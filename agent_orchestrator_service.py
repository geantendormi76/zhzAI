# /home/zhz/zhz_agent/agent_orchestrator_service.py

import os
import asyncio
import traceback
import json
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager
import re
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime 

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
service_logger = logging.getLogger(__name__)

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# --- 添加开始 ---
from core.prompts.manager_prompts import get_manager_agent_goal, MANAGER_AGENT_BACKSTORY
# --- 添加结束 ---

from core.llm_manager import get_llm_instance, CustomLiteLLMWrapper
from core.tools.enhanced_rag_tool import EnhancedRAGTool
from core.tools.excel_tool import ExcelOperationTool
from core.tools.search_tool import WebSearchTool
from core.tools.time_tool import GetCurrentTimeTool
from core.tools.calculator_tool import CalculateTool


AGENT_SERVICE_PORT = int(os.getenv("AGENT_SERVICE_PORT", 8090))
AGENT_SERVICE_HOST = "0.0.0.0"

class AgentTaskRequest(BaseModel):
    user_query: str = Field(description="用户的原始文本查询。")

class AgentTaskResponse(BaseModel):
    answer: str = Field(description="Agent 系统生成的最终答案或响应。")
    status: str = Field(default="success", description="执行状态: 'success', 'needs_clarification', 'error', 'processing_plan', 'task_created'.")
    intermediate_plan: Optional[Dict[str, Any]] = Field(None, description="如果status是'processing_plan', 这里包含Manager Agent的规划结果。")
    task_id: Optional[str] = Field(None, description="如果任务是异步的，返回任务ID。")
    error_message: Optional[str] = Field(None, description="如果发生错误，此字段包含错误信息。")
    debug_info: Optional[Dict[str, Any]] = Field(None, description="包含执行过程中的调试信息。")
    token_usage: Optional[Dict[str, Any]] = Field(None, description="LLM token 使用情况统计。")

class SubTaskDefinitionForManagerOutput(BaseModel):
    task_description: str = Field(description="用户的原始请求原文。")
    reasoning_for_plan: Optional[str] = Field(None, description="Manager Agent的决策思考过程。")
    selected_tool_names: List[str] = Field(description="选定的工具名称列表。如果直接回答，则为空列表。")
    direct_answer_content: Optional[str] = Field(None, description="如果选择直接回答，这里是答案内容。")
    tool_input_args: Optional[Dict[str, Any]] = Field(None, description="如果选择使用非Excel工具，这里是传递给该工具的参数。")
    excel_sqo_payload: Optional[List[Dict[str, Any]]] = Field(None, description="如果选择使用Excel工具，这里是SQO操作字典的列表。")

manager_llm: Optional[CustomLiteLLMWrapper] = None
worker_llm: Optional[CustomLiteLLMWrapper] = None
manager_agent_instance: Optional[Agent] = None
worker_agent_instance: Optional[Agent] = None
core_tools_instances: List[BaseTool] = []

# --- 覆盖开始 ---
CORE_TOOLS_ZHZ_AGENT = {
    "enhanced_rag_tool": "【核心RAG工具】用于从本地知识库查找信息、回答复杂问题，整合了向量、关键词和图谱检索。",
    "excel_operation_tool": "【Excel操作工具】通过结构化查询对象(SQO)对Excel文件执行复杂的数据查询、筛选、聚合等操作。此工具通过本地代理在Windows上运行。",
    "web_search_tool": "【网络搜索工具】使用DuckDuckGo搜索引擎在互联网上查找与用户查询相关的信息。此工具通过MCP调用。",
    "get_current_time_tool": "【时间工具】获取当前的日期和时间，可指定时区。此工具在Agent的Python环境中直接执行。",
    "calculate_tool": "【计算器工具】执行数学表达式的计算并返回数值结果。此工具在Agent的Python环境中直接执行。"
}
# --- 覆盖结束 ---
CORE_TOOL_NAMES_LIST = list(CORE_TOOLS_ZHZ_AGENT.keys()) # 这行会自动更新
TOOL_OPTIONS_STR_FOR_MANAGER = "\n".join( # 这行会自动更新
    [f"- '{name}': {desc}" for name, desc in CORE_TOOLS_ZHZ_AGENT.items()]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager_llm, worker_llm, manager_agent_instance, worker_agent_instance, core_tools_instances
    # --- [修改] 为了让 Manager 的 prompt 也只看到 RAG 工具，我们需要在函数作用域内临时修改这些 ---
    global CORE_TOOL_NAMES_LIST, TOOL_OPTIONS_STR_FOR_MANAGER 
    
    print("--- Agent Orchestrator Service: Lifespan startup ---")

    # --- 步骤 1: 初始化核心工具实例 ---
    print("Initializing core tool instances...")
    enhanced_rag_tool_instance = None 
    excel_operation_tool_instance = None
    web_search_tool_instance = None
    get_current_time_tool_instance = None
    calculate_tool_instance = None
    try:
        enhanced_rag_tool_instance = EnhancedRAGTool()
        excel_operation_tool_instance = ExcelOperationTool() # 尝试实例化
        web_search_tool_instance = WebSearchTool()       # 尝试实例化
        get_current_time_tool_instance = GetCurrentTimeTool()
        calculate_tool_instance = CalculateTool()

        core_tools_instances = []
        if enhanced_rag_tool_instance:
            core_tools_instances.append(enhanced_rag_tool_instance)
        if get_current_time_tool_instance:
            core_tools_instances.append(get_current_time_tool_instance)
        if calculate_tool_instance:
            core_tools_instances.append(calculate_tool_instance)
        if excel_operation_tool_instance:
            core_tools_instances.append(excel_operation_tool_instance)
        if web_search_tool_instance:
            core_tools_instances.append(web_search_tool_instance)

        if not core_tools_instances:
            print("CRITICAL ERROR: No core tools could be initialized. Aborting LLM/Agent setup.")
            # 在这种情况下，后续的LLM和Agent初始化可能会失败或无意义
            # 可以考虑在这里抛出异常或设置一个全局状态阻止服务启动
        else:
            print(f"Successfully initialized tools: {[tool.name for tool in core_tools_instances]}")
        print(f"--- FULL TOOL CONFIGURATION WILL BE USED BY MANAGER (if successfully initialized) ---")
        print(f"Effective CORE_TOOL_NAMES_LIST for Manager (based on global def): {CORE_TOOL_NAMES_LIST}") # 这是全局的
        print(f"Effective TOOL_OPTIONS_STR_FOR_MANAGER for Manager (based on global def):\n{TOOL_OPTIONS_STR_FOR_MANAGER}") # 这是全局的
        print(f"Actually initialized tools for Worker: {[tool.name for tool in core_tools_instances]}")
        print(f"--- END OF TOOL CONFIGURATION ---")
        
    except Exception as e:
        print(f"ERROR during core tool initialization: {e}", exc_info=True)
        core_tools_instances = [] # 确保出错时为空列表
        # 后续LLM/Agent初始化时，如果 core_tools_instances 为空，它们可能需要特殊处理或报错

    # ... 后续的LLM和Agent初始化代码 ...
    # 确保 Manager Agent 的 tools=[] (它不直接调用工具)
    # 确保 Worker Agent 的 tools=core_tools_instances (它需要所有可用的工具实例)


    # --- 步骤 2: 初始化 LLM 实例 ---
    # LLM 初始化时，agent_tools 参数将使用上面步骤中已更新（且只包含RAG工具）的 core_tools_instances
    print("Initializing LLM instances...")
    try:
        gemini_tool_config = {"function_calling_config": {"mode": "AUTO"}}
        # Manager LLM 初始化
        manager_llm = get_llm_instance(
            llm_type="cloud_gemini", 
            temperature=0.1, 
            max_tokens=4096, 
            tool_config=gemini_tool_config,
            agent_tools=core_tools_instances # 使用已更新的 core_tools_instances
        )
        if not manager_llm:
            print("Failed to initialize Manager LLM (Cloud Gemini). Attempting fallback...")
            manager_llm = get_llm_instance(
                llm_type="local_qwen", 
                temperature=0.1, 
                max_tokens=3072, 
                tool_config=gemini_tool_config,
                agent_tools=core_tools_instances 
            )
        
        # Worker LLM 初始化
        print("Initializing Worker LLM (attempting Cloud Gemini first)...")
        worker_gemini_tool_config = {"function_calling_config": {"mode": "AUTO"}} 
        worker_llm = get_llm_instance(
            llm_type="cloud_gemini", 
            temperature=0.5, 
            max_tokens=3072,
            tool_config=worker_gemini_tool_config,
            agent_tools=core_tools_instances # 使用已更新的 core_tools_instances
        )
        if not worker_llm:
            print("Failed to initialize Worker LLM (Cloud Gemini). Attempting fallback to local_qwen...")
            worker_llm = get_llm_instance(
                llm_type="local_qwen", 
                temperature=0.6, 
                max_tokens=3072,
                agent_tools=core_tools_instances
            )

        if manager_llm: print(f"Manager LLM initialized: {manager_llm.model_name}")
        else: print("CRITICAL: Failed to initialize Manager LLM.")
        if worker_llm: print(f"Worker LLM initialized: {worker_llm.model_name}")
        else: print("CRITICAL: Failed to initialize Worker LLM.")
    except Exception as e:
        print(f"FATAL ERROR during LLM initialization: {e}") 
        traceback.print_exc() 
        manager_llm = None; worker_llm = None

    # --- 步骤 3: 初始化 Agent 实例 ---
    if manager_llm:
        current_manager_goal = get_manager_agent_goal(TOOL_OPTIONS_STR_FOR_MANAGER)
        manager_agent_instance = Agent(
        role='资深AI任务分解与Excel查询规划师 (Senior AI Task Decomposition and Excel Query Planner)',
        goal=current_manager_goal,
        backstory=MANAGER_AGENT_BACKSTORY,
        llm=manager_llm,
        verbose=True,
        allow_delegation=False,
        tools=[] 
    )
        print(f"Manager Agent initialized with LLM: {manager_llm.model_name}")

    if worker_llm:
        worker_agent_instance = Agent(
            role='任务执行专家 (Task Execution Expert)',
            goal="根据Manager分配的具体任务描述和指定的工具，高效地执行任务并提供结果。",
            backstory="""我是一个AI执行者，专注于使用【Manager明确授权给我的工具】来解决问题。
                        我会严格遵循任务指令。如果任务是调用Excel工具并提供了SQO列表，我会按顺序迭代处理这些SQO，并整合结果。
                        对于像 'get_current_time_tool' 和 'calculate_tool' 这样的本地Python工具，我会直接在我的环境中执行它们。
                        对于其他工具，我会使用工具的名称（例如 'enhanced_rag_tool', 'excel_operation_tool', 'web_search_tool'）来调用它们。""",
            llm=worker_llm,
            verbose=True,
            allow_delegation=False,
            tools=core_tools_instances # Worker Agent 使用已更新（只含RAG工具）的 core_tools_instances
        )
        print(f"Worker Agent initialized with LLM: {worker_llm.model_name} and tools: {[t.name for t in worker_agent_instance.tools]}")

    if not manager_agent_instance or not worker_agent_instance:
        print("CRITICAL: One or more core agents failed to initialize. Service functionality will be severely limited.")
    elif not core_tools_instances and worker_agent_instance : 
        print("WARNING: Worker Agent initialized, but no core tools were successfully instantiated. Tool-based tasks will fail.")

    print("--- Agent Orchestrator Service: Lifespan startup complete ---")
    yield
    print("--- Agent Orchestrator Service: Lifespan shutdown ---")

app = FastAPI(
    title="Agent Orchestrator Service",
    description="接收用户请求，通过Manager/Worker Agent模型进行任务规划和执行。",
    version="0.1.1", # 版本更新
    lifespan=lifespan
)

@app.post("/v1/execute_task", response_model=AgentTaskResponse)
async def execute_task_endpoint(request: AgentTaskRequest):
    global manager_agent_instance, worker_agent_instance, core_tools_instances # 确保能访问全局 Agent 和工具实例
    
    print(f"Received agent task request: User Query='{request.user_query}'") # 使用 print 替代 logger

    if not manager_agent_instance or not worker_agent_instance:
        print("CRITICAL ERROR: Core agents are not initialized. Cannot process task.")
        raise HTTPException(status_code=503, detail="Service not ready: Core agents failed to initialize.")
    if not core_tools_instances:
        print("WARNING: Core tools are not initialized. Tool-based tasks may fail.")
    # --- 构建 Manager Task 的描述 ---
    # 我们将用户请求和可用的核心工具列表传递给 Manager Agent
    # Manager Agent 的 goal 和 backstory 已经包含了大部分指令
    # Task 的 description 主要用于传递动态信息，如当前用户查询
    manager_task_description_for_crewai = f"""
    用户的原始请求是：'{request.user_query}'

    你的任务是：
    1.  仔细分析此用户请求的意图。
    2.  严格回顾并遵循你在 `goal` 中被赋予的【决策规则与优先级】以及【决策示例】。
    3.  基于这些规则和示例，决定最佳的处理路径：是直接回答，还是选择一个最合适的工具。
    4.  如果选择了工具，请准备好调用该工具所需的参数。
    5.  严格按照 `SubTaskDefinitionForManagerOutput` 的JSON格式输出你的规划。`task_description` 字段必须是用户的原始请求原文: '{request.user_query}'。同时提供你的 `reasoning_for_plan`。
    """
    
    # --- Manager Task 的期望输出格式说明 ---
    manager_task_expected_output_description = f"""一个JSON对象，必须严格符合以下Pydantic模型的结构（你不需要输出 "SubTaskDefinitionForManagerOutput" 这个词本身）：
    {{
      "task_description": "string (固定为用户的原始请求: '{request.user_query}')",
      "reasoning_for_plan": "string (你的决策思考过程)",
      "selected_tool_names": ["list of strings (选定的工具名称列表。如果直接回答，则为空列表。如果使用工具，则为 ['enhanced_rag_tool'])"],
      "direct_answer_content": "string (可选, 仅当 selected_tool_names 为空列表时，这里是你的答案内容)",
      "tool_input_args": {{ "key": "value" }} (可选, 仅当 selected_tool_names 包含'enhanced_rag_tool'时，这里是给该工具的参数字典),
      "excel_sqo_payload": null # 当前Excel工具不可用，此字段应为null
    }}

    【重要输出规则】:
    - 如果你选择【直接回答】：`selected_tool_names` 必须是空列表 `[]`，`direct_answer_content` 必须包含你的答案，`tool_input_args` 和 `excel_sqo_payload` 应该为 `null` 或不存在。
    - 如果你选择使用【enhanced_rag_tool】：`selected_tool_names` 必须包含 `"enhanced_rag_tool"`，`direct_answer_content` 应该为 `null` 或不存在，`tool_input_args` 必须包含调用该工具所需的参数 (例如 `{{ "query": "{request.user_query}" }}` )，`excel_sqo_payload` 应该为 `null` 或不存在。

    示例输出 (直接回答):
    {{
      "task_description": "{request.user_query}",
      "reasoning_for_plan": "这是一个常识性问题，我可以根据我的内部知识直接回答。",
      "selected_tool_names": [],
      "direct_answer_content": "中国的首都是北京。",
      "tool_input_args": null,
      "excel_sqo_payload": null
    }}

    示例输出 (使用RAG工具):
    {{
      "task_description": "{request.user_query}",
      "reasoning_for_plan": "用户询问关于公司政策的问题，这需要从知识库中查找。",
      "selected_tool_names": ["enhanced_rag_tool"],
      "direct_answer_content": null,
      "tool_input_args": {{"query": "{request.user_query}", "top_k_vector": 5, "top_k_kg": 3, "top_k_bm25": 3}},
      "excel_sqo_payload": null
    }}
    请严格按照此JSON格式输出。
    """

    manager_task = Task(
        description=manager_task_description_for_crewai,
        expected_output=manager_task_expected_output_description,
        agent=manager_agent_instance,
        async_execution=False,
        output_pydantic=SubTaskDefinitionForManagerOutput
    )

    # --- 步骤 2: 执行 Manager Task ---
    print("Executing Manager Task...")
    manager_plan_object: Optional[SubTaskDefinitionForManagerOutput] = None
    manager_raw_output: Optional[str] = None

    try:
        manager_crew = Crew(
            agents=[manager_agent_instance],
            tasks=[manager_task],
            process=Process.sequential,
            verbose=True 
        )
        print(f"[{datetime.now()}] About to call manager_crew.kickoff()...") # <--- 添加日志
        manager_task_raw_result = manager_crew.kickoff(inputs={})
        print(f"[{datetime.now()}] manager_crew.kickoff() returned.") # <--- 添加日志

        # --- 新的解析逻辑，处理 CrewOutput ---
        actual_task_output: Optional[Any] = None # 用于存储实际的 TaskOutput 或字符串

        if hasattr(manager_task_raw_result, 'tasks_output') and isinstance(manager_task_raw_result.tasks_output, list) and manager_task_raw_result.tasks_output:
            # CrewOutput.tasks_output 是一个 TaskOutput 对象的列表
            actual_task_output = manager_task_raw_result.tasks_output[0] # 我们只有一个 manager_task
            print(f"Extracted TaskOutput from CrewOutput: {type(actual_task_output)}")
        elif hasattr(manager_task_raw_result, 'raw_output'): # 兼容直接返回 TaskOutput 的情况 (旧版或特定配置)
            actual_task_output = manager_task_raw_result
            print(f"kickoff returned a TaskOutput-like object directly: {type(actual_task_output)}")
        elif isinstance(manager_task_raw_result, str): # 直接返回字符串
            actual_task_output = manager_task_raw_result
            print(f"kickoff returned a raw string.")
        else:
            print(f"Warning: manager_crew.kickoff() returned an unexpected type: {type(manager_task_raw_result)}")
            actual_task_output = str(manager_task_raw_result)
        # --- 结束新的解析逻辑 ---


        # --- 后续的解析逻辑，现在基于 actual_task_output ---
        print(f"DEBUG: Type of actual_task_output: {type(actual_task_output)}")
        
        manager_raw_llm_output_str: Optional[str] = None
        if hasattr(actual_task_output, 'raw') and isinstance(actual_task_output.raw, str):
            manager_raw_llm_output_str = actual_task_output.raw.strip()
            print(f"DEBUG: actual_task_output.raw (LLM's original string output) IS:\n---\n{manager_raw_llm_output_str}\n---")
        else:
            print(f"DEBUG: actual_task_output.raw is not a string or does not exist. Value: {getattr(actual_task_output, 'raw', 'Attribute .raw not found')}")
            manager_raw_llm_output_str = str(actual_task_output) # Fallback

        # 优先尝试使用 CrewAI 已经解析好的 Pydantic 对象
        if hasattr(actual_task_output, 'pydantic_output') and \
           actual_task_output.pydantic_output is not None and \
           isinstance(actual_task_output.pydantic_output, SubTaskDefinitionForManagerOutput):
            print("DEBUG: Successfully using actual_task_output.pydantic_output.")
            manager_plan_object = actual_task_output.pydantic_output
        
        # 如果 Pydantic 对象不可用，但我们从 .raw 成功获取了字符串，则尝试解析它
        # 并且确保 manager_plan_object 之前没有被成功赋值
        elif manager_plan_object is None and manager_raw_llm_output_str: # <--- 添加 manager_plan_object is None 条件
            print(f"DEBUG: pydantic_output not available. Attempting to parse string from actual_task_output.raw:\n---\n{manager_raw_llm_output_str}\n---")
            json_to_parse_from_raw = manager_raw_llm_output_str # 使用我们从 .raw 获取的
            try:
                final_answer_marker = "## Final Answer:" # 虽然日志显示Qwen没输出这个，但保留以防万一
                if final_answer_marker in json_to_parse_from_raw:
                    json_to_parse_from_raw = json_to_parse_from_raw.split(final_answer_marker, 1)[-1].strip()
                
                cleaned_json_str = json_to_parse_from_raw.strip()
                if cleaned_json_str.startswith("```json"): cleaned_json_str = cleaned_json_str[len("```json"):].strip()
                if cleaned_json_str.endswith("```"): cleaned_json_str = cleaned_json_str[:-len("```")].strip()
                
                if not cleaned_json_str: raise ValueError("Cleaned JSON string (from .raw) is empty.")
                manager_plan_object = SubTaskDefinitionForManagerOutput(**json.loads(cleaned_json_str))
                print(f"Parsed Manager Plan (from actual_task_output.raw): {manager_plan_object.model_dump_json(indent=2)}")
            except (json.JSONDecodeError, ValueError, Exception) as e:
                print(f"Error parsing JSON from actual_task_output.raw: {e}. Raw content used: {json_to_parse_from_raw}")
        
        # 在所有尝试之后，如果 manager_plan_object 仍然是 None，才报告最终的解析失败
        if not manager_plan_object:
             # manager_raw_output 现在应该引用我们尝试过的 manager_raw_llm_output_str
             final_raw_output_for_debug = manager_raw_llm_output_str if manager_raw_llm_output_str else str(actual_task_output)
             print(f"Manager Task did not produce a valid Pydantic object after all attempts. Final Raw for debug: {final_raw_output_for_debug}")
             return AgentTaskResponse(answer="无法解析Manager Agent的规划结果。", status="error",
                                      error_message="Failed to parse manager plan after all attempts.",
                                      debug_info={"manager_raw_output_tried": final_raw_output_for_debug})
        
        # 如果 manager_plan_object 成功解析，则打印日志（这行已经在您的代码中）
        print(f"Parsed Manager Plan: {manager_plan_object.model_dump_json(indent=2)}")
        
    except Exception as e:
        print(f"Error executing Manager Task or its Crew: {e}") # <--- 修改后的第一行
        traceback.print_exc() # <--- 修改后的第二行
        return AgentTaskResponse(
            answer="执行Manager Agent任务时发生错误。", status="error", error_message=str(e),
            debug_info={"traceback": traceback.format_exc(), "manager_raw_output": manager_raw_output}
        )

    # --- 步骤 2: 根据 Manager 的规划执行后续操作 ---
    if not manager_plan_object.selected_tool_names and manager_plan_object.direct_answer_content:
        print(f"Manager decided to answer directly. Answer: {manager_plan_object.direct_answer_content}")
        return AgentTaskResponse(
            answer=manager_plan_object.direct_answer_content,
            status="success",
            debug_info={"manager_plan": manager_plan_object.model_dump()}
        )
    elif not manager_plan_object.selected_tool_names and not manager_plan_object.direct_answer_content:
        # 如果 Manager 既没有选择工具，也没有提供直接答案，这可能是一个规划错误
         print(f"Manager Warning: No tool selected and no direct answer provided. Reasoning: {manager_plan_object.reasoning_for_plan}")
         return AgentTaskResponse(
            answer=manager_plan_object.reasoning_for_plan or "Manager 未能提供明确的行动计划或答案。",
            status="success", # 或者 "error" 取决于您如何定义这种情况
            debug_info={"manager_plan": manager_plan_object.model_dump()}
        )

    # --- 步骤 3: 如果 Manager 规划使用工具，则创建并执行 Worker Task ---
    selected_tool_name = manager_plan_object.selected_tool_names[0] if manager_plan_object.selected_tool_names else None

    if not selected_tool_name:
        return AgentTaskResponse(
            answer="Manager规划使用工具但未在selected_tool_names中指定工具名称。", status="error",
            error_message="Tool name missing in manager plan's selected_tool_names.",
            debug_info={"manager_plan": manager_plan_object.model_dump()}
        )

    # 找到对应的工具实例
    tool_instance_for_worker = next((tool for tool in core_tools_instances if tool.name == selected_tool_name), None)

    if not tool_instance_for_worker:
        return AgentTaskResponse(
            answer=f"系统中未找到Manager规划使用的工具: {selected_tool_name}。", status="error",
            error_message=f"Tool '{selected_tool_name}' not found in core_tools_instances.",
            debug_info={"manager_plan": manager_plan_object.model_dump(), "available_tools": [t.name for t in core_tools_instances]}
        )

    print(f"Manager planned to use tool: {selected_tool_name}. Instance found: {tool_instance_for_worker is not None}")

    worker_task_description = ""
    worker_task_inputs = {} 

    if selected_tool_name == "enhanced_rag_tool":
        rag_query = manager_plan_object.tool_input_args.get("query", request.user_query) if manager_plan_object.tool_input_args else request.user_query
        top_k_v = manager_plan_object.tool_input_args.get("top_k_vector", 5) if manager_plan_object.tool_input_args else 5
        top_k_kg = manager_plan_object.tool_input_args.get("top_k_kg", 3) if manager_plan_object.tool_input_args else 3
        top_k_b = manager_plan_object.tool_input_args.get("top_k_bm25", 3) if manager_plan_object.tool_input_args else 3
        
        worker_task_description = f"请使用增强RAG工具回答以下问题：'{rag_query}'。\n使用以下参数进行检索：top_k_vector={top_k_v}, top_k_kg={top_k_kg}, top_k_bm25={top_k_b}。"
        worker_task_inputs = {
            "query": rag_query, 
            "top_k_vector": top_k_v, 
            "top_k_kg": top_k_kg, 
            "top_k_bm25": top_k_b
        }

    elif selected_tool_name == "excel_operation_tool":
        excel_sqo_list = manager_plan_object.excel_sqo_payload # excel_sqo_list 在这里赋值

        # --- 添加调试打印开始 ---
        print(f"DEBUG_EXCEL_TOOL: Type of manager_plan_object.excel_sqo_payload: {type(manager_plan_object.excel_sqo_payload)}")
        print(f"DEBUG_EXCEL_TOOL: Value of manager_plan_object.excel_sqo_payload: {manager_plan_object.excel_sqo_payload}")
        print(f"DEBUG_EXCEL_TOOL: Value of excel_sqo_list: {excel_sqo_list}")
        print(f"DEBUG_EXCEL_TOOL: Is excel_sqo_list None? {excel_sqo_list is None}")
        print(f"DEBUG_EXCEL_TOOL: Is excel_sqo_list an empty list? {isinstance(excel_sqo_list, list) and not excel_sqo_list}")
        # --- 添加调试打印结束 ---

        if not excel_sqo_list:
            return AgentTaskResponse(
                answer="Manager选择Excel工具但未提供SQO列表。", status="error",
                error_message="excel_sqo_payload is missing for excel_operation_tool.",
                debug_info={"manager_plan": manager_plan_object.model_dump()}
            )
        
        # --- 测试用硬编码 ---
        test_excel_file_path_on_windows = r"C:\\FlutterProjects\\data\\test2.xlsx" # 确保这个路径在Windows上是有效的
        test_excel_sheet_name: Union[str, int] = "Sheet1"
        # --- 测试用硬编码结束 ---
        
        worker_task_description = (
            f"你需要处理一个Excel文件相关的任务。目标文件路径是 '{test_excel_file_path_on_windows}'，工作表是 '{test_excel_sheet_name}'。\n"
            f"请严格按照以下SQO操作定义列表，顺序执行每一个操作，并整合所有操作的结果形成最终答案。\n"
            f"SQO操作定义列表: {json.dumps(excel_sqo_list, ensure_ascii=False)}\n\n"
            f"对于列表中的【每一个SQO操作定义字典】，你需要调用 '{tool_instance_for_worker.name}' 工具一次，"
            f"将该字典作为 'sqo_operation_definition' 参数，"
            f"同时传递 'target_excel_file_path': '{test_excel_file_path_on_windows}' 和 "
            f"'target_excel_sheet_name': '{test_excel_sheet_name}'。"
        )
        worker_task_inputs = {
            "excel_sqo_list_to_execute": excel_sqo_list, # Worker Agent会迭代这个
            "target_excel_file_path_for_worker": test_excel_file_path_on_windows, # Worker Agent使用这个传递给工具
            "target_excel_sheet_name_for_worker": test_excel_sheet_name # Worker Agent使用这个传递给工具
        }
        
    elif selected_tool_name == "web_search_tool": # 这是唯一且正确的 web_search_tool 分支
        search_query = manager_plan_object.tool_input_args.get("query", request.user_query) if manager_plan_object.tool_input_args else request.user_query
        max_results = manager_plan_object.tool_input_args.get("max_results", 5) if manager_plan_object.tool_input_args else 5
        worker_task_description = f"请使用网络搜索工具查找关于 '{search_query}' 的信息，返回最多 {max_results} 条结果。"
        worker_task_inputs = {"query": search_query, "max_results": max_results}
        
    elif selected_tool_name == "get_current_time_tool":
        timezone_str = manager_plan_object.tool_input_args.get("timezone", "Asia/Shanghai") if manager_plan_object.tool_input_args else "Asia/Shanghai"
        worker_task_description = f"请使用时间工具获取当前时间。时区参数为: '{timezone_str}'。"
        worker_task_inputs = {"timezone_str": timezone_str} 

    elif selected_tool_name == "calculate_tool":
        expression_str = manager_plan_object.tool_input_args.get("expression", "") if manager_plan_object.tool_input_args else ""
        if not expression_str:
             return AgentTaskResponse(
                answer="Manager选择计算器工具但未提供表达式。", status="error",
                error_message="expression is missing for calculate_tool.",
                debug_info={"manager_plan": manager_plan_object.model_dump()}
            )
        worker_task_description = f"请使用计算器工具计算以下表达式: '{expression_str}'。"
        worker_task_inputs = {"expression": expression_str}
        
    else: # 这是处理未知工具的 else
        return AgentTaskResponse(
            answer=f"未知的工具名称 '{selected_tool_name}' 被Manager规划。", status="error",
            error_message=f"Unknown tool '{selected_tool_name}' planned by manager.",
            debug_info={"manager_plan": manager_plan_object.model_dump()}
        )

    # 创建 Worker Task
    worker_task = Task(
        description=worker_task_description,
        expected_output="任务的执行结果，通常是一个字符串，其中包含答案或操作的状态。",
        agent=worker_agent_instance,
        tools=[tool_instance_for_worker], # 只给 Worker 当前任务需要的工具
        async_execution=False # Worker Task 通常也是同步的
    )

    print(f"Executing Worker Task with tool: {selected_tool_name}")
    print(f"Worker Task Description: {worker_task_description}")
    print(f"Worker Task Inputs: {worker_task_inputs}")
    
    worker_final_result: str = ""
    worker_crew_usage_metrics: Optional[Dict[str, Any]] = None # 用于存储 token usage
    token_usage_for_response: Optional[Dict[str, Any]] = None # <--- 在这里初始化
    try:
        # --- [关键修改] 为 Worker Task 创建并运行一个 Crew ---
        worker_crew = Crew(
            agents=[worker_agent_instance], # Worker Agent
            tasks=[worker_task],             # 它要执行的单个任务
            process=Process.sequential,
            verbose=True # 可以设为 True 或 2 来查看 Worker Crew 的详细日志
        )
        
        # 如果 worker_task_inputs 为空，则 inputs={}
        # CrewAI 的 kickoff 方法期望 inputs 是一个字典
        task_execution_inputs = worker_task_inputs if worker_task_inputs else {}
            
        # 执行 Worker Crew
        # kickoff 返回的是 CrewOutput 对象，或者在某些旧版本/配置下可能直接是结果字符串或 TaskOutput
        worker_crew_output = worker_crew.kickoff(inputs=task_execution_inputs)
        
        # 从 Worker Crew 的输出中提取结果
        # 这与我们处理 Manager Crew 输出的逻辑类似
        actual_worker_task_output: Optional[Any] = None
        if hasattr(worker_crew_output, 'tasks_output') and isinstance(worker_crew_output.tasks_output, list) and worker_crew_output.tasks_output:
            actual_worker_task_output = worker_crew_output.tasks_output[0] # 我们只有一个 worker_task
        elif hasattr(worker_crew_output, 'raw_output'): # 兼容直接返回 TaskOutput
            actual_worker_task_output = worker_crew_output
        elif isinstance(worker_crew_output, str): # 直接返回字符串
            actual_worker_task_output = worker_crew_output
        else: # 其他意外情况
            actual_worker_task_output = str(worker_crew_output)

        # 从 actual_worker_task_output 中提取最终的字符串结果
        if hasattr(actual_worker_task_output, 'raw') and isinstance(actual_worker_task_output.raw, str):
            worker_final_result = actual_worker_task_output.raw.strip()
        elif isinstance(actual_worker_task_output, str):
            worker_final_result = actual_worker_task_output.strip()
        else: # Fallback
            worker_final_result = str(actual_worker_task_output)
        
        print(f"Worker Task executed. Result: {worker_final_result}")
        
        # 获取 Worker Crew 的 token usage
        if hasattr(worker_crew, 'usage_metrics'):
            worker_crew_usage_metrics = worker_crew.usage_metrics
            print(f"DEBUG AGENT_ORCH: Raw worker_crew.usage_metrics object: {worker_crew_usage_metrics}")
            print(f"DEBUG AGENT_ORCH: Type of worker_crew.usage_metrics: {type(worker_crew_usage_metrics)}")
            if hasattr(worker_crew_usage_metrics, 'model_dump'):
                print(f"DEBUG AGENT_ORCH: worker_crew.usage_metrics.model_dump(): {worker_crew_usage_metrics.model_dump()}")
            elif isinstance(worker_crew_usage_metrics, dict):
                    print(f"DEBUG AGENT_ORCH: worker_crew.usage_metrics (is dict): {worker_crew_usage_metrics}")
            else:
                print(f"DEBUG AGENT_ORCH: worker_crew.usage_metrics is not a dict and has no model_dump.")
        else:
            print("DEBUG AGENT_ORCH: worker_crew does not have usage_metrics attribute.")
            worker_crew_usage_metrics = None # 确保它被定义
            # --- [结束关键修改] ---

            # 准备 token_usage_for_response
            token_usage_for_response = None
            if worker_crew_usage_metrics:
                if hasattr(worker_crew_usage_metrics, 'model_dump'):
                    token_usage_for_response = worker_crew_usage_metrics.model_dump()
                elif isinstance(worker_crew_usage_metrics, dict):
                    token_usage_for_response = worker_crew_usage_metrics
                else:
                    # 如果不是 Pydantic 模型或字典，尝试转换为字符串记录，但不作为结构化数据返回
                    service_logger.warning(f"Unexpected type for worker_crew_usage_metrics: {type(worker_crew_usage_metrics)}. Will not be included in structured token_usage.")

        return AgentTaskResponse(
                answer=worker_final_result,
                status="success",
                debug_info={
                    "manager_plan": manager_plan_object.model_dump(),
                    "worker_tool_used": selected_tool_name,
                    "worker_task_inputs": worker_task_inputs 
                },
                token_usage=token_usage_for_response # 使用处理后的 token_usage_for_response
            )

    except Exception as e:
    # 使用 traceback 打印详细错误
        print(f"Error executing Worker Task for tool {selected_tool_name}: {e}")
        traceback.print_exc() 
        return AgentTaskResponse(
            answer=f"执行工具 '{selected_tool_name}' 时发生错误。",
            status="error",
            error_message=str(e),
            debug_info={
                "manager_plan": manager_plan_object.model_dump(),
                "worker_tool_used": selected_tool_name,
                "worker_task_inputs": worker_task_inputs,
                "traceback": traceback.format_exc()
            }
        )

if __name__ == "__main__":
    print(f"--- Starting Agent Orchestrator FastAPI Service on {AGENT_SERVICE_HOST}:{AGENT_SERVICE_PORT} ---")
    uvicorn.run("agent_orchestrator_service:app", host=AGENT_SERVICE_HOST, port=AGENT_SERVICE_PORT, reload=True) # 确保模块名正确