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

# --- CrewAI ---
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# --- 自定义模块导入 ---
from core.llm_manager import get_llm_instance, CustomLiteLLMWrapper

# --- 导入我们新创建的工具 ---
from core.tools.enhanced_rag_tool import EnhancedRAGTool
from core.tools.excel_tool import ExcelOperationTool
from core.tools.search_tool import WebSearchTool

# --- 配置 ---
AGENT_SERVICE_PORT = int(os.getenv("AGENT_SERVICE_PORT", 8090))
AGENT_SERVICE_HOST = "0.0.0.0"

# --- Pydantic 模型定义 ---
# 沿用之前的 AgentTaskRequest 和 AgentTaskResponse
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

# 与 zhzai-agent/models.py 中 SubTaskDefinition 一致的结构，用于 Manager Agent 的输出
class SubTaskDefinitionForManagerOutput(BaseModel):
    task_description: str = Field(description="用户的原始请求原文。")
    reasoning_for_plan: Optional[str] = Field(None, description="Manager Agent的决策思考过程。")
    selected_tool_names: List[str] = Field(description="选定的工具名称列表。如果直接回答，则为空列表。")
    direct_answer_content: Optional[str] = Field(None, description="如果选择直接回答，这里是答案内容。")
    tool_input_args: Optional[Dict[str, Any]] = Field(None, description="如果选择使用非Excel工具，这里是传递给该工具的参数。")
    excel_sqo_payload: Optional[List[Dict[str, Any]]] = Field(None, description="如果选择使用Excel工具，这里是SQO操作字典的列表。")


# --- 全局变量 ---
manager_llm: Optional[CustomLiteLLMWrapper] = None
worker_llm: Optional[CustomLiteLLMWrapper] = None
manager_agent_instance: Optional[Agent] = None
worker_agent_instance: Optional[Agent] = None

core_tools_instances: List[BaseTool] = [] # BaseTool 是 CrewAI 工具的基类

# 我们新的核心工具名称 (与 zhzai-agent 不同)
CORE_TOOLS_ZHZ_AGENT = {
    "enhanced_rag_tool": "【核心RAG工具】用于从本地知识库查找信息、回答复杂问题，整合了向量、关键词和图谱检索。",
    "excel_operation_tool": "【Excel操作工具】通过结构化查询对象(SQO)对Excel文件执行复杂的数据查询、筛选、聚合等操作。",
    "web_search_tool": "【网络搜索工具】使用DuckDuckGo搜索引擎在互联网上查找与用户查询相关的信息。"
}
CORE_TOOL_NAMES_LIST = list(CORE_TOOLS_ZHZ_AGENT.keys())

# 将工具描述格式化为字符串，供Manager Prompt使用
TOOL_OPTIONS_STR_FOR_MANAGER = "\n".join(
    [f"- '{name}': {desc}" for name, desc in CORE_TOOLS_ZHZ_AGENT.items()]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager_llm, worker_llm, manager_agent_instance, worker_agent_instance, core_tools_instances
    print("--- Agent Orchestrator Service: Lifespan startup ---")

    # --- 步骤 1: 初始化 LLM 实例 (保持不变) ---
    print("Initializing LLM instances...")
    try:
        gemini_tool_config = {"function_calling_config": {"mode": "AUTO"}}
        manager_llm = get_llm_instance(
            llm_type="cloud_gemini", temperature=0.1, max_tokens=4096, tool_config=gemini_tool_config
        )
        if not manager_llm:
            print("Failed to initialize Manager LLM (Cloud Gemini). Attempting fallback...")
            manager_llm = get_llm_instance(
                llm_type="local_qwen", temperature=0.1, max_tokens=3072, tool_config=gemini_tool_config
            )
        
        worker_llm = get_llm_instance(llm_type="local_qwen", temperature=0.6, max_tokens=3072)

        if manager_llm: print(f"Manager LLM initialized: {manager_llm.model_name}")
        else: print("CRITICAL: Failed to initialize Manager LLM.")
        if worker_llm: print(f"Worker LLM initialized: {worker_llm.model_name}")
        else: print("CRITICAL: Failed to initialize Worker LLM.")
    except Exception as e:
        print(f"FATAL ERROR during LLM initialization: {e}", exc_info=True)
        manager_llm = None; worker_llm = None

    # --- 步骤 2: 初始化核心工具实例 ---
    print("Initializing core tool instances...")
    try:
        enhanced_rag_tool_instance = EnhancedRAGTool()
        excel_operation_tool_instance = ExcelOperationTool()
        web_search_tool_instance = WebSearchTool()
        
        core_tools_instances = [
            enhanced_rag_tool_instance,
            excel_operation_tool_instance,
            web_search_tool_instance,
        ]
        print(f"Core tools initialized: {[tool.name for tool in core_tools_instances]}")
    except Exception as e:
        print(f"ERROR during core tool initialization: {e}", exc_info=True)
        core_tools_instances = []

    # 初始化 Agent 实例
    if manager_llm:
        manager_agent_instance = Agent(
            role='资深AI任务分解与Excel查询规划师 (Senior AI Task Decomposition and Excel Query Planner)',
            goal=f"""【深入理解并分解】用户提出的复杂请求 (当前请求将通过任务描述提供) 成为一系列逻辑子任务。
【核心决策1 - 优先自主回答】：在选择任何工具之前，请首先判断你是否能基于自身知识库和推理能力【直接回答】用户的全部或核心部分请求。如果可以，你的主要输出应是包含直接答案的JSON。
【核心决策2 - Excel复杂查询处理】：如果用户的请求涉及到对Excel文件进行一个或多个复杂的数据查询、筛选、聚合或排序等操作，并且你无法直接回答，你【必须】选择 "{CORE_TOOL_NAMES_LIST[CORE_TOOL_NAMES_LIST.index('excel_operation_tool')]}" 工具。并且，你【必须】为这些Excel操作【构建一个结构化查询对象 (SQO) 的JSON列表】，并将其作为输出JSON中 `excel_sqo_payload` 字段的值。列表中的【每一个SQO字典】都需要包含一个明确的 "operation_type" 和该操作对应的参数，但【不要包含 "file_path" 或 "sheet_name"】。
【核心决策3 - 其他工具选择】：如果需要从本地知识库获取深度信息，选择 "{CORE_TOOL_NAMES_LIST[CORE_TOOL_NAMES_LIST.index('enhanced_rag_tool')]}"。如果需要网络实时信息，选择 "{CORE_TOOL_NAMES_LIST[CORE_TOOL_NAMES_LIST.index('web_search_tool')]}"。选择工具后，你需要准备好传递给该工具的参数，并将其放在输出JSON的 `tool_input_args` 字段中。
【核心决策4 - 通用请求/无适用工具】：如果用户请求是生成通用文本、编写简单代码、回答一般性知识问题，并且你已判断可以【直接回答】，则无需选择任何特定功能性工具。此时输出JSON中 `selected_tool_names` 应为空列表，`excel_sqo_payload` 和 `tool_input_args` 为null，答案内容在 `direct_answer_content` 字段。

最终，【严格按照指定的JSON格式】（即符合 `SubTaskDefinitionForManagerOutput` Pydantic模型）输出一个包含你的决策理由、用户原始请求、以及根据你的决策填充的 `direct_answer_content` 或 `selected_tool_names`、`tool_input_args`、`excel_sqo_payload` 的对象。

【可供选择的本系统核心工具及其描述】:
{TOOL_OPTIONS_STR_FOR_MANAGER}
""",
            backstory="""我是一位经验丰富的AI任务调度官和数据查询规划专家。我的核心工作流程如下：
1.  **【深度理解与CoD规划 (内部进行)】**：我会对用户请求进行彻底分析，优先判断是否能直接利用我的知识库和推理能力给出完整答案。
2.  **【工具选择与参数准备（如果无法直接回答）】**：
    a.  **Excel复杂查询**: 当识别出需要对Excel执行复杂操作时，我选择 "{CORE_TOOL_NAMES_LIST[CORE_TOOL_NAMES_LIST.index('excel_operation_tool')]}" 工具，并为其生成包含多个SQO操作定义的【JSON列表】作为 `excel_sqo_payload`。每个SQO包含 `operation_type` 和所需参数，但不含 `file_path` 或 `sheet_name`。
    b.  **RAG查询**: 若需从知识库获取信息，选择 "{CORE_TOOL_NAMES_LIST[CORE_TOOL_NAMES_LIST.index('enhanced_rag_tool')]}"，并准备 `tool_input_args`（例如 `{{ "query": "用户原始问题", "top_k_vector": 5, ... }}`）。
    c.  **网络搜索**: 若需实时网络信息，选择 "{CORE_TOOL_NAMES_LIST[CORE_TOOL_NAMES_LIST.index('web_search_tool')]}"，并准备 `tool_input_args`（例如 `{{ "query": "搜索关键词" }}`）。
    d.  **最简必要原则**: 只选择绝对必要的工具。
3.  **【严格的输出格式】**: 我的唯一输出是一个JSON对象，该对象必须符合本服务定义的 `SubTaskDefinitionForManagerOutput` Pydantic模型结构，包含 `task_description` (用户原始请求), `reasoning_for_plan`, 以及根据决策填充的 `selected_tool_names` (可为空), `direct_answer_content` (如果直接回答), `tool_input_args` (如果使用非Excel工具), 和 `excel_sqo_payload` (如果使用Excel工具)。

我【不】自己执行任何工具操作。我的职责是精准规划并输出结构化的任务定义。""",
            llm=manager_llm,
            verbose=True,
            allow_delegation=False,
            tools=[]
        )
        print(f"Manager Agent initialized with LLM: {manager_llm.model_name}")

    if worker_llm:
        # Worker Agent 现在拥有所有核心工具
        worker_agent_instance = Agent(
            role='任务执行专家 (Task Execution Expert)',
            goal="根据Manager分配的具体任务描述和指定的工具，高效地执行任务并提供结果。",
            backstory="""我是一个AI执行者，专注于使用【Manager明确授权给我的工具】来解决问题。
                        我会严格遵循任务指令。如果任务是调用Excel工具并提供了SQO列表，我会按顺序迭代处理这些SQO，并整合结果。
                        我会使用工具的名称（例如 'enhanced_rag_tool', 'excel_operation_tool', 'web_search_tool'）来调用它们。""",
            llm=worker_llm,
            verbose=True,
            allow_delegation=False,
            tools=core_tools_instances # <--- 将实例化的工具列表传递给 Worker Agent
        )
        print(f"Worker Agent initialized with LLM: {worker_llm.model_name} and tools: {[t.name for t in core_tools_instances]}")

    if not manager_agent_instance or not worker_agent_instance:
        print("CRITICAL: One or more core agents failed to initialize. Service functionality will be severely limited.")
    elif not core_tools_instances and worker_agent_instance : # 如果 Worker Agent 初始化了但没有工具
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
    请仔细分析以下用户请求：
    '{request.user_query}'

    你的目标是：
    1.  理解用户的核心意图。
    2.  **优先判断**：你能否基于你现有的知识直接、准确地回答这个问题？
        - 如果是，请在输出的JSON中填充 `direct_answer_content` 字段，并将 `selected_tool_names` 设为空列表。
    3.  **如果不能直接回答**：判断解决这个问题最核心的工具是什么。从你已知的核心工具 ({', '.join(CORE_TOOL_NAMES_LIST)}) 中选择【一个或多个必要】工具。
    4.  如果选择了 '{CORE_TOOL_NAMES_LIST[CORE_TOOL_NAMES_LIST.index('excel_operation_tool')]}'，请为Excel操作构建一个或多个SQO的【列表】，并将其放入 `excel_sqo_payload` 字段。
    5.  如果选择了其他工具（如 '{CORE_TOOL_NAMES_LIST[CORE_TOOL_NAMES_LIST.index('enhanced_rag_tool')]}' 或 '{CORE_TOOL_NAMES_LIST[CORE_TOOL_NAMES_LIST.index('web_search_tool')]}'），请准备好传递给该工具的参数，并将其放入 `tool_input_args` 字段。
    6.  严格按照 `SubTaskDefinitionForManagerOutput` 的JSON格式输出你的规划。`task_description` 字段必须是用户的原始请求原文: '{request.user_query}'。同时提供你的 `reasoning_for_plan`。
    """
    
    # --- Manager Task 的期望输出格式说明 ---
    # 这个 expected_output 对于指导LLM以正确的JSON格式返回至关重要
    manager_task_expected_output_description = f"""一个JSON对象，必须严格符合以下Pydantic模型的结构（你不需要输出 "SubTaskDefinitionForManagerOutput" 这个词本身）：
    {{
      "task_description": "string (固定为用户的原始请求: '{request.user_query}')",
      "reasoning_for_plan": "string (你的决策思考过程)",
      "selected_tool_names": ["list of strings (选定的工具名称列表。如果直接回答，则为空列表)"],
      "direct_answer_content": "string (可选, 仅当 selected_tool_names 为空列表时，这里是你的答案内容)",
      "tool_input_args": {{ "key": "value" }} (可选, 仅当 selected_tool_names 包含非Excel工具时，这里是给该工具的参数字典),
      "excel_sqo_payload": "[{{...}}, {{...}}] (可选, 仅当 selected_tool_names 包含'{CORE_TOOL_NAMES_LIST[CORE_TOOL_NAMES_LIST.index('excel_operation_tool')]}'时，这里是SQO操作字典的列表)"
    }}

    【重要输出规则】:
    - 如果你选择【直接回答】：`selected_tool_names` 必须是空列表 `[]`，`direct_answer_content` 必须包含你的答案，`tool_input_args` 和 `excel_sqo_payload` 应该为 `null` 或不存在。
    - 如果你选择使用【非Excel工具】(例如 RAG 或 Web Search)：`selected_tool_names` 必须包含该工具的名称，`direct_answer_content` 应该为 `null` 或不存在，`tool_input_args` 必须包含调用该工具所需的参数 (例如 `{{ "query": "{request.user_query}" }}` )，`excel_sqo_payload` 应该为 `null` 或不存在。
    - 如果你选择使用【Excel工具】 ('{CORE_TOOL_NAMES_LIST[CORE_TOOL_NAMES_LIST.index('excel_operation_tool')]}')：`selected_tool_names` 必须包含 `"{CORE_TOOL_NAMES_LIST[CORE_TOOL_NAMES_LIST.index('excel_operation_tool')]}"`，`direct_answer_content` 和 `tool_input_args` 应该为 `null` 或不存在，`excel_sqo_payload` 【必须】包含一个SQO操作定义的列表。

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
      "selected_tool_names": ["{CORE_TOOL_NAMES_LIST[CORE_TOOL_NAMES_LIST.index('enhanced_rag_tool')]}"],
      "direct_answer_content": null,
      "tool_input_args": {{"query": "{request.user_query}", "top_k_vector": 5, "top_k_kg": 3, "top_k_bm25": 3}},
      "excel_sqo_payload": null
    }}
    
    示例输出 (使用Excel工具，假设用户问“test.xlsx中有哪些区域以及每个区域的平均销售额”):
    {{
      "task_description": "test.xlsx中有哪些区域以及每个区域的平均销售额",
      "reasoning_for_plan": "用户需要从Excel文件中获取唯一区域列表，并对销售额按区域进行聚合计算平均值。这需要两个SQO操作。",
      "selected_tool_names": ["{CORE_TOOL_NAMES_LIST[CORE_TOOL_NAMES_LIST.index('excel_operation_tool')]}"],
      "direct_answer_content": null,
      "tool_input_args": null, 
      "excel_sqo_payload": [
        {{
          "operation_type": "get_unique_values",
          "parameters": {{ "column_name": "区域" }}
        }},
        {{
          "operation_type": "group_by_aggregate",
          "parameters": {{ "group_by_columns": ["区域"], "aggregation_column": "销售额", "aggregation_function": "mean" }}
        }}
      ]
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
    worker_task_inputs = {} # 用于传递给 task.execute(inputs=...)

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
        excel_sqo_list = manager_plan_object.excel_sqo_payload
        if not excel_sqo_list:
            return AgentTaskResponse(
                answer="Manager选择Excel工具但未提供SQO列表。", status="error",
                error_message="excel_sqo_payload is missing for excel_operation_tool.",
                debug_info={"manager_plan": manager_plan_object.model_dump()}
            )
        
        # TODO: 从用户原始请求 manager_plan_object.task_description 中提取 file_path 和 sheet_name
        # 这里我们先用一个占位符/默认值，您需要实现提取逻辑
        # 例如，使用正则表达式从 manager_plan_object.task_description 查找文件路径
        # file_path_match = re.search(r"文件\s*['\"]?([^'\"]+\.(?:xlsx|xls|csv))['\"]?", manager_plan_object.task_description)
        # excel_file_path = file_path_match.group(1) if file_path_match else "YOUR_DEFAULT_TEST_EXCEL_PATH.xlsx"
        excel_file_path = "/home/zhz/zhz_agent/data/test.xlsx" # 临时硬编码，后续需要动态获取
        excel_sheet_name: Union[str, int] = 0 # 临时硬编码

        worker_task_description = (
            f"你需要处理一个Excel文件相关的任务。文件路径是 '{excel_file_path}'，工作表是 '{excel_sheet_name}'。\n"
            f"请严格按照以下SQO操作列表，顺序执行每一个操作，并整合所有操作的结果形成最终答案。\n"
            f"SQO操作列表: {json.dumps(excel_sqo_list, ensure_ascii=False)}\n\n"
            f"对于列表中的每一个SQO字典，你需要调用 '{tool_instance_for_worker.name}' 工具一次，"
            f"将该SQO字典作为 'sqo_dict' 参数，同时传递 'file_path': '{excel_file_path}' 和 'sheet_name': '{excel_sheet_name}'。"
        )
        # Worker Task的inputs将是整个SQO列表以及文件和工作表信息
        # Worker Agent的Prompt需要指导它如何迭代处理这个列表
        worker_task_inputs = {
            "excel_sqo_list_to_execute": excel_sqo_list,
            "target_excel_file_path": excel_file_path,
            "target_excel_sheet_name": excel_sheet_name
        }
        
    elif selected_tool_name == "web_search_tool":
        search_query = manager_plan_object.tool_input_args.get("query", request.user_query) if manager_plan_object.tool_input_args else request.user_query
        max_results = manager_plan_object.tool_input_args.get("max_results", 5) if manager_plan_object.tool_input_args else 5

        worker_task_description = f"请使用网络搜索工具查找关于 '{search_query}' 的信息，返回最多 {max_results} 条结果。"
        worker_task_inputs = {"query": search_query, "max_results": max_results}
        
    else:
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
    try:
        # 如果 worker_task_inputs 为空，则不传递 inputs 参数
        task_execution_args = {}
        if worker_task_inputs:
            task_execution_args['inputs'] = worker_task_inputs
            
        worker_output = worker_task.execute(**task_execution_args)

        if isinstance(worker_output, str):
            worker_final_result = worker_output
        elif hasattr(worker_output, 'raw_output'): # CrewAI 0.30.0+
            worker_final_result = worker_output.raw_output
        elif hasattr(worker_output, 'raw'): # Older CrewAI
            worker_final_result = worker_output.raw
        else:
            worker_final_result = str(worker_output)
            
        print(f"Worker Task executed. Result: {worker_final_result}")
        
        # (可选) 获取 Worker Crew 的 token usage
        worker_token_usage = None
        if hasattr(worker_task, 'agent') and hasattr(worker_task.agent, 'crew') and worker_task.agent.crew and hasattr(worker_task.agent.crew, 'usage_metrics'):
             worker_token_usage = worker_task.agent.crew.usage_metrics
             print(f"Worker Crew token usage: {worker_token_usage}")


        return AgentTaskResponse(
            answer=worker_final_result,
            status="success",
            debug_info={
                "manager_plan": manager_plan_object.model_dump(),
                "worker_tool_used": selected_tool_name,
                "worker_task_inputs": worker_task_inputs
            },
            token_usage=worker_token_usage.model_dump() if worker_token_usage else None
        )

    except Exception as e:
        print(f"Error executing Worker Task for tool {selected_tool_name}: {e}", exc_info=True)
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