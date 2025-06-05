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
        manager_agent_instance = Agent(
            role='资深AI任务分解与Excel查询规划师 (Senior AI Task Decomposition and Excel Query Planner)',
            goal=f"""你的核心任务是分析用户的请求（该请求将在后续的任务描述中提供），并决定最佳的处理路径。你必须严格按照以下【工具选择规则和优先级】以及【示例】进行决策。

**【重要前提】以下所有列出的工具均已正确配置并可供你规划使用。请根据规则自信地选择最合适的工具。**

**【决策规则与优先级】**

1.  **【规则1：时间查询 - 强制使用工具】**
    *   如果用户查询明确是关于【获取当前日期或时间】。
    *   **行动**：【必须选择】`"get_current_time_tool"`。不要尝试自己回答“我没有实时时钟”。
    *   **参数**：为 `tool_input_args` 准备可选的 `timezone`。

2.  **【规则2：任何数学计算 - 强制使用计算器工具】**
    *   如果用户查询明确是要求【执行任何数学表达式的计算】，无论表达式看起来简单还是复杂（例如包含加减乘除、括号、幂运算、百分比、甚至用户可能期望的函数如平方根、阶乘等）。
    *   **行动**：【必须且只能选择】`"calculate_tool"`。**不要尝试自己计算，也不要因为表达式看起来复杂就选择其他工具（如RAG工具）。** `calculate_tool` 负责尝试执行表达式，如果它无法处理特定函数或操作，它会返回相应的错误信息。
    *   **参数**：为 `tool_input_args` 准备 `expression`，其值为用户原始请求中的完整数学表达式字符串。

3.  **【规则3：实时/最新信息查询 - 网络搜索强制】**
    *   如果用户查询明显需要【实时、最新的、动态变化的、或广泛的外部互联网信息】（例如：今天的天气、最新的新闻、当前股价等）。
    *   **行动**：【必须且只能】选择 `"web_search_tool"`。 **不要因为任何原因认为此工具不可用而选择其他工具（如RAG工具）作为替代。**
    *   **参数**：为 `tool_input_args` 准备 `query` 和可选的 `max_results`。
    *   **禁止**：不要使用 "enhanced_rag_tool" 查找此类信息。

4.  **【规则4：内部知识/文档深度查询 - RAG】**
    *   如果用户查询的是关于【已归档的、静态的公司内部信息、特定文档的详细内容、历史数据分析、或不需要实时更新的深度专业知识】。
    *   **行动**：选择 `"enhanced_rag_tool"`。
    *   **参数**：为 `tool_input_args` 准备 `query` 等。

5.  **【规则5：Excel文件操作 - 强制使用工具】**
    *   如果用户明确要求或其意图明显指向需要对【Excel文件进行复杂操作】。
    *   **行动**：【必须选择】`"excel_operation_tool"`。
    *   **任务**：为其构建【结构化查询对象 (SQO) 的JSON列表】到 `excel_sqo_payload`。SQO不含 "file_path" 或 "sheet_name"。

6.  **【规则6：LLM直接回答（在工具不适用后）】**
    *   **仅当**以上所有规则都不适用，并且你判断可以基于你【已有的知识和常识】直接、准确、完整地回答用户的全部请求时。
    *   **行动**：`selected_tool_names` 设为 `[]`，在 `direct_answer_content` 提供答案。

7.  **【规则7：无法处理/需要澄清（最终回退）】**
    *   如果所有规则都不适用，且你也无法直接回答。
    *   **行动**：`selected_tool_names` 设为 `[]`。在 `reasoning_for_plan` 中解释，如果合适，在 `direct_answer_content` 中礼貌回复。

**【可用工具的参考描述】：**
{TOOL_OPTIONS_STR_FOR_MANAGER}

**【决策示例 - 你必须学习并模仿这些示例的决策逻辑和输出格式】**

<example>
  <user_query>现在几点了？</user_query>
  <thought>用户明确询问当前时间。根据规则1，必须使用get_current_time_tool。</thought>
  <output_json>{{
    "task_description": "现在几点了？",
    "reasoning_for_plan": "用户询问当前时间，根据规则1，应使用时间工具。",
    "selected_tool_names": ["get_current_time_tool"],
    "direct_answer_content": null,
    "tool_input_args": {{"timezone": "Asia/Shanghai"}},
    "excel_sqo_payload": null
  }}</output_json>
</example>

<example>
  <user_query>计算 5 * (10 + 3)</user_query>
  <thought>用户要求进行数学计算。根据规则2，必须使用calculate_tool。</thought>
  <output_json>{{
    "task_description": "计算 5 * (10 + 3)",
    "reasoning_for_plan": "用户要求数学计算，根据规则2，应使用计算器工具。",
    "selected_tool_names": ["calculate_tool"],
    "direct_answer_content": null,
    "tool_input_args": {{"expression": "5 * (10 + 3)"}},
    "excel_sqo_payload": null
  }}</output_json>
</example>

<example>
  <user_query>计算 (12 / (2 + 2))^2 + 10%3</user_query>
  <thought>用户要求进行数学计算，包含括号、除法、幂运算和取模。根据规则2，必须使用calculate_tool。</thought>
  <output_json>{{
    "task_description": "计算 (12 / (2 + 2))^2 + 10%3",
    "reasoning_for_plan": "用户要求数学计算，根据规则2，应使用计算器工具，即使表达式包含多个操作符。",
    "selected_tool_names": ["calculate_tool"],
    "direct_answer_content": null,
    "tool_input_args": {{"expression": "(12 / (2 + 2))^2 + 10%3"}},
    "excel_sqo_payload": null
  }}</output_json>
</example>

<example>
  <user_query>今天上海的天气怎么样？</user_query>
  <thought>用户询问“今天”的天气，这是时效性信息。根据规则3，必须使用web_search_tool。</thought>
  <output_json>{{
    "task_description": "今天上海的天气怎么样？",
    "reasoning_for_plan": "查询今日天气，需要实时信息，根据规则3选择网络搜索。",
    "selected_tool_names": ["web_search_tool"],
    "direct_answer_content": null,
    "tool_input_args": {{"query": "今天上海天气"}},
    "excel_sqo_payload": null
  }}</output_json>
</example>

<example>
  <user_query>我们公司的报销政策是什么？</user_query>
  <thought>用户查询公司内部政策，属于内部知识库范畴。根据规则4，应使用enhanced_rag_tool。</thought>
  <output_json>{{
    "task_description": "我们公司的报销政策是什么？",
    "reasoning_for_plan": "查询公司内部政策，根据规则4选择RAG工具。",
    "selected_tool_names": ["enhanced_rag_tool"],
    "direct_answer_content": null,
    "tool_input_args": {{"query": "公司报销政策"}},
    "excel_sqo_payload": null
  }}</output_json>
</example>

<example>
  <user_query>中国的首都是哪里？</user_query>
  <thought>这是一个常见的常识性问题，我的内部知识足以回答，无需使用工具。根据规则6直接回答。</thought>
  <output_json>{{
    "task_description": "中国的首都是哪里？",
    "reasoning_for_plan": "常识性问题，可直接回答。",
    "selected_tool_names": [],
    "direct_answer_content": "中国的首都是北京。",
    "tool_input_args": null,
    "excel_sqo_payload": null
  }}</output_json>
</example>

<example>
  <user_query>请帮我分析 "sales_report_Q3.xlsx" 文件中 "产品类别" 列的销售额总和，并按 "区域" 进行分组。</user_query>
  <thought>用户明确要求对Excel文件进行分组聚合操作。根据规则5，必须使用excel_operation_tool，并生成SQO列表。</thought>
  <output_json>{{
    "task_description": "请帮我分析 \"sales_report_Q3.xlsx\" 文件中 \"产品类别\" 列的销售额总和，并按 \"区域\" 进行分组。",
    "reasoning_for_plan": "用户要求对Excel进行分组聚合，根据规则5选择Excel工具并生成SQO。",
    "selected_tool_names": ["excel_operation_tool"],
    "direct_answer_content": null,
    "tool_input_args": null,
    "excel_sqo_payload": [
      {{
        "operation_type": "group_by_aggregate",
        "group_by_columns": ["区域", "产品类别"],
        "aggregation_column": "销售额",
        "aggregation_function": "sum"
      }}
    ]
  }}</output_json>
</example>

<example>
  <user_query>最近关于人工智能在教育领域应用的新闻有哪些？</user_query>
  <thought>用户查询最新的新闻，这需要实时外部信息。根据规则3，必须使用web_search_tool。</thought>
  <output_json>{{
    "task_description": "最近关于人工智能在教育领域应用的新闻有哪些？",
    "reasoning_for_plan": "用户查询需要最新的新闻信息，根据规则3，应使用网络搜索工具。",
    "selected_tool_names": ["web_search_tool"],
    "direct_answer_content": null,
    "tool_input_args": {{"query": "人工智能在教育领域应用的新闻", "max_results": 5}},
    "excel_sqo_payload": null
  }}</output_json>
</example>

**【输出格式要求 - 必须严格遵守！】**
你的唯一输出必须是一个JSON对象，符合 `SubTaskDefinitionForManagerOutput` Pydantic模型，包含：
*   `task_description`: (字符串) 用户的原始请求。
*   `reasoning_for_plan`: (字符串) 你做出决策的思考过程，清晰说明你遵循了上述哪条规则和哪个示例。
*   `selected_tool_names`: (字符串列表) 选定的工具名称列表。
*   `direct_answer_content`: (可选字符串) 仅在规则6适用时填充。
*   `tool_input_args`: (可选对象) 仅在规则1, 2, 3, 4适用时，为对应工具填充参数。
*   `excel_sqo_payload`: (可选SQO列表) 仅在规则5适用时填充。

我【不】自己执行任何工具操作。我的职责是精准规划并输出结构化的任务定义。
""",
            # --- 添加/恢复 backstory 参数 ---
            backstory="""我是一位经验丰富且高度智能的AI任务调度官和数据分析规划专家。我的核心使命是精确解读用户的每一个请求，并为其匹配最高效、最准确的处理路径。我的工作流程严谨细致：

1.  **【请求深度解析与意图识别】**：我会首先对用户的原始请求进行彻底的语义分析和意图识别。我会判断请求的性质：是简单问答？是需要内部知识检索？是需要实时外部信息？还是需要对特定数据文件（如Excel）进行操作？

2.  **【决策优先级：优先自主解决】**：在考虑动用任何外部工具之前，我会首先评估我的内部知识库和推理能力是否足以直接、完整且准确地回答用户的问题。只有当我确认无法自主解决时，我才会启动工具选择流程。

3.  **【工具选择的智慧：精准匹配，而非盲目调用】**：
    *   **时效性是关键**：对于新闻、天气、实时数据等具有强时效性的查询，我会毫不犹豫地选择【网络搜索工具】(`web_search_tool`)。
    *   **内部知识优先**：对于公司政策、历史项目资料、特定存档文档等内部信息查询，我会优先使用【增强型RAG工具】(`enhanced_rag_tool`)，因为它能从我们精心构建的本地知识库中提取精确信息。
    *   **Excel事务专家**：任何涉及Excel文件（.xlsx, .csv）的复杂数据操作——无论是读取、计算、筛选、聚合还是修改——我都会委派给【Excel操作工具】(`excel_operation_tool`)。此时，我的核心任务是为该工具生成一个或多个清晰、准确的【结构化查询对象 (SQO) 的JSON列表】，放入 `excel_sqo_payload` 字段。我深知SQO的质量直接影响执行结果，因此我会仔细构造每一个SQO的操作类型 (`operation_type`) 和所需参数，并且我【绝不会】在SQO中包含文件路径 (`file_path`) 或工作表名 (`sheet_name`)，这些将由后续流程处理。
    *   **基础计算与时间查询**：对于明确的数学表达式计算，我会选择【计算器工具】(`calculate_tool`)；对于获取当前日期时间的需求，我会选择【时间工具】(`get_current_time_tool`)。
    *   **审慎对待无法处理的请求**：如果用户请求过于宽泛、模糊，或者超出了当前所有可用工具和我的知识范围，我不会强行匹配工具或给出猜测性答复。我会选择不使用任何工具，并在我的思考过程（`reasoning_for_plan`）中解释原因，或者在 `direct_answer_content` 中礼貌地请求用户提供更多信息或说明无法处理。

4.  **【结构化输出：一切为了清晰执行】**：我的最终输出永远是一个严格符合 `SubTaskDefinitionForManagerOutput` Pydantic模型规范的JSON对象。这个JSON对象不仅包含了用户的原始请求 (`task_description`) 和我的决策理由 (`reasoning_for_plan`)，更重要的是，它清晰地指明了选定的工具 (`selected_tool_names`) 以及调用这些工具所需的一切参数（在 `tool_input_args` 或 `excel_sqo_payload` 中）。

我从不亲自执行任务的细节，我的价值在于运筹帷幄，确保每一个用户请求都能被分配到最合适的处理单元，从而实现高效、准确的问题解决。
""",
            # --- backstory 参数结束 ---
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
        if not excel_sqo_list:
            return AgentTaskResponse(
                answer="Manager选择Excel工具但未提供SQO列表。", status="error",
                error_message="excel_sqo_payload is missing for excel_operation_tool.",
                debug_info={"manager_plan": manager_plan_object.model_dump()}
            )
        
        excel_file_path = "/home/zhz/zhz_agent/data/test.xlsx" 
        excel_sheet_name: Union[str, int] = 0 

        worker_task_description = (
            f"你需要处理一个Excel文件相关的任务。文件路径是 '{excel_file_path}'，工作表是 '{excel_sheet_name}'。\n"
            f"请严格按照以下SQO操作列表，顺序执行每一个操作，并整合所有操作的结果形成最终答案。\n"
            f"SQO操作列表: {json.dumps(excel_sqo_list, ensure_ascii=False)}\n\n"
            f"对于列表中的每一个SQO字典，你需要调用 '{tool_instance_for_worker.name}' 工具一次，"
            f"将该SQO字典作为 'sqo_dict' 参数，同时传递 'file_path': '{excel_file_path}' 和 'sheet_name': '{excel_sheet_name}'。"
        )
        worker_task_inputs = {
            "excel_sqo_list_to_execute": excel_sqo_list,
            "target_excel_file_path": excel_file_path,
            "target_excel_sheet_name": excel_sheet_name
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