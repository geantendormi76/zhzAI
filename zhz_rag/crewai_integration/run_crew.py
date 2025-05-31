# /home/zhz/zhz_agent/run_agent.py

import os
import json
import datetime

from crewai import Agent, Task, Crew, Process

# --- 导入我们自己的项目模块 (使用绝对导入) ---
from zhz_rag.crewai_integration.tools import HybridRAGTool, BaseMCPTool
from zhz_rag.config.pydantic_models import QueryRequest # 用于 RAG 工具的输入
from zhz_rag.utils.common_utils import call_mcpo_tool
from zhz_rag.llm.custom_crewai_llms import CustomGeminiLLM

# --- 环境配置 ---
from dotenv import load_dotenv
load_dotenv()

# --- CrewAI 基类和事件系统 ---
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Type
from crewai.llms.base_llm import BaseLLM as CrewAIBaseLLM
try:
    from crewai.utilities.events.base_event_listener import BaseEventListener as CrewAIBaseCallbackHandler
    from crewai.utilities.events import LLMCallStartedEvent, LLMCallCompletedEvent
    print("Successfully imported BaseEventListener and Event Types")
except ImportError:
    print("Failed to import BaseEventListener or Event Types, using dummy classes.")
    class CrewAIBaseCallbackHandler: pass
    class LLMCallStartedEvent: pass
    class LLMCallCompletedEvent: pass

# --- LiteLLM ---
import litellm

# --- 定义简单工具以供测试 ---
class SimpleToolInput(BaseModel):
    message: str = Field(description="A simple message string for the tool.")

class MySimpleTestTool(BaseTool):
    name: str = "MySimpleTestTool"
    description: str = "A very simple test tool that takes a message and returns it."
    args_schema: Type[BaseModel] = SimpleToolInput

    def _run(self, message: str) -> str:
        print(f"MySimpleTestTool received: {message}")
        return f"MySimpleTestTool processed: {message}"

# --- 配置 Agent 使用的 LLM 实例 ---
GEMINI_MODEL_NAME = "gemini/gemini-1.5-flash-latest"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("CRITICAL ERROR: GOOGLE_API_KEY or GEMINI_API_KEY not set.")
    exit(1)

# --- 定义详细的事件监听器 ---
class MyDetailedLogger(CrewAIBaseCallbackHandler):
    def __init__(self):
        super().__init__()
        print(f"[{datetime.datetime.now()}] MyDetailedLogger: 已初始化。")

    def setup_listeners(self, crewai_event_bus):
        print(f"[{datetime.datetime.now()}] MyDetailedLogger: 正在设置监听器...")

        @crewai_event_bus.on(LLMCallStartedEvent)
        def handle_llm_start(source, event: LLMCallStartedEvent):
            self.on_llm_start_logic(source, event)

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def handle_llm_completed(source, event: LLMCallCompletedEvent):
            self.on_llm_end_logic(source, event)

        print(f"[{datetime.datetime.now()}] MyDetailedLogger: 监听器设置完成。")

    def on_llm_start_logic(self, source, event: LLMCallStartedEvent):
        print(f"\n>>>> LLM 调用开始 (Event Logic) <<<<")
        llm_inputs = getattr(event, 'llm_inputs', {})
        messages = llm_inputs.get('messages')
        tools = llm_inputs.get('tools')
        print(f"来源 (Source): {source}")
        if messages:
            print("消息 (来自 event.llm_inputs):")
            if isinstance(messages, list) and len(messages) > 0:
                first_message = messages[0]
                if isinstance(first_message, dict) and 'content' in first_message:
                    content_snippet = str(first_message.get('content', ''))[:300]
                    print(f"   Role: {first_message.get('role')}, Content Snippet: {content_snippet}...")
                else:
                     print(f"  First message (raw): {first_message}")
            else:
                 print(f"  Messages (raw): {messages}")
        else:
            print("消息 (来自 event.llm_inputs): 无")
        if tools:
            print("工具 (来自 event.llm_inputs):")
            try:
                print(f"  {json.dumps(tools, indent=2, ensure_ascii=False)}")
            except Exception as e:
                print(f"  无法序列化工具为 JSON: {e}. 工具: {tools}")
        else:
            print("工具 (来自 event.llm_inputs): 无")
        print("----------------------------------")

    def on_llm_end_logic(self, source, event: LLMCallCompletedEvent):
        print(f"\n>>>> LLM 调用结束 (Event Logic) <<<<")
        response = getattr(event, 'llm_output', None)
        print(f"来源 (Source): {source}")
        if response:
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message') and choice.message:
                    print(f"  消息内容: {choice.message.content}")
                    if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                        print(f"  工具调用: {choice.message.tool_calls}")
                    else:
                        print(f"  工具调用: 无")
            elif hasattr(response, 'content'):
                print(f"  响应内容: {response.content}")
            else:
                print(f"  LLM 响应 (来自 event.llm_output): {str(response)[:500]}...")
        else:
            print("  在 event.llm_output 中未找到响应对象。")
        print("----------------------------------")

# --- 实例化 CustomGeminiLLM ---
custom_llm_tool_config = {"function_calling_config": {"mode": "AUTO"}}
zhz_agent_tool = HybridRAGTool()
researcher_tools = [zhz_agent_tool]

llm_for_agent = CustomGeminiLLM(
    model=GEMINI_MODEL_NAME,
    api_key=GEMINI_API_KEY,
    temperature=0.1,
    max_tokens=2048,
    tool_config=custom_llm_tool_config,
    agent_tools=researcher_tools # 传递工具列表以供缓存
)
print(f"Custom Agent LLM configured: {GEMINI_MODEL_NAME} with custom tool_config")

# --- 设置 BaseMCPTool 的调用器 ---
BaseMCPTool.set_mcpo_caller(call_mcpo_tool)

# --- 定义 Agents ---
researcher_agent = Agent(
    role='信息检索专家',
    goal='准确地回答用户查询，并且只使用提供的工具。',
    backstory=(
        "你是一位高级AI助手，专注于信息检索。"
        "你的专长在于高效地利用工具来查找最相关和最精确的答案来回应用户的查询。"
    ),
    llm=llm_for_agent,
    tools=researcher_tools,
    verbose=True,
    allow_delegation=False,
)

writer_agent = Agent(
    role='报告撰写专家',
    goal='根据提供的信息，撰写清晰、结构良好且富有洞察力的报告。',
    backstory=(
        "您是一位资深的报告撰写专家，拥有出色的分析和写作能力。"
        "您擅长将复杂的信息提炼成易于理解的报告，并能根据不同的输出状态（答案、澄清、错误）"
        "灵活调整报告内容和格式。"
    ),
    llm=llm_for_agent,
    verbose=True,
    allow_delegation=False,
)

# --- 定义 Tasks (包含上下文传递修复) ---
research_task_description = """你收到了来自用户的以下查询：

'{query}'

你应该使用提供的 `HybridRAGQueryTool` 工具来处理这个查询。
如果这个工具需要 `top_k_vector`, `top_k_kg`, 或 `top_k_bm25` 这些参数，请使用以下建议值：
top_k_vector: 5, top_k_kg: 3, top_k_bm25: 5。
在使用完必要的工具后，你的最终输出应该是（使用中文）：'我的最终答案是：' 
后跟你使用的工具返回的精确、完整的JSON字符串输出内容。"""

research_task_expected_output = "短语 '我的最终答案是：' 后跟你使用的工具返回的精确、完整的JSON字符串输出内容。"

research_task = Task(
    description=research_task_description,
    expected_output=research_task_expected_output,
    agent=researcher_agent,
)

report_writing_task = Task(
    description="""根据【前一个任务】（信息检索专家）提供的RAG工具输出（它是一个JSON字符串），生成一份报告或响应。
请仔细分析这个JSON字符串输出，它应该包含一个 'status' 字段。
1. 如果 'status' 是 'success'，则提取 'final_answer' 字段的内容，并基于此答案撰写一份简洁的报告。
2. 如果 'status' 是 'clarification_needed'，则提取 'clarification_question' 字段的内容，并向用户明确指出需要澄清的问题，例如：'系统需要澄清：[澄清问题]'。
3. 如果 'status' 是 'error'，则提取 'error_message' (或 'error') 字段的内容，并向用户报告错误，例如：'RAG服务发生错误：[错误信息]'。
你的最终输出必须是清晰、专业且符合上述情况的报告或响应。""",
    expected_output="一份清晰的报告，或者一个明确的澄清请求，或者一个错误报告。",
    agent=writer_agent,
    context=[research_task],
)

# --- 实例化监听器 ---
my_event_logger = MyDetailedLogger()

# --- 定义 Crew (添加 event_listeners) ---
office_brain_crew = Crew(
    agents=[researcher_agent, writer_agent],
    tasks=[research_task, report_writing_task],
    process=Process.sequential,
    verbose=True,
    event_listeners=[my_event_logger] # <<< --- 激活事件监听器 ---
)

# --- 启动 Crew ---
if __name__ == "__main__":
    print("--- 启动智能助手终端大脑 Crew (使用 CustomGeminiLLM 和事件监听器) ---")
    user_query_input = "公司2024年第一季度在华东和华北的总销售额一共是多少？"
    # --- 修复：kickoff inputs 只包含 query ---
    inputs = {'query': user_query_input}
    result = office_brain_crew.kickoff(inputs=inputs)
    print("\n\n=== 最终报告 ===\n")
    if hasattr(result, 'raw'):
        print(result.raw)
    else:
        print(result)
    print("\n--- Crew 任务完成 ---")