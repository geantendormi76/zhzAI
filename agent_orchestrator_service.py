# /home/zhz/zhz_agent/zhz_rag/api/agent_orchestrator_service.py

import os
import asyncio
import traceback
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from pydantic import BaseModel, Field

# --- LiteLLM (后续会在这里配置和使用) ---
import litellm 

# --- CrewAI (后续会在这里配置和使用) ---
# from crewai import Agent, Task, Crew, Process, LLM

# --- 自定义模块导入 (后续会逐步完善) ---
# from zhz_rag.utils.common_utils import call_mcpo_tool
# from zhz_rag.crewai_integration.tools import ... # 我们将定义新的核心工具类
# from zhz_agent_config.agents import create_manager_agent, create_worker_agent # 假设新的配置文件路径
# from zhz_agent_config.tasks import create_manager_task_definition, process_manager_output_and_execute_worker

# --- 配置 ---
AGENT_SERVICE_PORT = int(os.getenv("AGENT_SERVICE_PORT", 8090))
AGENT_SERVICE_HOST = "0.0.0.0"
# 本地 LLM 服务地址 (Qwen3)
LOCAL_QWEN_API_BASE = "http://localhost:8088/v1"
LOCAL_QWEN_MODEL_NAME_FOR_LITELLM = "local/qwen3-1.7b-gguf" # 与 test_litellm_local.py 中一致
# 云端 LiteLLM 网关地址 (用于 Gemini 等) - 先占位，后续替换为您的实际地址
CLOUD_LITELLM_GW_API_BASE = os.getenv("CLOUD_LITELLM_GW_API_BASE", "YOUR_CLOUD_LITELLM_GATEWAY_URL_HERE")
GEMINI_MODEL_NAME_FOR_LITELLM = "gemini/gemini-1.5-flash-latest" # 或您希望通过网关调用的模型

# --- Pydantic 模型定义 ---

class AgentTaskRequest(BaseModel):
    user_query: str = Field(description="用户的原始文本查询。")
    # local_file_path: Optional[str] = Field(None, description="如果操作涉及本地特定文件，请提供其绝对路径。")
    # session_id: Optional[str] = Field(None, description="可选的会话ID，用于支持多轮对话记忆。")

class AgentTaskResponse(BaseModel):
    answer: str = Field(description="Agent 系统生成的最终答案或响应。")
    status: str = Field(default="success", description="执行状态: 'success', 'needs_clarification', 'error', 'processing_plan'.")
    intermediate_plan: Optional[Dict[str, Any]] = Field(None, description="如果status是'processing_plan', 这里包含Manager Agent的规划结果。")
    error_message: Optional[str] = Field(None, description="如果发生错误，此字段包含错误信息。")
    debug_info: Optional[Dict[str, Any]] = Field(None, description="包含执行过程中的调试信息。")
    # token_usage: Optional[Dict[str, Any]] = Field(None, description="LLM token 使用情况统计。")

# --- 全局变量 (在 lifespan 中初始化) ---
# manager_llm: Optional[Any] = None # 将是 CrewAI LLM 兼容对象
# worker_llm: Optional[Any] = None  # 同上
# tool_kit: Dict[str, Any] = {}     # 存储实例化的工具

@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager_llm, worker_llm, tool_kit
    print("--- Agent Orchestrator Service: Lifespan startup ---")
    
    # TODO: 步骤3 - 初始化 LLM 实例 (通过 LiteLLM)
    # 例如:
    # manager_llm = ... # 创建一个使用 Gemini (通过云网关) 的 CrewAI LLM 包装器
    # worker_llm = ...  # 创建一个使用本地 Qwen3 的 CrewAI LLM 包装器

    # TODO: 步骤3 - 初始化核心工具实例
    # 例如:
    # rag_tool = EnhancedRAGTool()
    # excel_tool = ExecuteExcelSQOTool() # 确保它能正确调用本地代理
    # search_tool = DuckDuckGoSearchTool() # 确保它能正确调用MCP
    # tool_kit = {rag_tool.name: rag_tool, excel_tool.name: excel_tool, search_tool.name: search_tool}
    
    print("Agent Orchestrator Service: LLM 和工具将在后续步骤中在这里初始化。")
    yield
    print("--- Agent Orchestrator Service: Lifespan shutdown ---")

app = FastAPI(
    title="Agent Orchestrator Service",
    description="接收用户请求，通过Manager/Worker Agent模型进行任务规划和执行。",
    version="0.1.0",
    lifespan=lifespan
)

@app.post("/v1/execute_task", response_model=AgentTaskResponse)
async def execute_task_endpoint(request: AgentTaskRequest):
    print(f"Received agent task request: User Query='{request.user_query}'")
    
    # --- 临时的占位逻辑 ---
    # 在这里，我们将逐步集成 Manager Agent 的规划逻辑
    # 目前，我们先简单返回一个表示正在处理规划的响应

    # 模拟 Manager Agent 的初步规划 (后续替换为真实 CrewAI 调用)
    manager_plan = {
        "task_description": request.user_query,
        "selected_tool_names": ["TODO: Manager will select tools here"],
        "excel_sqo_payload": None, # 默认为 None
        "reasoning_for_plan": "TODO: Manager will explain its plan here, including why tools are needed or not."
    }

    if "excel" in request.user_query.lower() or ".xlsx" in request.user_query.lower():
        manager_plan["selected_tool_names"] = ["execute_excel_sqo"]
        manager_plan["excel_sqo_payload"] = [{"operation_type": "TODO: define_sqo_based_on_query"}]
        manager_plan["reasoning_for_plan"] = "用户请求涉及Excel文件，规划使用Excel工具。"
    elif "搜索" in request.user_query or "查找" in request.user_query or "最新" in request.user_query:
        manager_plan["selected_tool_names"] = ["duckduckgo_search"]
        manager_plan["reasoning_for_plan"] = "用户请求需要从网络获取信息，规划使用搜索工具。"
    elif "中国的首都是哪里" in request.user_query: # 模拟直接回答
        manager_plan["selected_tool_names"] = [] # 或者 ["direct_answer"]
        manager_plan["reasoning_for_plan"] = "这是一个常识性问题，LLM可以直接回答，无需工具。"
        # 理想情况下，这里 Manager Agent 会直接给出答案，我们暂时只返回规划
        # return AgentTaskResponse(answer="中国的首都是北京。", status="success", debug_info={"plan": manager_plan})

    return AgentTaskResponse(
        answer="Manager Agent is planning the task.", 
        status="processing_plan", # 一个新的状态，表示规划已生成，等待下一步执行
        intermediate_plan=manager_plan,
        debug_info={"message": "Full CrewAI logic not yet implemented."}
    )

if __name__ == "__main__":
    print(f"--- Starting Agent Orchestrator FastAPI Service on {AGENT_SERVICE_HOST}:{AGENT_SERVICE_PORT} ---")
    # 注意：如果这个服务由 mcpo 管理，这个 if __name__ == "__main__": 部分通常不会被 mcpo 调用。
    # mcpo 会直接通过模块和类名（如果服务本身是类）或直接运行脚本（如果服务是顶级 FastAPI app）来启动。
    # 为了方便独立测试，我们保留这个 uvicorn.run。
    uvicorn.run(app, host=AGENT_SERVICE_HOST, port=AGENT_SERVICE_PORT)