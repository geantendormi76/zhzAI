# /home/zhz/zhz_agent/core/tools/enhanced_rag_tool.py

import asyncio
from typing import Type, Optional, Dict, Any
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

# 假设 call_mcpo_tool 在 utils.common_utils 中
# 如果 agent_orchestrator_service.py 和 core 在同一个父级 zhz_agent 下
# 我们可以用 from ....utils import common_utils
# 或者更明确 from zhz_agent.utils.common_utils import call_mcpo_tool (如果PYTHONPATH设置正确或运行方式支持)
# 为了简单和在工具类内部直接可用，我们先用相对导入（如果目录结构支持）或绝对导入
from utils.common_utils import call_mcpo_tool # 假设 zhz_agent 在PYTHONPATH中或者运行上下文支持

# --- 日志记录 ---
import logging
logger = logging.getLogger(__name__)


class EnhancedRAGToolInput(BaseModel):
    query: str = Field(description="用户提出的原始查询文本。")
    top_k_vector: int = Field(default=5, description="期望检索的向量搜索结果数量。")
    top_k_kg: int = Field(default=3, description="期望检索的知识图谱结果数量。")
    top_k_bm25: int = Field(default=3, description="期望检索的 BM25 关键词搜索结果数量。")
    # top_k_final: int = Field(default=3, description="融合后最终返回的文档数。") # 这个参数通常在RAG服务内部处理或由FusionEngine处理

class EnhancedRAGTool(BaseTool):
    name: str = "enhanced_rag_tool"
    description: str = "【核心RAG工具】用于从本地知识库查找信息、回答复杂问题，整合了向量、关键词和图谱检索。"
    args_schema: Type[BaseModel] = EnhancedRAGToolInput
    # MCP 服务中 RAG 服务的名称和端点路径
    mcp_service_name: str = "zhz_rag_mcp_service" # 与 mcp_servers.json 中定义的服务名一致
    mcp_tool_path: str = "query_rag_v2" # RAG 服务提供的端点

    def _run(
        self,
        query: str,
        top_k_vector: int = 5,
        top_k_kg: int = 3,
        top_k_bm25: int = 3,
        **kwargs: Any # 捕获其他可能的参数
    ) -> str:
        """
        同步执行方法，CrewAI Agent 会调用这个。
        内部调用异步的 _arun 方法。
        """
        logger.info(f"EnhancedRAGTool._run called with query: '{query}'")
        payload = {
            "query": query,
            "top_k_vector": top_k_vector,
            "top_k_kg": top_k_kg,
            "top_k_bm25": top_k_bm25,
        }
        # CrewAI 的同步工具执行通常在一个单独的线程中运行 asyncio.run
        # 或者我们可以直接在这里使用 asyncio.run
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已经在运行的loop中，不能直接用 asyncio.run
                # 这种情况下，理想的做法是让 Agent 的执行流程本身是异步的
                # 但 CrewAI Agent 的 _run 通常是同步的。
                # 一个 hacky 的方法是创建一个新的线程来运行新的loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, self._acall_mcp(payload))
                    result = future.result(timeout=120) # 设置超时
            else:
                result = asyncio.run(self._acall_mcp(payload))
        except Exception as e:
            logger.error(f"Error running EnhancedRAGTool for query '{query}': {e}", exc_info=True)
            return f"Error executing RAG tool: {str(e)}"
        
        # _handle_mcp_result 应该返回一个字符串
        return self._handle_mcp_result(result)

    async def _acall_mcp(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        实际调用 MCP 服务的异步方法。
        """
        tool_path_on_mcp = f"{self.mcp_service_name}/{self.mcp_tool_path}"
        logger.info(f"EnhancedRAGTool: Calling MCP endpoint '{tool_path_on_mcp}' with payload: {payload}")
        mcp_response = await call_mcpo_tool(tool_path_on_mcp, payload)
        return mcp_response

    def _handle_mcp_result(self, mcp_response: Dict[str, Any]) -> str:
        """
        处理来自 MCP 服务的响应。
        RAG 服务成功时应该返回类似 HybridRAGResponse 的结构。
        """
        if mcp_response.get("error"):
            error_msg = f"Enhanced RAG tool failed: {mcp_response['error']}"
            logger.error(error_msg)
            return error_msg

        # 假设 RAG 服务成功时返回的 JSON 包含 "final_answer" 和 "retrieved_context_docs"
        final_answer = mcp_response.get("final_answer")
        retrieved_docs_raw = mcp_response.get("retrieved_context_docs")

        if final_answer is None:
            no_answer_msg = "RAG service did not provide a final answer."
            logger.warning(no_answer_msg)
            # 可以选择返回一个更友好的提示，或者包含调试信息
            # return f"{no_answer_msg} Raw MCP response: {json.dumps(mcp_response, ensure_ascii=False, indent=2)}"
            # 对于 Agent，可能直接返回一个指示性的字符串更好
            return "未能从知识库获取明确答案。"


        # CrewAI 工具的 _run 方法通常期望返回一个字符串
        # 我们可以将 RAG 的核心答案和一些上下文摘要组合起来
        # 或者只返回 final_answer，让 Agent 自行决定是否需要更多细节
        
        response_parts = [f"RAG Answer: {final_answer}"]
        if retrieved_docs_raw and isinstance(retrieved_docs_raw, list):
            response_parts.append("\n\nSupporting Context Snippets:")
            for i, doc_data in enumerate(retrieved_docs_raw[:2]): # 最多显示2个上下文片段的摘要
                content = doc_data.get("content", "N/A")
                source = doc_data.get("source_type", "N/A")
                score = doc_data.get("score", 0.0)
                response_parts.append(f"  - Source: {source}, Score: {score:.2f}, Content: {content[:100]}...")
        
        return "\n".join(response_parts)

# 确保 core/tools 目录也有一个 __init__.py 文件
# touch /home/zhz/zhz_agent/core/tools/__init__.py (如果不存在)