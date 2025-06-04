# /home/zhz/zhz_agent/core/tools/enhanced_rag_tool.py

import asyncio
from typing import Type, Optional, Dict, Any
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import json # <--- 确保导入 json
import traceback # <--- 确保导入 traceback

# 假设 call_mcpo_tool 在 utils.common_utils 中
# 如果 agent_orchestrator_service.py 和 core 在同一个父级 zhz_agent 下
# 我们可以用 from ....utils import common_utils
# 或者更明确 from zhz_agent.utils.common_utils import call_mcpo_tool (如果PYTHONPATH设置正确或运行方式支持)
# 为了简单和在工具类内部直接可用，我们先用相对导入（如果目录结构支持）或绝对导入
from utils.common_utils import call_mcpo_tool # 假设 zhz_agent 在PYTHONPATH中或者运行上下文支持

# --- 日志记录 ---
import logging
logger = logging.getLogger(__name__)
# 确保这个logger被配置了，例如在 agent_orchestrator_service.py 的开头配置根logger

class EnhancedRAGToolInput(BaseModel):
    query: str = Field(description="用户提出的原始查询文本。")
    top_k_vector: int = Field(default=5, description="期望检索的向量搜索结果数量。")
    top_k_kg: int = Field(default=3, description="期望检索的知识图谱结果数量。")
    top_k_bm25: int = Field(default=3, description="期望检索的 BM25 关键词搜索结果数量。")

class EnhancedRAGTool(BaseTool):
    name: str = "enhanced_rag_tool"
    description: str = "【核心RAG工具】用于从本地知识库查找信息、回答复杂问题，整合了向量、关键词和图谱检索。"
    args_schema: Type[BaseModel] = EnhancedRAGToolInput
    mcp_service_name: str = "zhz_rag_mcp_service" 
    mcp_tool_path: str = "query_rag_v2" 

    async def _acall_mcp(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        tool_path_on_mcp = f"{self.mcp_service_name}/{self.mcp_tool_path}"
        logger.critical(f"!!! EnhancedRAGTool._acall_mcp: ENTERING. Calling MCP endpoint '{tool_path_on_mcp}' with payload: {payload}")
        try:
            mcp_response = await call_mcpo_tool(tool_path_on_mcp, payload)
            logger.critical(f"!!! EnhancedRAGTool._acall_mcp: call_mcpo_tool RETURNED: {str(mcp_response)[:500]}")
            return mcp_response
        except Exception as e_acall:
            logger.critical(f"!!! EnhancedRAGTool._acall_mcp: EXCEPTION during call_mcpo_tool: {e_acall}", exc_info=True)
            return {"success": False, "error": f"Exception in _acall_mcp: {str(e_acall)}", "error_type": "ACALL_MCP_EXCEPTION", "traceback": traceback.format_exc()}

    def _run(
        self,
        query: str,
        top_k_vector: int = 5,
        top_k_kg: int = 3,
        top_k_bm25: int = 3,
        **kwargs: Any 
    ) -> str:
        logger.critical(f"!!! EnhancedRAGTool._run: ENTERING. Query: '{query}'")
        payload = {
            "query": query,
            "top_k_vector": top_k_vector,
            "top_k_kg": top_k_kg,
            "top_k_bm25": top_k_bm25,
        }
        raw_mcp_call_result = {"success": False, "error": "Initialization error in _run", "error_type": "RUN_INIT_ERROR"}

        try:
            logger.critical("!!! EnhancedRAGTool._run: Attempting to get event loop...")
            loop = asyncio.get_event_loop()
            logger.critical(f"!!! EnhancedRAGTool._run: Event loop obtained. Is running: {loop.is_running()}")

            if loop.is_running():
                logger.critical("!!! EnhancedRAGTool._run: Loop is running. Using ThreadPoolExecutor.")
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    logger.critical("!!! EnhancedRAGTool._run: Submitting _acall_mcp to executor...")
                    future = executor.submit(asyncio.run, self._acall_mcp(payload))
                    logger.critical("!!! EnhancedRAGTool._run: Waiting for future.result()...")
                    raw_mcp_call_result = future.result(timeout=120) 
                    logger.critical("!!! EnhancedRAGTool._run: future.result() RETURNED.")
            else:
                logger.critical("!!! EnhancedRAGTool._run: Loop is NOT running. Using asyncio.run directly.")
                raw_mcp_call_result = asyncio.run(self._acall_mcp(payload))
                logger.critical("!!! EnhancedRAGTool._run: asyncio.run(_acall_mcp) RETURNED.")
        
        except asyncio.TimeoutError as e_timeout: # 更具体地捕获 submit/result 的超时
            logger.critical(f"!!! EnhancedRAGTool._run: ASYNCIO TIMEOUT during RAG tool execution: {e_timeout}", exc_info=True)
            raw_mcp_call_result = {"success": False, "error": f"RAG tool execution timed out: {str(e_timeout)}", "error_type": "ASYNC_TIMEOUT_IN_RUN"}
        except Exception as e_run:
            logger.critical(f"!!! EnhancedRAGTool._run: EXCEPTION during RAG tool execution: {e_run}", exc_info=True)
            raw_mcp_call_result = {"success": False, "error": f"Exception in _run: {str(e_run)}", "error_type": "RUN_EXCEPTION", "traceback": traceback.format_exc()}
        
        logger.critical(f"!!! EnhancedRAGTool._run: Raw result before _handle_mcp_result: {str(raw_mcp_call_result)[:500]}")
        
        # 确保传递给 _handle_mcp_result 的是字典
        if not isinstance(raw_mcp_call_result, dict):
            logger.error(f"!!! EnhancedRAGTool._run: raw_mcp_call_result is not a dict, type: {type(raw_mcp_call_result)}. Wrapping in error dict.")
            raw_mcp_call_result = {"success": False, "error": f"Internal error: _acall_mcp did not return a dict. Got: {str(raw_mcp_call_result)[:200]}", "error_type": "INVALID_ACALL_RETURN_TYPE"}

        return self._handle_mcp_result(raw_mcp_call_result)


    async def _acall_mcp(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        tool_path_on_mcp = f"{self.mcp_service_name}/{self.mcp_tool_path}"
        logger.critical(f"!!! EnhancedRAGTool._acall_mcp: ENTERING. Calling MCP endpoint '{tool_path_on_mcp}' with payload: {payload}")
        try:
            mcp_response = await call_mcpo_tool(tool_path_on_mcp, payload)
            logger.critical(f"!!! EnhancedRAGTool._acall_mcp: call_mcpo_tool RETURNED: {str(mcp_response)[:500]}")
            # 确保返回的是字典，如果不是，也包装成错误字典
            if not isinstance(mcp_response, dict):
                logger.error(f"!!! EnhancedRAGTool._acall_mcp: call_mcpo_tool did not return a dict, got {type(mcp_response)}. Wrapping.")
                return {"success": False, "error": f"Internal error: call_mcpo_tool did not return a dict. Got: {str(mcp_response)[:200]}", "error_type": "INVALID_CALL_MCPO_RETURN_TYPE"}
            return mcp_response
        except Exception as e_acall:
            logger.critical(f"!!! EnhancedRAGTool._acall_mcp: EXCEPTION during call_mcpo_tool: {e_acall}", exc_info=True)
            return {"success": False, "error": f"Exception in _acall_mcp: {str(e_acall)}", "error_type": "ACALL_MCP_EXCEPTION", "traceback": traceback.format_exc()}


    def _handle_mcp_result(self, mcp_response: Dict[str, Any]) -> str:
        """
        处理来自 call_mcpo_tool 的结构化响应。
        """
        logger.info(f"EnhancedRAGTool._handle_mcp_result received: {str(mcp_response)[:500]}...")

        if not isinstance(mcp_response, dict):
            error_msg = f"TOOL_ERROR: {self.name} received an invalid response format from MCP call (expected dict, got {type(mcp_response)}). Content: {str(mcp_response)[:200]}"
            logger.error(error_msg)
            return error_msg

        if mcp_response.get("success") is True and "data" in mcp_response:
            rag_service_data = mcp_response["data"]
            status_from_rag = rag_service_data.get("status")
            
            if status_from_rag == "success":
                final_answer = rag_service_data.get("final_answer")
                retrieved_docs_raw = rag_service_data.get("retrieved_context_docs")

                if final_answer is None:
                    no_answer_msg = f"TOOL_INFO: {self.name} succeeded, but RAG service did not provide a final answer in its 'success' response."
                    logger.warning(no_answer_msg)
                    return "未能从知识库获取明确答案，但RAG服务调用成功。"

                response_parts = [f"RAG Answer: {final_answer}"]
                if retrieved_docs_raw and isinstance(retrieved_docs_raw, list) and retrieved_docs_raw:
                    response_parts.append("\n\nSupporting Context Snippets (Top 2):")
                    for i, doc_data in enumerate(retrieved_docs_raw[:2]): 
                        content = doc_data.get("content", "N/A")
                        source = doc_data.get("source_type", "N/A")
                        score = doc_data.get("score", 0.0)
                        score_str = f"{score:.2f}" if isinstance(score, float) else str(score)
                        response_parts.append(f"  - Source: {source}, Score: {score_str}, Content: {str(content)[:100]}...")
                
                final_tool_output_str = "\n".join(response_parts)
                logger.info(f"EnhancedRAGTool: Successfully processed RAG success response. Output for agent (first 200 chars): {final_tool_output_str[:200]}...")
                return final_tool_output_str

            elif status_from_rag == "clarification_needed":
                clarification_question = rag_service_data.get("clarification_question", "需要您提供更多信息。")
                uncertainty_reason = rag_service_data.get("debug_info", {}).get("uncertainty_reason", "未知原因")
                clarification_output = f"CLARIFICATION_NEEDED: {clarification_question} (Reason: {uncertainty_reason})"
                logger.info(f"EnhancedRAGTool: RAG service requires clarification. Output for agent: {clarification_output}")
                return clarification_output
                
            elif status_from_rag == "error": 
                rag_error_msg = rag_service_data.get("error_message", "RAG服务内部发生未知错误。")
                rag_error_code = rag_service_data.get("error_code", "RAG_UNKNOWN_ERROR")
                error_output = f"TOOL_ERROR: {self.name} failed. RAG service reported an error (Code: {rag_error_code}): {rag_error_msg}"
                logger.error(error_output)
                return error_output
            else: 
                unknown_status_msg = f"TOOL_ERROR: {self.name} received an unknown status '{status_from_rag}' from RAG service."
                logger.error(f"{unknown_status_msg} Full RAG data: {str(rag_service_data)[:300]}")
                return unknown_status_msg
        
        elif mcp_response.get("success") is False:
            error_msg_from_call = mcp_response.get("error", "MCP调用失败，原因未知。")
            error_type_from_call = mcp_response.get("error_type", "UNKNOWN_MCP_CALL_ERROR")
            status_code_from_call = mcp_response.get("status_code")
            
            formatted_error = f"TOOL_ERROR: {self.name} failed during MCP call. Type: {error_type_from_call}."
            if status_code_from_call:
                formatted_error += f" Status: {status_code_from_call}."
            formatted_error += f" Message: {str(error_msg_from_call)[:200]}" 
            
            logger.error(formatted_error)
            logger.debug(f"Full MCP error response in _handle_mcp_result: {mcp_response}")
            return formatted_error
        else:
            malformed_response_msg = f"TOOL_ERROR: {self.name} received a malformed or incomplete response from MCP call. Raw: {str(mcp_response)[:200]}"
            logger.error(malformed_response_msg)
            return malformed_response_msg