# /home/zhz/zhz_agent/core/tools/enhanced_rag_tool.py

import asyncio
from typing import Type, Optional, Dict, Any
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import json 
import traceback 

from utils.common_utils import call_mcpo_tool 

import logging
logger = logging.getLogger(__name__)


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
        """
        异步调用 MCP 工具服务，并处理其包装响应。
        如果 MCP 调用成功且其内部的 RAG 服务也成功，则返回 RAG 服务的实际数据。
        如果任何层面出错，则返回一个符合 RAG 服务错误格式的字典。
        """
        tool_path_on_mcp = f"{self.mcp_service_name}/{self.mcp_tool_path}"
        logger.critical(f"!!! EnhancedRAGTool._acall_mcp: ENTERING. Calling MCP endpoint '{tool_path_on_mcp}' with payload: {payload}")
        try:
            # call_mcpo_tool 返回的是一个包装后的响应，格式为:
            # {"success": True/False, "data": actual_tool_response_if_success, "error": error_msg_if_fail, ...}
            mcp_wrapper_response = await call_mcpo_tool(tool_path_on_mcp, payload)
            logger.critical(f"!!! EnhancedRAGTool._acall_mcp: call_mcpo_tool RETURNED wrapper: {str(mcp_wrapper_response)[:500]}")
            
            if isinstance(mcp_wrapper_response, dict) and mcp_wrapper_response.get("success") is True and "data" in mcp_wrapper_response:
                # MCP 调用成功，现在获取 RAG 服务本身的响应数据
                rag_service_actual_data = mcp_wrapper_response["data"]
                logger.info("!!! EnhancedRAGTool._acall_mcp: MCP call successful, returning actual RAG data.")
                # 确保 RAG 服务数据也是字典，如果不是，也视为错误
                if not isinstance(rag_service_actual_data, dict):
                    logger.error(f"!!! EnhancedRAGTool._acall_mcp: RAG service data ('data' field from MCP wrapper) is not a dict, got {type(rag_service_actual_data)}. Wrapping as error.")
                    return {
                        "status": "error", 
                        "error_message": f"Internal error: RAG service data was not a dict. Got: {str(rag_service_actual_data)[:200]}", 
                        "error_code": "INVALID_RAG_DATA_FROM_MCP"
                    }
                return rag_service_actual_data # 返回 RAG 服务的直接响应
            
            elif isinstance(mcp_wrapper_response, dict) and mcp_wrapper_response.get("success") is False:
                # MCP 调用本身失败（例如 HTTP 错误，超时等）
                error_msg = mcp_wrapper_response.get("error", "Unknown error from MCP call wrapper.")
                error_type = mcp_wrapper_response.get("error_type", "MCP_CALL_FAILED")
                logger.error(f"!!! EnhancedRAGTool._acall_mcp: MCP call failed. Type: {error_type}, Error: {error_msg}")
                return {
                    "status": "error", 
                    "error_message": error_msg,
                    "error_code": error_type 
                }
            else: 
                # call_mcpo_tool 返回了意外的格式 (不包含 "success" 键)
                logger.error(f"!!! EnhancedRAGTool._acall_mcp: call_mcpo_tool returned unexpected format: {type(mcp_wrapper_response)}. Content: {str(mcp_wrapper_response)[:200]}")
                return {
                    "status": "error", 
                    "error_message": f"Internal error: call_mcpo_tool returned unexpected format. Got: {str(mcp_wrapper_response)[:200]}", 
                    "error_code": "INVALID_MCP_WRAPPER_RESPONSE"
                }
        except Exception as e_acall:
            logger.critical(f"!!! EnhancedRAGTool._acall_mcp: EXCEPTION during call_mcpo_tool or its processing: {e_acall}", exc_info=True)
            return {
                "status": "error", 
                "error_message": f"Exception in _acall_mcp: {str(e_acall)}", 
                "error_code": "ACALL_MCP_UNHANDLED_EXCEPTION", 
                "traceback": traceback.format_exc()
            }

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
        # 初始化为 RAG 服务期望的错误格式
        rag_service_response_data: Dict[str, Any] = { 
            "status": "error", 
            "error_message": "Initialization error in _run before async call",
            "error_code": "RUN_INIT_ERROR"
        }

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
                    rag_service_response_data = future.result(timeout=120) 
                    logger.critical("!!! EnhancedRAGTool._run: future.result() RETURNED.")
            else:
                logger.critical("!!! EnhancedRAGTool._run: Loop is NOT running. Using asyncio.run directly.")
                rag_service_response_data = asyncio.run(self._acall_mcp(payload))
                logger.critical("!!! EnhancedRAGTool._run: asyncio.run(_acall_mcp) RETURNED.")
        
        except asyncio.TimeoutError as e_timeout: 
            logger.critical(f"!!! EnhancedRAGTool._run: ASYNCIO TIMEOUT during RAG tool execution: {e_timeout}", exc_info=True)
            rag_service_response_data = {
                "status": "error", 
                "error_message": f"RAG tool execution timed out: {str(e_timeout)}", 
                "error_code": "ASYNC_TIMEOUT_IN_RUN"
            }
        except Exception as e_run:
            logger.critical(f"!!! EnhancedRAGTool._run: EXCEPTION during RAG tool execution: {e_run}", exc_info=True)
            rag_service_response_data = {
                "status": "error", 
                "error_message": f"Exception in _run: {str(e_run)}", 
                "error_code": "RUN_UNHANDLED_EXCEPTION", 
                "traceback": traceback.format_exc()
            }
        
        logger.critical(f"!!! EnhancedRAGTool._run: Raw RAG service data before _handle_mcp_result: {str(rag_service_response_data)[:500]}")
        
        # 确保传递给 _handle_mcp_result 的是字典
        if not isinstance(rag_service_response_data, dict):
            logger.error(f"!!! EnhancedRAGTool._run: rag_service_response_data is not a dict, type: {type(rag_service_response_data)}. Wrapping in error dict.")
            rag_service_response_data = {
                "status": "error", 
                "error_message": f"Internal error: _acall_mcp did not return a dict as expected. Got: {str(rag_service_response_data)[:200]}", 
                "error_code": "INVALID_ACALL_RETURN_TYPE_FINAL"
            }

        return self._handle_mcp_result(rag_service_response_data)

    def _handle_mcp_result(self, rag_service_data: Dict[str, Any]) -> str:
        """
        处理来自 RAG 服务 (zhz_rag_mcp_service) 的直接响应。
        _acall_mcp 应该已经处理了 MCP 包装层，这里接收的是 RAG 服务本身的输出。
        """
        logger.info(f"EnhancedRAGTool._handle_mcp_result received RAG service data: {str(rag_service_data)[:500]}...")

        if not isinstance(rag_service_data, dict):
            error_msg = f"TOOL_ERROR: {self.name} received an invalid response format (expected dict from RAG service, got {type(rag_service_data)}). Content: {str(rag_service_data)[:200]}"
            logger.error(error_msg)
            return error_msg # 返回给 Agent 的错误信息

        status_from_rag = rag_service_data.get("status")
        
        if status_from_rag == "success":
            final_answer = rag_service_data.get("final_answer")
            retrieved_docs_raw = rag_service_data.get("retrieved_context_docs")

            if final_answer is None: # 即使 status 是 success，也可能没有 final_answer
                no_answer_msg = f"TOOL_INFO: {self.name} RAG service status is 'success', but did not provide a final_answer."
                logger.warning(no_answer_msg)
                # 对于 Agent，这仍然是一个有效的工具执行，只是结果是“未找到答案”
                return "RAG service processed successfully but found no specific answer." 

            response_parts = [f"RAG Answer: {str(final_answer).strip()}"] 
            if retrieved_docs_raw and isinstance(retrieved_docs_raw, list) and retrieved_docs_raw:
                response_parts.append("\n\nSupporting Context Snippets (Top 2):")
                for i, doc_data in enumerate(retrieved_docs_raw[:2]): 
                    content = doc_data.get("content", "N/A")
                    source = doc_data.get("source_type", "N/A")
                    score = doc_data.get("score") 
                    score_str = f"{score:.2f}" if isinstance(score, float) else str(score if score is not None else 'N/A')
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
            return error_output # 返回给 Agent 的错误信息
        else: 
            # 如果 status 字段不是 "success", "clarification_needed", 或 "error"
            unknown_status_msg = f"TOOL_ERROR: {self.name} received an unknown or missing status '{status_from_rag}' from RAG service."
            logger.error(f"{unknown_status_msg} Full RAG data: {str(rag_service_data)[:300]}")
            return unknown_status_msg # 返回给 Agent 的错误信息