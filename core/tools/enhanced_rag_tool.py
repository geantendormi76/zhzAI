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
        目标是返回 RAG 服务本身的业务响应字典。
        """
        tool_path_on_mcp = f"{self.mcp_service_name}/{self.mcp_tool_path}"
        logger.critical(f"!!! EnhancedRAGTool._acall_mcp: ENTERING. Calling MCP endpoint '{tool_path_on_mcp}' with payload: {payload}")
        try:
            mcp_wrapper_response = await call_mcpo_tool(tool_path_on_mcp, payload)

            logger.critical(f"!!! EnhancedRAGTool._acall_mcp: RECEIVED from call_mcpo_tool - Type: {type(mcp_wrapper_response)}")
            if isinstance(mcp_wrapper_response, dict):
                logger.critical(f"!!! EnhancedRAGTool._acall_mcp: RECEIVED keys: {list(mcp_wrapper_response.keys())}")
            logger.critical(f"!!! EnhancedRAGTool._acall_mcp: RECEIVED content preview: {str(mcp_wrapper_response)[:500]}")

            # 场景1: call_mcpo_tool 返回 MCP 框架的成功响应 ({"success": True, "data": RAG_response})
            if isinstance(mcp_wrapper_response, dict) and mcp_wrapper_response.get("success") is True:
                rag_service_data = mcp_wrapper_response.get("data")
                if isinstance(rag_service_data, dict):
                    logger.info("!!! EnhancedRAGTool._acall_mcp: MCP call successful (success:True), returning RAG 'data' field.")
                    return rag_service_data
                else:
                    logger.error(f"!!! EnhancedRAGTool._acall_mcp: MCP 'data' field is not a dict. Got {type(rag_service_data)}. Payload: {payload}")
                    return {
                        "status": "error", "original_query": payload.get("query", "N/A"), "retrieved_context_docs": [],
                        "error_message": f"Internal error: RAG service data from MCP was not a dict. Got: {str(rag_service_data)[:200]}",
                        "error_code": "INVALID_RAG_DATA_FROM_MCP_SUCCESS"
                    }

            # 场景2: call_mcpo_tool 返回 MCP 框架的失败响应 ({"success": False, "error": ...})
            elif isinstance(mcp_wrapper_response, dict) and mcp_wrapper_response.get("success") is False:
                error_msg = mcp_wrapper_response.get("error", "Unknown error from MCP call wrapper.")
                error_type = mcp_wrapper_response.get("error_type", "MCP_CALL_FAILED")
                status_code = mcp_wrapper_response.get("status_code")
                logger.error(f"!!! EnhancedRAGTool._acall_mcp: MCP call failed (success:False). Type: {error_type}, Error: {error_msg}, Status Code: {status_code}")
                return { # 构造一个符合RAG错误格式的字典
                    "status": "error", "original_query": payload.get("query", "N/A"), "retrieved_context_docs": [],
                    "error_message": error_msg, "error_code": error_type,
                    "debug_info": {"mcp_error_type": error_type, "mcp_status_code": status_code}
                }

            # 场景3: call_mcpo_tool 直接返回了 RAG 服务的业务 JSON (例如，包含 "status" 但不含 "success")
            # 这种情况通常是 MCP 代理透传了下游服务的响应。
            elif isinstance(mcp_wrapper_response, dict) and "status" in mcp_wrapper_response:
                logger.info("!!! EnhancedRAGTool._acall_mcp: Received a direct business response from RAG service (contains 'status' key).")
                return mcp_wrapper_response

            # 场景4: call_mcpo_tool 返回了其他无法识别的字典格式
            elif isinstance(mcp_wrapper_response, dict):
                logger.error(f"!!! EnhancedRAGTool._acall_mcp: call_mcpo_tool returned an unexpected dictionary format: {str(mcp_wrapper_response)[:200]}")
                return {
                    "status": "error", "original_query": payload.get("query", "N/A"), "retrieved_context_docs": [],
                    "error_message": f"Internal error: MCP wrapper response format unknown. Got: {str(mcp_wrapper_response)[:200]}",
                    "error_code": "INVALID_MCP_WRAPPER_DICT_FORMAT_UNKNOWN"
                }
            
            # 场景5: call_mcpo_tool 返回的不是字典
            else:
                logger.error(f"!!! EnhancedRAGTool._acall_mcp: call_mcpo_tool returned non-dict type: {type(mcp_wrapper_response)}. Content: {str(mcp_wrapper_response)[:200]}")
                return {
                    "status": "error", "original_query": payload.get("query", "N/A"), "retrieved_context_docs": [],
                    "error_message": f"Internal error: MCP wrapper returned non-dict. Got: {str(mcp_wrapper_response)[:200]}",
                    "error_code": "INVALID_MCP_WRAPPER_NON_DICT"
                }

        except Exception as e_acall:
            logger.critical(f"!!! EnhancedRAGTool._acall_mcp: EXCEPTION during call_mcpo_tool or its processing: {e_acall}", exc_info=True)
            return {
                "status": "error", "original_query": payload.get("query", "N/A"), "retrieved_context_docs": [],
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
        rag_service_response_data: Dict[str, Any] = {
            "status": "error", "original_query": query, "retrieved_context_docs": [],
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
                "status": "error", "original_query": query, "retrieved_context_docs": [],
                "error_message": f"RAG tool execution timed out: {str(e_timeout)}",
                "error_code": "ASYNC_TIMEOUT_IN_RUN"
            }
        except Exception as e_run:
            logger.critical(f"!!! EnhancedRAGTool._run: EXCEPTION during RAG tool execution: {e_run}", exc_info=True)
            rag_service_response_data = {
                "status": "error", "original_query": query, "retrieved_context_docs": [],
                "error_message": f"Exception in _run: {str(e_run)}",
                "error_code": "RUN_UNHANDLED_EXCEPTION",
                "traceback": traceback.format_exc()
            }

        logger.critical(f"!!! EnhancedRAGTool._run: Raw RAG service data before _handle_mcp_result: {str(rag_service_response_data)[:500]}")

        if not isinstance(rag_service_response_data, dict):
            logger.error(f"!!! EnhancedRAGTool._run: rag_service_response_data is not a dict, type: {type(rag_service_response_data)}. Wrapping in error dict.")
            rag_service_response_data = {
                "status": "error", "original_query": query, "retrieved_context_docs": [],
                "error_message": f"Internal error: _acall_mcp did not return a dict as expected. Got: {str(rag_service_response_data)[:200]}",
                "error_code": "INVALID_ACALL_RETURN_TYPE_FINAL"
            }
        # 确保 rag_service_response_data 至少包含 'status' 键，如果它在 _acall_mcp 中因为某些路径没有返回标准错误结构
        if "status" not in rag_service_response_data:
            logger.warning(f"!!! EnhancedRAGTool._run: rag_service_response_data missing 'status' key. Data: {str(rag_service_response_data)[:200]}. Defaulting to error status.")
            rag_service_response_data["status"] = "error"
            rag_service_response_data.setdefault("error_message", "Malformed response from RAG service call.")
            rag_service_response_data.setdefault("error_code", "MALFORMED_RAG_RESPONSE_IN_RUN")
            rag_service_response_data.setdefault("original_query", query)
            rag_service_response_data.setdefault("retrieved_context_docs", [])


        return self._handle_mcp_result(rag_service_response_data)

    def _handle_mcp_result(self, rag_service_data: Dict[str, Any]) -> str:
        """
        处理来自 RAG 服务 (zhz_rag_mcp_service) 的直接响应。
        """
        logger.info(f"EnhancedRAGTool._handle_mcp_result received RAG service data: {str(rag_service_data)[:500]}...")

        if not isinstance(rag_service_data, dict):
            error_msg = f"TOOL_ERROR: {self.name} received an invalid response format (expected dict from RAG service, got {type(rag_service_data)}). Content: {str(rag_service_data)[:200]}"
            logger.error(error_msg)
            return error_msg

        status_from_rag = rag_service_data.get("status")

        if status_from_rag == "success":
            final_answer = rag_service_data.get("final_answer")
            retrieved_docs_raw = rag_service_data.get("retrieved_context_docs")

            if final_answer is None:
                no_answer_msg = f"TOOL_INFO: {self.name} RAG service status is 'success', but did not provide a final_answer."
                logger.warning(no_answer_msg)
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
            return error_output
        else:
            unknown_status_msg = f"TOOL_ERROR: {self.name} received an unknown or missing status '{status_from_rag}' from RAG service."
            logger.error(f"{unknown_status_msg} Full RAG data: {str(rag_service_data)[:300]}")
            return unknown_status_msg