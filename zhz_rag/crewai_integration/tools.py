# zhz_agent/custom_crewai_tools.py

import os
import json
import asyncio
import traceback
from typing import Type, List, Dict, Any, Optional, ClassVar
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import httpx

# 从 zhz_agent.pydantic_models 导入 QueryRequest
from zhz_rag.config.pydantic_models import QueryRequest # 用于 RAG 工具的输入

# MCPO 代理的基地址
MCPO_BASE_URL = os.getenv("MCPO_BASE_URL", "http://localhost:8006")

class BaseMCPTool(BaseTool):
    mcpo_base_url: str = MCPO_BASE_URL
    _call_mcpo_func: ClassVar[callable] = None

    @classmethod
    def set_mcpo_caller(cls, caller: callable):
        cls._call_mcpo_func = caller

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def _call_mcpo_endpoint(self, service_and_tool_path: str, payload: dict) -> dict | str:
        api_url = f"{self.mcpo_base_url}/{service_and_tool_path}"
        cleaned_payload = {k: v for k, v in payload.items() if v is not None}
        print(f"BaseMCPTool: Calling mcpo endpoint: POST {api_url} with payload: {json.dumps(cleaned_payload, ensure_ascii=False)}")
        
        # --- [修改] 移除 proxies=None 参数 ---
        async with httpx.AsyncClient(trust_env=False) as client:
            response = None
            try:
                headers = {"Content-Type": "application/json"}
                response = await client.post(api_url, json=cleaned_payload, headers=headers, timeout=300.0)
                print(f"BaseMCPTool: mcpo status code for {service_and_tool_path}: {response.status_code}")
                print(f"BaseMCPTool: mcpo response headers for {service_and_tool_path}: {response.headers}") # <--- 新增日志
                # 尝试分块读取响应或提前获取少量内容进行日志记录，以防响应过大卡住 .text 或 .json()
                try:
                    response_text_snippet = await response.aread(num_bytes=1024) # 读取前1KB
                    print(f"BaseMCPTool: mcpo response text snippet (first 1KB) for {service_and_tool_path}: {response_text_snippet.decode(errors='ignore')}")
                except Exception as e_read:
                    print(f"BaseMCPTool: Error reading response snippet: {e_read}")

                if response.status_code == 200:
                    try:
                        # print(f"BaseMCPTool: mcpo raw response text for {service_and_tool_path}: {response.text}") # 如果怀疑内容问题，可以取消注释，但小心大响应
                        return response.json()
                    except json.JSONDecodeError:
                        print(f"BaseMCPTool Warning: mcpo returned status 200 but response is not JSON for '{service_and_tool_path}'. Returning raw text.")
                        return response.text
                else:
                    error_text = f"mcpo call to '{service_and_tool_path}' failed with status {response.status_code}. Response: {response.text[:500]}..."
                    print(f"BaseMCPTool Error: {error_text}")
                    return {"error": error_text, "status_code": response.status_code}
            except httpx.RequestError as exc:
                error_msg = f"BaseMCPTool HTTP RequestError calling mcpo tool '{service_and_tool_path}': {type(exc).__name__} - {exc}"
                print(f"BaseMCPTool Error: {error_msg}"); traceback.print_exc()
                return {"error": error_msg, "exception_type": type(exc).__name__}
            except Exception as exc:
                error_msg = f"BaseMCPTool Unexpected error calling mcpo tool '{service_and_tool_path}': {type(exc).__name__} - {exc}"
                response_text_snippet = response.text[:500] if response and hasattr(response, 'text') else "Response object not available or no text."
                print(f"BaseMCPTool Error: {error_msg}. Response snippet: {response_text_snippet}"); traceback.print_exc()
                return {"error": error_msg, "exception_type": type(exc).__name__}

    def _handle_tool_result(self, result: dict | str, tool_name_for_log: str) -> str:
        print(f"BaseMCPTool DEBUG: {tool_name_for_log} result from mcpo: {str(result)[:500]}...")
        parsed_result = result
        if isinstance(result, str):
            try:
                parsed_result = json.loads(result)
            except json.JSONDecodeError:
                if "error" in result.lower() or "failed" in result.lower() or "traceback" in result.lower():
                    return f"调用 {tool_name_for_log} 失败，返回非JSON错误文本: {result}"
                print(f"BaseMCPTool Info: Result for {tool_name_for_log} is a non-JSON string, returning as is.")
                return result
        if isinstance(parsed_result, dict):
            if "error" in parsed_result and "status_code" in parsed_result:
                return f"调用 {tool_name_for_log} 时发生HTTP错误：{parsed_result.get('error')}"
            if parsed_result.get("status") == "error":
                error_msg = parsed_result.get("error_message", "未知错误")
                error_code = parsed_result.get("error_code", "NO_CODE")
                return f"工具 {tool_name_for_log} 执行失败 (错误码: {error_code})：{error_msg}"
            try:
                return json.dumps(parsed_result, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"BaseMCPTool Error formatting successful dict result for {tool_name_for_log}: {e}")
                return str(parsed_result)
        print(f"BaseMCPTool Warning: Unexpected result format from {tool_name_for_log} mcpo call: {type(result)}, content: {str(result)[:200]}")
        return f"从 {tool_name_for_log} 服务收到的结果格式不正确或无法处理: {str(result)[:500]}"

    def _run_default_sync_wrapper(self, **kwargs) -> str:
        tool_name = getattr(self, 'name', self.__class__.__name__)
        print(f"BaseMCPTool INFO: Synchronous _run called for {tool_name} with args: {kwargs}.")
        result_str = ""
        try:
            # --- 改进的 asyncio.run 处理 ---
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            async def async_runner():
                return await self._arun(**kwargs)

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, async_runner())
                    result = future.result(timeout=120)
            else:
                result = asyncio.run(async_runner())
            result_str = str(result)
        except asyncio.TimeoutError:
            error_message = f"Tool {tool_name} execution timed out after 120 seconds."
            print(f"BaseMCPTool ERROR_TRACE: {error_message}"); result_str = error_message
        except RuntimeError as e:
            if "cannot run event loop while another loop is running" in str(e).lower() or "event loop is already running" in str(e).lower():
                error_message = (f"BaseMCPTool Error in {tool_name} _run: Nested asyncio event loop conflict. Original error: {e}")
            else:
                error_message = f"BaseMCPTool RuntimeError in {tool_name} _run: {type(e).__name__} - {e}"
            print(f"BaseMCPTool ERROR_TRACE: {error_message}"); traceback.print_exc();
            result_str = error_message
        except Exception as e:
            error_message = f"BaseMCPTool General Exception in {tool_name} _run: {type(e).__name__} - {e}"
            print(f"BaseMCPTool ERROR_TRACE: {error_message}"); traceback.print_exc(); result_str = error_message
        return result_str

class HybridRAGTool(BaseMCPTool):
    name: str = "HybridRAGQueryTool"
    description: str = (
        "【核心RAG工具】用于通过执行混合检索增强生成 (RAG) 搜索来回答用户问题。 "
        "该工具整合了向量检索、知识图谱检索和关键词检索，并进行智能融合和重排序。 "
        "当用户需要从知识库中获取信息、回答复杂问题或生成报告时，应调用此工具。"
    )
    args_schema: Type[BaseModel] = QueryRequest
    target_mcp_service_path: str = "zhz_agent_rag_service/query_rag_v2"

    async def _arun(self, query: str, top_k_vector: int, top_k_kg: int, top_k_bm25: int, **kwargs: Any) -> str: # --- 确认有 **kwargs ---
        tool_name_for_log = getattr(self, 'name', "HybridRAGTool")
        print(f"CrewAI Tool DEBUG: {tool_name_for_log}._arun called with query='{query}', top_k_vector={top_k_vector}, top_k_kg={top_k_kg}, top_k_bm25={top_k_bm25}, additional_kwargs={kwargs}")

        security_context = kwargs.get('security_context')
        if security_context:
            print(f"CrewAI Tool INFO: Received security_context (in HybridRAGTool): {str(security_context)[:200]}...")

        payload = {
            "query": query,
            "top_k_vector": top_k_vector,
            "top_k_kg": top_k_kg,
            "top_k_bm25": top_k_bm25
        }
        result = await self._call_mcpo_endpoint(self.target_mcp_service_path, payload)
        return self._handle_tool_result(result, self.name)

    def _run(self, query: str, top_k_vector: int, top_k_kg: int, top_k_bm25: int, **kwargs: Any) -> str: # --- 确认有 **kwargs ---
        return self._run_default_sync_wrapper(query=query, top_k_vector=top_k_vector, top_k_kg=top_k_kg, top_k_bm25=top_k_bm25, **kwargs)