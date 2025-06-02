# /home/zhz/zhz_agent/utils/common_utils.py

import httpx
import json
import traceback
import os
import logging

# --- 日志配置 ---
logger = logging.getLogger(__name__)
# (您可以根据需要添加更详细的日志配置)

# --- MCP 配置 ---
MCPO_BASE_URL = os.getenv("MCPO_BASE_URL", "http://localhost:8006")

async def call_mcpo_tool(tool_name_with_prefix: str, payload: dict) -> dict:
    """
    异步调用MCP工具服务。
    tool_name_with_prefix 例如: "zhz_rag_mcp_service/query_rag_v2" 或 "ddgsearch/search"
    payload 是传递给MCP工具的参数字典。
    返回一个字典，成功时包含工具的输出，失败时包含 "error" 键。
    """
    api_url = f"{MCPO_BASE_URL}/{tool_name_with_prefix}"
    cleaned_payload = {k: v for k, v in (payload or {}).items() if v is not None}

    logger.info(f"Calling MCP endpoint: POST {api_url} with payload: {json.dumps(cleaned_payload, ensure_ascii=False)}")

    async with httpx.AsyncClient(timeout=120.0) as client: # 增加超时
        response = None
        try:
            headers = {"Content-Type": "application/json"}
            response = await client.post(api_url, json=cleaned_payload, headers=headers)
            logger.info(f"MCP status code for {tool_name_with_prefix}: {response.status_code}")

            if response.status_code == 200:
                try:
                    result_data = response.json()
                    # 检查 MCP 服务本身是否返回了错误结构 (例如 MCP 框架的错误包装)
                    if isinstance(result_data, dict) and result_data.get("isError"): # 假设 MCP 错误格式
                        error_text = result_data.get("content", [{"text": "Unknown error from MCP tool"}])[0].get("text")
                        logger.error(f"MCP Tool '{tool_name_with_prefix}' execution failed (isError=true): {error_text}")
                        return {"error": f"Tool '{tool_name_with_prefix}' failed via MCP: {error_text}"}
                    return result_data # 假设成功时直接返回工具的 JSON 输出
                except json.JSONDecodeError:
                    logger.warning(f"MCP call to '{tool_name_with_prefix}' returned status 200 but response is not JSON. Raw text: {response.text[:500]}...")
                    return {"error": "Non-JSON response from MCP tool", "raw_response": response.text}
            else:
                error_text = f"MCP call to '{tool_name_with_prefix}' failed with status {response.status_code}. Response: {response.text[:500]}..."
                logger.error(error_text)
                return {"error": error_text, "status_code": response.status_code}

        except httpx.RequestError as exc:
            error_msg = f"HTTP RequestError calling MCP tool '{tool_name_with_prefix}': {type(exc).__name__} - {exc}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg, "exception_type": type(exc).__name__}
        except Exception as exc:
            error_msg = f"Unexpected error calling MCP tool '{tool_name_with_prefix}': {type(exc).__name__} - {exc}"
            response_text_snippet = response.text[:500] if response and hasattr(response, 'text') else "N/A"
            logger.error(f"{error_msg}. Response snippet: {response_text_snippet}", exc_info=True)
            return {"error": error_msg, "exception_type": type(exc).__name__}

# 确保 utils 目录也有一个 __init__.py 文件
# touch /home/zhz/zhz_agent/utils/__init__.py (如果不存在)