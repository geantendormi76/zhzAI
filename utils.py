# zhz_agent/utils.py

import httpx
import json
import traceback
import os
from dotenv import load_dotenv
load_dotenv()

MCPO_BASE_URL = os.getenv("MCPO_BASE_URL", "http://localhost:8006")

async def call_mcpo_tool(tool_name_with_prefix: str, payload: dict):
    """
    异步调用MCP工具服务。
    """
    api_url = f"{MCPO_BASE_URL}/{tool_name_with_prefix}"
    cleaned_payload = {k: v for k, v in payload.items() if v is not None}

    print(f"Calling mcpo endpoint: POST {api_url} with payload: {json.dumps(cleaned_payload, ensure_ascii=False)}")

    async with httpx.AsyncClient() as client:
        response = None  # 初始化response变量
        try:
            headers = {"Content-Type": "application/json"}
            response = await client.post(api_url, json=cleaned_payload, headers=headers, timeout=120.0)
            print(f"mcpo status code: {response.status_code}")

            if response.status_code == 200:
                try:
                    result_data = response.json()
                    if isinstance(result_data, dict) and result_data.get("isError"):
                        error_content_list = result_data.get("content", [{"type": "text", "text": "Unknown error from MCP tool"}])
                        error_text = "Unknown error from MCP tool"
                        for item in error_content_list:
                            if item.get("type") == "text":
                                error_text = item.get("text", error_text)
                                break
                        print(f"MCP Tool execution failed (isError=true): {error_text}")
                        try:
                            parsed_mcp_error = json.loads(error_text)
                            if isinstance(parsed_mcp_error, dict) and "error" in parsed_mcp_error:
                                return {"error": f"Tool '{tool_name_with_prefix}' failed via MCP: {parsed_mcp_error['error']}"}
                        except json.JSONDecodeError:
                            pass # 不是JSON，直接使用error_text
                        return {"error": f"Tool '{tool_name_with_prefix}' failed via MCP: {error_text}"}
                    return result_data
                except json.JSONDecodeError:
                    print(f"Warning: mcpo returned status 200 but response is not JSON for '{tool_name_with_prefix}'. Returning raw text.")
                    return {"content": [{"type": "text", "text": response.text}]} # 包装成MCP期望的格式之一
            else:
                error_text = f"mcpo call to '{tool_name_with_prefix}' failed with status {response.status_code}. Response: {response.text[:500]}..."
                print(error_text)
                return {"error": error_text}

        except httpx.RequestError as exc: # 更具体的网络请求错误
            error_msg = f"HTTP RequestError calling mcpo tool '{tool_name_with_prefix}': {type(exc).__name__} - {exc}"
            print(error_msg)
            traceback.print_exc()
            return {"error": error_msg}
        except Exception as exc: # 捕获其他所有异常
            error_msg = f"Unexpected error calling mcpo tool '{tool_name_with_prefix}': {type(exc).__name__} - {exc}"
            response_text_snippet = response.text[:500] if response and hasattr(response, 'text') else "Response object not available or no text."
            print(f"{error_msg}. Response snippet: {response_text_snippet}")
            traceback.print_exc()
            return {"error": error_msg}
