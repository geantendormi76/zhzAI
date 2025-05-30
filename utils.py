# zhz_agent/utils.py

import httpx
import json
import traceback
import os
from dotenv import load_dotenv
from datetime import datetime, timezone 
import uuid 
import logging 
import asyncio
from typing import Dict, Any, Optional 
load_dotenv()

# --- 添加一个utils模块的logger (新添加) ---
utils_logger = logging.getLogger("UtilsLogger")
utils_logger.setLevel(logging.INFO)
if not utils_logger.hasHandlers():
    _utils_console_handler = logging.StreamHandler()
    _utils_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _utils_console_handler.setFormatter(_utils_formatter)
    utils_logger.addHandler(_utils_console_handler)
    utils_logger.propagate = False
utils_logger.info("--- UtilsLogger configured ---")
# --- 结束logger配置 ---


MCPO_BASE_URL = os.getenv("MCPO_BASE_URL", "http://localhost:8006")

# --- 通用交互日志记录 (新添加) ---
# RAG_EVAL_DATA_DIR 的定义：utils.py 位于 zhz_agent 目录下
# 我们希望 rag_eval_data 也在 zhz_agent 目录下
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__)) # zhz_agent 目录
RAG_EVAL_DATA_DIR = os.path.join(_UTILS_DIR, 'rag_eval_data')

if not os.path.exists(RAG_EVAL_DATA_DIR):
    try:
        os.makedirs(RAG_EVAL_DATA_DIR)
        utils_logger.info(f"Successfully created directory for interaction logs: {RAG_EVAL_DATA_DIR}")
    except Exception as e:
        utils_logger.error(f"Error creating directory {RAG_EVAL_DATA_DIR}: {e}. Please create it manually.")

def get_interaction_log_filepath(log_type_prefix: str = "interaction") -> str:
    """获取当前交互日志文件的完整路径，按天分割。"""
    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    # 所有日志写入同一个文件，方便统一处理和按时间排序
    return os.path.join(RAG_EVAL_DATA_DIR, f"rag_interactions_{today_str}.jsonl")

def get_evaluation_result_log_filepath(evaluation_name: str = "cypher_eval") -> str: # <--- 新增函数
    """获取评估结果日志文件的完整路径，按天分割，并可指定评估类型。"""
    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    # 示例: rag_eval_data/eval_results_cypher_eval_20250530.jsonl
    return os.path.join(RAG_EVAL_DATA_DIR, f"eval_results_{evaluation_name}_{today_str}.jsonl")

async def log_interaction_data(interaction_data: dict, is_evaluation_result: bool = False, evaluation_name_for_file: Optional[str] = None): # <--- 修改参数
    """
    将单条交互数据或评估结果异步追加到JSONL文件中。
    """
    if is_evaluation_result:
        if not evaluation_name_for_file:
            # 如果是评估结果但没有指定文件名后缀，给一个默认的
            evaluation_name_for_file = interaction_data.get("task_type", "general_eval") 
        filepath = get_evaluation_result_log_filepath(evaluation_name=evaluation_name_for_file)
    else:
        filepath = get_interaction_log_filepath() # 原始交互日志文件名保持不变

    # 确保核心字段存在
    if "timestamp_utc" not in interaction_data:
        interaction_data["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    if "interaction_id" not in interaction_data:
        interaction_data["interaction_id"] = str(uuid.uuid4())

    try:
        def _write_sync():
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(interaction_data, ensure_ascii=False) + "\n")
        
        await asyncio.to_thread(_write_sync)
        # utils_logger.debug(f"Successfully logged interaction (type: {interaction_data.get('task_type', 'N/A')}) to {filepath}")
    except Exception as e:
        utils_logger.error(f"Failed to log interaction data to {filepath}: {e}", exc_info=True)
# --- 结束通用交互日志记录 ---

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
