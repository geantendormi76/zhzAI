# zhz_rag/utils/common_utils.py

import httpx
import json
import traceback
import os
import glob
from dotenv import load_dotenv
from datetime import datetime, timezone
import uuid
import logging
import asyncio #确保 asyncio 被导入
from typing import List, Dict, Any, Optional
import re
import unicodedata
import logging

load_dotenv()

# --- Logger Configuration ---
utils_logger = logging.getLogger("UtilsLogger")
utils_logger.setLevel(logging.INFO)
if not utils_logger.hasHandlers():
    _utils_console_handler = logging.StreamHandler()
    _utils_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _utils_console_handler.setFormatter(_utils_formatter)
    utils_logger.addHandler(_utils_console_handler)
    utils_logger.propagate = False
utils_logger.info("--- UtilsLogger configured ---")

# --- MCP Configuration ---
MCPO_BASE_URL = os.getenv("MCPO_BASE_URL", "http://localhost:8006")

_CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_ZHZ_RAG_PACKAGE_DIR = os.path.dirname(_CURRENT_FILE_DIR)

STORED_DATA_ROOT_DIR = os.path.join(_ZHZ_RAG_PACKAGE_DIR, 'stored_data')

RAG_INTERACTION_LOGS_DIR = os.path.join(STORED_DATA_ROOT_DIR, 'rag_interaction_logs')
EVALUATION_RESULTS_LOGS_DIR = os.path.join(STORED_DATA_ROOT_DIR, 'evaluation_results_logs')

FINETUNING_GENERATED_DATA_DIR = os.path.join(_ZHZ_RAG_PACKAGE_DIR, 'finetuning', 'generated_data')

# Ensure these directories exist
_DIRECTORIES_TO_CREATE = [
    STORED_DATA_ROOT_DIR,
    RAG_INTERACTION_LOGS_DIR,
    EVALUATION_RESULTS_LOGS_DIR,
    FINETUNING_GENERATED_DATA_DIR
]
for dir_path in _DIRECTORIES_TO_CREATE:
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
            utils_logger.info(f"Successfully created directory: {dir_path}")
        except Exception as e:
            utils_logger.error(f"Error creating directory {dir_path}: {e}. Consider creating it manually.")

# --- Log File Path Getters ---

def get_interaction_log_filepath() -> str:
    """Gets the full path for the current RAG interaction log file (daily rotation)."""
    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return os.path.join(RAG_INTERACTION_LOGS_DIR, f"rag_interactions_{today_str}.jsonl")

def get_evaluation_result_log_filepath(evaluation_name: str) -> str:
    """Gets the full path for an evaluation result log file (daily rotation, by evaluation name)."""
    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return os.path.join(EVALUATION_RESULTS_LOGS_DIR, f"eval_results_{evaluation_name}_{today_str}.jsonl")

def find_latest_rag_interaction_log(log_dir: str = RAG_INTERACTION_LOGS_DIR) -> Optional[str]:
    """
    Finds the latest RAG interaction log file (rag_interactions_*.jsonl) in the specified directory.
    Defaults to RAG_INTERACTION_LOGS_DIR.
    """
    utils_logger.debug(f"Searching for RAG interaction logs in: {log_dir}")
    rag_log_pattern = os.path.join(log_dir, "rag_interactions_*.jsonl")
    candidate_rag_logs = glob.glob(rag_log_pattern)

    if candidate_rag_logs:
        candidate_rag_logs.sort(key=os.path.getmtime, reverse=True)
        utils_logger.info(f"Automatically selected RAG interaction log: {candidate_rag_logs[0]}")
        return candidate_rag_logs[0]
    else:
        utils_logger.warning(f"No RAG interaction log files found matching pattern: {rag_log_pattern} in directory {log_dir}")
        return None

# --- Logging Function ---

async def log_interaction_data(
    interaction_data: Dict[str, Any],
    is_evaluation_result: bool = False,
    evaluation_name_for_file: Optional[str] = None
):
    """
    Asynchronously appends a single interaction data or evaluation result to a JSONL file.
    """
    if is_evaluation_result:
        if not evaluation_name_for_file:
            evaluation_name_for_file = interaction_data.get("task_type", "general_eval_result") # More specific default
        filepath = get_evaluation_result_log_filepath(evaluation_name=evaluation_name_for_file)
    else:
        filepath = get_interaction_log_filepath()

    if "timestamp_utc" not in interaction_data:
        interaction_data["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    if "interaction_id" not in interaction_data and not is_evaluation_result: # Eval results use original_interaction_id_ref
        interaction_data["interaction_id"] = str(uuid.uuid4())
    elif "interaction_id" not in interaction_data and is_evaluation_result and "original_interaction_id_ref" in interaction_data:
        # For eval results, ensure there's an ID, can be a new one for the eval log entry itself
         interaction_data["interaction_id"] = str(uuid.uuid4())

    try:
        # --- 新增 DEBUG 日志 ---
        print(f"COMMON_UTILS_LOG_DATA: Preparing to dump JSON. Keys in interaction_data: {list(interaction_data.keys())}")
        if "final_context_docs_full" in interaction_data:
            print(f"COMMON_UTILS_LOG_DATA: 'final_context_docs_full' IS PRESENT before dumps.")
            if interaction_data["final_context_docs_full"]:
                 print(f"COMMON_UTILS_LOG_DATA: 'final_context_docs_full' is NOT EMPTY before dumps. Length: {len(interaction_data['final_context_docs_full'])}")
                 try:
                    first_content = interaction_data["final_context_docs_full"][0].get("content", "CONTENT_KEY_MISSING")
                    print(f"COMMON_UTILS_LOG_DATA: First content in final_context_docs_full: {str(first_content)[:50]}...")
                 except:
                    pass # 简单忽略打印错误
            else:
                print(f"COMMON_UTILS_LOG_DATA: 'final_context_docs_full' IS EMPTY LIST before dumps.")
        else:
            print(f"COMMON_UTILS_LOG_DATA: 'final_context_docs_full' KEY IS MISSING before dumps!")
        # --- 结束新增 DEBUG 日志 ---

        def _write_sync():
            log_file_dir = os.path.dirname(filepath)
            if not os.path.exists(log_file_dir):
                try:
                    os.makedirs(log_file_dir, exist_ok=True)
                    # utils_logger.info(f"Created directory for log file: {log_file_dir}") # 使用print替代，避免日志级别问题
                    print(f"COMMON_UTILS_LOG_DATA: Created directory for log file: {log_file_dir}")
                except Exception as e_mkdir:
                    # utils_logger.error(f"Error creating directory {log_file_dir} for log file: {e_mkdir}")
                    print(f"COMMON_UTILS_LOG_DATA: Error creating directory {log_file_dir} for log file: {e_mkdir}")
                    return 

            with open(filepath, 'a', encoding='utf-8') as f:
                json_string_to_write = json.dumps(interaction_data, ensure_ascii=False, default=str)

                # --- 新增 DEBUG 日志 ---
                print(f"COMMON_UTILS_LOG_DATA: JSON string to write (first 300 chars): {json_string_to_write[:300]}...")
                if "\"final_context_docs_full\"" not in json_string_to_write: # 检查序列化后的字符串
                    print(f"COMMON_UTILS_LOG_DATA: CRITICAL! 'final_context_docs_full' NOT IN JSON string after dumps!")
                # --- 结束新增 DEBUG 日志 ---
                
                f.write(json_string_to_write + "\n")
        
        await asyncio.to_thread(_write_sync)
        # utils_logger.debug(f"Successfully logged data (type: {interaction_data.get('task_type', 'N/A')}) to {filepath}")
    except Exception as e:
        utils_logger.error(f"Failed to log interaction data to {filepath}: {e}", exc_info=True)

# --- MCP Tool Calling Utility ---

async def call_mcpo_tool(tool_name_with_prefix: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    异步调用MCP工具服务，并返回结构化的成功或错误响应。
    """
    api_url = f"{MCPO_BASE_URL}/{tool_name_with_prefix}"
    cleaned_payload = {k: v for k, v in (payload or {}).items() if v is not None}

    utils_logger.info(f"CALL_MCPO_TOOL: Attempting to call {api_url}") # 使用 utils_logger
    utils_logger.debug(f"CALL_MCPO_TOOL: Payload: {json.dumps(cleaned_payload, ensure_ascii=False)}") # 使用 utils_logger

    timeout_config = httpx.Timeout(120.0, connect=10.0, read=120.0, write=10.0) 
    
    async with httpx.AsyncClient(timeout=timeout_config) as client:
        response: Optional[httpx.Response] = None 
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ZhzAgent/1.0 (call_mcpo_tool)"
            }
            utils_logger.debug(f"CALL_MCPO_TOOL: Sending POST request to {api_url} with headers: {headers}") # 使用 utils_logger
            response = await client.post(api_url, json=cleaned_payload, headers=headers)
            
            utils_logger.info(f"CALL_MCPO_TOOL: Response from {api_url} - Status: {response.status_code}") # 使用 utils_logger
            utils_logger.debug(f"CALL_MCPO_TOOL: Response Headers: {response.headers}") # 使用 utils_logger
            
            try:
                response_text_snippet = response.text[:500] 
                utils_logger.debug(f"CALL_MCPO_TOOL: Response Text Snippet (first 500 chars): {response_text_snippet}") # 使用 utils_logger
            except Exception as e_read_snippet:
                utils_logger.warning(f"CALL_MCPO_TOOL: Could not read response text snippet: {e_read_snippet}") # 使用 utils_logger

            response.raise_for_status() 

            try:
                result_data = response.json()
                utils_logger.info(f"CALL_MCPO_TOOL: Successfully received and parsed JSON response from {api_url}.") # 使用 utils_logger
                if isinstance(result_data, dict) and result_data.get("isError"):
                    error_content_list = result_data.get("content", [{"type": "text", "text": "Unknown error from MCP tool"}])
                    error_text_from_mcp_payload = "Unknown error from MCP tool"
                    for item in error_content_list:
                        if item.get("type") == "text":
                            error_text_from_mcp_payload = item.get("text", error_text_from_mcp_payload)
                            break
                    utils_logger.error(f"CALL_MCPO_TOOL: MCP Tool '{tool_name_with_prefix}' reported an application-level error (isError=true): {error_text_from_mcp_payload}") # 使用 utils_logger
                    return {
                        "success": False,
                        "error": f"MCP tool '{tool_name_with_prefix}' reported failure: {error_text_from_mcp_payload}",
                        "error_type": "MCP_APPLICATION_ERROR",
                        "status_code": response.status_code
                    }
                return {
                        "success": True, 
                        "data": result_data 
                }
            except json.JSONDecodeError as e_json_decode:
                utils_logger.error(f"CALL_MCPO_TOOL: Response from {api_url} was 2xx but not valid JSON. Error: {e_json_decode}", exc_info=True) # 使用 utils_logger
                return {
                    "success": False,
                    "error": "MCP service returned a 2xx status but the response was not valid JSON.",
                    "error_type": "JSON_DECODE_ERROR",
                    "status_code": response.status_code,
                    "raw_response_snippet": response.text[:500] if response else "N/A"
                }

        except httpx.HTTPStatusError as exc_http_status:
            error_message = f"HTTP Error {exc_http_status.response.status_code} when calling {api_url}."
            utils_logger.error(f"CALL_MCPO_TOOL: {error_message} Response: {exc_http_status.response.text[:500]}", exc_info=True) # 使用 utils_logger
            error_detail_from_response = exc_http_status.response.text
            try:
                parsed_error_json = exc_http_status.response.json()
                if isinstance(parsed_error_json, dict) and "detail" in parsed_error_json:
                    error_detail_from_response = parsed_error_json["detail"]
                elif isinstance(parsed_error_json, dict) and "error" in parsed_error_json: 
                    error_detail_from_response = parsed_error_json["error"]
            except json.JSONDecodeError:
                pass 
            return {
                "success": False,
                "error": f"HTTP error from MCP service: {error_detail_from_response}",
                "error_type": "HTTP_STATUS_ERROR",
                "status_code": exc_http_status.response.status_code,
                "raw_response_snippet": exc_http_status.response.text[:500] if exc_http_status.response else "N/A"
            }
        except httpx.TimeoutException as exc_timeout:
            utils_logger.error(f"CALL_MCPO_TOOL: Timeout when calling {api_url}. Error: {exc_timeout}", exc_info=True) # 使用 utils_logger
            return {
                "success": False,
                "error": f"Request to MCP service timed out after {timeout_config.read if timeout_config else 'default'}s.",
                "error_type": "TIMEOUT_ERROR",
                "status_code": None 
            }
        except httpx.ConnectError as exc_connect:
            utils_logger.error(f"CALL_MCPO_TOOL: Connection error when calling {api_url}. Is the MCP service running at {MCPO_BASE_URL}? Error: {exc_connect}", exc_info=True) # 使用 utils_logger
            return {
                "success": False,
                "error": f"Could not connect to MCP service at {MCPO_BASE_URL}.",
                "error_type": "CONNECTION_ERROR",
                "status_code": None
            }
        except httpx.RequestError as exc_request_other: 
            utils_logger.error(f"CALL_MCPO_TOOL: Network request error when calling {api_url}. Error: {exc_request_other}", exc_info=True) # 使用 utils_logger
            return {
                "success": False,
                "error": f"A network request error occurred: {str(exc_request_other)}",
                "error_type": type(exc_request_other).__name__,
                "status_code": None
            }
        except Exception as exc_unexpected:
            utils_logger.error(f"CALL_MCPO_TOOL: Unexpected error when calling {api_url}. Error: {exc_unexpected}", exc_info=True) # 使用 utils_logger
            return {
                "success": False,
                "error": f"An unexpected error occurred during MCP call: {str(exc_unexpected)}",
                "error_type": type(exc_unexpected).__name__,
                "status_code": response.status_code if response else None, 
                "traceback": traceback.format_exc() 
            }

def load_jsonl_file(filepath: str, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
    """
    从 JSONL 文件加载数据。

    Args:
        filepath (str): JSONL 文件的路径。
        encoding (str): 文件编码，默认为 'utf-8'。

    Returns:
        List[Dict[str, Any]]: 从文件中加载的字典列表。如果文件不存在或解析出错，
                              会记录错误并返回空列表。
    """
    data_list: List[Dict[str, Any]] = []
    if not os.path.exists(filepath):
        utils_logger.error(f"File not found: {filepath}") # 使用已有的 utils_logger
        return data_list

    try:
        with open(filepath, 'r', encoding=encoding) as f:
            for line_number, line in enumerate(f, 1):
                try:
                    if line.strip(): # 确保行不是空的
                        data_list.append(json.loads(line.strip()))
                except json.JSONDecodeError as e_json:
                    utils_logger.warning(f"Skipping malformed JSON line {line_number} in {filepath}: {e_json}")
                except Exception as e_line:
                    utils_logger.warning(f"Error processing line {line_number} in {filepath}: {e_line}")
    except FileNotFoundError: # 再次捕获以防万一，虽然上面已经检查了
        utils_logger.error(f"File not found during open: {filepath}")
    except Exception as e_file:
        utils_logger.error(f"Error reading or processing file {filepath}: {e_file}", exc_info=True)
        return [] # 如果文件读取层面发生严重错误，返回空列表

    utils_logger.info(f"Successfully loaded {len(data_list)} entries from {filepath}")
    return data_list


def normalize_text_for_id(text: str) -> str:
    if not isinstance(text, str):
        return str(text) 
    
    try:
        normalized_text = unicodedata.normalize('NFKD', text)
        normalized_text = normalized_text.lower()
        normalized_text = normalized_text.strip()
        normalized_text = re.sub(r'\s+', ' ', normalized_text)
        return normalized_text
    except Exception as e:
        return text