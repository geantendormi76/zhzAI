# /home/zhz/zhz_agent/utils/common_utils.py

import httpx
import json
import traceback
import os
import glob
from dotenv import load_dotenv
from datetime import datetime, timezone
import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional
import re
import unicodedata

load_dotenv()

# --- Logger Configuration ---
# 使用一个统一的 logger 名称，方便管理
# 您可以根据需要在调用此模块的顶层配置这个 logger
# 例如，在 agent_orchestrator_service.py 的开头配置 "ZhzAgentUtils"
utils_logger = logging.getLogger("ZhzAgentUtils")
if not utils_logger.handlers: # 避免重复添加处理器
    utils_logger.setLevel(os.getenv("UTILS_LOG_LEVEL", "INFO").upper())
    _utils_console_handler = logging.StreamHandler()
    _utils_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _utils_console_handler.setFormatter(_utils_formatter)
    utils_logger.addHandler(_utils_console_handler)
    utils_logger.propagate = False # 通常在自定义logger中设置为False
utils_logger.info("--- ZhzAgentUtils Logger configured ---")


# --- MCP Configuration ---
MCPO_BASE_URL = os.getenv("MCPO_BASE_URL", "http://localhost:8006")

# --- Directory Paths (统一管理，基于此文件位置推断项目结构) ---
_CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# 假设 common_utils.py 在 zhz_agent/utils/ 或 zhz_agent/zhz_rag/utils/
# 我们需要项目根目录 zhz_agent/
_PROJECT_ROOT_GUESS_1 = os.path.abspath(os.path.join(_CURRENT_FILE_DIR, "..")) # 如果在 utils/ 下
_PROJECT_ROOT_GUESS_2 = os.path.abspath(os.path.join(_CURRENT_FILE_DIR, "..", "..")) # 如果在 zhz_rag/utils/ 下

# 尝试确定正确的项目根目录
if os.path.basename(_PROJECT_ROOT_GUESS_1) == "zhz_agent":
    PROJECT_ROOT_DIR = _PROJECT_ROOT_GUESS_1
elif os.path.basename(_PROJECT_ROOT_GUESS_2) == "zhz_agent":
    PROJECT_ROOT_DIR = _PROJECT_ROOT_GUESS_2
else:
    # 如果都猜不到，就用一个相对路径或者发出警告
    utils_logger.warning(f"Could not reliably determine PROJECT_ROOT_DIR from common_utils.py location. Using current dir as fallback for relative paths: {_CURRENT_FILE_DIR}")
    PROJECT_ROOT_DIR = _CURRENT_FILE_DIR # 或者您的固定路径

STORED_DATA_ROOT_DIR = os.path.join(PROJECT_ROOT_DIR, 'zhz_rag', 'stored_data')
RAG_INTERACTION_LOGS_DIR = os.path.join(STORED_DATA_ROOT_DIR, 'rag_interaction_logs')
EVALUATION_RESULTS_LOGS_DIR = os.path.join(STORED_DATA_ROOT_DIR, 'evaluation_results_logs')
FINETUNING_GENERATED_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'zhz_rag', 'finetuning', 'generated_data')

_DIRECTORIES_TO_CREATE = [
    STORED_DATA_ROOT_DIR, # 确保父目录也创建
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
    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return os.path.join(RAG_INTERACTION_LOGS_DIR, f"rag_interactions_{today_str}.jsonl")

def get_evaluation_result_log_filepath(evaluation_name: str) -> str:
    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return os.path.join(EVALUATION_RESULTS_LOGS_DIR, f"eval_results_{evaluation_name}_{today_str}.jsonl")

def find_latest_rag_interaction_log(log_dir: str = RAG_INTERACTION_LOGS_DIR) -> Optional[str]:
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
    if is_evaluation_result:
        if not evaluation_name_for_file:
            evaluation_name_for_file = interaction_data.get("task_type", "general_eval_result")
        filepath = get_evaluation_result_log_filepath(evaluation_name=evaluation_name_for_file)
    else:
        filepath = get_interaction_log_filepath()

    if "timestamp_utc" not in interaction_data:
        interaction_data["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    if "interaction_id" not in interaction_data and not is_evaluation_result:
        interaction_data["interaction_id"] = str(uuid.uuid4())
    elif "interaction_id" not in interaction_data and is_evaluation_result and "original_interaction_id_ref" in interaction_data:
         interaction_data["interaction_id"] = str(uuid.uuid4())

    try:
        def _write_sync():
            log_file_dir = os.path.dirname(filepath)
            if not os.path.exists(log_file_dir):
                try:
                    os.makedirs(log_file_dir, exist_ok=True)
                    utils_logger.info(f"Created directory for log file: {log_file_dir}")
                except Exception as e_mkdir:
                    utils_logger.error(f"Error creating directory {log_file_dir} for log file: {e_mkdir}")
                    return
            with open(filepath, 'a', encoding='utf-8') as f:
                json_string_to_write = json.dumps(interaction_data, ensure_ascii=False, default=str)
                f.write(json_string_to_write + "\n")
        await asyncio.to_thread(_write_sync)
    except Exception as e:
        utils_logger.error(f"Failed to log interaction data to {filepath}: {e}", exc_info=True)

# --- MCP Tool Calling Utility ---
async def call_mcpo_tool(tool_name_with_prefix: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    api_url = f"{MCPO_BASE_URL}/{tool_name_with_prefix}"
    cleaned_payload = {k: v for k, v in (payload or {}).items() if v is not None}

    utils_logger.info(f"CALL_MCPO_TOOL: Attempting to call {api_url}")
    utils_logger.debug(f"CALL_MCPO_TOOL: Payload: {json.dumps(cleaned_payload, ensure_ascii=False)}")

    timeout_config = httpx.Timeout(120.0, connect=10.0, read=120.0, write=10.0)
    
    async with httpx.AsyncClient(timeout=timeout_config) as client:
        response: Optional[httpx.Response] = None
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ZhzAgent/1.0 (call_mcpo_tool)"
            }
            utils_logger.debug(f"CALL_MCPO_TOOL: Sending POST request to {api_url} with headers: {headers}")
            response = await client.post(api_url, json=cleaned_payload, headers=headers)
            
            utils_logger.info(f"CALL_MCPO_TOOL: Response from {api_url} - Status: {response.status_code}")
            utils_logger.debug(f"CALL_MCPO_TOOL: Response Headers: {response.headers}")
            
            try:
                response_text_snippet = response.text[:500]
                utils_logger.debug(f"CALL_MCPO_TOOL: Response Text Snippet (first 500 chars): {response_text_snippet}")
            except Exception as e_read_snippet:
                utils_logger.warning(f"CALL_MCPO_TOOL: Could not read response text snippet: {e_read_snippet}")

            if response.status_code == 200:
                try:
                    result_data = response.json()
                    utils_logger.info(f"CALL_MCPO_TOOL: Successfully received and parsed JSON response from {api_url}.")
                    
                    if isinstance(result_data, dict) and result_data.get("isError"):
                        error_content_list = result_data.get("content", [{"type": "text", "text": "Unknown error from MCP tool"}])
                        error_text_from_mcp_payload = "Unknown error from MCP tool"
                        for item in error_content_list:
                            if item.get("type") == "text":
                                error_text_from_mcp_payload = item.get("text", error_text_from_mcp_payload)
                                break
                        utils_logger.error(f"CALL_MCPO_TOOL: MCP Tool '{tool_name_with_prefix}' reported an application-level error (isError=true): {error_text_from_mcp_payload}")
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
                except json.JSONDecodeError:
                    utils_logger.warning(f"CALL_MCPO_TOOL: Response from {api_url} was 200 OK but not valid JSON. Assuming plain text response. Raw text: {response.text[:200]}...")
                    return {
                        "success": True,
                        "data": { 
                            "content": [{"type": "text", "text": response.text}]
                        } 
                    }
            else:
                error_message = f"HTTP Error {response.status_code} when calling {api_url}."
                utils_logger.error(f"CALL_MCPO_TOOL: {error_message} Response: {response.text[:500]}", exc_info=False)
                error_detail_from_response = response.text
                try:
                    parsed_error_json = response.json()
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
                    "status_code": response.status_code,
                    "raw_response_snippet": response.text[:500] if response else "N/A"
                }
        except httpx.TimeoutException as exc_timeout:
            utils_logger.error(f"CALL_MCPO_TOOL: Timeout when calling {api_url}. Error: {exc_timeout}", exc_info=True)
            return {
                "success": False,
                "error": f"Request to MCP service timed out after {timeout_config.read if timeout_config else 'default'}s.",
                "error_type": "TIMEOUT_ERROR",
                "status_code": None 
            }
        except httpx.ConnectError as exc_connect:
            utils_logger.error(f"CALL_MCPO_TOOL: Connection error when calling {api_url}. Is the MCP service running at {MCPO_BASE_URL}? Error: {exc_connect}", exc_info=True)
            return {
                "success": False,
                "error": f"Could not connect to MCP service at {MCPO_BASE_URL}.",
                "error_type": "CONNECTION_ERROR",
                "status_code": None
            }
        except httpx.RequestError as exc_request_other: 
            utils_logger.error(f"CALL_MCPO_TOOL: Network request error when calling {api_url}. Error: {exc_request_other}", exc_info=True)
            return {
                "success": False,
                "error": f"A network request error occurred: {str(exc_request_other)}",
                "error_type": type(exc_request_other).__name__,
                "status_code": None
            }
        except Exception as exc_unexpected:
            utils_logger.error(f"CALL_MCPO_TOOL: Unexpected error when calling {api_url}. Error: {exc_unexpected}", exc_info=True)
            return {
                "success": False,
                "error": f"An unexpected error occurred during MCP call: {str(exc_unexpected)}",
                "error_type": type(exc_unexpected).__name__,
                "status_code": response.status_code if response else None, 
                "traceback": traceback.format_exc() 
            }

# --- JSONL File Loading Utility ---
def load_jsonl_file(filepath: str, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
    data_list: List[Dict[str, Any]] = []
    if not os.path.exists(filepath):
        utils_logger.error(f"File not found: {filepath}")
        return data_list
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            for line_number, line in enumerate(f, 1):
                try:
                    if line.strip():
                        data_list.append(json.loads(line.strip()))
                except json.JSONDecodeError as e_json:
                    utils_logger.warning(f"Skipping malformed JSON line {line_number} in {filepath}: {e_json}")
                except Exception as e_line:
                    utils_logger.warning(f"Error processing line {line_number} in {filepath}: {e_line}")
    except FileNotFoundError:
        utils_logger.error(f"File not found during open: {filepath}")
    except Exception as e_file:
        utils_logger.error(f"Error reading or processing file {filepath}: {e_file}", exc_info=True)
        return []
    utils_logger.info(f"Successfully loaded {len(data_list)} entries from {filepath}")
    return data_list

# --- Text Normalization Utility ---
def normalize_text_for_id(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    try:
        normalized_text = unicodedata.normalize('NFKD', text)
        normalized_text = normalized_text.lower()
        normalized_text = normalized_text.strip()
        normalized_text = re.sub(r'\s+', ' ', normalized_text) # Collapse multiple whitespaces
        return normalized_text
    except Exception: # pylint: disable=broad-except
        # Fallback to original text if normalization fails for any reason
        return text

# --- DB Utils (Moved from zhz_rag.utils.db_utils to here for consolidation if needed) ---
# from databases import Database # This would require 'databases' and 'aiosqlite' in requirements
# from sqlalchemy import create_engine
# from sqlalchemy.orm import declarative_base
# import pytz
# from apscheduler.schedulers.asyncio import AsyncIOScheduler
# from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

# ZHZ_AGENT_DIR_FOR_DB = PROJECT_ROOT_DIR # Use the determined project root
# DB_NAME = "ZHZ_AGENT_tasks.db" # Or your desired DB name
# DATABASE_FILE_PATH = os.path.join(ZHZ_AGENT_DIR_FOR_DB, DB_NAME) # Store DB in project root
# DATABASE_URL = f"sqlite+aiosqlite:///{DATABASE_FILE_PATH}"

# database_instance: Optional[Database] = None
# sqlalchemy_engine_instance = None
# SQLAlchemyBase = declarative_base()
# scheduler_instance: Optional[AsyncIOScheduler] = None

# def get_database() -> Database:
#     global database_instance
#     if database_instance is None:
#         database_instance = Database(DATABASE_URL)
#     return database_instance

# def get_sqlalchemy_engine():
#     global sqlalchemy_engine_instance
#     if sqlalchemy_engine_instance is None:
#         sqlalchemy_engine_instance = create_engine(DATABASE_URL.replace("+aiosqlite", ""))
#     return sqlalchemy_engine_instance

# def get_scheduler() -> AsyncIOScheduler:
#     global scheduler_instance
#     if scheduler_instance is None:
#         jobstore_url = f"sqlite:///{DATABASE_FILE_PATH}" # Scheduler uses its own tables in the same DB
#         jobstores = {
#             'default': SQLAlchemyJobStore(url=jobstore_url, tablename='apscheduler_jobs_v2_common_utils')
#         }
#         scheduler_instance = AsyncIOScheduler(jobstores=jobstores, timezone=pytz.utc)
#         logging.getLogger('apscheduler').setLevel(os.getenv("SCHEDULER_LOG_LEVEL", "WARNING").upper())
#         utils_logger.info(f"APScheduler (common_utils) initialized with timezone: {pytz.utc}")
#     return scheduler_instance

# async def connect_database():
#     db_to_connect = get_database()
#     if not db_to_connect.is_connected:
#         await db_to_connect.connect()
#         utils_logger.info("Database connected (common_utils).")

# async def disconnect_database():
#     db_to_disconnect = get_database()
#     if db_to_disconnect.is_connected:
#         await db_to_disconnect.disconnect()
#         utils_logger.info("Database disconnected (common_utils).")

# def create_db_tables():
#     engine = get_sqlalchemy_engine()
#     SQLAlchemyBase.metadata.create_all(bind=engine)
#     utils_logger.info("SQLAlchemy tables created via common_utils (if not exist).")

# def start_scheduler():
#     sched = get_scheduler()
#     if not sched.running:
#         sched.start()
#         utils_logger.info("APScheduler started via common_utils.")

# def shutdown_scheduler():
#     sched = get_scheduler()
#     if sched.running:
#         sched.shutdown()
#         utils_logger.info("APScheduler shutdown via common_utils.")

# (如果决定将数据库和调度器逻辑也合并到这里，请取消注释上面的DB相关部分，
# 并确保相关的依赖（databases, sqlalchemy, apscheduler, pytz）已添加到项目的 requirements.txt)