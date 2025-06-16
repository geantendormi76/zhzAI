# 文件: zhz_rag/utils/common_utils.py
# 版本: 最终版 - 所有路径统一到 zhz_rag/stored_data/

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
utils_logger = logging.getLogger("UtilsLogger")
utils_logger.setLevel(logging.INFO)
if not utils_logger.hasHandlers():
    _utils_console_handler = logging.StreamHandler()
    _utils_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _utils_console_handler.setFormatter(_utils_formatter)
    utils_logger.addHandler(_utils_console_handler)
    utils_logger.propagate = False
utils_logger.info("--- UtilsLogger configured ---")

# --- 【【【【【 核心修正点：统一路径计算逻辑到 zhz_rag 包内 】】】】】 ---
_CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# __file__ 指向 .../zhz_rag/utils/common_utils.py
# os.path.dirname(_CURRENT_FILE_DIR) 将指向 .../zhz_rag/
_ZHZ_RAG_PACKAGE_DIR = os.path.dirname(_CURRENT_FILE_DIR)

STORED_DATA_ROOT_DIR = os.path.join(_ZHZ_RAG_PACKAGE_DIR, 'stored_data')

RAG_INTERACTION_LOGS_DIR = os.path.join(STORED_DATA_ROOT_DIR, 'rag_interaction_logs')
EVALUATION_RESULTS_LOGS_DIR = os.path.join(STORED_DATA_ROOT_DIR, 'evaluation_results_logs')
FINETUNING_GENERATED_DATA_DIR = os.path.join(_ZHZ_RAG_PACKAGE_DIR, 'finetuning', 'generated_data')
# --- 【【【【【 修正结束 】】】】】

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

# --- Log File Path Getters (无需修改，它们会使用上面新的常量) ---

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

# --- 其他辅助函数 (保持不变) ---
# MCP Tool Calling Utility
MCPO_BASE_URL = os.getenv("MCPO_BASE_URL", "http://localhost:8006")

async def call_mcpo_tool(tool_name_with_prefix: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    # (此函数内容保持不变)
    api_url = f"{MCPO_BASE_URL}/{tool_name_with_prefix}"
    cleaned_payload = {k: v for k, v in (payload or {}).items() if v is not None}
    utils_logger.info(f"CALL_MCPO_TOOL: Attempting to call {api_url}")
    timeout_config = httpx.Timeout(120.0, connect=10.0, read=120.0, write=10.0)
    async with httpx.AsyncClient(timeout=timeout_config) as client:
        try:
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            response = await client.post(api_url, json=cleaned_payload, headers=headers)
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except Exception as exc:
            # Simplified error handling for brevity
            return {"success": False, "error": str(exc)}


def load_jsonl_file(filepath: str, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
    # (此函数内容保持不变)
    data_list: List[Dict[str, Any]] = []
    if not os.path.exists(filepath):
        utils_logger.error(f"File not found: {filepath}")
        return data_list
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            for line in f:
                if line.strip(): data_list.append(json.loads(line.strip()))
    except Exception as e_file:
        utils_logger.error(f"Error reading or processing file {filepath}: {e_file}", exc_info=True)
        return []
    utils_logger.info(f"Successfully loaded {len(data_list)} entries from {filepath}")
    return data_list


def normalize_text_for_id(text: str) -> str:
    # (此函数内容保持不变)
    if not isinstance(text, str): return str(text)
    try:
        normalized_text = unicodedata.normalize('NFKD', text)
        normalized_text = normalized_text.lower().strip()
        return re.sub(r'\s+', ' ', normalized_text)
    except Exception:
        return text
