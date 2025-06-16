# 文件: zhz_rag/utils/interaction_logger.py
import os
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import logging

# --- 配置此模块的logger ---
interaction_logger_module_logger = logging.getLogger("InteractionLoggerUtil") # 使用一个特定的名字
interaction_logger_module_logger.setLevel(os.getenv("INTERACTION_LOG_LEVEL", "INFO").upper())
interaction_logger_module_logger.propagate = False # 避免重复日志到根记录器

if not interaction_logger_module_logger.hasHandlers():
    _il_console_handler = logging.StreamHandler()
    _il_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _il_console_handler.setFormatter(_il_formatter)
    interaction_logger_module_logger.addHandler(_il_console_handler)
    interaction_logger_module_logger.info("--- InteractionLoggerUtil configured ---")


# --- 定义日志存储目录常量 (可以从 config.constants 导入，或在此处定义) ---
# 获取当前文件所在目录的父目录的父目录 (即 zhz_rag 的父目录，应该是 zhz_agent)
_PROJECT_ROOT_FOR_LOGS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RAG_INTERACTION_LOGS_DIR_DEFAULT = os.path.join(_PROJECT_ROOT_FOR_LOGS, 'stored_data', 'rag_interaction_logs')
EVALUATION_RESULTS_LOGS_DIR_DEFAULT = os.path.join(_PROJECT_ROOT_FOR_LOGS, 'stored_data', 'evaluation_results_logs')


async def log_interaction_data(
    interaction_data: Dict[str, Any],
    is_evaluation_result: bool = False,
    evaluation_name_for_file: Optional[str] = None,
    custom_log_dir: Optional[str] = None
):
    """
    异步将单条交互数据或评估结果追加到按天分割的JSONL文件中。

    Args:
        interaction_data (Dict[str, Any]): 要记录的交互数据字典。
        is_evaluation_result (bool): 如果为True，则记录到评估结果目录。默认为False。
        evaluation_name_for_file (Optional[str]): 如果是评估结果，用于文件名中区分评估类型。
                                                 例如 "answer_gemini_flash"。
        custom_log_dir (Optional[str]): 自定义日志目录路径，如果提供则覆盖默认目录。
    """
    try:
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        
        if is_evaluation_result:
            base_dir = custom_log_dir if custom_log_dir else EVALUATION_RESULTS_LOGS_DIR_DEFAULT
            file_prefix = "eval_results"
            if evaluation_name_for_file:
                file_prefix += f"_{evaluation_name_for_file}"
        else:
            base_dir = custom_log_dir if custom_log_dir else RAG_INTERACTION_LOGS_DIR_DEFAULT
            file_prefix = "rag_interactions"
            
        log_filename = f"{file_prefix}_{today_str}.jsonl"
        log_filepath = os.path.join(base_dir, log_filename)

        if not os.path.exists(base_dir):
            try:
                os.makedirs(base_dir, exist_ok=True)
                interaction_logger_module_logger.info(f"Created log directory: {base_dir}")
            except Exception as e_mkdir:
                interaction_logger_module_logger.error(f"Failed to create log directory {base_dir}: {e_mkdir}", exc_info=True)
                # 如果目录创建失败，可以选择记录到临时位置或直接返回，避免程序崩溃
                return

        def _write_sync():
            with open(log_filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(interaction_data, ensure_ascii=False) + "\n")
        
        await asyncio.to_thread(_write_sync)
        interaction_logger_module_logger.debug(f"Successfully logged data to {log_filepath}. Interaction ID: {interaction_data.get('interaction_id', 'N/A')}")

    except Exception as e:
        interaction_logger_module_logger.error(f"Failed to log interaction data: {e}", exc_info=True)
        interaction_logger_module_logger.error(f"Data that failed to log (first 500 chars): {str(interaction_data)[:500]}")