# 文件: zhz_rag/utils/interaction_logger.py
import os
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import logging
import aiofiles  # 使用 aiofiles 进行异步文件操作
import uuid
import traceback
import sys # <--- 添加此行

# --- 配置此模块的logger ---
# 使用一个特定的名字，以便在项目中其他地方可以按名获取，避免与根logger混淆
interaction_logger_module_logger = logging.getLogger("InteractionLoggerUtil")
# 建议从环境变量或配置文件读取日志级别，提供默认值
interaction_logger_module_logger.setLevel(os.getenv("INTERACTION_LOG_LEVEL", "INFO").upper())
# 设置propagate = False以防止日志消息被传递到父级logger（如root logger），避免重复输出
interaction_logger_module_logger.propagate = False

# 确保只添加一次处理器，防止重复配置
if not interaction_logger_module_logger.hasHandlers():
    _il_console_handler = logging.StreamHandler()
    _il_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _il_console_handler.setFormatter(_il_formatter)
    interaction_logger_module_logger.addHandler(_il_console_handler)
    interaction_logger_module_logger.info("--- InteractionLoggerUtil configured ---")


# --- 定义日志存储目录常量 ---
# __file__ 指向 .../zhz_rag/utils/interaction_logger.py
_CURRENT_FILE_DIR_IL = os.path.dirname(os.path.abspath(__file__))
# _ZHZ_RAG_PACKAGE_DIR_IL 指向 .../zhz_rag
_ZHZ_RAG_PACKAGE_DIR_IL = os.path.dirname(_CURRENT_FILE_DIR_IL)

# STORED_DATA_ROOT_DIR_IL 指向 .../zhz_rag/stored_data
_STORED_DATA_ROOT_DIR_IL = os.path.join(_ZHZ_RAG_PACKAGE_DIR_IL, 'stored_data')

RAG_INTERACTION_LOGS_DIR_DEFAULT = os.path.join(_STORED_DATA_ROOT_DIR_IL, 'rag_interaction_logs')
EVALUATION_RESULTS_LOGS_DIR_DEFAULT = os.path.join(_STORED_DATA_ROOT_DIR_IL, 'evaluation_results_logs')


async def _async_write_to_jsonl_robust(filepath: str, interaction_json_string: str):
    """
    一个健壮的异步函数，用于将字符串追加到文件。
    使用了 aiofiles 库来避免阻塞事件循环。
    """
    logger_to_use = interaction_logger_module_logger
    logger_to_use.debug(f"ASYNC_WRITE_ROBUST: Attempting to write to {filepath}")
    try:
        async with aiofiles.open(filepath, mode='a', encoding='utf-8') as f:
            await f.write(interaction_json_string)
            await f.flush() # aiofiles 的 flush 也是异步的
        logger_to_use.debug(f"ASYNC_WRITE_ROBUST: Successfully wrote and flushed to {filepath}")
    except Exception as e:
        logger_to_use.error(f"CRITICAL_LOG_FAILURE in _async_write_to_jsonl_robust: Failed to write to {filepath}. Error: {e}", exc_info=True)
        # 备用方案
        # print(f"CRITICAL_LOG_FAILURE: Could not write to {filepath}. Error: {e}")
        # traceback.print_exc()

def _sync_write_to_jsonl_robust(filepath: str, interaction_json_string: str):
    """
    一个健壮的同步函数，用于将字符串追加到文件，并确保数据刷入磁盘。
    """
    logger_to_use = interaction_logger_module_logger
    logger_to_use.debug(f"SYNC_WRITE_ROBUST: Attempting to write to {filepath}")
    try:
        # 'a' for append. '+' is not strictly needed for 'a' as it creates the file if it doesn't exist.
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(interaction_json_string)
            # 步骤1: 确保Python应用层缓冲区的内容写入操作系统缓冲区
            f.flush()
            # 步骤2: 请求操作系统将缓冲区内容实际写入磁盘，提供最强保证
            os.fsync(f.fileno())
        logger_to_use.debug(f"SYNC_WRITE_ROBUST: Successfully wrote and synced to {filepath}")
    except Exception as e:
        # 这种底层的关键日志如果失败，需要非常明确的错误提示
        logger_to_use.error(f"CRITICAL_LOG_FAILURE in _sync_write_to_jsonl_robust: Failed to write to {filepath}. Error: {e}", exc_info=True)


async def log_interaction_data(
    log_data: Dict[str, Any],
    is_evaluation_result: bool = False,
    evaluation_name_for_file: Optional[str] = None
):
    """
    Asynchronously logs interaction data to a JSONL file in the appropriate directory.

    Args:
        log_data (Dict[str, Any]): The dictionary containing the data to log.
        is_evaluation_result (bool): If True, logs to the evaluation results directory. 
                                     Otherwise, logs to the standard RAG interaction directory.
        evaluation_name_for_file (Optional[str]): A specific name for the evaluation file, e.g., 'answer_gemini'.
    """
    try:
        # --- 修正：根据 is_evaluation_result 选择正确的目录 ---
        if is_evaluation_result:
            target_dir = os.getenv("EVALUATION_RESULTS_LOGS_DIR", EVALUATION_RESULTS_LOGS_DIR_DEFAULT)
            if not evaluation_name_for_file:
                evaluation_name_for_file = "default_eval"
            # 文件名格式: eval_results_指定的名称_日期.jsonl
            log_filename = f"eval_results_{evaluation_name_for_file}_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        else:
            target_dir = os.getenv("RAG_INTERACTION_LOGS_DIR", RAG_INTERACTION_LOGS_DIR_DEFAULT)
            # 文件名格式: rag_interactions_日期.jsonl
            log_filename = f"rag_interactions_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"

        # 确保目标目录存在
        os.makedirs(target_dir, exist_ok=True)
        log_filepath = os.path.join(target_dir, log_filename)

        # 准备要写入的JSON字符串
        # 确保时间戳是字符串格式，避免JSON序列化问题
        if 'timestamp_utc' in log_data and isinstance(log_data['timestamp_utc'], datetime):
             log_data['timestamp_utc'] = log_data['timestamp_utc'].isoformat()
        
        log_entry_str = json.dumps(log_data, ensure_ascii=False)

        # 异步写入文件
        await _async_write_to_jsonl_robust(log_filepath, log_entry_str + '\n')
        
    except Exception as e:
        # 在独立的日志系统中记录日志本身的错误
        interaction_logger_module_logger.error(f"Failed to log interaction data. Error: {e}", exc_info=True)
        interaction_logger_module_logger.error(f"Original log data that failed: {str(log_data)[:500]}")

def get_logger(name: str) -> logging.Logger:
    """
    一个通用的函数，用于获取或创建具有标准配置的logger。
    这避免了在每个模块中重复配置logger。
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(os.getenv(f"{name.upper()}_LOG_LEVEL", "INFO").upper())
        logger.propagate = False
        handler = logging.StreamHandler(sys.stdout) # 确保日志输出到标准输出
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - PID:%(process)d - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger