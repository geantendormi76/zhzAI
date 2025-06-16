# 文件: zhz_rag/utils/interaction_logger.py
import os
import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import logging
import uuid
import traceback

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


# --- 定义日志存储目录常量 (可以从 config.constants 导入，或在此处定义) ---
# 获取当前文件所在目录的父目录的父目录 (即 zhz_rag 的父目录，应该是 zhz_agent)
_PROJECT_ROOT_FOR_LOGS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RAG_INTERACTION_LOGS_DIR_DEFAULT = os.path.join(_PROJECT_ROOT_FOR_LOGS, 'stored_data', 'rag_interaction_logs')
EVALUATION_RESULTS_LOGS_DIR_DEFAULT = os.path.join(_PROJECT_ROOT_FOR_LOGS, 'stored_data', 'evaluation_results_logs')


def _sync_write_to_jsonl_robust(filepath: str, interaction_json_string: str):
    """
    一个健壮的同步函数，用于将字符串追加到文件，并确保数据刷入磁盘。
    """
    logger_to_use = interaction_logger_module_logger
    logger_to_use.debug(f"SYNC_WRITE_ROBUST: Attempting to write to {filepath}")
    try:
        # 'a' for append. '+' is not strictly needed for 'a' as it creates the file if it doesn't exist.
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(interaction_json_string + "\n")
            # 步骤1: 确保Python应用层缓冲区的内容写入操作系统缓冲区
            f.flush()
            # 步骤2: 请求操作系统将缓冲区内容实际写入磁盘，提供最强保证
            os.fsync(f.fileno())
        logger_to_use.debug(f"SYNC_WRITE_ROBUST: Successfully wrote and synced to {filepath}")
    except Exception as e:
        # 这种底层的关键日志如果失败，需要非常明确的错误提示
        logger_to_use.error(f"CRITICAL_LOG_FAILURE in _sync_write_to_jsonl_robust: Failed to write to {filepath}. Error: {e}", exc_info=True)
        # 如果logger可能没有配置好，可以取消下面的注释作为备用方案
        # print(f"CRITICAL_LOG_FAILURE in _sync_write_to_jsonl_robust: Failed to write to {filepath}. Error: {e}")
        # traceback.print_exc()

async def log_interaction_data(
    interaction_data: Dict[str, Any],
    is_evaluation_result: bool = False,
    evaluation_name_for_file: Optional[str] = None,
    custom_log_dir: Optional[str] = None
):
    """
    异步将单条交互数据或评估结果追加到按天分割的JSONL文件中，
    此过程使用一个健壮的写入方法以确保数据持久化。

    Args:
        interaction_data (Dict[str, Any]): 要记录的交互数据字典。
        is_evaluation_result (bool): 如果为True，则记录到评估结果目录。默认为False。
        evaluation_name_for_file (Optional[str]): 如果是评估结果，用于文件名中区分评估类型。
        custom_log_dir (Optional[str]): 自定义日志目录路径，如果提供则覆盖默认目录。
    """
    logger_to_use = interaction_logger_module_logger
    filepath = "" # 初始化filepath以备在异常处理中使用
    try:
        # --- 数据准备逻辑 ---
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
        filepath = os.path.join(base_dir, log_filename)

        if not os.path.exists(base_dir):
            try:
                os.makedirs(base_dir, exist_ok=True)
                logger_to_use.info(f"Created log directory: {base_dir}")
            except Exception as e_mkdir:
                logger_to_use.error(f"Failed to create log directory {base_dir}: {e_mkdir}", exc_info=True)
                # 如果目录创建失败，直接返回，避免后续操作引发更多错误
                return

        # 确保时间戳和ID存在
        if "timestamp_utc" not in interaction_data:
            interaction_data["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        if "interaction_id" not in interaction_data:
            interaction_data["interaction_id"] = str(uuid.uuid4())
        
        # 将字典转换为JSON字符串
        json_string = json.dumps(interaction_data, ensure_ascii=False)
        
        # 通过 asyncio.to_thread 调用我们加强版的同步写入函数
        await asyncio.to_thread(_sync_write_to_jsonl_robust, filepath, json_string)
        
        logger_to_use.debug(f"Successfully queued robust log write to {filepath}. Interaction ID: {interaction_data.get('interaction_id', 'N/A')}")

    except Exception as e:
        logger_to_use.error(f"ERROR in log_interaction_data: Failed to queue log writing for {filepath}. Error: {e}", exc_info=True)
        # 记录失败时，打印部分数据以供调试
        data_str_for_log = str(interaction_data)
        logger_to_use.error(f"Data that failed to log (first 500 chars): {data_str_for_log[:500]}")