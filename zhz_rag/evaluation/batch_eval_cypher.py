# zhz_rag/evaluation/batch_eval_cypher.py
import asyncio
import json
import os
from typing import List, Dict, Any, Optional, TYPE_CHECKING # <--- 确保导入 TYPE_CHECKING
import glob
from datetime import datetime

# --- 从项目中导入必要的模块 ---
try:
    from zhz_rag.evaluation.evaluator import evaluate_cypher_with_gemini
    from zhz_rag.utils.common_utils import (
        find_latest_rag_interaction_log,
        load_jsonl_file
    )
    from zhz_rag.utils.common_utils import RAG_INTERACTION_LOGS_DIR
except ImportError as e:
    print(f"ERROR: Could not import necessary modules in batch_eval_cypher.py: {e}")
    print("Make sure this script is run in an environment where 'zhz_rag' package is accessible.")
    exit(1)

# --- 类型检查时导入资源类 ---
if TYPE_CHECKING:
    from zhz_rag_pipeline_dagster.zhz_rag_pipeline.resources import GeminiAPIResource

import logging

# --- 配置此脚本的logger ---
batch_cypher_eval_logger = logging.getLogger("BatchCypherEvaluationLogger")
batch_cypher_eval_logger.setLevel(logging.INFO)
if not batch_cypher_eval_logger.hasHandlers():
    _console_handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _console_handler.setFormatter(_formatter)
    batch_cypher_eval_logger.addHandler(_console_handler)
    batch_cypher_eval_logger.propagate = False # 通常不希望 utils 模块的日志传播到根
batch_cypher_eval_logger.info("--- BatchCypherEvaluationLogger configured ---")


async def run_cypher_batch_evaluation(
    gemini_resource_for_evaluator: 'GeminiAPIResource', # <--- 新增参数
    rag_interaction_log_filepath: str,
    app_version: str = "0.1.0",
    use_simulated_api: bool = False,
    api_call_delay: float = 4.1
) -> Dict[str, int]:
    """
    从指定的RAG交互日志文件中读取记录，筛选并评估Cypher查询。

    Args:
        rag_interaction_log_filepath (str): RAG交互日志文件的路径。
        app_version (str): 当前应用的日志版本标签。
        use_simulated_api (bool): 是否使用模拟的Gemini API响应。
        api_call_delay (float): 真实API调用之间的延迟秒数。

    Returns:
        Dict[str, int]: 包含评估统计信息的字典。
    """
    batch_cypher_eval_logger.info(f"Starting batch Cypher evaluation for log file: {rag_interaction_log_filepath}")
    batch_cypher_eval_logger.info(f"Parameters: app_version='{app_version}', use_simulated_api={use_simulated_api}, api_call_delay={api_call_delay}s")

    processed_count = 0
    evaluated_count = 0
    failed_to_extract_count = 0
    skipped_no_cypher_count = 0

    if use_simulated_api:
        batch_cypher_eval_logger.warning("Batch Cypher evaluation is USING SIMULATED Gemini responses.")
    else:
        batch_cypher_eval_logger.info("Batch Cypher evaluation is using REAL Gemini API calls.")

    # 使用新的通用函数加载日志数据
    interaction_logs = load_jsonl_file(rag_interaction_log_filepath)

    if not interaction_logs:
        batch_cypher_eval_logger.error(f"No data loaded from RAG interaction log file: {rag_interaction_log_filepath}. Exiting.")
        return {"processed": 0, "evaluated": 0, "skipped_no_cypher": 0, "failed_extract": 0, "file_not_found_or_empty": 1}

    for line_number, interaction_log in enumerate(interaction_logs, 1):
        processed_count += 1
        if interaction_log.get("task_type") == "cypher_generation":
            user_question = interaction_log.get("user_query")
            generated_cypher = interaction_log.get("processed_llm_output") # Qwen的原始输出
            original_id = interaction_log.get("interaction_id")

            if user_question and original_id: # 确保关键字段存在
                # 检查 generated_cypher 是否为空或表示无法生成
                if not generated_cypher or \
                   generated_cypher.strip() == "无法生成Cypher查询." or \
                   not generated_cypher.strip():
                    batch_cypher_eval_logger.info(f"Skipping evaluation for interaction_id '{original_id}' as Qwen indicated it couldn't generate Cypher or Cypher is empty. Content: '{generated_cypher}'")
                    skipped_no_cypher_count += 1
                    continue # 跳过此条记录

                batch_cypher_eval_logger.info(f"Evaluating Cypher for interaction_id: {original_id} - User Question: {user_question[:50]}...")
                
                evaluation_result = await evaluate_cypher_with_gemini(
                    gemini_resource=gemini_resource_for_evaluator, # <--- 传递资源实例
                    user_question=user_question,
                    generated_cypher=generated_cypher,
                    original_interaction_id=original_id,
                    app_version=app_version
                )

                if evaluation_result:
                    evaluated_count += 1
                    # 尝试从嵌套结构中获取 overall_quality_score_cypher
                    summary = evaluation_result.get("evaluation_summary", {})
                    overall_score = summary.get("overall_quality_score_cypher", "N/A")
                    batch_cypher_eval_logger.info(f"Successfully evaluated Cypher for interaction_id: {original_id}. Overall Score: {overall_score}")
                else:
                    batch_cypher_eval_logger.warning(f"Cypher evaluation returned None or failed for interaction_id: {original_id}")
                
                if not use_simulated_api:
                    batch_cypher_eval_logger.info(f"Waiting for {api_call_delay} seconds before next API call...")
                    await asyncio.sleep(api_call_delay)
            else:
                failed_to_extract_count += 1
                batch_cypher_eval_logger.warning(f"Skipping cypher_generation log entry {line_number} due to missing user_query or interaction_id. Log content: {str(interaction_log)[:200]}...")
        
        if processed_count > 0 and processed_count % 20 == 0: # 日志打印频率
            batch_cypher_eval_logger.info(f"Progress: Processed {processed_count} log entries. Evaluated {evaluated_count} Cypher queries so far. Skipped (no cypher/cannot generate): {skipped_no_cypher_count}. Failed to extract key fields: {failed_to_extract_count}.")

    summary = {
        "total_processed_from_log": processed_count,
        "cypher_queries_evaluated": evaluated_count,
        "skipped_qwen_could_not_generate": skipped_no_cypher_count,
        "failed_to_extract_fields_for_eval": failed_to_extract_count
    }
    batch_cypher_eval_logger.info(f"Batch Cypher evaluation finished. Summary: {summary}")
    return summary


if __name__ == "__main__":
    # 1. 自动查找最新的原始RAG交互日志文件
    # RAG_INTERACTION_LOGS_DIR 已从 common_utils 导入
    log_file_to_evaluate = find_latest_rag_interaction_log(RAG_INTERACTION_LOGS_DIR)

    # 2. 从环境变量决定是否模拟API调用 和 配置API调用延迟
    use_simulated_env = os.getenv("USE_SIMULATED_GEMINI_CYPHER_EVAL", "false").lower() == "true"
    api_delay_env = float(os.getenv("GEMINI_API_CALL_DELAY_SECONDS", "4.1")) # 从环境变量读取延迟

    app_version_tag_env = os.getenv("APP_VERSION_TAG", "0.1.3_batch_cypher_refactored")
    if use_simulated_env:
        app_version_tag_env += "_simulated"

    if log_file_to_evaluate:
        batch_cypher_eval_logger.info(f"Found RAG interaction log to process: {log_file_to_evaluate}")
        asyncio.run(run_cypher_batch_evaluation(
            rag_interaction_log_filepath=log_file_to_evaluate,
            app_version=app_version_tag_env,
            use_simulated_api=use_simulated_env,
            api_call_delay=api_delay_env
        ))
    # 如果文件不存在但明确要求模拟 (通常用于测试脚本本身，虽然没有输入数据意义不大)
    elif use_simulated_env:
        batch_cypher_eval_logger.warning(f"RAG interaction log file not found, but USE_SIMULATED_GEMINI_CYPHER_EVAL is true. Running with a dummy path (will process 0 entries).")
        asyncio.run(run_cypher_batch_evaluation(
            rag_interaction_log_filepath="dummy_non_existent_file.jsonl", # 传递一个虚拟路径
            app_version=app_version_tag_env + "_no_file",
            use_simulated_api=use_simulated_env,
            api_call_delay=api_delay_env
        ))
    else:
        batch_cypher_eval_logger.warning(f"No suitable RAG interaction log file found in '{RAG_INTERACTION_LOGS_DIR}' and not using simulated responses. Batch Cypher evaluation will not run.")