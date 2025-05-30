# zhz_agent/run_batch_cypher_evaluation.py
import asyncio
import json
import os
from typing import List, Dict, Any, Optional
import glob
from datetime import datetime # 确保导入

try:
    from zhz_agent.evaluation import evaluate_cypher_with_gemini
    from zhz_agent.constants import NEW_KG_SCHEMA_DESCRIPTION
    from zhz_agent.utils import get_interaction_log_filepath, find_latest_rag_interaction_log # <--- 确保这里导入了 find_latest_rag_interaction_log
except ImportError as e:
    print(f"ERROR: Could not import necessary modules: {e}")
    print("Make sure this script is run in an environment where 'zhz_agent' is accessible.")
    print("If running from outside the project root, you might need to set PYTHONPATH.")
    print("Example: PYTHONPATH=/path/to/your/project python -m zhz_agent.run_batch_cypher_evaluation")
    exit(1)

import logging

# 配置此脚本的logger
batch_cypher_eval_logger = logging.getLogger("BatchCypherEvaluationLogger") # 修改logger名称以区分
batch_cypher_eval_logger.setLevel(logging.INFO)
if not batch_cypher_eval_logger.hasHandlers():
    _console_handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _console_handler.setFormatter(_formatter)
    batch_cypher_eval_logger.addHandler(_console_handler)
    batch_cypher_eval_logger.info("--- BatchCypherEvaluationLogger configured ---")

# LOG_FILE_DIR 定义在脚本的顶层
LOG_FILE_DIR = "zhz_agent/rag_eval_data/"

async def batch_evaluate_cyphers_from_file(
    rag_interaction_log_filepath: str,
    app_version: str = "0.1.0",
    use_simulated_api: bool = False,
    api_call_delay: float = 4.1
) -> Dict[str, int]:
    """
    从指定的RAG交互日志文件中读取记录，筛选并评估Cypher查询。
    """
    batch_cypher_eval_logger.info(f"Starting batch Cypher evaluation for log file: {rag_interaction_log_filepath}")
    processed_count = 0
    evaluated_count = 0
    failed_to_extract_count = 0
    skipped_no_cypher_count = 0

    if use_simulated_api:
        batch_cypher_eval_logger.warning("Batch Cypher evaluation is USING SIMULATED Gemini responses.")
    else:
        batch_cypher_eval_logger.info("Batch Cypher evaluation is using REAL Gemini API calls.")

    if not os.path.exists(rag_interaction_log_filepath):
        batch_cypher_eval_logger.error(f"Input RAG interaction log file not found: {rag_interaction_log_filepath}")
        return {"processed": 0, "evaluated": 0, "skipped_no_cypher": 0, "failed_extract": 0}

    try:
        with open(rag_interaction_log_filepath, 'r', encoding='utf-8') as f_in:
            for line_number, line in enumerate(f_in, 1):
                try:
                    interaction_log = json.loads(line.strip())
                except json.JSONDecodeError:
                    batch_cypher_eval_logger.warning(f"Skipping malformed JSON line {line_number} in {rag_interaction_log_filepath}")
                    continue

                processed_count += 1
                if interaction_log.get("task_type") == "cypher_generation":
                    user_question = interaction_log.get("user_query")
                    generated_cypher = interaction_log.get("processed_llm_output")
                    original_id = interaction_log.get("interaction_id")

                    if user_question and original_id:
                        if not generated_cypher or generated_cypher.strip() == "无法生成Cypher查询." or not generated_cypher.strip():
                            batch_cypher_eval_logger.info(f"Skipping evaluation for interaction_id '{original_id}' as Qwen indicated it couldn't generate Cypher or Cypher is empty.")
                            skipped_no_cypher_count += 1
                            continue

                        batch_cypher_eval_logger.info(f"Evaluating Cypher for interaction_id: {original_id} - User Question: {user_question[:50]}...")
                        
                        evaluation_result = await evaluate_cypher_with_gemini(
                            user_question=user_question,
                            generated_cypher=generated_cypher,
                            # kg_schema_description is handled internally by evaluate_cypher_with_gemini
                            original_interaction_id=original_id,
                            app_version=app_version 
                        )

                        if evaluation_result:
                            evaluated_count += 1
                            batch_cypher_eval_logger.info(f"Successfully evaluated Cypher for interaction_id: {original_id}. Overall Score: {evaluation_result.get('evaluation_summary', {}).get('overall_quality_score_cypher')}")
                        else:
                            batch_cypher_eval_logger.warning(f"Cypher evaluation returned None or failed for interaction_id: {original_id}")
                        
                        if not use_simulated_api: 
                            batch_cypher_eval_logger.info(f"Waiting for {api_call_delay} seconds before next API call...")
                            await asyncio.sleep(api_call_delay)
                    else:
                        failed_to_extract_count +=1
                        batch_cypher_eval_logger.warning(f"Skipping cypher_generation line {line_number} due to missing user_query or interaction_id. Log content: {str(interaction_log)[:200]}...")
                
                if processed_count > 0 and processed_count % 20 == 0:
                    batch_cypher_eval_logger.info(f"Progress: Processed {processed_count} log entries. Evaluated {evaluated_count} Cypher queries so far. Skipped (no cypher): {skipped_no_cypher_count}. Failed to extract: {failed_to_extract_count}.")

    except Exception as e:
        batch_cypher_eval_logger.error(f"Error during batch Cypher evaluation: {e}", exc_info=True)

    summary = {"processed": processed_count, "evaluated": evaluated_count, "skipped_no_cypher": skipped_no_cypher_count, "failed_extract": failed_to_extract_count}
    batch_cypher_eval_logger.info(f"Batch Cypher evaluation finished. Summary: {summary}")
    return summary


if __name__ == "__main__":
    # 1. 自动查找最新的原始RAG交互日志文件
    log_file_to_evaluate = find_latest_rag_interaction_log(LOG_FILE_DIR)

    # 2. 从环境变量决定是否模拟API调用
    use_simulated = os.getenv("USE_SIMULATED_GEMINI_CYPHER_EVAL", "false").lower() == "true"
    app_version_tag = "0.1.2_batch_cypher_auto" # 更新版本标签
    if use_simulated:
        app_version_tag += "_simulated"

    if log_file_to_evaluate:
        asyncio.run(batch_evaluate_cyphers_from_file( # 调用新的函数名
            rag_interaction_log_filepath=log_file_to_evaluate,
            app_version=app_version_tag,
            use_simulated_api=use_simulated
        ))
    elif use_simulated:
        batch_cypher_eval_logger.warning(f"RAG interaction log file not found, but USE_SIMULATED_GEMINI_CYPHER_EVAL is true. Running with a dummy path (will process 0 entries).")
        asyncio.run(batch_evaluate_cyphers_from_file( # 调用新的函数名
            rag_interaction_log_filepath="dummy_non_existent_file.jsonl",
            app_version=app_version_tag + "_no_file",
            use_simulated_api=use_simulated
        ))
    else:
        batch_cypher_eval_logger.warning(f"No suitable RAG interaction log file found and not using simulated responses. Batch Cypher evaluation will not run.")