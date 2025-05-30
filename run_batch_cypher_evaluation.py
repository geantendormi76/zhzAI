# zhz_agent/run_batch_cypher_evaluation.py
import asyncio
import json
import os
from typing import List, Dict, Any

# 确保可以正确导入项目内的模块
try:
    from zhz_agent.evaluation import evaluate_cypher_with_gemini
    from zhz_agent.constants import NEW_KG_SCHEMA_DESCRIPTION
    from zhz_agent.utils import get_interaction_log_filepath # 用于找到原始日志文件
except ImportError as e:
    print(f"ERROR: Could not import necessary modules: {e}")
    print("Make sure this script is run in an environment where 'zhz_agent' is accessible.")
    print("If running from outside the project root, you might need to set PYTHONPATH.")
    print("Example: PYTHONPATH=/path/to/your/project python -m zhz_agent.run_batch_cypher_evaluation")
    exit(1)

import logging

# 配置此脚本的logger
batch_eval_logger = logging.getLogger("BatchCypherEvaluationLogger")
batch_eval_logger.setLevel(logging.INFO)
if not batch_eval_logger.hasHandlers():
    _console_handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _console_handler.setFormatter(_formatter)
    batch_eval_logger.addHandler(_console_handler)
    batch_eval_logger.info("--- BatchCypherEvaluationLogger configured ---")


async def run_batch_evaluation(input_log_filepath: str, app_version_for_eval: str = "0.1.0"):
    """
    读取包含RAG交互的JSONL文件，筛选出Cypher生成记录，并进行批量评估。
    """
    batch_eval_logger.info(f"Starting batch Cypher evaluation for log file: {input_log_filepath}")
    processed_count = 0
    evaluated_count = 0
    failed_to_extract_count = 0
    skipped_no_cypher_count = 0

    # 从环境变量判断是否使用模拟响应，以便在真实调用时才延时
    USE_SIMULATED_FLAG = os.getenv("USE_SIMULATED_GEMINI_CYPHER_EVAL", "false").lower() == "true"
    if USE_SIMULATED_FLAG:
        batch_eval_logger.warning("Batch evaluation is configured to USE SIMULATED Gemini responses.")
    else:
        batch_eval_logger.info("Batch evaluation is configured to use REAL Gemini API calls.")


    if not os.path.exists(input_log_filepath):
        batch_eval_logger.error(f"Input log file not found: {input_log_filepath}")
        return

    try:
        with open(input_log_filepath, 'r', encoding='utf-8') as f_in:
            for line_number, line in enumerate(f_in, 1):
                try:
                    interaction_log = json.loads(line.strip())
                except json.JSONDecodeError:
                    batch_eval_logger.warning(f"Skipping malformed JSON line {line_number} in {input_log_filepath}")
                    continue

                processed_count += 1
                if interaction_log.get("task_type") == "cypher_generation":
                    # 在llm.py中, user_query_for_log 对应的是 generate_cypher_query 的 user_question 参数
                    user_question = interaction_log.get("user_query") 
                    generated_cypher = interaction_log.get("processed_llm_output")
                    original_id = interaction_log.get("interaction_id")

                    if user_question and original_id: # generated_cypher 可能是空或"无法生成..."
                        if not generated_cypher or generated_cypher.strip() == "无法生成Cypher查询." or not generated_cypher.strip():
                            batch_eval_logger.info(f"Skipping evaluation for interaction_id '{original_id}' as Qwen indicated it couldn't generate Cypher or Cypher is empty.")
                            skipped_no_cypher_count += 1
                            continue

                        batch_eval_logger.info(f"Evaluating Cypher for interaction_id: {original_id} - User Question: {user_question[:50]}...")
                        
                        evaluation_result = await evaluate_cypher_with_gemini(
                            user_question=user_question,
                            generated_cypher=generated_cypher,
                            # kg_schema_description is now handled internally by evaluate_cypher_with_gemini
                            original_interaction_id=original_id,
                            app_version=app_version_for_eval
                        )

                        if evaluation_result:
                            evaluated_count += 1
                            batch_eval_logger.info(f"Successfully evaluated Cypher for interaction_id: {original_id}. Overall Score: {evaluation_result.get('evaluation_summary', {}).get('overall_quality_score_cypher')}")
                        else:
                            batch_eval_logger.warning(f"Evaluation returned None or failed for interaction_id: {original_id}")
                        
                        # 只在真实调用API时延时
                        if not USE_SIMULATED_FLAG: 
                            wait_time = 4.1 # 秒
                            batch_eval_logger.info(f"Waiting for {wait_time} seconds before next API call to respect rate limits...")
                            await asyncio.sleep(wait_time)
                    else:
                        failed_to_extract_count +=1
                        batch_eval_logger.warning(f"Skipping cypher_generation line {line_number} due to missing user_query or interaction_id. Log content: {str(interaction_log)[:200]}...")
                
                if processed_count > 0 and processed_count % 20 == 0: # 每处理20条日志打印一次进度
                    batch_eval_logger.info(f"Progress: Processed {processed_count} log entries. Evaluated {evaluated_count} Cypher queries so far. Skipped (no cypher): {skipped_no_cypher_count}. Failed to extract: {failed_to_extract_count}.")

    except Exception as e:
        batch_eval_logger.error(f"Error during batch evaluation: {e}", exc_info=True)

    batch_eval_logger.info(f"Batch Cypher evaluation finished. Total log entries processed: {processed_count}. Cypher queries evaluated: {evaluated_count}. Skipped (no cypher): {skipped_no_cypher_count}. Entries failed to extract data: {failed_to_extract_count}.")


if __name__ == "__main__":
    # 1. 确定要评估的日志文件路径
    #    默认评估当天的 "rag_interactions_YYYYMMDD.jsonl" 文件。
    #    如果需要评估特定文件，请修改下面的 log_file_to_evaluate 变量。
    log_file_to_evaluate = get_interaction_log_filepath() 
    
    # 示例：如果要评估名为 "rag_interactions_20250529.jsonl" 的特定文件
    specific_log_filename = "rag_interactions_20250530.jsonl"
    log_file_to_evaluate = os.path.join(os.path.dirname(get_interaction_log_filepath()), specific_log_filename)

    batch_eval_logger.info(f"Target log file for evaluation: {log_file_to_evaluate}")

    # 2. 通过环境变量控制是否使用模拟Gemini响应:
    #    在终端运行前设置:
    #    - 使用模拟: USE_SIMULATED_GEMINI_CYPHER_EVAL="true" python -m zhz_agent.run_batch_cypher_evaluation
    #    - 使用真实API: python -m zhz_agent.run_batch_cypher_evaluation 
    #                  (确保 GEMINI_API_KEY 和代理已在 .env 或环境中配置)

    should_run_simulated = os.getenv("USE_SIMULATED_GEMINI_CYPHER_EVAL", "false").lower() == "true"

    if os.path.exists(log_file_to_evaluate):
        asyncio.run(run_batch_evaluation(log_file_to_evaluate, app_version_for_eval="0.1.1_batch_eval_final"))
    elif should_run_simulated: # 如果文件不存在但明确要求模拟，也尝试运行（尽管可能没有输入）
        batch_eval_logger.warning(f"Log file {log_file_to_evaluate} not found, but USE_SIMULATED_GEMINI_CYPHER_EVAL is true. Proceeding with no input data (will likely process 0 entries).")
        asyncio.run(run_batch_evaluation(log_file_to_evaluate, app_version_for_eval="0.1.1_batch_eval_sim_no_file"))
    else:
        batch_eval_logger.warning(f"Log file {log_file_to_evaluate} not found and not using simulated responses. Batch evaluation will not run.")