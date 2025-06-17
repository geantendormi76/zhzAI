# zhz_rag/evaluation/batch_eval_cypher.py
import asyncio
import json
import os
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Union
import glob
from datetime import datetime

try:
    from zhz_rag.evaluation.evaluator import evaluate_cypher_with_gemini
    from zhz_rag.utils.common_utils import (
        find_latest_rag_interaction_log,
        load_jsonl_file,
        RAG_INTERACTION_LOGS_DIR
    )
except ImportError as e:
    print(f"ERROR: Could not import necessary modules in batch_eval_cypher.py: {e}")
    print("Make sure this script is run in an environment where 'zhz_rag' package is accessible.")
    exit(1)

if TYPE_CHECKING:
    from zhz_rag_pipeline_dagster.zhz_rag_pipeline.resources import GeminiAPIResource

import logging

batch_cypher_eval_logger = logging.getLogger("BatchCypherEvaluationLogger")
batch_cypher_eval_logger.setLevel(logging.DEBUG) # 设置为 DEBUG 以便查看详细日志
if not batch_cypher_eval_logger.hasHandlers():
    _console_handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _console_handler.setFormatter(_formatter)
    batch_cypher_eval_logger.addHandler(_console_handler)
    batch_cypher_eval_logger.propagate = False
batch_cypher_eval_logger.info("--- BatchCypherEvaluationLogger configured (Level: DEBUG) ---")


async def run_cypher_batch_evaluation(
    gemini_resource_for_evaluator: 'GeminiAPIResource',
    rag_interaction_log_filepath: str,
    app_version: str = "0.1.0",
    use_simulated_api: bool = False,
    api_call_delay: float = 4.1,
    target_task_types: Union[str, List[str]] = "cypher_generation_final_attempt_local_service",
    field_mapping: Optional[Dict[str, Union[str, List[str]]]] = None
) -> Dict[str, int]:
    batch_cypher_eval_logger.info(f"Starting batch Cypher evaluation for log file: {rag_interaction_log_filepath}")
    batch_cypher_eval_logger.info(f"Parameters: app_version='{app_version}', use_simulated_api={use_simulated_api}, api_call_delay={api_call_delay}s")
    batch_cypher_eval_logger.info(f"Target task types: {target_task_types}")
    batch_cypher_eval_logger.info(f"Field mapping: {field_mapping}")

    processed_count = 0
    evaluated_count = 0
    skipped_no_cypher_count = 0
    failed_to_extract_count = 0
    skipped_task_type_mismatch = 0

    if use_simulated_api:
        batch_cypher_eval_logger.warning("Batch Cypher evaluation is USING SIMULATED Gemini responses.")
    else:
        batch_cypher_eval_logger.info("Batch Cypher evaluation is using REAL Gemini API calls.")

    interaction_logs = load_jsonl_file(rag_interaction_log_filepath)

    if not interaction_logs:
        batch_cypher_eval_logger.error(f"No data loaded from RAG interaction log file: {rag_interaction_log_filepath}. Exiting.")
        return {"processed": 0, "evaluated": 0, "skipped_no_cypher":0, "failed_extract": 0, "skipped_task_type_mismatch":0, "file_not_found_or_empty": 1}

    if isinstance(target_task_types, str):
        target_task_types_list = [target_task_types]
    else:
        target_task_types_list = target_task_types

    default_field_map = {
        "user_query": ["user_query_for_task", "user_query", "original_user_query"],
        "generated_cypher": ["raw_llm_output", "processed_llm_output"], # raw_llm_output for cypher_generation_final_attempt_local_service
        "interaction_id": ["interaction_id", "original_interaction_id"]
    }
    current_field_map = default_field_map.copy()
    if field_mapping:
        for key, value in field_mapping.items():
            if isinstance(value, str):
                current_field_map[key] = [value]
            else:
                current_field_map[key] = value

    def get_field_value(log_entry: Dict[str, Any], field_key: str) -> Any:
        for actual_field_name in current_field_map.get(field_key, []):
            if actual_field_name in log_entry:
                return log_entry[actual_field_name]
        return None

    for line_number, interaction_log in enumerate(interaction_logs, 1):
        processed_count += 1
        current_task_type = interaction_log.get("task_type")

        if current_task_type not in target_task_types_list:
            skipped_task_type_mismatch +=1
            continue
            
        batch_cypher_eval_logger.debug(f"DEBUG_CYPHER_EVAL: Processing log entry {line_number} with task_type '{current_task_type}'")

        user_question = get_field_value(interaction_log, "user_query")
        generated_cypher_raw = get_field_value(interaction_log, "generated_cypher")
        original_id = get_field_value(interaction_log, "interaction_id")

        if user_question and original_id:
            generated_cypher_to_eval = None # 初始化
            if isinstance(generated_cypher_raw, str) and generated_cypher_raw.strip():
                # 对于 "kg_executed_query_for_eval" 类型的日志，"generated_query" 字段直接包含SQL语句
                # 我们假设字段映射已将 "generated_query" 映射到 generated_cypher_raw
                generated_cypher_to_eval = generated_cypher_raw.strip()
                batch_cypher_eval_logger.debug(f"Extracted query for eval (ID: {original_id}): '{generated_cypher_to_eval[:100]}...'")
            else:
                batch_cypher_eval_logger.warning(f"Log entry for ID {original_id} (task_type: {current_task_type}) has empty or non-string 'generated_query' (mapped to generated_cypher_raw). Value: {generated_cypher_raw}")

            if not generated_cypher_to_eval: # 再次检查，确保 generated_cypher_to_eval 有有效值
                batch_cypher_eval_logger.info(f"Skipping evaluation for interaction_id '{original_id}' as extracted Cypher is empty.")
                skipped_no_cypher_count += 1
                continue
            
            # We will evaluate "无法生成Cypher查询." as well, Gemini should score it appropriately.
            batch_cypher_eval_logger.info(f"Evaluating Cypher for interaction_id: {original_id} - User Question: {str(user_question)[:50]}... - Cypher: {str(generated_cypher_to_eval)[:100]}...")
            
            evaluation_result = await evaluate_cypher_with_gemini(
                gemini_resource=gemini_resource_for_evaluator,
                user_question=str(user_question),
                generated_cypher=str(generated_cypher_to_eval),
                original_interaction_id=str(original_id),
                app_version=app_version
            )

            if evaluation_result:
                evaluated_count += 1
                summary = evaluation_result.get("evaluation_summary", {})
                overall_score = summary.get("overall_quality_score_cypher", "N/A")
                batch_cypher_eval_logger.info(f"Successfully evaluated Cypher for interaction_id: {original_id}. Overall Score: {overall_score}")
            else:
                batch_cypher_eval_logger.warning(f"Cypher evaluation returned None or failed for interaction_id: {original_id}")
            
            if not use_simulated_api and evaluated_count > 0:
                batch_cypher_eval_logger.info(f"Waiting for {api_call_delay} seconds before next API call...")
                await asyncio.sleep(api_call_delay)
        else:
            failed_to_extract_count += 1
            batch_cypher_eval_logger.warning(f"Skipping cypher_generation log entry {line_number} due to missing user_query or interaction_id. Log content: {str(interaction_log)[:200]}...")
        
        if processed_count > 0 and processed_count % 10 == 0:
            batch_cypher_eval_logger.info(f"Progress: Processed {processed_count} log entries. Evaluated {evaluated_count} Cypher queries. Skipped (no cypher): {skipped_no_cypher_count}. Failed extract: {failed_to_extract_count}. Type mismatch: {skipped_task_type_mismatch}")

    summary = {
        "total_log_entries_read": processed_count,
        "target_task_type_entries_found": processed_count - skipped_task_type_mismatch,
        "cypher_queries_evaluated_successfully": evaluated_count,
        "skipped_empty_or_no_cypher": skipped_no_cypher_count,
        "failed_to_extract_fields_for_eval": failed_to_extract_count
    }
    batch_cypher_eval_logger.info(f"Batch Cypher evaluation finished. Summary: {summary}")
    return summary


if __name__ == "__main__":
    try:
        from zhz_rag_pipeline_dagster.zhz_rag_pipeline.resources import GeminiAPIResource, GeminiAPIResourceConfig
        
        gemini_model_name_env = os.getenv("GEMINI_MODEL_FOR_EVAL", "gemini/gemini-1.5-flash-latest")
        gemini_proxy_url_env = os.getenv("LITELLM_PROXY_URL") 

        gemini_resource_config = GeminiAPIResourceConfig(
            model_name=gemini_model_name_env,
            proxy_url=gemini_proxy_url_env
        )
        gemini_eval_resource = GeminiAPIResource(
            model_name=gemini_resource_config.model_name,
            proxy_url=gemini_resource_config.proxy_url,
            default_temperature=gemini_resource_config.default_temperature,
            default_max_tokens=gemini_resource_config.default_max_tokens
        )
        class MockContext: 
            def __init__(self):
                self.log = batch_cypher_eval_logger
        
        if hasattr(gemini_eval_resource, 'setup_for_execution'):
             gemini_eval_resource.setup_for_execution(MockContext())
        batch_cypher_eval_logger.info(f"GeminiAPIResource for Cypher evaluation initialized successfully.")

    except ImportError:
        batch_cypher_eval_logger.critical("CRITICAL: Could not import GeminiAPIResource. Ensure Dagster modules are in PYTHONPATH or installed.")
        gemini_eval_resource = None
    except Exception as e_res_init:
        batch_cypher_eval_logger.critical(f"CRITICAL: Error initializing GeminiAPIResource: {e_res_init}", exc_info=True)
        gemini_eval_resource = None

    log_file_to_evaluate = find_latest_rag_interaction_log(RAG_INTERACTION_LOGS_DIR)
    use_simulated_env = os.getenv("USE_SIMULATED_GEMINI_CYPHER_EVAL", "false").lower() == "true"
    api_delay_env = float(os.getenv("GEMINI_API_CALL_DELAY_SECONDS", "4.1"))
    app_version_tag_env = os.getenv("APP_VERSION_TAG", "0.1.4_batch_cypher_flexible")
    if use_simulated_env:
        app_version_tag_env += "_simulated"

    # --- 配置目标 task_type 和字段映射 ---
    cypher_gen_task_types = ["kg_executed_query_for_eval"] # <--- 查找新的 task_type
    cypher_gen_field_map = {
        "user_query": "user_query_for_task",      # 这个字段名在新的日志条目中是存在的
        "generated_cypher": "generated_query",    # 新的日志条目中，查询语句存储在 "generated_query" 字段
        "interaction_id": "interaction_id"        # interaction_id 仍然是主键
    }

    if not gemini_eval_resource:
        batch_cypher_eval_logger.error("Cannot proceed with Cypher evaluation as GeminiAPIResource is not available.")
    elif log_file_to_evaluate:
        batch_cypher_eval_logger.info(f"Found RAG interaction log to process for Cypher evaluation: {log_file_to_evaluate}")
        asyncio.run(run_cypher_batch_evaluation(
            gemini_resource_for_evaluator=gemini_eval_resource,
            rag_interaction_log_filepath=log_file_to_evaluate,
            app_version=app_version_tag_env,
            use_simulated_api=use_simulated_env,
            api_call_delay=api_delay_env,
            target_task_types=cypher_gen_task_types, # <--- 使用修改后的 task_types
            field_mapping=cypher_gen_field_map
        ))

    elif use_simulated_env:
        batch_cypher_eval_logger.warning(f"RAG interaction log file not found, but USE_SIMULATED_GEMINI_CYPHER_EVAL is true. Running with a dummy path.")
        if gemini_eval_resource:
            asyncio.run(run_cypher_batch_evaluation(
                gemini_resource_for_evaluator=gemini_eval_resource,
                rag_interaction_log_filepath="dummy_non_existent_file.jsonl", 
                app_version=app_version_tag_env + "_no_file",
                use_simulated_api=use_simulated_env,
                api_call_delay=api_delay_env,
                target_task_types=cypher_gen_task_types,
                field_mapping=cypher_gen_field_map
            ))
        else:
            batch_cypher_eval_logger.error("GeminiAPIResource for Cypher evaluation could not be initialized (even for simulated run). Aborting.")
    else:
        batch_cypher_eval_logger.warning(f"No suitable RAG interaction log file found in '{RAG_INTERACTION_LOGS_DIR}' and not using simulated responses. Batch Cypher evaluation will not run.")