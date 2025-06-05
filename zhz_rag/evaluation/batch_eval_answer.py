# zhz_rag/evaluation/batch_eval_answer.py
import asyncio
import json
import os
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Union
import glob
from datetime import datetime

try:
    from zhz_rag.evaluation.evaluator import evaluate_answer_with_gemini
    from zhz_rag.utils.common_utils import (
        find_latest_rag_interaction_log,
        load_jsonl_file,
        RAG_INTERACTION_LOGS_DIR
    )
    from zhz_rag.config.pydantic_models import RetrievedDocument
except ImportError as e:
    print(f"ERROR: Could not import necessary modules in batch_eval_answer.py: {e}")
    print("Make sure this script is run in an environment where 'zhz_rag' package is accessible.")
    exit(1)

if TYPE_CHECKING:
    from zhz_rag_pipeline_dagster.zhz_rag_pipeline.resources import GeminiAPIResource

import logging

batch_answer_eval_logger = logging.getLogger("BatchAnswerEvaluationLogger")
# 保留 DEBUG 级别，以便在需要时仍可查看详细日志，但常规 INFO 日志会更简洁
batch_answer_eval_logger.setLevel(logging.DEBUG) 
if not batch_answer_eval_logger.hasHandlers():
    _console_handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _console_handler.setFormatter(_formatter)
    batch_answer_eval_logger.addHandler(_console_handler)
    batch_answer_eval_logger.propagate = False
batch_answer_eval_logger.info("--- BatchAnswerEvaluationLogger configured (Level: DEBUG) ---")


def format_contexts_for_evaluation(context_docs_raw: List[Dict[str, Any]]) -> str:
    formatted_contexts = []
    if not context_docs_raw or not isinstance(context_docs_raw, list):
        batch_answer_eval_logger.warning("format_contexts_for_evaluation received no context or invalid format.")
        batch_answer_eval_logger.debug("DEBUG_FORMAT_CTX: context_docs_raw is empty or not a list.") 
        return "No context provided or context in unexpected format."

    batch_answer_eval_logger.debug(f"DEBUG_FORMAT_CTX: --- format_contexts_for_evaluation ---")
    batch_answer_eval_logger.debug(f"DEBUG_FORMAT_CTX: Received context_docs_raw length: {len(context_docs_raw)}")
    if context_docs_raw:
        batch_answer_eval_logger.debug(f"DEBUG_FORMAT_CTX: First item of context_docs_raw (type: {type(context_docs_raw[0])}): {str(context_docs_raw[0])[:500]}...")

    for i, doc_data in enumerate(context_docs_raw): 
        batch_answer_eval_logger.debug(f"DEBUG_FORMAT_CTX:   Processing doc_data item {i} (type: {type(doc_data)}): {str(doc_data)[:300]}...")
        try:
            content = doc_data.get("content", "[Content not available]") 
            source_type = doc_data.get("source_type", "unknown_source") 
            score = doc_data.get("score") 
            metadata = doc_data.get("metadata", {})
            chunk_id = metadata.get("chunk_id") or metadata.get("id") 

            context_str = f"--- Context Snippet {i+1} ---\n"
            context_str += f"Source Type: {source_type}\n"
            if score is not None:
                try:
                    context_str += f"Original Score: {float(score):.4f}\n"
                except (ValueError, TypeError):
                    context_str += f"Original Score: {score}\n" 
            if chunk_id:
                 context_str += f"Chunk ID: {chunk_id}\n"
            context_str += f"Content: {content}\n"
            formatted_contexts.append(context_str)
            batch_answer_eval_logger.debug(f"DEBUG_FORMAT_CTX:     Formatted context snippet {i+1} (content part first 100 chars): {str(content)[:100]}...")

        except Exception as e:
            batch_answer_eval_logger.warning(f"Could not parse a context document fully in format_contexts_for_evaluation: {doc_data}. Error: {e}")
            content = doc_data.get("content", "[Content not available]") 
            source_type = doc_data.get("source_type", "unknown_source") 
            formatted_contexts.append(f"--- Context Snippet {i+1} (Parsing Warning) ---\nSource Type: {source_type}\nContent: {content}\n")
    
    final_formatted_str = "\n\n".join(formatted_contexts) if formatted_contexts else "No context provided."
    batch_answer_eval_logger.debug(f"DEBUG_FORMAT_CTX: --- format_contexts_for_evaluation: Final formatted string (first 500 chars): {final_formatted_str[:500]}...")
    return final_formatted_str


async def run_answer_batch_evaluation(
    gemini_resource_for_evaluator: 'GeminiAPIResource',
    rag_interaction_log_filepath: str,
    app_version: str = "0.1.0",
    use_simulated_api: bool = False,
    api_call_delay: float = 4.1,
    target_task_types: Union[str, List[str]] = "rag_query_processing_full_log",
    field_mapping: Optional[Dict[str, Union[str, List[str]]]] = None
) -> Dict[str, int]:
    batch_answer_eval_logger.info(f"Starting batch Answer evaluation for log file: {rag_interaction_log_filepath}")
    batch_answer_eval_logger.info(f"Parameters: app_version='{app_version}', use_simulated_api={use_simulated_api}, api_call_delay={api_call_delay}s")
    batch_answer_eval_logger.info(f"Target task types: {target_task_types}")
    batch_answer_eval_logger.info(f"Field mapping: {field_mapping}")

    processed_count = 0
    evaluated_count = 0
    skipped_missing_data_count = 0
    skipped_task_type_mismatch = 0

    if use_simulated_api:
        batch_answer_eval_logger.warning("Batch Answer evaluation is USING SIMULATED Gemini responses.")
    else:
        batch_answer_eval_logger.info("Batch Answer evaluation is using REAL Gemini API calls.")

    interaction_logs = load_jsonl_file(rag_interaction_log_filepath)

    if not interaction_logs:
        batch_answer_eval_logger.error(f"No data loaded from RAG interaction log file: {rag_interaction_log_filepath}. Exiting.")
        return {"processed": 0, "evaluated": 0, "skipped_missing_data": 0, "skipped_task_type_mismatch": 0, "file_not_found_or_empty": 1}

    # --- 移除了临时筛选特定 interaction_id 的代码 ---

    if isinstance(target_task_types, str):
        target_task_types_list = [target_task_types]
    else:
        target_task_types_list = target_task_types

    default_field_map = {
        "user_query": ["user_query", "original_user_query", "query"],
        "generated_answer": ["generated_answer", "processed_llm_output", "final_answer_from_llm", "final_answer"],
        "interaction_id": ["interaction_id", "original_interaction_id"],
        "context_docs": ["final_context_docs_full", "final_context_docs_summary"]
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
        
        # --- 添加日志打印 interaction_log 的键，用于调试 ---
        batch_answer_eval_logger.debug(f"DEBUG_INTERACTION_LOG: Keys in interaction_log for entry {line_number}: {list(interaction_log.keys())}")
        if "final_context_docs_full" in interaction_log:
            batch_answer_eval_logger.debug(f"DEBUG_INTERACTION_LOG: interaction_log['final_context_docs_full'] (first item preview): {str(interaction_log['final_context_docs_full'][0])[:200] if interaction_log['final_context_docs_full'] else 'Empty or None'}")
        else:
            batch_answer_eval_logger.debug(f"DEBUG_INTERACTION_LOG: 'final_context_docs_full' NOT in interaction_log for entry {line_number}.")
        if "final_context_docs_summary" in interaction_log:
            batch_answer_eval_logger.debug(f"DEBUG_INTERACTION_LOG: interaction_log['final_context_docs_summary'] (first item preview): {str(interaction_log['final_context_docs_summary'][0])[:200] if interaction_log['final_context_docs_summary'] else 'Empty or None'}")
        else:
            batch_answer_eval_logger.debug(f"DEBUG_INTERACTION_LOG: 'final_context_docs_summary' NOT in interaction_log for entry {line_number}.")
        # --- 结束日志打印 ---


        user_question = get_field_value(interaction_log, "user_query")
        generated_answer = get_field_value(interaction_log, "generated_answer")
        original_id = get_field_value(interaction_log, "interaction_id")
        context_docs_raw = get_field_value(interaction_log, "context_docs")
        
        batch_answer_eval_logger.debug(
            f"Log entry {line_number}: task_type='{current_task_type}', id='{original_id}', "
            f"q_present={bool(user_question)}, ans_present={bool(generated_answer)}, ctx_present_is_list={isinstance(context_docs_raw, list) if context_docs_raw else False}"
        )

        if user_question and generated_answer and original_id and context_docs_raw and isinstance(context_docs_raw, list):
            batch_answer_eval_logger.info(f"DEBUG_EVAL: For ID {original_id}, UserQ: '{str(user_question)[:50]}...', GenAns: '{str(generated_answer)[:50]}...', Contexts count: {len(context_docs_raw)}")
            retrieved_contexts_str_for_eval = format_contexts_for_evaluation(context_docs_raw)
            
            batch_answer_eval_logger.info(f"Evaluating Answer for interaction_id: {original_id} - User Question: '{str(user_question)[:50]}...' - Generated Answer: '{str(generated_answer)[:50]}...'")
            
            evaluation_result = await evaluate_answer_with_gemini(
                gemini_resource=gemini_resource_for_evaluator,
                user_question=str(user_question),
                retrieved_contexts=retrieved_contexts_str_for_eval,
                generated_answer=str(generated_answer),
                original_interaction_id=str(original_id),
                app_version=app_version
            )

            if evaluation_result:
                evaluated_count += 1
                summary = evaluation_result.get("evaluation_summary", {})
                overall_score = summary.get("overall_answer_quality_score", "N/A")
                batch_answer_eval_logger.info(f"Successfully evaluated Answer for interaction_id: {original_id}. Overall Score: {overall_score}")
            else:
                batch_answer_eval_logger.warning(f"Answer evaluation returned None or failed for interaction_id: {original_id}")
            
            if not use_simulated_api and evaluated_count > 0 : 
                batch_answer_eval_logger.info(f"Waiting for {api_call_delay} seconds before next API call...")
                await asyncio.sleep(api_call_delay)
        else:
            skipped_missing_data_count += 1
            log_preview = {
                "interaction_id": original_id, 
                "task_type": current_task_type, 
                "user_question_found": bool(user_question),
                "generated_answer_found": bool(generated_answer),
                "context_docs_found_and_list": isinstance(context_docs_raw, list) if context_docs_raw else False
            }
            batch_answer_eval_logger.warning(f"Skipping RAG log entry {line_number} due to missing critical data. Details: {log_preview}")
        
        if processed_count > 0 and processed_count % 10 == 0:
            batch_answer_eval_logger.info(f"Progress: Processed {processed_count} log entries. Evaluated {evaluated_count} answers so far. Skipped (type mismatch): {skipped_task_type_mismatch}. Skipped (missing data): {skipped_missing_data_count}.")

    summary = {
        "total_log_entries_read": processed_count,
        "target_task_type_entries_found": processed_count - skipped_task_type_mismatch,
        "answers_evaluated_successfully": evaluated_count,
        "skipped_due_to_missing_data_in_target_entries": skipped_missing_data_count,
    }
    batch_answer_eval_logger.info(f"Batch Answer evaluation finished. Summary: {summary}")
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
                self.log = batch_answer_eval_logger
        
        if hasattr(gemini_eval_resource, 'setup_for_execution'):
             gemini_eval_resource.setup_for_execution(MockContext())
        batch_answer_eval_logger.info(f"GeminiAPIResource for evaluation initialized successfully using Dagster's resource class.")

    except ImportError:
        batch_answer_eval_logger.critical("CRITICAL: Could not import GeminiAPIResource. Ensure Dagster modules are in PYTHONPATH or installed.")
        gemini_eval_resource = None 
    except Exception as e_res_init:
        batch_answer_eval_logger.critical(f"CRITICAL: Error initializing GeminiAPIResource: {e_res_init}", exc_info=True)
        gemini_eval_resource = None


    log_file_to_evaluate = find_latest_rag_interaction_log(RAG_INTERACTION_LOGS_DIR)
    use_simulated_env = os.getenv("USE_SIMULATED_GEMINI_ANSWER_EVAL", "false").lower() == "true"
    api_delay_env = float(os.getenv("GEMINI_API_CALL_DELAY_SECONDS", "4.1"))
    app_version_tag_env = os.getenv("APP_VERSION_TAG", "0.1.4_batch_answer_flexible")
    if use_simulated_env:
        app_version_tag_env += "_simulated"

    rag_service_task_types = ["rag_query_processing_full_log"]
    rag_service_field_map = {
        "user_query": "original_user_query",
        "generated_answer": "final_answer_from_llm",
        "interaction_id": "interaction_id",
        "context_docs": ["final_context_docs_full", "final_context_docs_summary"] 
    }

    if not gemini_eval_resource:
        batch_answer_eval_logger.error("Cannot proceed with evaluation as GeminiAPIResource is not available.")
    elif log_file_to_evaluate:
        batch_answer_eval_logger.info(f"Found RAG interaction log to process for answer evaluation: {log_file_to_evaluate}")
        asyncio.run(run_answer_batch_evaluation(
            gemini_resource_for_evaluator=gemini_eval_resource,
            rag_interaction_log_filepath=log_file_to_evaluate,
            app_version=app_version_tag_env,
            use_simulated_api=use_simulated_env,
            api_call_delay=api_delay_env,
            target_task_types=rag_service_task_types, 
            field_mapping=rag_service_field_map      
        ))
    elif use_simulated_env:
        batch_answer_eval_logger.warning(f"RAG interaction log file not found, but USE_SIMULATED_GEMINI_ANSWER_EVAL is true. Running with a dummy path (will process 0 entries).")
        if gemini_eval_resource: # Check again if resource is available for simulated run
            asyncio.run(run_answer_batch_evaluation(
                gemini_resource_for_evaluator=gemini_eval_resource,
                rag_interaction_log_filepath="dummy_non_existent_file.jsonl", 
                app_version=app_version_tag_env + "_no_file",
                use_simulated_api=use_simulated_env,
                api_call_delay=api_delay_env,
                target_task_types=rag_service_task_types,
                field_mapping=rag_service_field_map
            ))
        else:
            batch_answer_eval_logger.error("GeminiAPIResource for evaluation could not be initialized (even for simulated run). Aborting simulated run.")
    else:
        batch_answer_eval_logger.warning(f"No suitable RAG interaction log file found in '{RAG_INTERACTION_LOGS_DIR}' and not using simulated responses. Batch Answer evaluation will not run.")