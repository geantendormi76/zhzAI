# zhz_rag/evaluation/batch_eval_answer.py
import asyncio
import json
import os
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import glob
from datetime import datetime

# --- 从项目中导入必要的模块 ---
try:
    from zhz_rag.evaluation.evaluator import evaluate_answer_with_gemini
    from zhz_rag.utils.common_utils import (
        find_latest_rag_interaction_log,
        load_jsonl_file,
        RAG_INTERACTION_LOGS_DIR
    )
    from zhz_rag.config.pydantic_models import RetrievedDocument
except ImportError as e:
    # ... (错误处理保持不变) ...
    print(f"ERROR: Could not import necessary modules in batch_eval_answer.py: {e}")
    print("Make sure this script is run in an environment where 'zhz_rag' package is accessible.")
    exit(1)

# --- 类型检查时导入资源类 ---
if TYPE_CHECKING:
    from zhz_rag_pipeline_dagster.zhz_rag_pipeline.resources import GeminiAPIResource

import logging

# --- 配置此脚本的logger ---
batch_answer_eval_logger = logging.getLogger("BatchAnswerEvaluationLogger")
batch_answer_eval_logger.setLevel(logging.INFO)
if not batch_answer_eval_logger.hasHandlers():
    _console_handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _console_handler.setFormatter(_formatter)
    batch_answer_eval_logger.addHandler(_console_handler)
    batch_answer_eval_logger.propagate = False
batch_answer_eval_logger.info("--- BatchAnswerEvaluationLogger configured ---")


def format_contexts_for_evaluation(context_docs_raw: List[Dict[str, Any]]) -> str:
    """
    将从日志中解析出的上下文文档列表格式化为单一字符串，以便传递给评估LLM。
    这个函数与您之前在 batch_eval_answer.py 中的版本保持一致。
    """
    formatted_contexts = []
    if not context_docs_raw or not isinstance(context_docs_raw, list):
        batch_answer_eval_logger.warning("format_contexts_for_evaluation received no context or invalid format.")
        return "No context provided or context in unexpected format."

    for i, doc_data in enumerate(context_docs_raw):
        try:
            # 尝试使用RetrievedDocument模型解析，如果原始日志中已经是这个结构
            # 或者直接从字典中获取字段
            content = doc_data.get("content", "[Content not available]")
            source_type = doc_data.get("source_type", "unknown_source")
            score = doc_data.get("score")
            metadata = doc_data.get("metadata", {})
            chunk_id = metadata.get("chunk_id") or metadata.get("id") # 兼容不同日志格式

            context_str = f"--- Context Snippet {i+1} ---\n"
            context_str += f"Source Type: {source_type}\n"
            if score is not None:
                try:
                    context_str += f"Original Score: {float(score):.4f}\n"
                except (ValueError, TypeError):
                    context_str += f"Original Score: {score}\n" # 如果分数不是数字，直接用原始值
            if chunk_id:
                 context_str += f"Chunk ID: {chunk_id}\n"
            context_str += f"Content: {content}\n"
            formatted_contexts.append(context_str)
        except Exception as e:
            batch_answer_eval_logger.warning(f"Could not parse a context document fully in format_contexts_for_evaluation: {doc_data}. Error: {e}")
            content = doc_data.get("content", "[Content not available]")
            source_type = doc_data.get("source_type", "unknown_source")
            formatted_contexts.append(f"--- Context Snippet {i+1} (Parsing Warning) ---\nSource Type: {source_type}\nContent: {content}\n")
            
    return "\n\n".join(formatted_contexts) if formatted_contexts else "No context provided."


async def run_answer_batch_evaluation(
    gemini_resource_for_evaluator: 'GeminiAPIResource', # <--- 新增参数
    rag_interaction_log_filepath: str,
    app_version: str = "0.1.0",
    use_simulated_api: bool = False,
    api_call_delay: float = 4.1
) -> Dict[str, int]:
    """
    从指定的RAG交互日志文件中读取记录，筛选并评估最终答案。

    Args:
        rag_interaction_log_filepath (str): RAG交互日志文件的路径。
        app_version (str): 当前应用的日志版本标签。
        use_simulated_api (bool): 是否使用模拟的Gemini API响应。
        api_call_delay (float): 真实API调用之间的延迟秒数。

    Returns:
        Dict[str, int]: 包含评估统计信息的字典。
    """
    batch_answer_eval_logger.info(f"Starting batch Answer evaluation for log file: {rag_interaction_log_filepath}")
    batch_answer_eval_logger.info(f"Parameters: app_version='{app_version}', use_simulated_api={use_simulated_api}, api_call_delay={api_call_delay}s")

    processed_count = 0
    evaluated_count = 0
    failed_to_extract_count = 0
    skipped_missing_data_count = 0 # 用于记录因数据不全而跳过的情况

    if use_simulated_api:
        batch_answer_eval_logger.warning("Batch Answer evaluation is USING SIMULATED Gemini responses.")
    else:
        batch_answer_eval_logger.info("Batch Answer evaluation is using REAL Gemini API calls.")

    # 使用新的通用函数加载日志数据
    interaction_logs = load_jsonl_file(rag_interaction_log_filepath)

    if not interaction_logs:
        batch_answer_eval_logger.error(f"No data loaded from RAG interaction log file: {rag_interaction_log_filepath}. Exiting.")
        return {"processed": 0, "evaluated": 0, "skipped_missing_data": 0, "failed_extract": 0, "file_not_found_or_empty": 1}

    for line_number, interaction_log in enumerate(interaction_logs, 1):
        processed_count += 1
        # 我们关注的是成功完成RAG查询处理的日志条目
        if interaction_log.get("task_type") == "rag_query_processing_success":
            user_question = interaction_log.get("user_query")
            generated_answer = interaction_log.get("processed_llm_output") # RAG的最终答案
            original_id = interaction_log.get("interaction_id")
            
            # retrieved_context_docs 应该是由 FusionEngine 融合重排后的最终上下文文档列表
            # 在 rag_mcp_service.py 中，它被记录在 "retrieved_context_docs" 字段下
            context_docs_raw = interaction_log.get("retrieved_context_docs")

            if not context_docs_raw and interaction_log.get("debug_info"): # 备用方案，以防万一
                context_docs_raw = interaction_log.get("debug_info",{}).get("retrieved_context_docs")

            if user_question and generated_answer and original_id and context_docs_raw and isinstance(context_docs_raw, list):
                retrieved_contexts_str_for_eval = format_contexts_for_evaluation(context_docs_raw)
                
                batch_answer_eval_logger.info(f"Evaluating Answer for interaction_id: {original_id} - User Question: {user_question[:50]}...")
                
                evaluation_result = await evaluate_answer_with_gemini(
                    gemini_resource=gemini_resource_for_evaluator, # <--- 传递资源实例
                    user_question=user_question,
                    retrieved_contexts=retrieved_contexts_str_for_eval,
                    generated_answer=generated_answer,
                    original_interaction_id=original_id,
                    app_version=app_version
                )

                if evaluation_result:
                    evaluated_count += 1
                    summary = evaluation_result.get("evaluation_summary", {})
                    overall_score = summary.get("overall_answer_quality_score", "N/A")
                    batch_answer_eval_logger.info(f"Successfully evaluated Answer for interaction_id: {original_id}. Overall Score: {overall_score}")
                else:
                    batch_answer_eval_logger.warning(f"Answer evaluation returned None or failed for interaction_id: {original_id}")
                
                if not use_simulated_api:
                    batch_answer_eval_logger.info(f"Waiting for {api_call_delay} seconds before next API call...")
                    await asyncio.sleep(api_call_delay)
            else:
                skipped_missing_data_count += 1
                log_preview = {k: v for k, v in interaction_log.items() if k in ["interaction_id", "task_type", "user_query"]}
                log_preview["generated_answer_present"] = bool(generated_answer)
                log_preview["context_docs_present_and_list"] = isinstance(context_docs_raw, list) if context_docs_raw else False
                batch_answer_eval_logger.warning(f"Skipping RAG success log entry {line_number} due to missing critical data. Details: {log_preview}")
        
        if processed_count > 0 and processed_count % 10 == 0: # 日志打印频率
            batch_answer_eval_logger.info(f"Progress: Processed {processed_count} log entries. Evaluated {evaluated_count} answers so far. Skipped (missing data): {skipped_missing_data_count}.")

    summary = {
        "total_processed_from_log": processed_count,
        "answers_evaluated": evaluated_count,
        "skipped_due_to_missing_data": skipped_missing_data_count,
        "failed_to_extract_fields_for_eval": failed_to_extract_count # 这个字段可能在此脚本中意义不大，因为我们主要按task_type筛选
    }
    batch_answer_eval_logger.info(f"Batch Answer evaluation finished. Summary: {summary}")
    return summary


if __name__ == "__main__":
    # 1. 自动查找最新的原始RAG交互日志文件
    log_file_to_evaluate = find_latest_rag_interaction_log(RAG_INTERACTION_LOGS_DIR)

    # 2. 从环境变量决定是否模拟API调用 和 配置API调用延迟
    use_simulated_env = os.getenv("USE_SIMULATED_GEMINI_ANSWER_EVAL", "false").lower() == "true"
    api_delay_env = float(os.getenv("GEMINI_API_CALL_DELAY_SECONDS", "4.1"))

    app_version_tag_env = os.getenv("APP_VERSION_TAG", "0.1.3_batch_answer_refactored")
    if use_simulated_env:
        app_version_tag_env += "_simulated"

    if log_file_to_evaluate:
        batch_answer_eval_logger.info(f"Found RAG interaction log to process for answer evaluation: {log_file_to_evaluate}")
        asyncio.run(run_answer_batch_evaluation(
            rag_interaction_log_filepath=log_file_to_evaluate,
            app_version=app_version_tag_env,
            use_simulated_api=use_simulated_env,
            api_call_delay=api_delay_env
        ))
    elif use_simulated_env:
        batch_answer_eval_logger.warning(f"RAG interaction log file not found, but USE_SIMULATED_GEMINI_ANSWER_EVAL is true. Running with a dummy path (will process 0 entries).")
        asyncio.run(run_answer_batch_evaluation(
            rag_interaction_log_filepath="dummy_non_existent_file.jsonl",
            app_version=app_version_tag_env + "_no_file",
            use_simulated_api=use_simulated_env,
            api_call_delay=api_delay_env
        ))
    else:
        batch_answer_eval_logger.warning(f"No suitable RAG interaction log file found in '{RAG_INTERACTION_LOGS_DIR}' and not using simulated responses. Batch Answer evaluation will not run.")