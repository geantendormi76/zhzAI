# zhz_agent/run_batch_answer_evaluation.py
import asyncio
import json
import os
from typing import List, Dict, Any, Optional # <--- 确保导入 Optional
import glob
from datetime import datetime # 确保导入

try:
    from zhz_rag.evaluation.evaluator import evaluate_answer_with_gemini
    from zhz_rag.utils.common_utils import RAG_INTERACTION_LOGS_DIR, find_latest_rag_interaction_log # 导入常量和函数
    from zhz_rag.config.pydantic_models import RetrievedDocument
except ImportError as e:
    print(f"ERROR: Could not import necessary modules: {e}")
    print("Make sure this script is run in an environment where 'zhz_agent' is accessible.")
    print("If running from outside the project root, you might need to set PYTHONPATH.")
    print("Example: PYTHONPATH=/path/to/your/project python -m zhz_agent.run_batch_answer_evaluation")
    exit(1)

import logging

# 配置此脚本的logger
batch_answer_eval_logger = logging.getLogger("BatchAnswerEvaluationLogger")
batch_answer_eval_logger.setLevel(logging.INFO)
if not batch_answer_eval_logger.hasHandlers():
    _console_handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _console_handler.setFormatter(_formatter)
    batch_answer_eval_logger.addHandler(_console_handler)
    batch_answer_eval_logger.info("--- BatchAnswerEvaluationLogger configured ---")

def format_contexts_for_evaluation(context_docs: List[Dict[str, Any]]) -> str:
    """
    将从日志中解析出的上下文文档列表格式化为单一字符串，以便传递给评估LLM。
    可以考虑加入来源和分数信息。
    """
    formatted_contexts = []
    for i, doc_data in enumerate(context_docs):
        try:
            doc = RetrievedDocument(**doc_data)
            context_str = f"--- Context Snippet {i+1} ---\n"
            context_str += f"Source Type: {doc.source_type}\n"
            if doc.score is not None:
                context_str += f"Original Score: {doc.score:.4f}\n"
            if doc.metadata and doc.metadata.get("chunk_id"):
                 context_str += f"Chunk ID: {doc.metadata.get('chunk_id')}\n"
            elif doc.metadata and doc.metadata.get("id"):
                 context_str += f"ID: {doc.metadata.get('id')}\n"
            context_str += f"Content: {doc.content}\n"
            formatted_contexts.append(context_str)
        except Exception as e:
            batch_answer_eval_logger.warning(f"Could not parse a context document fully: {doc_data}. Error: {e}")
            content = doc_data.get("content", "[Content not available]")
            source_type = doc_data.get("source_type", "unknown_source")
            formatted_contexts.append(f"--- Context Snippet {i+1} (Parsing Warning) ---\nSource Type: {source_type}\nContent: {content}\n")
            
    return "\n\n".join(formatted_contexts) if formatted_contexts else "No context provided." # 使用双换行分隔上下文片段

async def batch_evaluate_answers_from_file(
    rag_interaction_log_filepath: str,
    app_version: str = "0.1.0",
    use_simulated_api: bool = False,
    api_call_delay: float = 4.1
) -> Dict[str, int]: # 返回一个包含统计信息的字典
    """
    从指定的RAG交互日志文件中读取记录，筛选并评估答案。
    """
    batch_answer_eval_logger.info(f"Starting batch Answer evaluation for log file: {rag_interaction_log_filepath}")
    processed_count = 0
    evaluated_count = 0
    failed_to_extract_count = 0
    skipped_no_answer_count = 0

    if use_simulated_api:
        batch_answer_eval_logger.warning("Batch answer evaluation is USING SIMULATED Gemini responses.")
    else:
        batch_answer_eval_logger.info("Batch answer evaluation is using REAL Gemini API calls.")

    if not os.path.exists(rag_interaction_log_filepath):
        batch_answer_eval_logger.error(f"Input RAG interaction log file not found: {rag_interaction_log_filepath}")
        return {"processed": 0, "evaluated": 0, "skipped_no_answer": 0, "failed_extract": 0}

    try:
        with open(rag_interaction_log_filepath, 'r', encoding='utf-8') as f_in:
            for line_number, line in enumerate(f_in, 1):
                try:
                    interaction_log = json.loads(line.strip())
                except json.JSONDecodeError:
                    batch_answer_eval_logger.warning(f"Skipping malformed JSON line {line_number} in {rag_interaction_log_filepath}")
                    continue

                processed_count += 1
                if interaction_log.get("task_type") == "rag_query_processing_success":
                    user_question = interaction_log.get("user_query")
                    generated_answer = interaction_log.get("processed_llm_output")
                    original_id = interaction_log.get("interaction_id")
                    context_docs_raw = interaction_log.get("retrieved_context_docs")
                    
                    if not context_docs_raw and interaction_log.get("debug_info"):
                        context_docs_raw = interaction_log.get("debug_info",{}).get("retrieved_context_docs")

                    if user_question and generated_answer and original_id and context_docs_raw and isinstance(context_docs_raw, list):
                        retrieved_contexts_str = format_contexts_for_evaluation(context_docs_raw)
                        
                        batch_answer_eval_logger.info(f"Evaluating Answer for interaction_id: {original_id} - User Question: {user_question[:50]}...")
                        
                        evaluation_result = await evaluate_answer_with_gemini(
                            user_question=user_question,
                            retrieved_contexts=retrieved_contexts_str,
                            generated_answer=generated_answer,
                            original_interaction_id=original_id,
                            app_version=app_version
                        )

                        if evaluation_result:
                            evaluated_count += 1
                            batch_answer_eval_logger.info(f"Successfully evaluated Answer for interaction_id: {original_id}. Overall Score: {evaluation_result.get('evaluation_summary', {}).get('overall_answer_quality_score')}")
                        else:
                            batch_answer_eval_logger.warning(f"Answer evaluation returned None or failed for interaction_id: {original_id}")
                        
                        if not use_simulated_api: 
                            batch_answer_eval_logger.info(f"Waiting for {api_call_delay} seconds before next API call...")
                            await asyncio.sleep(api_call_delay)
                    else:
                        failed_to_extract_count +=1
                        if not generated_answer: skipped_no_answer_count +=1
                        batch_answer_eval_logger.warning(f"Skipping RAG success log line {line_number} due to missing user_question, generated_answer, context_docs, or interaction_id. Log: {str(interaction_log)[:200]}")
                
                if processed_count > 0 and processed_count % 10 == 0:
                    batch_answer_eval_logger.info(f"Progress: Processed {processed_count} log entries. Evaluated {evaluated_count} answers so far. Skipped (no answer/data): {skipped_no_answer_count}. Failed to extract: {failed_to_extract_count}.")

    except Exception as e:
        batch_answer_eval_logger.error(f"Error during batch answer evaluation: {e}", exc_info=True)

    summary = {"processed": processed_count, "evaluated": evaluated_count, "skipped_no_answer": skipped_no_answer_count, "failed_extract": failed_to_extract_count}
    batch_answer_eval_logger.info(f"Batch Answer evaluation finished. Summary: {summary}")
    return summary

if __name__ == "__main__":
    # 1. 自动查找最新的原始RAG交互日志文件
    log_file_to_evaluate = find_latest_rag_interaction_log(RAG_INTERACTION_LOGS_DIR)

    # 2. 从环境变量决定是否模拟API调用
    use_simulated = os.getenv("USE_SIMULATED_GEMINI_ANSWER_EVAL", "false").lower() == "true"
    app_version_tag = "0.1.2_batch_answer_auto"
    if use_simulated:
        app_version_tag += "_simulated"

    if log_file_to_evaluate: 
        asyncio.run(batch_evaluate_answers_from_file(
            rag_interaction_log_filepath=log_file_to_evaluate,
            app_version=app_version_tag,
            use_simulated_api=use_simulated
        ))
    # 如果文件不存在但明确要求模拟 (通常用于测试脚本本身，虽然没有输入数据意义不大)
    elif use_simulated: 
        batch_answer_eval_logger.warning(f"RAG output log file not found, but USE_SIMULATED_GEMINI_ANSWER_EVAL is true. Running with a dummy path (will process 0 entries).")
        asyncio.run(batch_evaluate_answers_from_file(
            rag_interaction_log_filepath="dummy_non_existent_file.jsonl", # 传递一个虚拟路径
            app_version=app_version_tag + "_no_file",
            use_simulated_api=use_simulated
        ))
    else:
        batch_answer_eval_logger.warning(f"No suitable RAG output log file found and not using simulated responses. Batch answer evaluation will not run.")