# zhz_agent/run_batch_answer_evaluation.py
import asyncio
import json
import os
from typing import List, Dict, Any
import glob
from datetime import datetime # 确保导入

try:
    from zhz_agent.evaluation import evaluate_answer_with_gemini
    from zhz_agent.utils import get_interaction_log_filepath
    from zhz_agent.pydantic_models import RetrievedDocument
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
    _formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
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
            # 尝试用Pydantic模型解析，以获取类型提示和验证，但如果日志中的结构简单，直接访问字典键也可以
            doc = RetrievedDocument(**doc_data)
            context_str = f"--- Context Snippet {i+1} ---\n"
            context_str += f"Source Type: {doc.source_type}\n"
            if doc.score is not None:
                context_str += f"Original Score: {doc.score:.4f}\n"
            # 如果metadata中有chunk_id或其他有用信息，也可以加入
            if doc.metadata and doc.metadata.get("chunk_id"):
                 context_str += f"Chunk ID: {doc.metadata.get('chunk_id')}\n"
            elif doc.metadata and doc.metadata.get("id"): # 有些地方可能存的是id
                 context_str += f"ID: {doc.metadata.get('id')}\n"

            context_str += f"Content: {doc.content}\n"
            formatted_contexts.append(context_str)
        except Exception as e:
            batch_answer_eval_logger.warning(f"Could not parse a context document fully: {doc_data}. Error: {e}")
            # 即使解析Pydantic失败，也尝试提取基本内容
            content = doc_data.get("content", "[Content not available]")
            source_type = doc_data.get("source_type", "unknown_source")
            formatted_contexts.append(f"--- Context Snippet {i+1} (Parsing Warning) ---\nSource Type: {source_type}\nContent: {content}\n")
            
    return "\n".join(formatted_contexts) if formatted_contexts else "No context provided."


async def run_batch_evaluation_for_answers(input_log_filepath: str, app_version_for_eval: str = "0.1.0"):
    """
    读取包含RAG交互的JSONL文件，筛选出成功的RAG查询结果，并对生成的答案进行批量评估。
    """
    batch_answer_eval_logger.info(f"Starting batch Answer evaluation for log file: {input_log_filepath}")
    processed_count = 0
    evaluated_count = 0
    failed_to_extract_count = 0
    skipped_no_answer_count = 0

    USE_SIMULATED_FLAG = os.getenv("USE_SIMULATED_GEMINI_ANSWER_EVAL", "false").lower() == "true"
    if USE_SIMULATED_FLAG:
        batch_answer_eval_logger.warning("Batch answer evaluation is configured to USE SIMULATED Gemini responses.")
    else:
        batch_answer_eval_logger.info("Batch answer evaluation is configured to use REAL Gemini API calls.")

    if not os.path.exists(input_log_filepath):
        batch_answer_eval_logger.error(f"Input log file not found: {input_log_filepath}")
        return

    try:
        with open(input_log_filepath, 'r', encoding='utf-8') as f_in:
            for line_number, line in enumerate(f_in, 1):
                try:
                    interaction_log = json.loads(line.strip())
                except json.JSONDecodeError:
                    batch_answer_eval_logger.warning(f"Skipping malformed JSON line {line_number} in {input_log_filepath}")
                    continue

                processed_count += 1
                # 我们关注的是RAG流程成功生成答案的日志
                if interaction_log.get("task_type") == "rag_query_processing_success":
                    user_question = interaction_log.get("user_query")
                    generated_answer = interaction_log.get("processed_llm_output") # 这是RAG的最终答案
                    original_id = interaction_log.get("interaction_id")
                    
                    # 从debug_info或直接从顶层获取上下文文档列表
                    # 假设在rag_service.py中，我们将retrieved_context_docs存入了顶层日志
                    context_docs_raw = interaction_log.get("retrieved_context_docs") 
                    if not context_docs_raw and interaction_log.get("debug_info"): # 兼容旧格式
                        context_docs_raw = interaction_log.get("debug_info",{}).get("retrieved_context_docs")


                    if user_question and generated_answer and original_id and context_docs_raw and isinstance(context_docs_raw, list):
                        retrieved_contexts_str = format_contexts_for_evaluation(context_docs_raw)
                        
                        batch_answer_eval_logger.info(f"Evaluating Answer for interaction_id: {original_id} - User Question: {user_question[:50]}...")
                        
                        evaluation_result = await evaluate_answer_with_gemini(
                            user_question=user_question,
                            retrieved_contexts=retrieved_contexts_str,
                            generated_answer=generated_answer,
                            original_interaction_id=original_id,
                            app_version=app_version_for_eval
                        )

                        if evaluation_result:
                            evaluated_count += 1
                            batch_answer_eval_logger.info(f"Successfully evaluated Answer for interaction_id: {original_id}. Overall Score: {evaluation_result.get('evaluation_summary', {}).get('overall_answer_quality_score')}")
                        else:
                            batch_answer_eval_logger.warning(f"Answer evaluation returned None or failed for interaction_id: {original_id}")
                        
                        if not USE_SIMULATED_FLAG: 
                            wait_time = 4.1 
                            batch_answer_eval_logger.info(f"Waiting for {wait_time} seconds before next API call...")
                            await asyncio.sleep(wait_time)
                    else:
                        failed_to_extract_count +=1
                        if not generated_answer: skipped_no_answer_count +=1
                        batch_answer_eval_logger.warning(f"Skipping RAG success log line {line_number} due to missing user_question, generated_answer, context_docs, or interaction_id. Log: {str(interaction_log)[:200]}")
                
                if processed_count > 0 and processed_count % 10 == 0: # 每处理10条日志打印一次进度
                    batch_answer_eval_logger.info(f"Progress: Processed {processed_count} log entries. Evaluated {evaluated_count} answers so far. Skipped (no answer/data): {skipped_no_answer_count}. Failed to extract: {failed_to_extract_count}.")

    except Exception as e:
        batch_answer_eval_logger.error(f"Error during batch answer evaluation: {e}", exc_info=True)

    batch_answer_eval_logger.info(f"Batch Answer evaluation finished. Total log entries processed: {processed_count}. Answers evaluated: {evaluated_count}. Skipped (no answer/data): {skipped_no_answer_count}. Entries failed to extract data: {failed_to_extract_count}.")


if __name__ == "__main__":
    # 1. 确定要评估的原始RAG交互日志文件路径
    #    这个文件应该包含 task_type: "rag_query_processing_success" 的记录
    #    并且这些记录中应该有 "user_query", "processed_llm_output" (作为答案), 和 "retrieved_context_docs"
    
    # 默认评估当天的 "rag_interactions_YYYYMMDD.jsonl" 文件。
    # 您可能需要修改这里以指向包含历史数据的特定日志文件。
    log_file_to_evaluate_rag_outputs = get_interaction_log_filepath() 
    
    # 示例：如果要评估名为 "rag_interactions_20250530.jsonl" 的特定文件
    # specific_log_filename = "rag_interactions_20250530.jsonl" # 假设这是包含了RAG输出的日志
    # log_file_to_evaluate_rag_outputs = os.path.join(os.path.dirname(get_interaction_log_filepath()), specific_log_filename)

    batch_answer_eval_logger.info(f"Target RAG output log file for answer evaluation: {log_file_to_evaluate_rag_outputs}")

    # 2. 通过环境变量控制是否使用模拟Gemini响应:
    #    在终端运行前设置:
    #    - 使用模拟: USE_SIMULATED_GEMINI_ANSWER_EVAL="true" python -m zhz_agent.run_batch_answer_evaluation
    #    - 使用真实API: python -m zhz_agent.run_batch_answer_evaluation
    #                  (确保 GEMINI_API_KEY 和代理已在 .env 或环境中配置)

    should_run_simulated_answer = os.getenv("USE_SIMULATED_GEMINI_ANSWER_EVAL", "false").lower() == "true"

    if os.path.exists(log_file_to_evaluate_rag_outputs):
        asyncio.run(run_batch_evaluation_for_answers(log_file_to_evaluate_rag_outputs, app_version_for_eval="0.1.1_batch_answer_eval"))
    elif should_run_simulated_answer:
        batch_answer_eval_logger.warning(f"RAG output log file {log_file_to_evaluate_rag_outputs} not found, but USE_SIMULATED_GEMINI_ANSWER_EVAL is true. Proceeding with no input data.")
        asyncio.run(run_batch_evaluation_for_answers(log_file_to_evaluate_rag_outputs, app_version_for_eval="0.1.1_batch_answer_eval_sim_no_file"))
    else:
        batch_answer_eval_logger.warning(f"RAG output log file {log_file_to_evaluate_rag_outputs} not found and not using simulated responses. Batch answer evaluation will not run.")