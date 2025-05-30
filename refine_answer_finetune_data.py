# zhz_agent/refine_answer_finetune_data.py
import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional
import glob
from datetime import datetime

# 假设 utils.py 和 constants.py 在同一个 zhz_agent 包内
try:
    from zhz_agent.utils import find_latest_rag_interaction_log
    from zhz_agent.pydantic_models import RetrievedDocument
    # NO_ANSWER_PHRASE_ANSWER_CLEAN 将从 llm.py 导入，或者在constants.py中定义
    # 我们需要与 llm.py -> generate_answer_from_context 一致的 "无法回答" 短语
    from zhz_agent.llm import NO_ANSWER_PHRASE_ANSWER_CLEAN 
except ImportError as e:
    print(f"ERROR: Could not import necessary modules for refine_answer_finetune_data: {e}")
    exit(1)

import logging

# 配置此脚本的logger
refine_answer_logger = logging.getLogger("RefineAnswerFinetuneDataLogger")
refine_answer_logger.setLevel(logging.INFO)
if not refine_answer_logger.hasHandlers():
    _console_handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _console_handler.setFormatter(_formatter)
    refine_answer_logger.addHandler(_console_handler)
    refine_answer_logger.info("--- RefineAnswerFinetuneDataLogger configured ---")

# --- 配置 ---
RAG_LOG_DIR = "zhz_agent/rag_eval_data/"
EVAL_LOG_DIR = "zhz_agent/rag_eval_data/"
FINETUNE_DATA_DIR = "zhz_agent/finetune_data/"
os.makedirs(FINETUNE_DATA_DIR, exist_ok=True)

# --- 与 run_batch_answer_evaluation.py 中类似的上下文格式化函数 ---
def format_contexts_for_prompt(context_docs_raw: List[Dict[str, Any]]) -> str:
    """
    将从日志中解析出的上下文文档列表格式化为单一字符串，用于构建LLM的输入Prompt。
    这个格式应该与 llm.py -> generate_answer_from_context 中构建上下文的方式一致。
    """
    context_strings_for_llm = []
    if not context_docs_raw:
        return "No context provided."
        
    for i, doc_data in enumerate(context_docs_raw):
        try:
            # 尝试使用RetrievedDocument模型解析，如果原始日志中已经是这个结构
            # 但通常日志中可能是字典列表
            doc_content = doc_data.get("content", "[Content not available]")
            doc_source = doc_data.get("source_type", "unknown_source")
            doc_score = doc_data.get("score")
            
            # 与 rag_service.py 中准备上下文给LLM的格式保持一致
            # 在 rag_service.py 中是:
            # f"Source Type: {doc.source_type}, Score: {doc.score:.4f}\nContent: {doc.content}"
            # 我们这里也尽量模拟，但日志中的score可能不存在或格式不同
            header = f"Source Type: {doc_source}"
            if doc_score is not None:
                try:
                    header += f", Score: {float(doc_score):.4f}"
                except ValueError:
                    header += f", Score: {doc_score}" # 如果分数不是数字，直接用原始值
            
            context_strings_for_llm.append(f"{header}\nContent: {doc_content}")

        except Exception as e:
            refine_answer_logger.warning(f"Could not parse a context document fully for prompt: {doc_data}. Error: {e}")
            content = doc_data.get("content", "[Content not available]")
            context_strings_for_llm.append(f"Content: {content}") # 简化版

    return "\n\n---\n\n".join(context_strings_for_llm) if context_strings_for_llm else "No context provided."


def construct_qwen_answer_input_prompt(user_question: str, formatted_context: str) -> str:
    """
    根据用户问题和格式化的上下文构建Qwen生成答案时的完整输入Prompt。
    这个函数必须与 llm.py 中 generate_answer_from_context 内部构建Prompt的逻辑完全一致。
    """
    # --- 从 llm.py 的 generate_answer_from_context 函数复制并粘贴完整的 prompt 模板 ---
    # 注意：这里需要确保模板与 llm.py 中的完全一致
    prompt = f"""
<|im_start|>system
你是一个AI问答助手。你的任务是根据【上下文信息】回答【用户问题】。

**核心指令：**

1.  **尝试直接回答：** 请首先仔细阅读【上下文信息】，如果其中包含能直接回答【用户问题】的内容，请用上下文中的信息直接、简洁地回答。
2.  **忠实原文：** 你的回答必须严格基于【上下文信息】，禁止加入任何外部知识或个人观点。
3.  **如果无法回答：** 如果你分析了【上下文信息】后，确认其中确实没有能回答【用户问题】的明确信息，那么请只回答以下这句话：
    "根据目前提供的资料，我无法找到关于您问题的明确信息。"
    **不要添加任何其他解释、建议或反问。**

**请直接给出答案，或者只给出上述那句固定的“无法找到信息”的回复。**
<|im_start|>user
用户问题: {user_question}

上下文信息:
{formatted_context}
<|im_end|>
<|im_start|>assistant
"""
    return prompt

def load_logs_to_dict(filepath: str, key_field: str = "interaction_id") -> Dict[str, Dict[str, Any]]:
    """将JSONL文件加载到一个以指定字段为键的字典中。"""
    data_dict = {}
    if not os.path.exists(filepath):
        refine_answer_logger.error(f"Log file not found: {filepath}")
        return data_dict
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())
                if key_field in log_entry and log_entry[key_field]: # 确保key_field的值不是None或空
                    data_dict[log_entry[key_field]] = log_entry
                elif key_field == "original_interaction_id_ref" and log_entry.get("original_interaction_id_ref"):
                    data_dict[log_entry["original_interaction_id_ref"]] = log_entry
            except json.JSONDecodeError:
                refine_answer_logger.warning(f"Skipping malformed JSON line in {filepath}")
    return data_dict

def generate_finetune_samples_for_answer(
    rag_interaction_logs: Dict[str, Dict[str, Any]],
    answer_evaluation_logs: Dict[str, Dict[str, Any]]
) -> List[Dict[str, str]]:
    finetune_samples = []
    processed_ids = set()

    refine_answer_logger.info(f"Processing {len(rag_interaction_logs)} RAG interaction logs and {len(answer_evaluation_logs)} Answer evaluation logs.")

    for interaction_id, rag_log in rag_interaction_logs.items():
        if rag_log.get("task_type") != "rag_query_processing_success":
            continue

        if interaction_id in processed_ids:
            continue
        processed_ids.add(interaction_id)

        user_question = rag_log.get("user_query")
        qwen_generated_answer_raw = rag_log.get("processed_llm_output") # Qwen的原始答案
        # retrieved_context_docs 在 rag_log 中可能是 "retrieved_context_docs" 或 "retrieved_documents_summary"
        # 我们需要原始的、完整的上下文文档
        retrieved_context_docs_raw = rag_log.get("retrieved_context_docs") 
        
        if not retrieved_context_docs_raw and rag_log.get("debug_info"): # 尝试从debug_info获取
             retrieved_context_docs_raw = rag_log.get("debug_info",{}).get("retrieved_context_docs")


        if qwen_generated_answer_raw is None or not qwen_generated_answer_raw.strip():
            qwen_generated_answer = NO_ANSWER_PHRASE_ANSWER_CLEAN # 空答案视为无法回答
        else:
            qwen_generated_answer = qwen_generated_answer_raw.strip()

        if not user_question or not retrieved_context_docs_raw:
            refine_answer_logger.warning(f"Skipping RAG log {interaction_id} due to missing user_question or retrieved_context_docs.")
            continue
        
        # 构建Prompt
        formatted_contexts_for_prompt = format_contexts_for_prompt(retrieved_context_docs_raw)
        qwen_answer_input_prompt = construct_qwen_answer_input_prompt(user_question, formatted_contexts_for_prompt)

        ideal_answer_output = None
        source_of_ideal = "unknown"
        gemini_scores_for_log = {}

        eval_log = answer_evaluation_logs.get(interaction_id)

        if eval_log and eval_log.get("eval_llm_processed_output_json"):
            eval_json = eval_log["eval_llm_processed_output_json"]
            summary_eval = eval_json.get("evaluation_summary", {})
            dimensions_eval = eval_json.get("dimensions", {})
            
            overall_score_str = summary_eval.get("overall_answer_quality_score")
            faithfulness_score_str = dimensions_eval.get("faithfulness", {}).get("score")
            relevance_score_str = dimensions_eval.get("relevance", {}).get("score")
            completeness_score_str = dimensions_eval.get("completeness", {}).get("score")
            context_sufficiency = dimensions_eval.get("completeness", {}).get("context_sufficiency_assessment", "Unknown")
            gemini_suggestion_answer = eval_json.get("suggestion_for_answer_improvement", "").strip()

            try:
                overall_score = int(overall_score_str) if overall_score_str is not None else 0
                faithfulness_score = int(faithfulness_score_str) if faithfulness_score_str is not None else 0
                relevance_score = int(relevance_score_str) if relevance_score_str is not None else 0
                completeness_score = int(completeness_score_str) if completeness_score_str is not None else 0
                gemini_scores_for_log = {
                    "overall": overall_score,
                    "faithfulness": faithfulness_score,
                    "relevance": relevance_score,
                    "completeness": completeness_score,
                    "context_sufficiency": context_sufficiency
                }
            except (ValueError, TypeError) as e:
                refine_answer_logger.warning(f"Could not parse one or more scores for {interaction_id}: {e}")
                overall_score = faithfulness_score = relevance_score = completeness_score = 0
                gemini_scores_for_log = { # 记录解析失败
                    "overall": "parse_error", "faithfulness": "parse_error", 
                    "relevance": "parse_error", "completeness": "parse_error",
                    "context_sufficiency": context_sufficiency
                }

            # --- Completion选择逻辑 (初步) ---
            # 规则1: Qwen的答案本身就是标准的“无法回答”，并且Gemini评估上下文不足
            if qwen_generated_answer == NO_ANSWER_PHRASE_ANSWER_CLEAN and \
                context_sufficiency == "Insufficient" and \
                overall_score >= 4 : # Gemini 认为这个“无法回答”是高质量的
                    ideal_answer_output = NO_ANSWER_PHRASE_ANSWER_CLEAN
                    source_of_ideal = "qwen_no_answer_confirmed_by_gemini_context_insufficient_high_score"

            # 规则2: Gemini 评分很高 (例如 overall, faithfulness, relevance 都 >= 4)
            elif overall_score >= 4 and faithfulness_score >= 4 and relevance_score >= 4:
                ideal_answer_output = qwen_generated_answer
                source_of_ideal = "qwen_high_score_by_gemini"

            # 规则3: 上下文不足，且Qwen的答案不是标准的“无法回答”，但Gemini建议应指出信息不足
            elif context_sufficiency == "Insufficient" and \
                 "information is not available" in gemini_suggestion_answer.lower() or \
                 "context does not contain" in gemini_suggestion_answer.lower() or \
                 "cannot be answered" in gemini_suggestion_answer.lower():
                ideal_answer_output = NO_ANSWER_PHRASE_ANSWER_CLEAN
                source_of_ideal = "gemini_suggests_no_answer_due_to_insufficient_context"
            
            # 规则4: Gemini 给出了非常具体的、可直接采纳的改进建议 (这部分较难自动化判断，初期可跳过或标记人工)
            # 例如，如果 suggestion 是 "答案应为 'XXX' 而不是 'YYY'"
            # 暂时，我们只记录这类情况，不直接采纳建议作为 completion
            elif gemini_suggestion_answer and gemini_suggestion_answer != "No improvement needed." and len(gemini_suggestion_answer) < 150 : # 假设简短的建议更可能是直接替换
                 refine_answer_logger.info(f"Answer log {interaction_id} (Qwen: '{qwen_generated_answer[:100]}...') has a potentially actionable Gemini suggestion: '{gemini_suggestion_answer}'. Needs manual review for completion.")
                 # 可以在这里将 gemini_suggestion_answer 存入一个特殊字段，供人工审核后决定是否用作 completion
                 # continue # 暂时跳过，等待人工审核流程

            # 规则5: 其他情况，需要人工审核
            else:
                refine_answer_logger.info(f"Answer log {interaction_id} (Qwen: '{qwen_generated_answer[:100]}...') needs manual review. Gemini scores: {gemini_scores_for_log}, Suggestion: '{gemini_suggestion_answer[:100]}...'")
                continue

        else:
            refine_answer_logger.warning(f"No valid Gemini evaluation found for Answer log {interaction_id}. Qwen's output: '{qwen_generated_answer[:100]}...'. Skipping for finetune data.")
            continue
            
        if ideal_answer_output is not None:
            finetune_samples.append({
                "prompt": qwen_answer_input_prompt,
                "completion": ideal_answer_output.strip(),
                "original_qwen_answer": qwen_generated_answer,
                "gemini_scores": gemini_scores_for_log,
                "gemini_suggestion": gemini_suggestion_answer if eval_log else None,
                "source_of_ideal": source_of_ideal,
                "interaction_id": interaction_id
            })

    refine_answer_logger.info(f"Generated {len(finetune_samples)} Answer finetuning samples.")
    return finetune_samples


if __name__ == "__main__":
    rag_log_file = find_latest_rag_interaction_log(RAG_LOG_DIR)
    
    eval_log_file = None
    if rag_log_file:
        rag_log_basename = os.path.basename(rag_log_file)
        date_str_match = "".join(filter(str.isdigit, rag_log_basename))
        if len(date_str_match) >= 8:
            date_str = date_str_match[:8]
            evaluation_name = "answer_gemini_flash" # 与 evaluation.py 中一致
            eval_file_name = f"eval_results_{evaluation_name}_{date_str}.jsonl"
            eval_log_file = os.path.join(EVAL_LOG_DIR, eval_file_name)
            refine_answer_logger.info(f"Attempting to load Answer evaluation results from: {eval_log_file}")
        else:
            refine_answer_logger.error(f"Could not reliably extract date from RAG log filename: {rag_log_basename}")

    if not rag_log_file or not eval_log_file or not os.path.exists(eval_log_file):
        refine_answer_logger.error("Required log files for answer finetune data generation not found. Exiting.")
    else:
        rag_interactions = load_logs_to_dict(rag_log_file, key_field="interaction_id")
        answer_evaluations = load_logs_to_dict(eval_log_file, key_field="original_interaction_id_ref")

        if rag_interactions and answer_evaluations:
            finetune_data = generate_finetune_samples_for_answer(rag_interactions, answer_evaluations)
            
            if finetune_data:
                today_for_filename = datetime.now().strftime("%Y%m%d")
                output_filepath = os.path.join(FINETUNE_DATA_DIR, f"answer_finetune_samples_{today_for_filename}.jsonl")
                
                with open(output_filepath, 'w', encoding='utf-8') as f_out:
                    for sample in finetune_data:
                        f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                refine_answer_logger.info(f"Successfully saved {len(finetune_data)} Answer finetuning samples to: {output_filepath}")
                
                try:
                    df = pd.DataFrame(finetune_data)
                    csv_output_filepath = os.path.join(FINETUNE_DATA_DIR, f"answer_finetune_samples_review_{today_for_filename}.csv")
                    df.to_csv(csv_output_filepath, index=False, encoding='utf-8-sig')
                    refine_answer_logger.info(f"Reviewable CSV for answers saved to: {csv_output_filepath}")
                except Exception as e_csv:
                    refine_answer_logger.error(f"Failed to save answer review CSV: {e_csv}")
            else:
                refine_answer_logger.info("No answer finetuning samples were generated.")
        else:
            refine_answer_logger.error("Failed to load data from log files for answer finetuning.")