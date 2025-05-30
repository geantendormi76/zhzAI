# zhz_agent/refine_cypher_finetune_data.py
import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import glob
from datetime import datetime

# 假设 utils.py 和 constants.py 在同一个 zhz_agent 包内
try:
    from zhz_agent.utils import get_interaction_log_filepath, get_evaluation_result_log_filepath, find_latest_rag_interaction_log # <--- 修改这里
    from zhz_agent.constants import NEW_KG_SCHEMA_DESCRIPTION
except ImportError as e:
    print(f"ERROR: Could not import necessary modules: {e}")
    # ... (错误处理)
    exit(1)

import logging

# 配置此脚本的logger
refine_logger = logging.getLogger("RefineFinetuneDataLogger")
refine_logger.setLevel(logging.INFO)
if not refine_logger.hasHandlers():
    _console_handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _console_handler.setFormatter(_formatter)
    refine_logger.addHandler(_console_handler)
    refine_logger.info("--- RefineFinetuneDataLogger configured ---")

# --- 配置 ---
# 原始RAG交互日志的目录 (包含cypher_generation类型)
RAG_LOG_DIR = "zhz_agent/rag_eval_data/"
# Gemini评估结果日志的目录
EVAL_LOG_DIR = "zhz_agent/rag_eval_data/"
# 输出微调数据文件的目录
FINETUNE_DATA_DIR = "zhz_agent/finetune_data/"
os.makedirs(FINETUNE_DATA_DIR, exist_ok=True)


def load_logs_to_dict(filepath: str, key_field: str = "interaction_id") -> Dict[str, Dict[str, Any]]:
    """将JSONL文件加载到一个以指定字段为键的字典中。"""
    data_dict = {}
    if not os.path.exists(filepath):
        refine_logger.error(f"Log file not found: {filepath}")
        return data_dict
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())
                if key_field in log_entry:
                    data_dict[log_entry[key_field]] = log_entry
                # 对于评估日志，我们可能需要用 original_interaction_id_ref 作为键
                elif key_field == "original_interaction_id_ref" and log_entry.get("original_interaction_id_ref"):
                    data_dict[log_entry["original_interaction_id_ref"]] = log_entry
            except json.JSONDecodeError:
                refine_logger.warning(f"Skipping malformed JSON line in {filepath}")
    return data_dict

def construct_qwen_input_prompt(user_question: str, schema_description: str) -> str:
    """
    根据用户问题和Schema描述构建Qwen生成Cypher时的完整输入Prompt。
    这个函数应该与 llm.py 中 generate_cypher_query 内部构建Prompt的逻辑一致。
    """
    # 这是我们在 llm.py -> generate_cypher_query 中使用的Prompt模板
    # 我们需要确保这里的模板与Qwen实际接收到的一致
    # 注意：这里使用了最新的V7版本（或您当前使用的版本）的Schema描述作为基础
    # 如果您的 generate_cypher_query 中的模板不同，请相应调整
    prompt = f"""<|im_start|>system
{schema_description}
<|im_end|>
<|im_start|>user
用户问题: {user_question}
<|im_end|>
<|im_start|>assistant
"""
    return prompt

def generate_finetune_samples_for_cypher(
    rag_interaction_logs: Dict[str, Dict[str, Any]],
    cypher_evaluation_logs: Dict[str, Dict[str, Any]]
) -> List[Dict[str, str]]:
    """
    根据原始交互日志和Gemini评估日志，生成用于Cypher微调的样本。
    返回一个列表，每个元素是 {"prompt": "...", "completion": "..."}
    """
    finetune_samples = []
    processed_ids = set()

    refine_logger.info(f"Processing {len(rag_interaction_logs)} RAG interaction logs and {len(cypher_evaluation_logs)} Cypher evaluation logs.")

    for interaction_id, rag_log in rag_interaction_logs.items():
        if rag_log.get("task_type") != "cypher_generation":
            continue

        if interaction_id in processed_ids:
            continue
        processed_ids.add(interaction_id)

        user_question = rag_log.get("user_query") # 这是 llm.py 中记录的 user_query_for_log
        qwen_generated_cypher = rag_log.get("processed_llm_output")
        # 原始输入给Qwen的Prompt，理论上应该从rag_log的 "llm_input_prompt" 获取
        # 如果没有记录完整prompt，我们就用 user_question 和最新的 Schema Prompt 来重构
        qwen_input_prompt = rag_log.get("llm_input_prompt")
        if not qwen_input_prompt: # 如果原始日志没有存完整输入prompt
            if user_question:
                qwen_input_prompt = construct_qwen_input_prompt(user_question, NEW_KG_SCHEMA_DESCRIPTION)
            else:
                refine_logger.warning(f"Skipping Cypher log {interaction_id} due to missing user_question for prompt reconstruction.")
                continue
        
        if not user_question or qwen_generated_cypher is None: # qwen_generated_cypher可能是空字符串
            refine_logger.warning(f"Skipping Cypher log {interaction_id} due to missing user_question or qwen_generated_cypher.")
            continue

        ideal_cypher_output = None
        source_of_ideal = "unknown"

        eval_log = cypher_evaluation_logs.get(interaction_id)

        if eval_log and eval_log.get("eval_llm_processed_output_json"):
            eval_json = eval_log["eval_llm_processed_output_json"]
            overall_score = eval_json.get("evaluation_summary", {}).get("overall_quality_score_cypher")
            gemini_suggestion = eval_json.get("suggestion_for_improvement_cypher", "").strip()

            try: # 尝试将评分转为整数
                overall_score = int(overall_score)
            except (ValueError, TypeError):
                refine_logger.warning(f"Could not parse overall_quality_score_cypher for {interaction_id}: {overall_score}")
                overall_score = 0 # 默认给个低分

            if overall_score >= 4: # 假设4分及以上认为是高质量的
                ideal_cypher_output = qwen_generated_cypher
                source_of_ideal = "qwen_high_score"
            elif gemini_suggestion and gemini_suggestion != "无法生成Cypher查询。" and "cannot be improved" not in gemini_suggestion.lower() and "needs to be extended" not in gemini_suggestion.lower() and "MATCH" in gemini_suggestion.upper(): # Gemini给出了具体的Cypher建议
                # TODO (可选的LLM辅助提纯): 在这里可以加入调用LLM验证gemini_suggestion的逻辑
                ideal_cypher_output = gemini_suggestion
                source_of_ideal = "gemini_suggestion"
            elif "无法生成Cypher查询" in gemini_suggestion or (overall_score <= 2 and ("hallucinated" in eval_log["eval_llm_raw_output"].lower() or "schema violation" in eval_log["eval_llm_raw_output"].lower())):
                # 如果Gemini建议无法生成，或者低分且明确提到幻觉/Schema违规
                ideal_cypher_output = "无法生成Cypher查询。"
                source_of_ideal = "gemini_cannot_generate"
            else:
                # 其他情况，可能需要人工审核和提供黄金标准
                refine_logger.info(f"Cypher log {interaction_id} (Qwen: '{qwen_generated_cypher}') needs manual review. Gemini score: {overall_score}, Suggestion: '{gemini_suggestion}'")
                # 暂时跳过，或者标记后导出给人工处理
                continue 
        else:
            # 没有找到对应的Gemini评估日志，或者评估日志无效
            # 我们可以选择跳过，或者如果信任Qwen的原始输出（不推荐），或者有其他标准
            refine_logger.warning(f"No valid Gemini evaluation found for Cypher log {interaction_id}. Qwen's output: '{qwen_generated_cypher}'. Skipping for finetune data.")
            continue
            
        if ideal_cypher_output is not None:
            finetune_samples.append({
                "prompt": qwen_input_prompt, # 这是给Qwen的完整输入
                "completion": ideal_cypher_output, # 这是期望Qwen的输出
                "original_qwen_cypher": qwen_generated_cypher, # 保留原始输出供参考
                "gemini_score": overall_score if eval_log else None, # 保留Gemini评分
                "source_of_ideal": source_of_ideal, # 记录理想输出的来源
                "interaction_id": interaction_id
            })

    refine_logger.info(f"Generated {len(finetune_samples)} Cypher finetuning samples.")
    return finetune_samples


if __name__ == "__main__":
    # 1. 确定要处理的原始RAG交互日志文件 (包含cypher_generation)
    #    和对应的Gemini评估结果日志文件 (包含cypher_evaluation_result)
    
    # 自动查找最新的原始RAG交互日志
    rag_log_file = find_latest_rag_interaction_log(RAG_LOG_DIR) # utils.py中的函数
    
    # 构造对应的Gemini Cypher评估结果文件名
    # 假设评估文件名与原始日志文件名日期部分相同，且评估类型固定
    eval_log_file = None
    if rag_log_file:
        rag_log_basename = os.path.basename(rag_log_file)
        date_str_match = "".join(filter(str.isdigit, rag_log_basename)) # 提取文件名中的日期部分
        if len(date_str_match) >= 8: # 确保提取到至少YYYYMMDD
            date_str = date_str_match[:8]

            # 根据 evaluation.py 中 log_interaction_data 的 evaluation_name_for_file 参数构造
            evaluation_name = "cypher_gemini_flash" 
            eval_file_name = f"eval_results_{evaluation_name}_{date_str}.jsonl"
            
            eval_log_file = os.path.join(EVAL_LOG_DIR, eval_file_name)
            refine_logger.info(f"Attempting to load Cypher evaluation results from: {eval_log_file}")
        else:
            refine_logger.error(f"Could not reliably extract date from RAG log filename: {rag_log_basename}")
    
    if not rag_log_file or not eval_log_file or not os.path.exists(eval_log_file):
        refine_logger.error("Required log files not found. Exiting.")
        if not rag_log_file: refine_logger.error(f"RAG interaction log missing (expected pattern rag_interactions_*.jsonl in {RAG_LOG_DIR})")
        if rag_log_file and (not eval_log_file or not os.path.exists(eval_log_file)): refine_logger.error(f"Cypher evaluation result log missing (expected: {eval_log_file})")
    else:
        rag_interactions = load_logs_to_dict(rag_log_file, key_field="interaction_id")
        cypher_evaluations = load_logs_to_dict(eval_log_file, key_field="original_interaction_id_ref")

        if rag_interactions and cypher_evaluations:
            finetune_data = generate_finetune_samples_for_cypher(rag_interactions, cypher_evaluations)
            
            if finetune_data:
                # 获取当前日期用于文件名
                today_for_filename = datetime.now().strftime("%Y%m%d")
                output_filepath = os.path.join(FINETUNE_DATA_DIR, f"cypher_finetune_samples_{today_for_filename}.jsonl")
                
                with open(output_filepath, 'w', encoding='utf-8') as f_out:
                    for sample in finetune_data:
                        f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                refine_logger.info(f"Successfully saved {len(finetune_data)} Cypher finetuning samples to: {output_filepath}")
                
                # 也可以同时保存一个CSV版本供人工审查
                try:
                    df = pd.DataFrame(finetune_data)
                    csv_output_filepath = os.path.join(FINETUNE_DATA_DIR, f"cypher_finetune_samples_review_{today_for_filename}.csv")
                    df.to_csv(csv_output_filepath, index=False, encoding='utf-8-sig')
                    refine_logger.info(f"Reviewable CSV saved to: {csv_output_filepath}")
                except Exception as e_csv:
                    refine_logger.error(f"Failed to save review CSV: {e_csv}")
            else:
                refine_logger.info("No finetuning samples were generated.")
        else:
            refine_logger.error("Failed to load data from log files.")