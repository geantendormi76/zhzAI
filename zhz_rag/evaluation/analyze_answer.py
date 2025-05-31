# analyze_answer_evaluations.py
import json
import os
import pandas as pd
from typing import List, Dict, Any
from collections import Counter

# --- 配置 ---
LOG_FILE_DIR = "zhz_rag/stored_data/evaluation_results_logs/" # 日志文件所在目录

# --- 修改这里以指向新的答案评估结果文件名 ---
# 您需要根据 evaluation.py 中 evaluation_name_for_file 的设置来构造正确的文件名
EVALUATION_NAME_ANSWER = "answer_gemini_flash" # <<<--- 确保这个与 evaluation.py 中的一致
# from datetime import datetime, timezone
# today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
# LOG_FILE_NAME_ANSWER = f"eval_results_{EVALUATION_NAME_ANSWER}_{today_str}.jsonl"
LOG_FILE_NAME_ANSWER = f"eval_results_{EVALUATION_NAME_ANSWER}_20250530.jsonl" # <<<--- 请修改为您实际评估结果的日志文件名 (注意日期)

LOG_FILE_PATH_ANSWER = os.path.join(LOG_FILE_DIR, LOG_FILE_NAME_ANSWER)

OUTPUT_CSV_FILE_ANSWER = os.path.join(LOG_FILE_DIR, f"analysis_{EVALUATION_NAME_ANSWER}_{os.path.splitext(LOG_FILE_NAME_ANSWER)[0].split('_')[-1]}.csv")

def load_answer_evaluation_logs(filepath: str) -> List[Dict[str, Any]]:
    """从JSONL文件中加载所有 'answer_evaluation_result' 类型的日志。"""
    eval_logs = []
    if not os.path.exists(filepath):
        print(f"错误：答案评估结果日志文件未找到: {filepath}")
        return eval_logs
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            try:
                log_entry = json.loads(line.strip())
                # 我们现在直接读取评估结果文件，所以主要检查 task_type 和是否有评估结果
                if log_entry.get("task_type") == "answer_evaluation_result" and log_entry.get("eval_llm_processed_output_json"):
                    eval_logs.append(log_entry)
            except json.JSONDecodeError:
                print(f"警告：跳过格式错误的JSON行 {line_number} 在文件 {filepath}")
            except Exception as e:
                print(f"警告：处理行 {line_number} 时发生错误: {e}")
    print(f"从 {filepath} 加载了 {len(eval_logs)} 条 'answer_evaluation_result' 日志。")
    return eval_logs

def extract_answer_evaluation_details(log_entry: Dict[str, Any]) -> Dict[str, Any]:
    """从单条答案评估日志中提取关键信息。"""
    details = {}
    eval_data = log_entry.get("eval_llm_processed_output_json", {}) # 这是Gemini评估的JSON输出
    
    details["interaction_id_ref"] = log_entry.get("original_interaction_id_ref")
    details["user_question"] = log_entry.get("user_question_for_eval")
    details["generated_answer"] = log_entry.get("generated_answer_for_eval")
    # details["retrieved_contexts_char_count"] = log_entry.get("retrieved_contexts_for_eval_char_count") # 可选
    
    if eval_data and isinstance(eval_data, dict):
        summary = eval_data.get("evaluation_summary", {})
        dimensions = eval_data.get("dimensions", {})
        
        details["overall_answer_quality_score"] = summary.get("overall_answer_quality_score")
        details["main_strengths_answer"] = summary.get("main_strengths_answer")
        details["main_weaknesses_answer"] = summary.get("main_weaknesses_answer")
        
        faithfulness = dimensions.get("faithfulness", {})
        details["faithfulness_score"] = faithfulness.get("score")
        details["faithfulness_reasoning"] = faithfulness.get("reasoning")
        details["faithfulness_problematic_segments"] = "; ".join(faithfulness.get("problematic_answer_segments_faithfulness", []))

        relevance = dimensions.get("relevance", {})
        details["relevance_score"] = relevance.get("score")
        details["relevance_reasoning"] = relevance.get("reasoning")
        
        completeness = dimensions.get("completeness", {})
        details["completeness_context_sufficiency"] = completeness.get("context_sufficiency_assessment")
        details["completeness_context_reasoning"] = completeness.get("context_sufficiency_reasoning")
        details["completeness_score"] = completeness.get("score")
        details["completeness_reasoning"] = completeness.get("reasoning")
        
        coherence = dimensions.get("coherence_fluency", {})
        details["coherence_score"] = coherence.get("score")
        details["coherence_reasoning"] = coherence.get("reasoning")

        actionability = dimensions.get("actionability_usability", {})
        details["actionability_score"] = actionability.get("score")
        details["actionability_reasoning"] = actionability.get("reasoning")
        
        details["gemini_suggestion_answer"] = eval_data.get("suggestion_for_answer_improvement")
    else:
        print(f"警告: interaction_id_ref {details.get('interaction_id_ref')} 的 eval_llm_processed_output_json 为空或格式不正确。")

    return details

def analyze_answer_evaluations_summary(eval_data_list: List[Dict[str, Any]]):
    """对提取的答案评估数据进行初步分析总结。"""
    if not eval_data_list:
        print("没有答案评估数据可供分析。")
        return

    df = pd.DataFrame(eval_data_list)
    
    score_columns = [
        "overall_answer_quality_score", "faithfulness_score", "relevance_score",
        "completeness_score", "coherence_score", "actionability_score"
    ]
    for col in score_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("\n--- 初步答案评估分析 ---")
    print(f"总评估条目数: {len(df)}")

    for col_name in score_columns:
        if col_name in df.columns and not df[col_name].isnull().all():
            print(f"\n维度: {col_name}")
            print(df[col_name].describe())
            print("评分分布:")
            print(df[col_name].value_counts(dropna=False).sort_index())
        else:
            print(f"\n维度: {col_name} - 无有效数据或全为空值")
            
    # 可以添加对特定文本字段的分析，例如最常见的weaknesses等，但需要更复杂的文本处理
    # 例如，统计 context_sufficiency_assessment 的分布
    if "completeness_context_sufficiency" in df.columns:
        print("\n上下文充分性评估分布:")
        print(df["completeness_context_sufficiency"].value_counts(dropna=False))

    try:
        df.to_csv(OUTPUT_CSV_FILE_ANSWER, index=False, encoding='utf-8-sig')
        print(f"\n答案评估分析结果已保存到: {OUTPUT_CSV_FILE_ANSWER}")
    except Exception as e:
        print(f"\n保存答案评估CSV文件失败: {e}")

if __name__ == "__main__":
    print(f"正在从答案评估结果日志文件加载数据: {LOG_FILE_PATH_ANSWER}")
    answer_evaluation_logs = load_answer_evaluation_logs(LOG_FILE_PATH_ANSWER)
    
    if answer_evaluation_logs:
        extracted_answer_details_list = []
        for log in answer_evaluation_logs:
            details = extract_answer_evaluation_details(log)
            extracted_answer_details_list.append(details)
        
        analyze_answer_evaluations_summary(extracted_answer_details_list)
    else:
        print("未能加载任何答案评估日志，分析中止。")