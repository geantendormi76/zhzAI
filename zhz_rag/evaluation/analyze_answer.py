# zhz_rag/evaluation/analyze_answer.py
import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional
from collections import Counter
from datetime import datetime
import glob 

# --- 从项目中导入必要的模块 ---
try:
    from zhz_rag.utils.common_utils import (
        load_jsonl_file, # <--- 使用新的通用函数
        EVALUATION_RESULTS_LOGS_DIR # 导入评估日志目录常量
    )
except ImportError as e:
    print(f"ERROR: Could not import necessary modules in analyze_answer.py: {e}")
    print("Make sure this script is run in an environment where 'zhz_rag' package is accessible.")
    exit(1)

import logging

# --- 配置此脚本的logger ---
analyze_answer_logger = logging.getLogger("AnalyzeAnswerLogger")
analyze_answer_logger.setLevel(logging.INFO)
if not analyze_answer_logger.hasHandlers():
    _console_handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _console_handler.setFormatter(_formatter)
    analyze_answer_logger.addHandler(_console_handler)
    analyze_answer_logger.propagate = False
analyze_answer_logger.info("--- AnalyzeAnswerLogger configured ---")

# --- 核心功能函数 ---

def extract_answer_evaluation_details(log_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    从单条已解析的答案评估日志条目中提取关键信息。
    这个函数与您之前在 analyze_answer.py 中的版本基本一致，稍作调整。
    """
    details = {}
    eval_data = log_entry.get("eval_llm_processed_output_json")

    if not eval_data or not isinstance(eval_data, dict):
        analyze_answer_logger.warning(f"Skipping log entry due to missing or invalid 'eval_llm_processed_output_json'. Interaction ID ref: {log_entry.get('original_interaction_id_ref')}")
        return None

    details["interaction_id_ref"] = log_entry.get("original_interaction_id_ref")
    details["user_question"] = log_entry.get("user_question_for_eval")
    details["generated_answer"] = log_entry.get("generated_answer_for_eval")
    
    summary = eval_data.get("evaluation_summary", {})
    dimensions = eval_data.get("dimensions", {})
    
    details["overall_answer_quality_score"] = summary.get("overall_answer_quality_score")
    details["main_strengths_answer"] = summary.get("main_strengths_answer")
    details["main_weaknesses_answer"] = summary.get("main_weaknesses_answer")
    
    faithfulness = dimensions.get("faithfulness", {})
    details["faithfulness_score"] = faithfulness.get("score")
    details["faithfulness_reasoning"] = faithfulness.get("reasoning")
    # 确保 problematic_answer_segments_faithfulness 是列表，然后 join
    problematic_segments = faithfulness.get("problematic_answer_segments_faithfulness", [])
    if isinstance(problematic_segments, list):
        details["faithfulness_problematic_segments"] = "; ".join(problematic_segments)
    else:
        details["faithfulness_problematic_segments"] = str(problematic_segments) # 以防万一不是列表

    relevance = dimensions.get("relevance", {})
    details["relevance_score"] = relevance.get("score")
    details["relevance_reasoning"] = relevance.get("reasoning")
    
    completeness = dimensions.get("completeness", {})
    details["completeness_context_sufficiency"] = completeness.get("context_sufficiency_assessment")
    details["completeness_context_reasoning"] = completeness.get("context_sufficiency_reasoning")
    details["completeness_score"] = completeness.get("score")
    details["completeness_reasoning"] = completeness.get("reasoning")
    
    coherence = dimensions.get("coherence_fluency", {}) # 键名可能与prompt中的一致
    details["coherence_score"] = coherence.get("score")
    details["coherence_reasoning"] = coherence.get("reasoning")

    actionability = dimensions.get("actionability_usability", {}) # 键名可能与prompt中的一致
    details["actionability_score"] = actionability.get("score")
    details["actionability_reasoning"] = actionability.get("reasoning")
    
    details["gemini_suggestion_answer"] = eval_data.get("suggestion_for_answer_improvement")

    return details

def perform_answer_evaluation_analysis(
    evaluation_log_filepath: str,
    output_csv_filepath: str
) -> bool:
    """
    加载答案评估日志，进行分析，并保存结果到CSV。

    Args:
        evaluation_log_filepath (str): 答案评估结果日志文件的路径。
        output_csv_filepath (str): 分析结果CSV文件的保存路径。

    Returns:
        bool: 如果分析和保存成功则返回True，否则返回False。
    """
    analyze_answer_logger.info(f"Starting Answer evaluation analysis for log file: {evaluation_log_filepath}")
    analyze_answer_logger.info(f"Analysis results will be saved to: {output_csv_filepath}")

    evaluation_logs = load_jsonl_file(evaluation_log_filepath)

    if not evaluation_logs:
        analyze_answer_logger.warning(f"No evaluation logs found or loaded from {evaluation_log_filepath}. Analysis aborted.")
        return False

    extracted_details_list = []
    for log_entry in evaluation_logs:
        if log_entry.get("task_type") == "answer_evaluation_result": # 确保是答案评估日志
            details = extract_answer_evaluation_details(log_entry)
            if details:
                extracted_details_list.append(details)
        else:
            analyze_answer_logger.debug(f"Skipping log entry with task_type '{log_entry.get('task_type')}' as it's not 'answer_evaluation_result'.")


    if not extracted_details_list:
        analyze_answer_logger.info("No valid Answer evaluation details extracted from the logs. No CSV will be generated.")
        return False

    df = pd.DataFrame(extracted_details_list)
    
    score_columns = [
        "overall_answer_quality_score", "faithfulness_score", "relevance_score",
        "completeness_score", "coherence_score", "actionability_score"
    ]
    for col in score_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    analyze_answer_logger.info(f"\n--- Preliminary Answer Evaluation Analysis (from {len(extracted_details_list)} entries) ---")
    analyze_answer_logger.info(f"Total evaluation entries processed: {len(df)}")

    for col_name in score_columns:
        if col_name in df.columns and not df[col_name].isnull().all():
            analyze_answer_logger.info(f"\nDimension: {col_name}")
            analyze_answer_logger.info(f"{df[col_name].describe()}")
            analyze_answer_logger.info("Score Distribution:")
            analyze_answer_logger.info(f"{df[col_name].value_counts(dropna=False).sort_index()}")
        else:
            analyze_answer_logger.info(f"\nDimension: {col_name} - No data or all NaN.")
            
    if "completeness_context_sufficiency" in df.columns and not df["completeness_context_sufficiency"].isnull().all():
        analyze_answer_logger.info("\nContext Sufficiency Assessment Distribution:")
        analyze_answer_logger.info(f"{df['completeness_context_sufficiency'].value_counts(dropna=False)}")
    else:
        analyze_answer_logger.info("\nContext Sufficiency Assessment Distribution: No data.")

    try:
        os.makedirs(os.path.dirname(output_csv_filepath), exist_ok=True)
        df.to_csv(output_csv_filepath, index=False, encoding='utf-8-sig')
        analyze_answer_logger.info(f"\nAnalysis results saved to: {output_csv_filepath}")
        return True
    except Exception as e:
        analyze_answer_logger.error(f"\nFailed to save CSV file: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    EVALUATION_NAME_FOR_ANSWER = "answer_gemini_flash" 
    
    # --- 动态查找最新的评估结果日志文件 ---
    eval_logs_pattern = os.path.join(EVALUATION_RESULTS_LOGS_DIR, f"eval_results_{EVALUATION_NAME_FOR_ANSWER}_*.jsonl")
    all_eval_logs = sorted(glob.glob(eval_logs_pattern), key=os.path.getmtime, reverse=True)
    
    log_file_path_answer: Optional[str] = None
    output_csv_path_answer: Optional[str] = None

    if all_eval_logs:
        log_file_path_answer = all_eval_logs[0] # 获取最新的一个
        analyze_answer_logger.info(f"Found latest Answer evaluation log for analysis: {log_file_path_answer}")
        
        # 根据找到的日志文件名构造输出的 CSV 文件名
        base_log_name = os.path.basename(log_file_path_answer)
        # 从 "eval_results_answer_gemini_flash_YYYYMMDD.jsonl" 生成 "analysis_answer_gemini_flash_YYYYMMDD.csv"
        if base_log_name.startswith(f"eval_results_{EVALUATION_NAME_FOR_ANSWER}_") and base_log_name.endswith(".jsonl"):
            date_part_from_filename = base_log_name[len(f"eval_results_{EVALUATION_NAME_FOR_ANSWER}_"):-len(".jsonl")]
            output_csv_name_answer = f"analysis_{EVALUATION_NAME_FOR_ANSWER}_{date_part_from_filename}.csv"
            output_csv_path_answer = os.path.join(EVALUATION_RESULTS_LOGS_DIR, output_csv_name_answer)
        else: # Fallback naming for CSV
            today_str = datetime.now().strftime("%Y%m%d")
            output_csv_name_answer = f"analysis_{EVALUATION_NAME_FOR_ANSWER}_{today_str}_fallback.csv"
            output_csv_path_answer = os.path.join(EVALUATION_RESULTS_LOGS_DIR, output_csv_name_answer)
        analyze_answer_logger.info(f"Analysis CSV report will be saved to: {output_csv_path_answer}")
    else:
        analyze_answer_logger.error(f"No Answer evaluation log files found matching pattern: {eval_logs_pattern}")

    if log_file_path_answer and output_csv_path_answer and os.path.exists(log_file_path_answer):
        perform_answer_evaluation_analysis(
            evaluation_log_filepath=log_file_path_answer,
            output_csv_filepath=output_csv_path_answer
        )
    else:
        analyze_answer_logger.info("Answer evaluation analysis will not run as no suitable log file was identified or output path could not be determined.")