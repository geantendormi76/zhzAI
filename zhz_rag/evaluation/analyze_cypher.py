# zhz_rag/evaluation/analyze_cypher.py
import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional
from collections import Counter

# --- 从项目中导入必要的模块 ---
try:
    from zhz_rag.utils.common_utils import (
        load_jsonl_file, # <--- 使用新的通用函数
        EVALUATION_RESULTS_LOGS_DIR # 导入评估日志目录常量
    )
except ImportError as e:
    print(f"ERROR: Could not import necessary modules in analyze_cypher.py: {e}")
    print("Make sure this script is run in an environment where 'zhz_rag' package is accessible.")
    exit(1)

import logging

# --- 配置此脚本的logger ---
analyze_cypher_logger = logging.getLogger("AnalyzeCypherLogger")
analyze_cypher_logger.setLevel(logging.INFO)
if not analyze_cypher_logger.hasHandlers():
    _console_handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _console_handler.setFormatter(_formatter)
    analyze_cypher_logger.addHandler(_console_handler)
    analyze_cypher_logger.propagate = False
analyze_cypher_logger.info("--- AnalyzeCypherLogger configured ---")

# --- 核心功能函数 ---

def extract_cypher_evaluation_details(log_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    从单条已解析的Cypher评估日志条目中提取关键信息。
    这个函数与您之前在 analyze_cypher.py 中的版本基本一致，稍作调整以适应通用加载。
    """
    details = {}
    # eval_llm_processed_output_json 字段包含了Gemini评估的JSON输出
    eval_data = log_entry.get("eval_llm_processed_output_json")

    if not eval_data or not isinstance(eval_data, dict):
        analyze_cypher_logger.warning(f"Skipping log entry due to missing or invalid 'eval_llm_processed_output_json'. Interaction ID ref: {log_entry.get('original_interaction_id_ref')}")
        return None # 如果核心评估数据缺失，则跳过此条目

    details["interaction_id_ref"] = log_entry.get("original_interaction_id_ref")
    details["user_question"] = log_entry.get("user_question_for_eval")
    details["generated_cypher"] = log_entry.get("generated_cypher_for_eval")
    
    summary = eval_data.get("evaluation_summary", {})
    dimensions = eval_data.get("dimensions", {})
    
    details["overall_quality_score"] = summary.get("overall_quality_score_cypher")
    details["main_strength"] = summary.get("main_strength_cypher")
    details["main_weakness"] = summary.get("main_weakness_cypher")
    
    syntax = dimensions.get("syntactic_correctness", {})
    details["syntax_score"] = syntax.get("score")
    details["syntax_reasoning"] = syntax.get("reasoning")
    
    schema = dimensions.get("schema_adherence", {})
    details["schema_overall_score"] = schema.get("overall_score")
    details["schema_node_label_correct"] = schema.get("node_label_correctness", {}).get("check_result")
    details["schema_entity_type_correct"] = schema.get("entity_type_property_correctness", {}).get("check_result")
    details["schema_rel_type_correct"] = schema.get("relationship_type_correctness", {}).get("check_result")
    details["schema_prop_name_correct"] = schema.get("property_name_correctness", {}).get("check_result")
    details["schema_hallucinated_present"] = schema.get("hallucinated_schema_elements", {}).get("check_result_hallucination_present")
    details["schema_hallucinated_elements"] = ", ".join(schema.get("hallucinated_schema_elements", {}).get("elements_found", []))
    details["schema_reasoning"] = schema.get("reasoning")
    
    intent = dimensions.get("intent_accuracy", {})
    details["intent_score"] = intent.get("score")
    details["intent_explanation_cypher"] = intent.get("explanation_of_cypher_retrieval")
    details["intent_alignment_notes"] = intent.get("semantic_alignment_with_question")
    details["intent_key_elements_notes"] = intent.get("key_element_coverage_notes")
    details["intent_reasoning"] = intent.get("reasoning")
    
    details["qwen_error_patterns"] = ", ".join(eval_data.get("qwen_error_patterns_identified", []))
    details["gemini_suggestion"] = eval_data.get("suggestion_for_improvement_cypher")

    return details

def perform_cypher_evaluation_analysis(
    evaluation_log_filepath: str,
    output_csv_filepath: str
) -> bool:
    """
    加载Cypher评估日志，进行分析，并保存结果到CSV。

    Args:
        evaluation_log_filepath (str): Cypher评估结果日志文件的路径。
        output_csv_filepath (str): 分析结果CSV文件的保存路径。

    Returns:
        bool: 如果分析和保存成功则返回True，否则返回False。
    """
    analyze_cypher_logger.info(f"Starting Cypher evaluation analysis for log file: {evaluation_log_filepath}")
    analyze_cypher_logger.info(f"Analysis results will be saved to: {output_csv_filepath}")

    # 使用通用函数加载评估日志
    # 注意：evaluate_cypher_with_gemini 保存的日志中 task_type 是 "cypher_evaluation_by_gemini"
    # load_jsonl_file 不关心 task_type，它会加载所有行
    evaluation_logs = load_jsonl_file(evaluation_log_filepath)

    if not evaluation_logs:
        analyze_cypher_logger.warning(f"No evaluation logs found or loaded from {evaluation_log_filepath}. Analysis aborted.")
        return False

    extracted_details_list = []
    for log_entry in evaluation_logs:
        # 确保只处理Cypher评估结果
        if log_entry.get("task_type") == "cypher_evaluation_by_gemini":
            details = extract_cypher_evaluation_details(log_entry)
            if details: # extract_cypher_evaluation_details 可能会返回 None
                extracted_details_list.append(details)
        else:
            analyze_cypher_logger.debug(f"Skipping log entry with task_type '{log_entry.get('task_type')}' as it's not 'cypher_evaluation_by_gemini'.")


    if not extracted_details_list:
        analyze_cypher_logger.info("No valid Cypher evaluation details extracted from the logs. No CSV will be generated.")
        return False

    df = pd.DataFrame(extracted_details_list)
    
    score_columns = ["overall_quality_score", "syntax_score", "schema_overall_score", "intent_score"]
    for col in score_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    analyze_cypher_logger.info(f"\n--- Preliminary Cypher Evaluation Analysis (from {len(extracted_details_list)} entries) ---")
    analyze_cypher_logger.info(f"Total evaluation entries processed: {len(df)}")

    if "overall_quality_score" in df.columns and not df["overall_quality_score"].isnull().all():
        analyze_cypher_logger.info("\n1. Overall Quality Score:")
        analyze_cypher_logger.info(f"{df['overall_quality_score'].describe()}")
        analyze_cypher_logger.info("\nScore Distribution:")
        analyze_cypher_logger.info(f"{df['overall_quality_score'].value_counts(dropna=False).sort_index()}")
    else:
        analyze_cypher_logger.info("\n1. Overall Quality Score: No data or all NaN.")


    if "schema_overall_score" in df.columns and not df["schema_overall_score"].isnull().all():
        analyze_cypher_logger.info("\n2. Schema Adherence Overall Score:")
        analyze_cypher_logger.info(f"{df['schema_overall_score'].describe()}")
        analyze_cypher_logger.info("\nScore Distribution:")
        analyze_cypher_logger.info(f"{df['schema_overall_score'].value_counts(dropna=False).sort_index()}")
        
        schema_sub_checks = [
            "schema_node_label_correct", "schema_entity_type_correct", 
            "schema_rel_type_correct", "schema_prop_name_correct", 
            "schema_hallucinated_present"
        ]
        analyze_cypher_logger.info("\nSchema Adherence Sub-item Issues (False means issue, Hallucinated True means issue):")
        for check in schema_sub_checks:
            if check in df.columns:
                if check == "schema_hallucinated_present":
                    issue_count = df[df[check] == True].shape[0]
                    analyze_cypher_logger.info(f"  - {check} (Hallucination Present): {issue_count} entries")
                else:
                    issue_count = df[df[check] == False].shape[0]
                    analyze_cypher_logger.info(f"  - {check} (Incorrect): {issue_count} entries")
    else:
        analyze_cypher_logger.info("\n2. Schema Adherence Overall Score: No data or all NaN.")


    if "intent_score" in df.columns and not df["intent_score"].isnull().all():
        analyze_cypher_logger.info("\n3. Intent Accuracy Score:")
        analyze_cypher_logger.info(f"{df['intent_score'].describe()}")
        analyze_cypher_logger.info("\nScore Distribution:")
        analyze_cypher_logger.info(f"{df['intent_score'].value_counts(dropna=False).sort_index()}")
    else:
        analyze_cypher_logger.info("\n3. Intent Accuracy Score: No data or all NaN.")


    if "qwen_error_patterns" in df.columns and not df["qwen_error_patterns"].isnull().all():
        analyze_cypher_logger.info("\n4. Identified Qwen Error Patterns (Top 5):")
        all_patterns = []
        for pattern_list_str in df["qwen_error_patterns"].dropna():
            if pattern_list_str and isinstance(pattern_list_str, str):
                all_patterns.extend([p.strip() for p in pattern_list_str.split(",") if p.strip()])
        pattern_counts = Counter(all_patterns)
        analyze_cypher_logger.info(f"{pattern_counts.most_common(5)}")
    else:
        analyze_cypher_logger.info("\n4. Identified Qwen Error Patterns: No data.")
        
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_csv_filepath), exist_ok=True)
        df.to_csv(output_csv_filepath, index=False, encoding='utf-8-sig')
        analyze_cypher_logger.info(f"\nAnalysis results saved to: {output_csv_filepath}")
        return True
    except Exception as e:
        analyze_cypher_logger.error(f"\nFailed to save CSV file: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # --- Configuration for running the analysis ---
    EVALUATION_NAME_FOR_CYPHER = "cypher_gemini_flash" # Must match the name used in batch_eval_cypher.py / evaluator.py

    # Try to find the latest log file for the given evaluation name
    # This requires a naming convention for eval logs, e.g., eval_results_<name>_<date>.jsonl
    # For simplicity, we'll construct a potential filename based on today's date,
    # but in a real scenario, you might want a more robust way to find the latest.
    
    # For demonstration, let's try to find a log from a recent specific date if today's doesn't exist.
    # You might need to adjust this logic or manually specify the file.
    
    # We will try to use the file name you provided in the previous run's log:
    # eval_results_cypher_gemini_flash_20250530.jsonl
    # This assumes the "date" part of the filename is consistent.
    
    # Let's try to find the log file used in the last successful Cypher evaluation run (20250530)
    # This is a placeholder, in a real system you might have a more robust way to get this.
    target_date_str = "20250530" # From your previous successful run log
    
    log_file_name_cypher = f"eval_results_{EVALUATION_NAME_FOR_CYPHER}_{target_date_str}.jsonl"
    log_file_path_cypher = os.path.join(EVALUATION_RESULTS_LOGS_DIR, log_file_name_cypher)

    output_csv_name_cypher = f"analysis_{EVALUATION_NAME_FOR_CYPHER}_{target_date_str}.csv"
    output_csv_path_cypher = os.path.join(EVALUATION_RESULTS_LOGS_DIR, output_csv_name_cypher)

    if not os.path.exists(log_file_path_cypher):
        analyze_cypher_logger.error(f"Cypher evaluation log file for analysis not found: {log_file_path_cypher}")
        analyze_cypher_logger.info("Please ensure the filename and date match an existing evaluation log.")
        # As a fallback, you could try finding the *absolute* latest eval log if the dated one isn't found,
        # but that might not always be what you want.
        # Example:
        # eval_logs_pattern = os.path.join(EVALUATION_RESULTS_LOGS_DIR, f"eval_results_{EVALUATION_NAME_FOR_CYPHER}_*.jsonl")
        # all_eval_logs = sorted(glob.glob(eval_logs_pattern), key=os.path.getmtime, reverse=True)
        # if all_eval_logs:
        #     log_file_path_cypher = all_eval_logs[0]
        #     date_part_from_filename = os.path.basename(log_file_path_cypher).split('_')[-1].split('.')[0]
        #     output_csv_path_cypher = os.path.join(EVALUATION_RESULTS_LOGS_DIR, f"analysis_{EVALUATION_NAME_FOR_CYPHER}_{date_part_from_filename}.csv")
        #     analyze_cypher_logger.info(f"Falling back to latest available Cypher eval log: {log_file_path_cypher}")
        # else:
        #     analyze_cypher_logger.error(f"No Cypher evaluation logs found matching pattern: {eval_logs_pattern}")
        #     log_file_path_cypher = None # Ensure it's None if no fallback
    
    if log_file_path_cypher and os.path.exists(log_file_path_cypher) : # Check again if fallback was attempted
        perform_cypher_evaluation_analysis(
            evaluation_log_filepath=log_file_path_cypher,
            output_csv_filepath=output_csv_path_cypher
        )
    else:
        analyze_cypher_logger.info("Cypher evaluation analysis will not run as no suitable log file was identified.")