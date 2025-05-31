# analyze_cypher_evaluations.py
import json
import os
import pandas as pd
from typing import List, Dict, Any
from collections import Counter

# --- 配置 ---
# 假设您的日志文件与此脚本在同一目录下或可以通过相对/绝对路径访问
# 您需要将此路径替换为实际的 .jsonl 文件路径
LOG_FILE_DIR = "zhz_agent/rag_eval_data/" # 日志文件所在目录
# 默认分析当天的日志，您可以修改为特定的文件名
# from datetime import datetime, timezone
# today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
# LOG_FILE_NAME = f"rag_interactions_{today_str}.jsonl"
EVALUATION_NAME = "cypher_gemini_flash" # <<<--- 确保这个与 evaluation.py 中的一致
# from datetime import datetime, timezone
# today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
# LOG_FILE_NAME = f"eval_results_{EVALUATION_NAME}_{today_str}.jsonl"
LOG_FILE_NAME = f"eval_results_{EVALUATION_NAME}_20250530.jsonl" # <<<--- 请修改为您实际评估结果的日志文件名 (注意日期)

LOG_FILE_PATH = os.path.join(LOG_FILE_DIR, LOG_FILE_NAME)

OUTPUT_CSV_FILE = os.path.join(LOG_FILE_DIR, f"analysis_{EVALUATION_NAME}_{os.path.splitext(LOG_FILE_NAME)[0].split('_')[-1]}.csv") # <<<--- 新的定义

def load_evaluation_logs(filepath: str) -> List[Dict[str, Any]]:
    """从JSONL文件中加载所有评估结果日志。"""
    eval_logs = []
    if not os.path.exists(filepath):
        print(f"错误：评估结果日志文件未找到: {filepath}")
        return eval_logs
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            try:
                log_entry = json.loads(line.strip())
                # 现在我们直接读取评估结果文件，所以不需要再按 task_type 筛选了
                # 只需要确保 eval_llm_processed_output_json 存在
                if log_entry.get("eval_llm_processed_output_json"):
                    eval_logs.append(log_entry)
            except json.JSONDecodeError:
                print(f"警告：跳过格式错误的JSON行 {line_number} 在文件 {filepath}")
            except Exception as e:
                print(f"警告：处理行 {line_number} 时发生错误: {e}")
    
    print(f"从 {filepath} 加载了 {len(eval_logs)} 条 'cypher_evaluation_by_gemini' 日志。")
    return eval_logs

def extract_evaluation_details(log_entry: Dict[str, Any]) -> Dict[str, Any]:
    """从单条评估日志中提取关键信息。"""
    details = {}
    eval_data = log_entry.get("eval_llm_processed_output_json", {})
    
    details["interaction_id_ref"] = log_entry.get("original_interaction_id_ref")
    details["user_question"] = log_entry.get("user_question_for_eval")
    details["generated_cypher"] = log_entry.get("generated_cypher_for_eval")
    
    if eval_data and isinstance(eval_data, dict): # 确保 eval_data 是字典
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
    else:
        print(f"警告: interaction_id_ref {details.get('interaction_id_ref')} 的 eval_llm_processed_output_json 为空或格式不正确。")

    return details

def analyze_evaluations(eval_data_list: List[Dict[str, Any]]):
    """对提取的评估数据进行初步分析。"""
    if not eval_data_list:
        print("没有评估数据可供分析。")
        return

    df = pd.DataFrame(eval_data_list)
    
    # 将评分列转换为数值类型，错误时设为NaN
    score_columns = ["overall_quality_score", "syntax_score", "schema_overall_score", "intent_score"]
    for col in score_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("\n--- 初步评估分析 ---")
    print(f"总评估条目数: {len(df)}")

    if "overall_quality_score" in df.columns:
        print("\n1. 整体质量评分 (Overall Quality Score):")
        print(df["overall_quality_score"].describe())
        print("\n评分分布:")
        print(df["overall_quality_score"].value_counts(dropna=False).sort_index())

    if "schema_overall_score" in df.columns:
        print("\n2. Schema遵循度总体评分 (Schema Adherence Overall Score):")
        print(df["schema_overall_score"].describe())
        print("\n评分分布:")
        print(df["schema_overall_score"].value_counts(dropna=False).sort_index())
        
        # Schema子项统计 (只统计False的情况，即有问题的)
        schema_sub_checks = [
            "schema_node_label_correct", "schema_entity_type_correct", 
            "schema_rel_type_correct", "schema_prop_name_correct", 
            "schema_hallucinated_present" # True 表示有问题
        ]
        print("\nSchema遵循度子项问题统计 (False表示通过, Hallucinated True表示有问题):")
        for check in schema_sub_checks:
            if check in df.columns:
                if check == "schema_hallucinated_present": # 这个是True代表有问题
                    issue_count = df[df[check] == True].shape[0]
                    print(f"  - {check} (存在幻觉): {issue_count} 条")
                else: # 其他的是False代表有问题
                    issue_count = df[df[check] == False].shape[0]
                    print(f"  - {check} (未通过): {issue_count} 条")


    if "intent_score" in df.columns:
        print("\n3. 意图准确性评分 (Intent Accuracy Score):")
        print(df["intent_score"].describe())
        print("\n评分分布:")
        print(df["intent_score"].value_counts(dropna=False).sort_index())

    if "qwen_error_patterns" in df.columns:
        print("\n4. 识别出的Qwen错误模式 (Top 5):")
        all_patterns = []
        for pattern_list_str in df["qwen_error_patterns"].dropna():
            if pattern_list_str: # 确保不是空字符串
                all_patterns.extend([p.strip() for p in pattern_list_str.split(",")])
        pattern_counts = Counter(all_patterns)
        print(pattern_counts.most_common(5))
        
    # 保存到CSV
    try:
        df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
        print(f"\n分析结果已保存到: {OUTPUT_CSV_FILE}")
    except Exception as e:
        print(f"\n保存CSV文件失败: {e}")

if __name__ == "__main__":
    print(f"正在从评估结果日志文件加载数据: {LOG_FILE_PATH}")
    evaluation_logs = load_evaluation_logs(LOG_FILE_PATH)
    
    if evaluation_logs:
        extracted_details_list = []
        for log in evaluation_logs:
            details = extract_evaluation_details(log)
            extracted_details_list.append(details)
        
        analyze_evaluations(extracted_details_list)
    else:
        print("未能加载任何评估日志，分析中止。")