# zhz_agent/refine_cypher_finetune_data.py
import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import glob
from datetime import datetime

# 假设 utils.py 和 constants.py 在同一个 zhz_agent 包内
try:
    from zhz_rag.utils.common_utils import get_interaction_log_filepath, get_evaluation_result_log_filepath, find_latest_rag_interaction_log # <--- 修改这里
    from zhz_rag.config.constants import NEW_KG_SCHEMA_DESCRIPTION
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
RAG_LOG_DIR = "zhz_rag/stored_data/rag_interaction_logs/"
# Gemini评估结果日志的目录
EVAL_LOG_DIR = "zhz_rag/stored_data/evaluation_results_logs/"
# 输出微调数据文件的目录
FINETUNE_DATA_DIR = "zhz_rag/finetuning/generated_data/"
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
你是顶级Neo4j Cypher查询生成专家。你的任务是根据用户问题和严格提供的【知识图谱Schema】，生成一个【语法正确】、【逻辑合理】且【高效】的Cypher查询。

**【核心指令与约束 - 必须严格遵守！】**

1.  **【Schema绝对绑定 - 最高优先级】**:
    *   你生成的Cypher查询中所有用到的【节点标签】、【关系类型】、【属性名称】及其对应的【数据类型】，都**必须严格存在于**下面提供的 "知识图谱Schema描述" 中。
    *   在构建查询的每一步，都要反复与Schema核对。**严禁臆断、猜测或使用任何Schema中未明确定义的元素。**
    *   **属性名称的大小写和确切拼写必须与Schema完全一致。**
    *   **关系类型的名称和方向必须与Schema完全一致。** 例如，如果Schema定义为 `(Person)-[:WORKS_ON]->(Project)`，则查询中不能是 `(Project)-[:WORKS_ON]->(Person)`，除非Schema中也定义了反向关系。

2.  **【纯净输出格式 - 严格要求】**:
    *   如果能生成有效查询，你的回答**必须只包含纯粹的Cypher查询语句本身**。
    *   如果根据问题和Schema无法生成有效的Cypher查询（例如，问题超出了Schema表达能力，或问题本身逻辑不通），则**必须只输出固定的短语：“无法生成Cypher查询。”**
    *   **绝对禁止**在有效的Cypher语句前后添加任何前缀（如“Cypher查询: ”）、后缀、解释、注释、markdown标记（如 ```cypher ```）或任何其他多余的文本。

3.  **【属性与值的使用】**:
    *   当在`WHERE`子句中对属性进行匹配时，确保值的类型与Schema中定义的属性类型一致。例如，如果`name`是字符串，则匹配 `name: '张三'`；如果`year`是数字，则匹配 `year: 2023`。
    *   对于数值计算（如`SUM`, `AVG`），**必须**使用Schema中明确指定的数字类型属性（例如，`SalesAmount`节点的 `numeric_amount`）。

4.  **【查询构建逻辑指引】**:
    *   **实体识别**: 准确识别用户问题中的核心实体及其在Schema中对应的节点标签和属性。
    *   **关系路径**: 基于问题和Schema构建清晰的`MATCH`路径。
    *   **条件过滤**: 使用`WHERE`子句添加必要的过滤条件。
    *   **结果返回**: 使用`RETURN`子句指定需要返回的信息，并用`AS`为返回的列指定清晰、合法的别名（字母或下划线开头）。
    *   **多步查询**: 对于需要关联多个信息点的问题，合理使用`WITH`传递中间结果。
    *   **聚合**: 如需统计或汇总，正确使用`COUNT()`, `SUM()`, `COLLECT()`等聚合函数。

**【知识图谱Schema描述】**:
{schema_description}

**【查询示例 - 严格基于上述Schema】**:

*   用户问题: "张三参与了哪个项目？"
    Cypher查询: MATCH (p:Person {{name: '张三'}})-[:WORKS_ON]->(proj:Project) RETURN proj.name AS projectName

*   用户问题: "华东区域2024年第一季度的销售额是多少？"
    Cypher查询: MATCH (r:Region {{name: '华东'}})-[:HAS_SALES_AMOUNT]->(sa:SalesAmount {{period: '2024年第一季度'}}) RETURN sa.numeric_amount AS salesAmount, sa.unit AS salesUnit

*   用户问题: "查询所有产品的名称。"
    Cypher查询: MATCH (prod:Product) RETURN prod.name AS productName

*   用户问题: "项目X有哪些人参与？"
    Cypher查询: MATCH (p:Person)-[:WORKS_ON]->(proj:Project {{name: '项目X'}}) RETURN p.name AS participantName

*   用户问题: "2024年第一季度所有区域的总销售额是多少？"
    Cypher查询: MATCH (r:Region)-[:HAS_SALES_AMOUNT]->(sa:SalesAmount {{period: '2024年第一季度'}}) RETURN sum(sa.numeric_amount) AS totalSales, sa.unit AS commonUnit LIMIT 1 
    (此查询假设所有相关销售额的单位是相同的，并取第一个出现的单位作为代表)

*   用户问题: "与新产品A相关的文档ID是什么？"
    Cypher查询: MATCH (p:Product {{name: '新产品A'}})-[:RELATED_TO]->(d:Document) RETURN d.id AS documentId

*   用户问题: "公司CEO是谁？" (假设Schema中没有CEO信息)
    Cypher查询: 无法生成Cypher查询。

现在，请根据以下用户问题和上述Schema及规则生成Cypher查询。
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

        user_question = rag_log.get("user_query")
        qwen_generated_cypher_raw = rag_log.get("processed_llm_output") # 这是Qwen原始输出

        # --- 改进点: 处理Qwen输出为空或仅包含空白的情况 ---
        if qwen_generated_cypher_raw is None or not qwen_generated_cypher_raw.strip():
            qwen_generated_cypher = "无法生成Cypher查询." # 将空输出也视为无法生成
            refine_logger.info(f"Interaction {interaction_id}: Qwen output was empty/None, treating as '无法生成Cypher查询.'.")
        else:
            qwen_generated_cypher = qwen_generated_cypher_raw.strip()


        qwen_input_prompt = rag_log.get("llm_input_prompt")
        if not qwen_input_prompt:
            if user_question:
                qwen_input_prompt = construct_qwen_input_prompt(user_question, NEW_KG_SCHEMA_DESCRIPTION)
            else:
                refine_logger.warning(f"Skipping Cypher log {interaction_id} due to missing user_question for prompt reconstruction.")
                continue
        
        if not user_question: # qwen_generated_cypher 已确保非None
            refine_logger.warning(f"Skipping Cypher log {interaction_id} due to missing user_question.")
            continue

        ideal_cypher_output = None
        source_of_ideal = "unknown"
        gemini_score_for_log = None # 用于记录

        eval_log = cypher_evaluation_logs.get(interaction_id)

        if eval_log and eval_log.get("eval_llm_processed_output_json"):
            eval_json = eval_log["eval_llm_processed_output_json"]
            overall_score_str = eval_json.get("evaluation_summary", {}).get("overall_quality_score_cypher")
            gemini_suggestion_raw = eval_json.get("suggestion_for_improvement_cypher", "").strip()
            
            try:
                overall_score = int(overall_score_str)
                gemini_score_for_log = overall_score
            except (ValueError, TypeError):
                refine_logger.warning(f"Could not parse overall_quality_score_cypher for {interaction_id}: {overall_score_str}")
                overall_score = 0 # 默认给个低分
                gemini_score_for_log = 0

            # --- 规则1: Qwen自己就说无法生成 ---
            if qwen_generated_cypher == "无法生成Cypher查询.":
                # 如果Gemini也认为无法生成或评分低，那么采纳
                if "无法生成Cypher查询" in gemini_suggestion_raw or overall_score <= 2:
                    ideal_cypher_output = "无法生成Cypher查询."
                    source_of_ideal = "qwen_and_gemini_cannot_generate"
                # 如果Qwen说无法生成，但Gemini给出了高分建议，这很奇怪，需要人工看
                elif overall_score >=4 and "MATCH" in gemini_suggestion_raw.upper():
                     refine_logger.info(f"Cypher log {interaction_id}: Qwen said '无法生成', but Gemini suggested a high-score query '{gemini_suggestion_raw}'. Needs manual review.")
                     continue
                else: # Qwen说无法生成，Gemini建议不明确或中低分，也采纳Qwen的
                    ideal_cypher_output = "无法生成Cypher查询."
                    source_of_ideal = "qwen_cannot_generate_gemini_unclear"


            # --- 规则2: Qwen生成了查询，看Gemini评估 ---
            elif overall_score >= 4: # Gemini认为Qwen的输出质量高
                ideal_cypher_output = qwen_generated_cypher
                source_of_ideal = "qwen_high_score_by_gemini"

            
            # --- 规则3: Qwen的查询质量不高，但Gemini给出了具体的、看起来像Cypher的建议 ---
            elif gemini_suggestion_raw and \
                "无法生成Cypher查询" not in gemini_suggestion_raw and \
                "cannot be improved" not in gemini_suggestion_raw.lower() and \
                "needs to be extended" not in gemini_suggestion_raw.lower() and \
                ("MATCH " in gemini_suggestion_raw.upper() or \
                    "RETURN " in gemini_suggestion_raw.upper() or \
                    "CREATE " in gemini_suggestion_raw.upper() or \
                    "MERGE " in gemini_suggestion_raw.upper() or \
                    "WITH " in gemini_suggestion_raw.upper() or \
                    "OPTIONAL MATCH " in gemini_suggestion_raw.upper()
                ):

                # 简化处理：直接将 Gemini 的原始建议作为 completion 的候选
                # 清洗工作主要交给人工审核阶段
                # 我们仍然可以做非常基础的清理，比如首尾空格和常见的 markdown
                
                temp_completion = gemini_suggestion_raw.strip()
                if temp_completion.startswith("```") and temp_completion.endswith("```"):
                    temp_completion = temp_completion[3:-3].strip()
                    if temp_completion.lower().startswith("cypher"):
                        temp_completion = temp_completion[len("cypher"):].strip()
                elif temp_completion.startswith("`") and temp_completion.endswith("`"):
                    temp_completion = temp_completion[1:-1].strip()

                # 只要建议中包含核心Cypher关键字，我们就认为它有价值被审核
                core_cypher_keywords_check = ["MATCH", "RETURN", "CREATE", "MERGE", "WITH", "OPTIONAL MATCH"]
                suggestion_contains_cypher_keyword = False
                if temp_completion:
                    for core_keyword in core_cypher_keywords_check:
                        if core_keyword in temp_completion.upper():
                            suggestion_contains_cypher_keyword = True
                            break
                
                if suggestion_contains_cypher_keyword:
                    ideal_cypher_output = temp_completion # 使用初步清理后的建议
                    source_of_ideal = "gemini_suggestion_for_review" # 明确标记为需要审核
                    refine_logger.info(f"Interaction {interaction_id}: Gemini suggestion adopted for review. Raw: '{gemini_suggestion_raw[:150]}...', Processed for completion: '{ideal_cypher_output[:150]}...'")
                else:
                    refine_logger.warning(f"Interaction {interaction_id}: Gemini suggestion '{gemini_suggestion_raw[:150]}...' did not appear to contain core Cypher keywords after basic cleaning. Skipping.")
                    continue

            
            # --- 规则4: Gemini明确建议“无法生成” 或 Qwen的查询质量低且有严重问题 ---
            elif "无法生成Cypher查询" in gemini_suggestion_raw or \
                 (overall_score <= 2 and ("hallucinated" in eval_log.get("eval_llm_raw_output", "").lower() or \
                                         "schema violation" in eval_log.get("eval_llm_raw_output", "").lower() or \
                                         "syntax error" in eval_log.get("eval_llm_raw_output", "").lower())):
                ideal_cypher_output = "无法生成Cypher查询."
                source_of_ideal = "gemini_explicitly_cannot_generate_or_qwen_low_quality"
            
            # --- 规则5: 其他情况，需要人工审核 ---
            else:
                refine_logger.info(f"Cypher log {interaction_id} (Qwen: '{qwen_generated_cypher[:100]}...') needs manual review. Gemini score: {overall_score}, Suggestion: '{gemini_suggestion_raw[:100]}...'")
                continue 
        
        # --- 如果没有Gemini评估日志 ---
        else:
            refine_logger.warning(f"No valid Gemini evaluation found for Cypher log {interaction_id}. Qwen's output: '{qwen_generated_cypher[:100]}...'. Skipping for finetune data.")
            continue
            
        if ideal_cypher_output is not None:
            finetune_samples.append({
                "prompt": qwen_input_prompt,
                "completion": ideal_cypher_output.strip(), # 确保completion也strip
                "original_qwen_cypher": qwen_generated_cypher,
                "gemini_score": gemini_score_for_log,
                "source_of_ideal": source_of_ideal,
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