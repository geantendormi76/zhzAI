import dagster as dg
import os
from typing import Dict, List, Any # Optional 可能之后会用到

# 从项目中导入我们重构的批量评估函数和相关工具/常量
from zhz_rag.evaluation.batch_eval_cypher import run_cypher_batch_evaluation
from zhz_rag.evaluation.batch_eval_answer import run_answer_batch_evaluation
from zhz_rag.evaluation.analyze_cypher import perform_cypher_evaluation_analysis
from zhz_rag.evaluation.analyze_answer import perform_answer_evaluation_analysis
from zhz_rag.utils.common_utils import (
    find_latest_rag_interaction_log,
    RAG_INTERACTION_LOGS_DIR,
    EVALUATION_RESULTS_LOGS_DIR,
    get_evaluation_result_log_filepath
)
# 导入 GeminiAPIResource 以声明资源依赖
from .resources import GeminiAPIResource # 假设 evaluation_assets.py 与 resources.py 在同一包内

# --- 资产定义 ---

@dg.asset(
    name="latest_rag_interaction_log_for_evaluation",
    description="Provides the filepath of the latest RAG interaction log to be used for evaluation.",
    group_name="evaluation_pipeline",
    compute_kind="python" # 可选，指明计算类型
)
def latest_rag_interaction_log_for_evaluation_asset(context: dg.AssetExecutionContext) -> str:
    """
    Finds and returns the path to the latest RAG interaction log file.
    """
    log_filepath = find_latest_rag_interaction_log(RAG_INTERACTION_LOGS_DIR)
    if not log_filepath or not os.path.exists(log_filepath):
        error_msg = f"No RAG interaction log file found in directory: {RAG_INTERACTION_LOGS_DIR}"
        context.log.error(error_msg)
        raise dg.Failure(description=error_msg)
    
    context.log.info(f"Using RAG interaction log for evaluation: {log_filepath}")
    context.add_output_metadata({"log_filepath": log_filepath, "filename": os.path.basename(log_filepath)})
    return log_filepath

@dg.asset(
    name="batch_cypher_evaluations_log", # 资产名称最好能反映它产出的是日志文件
    description="Runs batch evaluation of Cypher queries and produces an evaluation log file.",
    group_name="evaluation_pipeline",
    compute_kind="python",
    # deps=[latest_rag_interaction_log_for_evaluation_asset] # 通过函数参数自动推断依赖
)
async def batch_cypher_evaluation_log_asset(
    context: dg.AssetExecutionContext,
    gemini_api: GeminiAPIResource,
    latest_rag_interaction_log_for_evaluation: str # <--- 修改参数名
) -> dg.Output[str]:
    context.log.info(f"Starting batch Cypher evaluation using log file: {latest_rag_interaction_log_for_evaluation}") # <--- 使用新参数名
    
    # 从 Dagster 配置中获取参数，或使用默认/环境变量
    # 这里我们先用之前脚本中的方式，未来可以转为 Dagster run_config
    app_version = os.getenv("APP_VERSION_TAG", "dagster_cypher_eval_0.2")
    # 对于 use_simulated_api，在 Dagster 中通常会通过资源配置或 op_config 来控制，
    # 而不是直接依赖环境变量，这样更灵活。但为了保持与脚本一致，暂时保留。
    use_simulated = os.getenv("USE_SIMULATED_GEMINI_CYPHER_EVAL", "false").lower() == "true"
    api_delay = float(os.getenv("GEMINI_API_CALL_DELAY_SECONDS", "4.1"))

    if use_simulated:
        context.log.warning("Cypher evaluation asset is using SIMULATED Gemini API calls.")

    # 调用我们重构的、现在接受 gemini_resource 的批量评估函数
    eval_stats = await run_cypher_batch_evaluation(
        gemini_resource_for_evaluator=gemini_api, # 传递注入的 Dagster 资源
        rag_interaction_log_filepath=latest_rag_interaction_log_for_evaluation,
        app_version=app_version,
        use_simulated_api=use_simulated, # 这个参数现在由 run_cypher_batch_evaluation 内部处理
        api_call_delay=api_delay
    )
    context.log.info(f"Batch Cypher evaluation completed. Statistics: {eval_stats}")

    # 确定输出的评估结果日志文件名 (与 evaluator.py 中一致)
    output_log_filepath = get_evaluation_result_log_filepath(evaluation_name="cypher_gemini_flash")
    
    # 确保目录存在 (get_evaluation_result_log_filepath 内部的 log_interaction_data 会处理)
    # 但这里我们也可以提前确保，或者依赖 log_interaction_data
    os.makedirs(os.path.dirname(output_log_filepath), exist_ok=True)
            
    metadata = {"evaluation_stats": eval_stats, "output_filepath": output_log_filepath}
    if eval_stats.get("cypher_queries_evaluated", 0) == 0:
        metadata["warning"] = "No Cypher queries were evaluated. Output log might be empty."
        context.log.warning(metadata["warning"])

    return dg.Output(output_log_filepath, metadata=metadata)


@dg.asset(
    name="batch_answer_evaluations_log", # 资产名称
    description="Runs batch evaluation of generated answers from RAG logs using Gemini.",
    group_name="evaluation_pipeline",
    compute_kind="python",
    # deps=[latest_rag_interaction_log_for_evaluation_asset] # 通过函数参数自动推断依赖
)
async def batch_answer_evaluation_log_asset(
    context: dg.AssetExecutionContext,
    gemini_api: GeminiAPIResource,
    latest_rag_interaction_log_for_evaluation: str # <--- 修改参数名
) -> dg.Output[str]:
    context.log.info(f"Starting batch Answer evaluation using log file: {latest_rag_interaction_log_for_evaluation}") # <--- 使用新参数名
    app_version = os.getenv("APP_VERSION_TAG", "dagster_answer_eval_0.2")
    use_simulated = os.getenv("USE_SIMULATED_GEMINI_ANSWER_EVAL", "false").lower() == "true"
    api_delay = float(os.getenv("GEMINI_API_CALL_DELAY_SECONDS", "4.1"))

    if use_simulated:
        context.log.warning("Answer evaluation asset is using SIMULATED Gemini API calls.")

    eval_stats = await run_answer_batch_evaluation(
        gemini_resource_for_evaluator=gemini_api, # 传递注入的 Dagster 资源
        rag_interaction_log_filepath=latest_rag_interaction_log_for_evaluation,
        app_version=app_version,
        use_simulated_api=use_simulated, # 这个参数现在由 run_answer_batch_evaluation 内部处理
        api_call_delay=api_delay
    )
    context.log.info(f"Batch Answer evaluation completed. Statistics: {eval_stats}")

    output_log_filepath = get_evaluation_result_log_filepath(evaluation_name="answer_gemini_flash")
    os.makedirs(os.path.dirname(output_log_filepath), exist_ok=True)

    metadata = {"evaluation_stats": eval_stats, "output_filepath": output_log_filepath}
    if eval_stats.get("answers_evaluated", 0) == 0:
        metadata["warning"] = "No answers were evaluated. Output log might be empty."
        context.log.warning(metadata["warning"])
        
    return dg.Output(output_log_filepath, metadata=metadata)

@dg.asset(
    name="cypher_evaluation_analysis_report", # 资产名称
    description="Generates a CSV analysis report from Cypher evaluation results.",
    group_name="evaluation_pipeline",
    compute_kind="python",
    # deps=[batch_cypher_evaluation_log_asset] # 通过函数参数自动推断依赖
)
def cypher_analysis_report_asset(
    context: dg.AssetExecutionContext,
    batch_cypher_evaluations_log: str # 上游资产的输出 (即 cypher 评估日志文件的路径)
) -> dg.Output[str]: # 输出 CSV 报告文件的路径
    """
    Analyzes Cypher evaluation logs and produces a CSV report.
    """
    context.log.info(f"Starting Cypher evaluation analysis using log file: {batch_cypher_evaluations_log}")

    if not os.path.exists(batch_cypher_evaluations_log):
        error_msg = f"Input Cypher evaluation log file not found: {batch_cypher_evaluations_log}"
        context.log.error(error_msg)
        raise dg.Failure(description=error_msg)

    # 构建输出CSV文件的路径
    # 我们希望CSV文件也存储在 EVALUATION_RESULTS_LOGS_DIR 目录下
    # 文件名可以基于输入日志名或固定一个模式
    base_input_log_name = os.path.basename(batch_cypher_evaluations_log)
    # 从 "eval_results_cypher_gemini_flash_YYYYMMDD.jsonl" 生成 "analysis_cypher_gemini_flash_YYYYMMDD.csv"
    if base_input_log_name.startswith("eval_results_") and base_input_log_name.endswith(".jsonl"):
        analysis_file_name = "analysis_" + base_input_log_name[len("eval_results_"):-len(".jsonl")] + ".csv"
    else: # Fallback naming
        analysis_file_name = f"analysis_cypher_report_{context.run_id[:8]}.csv"
    
    output_csv_filepath = os.path.join(EVALUATION_RESULTS_LOGS_DIR, analysis_file_name)
    
    success = perform_cypher_evaluation_analysis(
        evaluation_log_filepath=batch_cypher_evaluations_log,
        output_csv_filepath=output_csv_filepath
    )

    if success:
        context.log.info(f"Cypher evaluation analysis report generated: {output_csv_filepath}")
        return dg.Output(output_csv_filepath, metadata={"output_csv_filepath": output_csv_filepath, "source_log": base_input_log_name})
    else:
        error_msg = f"Cypher evaluation analysis failed for log file: {batch_cypher_evaluations_log}"
        context.log.error(error_msg)
        raise dg.Failure(description=error_msg)


@dg.asset(
    name="answer_evaluation_analysis_report", # 资产名称
    description="Generates a CSV analysis report from Answer evaluation results.",
    group_name="evaluation_pipeline",
    compute_kind="python",
    # deps=[batch_answer_evaluations_log_asset] # 通过函数参数自动推断依赖
)
def answer_analysis_report_asset(
    context: dg.AssetExecutionContext,
    batch_answer_evaluations_log: str # 上游资产的输出 (即 answer 评估日志文件的路径)
) -> dg.Output[str]: # 输出 CSV 报告文件的路径
    """
    Analyzes Answer evaluation logs and produces a CSV report.
    """
    context.log.info(f"Starting Answer evaluation analysis using log file: {batch_answer_evaluations_log}")

    if not os.path.exists(batch_answer_evaluations_log):
        error_msg = f"Input Answer evaluation log file not found: {batch_answer_evaluations_log}"
        context.log.error(error_msg)
        raise dg.Failure(description=error_msg)

    base_input_log_name = os.path.basename(batch_answer_evaluations_log)
    if base_input_log_name.startswith("eval_results_") and base_input_log_name.endswith(".jsonl"):
        analysis_file_name = "analysis_" + base_input_log_name[len("eval_results_"):-len(".jsonl")] + ".csv"
    else: # Fallback naming
        analysis_file_name = f"analysis_answer_report_{context.run_id[:8]}.csv"
        
    output_csv_filepath = os.path.join(EVALUATION_RESULTS_LOGS_DIR, analysis_file_name)

    success = perform_answer_evaluation_analysis(
        evaluation_log_filepath=batch_answer_evaluations_log,
        output_csv_filepath=output_csv_filepath
    )

    if success:
        context.log.info(f"Answer evaluation analysis report generated: {output_csv_filepath}")
        return dg.Output(output_csv_filepath, metadata={"output_csv_filepath": output_csv_filepath, "source_log": base_input_log_name})
    else:
        error_msg = f"Answer evaluation analysis failed for log file: {batch_answer_evaluations_log}"
        context.log.error(error_msg)
        raise dg.Failure(description=error_msg)

# 将所有评估相关的资产收集到一个列表中，方便在 definitions.py 中引用
all_evaluation_assets = [
    latest_rag_interaction_log_for_evaluation_asset,
    batch_cypher_evaluation_log_asset,
    batch_answer_evaluation_log_asset,
    cypher_analysis_report_asset, # <--- 新增
    answer_analysis_report_asset, # <--- 新增
]