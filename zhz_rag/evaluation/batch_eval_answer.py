# 文件: zhz_rag/evaluation/batch_eval_answer.py
import os
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import argparse
import httpx # 使用httpx进行异步HTTP请求

# 确保项目根目录在sys.path中，以便正确导入
# 这部分可能需要根据你的项目结构调整
try:
    from zhz_rag.llm.llm_interface import NO_ANSWER_PHRASE_ANSWER_CLEAN
    from zhz_rag.config.pydantic_models import RetrievedDocument
    from zhz_rag.utils.gemini_api_utils import GeminiAPIClient # <--- 修改
    from zhz_rag.evaluation.evaluator import evaluate_answer_with_gemini
except ImportError:
    import sys
    # A more robust way to add the project root to the path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from zhz_rag.llm.llm_interface import NO_ANSWER_PHRASE_ANSWER_CLEAN
    from zhz_rag.config.pydantic_models import RetrievedDocument
    from zhz_rag.utils.gemini_api_utils import GeminiAPIClient # <--- 修正这一行
    from zhz_rag.evaluation.evaluator import evaluate_answer_with_gemini

# --- 配置日志 ---
batch_answer_eval_logger = logging.getLogger("BatchAnswerEvaluationLogger")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- 定义常量 ---
RAG_API_URL = "http://localhost:8081/api/v1/rag/query"
DEFAULT_TEST_DATASET_PATH = Path(__file__).parent / "test_datasets" / "evaluation_questions_v1.txt"
REQUEST_TIMEOUT = 120.0  # API请求的超时时间，单位秒
DELAY_BETWEEN_REQUESTS = 5 # 每个评估请求之间的延迟，避免API过载

async def call_rag_api(client: httpx.AsyncClient, question: str) -> Optional[Dict[str, Any]]:
    """
    异步调用RAG API服务并返回结果。
    """
    payload = {
        "query": question,
        "top_k_vector": 3,
        "top_k_bm25": 3,
        "top_k_kg": 3,
        "top_k_final": 5  # 召回更多上下文给评估器
    }
    try:
        batch_answer_eval_logger.info(f"Sending request to RAG API for question: '{question}'")
        response = await client.post(RAG_API_URL, json=payload, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            batch_answer_eval_logger.info("RAG API returned a successful response.")
            return response.json()
        else:
            batch_answer_eval_logger.error(
                f"RAG API returned an error. Status: {response.status_code}, "
                f"Response: {response.text[:200]}"
            )
            return None
    except httpx.RequestError as e:
        batch_answer_eval_logger.error(f"Error calling RAG API for question '{question}': {e}", exc_info=True)
        return None

async def main_evaluation_runner(
    questions_file: Path, 
    gemini_resource: GeminiAPIClient, 
    app_version_tag: str,
    use_simulated_api: bool,
    api_call_delay: int
):
    """
    读取问题文件，调用RAG API，然后使用Gemini评估答案。
    """
    if not questions_file.exists():
        batch_answer_eval_logger.error(f"Test questions file not found at: {questions_file}")
        return

    with open(questions_file, 'r', encoding='utf-8') as f:
        # --- 优化：跳过注释行 (#) 和空行 ---
        questions = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    batch_answer_eval_logger.info(f"Loaded {len(questions)} questions from '{questions_file.name}'. Starting evaluation...")
    
    successful_evals = 0
    failed_evals = 0
    
    async with httpx.AsyncClient() as client:
        for i, question in enumerate(questions, 1):
            batch_answer_eval_logger.info(f"--- Processing question {i}/{len(questions)}: '{question}' ---")
            
            # 1. 调用RAG API获取答案
            rag_response = await call_rag_api(client, question)
            
            if rag_response is None:
                batch_answer_eval_logger.warning(f"Skipping evaluation for '{question}' due to RAG API failure.")
                failed_evals += 1
                await asyncio.sleep(api_call_delay) # 使用传入的延迟参数
                continue

            # 2. 从RAG API响应中提取评估所需的信息
            answer_text = rag_response.get("answer")
            retrieved_sources_raw = rag_response.get("retrieved_sources", [])
            
            # 即使有答案，如果没有上下文来源，对于研究型问题也认为是失败
            if not answer_text or not retrieved_sources_raw:
                batch_answer_eval_logger.warning(
                    f"RAG API response for '{question}' is incomplete. "
                    f"Answer: '{answer_text}', Sources: {len(retrieved_sources_raw)}. Skipping evaluation."
                )
                failed_evals += 1
                await asyncio.sleep(api_call_delay) # 使用传入的延迟参数
                continue

            # 3. 准备评估函数的输入
            retrieved_docs_for_eval = [RetrievedDocument(**doc) for doc in retrieved_sources_raw]
            
            # 4. 调用Gemini进行评估
            await evaluate_answer_with_gemini(
                gemini_resource_for_evaluator=gemini_resource,
                user_question=question,
                retrieved_contexts=retrieved_docs_for_eval,
                generated_answer=answer_text,
                app_version=app_version_tag,
                use_simulated_api=use_simulated_api,
                api_call_delay=api_call_delay
            )
            successful_evals += 1
            
            batch_answer_eval_logger.info(f"Successfully evaluated question {i}. Sleeping for {api_call_delay} seconds...")
            await asyncio.sleep(api_call_delay) # 使用传入的延迟参数

    batch_answer_eval_logger.info("--- Batch Answer Evaluation Finished ---")
    batch_answer_eval_logger.info(f"Summary: Successfully evaluated {successful_evals} questions, Failed/Skipped {failed_evals} questions.")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch evaluation of RAG answers using a questions file.")
    
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=DEFAULT_TEST_DATASET_PATH,
        help=f"Path to the .txt file containing evaluation questions. Defaults to {DEFAULT_TEST_DATASET_PATH}"
    )
    parser.add_argument(
        "--app-version",
        type=str,
        default="0.2.0_eval_run",
        help="A version tag for this evaluation run."
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use simulated Gemini API responses for testing the evaluation script itself."
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=DELAY_BETWEEN_REQUESTS,
        help=f"Delay in seconds between evaluation calls to avoid rate limiting. Defaults to {DELAY_BETWEEN_REQUESTS}."
    )
    
    args = parser.parse_args()
    
# 初始化Gemini客户端
    try:
        gemini_client = GeminiAPIClient.from_env() # <--- 修改
        batch_answer_eval_logger.info("GeminiAPIClient for Answer evaluation initialized successfully.")
    except Exception as e:
        batch_answer_eval_logger.error(f"Failed to initialize GeminiAPIClient: {e}", exc_info=True)
        gemini_client = None

    if not gemini_client:
        batch_answer_eval_logger.error("Cannot proceed with evaluation as GeminiAPIClient is not available.")
    else:
        asyncio.run(main_evaluation_runner(
            questions_file=args.questions_file,
            gemini_resource=gemini_client, # <--- 修改
            app_version_tag=args.app_version,
            use_simulated_api=args.simulate,
            api_call_delay=args.delay
        ))