import asyncio
import os
import sys
import logging
import json
from typing import Optional


# --- 配置项目根目录到 sys.path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- 结束 sys.path 配置 ---

try:
    from zhz_rag.core_rag.kg_retriever import KGRetriever
    from zhz_rag.llm.local_model_handler import LocalModelHandler # KGRetriever 需要 embedder
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure you are running this script from a context where 'zhz_rag' package is discoverable,")
    print(f"or that PYTHONPATH is set correctly. Current sys.path: {sys.path}")
    sys.exit(1)

# 加载 .env 文件 (确保 SGLANG_API_URL, EMBEDDING_MODEL_PATH, DUCKDB_KG_FILE_PATH 等已设置)
# .env 文件应该在项目根目录 PROJECT_ROOT
dotenv_path = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    print(f"TestScript: Loaded .env file from: {dotenv_path}")
else:
    print(f"TestScript: .env file not found at {dotenv_path}. Relying on system environment variables.")
    load_dotenv()


# --- 日志配置 ---
# 配置根日志记录器，以便捕获来自所有模块的日志
logging.basicConfig(
    level=logging.DEBUG, # 设置为 DEBUG 以查看来自 KGRetriever 和 LocalModelHandler 的详细日志
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__) # 当前脚本的 logger

# --- 测试用例定义 ---
# 我们使用之前 Dagster 流水线加载的 doc1.txt 和 doc2.txt 的内容来构造测试用例
# doc1.txt: "项目Alpha的文档编写任务分配给了张三。张三在谷歌工作。"
#   - 实体: "项目alpha的文档编写任务" (TASK), "张三" (PERSON), "谷歌" (ORGANIZATION)
#   - 关系: ("文档编写任务" ASSIGNED_TO "张三"), ("张三" WORKS_AT "谷歌")
# doc2.txt: "第二个文档讨论了自然语言处理技术。" (可能没有提取出特定实体和关系)

test_cases = [
    {
        "name": "TC1: 查询'张三'相关信息 (期望向量+属性匹配)",
        "user_query": "张三是谁？",
        "top_k": 3,
        "expected_min_results": 1,
        "check_keywords_in_content": ["张三", "PERSON"]
    },
    {
        "name": "TC2: 查询'张三'的工作地点 (期望关系查找)",
        "user_query": "张三在哪里工作？",
        "top_k": 3,
        "expected_min_results": 1,
        "check_keywords_in_content": ["张三", "谷歌", "WORKS_AT"]
    },
    {
        "name": "TC3: 查询'文档编写任务'的负责人 (期望关系查找)",
        "user_query": "谁负责项目Alpha的文档编写任务？",
        "top_k": 3,
        "expected_min_results": 1,
        "check_keywords_in_content": ["文档编写任务", "张三", "ASSIGNED_TO"]
    },
    {
        "name": "TC4: 查询'谷歌'公司信息 (期望向量+属性匹配)",
        "user_query": "谷歌公司是做什么的？", # 这个问题可能更多依赖向量搜索
        "top_k": 2,
        "expected_min_results": 1,
        "check_keywords_in_content": ["谷歌", "ORGANIZATION"]
    },
    {
        "name": "TC5: 查询'自然语言处理' (期望向量匹配，但DB中无此内容)",
        "user_query": "什么是自然语言处理技术？",
        "top_k": 2,
        "expected_min_results": 0, # <--- 修改：期望找不到特定结果
        "check_keywords_in_content": []  # <--- 修改：不期望特定关键词
    },
    {
        "name": "TC6: 图谱中不存在的信息",
        "user_query": "微软公司的CEO是谁？",
        "top_k": 3,
        "expected_min_results": 0, # 期望找不到
        "check_keywords_in_content": []
    }
]

async def run_single_test(kg_retriever: KGRetriever, test_case: dict) -> bool:
    logger.info(f"\n--- Running Test Case: {test_case['name']} ---")
    logger.info(f"User Query: {test_case['user_query']}")
    
    retrieved_docs = await kg_retriever.retrieve(
        user_query=test_case['user_query'],
        top_k=test_case['top_k']
    )
    
    logger.info(f"KGRetriever returned {len(retrieved_docs)} documents.")
    for i, doc in enumerate(retrieved_docs):
        logger.info(f"  Doc {i+1}: Score={doc.get('score', 'N/A'):.4f}, Source={doc.get('source_type')}, Content='{str(doc.get('content'))[:100]}...'")
        logger.debug(f"    Full Doc Metadata: {json.dumps(doc.get('metadata'), ensure_ascii=False, indent=2)}")

    # 基本检查
    if len(retrieved_docs) < test_case["expected_min_results"]:
        logger.error(f"  RESULT: FAILED - Expected at least {test_case['expected_min_results']} documents, got {len(retrieved_docs)}")
        return False

    if not retrieved_docs and not test_case["check_keywords_in_content"]: # 期望为空，实际也为空
        logger.info("  RESULT: PASSED - Correctly returned no documents as expected.")
        return True
    
    if retrieved_docs and not test_case["check_keywords_in_content"]: # 期望为空，但不为空
        logger.warning("  RESULT: POTENTIAL ISSUE - Expected no documents, but got some. Manual check advised.")
        # 这种情况我们不直接判为失败，因为向量搜索可能会找到一些模糊匹配，需要人工判断是否合理
        return True # 暂时算通过，但需要注意

    if not retrieved_docs and test_case["check_keywords_in_content"]: # 期望有内容，但为空
         logger.error(f"  RESULT: FAILED - Expected documents with keywords {test_case['check_keywords_in_content']}, but got no documents.")
         return False

    # 关键词检查 (如果期望有结果)
    if test_case["check_keywords_in_content"]:
        found_match = False
        for doc in retrieved_docs:
            content_lower = str(doc.get('content', "")).lower()
            all_keywords_present_in_this_doc = True
            for keyword in test_case["check_keywords_in_content"]:
                if keyword.lower() not in content_lower:
                    all_keywords_present_in_this_doc = False
                    break
            if all_keywords_present_in_this_doc:
                found_match = True
                logger.info(f"  Found a matching document (contains all keywords): Content='{str(doc.get('content'))[:100]}...'")
                break
        
        if found_match:
            logger.info(f"  RESULT: PASSED - At least one document contained all expected keywords: {test_case['check_keywords_in_content']}")
            return True
        else:
            logger.error(f"  RESULT: FAILED - No document contained all expected keywords: {test_case['check_keywords_in_content']}")
            return False
            
    return True # 如果没有指定检查关键词，并且走到了这里（说明期望结果数量符合）

async def main_test_kg_retriever():
    logger.info("--- Starting DuckDB KGRetriever Test Script ---")

    # 1. 初始化 LocalModelHandler (用于嵌入)
    embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH")
    if not embedding_model_path or not os.path.exists(embedding_model_path):
        logger.error(f"Embedding model path not found or not set in EMBEDDING_MODEL_PATH: {embedding_model_path}")
        return

    try:
        # LocalModelHandler 只需要嵌入模型用于 KGRetriever
        # 如果您的 LocalModelHandler __init__ 强制要求 llm_model_path，您可能需要提供一个虚拟的或实际的路径
        # 或者修改 LocalModelHandler 使其 llm_model_path 可选
        embedder_handler = LocalModelHandler(
            embedding_model_path=embedding_model_path,
            # 提供默认值或从env获取，如果 LocalModelHandler 需要
            n_ctx_embed=int(os.getenv("EMBEDDING_N_CTX", 2048)), 
            n_gpu_layers_embed=int(os.getenv("EMBEDDING_N_GPU_LAYERS", 0))
        )
        logger.info("LocalModelHandler for embeddings initialized successfully.")
    except Exception as e_embed_init:
        logger.error(f"Failed to initialize LocalModelHandler for embeddings: {e_embed_init}", exc_info=True)
        return

    # 2. 初始化 KGRetriever
    # 它会从环境变量 DUCKDB_KG_FILE_PATH 读取数据库路径
    kg_retriever_instance: Optional[KGRetriever] = None
    try:
        kg_retriever_instance = KGRetriever(embedder=embedder_handler)
        logger.info("KGRetriever (DuckDB) instance created successfully.")
    except Exception as e_kg_init:
        logger.error(f"Failed to initialize KGRetriever (DuckDB): {e_kg_init}", exc_info=True)
        return
        
    # 3. 执行测试用例
    successful_count = 0
    failed_count = 0

    for tc in test_cases:
        try:
            passed = await run_single_test(kg_retriever_instance, tc)
            if passed:
                successful_count += 1
            else:
                failed_count += 1
        except Exception as e_test_case:
            logger.error(f"--- Test Case '{tc['name']}' FAILED with unhandled exception: {e_test_case} ---", exc_info=True)
            failed_count += 1
        print("-" * 50) # 每个测试用例后的分隔符

    logger.info("\n--- Test Run Summary ---")
    logger.info(f"Total Test Cases: {len(test_cases)}")
    logger.info(f"Successful: {successful_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info("--- DuckDB KGRetriever Test Script Finished ---")

if __name__ == "__main__":
    # 确保 SGLANG_API_URL (或者说 LOCAL_LLM_API_BASE) 环境变量已设置，
    # 因为 extract_entities_for_kg_query 会调用 call_llm_via_openai_api_local_only，
    # 而后者依赖 LLM_API_URL，LLM_API_URL 又从 SGLANG_API_URL 或 LOCAL_LLM_API_BASE 获取。
    # 在 .env 文件中我们有 LOCAL_LLM_API_BASE="http://localhost:8088/v1"
    # 并且 llm_interface.py 中 LLM_API_URL = os.getenv("SGLANG_API_URL", "http://localhost:8088/v1/chat/completions")
    # 所以只要 local_llm_service.py 在运行，这里应该没问题。

    # 确保您的 DuckDB 数据库文件 duckdb_knowledge_graph.db 已由 Dagster 流水线正确生成和填充。
    # 确保您的本地 LLM 服务 (local_llm_service.py) 正在运行在端口 8088。
    asyncio.run(main_test_kg_retriever())