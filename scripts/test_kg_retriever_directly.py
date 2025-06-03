# /home/zhz/zhz_agent/scripts/manual_tests/test_kg_retriever_directly.py
import asyncio
import os
import sys
import logging

# --- 配置项目根目录到 sys.path ---
# 这使得我们可以像在项目根目录运行一样导入模块
# 例如 from zhz_rag.core_rag.kg_retriever import KGRetriever
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- 结束 sys.path 配置 ---

try:
    from zhz_rag.core_rag.kg_retriever import KGRetriever
    from zhz_rag.llm.sglang_wrapper import generate_cypher_query # KGRetriever 依赖它
    from zhz_rag.config.constants import NEW_KG_SCHEMA_DESCRIPTION # generate_cypher_query 依赖它
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure you are running this script from a context where 'zhz_rag' package is discoverable,")
    print(f"or that PYTHONPATH is set correctly. Current sys.path: {sys.path}")
    sys.exit(1)

# --- 日志配置 (与 KGRetriever 内部的日志级别和格式一致或更详细) ---
# KGRetriever 和 sglang_wrapper 内部都有自己的 logger
# 为了看到它们的输出，我们可以配置根 logger，或者确保它们的 logger 能输出到控制台
logging.basicConfig(
    level=logging.DEBUG, # 设置为 DEBUG 可以看到更多信息
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)] # 输出到标准输出
)
logger = logging.getLogger(__name__)

async def main():
    logger.info("--- Starting KGRetriever Direct Test ---")

    # 确保 local_llm_service.py 正在运行 (端口 8088)
    # 确保 KuzuDB 数据库路径正确且包含数据

    # 从环境变量获取 KuzuDB 路径，如果 KGRetriever 内部也这样做的话
    # KGRetriever 默认使用 KUZU_DB_PATH_ENV，其默认值是 "/home/zhz/zhz_agent/zhz_rag/stored_data/kuzu_default_db"
    # 如果您的 .env 文件或环境变量中设置了不同的 KUZU_DB_PATH，请确保这里也一致，或者让 KGRetriever 使用其默认值。
    kuzu_db_path_for_test = os.getenv("KUZU_DB_PATH", "/home/zhz/zhz_agent/zhz_rag/stored_data/kuzu_default_db")
    logger.info(f"Using KuzuDB path for test: {kuzu_db_path_for_test}")


    # 实例化 KGRetriever
    # 它会使用 sglang_wrapper.generate_cypher_query 作为默认的 Cypher 生成函数
    # generate_cypher_query 会调用 local_llm_service.py
    try:
        kg_retriever = KGRetriever(db_path=kuzu_db_path_for_test)
        logger.info("KGRetriever instance created.")
    except Exception as e_init:
        logger.error(f"Failed to initialize KGRetriever: {e_init}", exc_info=True)
        return

    test_queries = [
        {"query": "张三在哪里工作？", "top_k": 2},
        {"query": "项目Alpha的文档编写任务分配给了谁？", "top_k": 2},
        {"query": "法国的首都是哪里？", "top_k": 2} # 测试无法生成 Cypher 的情况
    ]

    for test_case in test_queries:
        user_query = test_case["query"]
        top_k_results = test_case["top_k"]
        logger.info(f"\n--- Testing query: '{user_query}' with top_k={top_k_results} ---")
        
        try:
            retrieved_kg_docs = await kg_retriever.retrieve_with_llm_cypher(
                query=user_query,
                top_k=top_k_results
            )

            if retrieved_kg_docs:
                logger.info(f"Retrieved {len(retrieved_kg_docs)} documents from KG for query '{user_query}':")
                for i, doc in enumerate(retrieved_kg_docs):
                    logger.info(f"  Doc {i+1}:")
                    logger.info(f"    Source Type: {doc.get('source_type')}")
                    logger.info(f"    Content: {doc.get('content')}")
                    logger.info(f"    Score: {doc.get('score')}")
                    logger.info(f"    Metadata: {doc.get('metadata')}")
            else:
                logger.info(f"No documents retrieved from KG for query '{user_query}'. This might be expected if Cypher was '无法生成'.")

        except Exception as e_retrieve:
            logger.error(f"Error during KG retrieval for query '{user_query}': {e_retrieve}", exc_info=True)
        
        logger.info("--- End of test case ---")

    # 关闭 KGRetriever (如果它有 close 方法且需要显式关闭)
    # KGRetriever 的 close 方法会删除 _db 对象，依赖其 __del__
    if hasattr(kg_retriever, 'close'):
        kg_retriever.close()
        logger.info("KGRetriever closed.")

if __name__ == "__main__":
    # 确保 local_llm_service.py 在端口 8088 上运行
    # 确保 KuzuDB 数据库路径正确且包含数据
    if not os.getenv("SGLANG_API_URL"): # sglang_wrapper.py 会使用这个环境变量
        os.environ["SGLANG_API_URL"] = "http://localhost:8088/v1/chat/completions" # 指向我们的本地服务
        logger.info(f"SGLANG_API_URL not set, defaulting to: {os.environ['SGLANG_API_URL']}")

    asyncio.run(main())
    logger.info("--- KGRetriever Direct Test Finished ---")