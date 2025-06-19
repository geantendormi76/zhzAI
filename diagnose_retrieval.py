# diagnose_retrieval.py

import asyncio
import os
import sys
import logging
from dotenv import load_dotenv

# --- 配置项目根路径，确保能正确导入模块 ---
# 假设此脚本位于项目根目录，或者 zhz_rag 的上一级目录
# 如果脚本在 zhz_rag/scripts/ 下，则 project_root 可能需要调整为 Path(__file__).resolve().parents[2]
from pathlib import Path
project_root = Path(__file__).resolve().parent # 或者 .parents[1] 等，取决于脚本位置
sys.path.append(str(project_root))
sys.path.append(str(project_root / "zhz_rag_pipeline_dagster")) # 为了 GGUFEmbeddingResource

# --- 加载 .env 文件 ---
dotenv_path = project_root / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print(f"DiagnoseScript: Loaded .env from {dotenv_path}")
else:
    print(f"DiagnoseScript: .env file not found at {dotenv_path}. Relying on environment variables.")

# --- 导入必要的模块 ---
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.resources import GGUFEmbeddingResource
from zhz_rag.llm.local_model_handler import LlamaCppEmbeddingFunction as LocalModelHandlerWrapper
from zhz_rag.core_rag.retrievers.embedding_functions import LlamaCppEmbeddingFunction as CoreRAGEmbeddingFunction
from zhz_rag.core_rag.retrievers.chromadb_retriever import ChromaDBRetriever
from zhz_rag.core_rag.retrievers.file_bm25_retriever import FileBM25Retriever

# --- 日志配置 ---
# 可以根据需要设置更详细的日志级别，例如 logging.DEBUG
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("DiagnoseRetrieval")

# --- 辅助类：模拟Dagster上下文 ---
class FakeDagsterContext:
    def __init__(self, logger_instance):
        self.log = logger_instance

async def main_diagnose():
    logger.info("--- Starting Retrieval Diagnosis ---")

    # --- 1. 初始化 GGUFEmbeddingResource ---
    embedding_api_url = os.getenv("EMBEDDING_API_URL", "http://127.0.0.1:8089")
    logger.info(f"Using Embedding API URL: {embedding_api_url}")
    
    fake_dagster_logger = logging.getLogger("FakeDagsterLogger") # 为 GGUFResource 创建一个 logger
    fake_dagster_logger.setLevel(logging.INFO) # 或 DEBUG
    if not fake_dagster_logger.hasHandlers():
        ch = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fake_dagster_logger.addHandler(ch)
        fake_dagster_logger.propagate = False

    fake_context = FakeDagsterContext(fake_dagster_logger)
    
    gguf_embed_resource = None
    try:
        gguf_embed_resource = GGUFEmbeddingResource(api_url=embedding_api_url)
        # GGUFEmbeddingResource.setup_for_execution 是同步的
        await asyncio.to_thread(gguf_embed_resource.setup_for_execution, fake_context)
        logger.info(f"GGUFEmbeddingResource initialized. Dimension: {gguf_embed_resource.get_embedding_dimension()}")
            # --- Test GGUFEmbeddingResource directly ---
        logger.info("--- Testing GGUFEmbeddingResource.encode() directly ---")
        test_text_for_embedding = "这是一个简单的测试句子，用于检查嵌入向量。"
        try:
            # GGUFEmbeddingResource.encode 是同步包装的异步，所以 to_thread 是合适的
            # 或者如果 encode 已经是 async def，则直接 await
            # 根据 GGUFEmbeddingResource 的实现，它内部使用 asyncio.run_coroutine_threadsafe 或 asyncio.run
            # 所以在异步函数中用 to_thread 是安全的
            embedding_vector_list = await asyncio.to_thread(gguf_embed_resource.encode, [test_text_for_embedding])
            if embedding_vector_list and embedding_vector_list[0]:
                vector_to_print = embedding_vector_list[0]
                logger.info(f"Embedding for '{test_text_for_embedding}':")
                logger.info(f"  Vector (first 10 elements): {vector_to_print[:10]}")
                logger.info(f"  Vector length: {len(vector_to_print)}")
                # 检查向量是否全为零或包含 NaN
                is_all_zeros = all(v == 0.0 for v in vector_to_print)
                has_nan = any(v != v for v in vector_to_print) # NaN is not equal to itself
                logger.info(f"  Is all zeros: {is_all_zeros}")
                logger.info(f"  Has NaN: {has_nan}")
            else:
                logger.error("GGUFEmbeddingResource.encode() returned empty or invalid result.")
        except Exception as e:
            logger.error(f"Error during direct GGUFEmbeddingResource.encode() test: {e}", exc_info=True)
        logger.info("--- Finished direct GGUFEmbeddingResource.encode() test ---")
        # --- End of GGUFEmbeddingResource direct test ---
    except Exception as e:
        logger.error(f"Failed to initialize GGUFEmbeddingResource: {e}", exc_info=True)
        return

    # --- 2. 初始化 LocalModelHandlerWrapper ---
    actual_embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH") # For embedding_model_path attribute
    try:
        model_handler_wrapper = LocalModelHandlerWrapper(
            resource=gguf_embed_resource,
            embedding_model_path_for_handler=actual_embedding_model_path
        )
        logger.info("LocalModelHandlerWrapper initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize LocalModelHandlerWrapper: {e}", exc_info=True)
        if gguf_embed_resource:
            await asyncio.to_thread(gguf_embed_resource.teardown_for_execution, fake_context)
        return

    # --- 3. 初始化 CoreRAGEmbeddingFunction (用于ChromaDB) ---
    try:
        core_rag_embed_fn = CoreRAGEmbeddingFunction(model_handler=model_handler_wrapper)
        logger.info("CoreRAGEmbeddingFunction initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize CoreRAGEmbeddingFunction: {e}", exc_info=True)
        if gguf_embed_resource:
            await asyncio.to_thread(gguf_embed_resource.teardown_for_execution, fake_context)
        return

    # --- 4. 初始化 ChromaDBRetriever ---
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME", "zhz_rag_collection")
    if not chroma_persist_dir:
        logger.error("CHROMA_PERSIST_DIRECTORY not set in .env")
        if gguf_embed_resource:
            await asyncio.to_thread(gguf_embed_resource.teardown_for_execution, fake_context)
        return
        
    chroma_retriever = None
    try:
        chroma_retriever = ChromaDBRetriever(
            collection_name=chroma_collection_name,
            persist_directory=chroma_persist_dir,
            embedding_function=core_rag_embed_fn # 使用 core_rag 的嵌入函数实例
        )
        collection_count = chroma_retriever._collection.count() if chroma_retriever._collection else "N/A (collection not loaded)"
        logger.info(f"ChromaDBRetriever initialized. Collection: {chroma_collection_name}, Count: {collection_count}")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDBRetriever: {e}", exc_info=True)
        if gguf_embed_resource:
            await asyncio.to_thread(gguf_embed_resource.teardown_for_execution, fake_context)
        return

    # --- 5. 初始化 FileBM25Retriever ---
    bm25_index_dir = os.getenv("BM25_INDEX_DIRECTORY")
    if not bm25_index_dir:
        logger.error("BM25_INDEX_DIRECTORY not set in .env")
        if gguf_embed_resource:
            await asyncio.to_thread(gguf_embed_resource.teardown_for_execution, fake_context)
        return

    bm25_retriever = None
    try:
        doc_count_bm25 = len(bm25_retriever._doc_ids) if hasattr(bm25_retriever, '_doc_ids') and bm25_retriever._doc_ids is not None else "N/A"
        logger.info(f"FileBM25Retriever initialized. Indexed documents: {doc_count_bm25}")
    except Exception as e:
        logger.error(f"Failed to initialize FileBM25Retriever: {e}", exc_info=True)
        if gguf_embed_resource:
            await asyncio.to_thread(gguf_embed_resource.teardown_for_execution, fake_context)
        return

    # --- 6. 执行检索测试 ---
    test_queries = [
        {
            "description": "Test sample.md content",
            "query": "项目Alpha的文档编写任务分配给了张三", # 请确认这句话确实在 sample.md 中
            "target_doc_hint": "sample.md"
        },
        {
            "description": "Test simple_table.xlsx content (Product B price)",
            "query": "产品B 4800", # 假设表格解析后包含 "产品B" 和 "4800"
            "target_doc_hint": "simple_table.xlsx"
        },
        {
            "description": "Test sample_article.html content",
            # 您需要从 sample_article.html 中选取一句特有的、不太可能在其他文档中重复的句子
            "query": "RAG框架通过结合预训练语言模型强大的生成能力和外部知识库的精确信息检索能力", 
            "target_doc_hint": "sample_article.html"
        },
        # 您可以添加更多测试用例
    ]

    for test_case in test_queries:
        logger.info(f"\n--- Running Test: {test_case['description']} ---")
        logger.info(f"Query: '{test_case['query']}' (Hint: should be in {test_case['target_doc_hint']})")

        # ChromaDB 检索
        if chroma_retriever:
            logger.info("--- ChromaDB Retrieval Results ---")
            try:
                # ChromaDBRetriever.retrieve 是异步的
                chroma_results = await chroma_retriever.retrieve(query_text=test_case['query'], n_results=3)
                if chroma_results:
                    for i, doc in enumerate(chroma_results):
                        logger.info(f"  Chroma Result {i+1}:")
                        logger.info(f"    ID: {doc.get('id')}")
                        logger.info(f"    Source Type: {doc.get('source_type')}")
                        logger.info(f"    Score (Distance): {doc.get('score')}") # 对于ChromaDB，score通常是距离
                        logger.info(f"    Metadata: {doc.get('metadata')}")
                        logger.info(f"    Content Snippet: {doc.get('content', '')[:200]}...") # 打印前200字符
                else:
                    logger.info("  No results from ChromaDB.")
            except Exception as e:
                logger.error(f"  Error during ChromaDB retrieval: {e}", exc_info=True)
        
        # BM25 检索
        if bm25_retriever:
            logger.info("--- BM25 Retrieval Results ---")
            try:
                # FileBM25Retriever.retrieve 是同步的，需要用 to_thread
                bm25_results = await asyncio.to_thread(bm25_retriever.retrieve, test_case['query'], n_results=3)
                if bm25_results:
                    for i, doc in enumerate(bm25_results):
                        logger.info(f"  BM25 Result {i+1}:")
                        logger.info(f"    ID: {doc.get('id')}")
                        logger.info(f"    Source Type: {doc.get('source_type')}")
                        logger.info(f"    Score: {doc.get('score')}")
                        logger.info(f"    Metadata (from BM25, might be limited): {doc.get('metadata')}") # BM25通常只存ID
                        # 如果需要内容，BM25结果通常需要后续从ChromaDB或其他地方根据ID获取
                        # logger.info(f"    Content (BM25 does not store content directly, ID: {doc.get('id')})")
                else:
                    logger.info("  No results from BM25.")
            except Exception as e:
                logger.error(f"  Error during BM25 retrieval: {e}", exc_info=True)

    # --- 7. 清理资源 ---
    if gguf_embed_resource:
        logger.info("Tearing down GGUFEmbeddingResource...")
        await asyncio.to_thread(gguf_embed_resource.teardown_for_execution, fake_context)
        logger.info("GGUFEmbeddingResource torn down.")

    logger.info("--- Diagnosis Script Finished ---")

if __name__ == "__main__":
    # 确保 embedding_api_service.py 正在运行
    print("IMPORTANT: Make sure your embedding_api_service (port 8089) is running before executing this script.")
    asyncio.run(main_diagnose())