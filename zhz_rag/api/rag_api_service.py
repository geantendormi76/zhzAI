# 文件: zhz_rag/api/rag_api_service.py (已修正，完整覆盖)

import os
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import logging
import sys

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from dataclasses import dataclass
from dotenv import load_dotenv


# --- 添加开始：明确指定 .env 文件路径 ---
# 获取 rag_api_service.py 文件所在的目录
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
# .env 文件通常在项目根目录，即 zhz_rag 包的上两级目录 (zhz_agent/)
# zhz_agent/zhz_rag/api/rag_api_service.py
# zhz_agent/zhz_rag/
# zhz_agent/  <-- .env 文件在这里
_project_root_dir = os.path.abspath(os.path.join(_current_file_dir, "..", ".."))
_dotenv_path = os.path.join(_project_root_dir, ".env")

if os.path.exists(_dotenv_path):
    load_dotenv(dotenv_path=_dotenv_path)
    print(f"RagApiService: Successfully loaded .env file from: {_dotenv_path}")
else:
    print(f"RagApiService: .env file not found at {_dotenv_path}. Relying on system environment variables or defaults.")
    # 即使没找到，也执行一次 load_dotenv()，它可能会从其他地方加载或什么都不做
    load_dotenv()


# --- [MODIFIED] Import new handlers and functions ---
from zhz_rag.config.pydantic_models import QueryRequest, HybridRAGResponse, RetrievedDocument
from zhz_rag.llm.llm_interface import generate_answer_from_context, generate_expanded_queries, NO_ANSWER_PHRASE_ANSWER_CLEAN
# --- 添加 LocalModelHandler 的导入 ---
from zhz_rag.llm.local_model_handler import LocalModelHandler
from zhz_rag.core_rag.retrievers.chromadb_retriever import ChromaDBRetriever
from zhz_rag.core_rag.retrievers.file_bm25_retriever import FileBM25Retriever
from zhz_rag.core_rag.kg_retriever import KGRetriever
# --- 添加 LlamaCppEmbeddingFunction 的导入 ---
from zhz_rag.core_rag.retrievers.embedding_functions import LlamaCppEmbeddingFunction
from zhz_rag.core_rag.fusion_engine import FusionEngine

# --- 日志配置 ---
api_logger = logging.getLogger("RAGApiServiceLogger")
api_logger.setLevel(logging.INFO)
if not api_logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    api_logger.addHandler(handler)
    api_logger.propagate = False

load_dotenv()

# --- 应用上下文 Dataclass ---
@dataclass
class RAGAppContext:
    model_handler: LocalModelHandler
    chroma_retriever: ChromaDBRetriever
    kg_retriever: KGRetriever
    file_bm25_retriever: FileBM25Retriever
    fusion_engine: FusionEngine

# --- FastAPI 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    api_logger.info("--- RAG API Service: Initializing RAG components with Qwen3 GGUF models ---")
    
    # --- 读取环境变量 ---
    # LLM GGUF 模型路径 (用于 LocalModelHandler 中的 LLM 功能，如果 rag_api_service 也用它生成答案的话)
    # 注意: 当前 rag_api_service 中的 generate_answer_from_context 等函数是调用外部的 local_llm_service.py
    # 所以 llm_model_path_for_handler 可能暂时不直接用于答案生成，但 LocalModelHandler 初始化需要它。
    # 如果您希望 LocalModelHandler 只负责嵌入，可以将 llm_model_path 设置为 None。
    # 我们先假设您可能希望 LocalModelHandler 也能处理 LLM 任务。
    llm_gguf_model_path_for_handler = os.getenv("LOCAL_LLM_GGUF_MODEL_PATH") # 您需要在 .env 中定义这个变量
    if not llm_gguf_model_path_for_handler:
        api_logger.warning("LOCAL_LLM_GGUF_MODEL_PATH not set in .env. LLM features of LocalModelHandler might be unavailable.")

    embedding_gguf_model_path = os.getenv("EMBEDDING_MODEL_PATH") # 这个我们已经更新了
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY")
    bm25_index_dir = os.getenv("BM25_INDEX_DIRECTORY")
    kuzu_db_path = os.getenv("KUZU_DB_PATH")


    if not all([embedding_gguf_model_path, chroma_persist_dir, bm25_index_dir, kuzu_db_path]):
        api_logger.critical("One or more critical paths (embedding, chroma, bm25, kuzu) are not set in environment variables!")
        # 根据您的需求，这里可以决定是否抛出异常使服务启动失败
        app.state.rag_context = None # 明确设置 rag_context 为 None
        yield # 即使部分失败，也需要 yield 以完成生命周期
        return

    try:
        # 1. 初始化 LocalModelHandler
        #    LocalModelHandler 现在负责加载GGUF嵌入模型和可选的GGUF LLM模型
        api_logger.info(f"Initializing LocalModelHandler with Embedding GGUF: {embedding_gguf_model_path}")
        if llm_gguf_model_path_for_handler:
             api_logger.info(f"LocalModelHandler will also attempt to load LLM GGUF: {llm_gguf_model_path_for_handler}")
        
        model_handler = LocalModelHandler(
            llm_model_path=llm_gguf_model_path_for_handler, # 可选，如果只用嵌入则为 None
            embedding_model_path=embedding_gguf_model_path,
            # 可以从 .env 读取 n_ctx, n_gpu_layers 等参数，或使用 LocalModelHandler 的默认值
            n_gpu_layers_embed=int(os.getenv("EMBED_N_GPU_LAYERS", 0)), # 示例
            n_gpu_layers_llm=int(os.getenv("LLM_N_GPU_LAYERS", 0))   # 示例
        )
        if not model_handler.embedding_model:
            raise RuntimeError("Failed to load GGUF embedding model in LocalModelHandler.")

        # 2. 创建 LlamaCppEmbeddingFunction
        api_logger.info("Creating LlamaCppEmbeddingFunction...")
        custom_embed_fn = LlamaCppEmbeddingFunction(model_handler=model_handler)

        # 3. 初始化 ChromaDBRetriever，传入自定义嵌入函数
        api_logger.info(f"Initializing ChromaDBRetriever with persist_directory: {chroma_persist_dir}")
        chroma_retriever_instance = ChromaDBRetriever(
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "rag_documents"), # 从env或默认
            persist_directory=chroma_persist_dir,
            embedding_function=custom_embed_fn # <--- 关键：传入自定义嵌入函数
        )

        # 4. 初始化 FileBM25Retriever (保持不变)
        api_logger.info(f"Initializing FileBM25Retriever with index_directory: {bm25_index_dir}")
        file_bm25_retriever_instance = FileBM25Retriever(index_directory_path=bm25_index_dir)

        # 5. 初始化 KGRetriever
        #    KGRetriever 的 __init__ 需要一个 embedder 参数用于其内部的向量搜索。
        #    我们可以直接将 model_handler 传递给它，KGRetriever 内部应调用 model_handler.embed_query()
        api_logger.info(f"Initializing KGRetriever with db_path: {kuzu_db_path}")
        kg_retriever_instance = KGRetriever(
            db_path=kuzu_db_path,
            embedder=model_handler # <--- 将 model_handler 作为 embedder 传递
        )

        # 6. 初始化 FusionEngine (它内部会从env读取RERANKER_MODEL_PATH)
        api_logger.info("Initializing FusionEngine...")
        fusion_engine_instance = FusionEngine(logger=api_logger)

        app.state.rag_context = RAGAppContext(
            model_handler=model_handler,
            chroma_retriever=chroma_retriever_instance,
            kg_retriever=kg_retriever_instance,
            file_bm25_retriever=file_bm25_retriever_instance,
            fusion_engine=fusion_engine_instance
        )
        api_logger.info("--- RAG components initialized successfully using Qwen3 GGUF models. Service is ready. ---")
    except Exception as e:
        api_logger.critical(f"FATAL: Failed to initialize RAG components during startup: {e}", exc_info=True)
        app.state.rag_context = None # 确保出错时 rag_context 为 None
    
    yield
    
    api_logger.info("--- RAG API Service: Cleaning up resources ---")
    rag_context_to_clean: Optional[RAGAppContext] = getattr(app.state, 'rag_context', None)
    if rag_context_to_clean and rag_context_to_clean.kg_retriever:
        rag_context_to_clean.kg_retriever.close() # KGRetriever 有 close 方法
    # LocalModelHandler 中的 Llama 对象会在其被垃圾回收时自动释放资源
    api_logger.info("--- Cleanup complete. ---")

# --- FastAPI 应用实例 ---
app = FastAPI(
    title="Upgraded Standalone RAG API Service",
    description="Provides API access to the RAG framework, now powered by Qwen3 models.",
    version="2.0.0", # Major version upgrade!
    lifespan=lifespan
)

# --- API 端点 ---
@app.post("/api/v1/rag/query", response_model=HybridRAGResponse)
async def query_rag_endpoint(request: Request, query_request: QueryRequest):
    api_logger.info(f"\n--- Received RAG query: '{query_request.query}' ---")

    CONCURRENT_CPU_OPERATIONS_LIMIT = int(os.getenv("RAG_CPU_CONCURRENCY_LIMIT", "2"))
    cpu_bound_semaphore = asyncio.Semaphore(CONCURRENT_CPU_OPERATIONS_LIMIT)
    api_logger.info(f"Using Semaphore with limit: {CONCURRENT_CPU_OPERATIONS_LIMIT} for CPU-bound tasks.")

    async def run_with_semaphore(semaphore: asyncio.Semaphore, coro_or_func_name: str, coro_obj):
        """执行一个协程，并使用信号量控制并发。"""
        async with semaphore:
            api_logger.debug(f"Semaphore acquired for async task: {coro_or_func_name}")
            result = await coro_obj
            api_logger.debug(f"Semaphore released for async task: {coro_or_func_name}")
            return result
    
    async def run_sync_with_semaphore(semaphore: asyncio.Semaphore, func_name: str, func_obj, *args):
        """在线程中执行一个同步阻塞函数，并使用信号量控制并发。"""
        async with semaphore:
            api_logger.debug(f"Semaphore acquired for sync task: {func_name}")
            result = await asyncio.to_thread(func_obj, *args)
            api_logger.debug(f"Semaphore released for sync task: {func_name}")
            return result
    # --- 结束添加 ---

    app_ctx: RAGAppContext = request.app.state.rag_context
    if not app_ctx:
        # 这个 HTTPException 会被 FastAPI 框架捕获并返回给客户端
        raise HTTPException(status_code=503, detail="RAG service is not properly initialized.")

    try:
        api_logger.info("Acquiring semaphore for 'generate_expanded_queries'...")
        expanded_queries = await run_with_semaphore(
            cpu_bound_semaphore,
            "generate_expanded_queries",
            generate_expanded_queries(query_request.query)
        )
        api_logger.info(f"Query expansion completed. Expanded queries: {expanded_queries}")
        
        retrieval_tasks = []
        for q_text in expanded_queries:
            api_logger.debug(f"Preparing retrieval tasks for query part: '{q_text}'")
            retrieval_tasks.append(
                run_sync_with_semaphore(
                    cpu_bound_semaphore, 
                    "chroma_retriever.retrieve",
                    app_ctx.chroma_retriever.retrieve, 
                    q_text, 
                    query_request.top_k_vector
                )
            )
            retrieval_tasks.append(
                run_sync_with_semaphore(
                    cpu_bound_semaphore,
                    "file_bm25_retriever.retrieve",
                    app_ctx.file_bm25_retriever.retrieve, 
                    q_text, 
                    query_request.top_k_bm25
                )
            )
            retrieval_tasks.append(
                run_with_semaphore(
                    cpu_bound_semaphore,
                    "kg_retriever.retrieve",
                    app_ctx.kg_retriever.retrieve(q_text, query_request.top_k_kg)
                )
            )
        
        api_logger.info(f"Gathering {len(retrieval_tasks)} retrieval tasks...")
        results_from_gather = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
        api_logger.info("All retrieval tasks completed.")

        all_retrieved_docs: List[RetrievedDocument] = []
        bm25_results_to_enrich: List[Dict[str, Any]] = []

        for i, result_group in enumerate(results_from_gather):
            if isinstance(result_group, Exception):
                api_logger.error(f"An error occurred in retrieval task {i}: {result_group}", exc_info=result_group)
                continue
            if not result_group or not isinstance(result_group, list): 
                api_logger.warning(f"Retrieval task {i} returned empty or non-list result: {result_group}")
                continue

            for item in result_group:
                if isinstance(item, RetrievedDocument):
                    all_retrieved_docs.append(item)
                elif isinstance(item, dict) and 'id' in item and 'score' in item and item.get('source_type') != "knowledge_graph_kuzu": # 确保不是KG的结果
                    # 假设只有BM25会返回这种简单字典格式，需要内容补全
                    bm25_results_to_enrich.append(item)
                elif isinstance(item, dict) and item.get('source_type') == "knowledge_graph_kuzu": # KG retriever 返回的是字典
                    try:
                        all_retrieved_docs.append(RetrievedDocument(**item))
                    except Exception as e_kg_parse:
                        api_logger.error(f"Failed to parse KG result into RetrievedDocument: {item}, error: {e_kg_parse}")

        api_logger.info(f"Processed gather results. Total docs before BM25 enrichment: {len(all_retrieved_docs)}. BM25 results to enrich: {len(bm25_results_to_enrich)}")
        
        if bm25_results_to_enrich:
            bm25_ids = list(set([res['id'] for res in bm25_results_to_enrich]))
            if bm25_ids:
                api_logger.info(f"Enriching {len(bm25_ids)} BM25 results by fetching texts from ChromaDB.")
                # get_texts_by_ids 是同步的，也需要用 to_thread + semaphore
                texts_map = await run_sync_with_semaphore(
                    cpu_bound_semaphore,
                    "chroma_retriever.get_texts_by_ids",
                    app_ctx.chroma_retriever.get_texts_by_ids,
                    bm25_ids
                )
                api_logger.debug(f"Texts map from ChromaDB for BM25 enrichment: {texts_map}")
                for res in bm25_results_to_enrich:
                    chunk_id = res['id']
                    content_text = texts_map.get(chunk_id, f"[Content for BM25 chunk_id {chunk_id} not found in ChromaDB]")
                    all_retrieved_docs.append(RetrievedDocument(
                        source_type="keyword_bm25s",
                        content=content_text,
                        score=res.get('score'),
                        metadata={"chunk_id": chunk_id}
                    ))
                api_logger.info(f"BM25 results enriched. Total docs now: {len(all_retrieved_docs)}")
        
        if not all_retrieved_docs:
             api_logger.info("No documents retrieved from any source after processing all tasks.")
             return HybridRAGResponse(answer=NO_ANSWER_PHRASE_ANSWER_CLEAN, original_query=query_request.query, retrieved_sources=[])

        api_logger.info(f"Acquiring semaphore for 'fusion_engine.fuse_results' (total {len(all_retrieved_docs)} docs)...")
        final_context_docs = await run_with_semaphore(
            cpu_bound_semaphore,
            "fusion_engine.fuse_results",
            app_ctx.fusion_engine.fuse_results(all_retrieved_docs, query_request.query, query_request.top_k_final)
        )
        api_logger.info(f"Fusion engine processing completed. Final context docs: {len(final_context_docs)}")
        
        if not final_context_docs:
            final_answer = NO_ANSWER_PHRASE_ANSWER_CLEAN
            api_logger.info("No relevant context found after fusion. Returning no answer.")
        else:
            context_strings = []
            for doc in final_context_docs:
                score_to_display = doc.score # FusionEngine 应该已经统一了score
                score_str = f"{score_to_display:.4f}" if isinstance(score_to_display, float) else str(score_to_display if score_to_display is not None else 'N/A')
                context_strings.append(
                    f"Source Type: {doc.source_type}, Score: {score_str}\nContent: {doc.content}"
                )
            fused_context = "\n\n---\n\n".join(context_strings)
            api_logger.info(f"Fused context for LLM (length: {len(fused_context)} chars): {fused_context[:500]}...")

            api_logger.info("Acquiring semaphore for 'generate_answer_from_context'...")
            final_answer = await run_with_semaphore(
                cpu_bound_semaphore,
                "generate_answer_from_context",
                generate_answer_from_context(query_request.query, fused_context)
            ) or NO_ANSWER_PHRASE_ANSWER_CLEAN
            api_logger.info(f"Answer generation completed. Final answer: {final_answer[:200]}...")

        return HybridRAGResponse(answer=final_answer, original_query=query_request.query, retrieved_sources=final_context_docs)

    except Exception as e:
        api_logger.error(f"Critical error in query_rag_endpoint: {e}", exc_info=True)
        # 确保即使在这里出错，也返回一个符合 HybridRAGResponse 结构或至少是FastAPI能处理的HTTPException
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# --- Main execution block ---
if __name__ == "__main__":
    api_logger.info("Starting Standalone RAG API Service...")
    uvicorn.run(app, host="0.0.0.0", port=8081)