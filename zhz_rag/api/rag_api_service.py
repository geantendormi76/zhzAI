# 文件: zhz_rag/api/rag_api_service.py
# 版本: 生产者-消费者队列方案 (已修复f-string)

import os
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import logging
import sys
import uvicorn
import traceback
from fastapi import FastAPI, Request, HTTPException
from dataclasses import dataclass
from dotenv import load_dotenv
import uuid
from datetime import datetime, timezone

# --- .env 文件加载 (保持不变) ---
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_dir = os.path.abspath(os.path.join(_current_file_dir, "..", ".."))
_dotenv_path = os.path.join(_project_root_dir, ".env")
if os.path.exists(_dotenv_path):
    load_dotenv(dotenv_path=_dotenv_path)
    print(f"RagApiService: Successfully loaded .env file from: {_dotenv_path}")
else:
    print(f"RagApiService: .env file not found at {_dotenv_path}. Relying on system environment variables or defaults.")
    load_dotenv()

# --- 导入应用模块 (保持不变) ---
from zhz_rag.config.pydantic_models import QueryRequest, HybridRAGResponse, RetrievedDocument
from zhz_rag.llm.llm_interface import generate_answer_from_context, generate_expanded_queries, NO_ANSWER_PHRASE_ANSWER_CLEAN
from zhz_rag.llm.local_model_handler import LocalModelHandler
from zhz_rag.core_rag.retrievers.chromadb_retriever import ChromaDBRetriever
from zhz_rag.core_rag.retrievers.file_bm25_retriever import FileBM25Retriever
from zhz_rag.core_rag.kg_retriever import KGRetriever
from zhz_rag.core_rag.retrievers.embedding_functions import LlamaCppEmbeddingFunction
from zhz_rag.core_rag.fusion_engine import FusionEngine
from zhz_rag.utils.interaction_logger import log_interaction_data

# --- 日志配置 (保持不变) ---
api_logger = logging.getLogger("RAGApiServiceLogger")
api_logger.setLevel(logging.INFO)
if not api_logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    api_logger.addHandler(handler)
    api_logger.propagate = False

# --- 生产者-消费者队列 (保持不变) ---
log_queue = asyncio.Queue()

async def log_writer_task():
    api_logger.info("Log writer task started and is waiting for log entries.")
    while True:
        try:
            log_entry_to_write = await log_queue.get()
            await log_interaction_data(log_entry_to_write)
            log_queue.task_done()
            api_logger.info(f"Log writer successfully wrote interaction ID: {log_entry_to_write.get('interaction_id')}")
        except Exception as e:
            api_logger.error(f"Critical error in log_writer_task: {e}", exc_info=True)

# --- 应用上下文和生命周期管理 (保持不变) ---
@dataclass
class RAGAppContext:
    model_handler: LocalModelHandler
    chroma_retriever: ChromaDBRetriever
    kg_retriever: KGRetriever
    file_bm25_retriever: FileBM25Retriever
    fusion_engine: FusionEngine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # (这部分所有组件的初始化代码都保持不变)
    api_logger.info("--- RAG API Service: Initializing RAG components... ---")
    llm_gguf_model_path_for_handler = os.getenv("LOCAL_LLM_GGUF_MODEL_PATH")
    embedding_gguf_model_path = os.getenv("EMBEDDING_MODEL_PATH")
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY")
    bm25_index_dir = os.getenv("BM25_INDEX_DIRECTORY")
    duckdb_file_path_for_api = os.getenv("DUCKDB_KG_FILE_PATH")
    embedding_pool_size = int(os.getenv("EMBEDDING_SUBPROCESS_POOL_SIZE", "2"))
    
    # (所有环境变量检查和组件初始化逻辑都保持不变)
    try:
        model_handler = LocalModelHandler(
            llm_model_path=llm_gguf_model_path_for_handler, embedding_model_path=embedding_gguf_model_path,
            n_gpu_layers_embed=int(os.getenv("EMBEDDING_N_GPU_LAYERS", 0)), n_gpu_layers_llm=int(os.getenv("LLM_N_GPU_LAYERS", 0)),
            embedding_pool_size=embedding_pool_size
        )
        custom_embed_fn = LlamaCppEmbeddingFunction(model_handler=model_handler)
        chroma_retriever_instance = ChromaDBRetriever(
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "rag_documents"), persist_directory=chroma_persist_dir,
            embedding_function=custom_embed_fn
        )
        file_bm25_retriever_instance = FileBM25Retriever(index_directory=bm25_index_dir)
        kg_retriever_instance = KGRetriever(db_file_path=duckdb_file_path_for_api, embedder=model_handler)
       
        # use_rrf_env = os.getenv("FUSION_USE_RRF", "true").lower() == "true"
        rrf_k_env_str = os.getenv("RRF_K_VALUE", "60")
        try:
            rrf_k_setting = int(rrf_k_env_str)
        except ValueError:
            api_logger.warning(f"Invalid RRF_K_VALUE '{rrf_k_env_str}' in environment. Defaulting to 60.")
            rrf_k_setting = 60
        api_logger.info(f"FusionEngine will use RRF with k={rrf_k_setting}") # <--- 添加日志确认k值
        
        fusion_engine_instance = FusionEngine(logger=api_logger, rrf_k=rrf_k_setting) # <--- 修改这里，只传递rrf_k
        
        
        app.state.rag_context = RAGAppContext(
            model_handler=model_handler, chroma_retriever=chroma_retriever_instance,
            kg_retriever=kg_retriever_instance, file_bm25_retriever=file_bm25_retriever_instance,
            fusion_engine=fusion_engine_instance
        )
        api_logger.info("--- RAG components initialized successfully. ---")
        asyncio.create_task(log_writer_task())
    except Exception as e:
        api_logger.critical(f"FATAL: Failed to initialize RAG components: {e}", exc_info=True)
        app.state.rag_context = None
    
    yield
    
    api_logger.info("--- RAG API Service: Cleaning up resources ---")
    if app.state.rag_context and hasattr(app.state.rag_context.model_handler, 'close_embedding_pool'):
        app.state.rag_context.model_handler.close_embedding_pool()
    api_logger.info("--- Cleanup complete. ---")

# --- FastAPI 应用实例 (保持不变) ---
app = FastAPI(
    title="Upgraded Standalone RAG API Service",
    description="Provides API access to the RAG framework, now powered by Qwen3 models.",
    version="2.0.2_fstring_fix", # 版本号更新
    lifespan=lifespan
)

# ... (run_with_semaphore 等辅助函数保持不变) ...
async def run_with_semaphore(semaphore: asyncio.Semaphore, coro_or_func_name: str, coro_obj):
    async with semaphore:
        api_logger.debug(f"Semaphore acquired for async task: {coro_or_func_name}")
        result = await coro_obj
        api_logger.debug(f"Semaphore released for async task: {coro_or_func_name}")
        return result

async def run_sync_with_semaphore(semaphore: asyncio.Semaphore, func_name: str, func_obj, *args):
    async with semaphore:
        api_logger.debug(f"Semaphore acquired for sync task: {func_name}")
        result = await asyncio.to_thread(func_obj, *args)
        api_logger.debug(f"Semaphore released for sync task: {func_name}")
        return result

# --- API 端点 ---
@app.post("/api/v1/rag/query", response_model=HybridRAGResponse)
async def query_rag_endpoint(request: Request, query_request: QueryRequest):
    # (try 块之前的所有代码都保持不变)
    api_logger.info(f"\n--- Received RAG query: '{query_request.query}' ---")
    start_time_total = datetime.now(timezone.utc)
    cpu_bound_semaphore = asyncio.Semaphore(int(os.getenv("RAG_CPU_CONCURRENCY_LIMIT", "2")))
    app_ctx: RAGAppContext = request.app.state.rag_context
    if not app_ctx:
        raise HTTPException(status_code=503, detail="RAG service is not properly initialized.")

    final_answer_for_log = "Error: Processing incomplete"
    final_context_docs_for_log: List[Dict[str, Any]] = []
    expanded_queries_for_log: Optional[List[str]] = None
    response_to_return: Optional[HybridRAGResponse] = None
    exception_occurred: Optional[Exception] = None
    interaction_id_for_log = str(uuid.uuid4())

    try:
        # (召回和融合的逻辑保持不变)
        expanded_queries = await run_with_semaphore(cpu_bound_semaphore, "generate_expanded_queries", generate_expanded_queries(query_request.query))
        expanded_queries_for_log = expanded_queries
        retrieval_tasks = [
            run_with_semaphore(cpu_bound_semaphore, "chroma.retrieve", app_ctx.chroma_retriever.retrieve(q, query_request.top_k_vector)) for q in expanded_queries
        ] + [
            run_sync_with_semaphore(cpu_bound_semaphore, "bm25.retrieve", app_ctx.file_bm25_retriever.retrieve, q, query_request.top_k_bm25) for q in expanded_queries
        ] + [
            run_with_semaphore(cpu_bound_semaphore, "kg.retrieve", app_ctx.kg_retriever.retrieve(q, query_request.top_k_kg)) for q in expanded_queries
        ]
        results_from_gather = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
        # (处理召回结果的循环保持不变)
        all_retrieved_docs: List[RetrievedDocument] = []
        bm25_results_to_enrich: List[Dict[str, Any]] = []
        for task_result_group in results_from_gather:
            if isinstance(task_result_group, list):
                for item in task_result_group:
                    if isinstance(item, dict):
                        source_type = item.get("source_type")
                        if source_type in ["duckdb_kg", "vector_chromadb"]:
                            all_retrieved_docs.append(RetrievedDocument(**item))
                        elif source_type == "keyword_bm25":
                            bm25_results_to_enrich.append(item)
        if bm25_results_to_enrich:
            bm25_ids = list(set([res['id'] for res in bm25_results_to_enrich if 'id' in res]))
            if bm25_ids:
                texts_map = await app_ctx.chroma_retriever.get_texts_by_ids(bm25_ids)
                for res in bm25_results_to_enrich:
                    all_retrieved_docs.append(RetrievedDocument(source_type="keyword_bm25", content=texts_map.get(res['id'], ""), score=res.get('score'), metadata={"chunk_id": res['id']}))
        
        if not all_retrieved_docs:
            final_answer_for_log = NO_ANSWER_PHRASE_ANSWER_CLEAN
            response_to_return = HybridRAGResponse(answer=final_answer_for_log, original_query=query_request.query, retrieved_sources=[])
        
        if response_to_return is None:
            final_context_docs_obj = await run_with_semaphore(
                cpu_bound_semaphore, "fusion_engine.fuse_results",
                app_ctx.fusion_engine.fuse_results(all_retrieved_docs, query_request.query, query_request.top_k_final)
            )
            final_context_docs_for_log = [doc.model_dump(exclude_none=True) for doc in final_context_docs_obj]
            if not final_context_docs_obj:
                final_answer_for_log = NO_ANSWER_PHRASE_ANSWER_CLEAN
            else:
                # 【【【【【 这里的 f-string 已被修复 】】】】】
                context_strings = [
                    f"Source Type: {doc.source_type}, Score: {f'{doc.score:.4f}' if isinstance(doc.score, float) else doc.score}\nContent: {doc.content}"
                    for doc in final_context_docs_obj
                ]
                fused_context = "\n\n---\n\n".join(context_strings)
                generated_final_answer = await run_with_semaphore(
                    cpu_bound_semaphore, "generate_answer_from_context",
                    generate_answer_from_context(query_request.query, fused_context)
                )
                final_answer_for_log = generated_final_answer or NO_ANSWER_PHRASE_ANSWER_CLEAN
            
            response_to_return = HybridRAGResponse(answer=final_answer_for_log, original_query=query_request.query, retrieved_sources=final_context_docs_obj if final_context_docs_obj else [])

    except Exception as e:
        api_logger.error(f"Critical error in query_rag_endpoint: {e}", exc_info=True)
        exception_occurred = e
        # (异常处理逻辑保持不变)
        final_answer_for_log = f"Error: Processing failed due to {type(e).__name__}"
        response_to_return = HybridRAGResponse(
            answer=f"An internal error occurred: {str(e)}", original_query=query_request.query,
            retrieved_sources=[], debug_info={"error": str(e), "type": type(e).__name__}
        )

    finally:
        # (finally 块中的所有逻辑，包括放入队列，都保持不变)
        processing_time_seconds = (datetime.now(timezone.utc) - start_time_total).total_seconds()
        api_logger.info(f"!!!!!!!!!! FINALLY BLOCK ENTERED. ... !!!!!!!!!!")
        current_final_answer = final_answer_for_log if 'final_answer_for_log' in locals() and final_answer_for_log is not None else "Error: final_answer_for_log not set in finally"
        current_final_docs = final_context_docs_for_log if 'final_context_docs_for_log' in locals() and final_context_docs_for_log is not None else []
        current_expanded_q_count = len(expanded_queries_for_log) if 'expanded_queries_for_log' in locals() and expanded_queries_for_log is not None else 0
        interaction_log_entry = {
            "interaction_id": interaction_id_for_log, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "task_type": "rag_query_processing_full_log", "original_user_query": query_request.query,
            "final_answer_from_llm": current_final_answer, "final_context_docs_full": current_final_docs,
            "retrieval_parameters": query_request.model_dump(), "expanded_queries_count": current_expanded_q_count,
            "processing_time_seconds": round(processing_time_seconds, 3)
        }
        if exception_occurred:
            interaction_log_entry["error_details"] = f"{type(exception_occurred).__name__}: {str(exception_occurred)}"
            interaction_log_entry["error_traceback"] = traceback.format_exc() if hasattr(exception_occurred, '__traceback__') else "No traceback available"
        
        try:
            await log_queue.put(interaction_log_entry)
            api_logger.info(f"!!!!!!!!!! FINALLY BLOCK: Full RAG interaction log ADDED TO QUEUE. !!!!!!!!!!")
        except Exception as log_e_final:
            api_logger.error(f"!!!!!!!!!! FINALLY BLOCK CRITICAL ERROR: ...", exc_info=True)
        
        if exception_occurred:
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(exception_occurred)}")
        if response_to_return is None:
             raise HTTPException(status_code=500, detail="Internal Server Error: Response generation failed unexpectedly.")
        return response_to_return

# --- Main execution block (保持不变) ---
if __name__ == "__main__":
    api_logger.info("Starting Standalone RAG API Service with Producer-Consumer Logger...")
    uvicorn.run("zhz_rag.api.rag_api_service:app", host="0.0.0.0", port=8081, reload=False)
