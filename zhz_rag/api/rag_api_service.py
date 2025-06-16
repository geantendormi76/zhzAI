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
import uuid
from datetime import datetime, timezone

# --- 添加开始：明确指定 .env 文件路径 ---
# 获取 rag_api_service.py 文件所在的目录
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
# .env 文件通常在项目根目录，即 zhz_rag 包的上两级目录 (zhz_agent/)
_project_root_dir = os.path.abspath(os.path.join(_current_file_dir, "..", ".."))
_dotenv_path = os.path.join(_project_root_dir, ".env")

if os.path.exists(_dotenv_path):
    load_dotenv(dotenv_path=_dotenv_path)
    print(f"RagApiService: Successfully loaded .env file from: {_dotenv_path}")
else:
    print(f"RagApiService: .env file not found at {_dotenv_path}. Relying on system environment variables or defaults.")
    # 即使没找到，也执行一次 load_dotenv()，它可能会从其他地方加载或什么都不做
    load_dotenv()

# --- 导入应用模块 ---
from zhz_rag.config.pydantic_models import QueryRequest, HybridRAGResponse, RetrievedDocument
from zhz_rag.llm.llm_interface import generate_answer_from_context, generate_expanded_queries, NO_ANSWER_PHRASE_ANSWER_CLEAN
from zhz_rag.llm.local_model_handler import LocalModelHandler
from zhz_rag.core_rag.retrievers.chromadb_retriever import ChromaDBRetriever
from zhz_rag.core_rag.retrievers.file_bm25_retriever import FileBM25Retriever
from zhz_rag.core_rag.kg_retriever import KGRetriever
from zhz_rag.core_rag.retrievers.embedding_functions import LlamaCppEmbeddingFunction
from zhz_rag.core_rag.fusion_engine import FusionEngine
# 外部AI建议新增的导入
from zhz_rag.utils.interaction_logger import log_interaction_data

# --- 日志配置 ---
api_logger = logging.getLogger("RAGApiServiceLogger")
api_logger.setLevel(logging.INFO)
if not api_logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
    handler.setFormatter(formatter)
    api_logger.addHandler(handler)
    api_logger.propagate = False

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
    llm_gguf_model_path_for_handler = os.getenv("LOCAL_LLM_GGUF_MODEL_PATH")
    embedding_gguf_model_path = os.getenv("EMBEDDING_MODEL_PATH")
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY")
    bm25_index_dir = os.getenv("BM25_INDEX_DIRECTORY")
    duckdb_file_path_for_api = os.getenv("DUCKDB_KG_FILE_PATH")

    embedding_pool_size_str = os.getenv("EMBEDDING_SUBPROCESS_POOL_SIZE")
    embedding_pool_size = None
    if embedding_pool_size_str and embedding_pool_size_str.isdigit():
        embedding_pool_size = int(embedding_pool_size_str)
        api_logger.info(f"RAG API Service: Embedding subprocess pool size set from env: {embedding_pool_size}")
    else:
        api_logger.info(f"RAG API Service: Embedding subprocess pool size not set or invalid in env, LocalModelHandler will use default.")

    # --- 环境变量检查 ---
    required_paths_for_log = {
        "EMBEDDING_MODEL_PATH": embedding_gguf_model_path,
        "CHROMA_PERSIST_DIRECTORY": chroma_persist_dir,
        "BM25_INDEX_DIRECTORY": bm25_index_dir,
        "DUCKDB_KG_FILE_PATH": duckdb_file_path_for_api
    }
    missing_paths = [name for name, path in required_paths_for_log.items() if not path]

    if missing_paths:
        api_logger.critical(f"Critical environment variables not set: {', '.join(missing_paths)}")
        app.state.rag_context = None
        yield
        return

    try:
        # 1. 初始化 LocalModelHandler
        api_logger.info(f"Initializing LocalModelHandler with Embedding GGUF: {embedding_gguf_model_path}")
        if llm_gguf_model_path_for_handler:
            api_logger.info(f"LocalModelHandler will also attempt to load LLM GGUF: {llm_gguf_model_path_for_handler}")
        
        model_handler = LocalModelHandler(
            llm_model_path=llm_gguf_model_path_for_handler,
            embedding_model_path=embedding_gguf_model_path,
            n_gpu_layers_embed=int(os.getenv("EMBEDDING_N_GPU_LAYERS", 0)),
            n_gpu_layers_llm=int(os.getenv("LLM_N_GPU_LAYERS", 0)),
            embedding_pool_size=embedding_pool_size
        )
        api_logger.info("LocalModelHandler instance created. Embedding operations will use subprocess pool.")

        # 2. 创建 LlamaCppEmbeddingFunction
        api_logger.info("Creating LlamaCppEmbeddingFunction...")
        custom_embed_fn = LlamaCppEmbeddingFunction(model_handler=model_handler)

        # 3. 初始化 ChromaDBRetriever
        api_logger.info(f"Initializing ChromaDBRetriever with persist_directory: {chroma_persist_dir}")
        chroma_retriever_instance = ChromaDBRetriever(
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "rag_documents"),
            persist_directory=chroma_persist_dir,
            embedding_function=custom_embed_fn
        )

        # 4. 初始化 FileBM25Retriever
        api_logger.info(f"Initializing FileBM25Retriever with index_directory: {bm25_index_dir}")
        file_bm25_retriever_instance = FileBM25Retriever(index_directory=bm25_index_dir)

        # 5. 初始化 KGRetriever
        api_logger.info(f"Initializing KGRetriever (DuckDB) with db_file_path: {duckdb_file_path_for_api}")
        kg_retriever_instance = KGRetriever(
            db_file_path=duckdb_file_path_for_api,
            embedder=model_handler 
        )
        api_logger.info("KGRetriever (DuckDB) initialized for RAG API Service.")
        
        # 6. 初始化 FusionEngine
        api_logger.info("Initializing FusionEngine...")
        fusion_engine_instance = FusionEngine(logger=api_logger)

        app.state.rag_context = RAGAppContext(
            model_handler=model_handler,
            chroma_retriever=chroma_retriever_instance,
            kg_retriever=kg_retriever_instance,
            file_bm25_retriever=file_bm25_retriever_instance,
            fusion_engine=fusion_engine_instance
        )
        api_logger.info("--- RAG components initialized successfully. Embedding uses subprocesses. Service is ready. ---")
    except Exception as e:
        api_logger.critical(f"FATAL: Failed to initialize RAG components during startup: {e}", exc_info=True)
        app.state.rag_context = None 
    
    yield
    
    # --- 应用关闭阶段 ---
    api_logger.info("--- RAG API Service: Cleaning up resources ---")
    rag_context_to_clean: Optional[RAGAppContext] = getattr(app.state, 'rag_context', None)
    
    if rag_context_to_clean:
        if rag_context_to_clean.kg_retriever and hasattr(rag_context_to_clean.kg_retriever, 'close'):
            try:
                rag_context_to_clean.kg_retriever.close() 
                api_logger.info("KGRetriever closed.")
            except Exception as e_kg_close:
                api_logger.error(f"Error closing KGRetriever: {e_kg_close}", exc_info=True)

        if rag_context_to_clean.model_handler:
            api_logger.info("Attempting to close LocalModelHandler's embedding pool...")
            try:
                rag_context_to_clean.model_handler.close_embedding_pool()
            except Exception as e_pool_close:
                api_logger.error(f"Error explicitly closing LocalModelHandler's embedding pool: {e_pool_close}", exc_info=True)
    
    api_logger.info("--- Cleanup complete. ---")

# --- FastAPI 应用实例 ---
app = FastAPI(
    title="Upgraded Standalone RAG API Service",
    description="Provides API access to the RAG framework, now powered by Qwen3 models.",
    version="2.0.0",
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

    app_ctx: RAGAppContext = request.app.state.rag_context
    if not app_ctx:
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
                run_with_semaphore(
                    cpu_bound_semaphore, 
                    "chroma_retriever.retrieve",
                    app_ctx.chroma_retriever.retrieve(q_text, query_request.top_k_vector) 
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

        # --- 从这里开始是采纳外部AI建议修改的核心逻辑 ---
        all_retrieved_docs: List[RetrievedDocument] = []
        bm25_results_to_enrich: List[Dict[str, Any]] = []

        for i, task_result_group in enumerate(results_from_gather):
            if isinstance(task_result_group, Exception):
                api_logger.error(f"An error occurred in retrieval task {i}: {task_result_group}", exc_info=task_result_group)
                continue
            if not task_result_group: 
                api_logger.info(f"Retrieval task {i} returned empty result.")
                continue
            if not isinstance(task_result_group, list): 
                api_logger.warning(f"Retrieval task {i} returned non-list result: {type(task_result_group)} - {str(task_result_group)[:100]}")
                continue

            for item_idx, item in enumerate(task_result_group):
                if not isinstance(item, dict):
                    api_logger.warning(f"Task {i}, Item {item_idx}: Skipping non-dict item: {type(item)} - {str(item)[:100]}")
                    continue

                source_type = item.get("source_type")
                item_content = item.get("content")
                item_score = item.get("score")
                item_metadata = item.get("metadata", {})
                
                api_logger.debug(f"Task {i}, Item {item_idx}: Processing item with source_type='{source_type}', content_present={item_content is not None}, score={item_score}")

                if source_type == "duckdb_kg":
                    try:
                        if item_content is None:
                            item_content = f"[Content missing for KG doc {item_metadata.get('duckdb_retrieved_id_prop', 'UNKNOWN_ID')}]"
                            api_logger.warning(f"Task {i}, Item {item_idx} (KG): Content was None, using fallback. Metadata: {item_metadata}")
                        
                        doc_to_add = RetrievedDocument(
                            source_type=str(source_type),
                            content=str(item_content),
                            score=item_score,
                            metadata=item_metadata
                        )
                        all_retrieved_docs.append(doc_to_add)
                        api_logger.debug(f"Task {i}, Item {item_idx} (KG): Added to all_retrieved_docs. Chunk ID from meta: {item_metadata.get('duckdb_retrieved_id_prop')}")
                    except Exception as e_kg_parse:
                        api_logger.error(f"Task {i}, Item {item_idx} (KG): Failed to parse into RetrievedDocument: {item}, error: {e_kg_parse}")
                
                elif source_type == "vector_chromadb":
                    try:
                        chroma_item_id = item.get("id")
                        if item_content is None:
                            item_content = f"[Content missing for ChromaDB doc {chroma_item_id or 'UNKNOWN_ID'}]"
                            api_logger.warning(f"Task {i}, Item {item_idx} (ChromaDB): Content was None, using fallback. Chroma ID: {chroma_item_id}")

                        final_metadata = item_metadata.copy()
                        if "chunk_id" not in final_metadata and chroma_item_id:
                            final_metadata["chunk_id_from_chroma_retrieve"] = chroma_item_id
                        
                        doc_to_add = RetrievedDocument(
                            source_type=str(source_type),
                            content=str(item_content),
                            score=item_score,
                            metadata=final_metadata
                        )
                        all_retrieved_docs.append(doc_to_add)
                        api_logger.debug(f"Task {i}, Item {item_idx} (ChromaDB): Added to all_retrieved_docs. Chroma ID: {chroma_item_id}, Chunk ID in meta: {final_metadata.get('chunk_id') or final_metadata.get('chunk_id_from_chroma_retrieve')}")
                    except Exception as e_chroma_parse:
                        api_logger.error(f"Task {i}, Item {item_idx} (ChromaDB): Failed to parse into RetrievedDocument: {item}, error: {e_chroma_parse}")
                
                elif source_type == "keyword_bm25":
                    bm25_item_id = item.get("id")
                    if bm25_item_id is not None and item_score is not None:
                        item_for_enrich = {"id": bm25_item_id, "score": item_score, "source_type": source_type}
                        if item_metadata:
                            item_for_enrich["metadata"] = item_metadata
                        bm25_results_to_enrich.append(item_for_enrich)
                        api_logger.debug(f"Task {i}, Item {item_idx} (BM25): Added to bm25_results_to_enrich. ID: {bm25_item_id}")
                    else:
                        api_logger.warning(f"Task {i}, Item {item_idx} (BM25): Item missing 'id' or 'score', skipped for enrichment: {item}")
                
                else: 
                    api_logger.warning(f"Task {i}, Item {item_idx}: Unknown or unhandled source_type='{source_type}'. Item skipped: {str(item)[:200]}")
        
        api_logger.info(f"Processed gather results. Docs directly added (all_retrieved_docs): {len(all_retrieved_docs)}. BM25 results to enrich: {len(bm25_results_to_enrich)}")
        
        if bm25_results_to_enrich:
            bm25_ids = list(set([res['id'] for res in bm25_results_to_enrich if 'id' in res]))
            if bm25_ids:
                api_logger.info(f"Enriching {len(bm25_ids)} unique BM25 document IDs by fetching texts from ChromaDB.")
                texts_map = await app_ctx.chroma_retriever.get_texts_by_ids(bm25_ids)
                
                for res in bm25_results_to_enrich:
                    chunk_id = res['id']
                    content_text = texts_map.get(chunk_id)
                    
                    if content_text is None or "[Content for chunk_id" in content_text:
                        # 如果从ChromaDB未找到内容，或内容是占位符，记录警告
                        api_logger.warning(f"BM25 enrichment: Content for chunk_id {chunk_id} was not found or invalid in ChromaDB. Using fallback text.")
                        content_text = f"[Content for BM25 chunk_id {chunk_id} not found in ChromaDB]"

                    # 将补全了内容的BM25结果转换为RetrievedDocument并添加到主列表
                    all_retrieved_docs.append(RetrievedDocument(
                        source_type="keyword_bm25",
                        content=content_text,
                        score=res.get('score'),
                        metadata=res.get('metadata', {"chunk_id": chunk_id})
                    ))
                api_logger.info(f"BM25 results enriched. Total docs now: {len(all_retrieved_docs)}")
        # --- 核心修改逻辑结束 ---

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
                score_to_display = doc.score
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
            api_logger.info(f"Answer generation completed.  {final_answer[:200]}...")

        final_response = HybridRAGResponse(answer=final_answer, original_query=query_request.query, retrieved_sources=final_context_docs)

        # --- 外部AI建议新增的代码块 ---
        interaction_log_entry = {
            "interaction_id": str(uuid.uuid4()), 
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "task_type": "rag_query_processing_full_log", # <--- 确认这里的值
            "original_user_query": query_request.query,
            "final_answer_from_llm": final_answer,
            "final_context_docs_full": [doc.model_dump(exclude_none=True) for doc in final_context_docs],
            "retrieval_parameters": query_request.model_dump() 
        }
        await log_interaction_data(interaction_log_entry)
        # --- 新增代码块结束 ---
        
        return final_response

    except Exception as e:
        api_logger.error(f"Critical error in query_rag_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# --- Main execution block ---
if __name__ == "__main__":
    api_logger.info("Starting Standalone RAG API Service...")
    uvicorn.run("zhz_rag.api.rag_api_service:app", host="0.0.0.0", port=8081, reload=True)