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
from cachetools import TTLCache
import hashlib

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
from zhz_rag.llm.llm_interface import (
    generate_answer_from_context, 
    generate_expansion_and_entities, 
    NO_ANSWER_PHRASE_ANSWER_CLEAN
)
from zhz_rag.llm.rag_prompts import (
    get_answer_generation_messages, 
    get_table_qa_messages
)
from zhz_rag.llm.local_model_handler import LlamaCppEmbeddingFunction as LocalModelHandlerWrapper
from zhz_rag.core_rag.retrievers.chromadb_retriever import ChromaDBRetriever
from zhz_rag.core_rag.retrievers.file_bm25_retriever import FileBM25Retriever
from zhz_rag.core_rag.kg_retriever import KGRetriever
from zhz_rag.core_rag.retrievers.embedding_functions import LlamaCppEmbeddingFunction # Re-importing from here for clarity after changes
from zhz_rag.core_rag.fusion_engine import FusionEngine
from zhz_rag.utils.interaction_logger import log_interaction_data
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.resources import GGUFEmbeddingResource # <--- 新增导入

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

# --- 应用上下文和生命周期管理 ---
@dataclass
class RAGAppContext:
    model_handler: LlamaCppEmbeddingFunction # This will now encapsulate GGUFEmbeddingResource
    chroma_retriever: ChromaDBRetriever
    kg_retriever: KGRetriever
    file_bm25_retriever: FileBM25Retriever
    fusion_engine: FusionEngine
    answer_cache: TTLCache
    gguf_embedding_resource: GGUFEmbeddingResource # Keep a direct reference for teardown
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    api_logger.info("--- RAG API Service: Initializing RAG components... ---")
    llm_gguf_model_path_for_handler = os.getenv("LOCAL_LLM_GGUF_MODEL_PATH")
    embedding_gguf_model_path = os.getenv("EMBEDDING_MODEL_PATH") # This will now be used by GGUFEmbeddingResource
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY")
    bm25_index_dir = os.getenv("BM25_INDEX_DIRECTORY")
    duckdb_file_path_for_api = os.getenv("DUCKDB_KG_FILE_PATH")
    embedding_pool_size = int(os.getenv("EMBEDDING_SUBPROCESS_POOL_SIZE", "2")) # Not directly used by GGUFEmbeddingResource

    # Environment variable checks (kept for completeness, assuming they are in .env)
    if not llm_gguf_model_path_for_handler:
        api_logger.warning("LOCAL_LLM_GGUF_MODEL_PATH not set. LLM functionality may be limited.")
    if not embedding_gguf_model_path:
        api_logger.warning("EMBEDDING_MODEL_PATH not set. Embedding functionality may be limited.")
    if not chroma_persist_dir:
        api_logger.warning("CHROMA_PERSIST_DIRECTORY not set. ChromaDB will use in-memory mode or fail.")
    if not bm25_index_dir:
        api_logger.warning("BM25_INDEX_DIRECTORY not set. BM25 retrieval will not function.")
    if not duckdb_file_path_for_api:
        api_logger.warning("DUCKDB_KG_FILE_PATH not set. KG retrieval will not function.")
    
    try:
        # 1. Initialize GGUFEmbeddingResource (这部分保持不变)
        embedding_api_url = os.getenv("EMBEDDING_API_URL", "http://127.0.0.1:8089")
        api_logger.info(f"Attempting to initialize GGUFEmbeddingResource with API URL: {embedding_api_url}")
        class FakeDagsterContext:
            def __init__(self, logger_instance): self.log = logger_instance
        fake_dagster_context = FakeDagsterContext(api_logger)
        gguf_embed_resource = GGUFEmbeddingResource(api_url=embedding_api_url)
        await asyncio.to_thread(gguf_embed_resource.setup_for_execution, fake_dagster_context) 
        api_logger.info(f"GGUFEmbeddingResource setup called. Dimension: {gguf_embed_resource.get_embedding_dimension()}")

        # 2. Create the "model_handler" instance using the class from local_model_handler.py
        #    This instance will be passed to the LlamaCppEmbeddingFunction from core_rag.
        #    Pass the embedding_model_path it might need for validation or logging.
        actual_embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH")
        the_model_handler_for_core_rag = LocalModelHandlerWrapper(
            resource=gguf_embed_resource,
            embedding_model_path_for_handler=actual_embedding_model_path
        )
        api_logger.info("LocalModelHandlerWrapper (acting as model_handler) initialized.")

        # 3. Now initialize LlamaCppEmbeddingFunction (from core_rag) using the_model_handler_for_core_rag
        #    This is the embedding function that ChromaDB will use.
        #    The parameter name is 'model_handler'.
        chroma_embedding_function = LlamaCppEmbeddingFunction(model_handler=the_model_handler_for_core_rag)
        api_logger.info("LlamaCppEmbeddingFunction (for ChromaDB) initialized with LocalModelHandlerWrapper.")
        
        # 4. Update RAGAppContext and other components
        #    'model_handler' in RAGAppContext should probably be the_model_handler_for_core_rag
        #    or chroma_embedding_function depending on its intended use elsewhere.
        #    Let's assume 'model_handler' in RAGAppContext is the higher-level one.
        
        #    'custom_embed_fn' for ChromaDBRetriever is chroma_embedding_function
        #    'embedder' for KGRetriever should also be chroma_embedding_function (or compatible)

        app_ctx_model_handler = chroma_embedding_function # This is the one used by Chroma and KG

        chroma_retriever_instance = ChromaDBRetriever(
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "zhz_rag_collection"), # <--- 修改这里
            persist_directory=chroma_persist_dir,
            embedding_function=app_ctx_model_handler # 或者之前用的 custom_embed_fn，确保它正确初始化
        )
        file_bm25_retriever_instance = FileBM25Retriever(index_directory=bm25_index_dir)
        
        kg_retriever_instance = KGRetriever(
            db_file_path=duckdb_file_path_for_api, 
            embedder=app_ctx_model_handler 
        )
       
        rrf_k_env_str = os.getenv("RRF_K_VALUE", "60")
        try:
            rrf_k_setting = int(rrf_k_env_str)
        except ValueError:
            api_logger.warning(f"Invalid RRF_K_VALUE '{rrf_k_env_str}' in environment. Defaulting to 60.")
            rrf_k_setting = 60
        
        fusion_engine_instance = FusionEngine(logger=api_logger, rrf_k=rrf_k_setting)
        
        final_answer_cache = TTLCache(maxsize=100, ttl=900) # Cache 100 final answers for 15 minutes
        
        app.state.rag_context = RAGAppContext(
            model_handler=app_ctx_model_handler, # <--- 更新这里
            chroma_retriever=chroma_retriever_instance,
            kg_retriever=kg_retriever_instance, 
            file_bm25_retriever=file_bm25_retriever_instance,
            fusion_engine=fusion_engine_instance,
            answer_cache=final_answer_cache,
            gguf_embedding_resource=gguf_embed_resource # Store the resource for proper teardown
        )
        
        api_logger.info("--- RAG components initialized successfully. ---")
        asyncio.create_task(log_writer_task())
    except Exception as e:
        api_logger.critical(f"FATAL: Failed to initialize RAG components: {e}", exc_info=True)
        app.state.rag_context = None
    
    yield # This is the FastAPI lifespan's yield point
    
    api_logger.info("--- RAG API Service: Cleaning up resources ---")
    
    # Clean up GGUFEmbeddingResource using its teardown_for_execution method
    if app.state.rag_context and app.state.rag_context.gguf_embedding_resource:
        if hasattr(app.state.rag_context.gguf_embedding_resource, 'teardown_for_execution'):
            api_logger.info("Calling teardown_for_execution on GGUFEmbeddingResource...")
            class FakeDagsterContext: # Temporary helper class for teardown
                def __init__(self, logger_instance):
                    self.log = logger_instance
            fake_dagster_context_teardown = FakeDagsterContext(api_logger)
            await asyncio.to_thread(app.state.rag_context.gguf_embedding_resource.teardown_for_execution, fake_dagster_context_teardown)
            api_logger.info("GGUFEmbeddingResource teardown_for_execution called.")
        else:
            api_logger.warning("GGUFEmbeddingResource does not have a teardown_for_execution method.")
    else:
        api_logger.warning("No RAGAppContext or GGUFEmbeddingResource found for teardown.")

    api_logger.info("--- Cleanup complete. ---")

# --- FastAPI 应用实例 (保持不变) ---
app = FastAPI(
    title="Upgraded Standalone RAG API Service",
    description="Provides API access to the RAG framework, now powered by Qwen3 models.",
    version="2.0.2_fstring_fix", # Version updated
    lifespan=lifespan
)

# --- Helper functions (kept unchanged) ---
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
    api_logger.info(f"\n--- Received RAG query: '{query_request.query}' ---")
    start_time_total = datetime.now(timezone.utc)
    cpu_bound_semaphore = asyncio.Semaphore(int(os.getenv("RAG_CPU_CONCURRENCY_LIMIT", "2")))
    app_ctx: RAGAppContext = request.app.state.rag_context
    if not app_ctx:
        raise HTTPException(status_code=503, detail="RAG service is not properly initialized.")

    response_to_return: Optional[HybridRAGResponse] = None
    exception_occurred: Optional[Exception] = None
    interaction_id_for_log = str(uuid.uuid4())
    log_data_for_finally: Dict[str, Any] = {}

    try:
        cache_key = hashlib.md5(query_request.model_dump_json().encode('utf-8')).hexdigest()
        cached_response = app_ctx.answer_cache.get(cache_key)
        if cached_response is not None:
            api_logger.info(f"FINAL ANSWER CACHE HIT for query: '{query_request.query}'")
            response_to_return = cached_response
            log_data_for_finally = {
                "final_answer": response_to_return.answer,
                "final_docs": [doc.model_dump() for doc in response_to_return.retrieved_sources],
                "expanded_queries": ["FROM_CACHE"],
            }
            return response_to_return

        api_logger.info(f"FINAL ANSWER CACHE MISS for query: '{query_request.query}'")
        
        api_logger.info("--- Step 1: Performing unified LLM call for planning (Expansion, KG Extraction, Filter) ---")
        llm_planning_output = await generate_expansion_and_entities(query_request.query)

        if not llm_planning_output:
            raise HTTPException(status_code=500, detail="Failed to get planning output from LLM.")

        expanded_queries = llm_planning_output.expanded_queries
        kg_extraction_info = llm_planning_output.extracted_entities_for_kg
        metadata_filter = llm_planning_output.metadata_filter
        
        api_logger.info(f"--- Step 2: Starting parallel retrieval for {len(expanded_queries)} queries. Metadata Filter: {metadata_filter} ---")
        
        # --- START: 核心修改 ---
        # 1. 对 KG 和 BM25 的检索任务保持不变，因为它们不受此元数据过滤器的影响
        kg_task = app_ctx.kg_retriever.retrieve(query_request.query, kg_extraction_info, query_request.top_k_kg)
        bm25_tasks = [
            run_sync_with_semaphore(
                cpu_bound_semaphore, 
                f"bm25.retrieve({q[:20]}..)", 
                app_ctx.file_bm25_retriever.retrieve, 
                q, 
                query_request.top_k_bm25
            ) for q in expanded_queries
        ]

        # 2. 对 ChromaDB 的检索任务进行智能过滤
        chroma_tasks = []
        for q in expanded_queries:
            # 只有当查询是原始查询且规划器生成了过滤器时，才应用过滤器
            # 这确保了针对特定文件的查询是精确的
            # 而扩展出的其他相关问题则在整个知识库中搜索，以保证召回率
            current_filter = metadata_filter if q == query_request.query else None
            
            api_logger.info(f"Chroma task for query '{q[:30]}...' will use filter: {current_filter}")
            
            chroma_tasks.append(
                run_with_semaphore(
                    cpu_bound_semaphore, 
                    f"chroma.retrieve({q[:20]}..)", 
                    app_ctx.chroma_retriever.retrieve(
                        q, 
                        query_request.top_k_vector, 
                        where_filter=current_filter
                    )
                )
            )
        # --- END: 核心修改 ---

        retrieval_tasks = chroma_tasks + bm25_tasks + [run_with_semaphore(cpu_bound_semaphore, "kg.retrieve", kg_task)]

        results_from_gather = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
        
        # ... (后续的融合、生成答案和日志记录逻辑保持不变) ...
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
        
        api_logger.info(f"--- Step 3: Fusing {len(all_retrieved_docs)} raw documents and generating final answer ---")
        if not all_retrieved_docs:
            final_answer = NO_ANSWER_PHRASE_ANSWER_CLEAN
            final_context_docs_obj = []
        else:
            final_context_docs_obj = await app_ctx.fusion_engine.fuse_results(
                all_retrieved_docs, query_request.query, query_request.top_k_final
            )
            if not final_context_docs_obj:
                final_answer = NO_ANSWER_PHRASE_ANSWER_CLEAN
            else:
                context_strings = [
                    f"Source Type: {doc.source_type}, Score: {f'{doc.score:.4f}' if isinstance(doc.score, float) else doc.score}\nContent: {doc.content}"
                    for doc in final_context_docs_obj
                ]
                fused_context = "\n\n---\n\n".join(context_strings)

                contains_table = any(
                    doc.metadata.get("source_element_type") == "table" or 
                    doc.metadata.get("source_element_type") == "table_row_chunk"
                    for doc in final_context_docs_obj if doc.metadata
                )

                if contains_table:
                    api_logger.info("Table detected in context. Using Table-QA prompt builder.")
                    prompt_builder_to_use = get_table_qa_messages
                else:
                    api_logger.info("No table detected in context. Using general answer prompt builder.")
                    prompt_builder_to_use = get_answer_generation_messages
                
                generated_final_answer = await generate_answer_from_context(
                    user_query=query_request.query, 
                    context_str=fused_context,
                    prompt_builder=prompt_builder_to_use
                )
                final_answer = generated_final_answer
        
        response_to_return = HybridRAGResponse(answer=final_answer, original_query=query_request.query, retrieved_sources=final_context_docs_obj)
        
        log_data_for_finally = {
            "final_answer": final_answer,
            "final_docs": [doc.model_dump() for doc in final_context_docs_obj],
            "expanded_queries": expanded_queries,
        }
        
        if response_to_return.answer != "Error: Processing failed due to " and not response_to_return.debug_info:
             app_ctx.answer_cache[cache_key] = response_to_return
             api_logger.info(f"FINAL ANSWER CACHED for query: '{query_request.query}'")

    except Exception as e:
        api_logger.error(f"Critical error in query_rag_endpoint: {e}", exc_info=True)
        exception_occurred = e
        response_to_return = HybridRAGResponse(
            answer=f"An internal error occurred: {str(e)}", original_query=query_request.query,
            retrieved_sources=[], debug_info={"error": str(e), "type": type(e).__name__}
        )
        log_data_for_finally = { "final_answer": response_to_return.answer, "final_docs": [], "expanded_queries": [], }

    finally:
        processing_time_seconds = (datetime.now(timezone.utc) - start_time_total).total_seconds()
        interaction_log_entry = {
            "interaction_id": interaction_id_for_log, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "task_type": "rag_query_processing_full_log", "original_user_query": query_request.query,
            "final_answer_from_llm": log_data_for_finally.get("final_answer", "N/A"),
            "final_context_docs_full": log_data_for_finally.get("final_docs", []),
            "retrieval_parameters": query_request.model_dump(),
            "expanded_queries_count": len(log_data_for_finally.get("expanded_queries", [])),
            "processing_time_seconds": round(processing_time_seconds, 3)
        }
        if exception_occurred:
            interaction_log_entry["error_details"] = f"{type(exception_occurred).__name__}: {str(exception_occurred)}"
            interaction_log_entry["error_traceback"] = traceback.format_exc() if hasattr(exception_occurred, '__traceback__') else "No traceback available"
        
        try:
            await log_queue.put(interaction_log_entry)
            api_logger.info(f"Log queue put successful for interaction: {interaction_id_for_log}")
        except Exception as log_e_final:
            api_logger.error(f"Failed to queue log for interaction {interaction_id_for_log}: {log_e_final}", exc_info=True)
        
        if exception_occurred:
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(exception_occurred)}")
        
        if response_to_return is None:
             raise HTTPException(status_code=500, detail="Internal Server Error: Response generation failed unexpectedly.")
        
        return response_to_return
    


if __name__ == "__main__":
    api_logger.info("Starting Standalone RAG API Service with Producer-Consumer Logger...")
    uvicorn.run("zhz_rag.api.rag_api_service:app", host="0.0.0.0", port=8081, reload=False)
