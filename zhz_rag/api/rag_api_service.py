# /home/zhz/zhz_agent/zhz_rag/api/rag_api_service.py
# 版本: 3.1.0 - 手动实现 Small-to-Big Retrieval (更新异步检索调用, 修复prompts导入)

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

# LangChain 相关导入 - 我们仍然需要 Document 和 InMemoryStore
from langchain.storage import InMemoryStore
from langchain.docstore.document import Document as LangchainDocument

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

# --- 导入我们自己的模块 ---
from zhz_rag.config.pydantic_models import QueryRequest, HybridRAGResponse, RetrievedDocument
from zhz_rag.llm.llm_interface import (
    generate_answer_from_context,
    generate_query_plan,  # <--- 修改点
    NO_ANSWER_PHRASE_ANSWER_CLEAN
)
# 修复: 移除 get_table_qa_messages 的导入，因为它导致了 AttributeError
from zhz_rag.llm.rag_prompts import get_answer_generation_messages 
from zhz_rag.core_rag.retrievers.chromadb_retriever import ChromaDBRetriever
from zhz_rag.core_rag.retrievers.embedding_functions import LlamaCppEmbeddingFunction
from zhz_rag.llm.local_model_handler import LlamaCppEmbeddingFunction as LocalModelHandlerWrapper
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.resources import GGUFEmbeddingResource
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

# --- 应用上下文 Dataclass ---
@dataclass
class RAGAppContext:
    chroma_retriever: ChromaDBRetriever # 我们现在直接使用自己的 retriever
    docstore: InMemoryStore             # docstore 仍然用来存储父文档
    gguf_embedding_resource: GGUFEmbeddingResource
    answer_cache: TTLCache

@asynccontextmanager
async def lifespan(app: FastAPI):
    api_logger.info("--- RAG API Service (v3.1): Initializing for Manual Small-to-Big Retrieval... ---")
    
    # --- 资源初始化 (大部分保持不变) ---
    embedding_api_url = os.getenv("EMBEDDING_API_URL", "http://127.0.0.1:8089")
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME", "zhz_rag_collection")

    class FakeDagsterContext:
        def __init__(self, logger_instance): self.log = logger_instance
    
    gguf_embed_resource = GGUFEmbeddingResource(api_url=embedding_api_url)
    await asyncio.to_thread(gguf_embed_resource.setup_for_execution, FakeDagsterContext(api_logger))
    model_handler = LocalModelHandlerWrapper(resource=gguf_embed_resource)
    chroma_embedding_function = LlamaCppEmbeddingFunction(model_handler=model_handler)
    
    try:
        # 直接实例化我们自己的 ChromaDBRetriever
        chroma_retriever_instance = ChromaDBRetriever(
            collection_name=chroma_collection_name,
            persist_directory=chroma_persist_dir,
            embedding_function=chroma_embedding_function
        )
        api_logger.info(f"Initialized ChromaDBRetriever. Collection: '{chroma_collection_name}'")

        # --- 在生命周期函数中构建 docstore ---
        api_logger.info("Building docstore from ChromaDB metadata upon service startup...")
        docstore = InMemoryStore()
        
        try:
            # 确保 _collection 存在且可以被访问
            count = chroma_retriever_instance._collection.count()
            if count > 0:
                all_chunks_from_db = chroma_retriever_instance._collection.get(include=["metadatas", "documents"])
                
                parent_docs_map: Dict[str, Dict[str, Any]] = {}
                for i, metadata in enumerate(all_chunks_from_db['metadatas']):
                    parent_id = metadata.get("parent_id")
                    if parent_id:
                        if parent_id not in parent_docs_map:
                            # 存储父文档的元数据和所有子块的内容
                            parent_docs_map[parent_id] = {"metadata": metadata, "texts": []}
                        parent_docs_map[parent_id]["texts"].append(all_chunks_from_db['documents'][i])
                
                docs_to_store_in_docstore = [
                    LangchainDocument(
                        page_content="\n\n".join(sorted(data["texts"])), 
                        metadata={**data["metadata"], "doc_id": parent_id} # Merging metadata, ensuring doc_id
                    ) for parent_id, data in parent_docs_map.items()
                ]
                docstore.mset([(doc.metadata["doc_id"], doc) for doc in docs_to_store_in_docstore])
                api_logger.info(f"Docstore built successfully with {len(docs_to_store_in_docstore)} parent documents.")
            else:
                api_logger.warning(f"ChromaDB collection '{chroma_collection_name}' is empty. Docstore will be empty.")

        except Exception as e:
            api_logger.error(f"Failed to build docstore during startup: {e}", exc_info=True)
            # 即使构建失败，也继续，只是手动检索可能无法完全发挥作用
            docstore = InMemoryStore() # Ensure docstore is initialized even on failure

        app.state.rag_context = RAGAppContext(
            chroma_retriever=chroma_retriever_instance,
            docstore=docstore,
            gguf_embedding_resource=gguf_embed_resource,
            answer_cache=TTLCache(maxsize=100, ttl=900)
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
    title="Advanced RAG API Service with Manual Small-to-Big Retrieval",
    description="Provides API access to the RAG framework, now with manual small-to-big retrieval.",
    version="3.1.0", # Version updated
    lifespan=lifespan
)

# --- API 端点 ---
@app.post("/api/v1/rag/query", response_model=HybridRAGResponse)
async def query_rag_endpoint(request: Request, query_request: QueryRequest):
    api_logger.info(f"\n--- Received RAG query (v3.1): '{query_request.query}' ---")
    start_time_total = datetime.now(timezone.utc)
    app_ctx: RAGAppContext = request.app.state.rag_context
    if not app_ctx:
        raise HTTPException(status_code=503, detail="RAG service is not properly initialized.")

    response_to_return: Optional[HybridRAGResponse] = None
    exception_occurred: Optional[Exception] = None
    interaction_id_for_log = str(uuid.uuid4())
    log_data_for_finally: Dict[str, Any] = {}

    try:
        # Cache check before any heavy processing
        cache_key = hashlib.md5(query_request.model_dump_json().encode('utf-8')).hexdigest()
        cached_response = app_ctx.answer_cache.get(cache_key)
        if cached_response is not None:
            api_logger.info(f"FINAL ANSWER CACHE HIT for query: '{query_request.query}'")
            response_to_return = cached_response
            log_data_for_finally = {
                "final_answer": response_to_return.answer,
                "final_docs": [doc.model_dump() for doc in response_to_return.retrieved_sources],
                "expanded_queries": ["FROM_CACHE"], # Indicate cache hit in logs
            }
            return response_to_return

        api_logger.info(f"FINAL ANSWER CACHE MISS for query: '{query_request.query}'")

        # --- START: V3.2 - 规划与元数据过滤 ---
        # 1. 调用LLM规划器生成查询计划
        api_logger.info(f"--- Step 1.1: Generating query plan for: '{query_request.query}' ---")
        query_plan = await generate_query_plan(user_query=query_request.query)
        
        # 如果规划失败，则使用原始查询和空过滤器
        if not query_plan:
            api_logger.warning("Query plan generation failed. Falling back to basic query.")
            search_query = query_request.query
            metadata_filter = {}
        else:
            search_query = query_plan.query
            metadata_filter = query_plan.metadata_filter
            api_logger.info(f"Generated Plan -> Search Query: '{search_query}', Metadata Filter: {metadata_filter}")

            # --- 新增：简化元数据过滤器 ---
            # ChromaDB 要求 $and/$or 列表至少有两个元素。如果只有一个，我们需要将其简化。
            if metadata_filter and ("$and" in metadata_filter) and len(metadata_filter["$and"]) == 1:
                metadata_filter = metadata_filter["$and"][0]
                api_logger.info(f"Simplified single-element $and filter to: {metadata_filter}")
            elif metadata_filter and ("$or" in metadata_filter) and len(metadata_filter["$or"]) == 1:
                metadata_filter = metadata_filter["$or"][0]
                api_logger.info(f"Simplified single-element $or filter to: {metadata_filter}")
            # --- 新增结束 ---
            
        # 2. 使用规划后的查询和过滤器，检索小块
        api_logger.info(f"--- Step 1.2: Retrieving child chunks with generated plan ---")
        retrieved_child_chunks = await app_ctx.chroma_retriever.retrieve(
            query_text=search_query,
            n_results=query_request.top_k_vector,
            where_filter=metadata_filter if metadata_filter else None # <--- 应用元数据过滤器
        )
        api_logger.info(f"Retrieved {len(retrieved_child_chunks)} child chunks with filter.")

        # 3. 从小块中提取父ID (逻辑不变)
        parent_ids = [
            chunk['metadata']['parent_id'] 
            for chunk in retrieved_child_chunks 
            if chunk.get('metadata') and chunk['metadata'].get('parent_id')
        ]
        unique_parent_ids = list(set(parent_ids))
        api_logger.info(f"Found {len(unique_parent_ids)} unique parent document IDs.")

        # 4. 从 docstore 获取父文档 (逻辑不变)
        parent_docs = app_ctx.docstore.mget(unique_parent_ids)
        
        # 5. 准备最终上下文 (逻辑不变)
        final_context_docs_obj = [
            RetrievedDocument(
                source_type="parent_document_retrieval",
                content=doc.page_content,
                score=1.0,
                metadata=doc.metadata
            ) for doc in parent_docs if doc is not None
        ]
        
        # 6. 生成答案
        api_logger.info(f"--- Step 2: Generating final answer from {len(final_context_docs_obj)} parent documents ---")
        if not final_context_docs_obj:
            final_answer = NO_ANSWER_PHRASE_ANSWER_CLEAN
        else:
            context_strings = [
                f"Source Document ID: {doc.metadata.get('doc_id', 'N/A')}\nContent: {doc.content}" 
                for doc in final_context_docs_obj
            ]
            fused_context = "\n\n---\n\n".join(context_strings)
            
            # 修复: 直接使用 get_answer_generation_messages 避免 get_table_qa_messages 报错
            # 如果需要表格QA功能，确保 zhz_rag.llm.rag_prompts 中正确定义并导出 get_table_qa_messages
            # For now, simplifying to avoid the AttributeError.
            prompt_builder_to_use = get_answer_generation_messages
            
            generated_final_answer = await generate_answer_from_context(
                user_query=query_request.query, 
                context_str=fused_context,
                prompt_builder=prompt_builder_to_use
            )
            final_answer = generated_final_answer if generated_final_answer else NO_ANSWER_PHRASE_ANSWER_CLEAN
        
        response_to_return = HybridRAGResponse(
            answer=final_answer, 
            original_query=query_request.query, 
            retrieved_sources=final_context_docs_obj
        )
        
        log_data_for_finally = {
            "final_answer": final_answer,
            "final_docs": [doc.model_dump() for doc in final_context_docs_obj],
            "expanded_queries": [query_request.query], # Only original query used for manual retrieval directly
        }
        
        # Cache the successful response
        if response_to_return.answer != "Error: Processing failed due to " and not response_to_return.debug_info:
            app_ctx.answer_cache[cache_key] = response_to_return
            api_logger.info(f"FINAL ANSWER CACHED for query: '{query_request.query}'")

    except Exception as e:
        api_logger.error(f"Critical error in query_rag_endpoint (v3.1): {e}", exc_info=True)
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
            "task_type": "rag_query_processing_full_log_v3_1", # Updated task type for v3.1
            "original_user_query": query_request.query,
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
    api_logger.info("Starting Standalone RAG API Service with Manual Small-to-Big Retrieval...")
    uvicorn.run("zhz_rag.api.rag_api_service:app", host="0.0.0.0", port=8081, reload=False)
