# /home/zhz/zhz_agent/zhz_rag/api/rag_api_service.py
# 版本: 3.1.0 - 手动实现 Small-to-Big Retrieval (更新异步检索调用, 修复prompts导入)

import pandas as pd
import io
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
import json
# LangChain 相关导入 - 我们仍然需要 Document 和 InMemoryStore
from langchain.storage import InMemoryStore
from langchain.docstore.document import Document as LangchainDocument

from zhz_rag.core_rag.fusion_engine import FusionEngine # <-- 添加导入

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
    generate_query_plan,
    generate_table_lookup_instruction,
    generate_actionable_suggestion,
    generate_expanded_queries,
    generate_document_summary, # <--- 确保导入
    NO_ANSWER_PHRASE_ANSWER_CLEAN
)
from zhz_rag.utils.hardware_manager import HardwareManager

# 修复: 移除 get_table_qa_messages 的导入，因为它导致了 AttributeError
from zhz_rag.llm.rag_prompts import get_answer_generation_messages, get_table_qa_messages, get_fusion_messages # <--- 修正导入 get_fusion_messages
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
    handler.setFormatter(handler)
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
    llm_gbnf_instance: Any
    fusion_engine: FusionEngine # <-- 添加这一行


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_logger.info("--- RAG API Service (v5.4 - HAL Corrected): Initializing... ---")
    
    # --- 1. 使用 HAL 获取硬件建议 ---
    hal = HardwareManager()
    
    # --- 2. 在应用层加载 GBNF LLM 模型 ---
    gbnf_llm = None
    model_path = os.getenv("LOCAL_LLM_GGUF_MODEL_PATH")
    if not model_path or not os.path.exists(model_path):
        api_logger.error(f"LLM model path not found or invalid: {model_path}. GBNF features will be disabled.")
    else:
        try:
            model_size_gb = os.path.getsize(model_path) / (1024**3)
            # 假设层数，这个值对于推荐很重要
            model_total_layers = 28 # Qwen3-1.7B
            
            n_gpu_layers = hal.recommend_llm_gpu_layers(
                model_total_layers=model_total_layers,
                model_size_on_disk_gb=model_size_gb
            )
            
            api_logger.info(f"Loading GBNF LLM from: {model_path} with {n_gpu_layers} layers offloaded to GPU.")
            # 导入Llama类
            from llama_cpp import Llama
            gbnf_llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=int(os.getenv("LLM_N_CTX", 4096)),
                verbose=False
            )
            api_logger.info("GBNF LLM instance pre-loaded successfully in API lifespan.")
        except Exception as e:
            api_logger.critical(f"FATAL: Failed to pre-load GBNF LLM model: {e}", exc_info=True)
            gbnf_llm = None

    # --- 3. 初始化其他服务 ---
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
        chroma_retriever_instance = ChromaDBRetriever(
            collection_name=chroma_collection_name,
            persist_directory=chroma_persist_dir,
            embedding_function=chroma_embedding_function
        )
        api_logger.info(f"Initialized ChromaDBRetriever. Collection: '{chroma_collection_name}'")

        api_logger.info("Building docstore from ChromaDB metadata upon service startup...")
        docstore = InMemoryStore()
        
        try:
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
            docstore = InMemoryStore() # Ensure docstore is initialized even on failure
            
        # --- 新增：初始化 FusionEngine ---
        fusion_engine_instance = FusionEngine(logger=api_logger)

        app.state.rag_context = RAGAppContext(
            chroma_retriever=chroma_retriever_instance,
            docstore=docstore,
            gguf_embedding_resource=gguf_embed_resource,
            answer_cache=TTLCache(maxsize=100, ttl=900),
            llm_gbnf_instance=gbnf_llm,
            fusion_engine=fusion_engine_instance # <-- 传递实例
        )
        
        api_logger.info("--- RAG components initialized successfully. ---")
        
        # The previous version had a duplicate assignment of app.state.rag_context here, removed it.
        # app.state.rag_context = RAGAppContext(
        #     chroma_retriever=chroma_retriever_instance,
        #     docstore=docstore,
        #     gguf_embedding_resource=gguf_embed_resource,
        #     answer_cache=TTLCache(maxsize=100, ttl=900),
        #     llm_gbnf_instance=gbnf_llm # 存储在应用启动时加载的实例
        # )
        # api_logger.info("--- RAG components initialized successfully. ---")
        asyncio.create_task(log_writer_task())
    except Exception as e:
        api_logger.critical(f"FATAL: Failed to initialize RAG components: {e}", exc_info=True)
        app.state.rag_context = None
    
    yield
    
    api_logger.info("--- RAG API Service: Cleaning up resources ---")
    
    if hasattr(app.state, 'rag_context') and app.state.rag_context and app.state.rag_context.gguf_embedding_resource:
        if hasattr(app.state.rag_context.gguf_embedding_resource, 'teardown_for_execution'):
            api_logger.info("Calling teardown_for_execution on GGUFEmbeddingResource...")
            class FakeDagsterContextTeardown:
                def __init__(self, logger_instance):
                    self.log = logger_instance
            fake_dagster_context_teardown = FakeDagsterContextTeardown(api_logger)
            await asyncio.to_thread(app.state.rag_context.gguf_embedding_resource.teardown_for_execution, fake_dagster_context_teardown)
            api_logger.info("GGUFEmbeddingResource teardown_for_execution called.")
        else:
            api_logger.warning("GGUFEmbeddingResource does not have a teardown_for_execution method.")
    else:
        api_logger.warning("No RAGAppContext or GGUFEmbeddingResource found for teardown.")

    if hasattr(app.state, 'rag_context') and app.state.rag_context and app.state.rag_context.llm_gbnf_instance:
        app.state.rag_context.llm_gbnf_instance = None
        api_logger.info("GBNF LLM instance released.")

    api_logger.info("--- Cleanup complete. ---")


# --- FastAPI 应用实例 (保持不变) ---
app = FastAPI(
    title="Advanced RAG API Service with Manual Small-to-Big Retrieval",
    description="Provides API access to the RAG framework, now with manual small-to-big retrieval.",
    version="3.1.0", # Version updated
    lifespan=lifespan
)

# --- API 端点 (V3.5 - Final Fix) ---
# --- API 端点 (V7.0 - Step-by-Step Synthesis) ---
@app.post("/api/v1/rag/query", response_model=HybridRAGResponse)
async def query_rag_endpoint(request: Request, query_request: QueryRequest):
    api_logger.info(f"\n--- Received RAG query (v7.0 - Step-by-Step Synthesis): '{query_request.query}' ---") # Updated version
    start_time_total = datetime.now(timezone.utc)
    app_ctx: RAGAppContext = request.app.state.rag_context
    if not app_ctx or not app_ctx.llm_gbnf_instance or not app_ctx.fusion_engine: # Added fusion_engine check
        raise HTTPException(status_code=503, detail="RAG service or its core components are not initialized.")

    interaction_id_for_log = str(uuid.uuid4())
    exception_occurred: Optional[Exception] = None
    response_to_return: Optional[HybridRAGResponse] = None

    try:
        # Cache logic remains the same
        cache_key = hashlib.md5(query_request.model_dump_json().encode('utf-8')).hexdigest()
        cached_response = app_ctx.answer_cache.get(cache_key)
        if cached_response is not None:
            api_logger.info(f"FINAL ANSWER CACHE HIT for query: '{query_request.query}'")
            return cached_response

        # Stage 1 & 2: Expansion, Planning, Retrieval (remains the same)
        api_logger.info("--- Step 1 & 2: Expansion, Planning, Retrieval ---")
        query_plan = await generate_query_plan(app_ctx.llm_gbnf_instance, user_query=query_request.query)
        sub_queries = await generate_expanded_queries(app_ctx.llm_gbnf_instance, original_query=query_request.query)
        unique_queries = list(dict.fromkeys([query_plan.query] + sub_queries if query_plan else sub_queries))
        metadata_filter = query_plan.metadata_filter if query_plan else {}
        if metadata_filter and ("$and" in metadata_filter) and len(metadata_filter["$and"]) == 1:
            metadata_filter = metadata_filter["$and"][0]
        
        retrieval_tasks = [app_ctx.chroma_retriever.retrieve(q, query_request.top_k_vector, metadata_filter or None) for q in unique_queries]
        all_child_chunks_results = await asyncio.gather(*retrieval_tasks)
        all_parent_ids = {chunk['metadata']['parent_id'] for chunk_list in all_child_chunks_results for chunk in chunk_list if 'parent_id' in chunk.get('metadata', {})}
        parent_docs = app_ctx.docstore.mget(list(all_parent_ids))
        valid_parent_docs = [doc for doc in parent_docs if doc]
        api_logger.info(f"Retrieved {len(valid_parent_docs)} unique candidate documents.")

        # Stage 3: Reranking (remains the same)
        api_logger.info(f"--- Step 3: Reranking {len(valid_parent_docs)} documents... ---")
        reranked_docs = await app_ctx.fusion_engine.rerank_documents(
            query=query_request.query, # Use the original query for relevance scoring
            documents=[RetrievedDocument(content=doc.page_content, metadata=doc.metadata, score=0.0, source_type="retrieved_parent") for doc in valid_parent_docs],
            top_n=5 # We'll feed the top 5 most relevant docs to the LLM
        )
        api_logger.info(f"Reranking complete. Top {len(reranked_docs)} documents selected.")

        # --- NEW Stage 4: Step-by-Step Synthesis ---
        api_logger.info(f"--- Step 4: Step-by-Step Synthesis on {len(reranked_docs)} reranked documents ---")
        final_answer = NO_ANSWER_PHRASE_ANSWER_CLEAN
        failure_reason = ""
        
        if not reranked_docs:
            failure_reason = "知识库中未能找到任何与您问题相关的信息。"
        else:
            # Table expert logic is still prioritized
            is_single_table_context = (len(reranked_docs) == 1 and reranked_docs[0].metadata.get("paragraph_type") == "table")
            if is_single_table_context:
                api_logger.info("Dispatching to Table QA Hybrid Expert.")
                table_doc = reranked_docs[0]
                try:
                    df = pd.read_csv(io.StringIO(table_doc.content), sep='|', skipinitialspace=True).dropna(axis=1, how='all').iloc[1:]
                    df.columns = [col.strip() for col in df.columns]
                    index_col_name = df.columns[0]
                    df = df.set_index(index_col_name)
                    instruction = await generate_table_lookup_instruction(
                        user_query=query_request.query,
                        table_column_names=[index_col_name] + df.columns.tolist()
                    )
                    if instruction and "row_identifier" in instruction and "column_identifier" in instruction:
                        row_id, col_id = instruction.get("row_identifier"), instruction.get("column_identifier")
                        if row_id in df.index and col_id in df.columns:
                            value = df.at[row_id, col_id]
                            final_answer = f"根据查找到的表格信息，{row_id}的{col_id}是{value}。"
                        else:
                            failure_reason = f"模型指令无法执行：在表格中未能同时找到行'{row_id}'和列'{col_id}'。"
                    else:
                        failure_reason = "模型未能从问题中生成有效的表格查询指令。"
                except Exception as e_pandas:
                    failure_reason = f"处理表格数据时遇到代码错误: {e_pandas}"
                
                if failure_reason:
                    api_logger.warning(f"Table QA Expert failed: {failure_reason}. Downgrading to Fusion Expert.")
            
            if final_answer == NO_ANSWER_PHRASE_ANSWER_CLEAN:
                # --- 分步合成开始 ---
                # 1. 并行生成所有文档的摘要
                summary_tasks = [generate_document_summary(app_ctx.llm_gbnf_instance, user_query=query_request.query, document_content=doc.content) for doc in reranked_docs]
                summaries = await asyncio.gather(*summary_tasks)
                
                # 2. 过滤掉不相关的摘要，并构建融合上下文
                relevant_summaries = []
                for doc, summary in zip(reranked_docs, summaries):
                    if summary:
                        # 在摘要前附加上下文来源，让最终融合时模型知道信息出处
                        filename = doc.metadata.get('filename', '未知文档')
                        relevant_summaries.append(f"根据文档《{filename}》的信息：{summary}")
                
                api_logger.info(f"Generated {len(relevant_summaries)} relevant summaries.")

                if not relevant_summaries:
                    failure_reason = "虽然检索到了相关文档，但无法从中提炼出与您问题直接相关的核心信息。"
                else:
                    # 3. 将精炼后的摘要交给最终的融合专家
                    fusion_context = "\n\n".join(relevant_summaries)
                    final_answer = await generate_answer_from_context(user_query=query_request.query, context_str=fusion_context, prompt_builder=lambda q, c: get_fusion_messages(q, c))
            
        # Final failure handling and response assembly (remains the same)
        if not final_answer or NO_ANSWER_PHRASE_ANSWER_CLEAN in final_answer:
            if not failure_reason: failure_reason = "根据检索到的上下文信息，无法直接回答您的问题。"
            suggestion = await generate_actionable_suggestion(app_ctx.llm_gbnf_instance, user_query=query_request.query, failure_reason=failure_reason)
            final_answer = f"{failure_reason} {suggestion}" if suggestion else failure_reason

        response_to_return = HybridRAGResponse(answer=final_answer, original_query=query_request.query, retrieved_sources=reranked_docs)
        if not failure_reason and NO_ANSWER_PHRASE_ANSWER_CLEAN not in final_answer:
            app_ctx.answer_cache[cache_key] = response_to_return
            api_logger.info(f"FINAL ANSWER CACHED.")
    except Exception as e:
        exception_occurred = e
        api_logger.error(f"Critical error in query_rag_endpoint: {e}", exc_info=True)
        response_to_return = HybridRAGResponse(answer=f"An internal server error occurred: {e}", original_query=query_request.query, retrieved_sources=[])
    finally:
        processing_time_seconds = (datetime.now(timezone.utc) - start_time_total).total_seconds()
        log_data_for_finally = {
            "interaction_id": interaction_id_for_log, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "task_type": "rag_query_processing_v7_0_step_by_step_synthesis", # Updated task_type
            "original_user_query": query_request.query,
            "final_answer_from_llm": response_to_return.answer if response_to_return else "N/A",
            "final_context_docs_full": [doc.model_dump() for doc in response_to_return.retrieved_sources] if response_to_return else [],
            "retrieval_parameters": query_request.model_dump(),
            "processing_time_seconds": round(processing_time_seconds, 3)
        }
        if exception_occurred:
            log_data_for_finally["error_details"] = f"{type(exception_occurred).__name__}: {str(exception_occurred)}"
            log_data_for_finally["error_traceback"] = traceback.format_exc()
        
        await log_queue.put(log_data_for_finally)
        
        if exception_occurred:
            raise HTTPException(status_code=500, detail=str(exception_occurred))
        
        if response_to_return is None:
            response_to_return = HybridRAGResponse(answer="An unexpected error occurred during response generation.", original_query=query_request.query, retrieved_sources=[])
        
        return response_to_return
    
    
if __name__ == "__main__":
    api_logger.info("Starting Standalone RAG API Service with Manual Small-to-Big Retrieval...")
    uvicorn.run("zhz_rag.api.rag_api_service:app", host="0.0.0.0", port=8081, reload=False)
