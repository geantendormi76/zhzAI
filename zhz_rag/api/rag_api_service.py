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
    generate_expanded_queries, # <--- 添加这一行
    NO_ANSWER_PHRASE_ANSWER_CLEAN
)
# 修复: 移除 get_table_qa_messages 的导入，因为它导致了 AttributeError
from zhz_rag.llm.rag_prompts import get_answer_generation_messages, get_table_qa_messages
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

# --- API 端点 (V3.5 - Final Fix) ---
@app.post("/api/v1/rag/query", response_model=HybridRAGResponse)
async def query_rag_endpoint(request: Request, query_request: QueryRequest):
    api_logger.info(f"\n--- Received RAG query (v3.5 - Final Fix): '{query_request.query}' ---")
    start_time_total = datetime.now(timezone.utc)
    app_ctx: RAGAppContext = request.app.state.rag_context
    if not app_ctx:
        raise HTTPException(status_code=503, detail="RAG service is not properly initialized.")

    interaction_id_for_log = str(uuid.uuid4())
    
    # --- 修复1：确保 exception_occurred 在任何路径下都已定义 ---
    exception_occurred: Optional[Exception] = None
    response_to_return: Optional[HybridRAGResponse] = None

    try:
        # --- 缓存逻辑保持不变 ---
        cache_key = hashlib.md5(query_request.model_dump_json().encode('utf-8')).hexdigest()
        cached_response = app_ctx.answer_cache.get(cache_key)
        if cached_response is not None:
            api_logger.info(f"FINAL ANSWER CACHE HIT for query: '{query_request.query}'")
            return cached_response

        # --- 1. 查询扩展 ---
        api_logger.info(f"--- Step 1: Expanding original query ---")
        sub_queries = await generate_expanded_queries(query_request.query)
        api_logger.info(f"Generated {len(sub_queries)} sub-queries: {sub_queries}")
        
        # --- 2. 对每个子问题执行RAG，并收集所有独特的上下文 ---
        all_retrieved_docs_map: Dict[str, RetrievedDocument] = {}
        
        for sub_query in sub_queries:
            api_logger.info(f"--- Processing sub-query: '{sub_query}' ---")
            
            # a. 为子问题生成查询计划
            query_plan = await generate_query_plan(user_query=sub_query)
            search_query = query_plan.query if query_plan else sub_query
            metadata_filter = query_plan.metadata_filter if query_plan and query_plan.metadata_filter else {}
            
            # --- 修复2：在循环内部应用过滤器规范化逻辑 ---
            if isinstance(metadata_filter, dict) and len(metadata_filter) > 1 and not metadata_filter.keys() & {'$and', '$or', '$not'}:
                metadata_filter = {"$and": [{key: value} for key, value in metadata_filter.items()]}
            if isinstance(metadata_filter, dict):
                for op in ["$and", "$or"]:
                    if op in metadata_filter and isinstance(metadata_filter[op], list):
                        cleaned_list = [item for item in metadata_filter[op] if item and isinstance(item, dict)]
                        if cleaned_list: metadata_filter[op] = cleaned_list
                        else: metadata_filter.pop(op)
                if not metadata_filter: metadata_filter = {}
            if metadata_filter and ("$and" in metadata_filter) and len(metadata_filter["$and"]) == 1:
                metadata_filter = metadata_filter["$and"][0]
            elif metadata_filter and ("$or" in metadata_filter) and len(metadata_filter["$or"]) == 1:
                metadata_filter = metadata_filter["$or"][0]
            api_logger.info(f"Normalized filter for sub-query: {metadata_filter}")
            # --- 过滤器规范化结束 ---
            
            # b. 检索小块
            retrieved_child_chunks = await app_ctx.chroma_retriever.retrieve(
                query_text=search_query,
                n_results=query_request.top_k_vector,
                where_filter=metadata_filter if metadata_filter else None
            )

            # c. 获取父文档 (小块->大块)
            parent_ids = list(set(chunk['metadata']['parent_id'] for chunk in retrieved_child_chunks if chunk.get('metadata', {}).get('parent_id')))
            parent_docs = app_ctx.docstore.mget(parent_ids)
            
            # d. 收集并去重上下文
            for doc in parent_docs:
                if doc and doc.metadata.get("doc_id") not in all_retrieved_docs_map:
                    retrieved_doc_obj = RetrievedDocument(source_type="parent_document_from_expansion", content=doc.page_content, score=1.0, metadata=doc.metadata)
                    all_retrieved_docs_map[doc.metadata["doc_id"]] = retrieved_doc_obj

        final_context_docs_obj = list(all_retrieved_docs_map.values())
        api_logger.info(f"Collected {len(final_context_docs_obj)} unique parent documents from all sub-queries.")

        # --- 3. 基于所有收集到的上下文，生成最终答案 ---
        final_answer = NO_ANSWER_PHRASE_ANSWER_CLEAN
        failure_reason = ""
        
        if not final_context_docs_obj:
            failure_reason = "经过查询扩展后，在知识库中仍未找到任何相关文档。"
        else:
            # 检查是否有表格，并决定是否使用表格专家
            is_table_context = any('Table' in doc.metadata.get('paragraph_type', '') for doc in final_context_docs_obj)
            
            if is_table_context and len(final_context_docs_obj) == 1:
                # 如果只找到了一个表格上下文，则使用混合方法
                api_logger.info("Single table context detected. Activating Table QA Hybrid Expert.")
                table_doc = final_context_docs_obj[0]
                try:
                    df = pd.read_csv(io.StringIO(table_doc.content), sep='|', skipinitialspace=True).dropna(axis=1, how='all').iloc[1:]
                    df.columns = [col.strip() for col in df.columns]
                    index_col_name = df.columns[0]
                    df = df.set_index(index_col_name)
                    
                    instruction = await generate_table_lookup_instruction(
                        user_query=query_request.query, # 注意：这里用原始问题来生成指令
                        table_column_names=[index_col_name] + df.columns.tolist()
                    )
                    
                    if instruction:
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
            else:
                # 对于多个上下文或纯文本上下文，使用通用总结专家
                api_logger.info("Multiple/text contexts detected. Using General Summarization Expert.")
                context_strings = []
                for doc in final_context_docs_obj:
                    metadata_str = json.dumps(doc.metadata, indent=2, ensure_ascii=False)
                    context_strings.append(f"---\n### Source Document Metadata:\n```json\n{metadata_str}\n```\n\n### Document Content:\n{doc.content}\n---")
                final_context_str = "\n\n".join(context_strings)

                generated_final_answer = await generate_answer_from_context(user_query=query_request.query, context_str=final_context_str)
                if not generated_final_answer or NO_ANSWER_PHRASE_ANSWER_CLEAN in generated_final_answer:
                    failure_reason = "已检索到相关上下文，但模型无法从中提炼出明确答案。"
                else:
                    final_answer = generated_final_answer

        # 4. 如果失败，生成智能建议
        if failure_reason:
            api_logger.info(f"Answer generation failed. Reason: {failure_reason}. Generating suggestions...")
            suggestion = await generate_actionable_suggestion(user_query=query_request.query, failure_reason=failure_reason)
            if suggestion:
                final_answer = f"{NO_ANSWER_PHRASE_ANSWER_CLEAN} {suggestion}"

        # --- 整合最终结果 ---
        response_to_return = HybridRAGResponse(
            answer=final_answer, 
            original_query=query_request.query, 
            retrieved_sources=final_context_docs_obj
        )

        if not failure_reason:
            app_ctx.answer_cache[cache_key] = response_to_return
            api_logger.info(f"FINAL ANSWER CACHED for query: '{query_request.query}'")
        
    except Exception as e:
        exception_occurred = e
        api_logger.error(f"Critical error in query_rag_endpoint: {e}", exc_info=True)
        # 直接在这里构建失败的响应，而不是依赖finally块
        response_to_return = HybridRAGResponse(answer=f"An internal server error occurred: {e}", original_query=query_request.query, retrieved_sources=[])
    
    finally:
        # --- 日志记录逻辑 ---
        processing_time_seconds = (datetime.now(timezone.utc) - start_time_total).total_seconds()
        log_data_for_finally = {
            "interaction_id": interaction_id_for_log, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "task_type": "rag_query_processing_v3_3",
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
        
        # This should ideally be handled within the try block to ensure response_to_return is always set
        # But as a final fallback for the `finally` block, ensure it exists or raise an error
        if response_to_return is None:
            raise HTTPException(status_code=500, detail="Response generation failed unexpectedly after logging.")
        
        return response_to_return
    

if __name__ == "__main__":
    api_logger.info("Starting Standalone RAG API Service with Manual Small-to-Big Retrieval...")
    uvicorn.run("zhz_rag.api.rag_api_service:app", host="0.0.0.0", port=8081, reload=False)
