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
    generate_table_lookup_instruction, # <--- 确保导入这个新函数
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
    api_logger.info(f"\n--- Received RAG query (v3.2 - Expert Dispatch): '{query_request.query}' ---")
    start_time_total = datetime.now(timezone.utc)
    app_ctx: RAGAppContext = request.app.state.rag_context
    if not app_ctx:
        raise HTTPException(status_code=503, detail="RAG service is not properly initialized.")

    response_to_return: Optional[HybridRAGResponse] = None
    exception_occurred: Optional[Exception] = None
    interaction_id_for_log = str(uuid.uuid4())
    log_data_for_finally: Dict[str, Any] = {}

    try:
        # --- 缓存逻辑保持不变 ---
        cache_key = hashlib.md5(query_request.model_dump_json().encode('utf-8')).hexdigest()
        cached_response = app_ctx.answer_cache.get(cache_key)
        if cached_response is not None:
            api_logger.info(f"FINAL ANSWER CACHE HIT for query: '{query_request.query}'")
            return cached_response

        # --- START: V3.2 - 最终版专家分派逻辑 ---
        
        # 1. 生成查询计划
        api_logger.info(f"--- Step 1: Generating query plan for: '{query_request.query}' ---")
        query_plan = await generate_query_plan(user_query=query_request.query)
        search_query = query_plan.query if query_plan else query_request.query
        metadata_filter = query_plan.metadata_filter if query_plan and query_plan.metadata_filter else {}

        # --- V2: 规范化并简化元数据过滤器 ---
        api_logger.info(f"Normalizing filter: {metadata_filter}")

        # 步骤a: 规范化 - 如果一个字典里有多个键，自动用$and包装
        if isinstance(metadata_filter, dict) and len(metadata_filter) > 1 and not metadata_filter.keys() & {'$and', '$or', '$not'}:
            # 只有在没有逻辑操作符的情况下，才进行包装
            new_filter_list = [{key: value} for key, value in metadata_filter.items()]
            metadata_filter = {"$and": new_filter_list}
            api_logger.info(f"Wrapped multiple conditions into $and: {metadata_filter}")

        # 步骤b: 净化 - 移除$and/$or列表中可能存在的空字典
        if isinstance(metadata_filter, dict):
            for op in ["$and", "$or"]:
                if op in metadata_filter and isinstance(metadata_filter[op], list):
                    cleaned_list = [item for item in metadata_filter[op] if item and isinstance(item, dict)]
                    if cleaned_list:
                        metadata_filter[op] = cleaned_list
                    else:
                        metadata_filter.pop(op)
            if not metadata_filter:
                metadata_filter = {}

        # 步骤c: 简化 - 如果$and/$or中只剩一个元素，则剥离包装
        if metadata_filter and ("$and" in metadata_filter) and len(metadata_filter["$and"]) == 1:
            metadata_filter = metadata_filter["$and"][0]
            api_logger.info(f"Simplified single-element $and filter to: {metadata_filter}")
        elif metadata_filter and ("$or" in metadata_filter) and len(metadata_filter["$or"]) == 1:
            metadata_filter = metadata_filter["$or"][0]
            api_logger.info(f"Simplified single-element $or filter to: {metadata_filter}")
        
        api_logger.info(f"Final normalized filter: {metadata_filter}")
        # --- 规范化与简化结束 ---

        # 3. 检索小块
        api_logger.info(f"--- Step 2: Retrieving child chunks with plan ---")
        retrieved_child_chunks = await app_ctx.chroma_retriever.retrieve(
            query_text=search_query,
            n_results=query_request.top_k_vector,
            where_filter=metadata_filter if metadata_filter else None
        )
        api_logger.info(f"Retrieved {len(retrieved_child_chunks)} child chunks.")

        final_context_str = ""
        final_context_docs_obj: List[RetrievedDocument] = []
        prompt_builder_to_use = get_answer_generation_messages # 默认为通用专家

        # 4. 专家分派逻辑 (V3.3 - Hybrid Approach Implemented)
        final_answer = NO_ANSWER_PHRASE_ANSWER_CLEAN
        final_context_docs_obj: List[RetrievedDocument] = []

        if retrieved_child_chunks:
            top_chunk = retrieved_child_chunks[0]
            top_chunk_meta = top_chunk.get('metadata', {})
            top_chunk_type = top_chunk_meta.get('paragraph_type')
            top_chunk_content = top_chunk.get('document', '')

            # --- 表格专家路径 (Hybrid: LLM for Instruction, Code for Execution) ---
            if top_chunk_type and 'Table' in top_chunk_type and top_chunk_content:
                api_logger.info(f"EXPERT DISPATCH: Top chunk is a '{top_chunk_type}'. Activating Table QA Hybrid Expert.")
                
                # --- START: 修改点 ---
                metadata_str = json.dumps(top_chunk_meta, indent=2, ensure_ascii=False)
                final_context_str = (
                    f"---\n"
                    f"### Source Document Metadata:\n"
                    f"```json\n{metadata_str}\n```\n\n"
                    f"### Table Content (Markdown):\n{top_chunk_content}\n"
                    f"---"
                )
                # --- END: 修改点 ---
                
                # a. 将表格Markdown加载到DataFrame
                try:
                    df = pd.read_csv(io.StringIO(top_chunk_content), sep='|', skipinitialspace=True).dropna(axis=1, how='all').iloc[1:]
                    df.columns = [col.strip() for col in df.columns]
                    # 获取第一列的列名作为索引名
                    index_col_name = df.columns[0]
                    df = df.set_index(index_col_name)
                    api_logger.info(f"Successfully loaded table into DataFrame. Index: '{index_col_name}', Columns: {df.columns.tolist()}")

                    # b. 调用LLM生成查找指令
                    instruction = await generate_table_lookup_instruction(
                        user_query=query_request.query,
                        table_column_names=[index_col_name] + df.columns.tolist() # 将索引名也加入列名列表
                    )

                    # c. 使用代码执行指令
                    if instruction:
                        row_id = instruction.get("row_identifier")
                        col_id = instruction.get("column_identifier")
                        api_logger.info(f"Executing instruction: Find row='{row_id}', column='{col_id}'")
                        
                        # d. 在DataFrame中精确查找
                        if row_id in df.index and col_id in df.columns:
                            value = df.at[row_id, col_id]
                            final_answer = f"根据查找到的表格信息，{row_id}的{col_id}是{value}。"
                            api_logger.info(f"SUCCESS: Found value '{value}' at ('{row_id}', '{col_id}').")
                        else:
                            api_logger.warning(f"Instruction execution failed: Row '{row_id}' or Column '{col_id}' not found in DataFrame.")
                            final_answer = f"我在表格中找到了相关信息，但无法精确定位到'{row_id}'的'{col_id}'。"

                    else:
                        api_logger.warning("LLM failed to generate a valid table lookup instruction. Falling back to text summary.")
                        # 如果指令生成失败，可以回退到通用总结模式
                        final_context_str = top_chunk_content
                        generated_final_answer = await generate_answer_from_context(
                            user_query=query_request.query, 
                            context_str=final_context_str,
                            prompt_builder=get_answer_generation_messages
                        )
                        final_answer = generated_final_answer if generated_final_answer else NO_ANSWER_PHRASE_ANSWER_CLEAN

                    # 记录用于生成答案的上下文（即这个表格本身）
                    final_context_docs_obj = [RetrievedDocument(
                        source_type="hybrid_table_qa_execution",
                        content=top_chunk_content,
                        score=top_chunk.get('distance', 1.0),
                        metadata=top_chunk_meta
                    )]

                except Exception as e_pandas:
                    api_logger.error(f"Error during Hybrid Table QA execution: {e_pandas}", exc_info=True)
                    final_answer = "处理表格数据时遇到错误。"

            # --- 通用专家路径 (小块 -> 大块) ---
            else:
                api_logger.info(f"EXPERT DISPATCH: Top chunk is '{top_chunk_type}'. Using General Context (Small-to-Big) Expert.")
                parent_ids = list(set(
                    chunk['metadata']['parent_id'] 
                    for chunk in retrieved_child_chunks if chunk.get('metadata') and chunk['metadata'].get('parent_id')
                ))
                parent_docs = app_ctx.docstore.mget(parent_ids)

                # --- START: 修改点 ---
                context_strings = []
                for doc in final_context_docs_obj:
                    # 将元数据字典转换为格式化的JSON字符串，以便LLM阅读
                    metadata_str = json.dumps(doc.metadata, indent=2, ensure_ascii=False)
                    context_strings.append(
                        f"---\n"
                        f"### Source Document Metadata:\n"
                        f"```json\n{metadata_str}\n```\n\n"
                        f"### Document Content:\n{doc.content}\n"
                        f"---"
                    )
                final_context_str = "\n\n".join(context_strings)
                # --- END: 修改点 ---

                final_context_docs_obj = [
                    RetrievedDocument(
                        source_type="parent_document_retrieval",
                        content=doc.page_content,
                        score=1.0,
                        metadata=doc.metadata
                    ) for doc in parent_docs if doc is not None
                ]
                context_strings = [f"Source: {doc.metadata.get('filename', 'N/A')}\nContent:\n{doc.content}" for doc in final_context_docs_obj]
                final_context_str = "\n\n---\n\n".join(context_strings)
                
                if not final_context_str:
                    final_answer = NO_ANSWER_PHRASE_ANSWER_CLEAN
                else:
                    generated_final_answer = await generate_answer_from_context(
                        user_query=query_request.query, 
                        context_str=final_context_str,
                        prompt_builder=get_answer_generation_messages
                    )
                    final_answer = generated_final_answer if generated_final_answer else NO_ANSWER_PHRASE_ANSWER_CLEAN
        
        # --- 整合最终结果 ---
        response_to_return = HybridRAGResponse(
            answer=final_answer, 
            original_query=query_request.query, 
            retrieved_sources=final_context_docs_obj
        )

        response_to_return = HybridRAGResponse(
            answer=final_answer, 
            original_query=query_request.query, 
            retrieved_sources=final_context_docs_obj
        )
        
        if final_answer != NO_ANSWER_PHRASE_ANSWER_CLEAN:
            app_ctx.answer_cache[cache_key] = response_to_return
            api_logger.info(f"FINAL ANSWER CACHED for query: '{query_request.query}'")
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
