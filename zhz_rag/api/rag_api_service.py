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
from dataclasses import dataclass, field


# --- .env 文件加载 ---
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_dir = os.path.abspath(os.path.join(_current_file_dir, "..", ".."))
_dotenv_path = os.path.join(_project_root_dir, ".env")
if os.path.exists(_dotenv_path):
    load_dotenv(dotenv_path=_dotenv_path)
else:
    load_dotenv()

# --- 导入我们自己的模块 ---
from zhz_rag.config.pydantic_models import QueryRequest, HybridRAGResponse, RetrievedDocument
from zhz_rag.llm.llm_interface import (
    generate_answer_from_context,
    generate_query_plan,
    generate_table_lookup_instruction,
    generate_actionable_suggestion,
    generate_expanded_queries,
    generate_document_summary, 
    NO_ANSWER_PHRASE_ANSWER_CLEAN
)
from zhz_rag.utils.hardware_manager import HardwareManager

from zhz_rag.llm.rag_prompts import get_answer_generation_messages, get_table_qa_messages, get_fusion_messages
from zhz_rag.core_rag.retrievers.chromadb_retriever import ChromaDBRetriever
from zhz_rag.core_rag.retrievers.file_bm25_retriever import FileBM25Retriever
from zhz_rag.core_rag.retrievers.embedding_functions import LlamaCppEmbeddingFunction
from zhz_rag.llm.local_model_handler import LlamaCppEmbeddingFunction as LocalModelHandlerWrapper
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.resources import GGUFEmbeddingResource
from zhz_rag.utils.interaction_logger import log_interaction_data


from enum import Enum
from zhz_rag.core_rag.fusion_engine import FusionEngine 
class HardwareTier(Enum):
    HIGH = "HIGH"
    MID = "MID"
    LOW = "LOW"

def _get_hardware_tier(hw_manager: HardwareManager) -> HardwareTier:
    """根据硬件信息判断硬件等级。"""
    hw_info = hw_manager.get_hardware_info()
    if not hw_info:
        return HardwareTier.LOW
    
    # 简化版策略：VRAM > 10GB 认为是高配，VRAM > 6GB认为是中配
    if hw_info.gpu_available and hw_info.gpu_vram_total_gb > 10:
        return HardwareTier.HIGH
    elif hw_info.gpu_available and hw_info.gpu_vram_total_gb > 6:
        return HardwareTier.MID
    else:
        return HardwareTier.LOW


# --- 日志配置 ---
api_logger = logging.getLogger("RAGApiServiceLogger")
api_logger.setLevel(logging.INFO)
if not api_logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
    handler.setFormatter(handler)
    api_logger.addHandler(handler)
    api_logger.propagate = False

# --- 生产者-消费者队列 ---
log_queue = asyncio.Queue()

async def log_writer_task():
    """
    后台任务，用于从队列中获取日志条目并写入交互数据。
    """
    while True:
        try:
            log_entry_to_write = await log_queue.get()
            await log_interaction_data(log_entry_to_write) 
            log_queue.task_done()
        except Exception as e:
            api_logger.error(f"Critical error in log_writer_task: {e}", exc_info=True)

# --- 应用上下文 Dataclass ---
@dataclass
class RAGAppContext:
    """
    RAG应用程序的上下文，包含所有核心组件的实例。
    """
    chroma_retriever: ChromaDBRetriever 
    bm25_retriever: Optional[FileBM25Retriever]
    docstore: InMemoryStore
    gguf_embedding_resource: GGUFEmbeddingResource
    answer_cache: TTLCache
    llm_gbnf_instance: Any
    fusion_engine: FusionEngine
    hardware_tier: HardwareTier


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI应用的生命周期管理器。
    负责在应用启动时初始化RAG组件，并在应用关闭时清理资源。
    """
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
            model_total_layers = 28 # Qwen3-1.7B
            
            n_gpu_layers = hal.recommend_llm_gpu_layers(
                model_total_layers=model_total_layers,
                model_size_on_disk_gb=model_size_gb
            )
            
            from llama_cpp import Llama
            gbnf_llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=int(os.getenv("LLM_N_CTX", 4096)),
                verbose=False
            )
        except Exception as e:
            api_logger.critical(f"FATAL: Failed to pre-load GBNF LLM model: {e}", exc_info=True)
            gbnf_llm = None

    # --- 3. 初始化其他服务 ---
    embedding_api_url = os.getenv("EMBEDDING_API_URL", "http://127.0.0.1:8089")
    chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME", "zhz_rag_collection")
    bm25_index_dir = os.getenv("BM25_INDEX_DIRECTORY")


    class FakeDagsterContext:
        def __init__(self, logger_instance): self.log = logger_instance
    
    gguf_embed_resource = GGUFEmbeddingResource(api_url=embedding_api_url)
    await asyncio.to_thread(gguf_embed_resource.setup_for_execution, FakeDagsterContext(api_logger))
    model_handler = LocalModelHandlerWrapper(resource=gguf_embed_resource)
    chroma_embedding_function = LlamaCppEmbeddingFunction(model_handler=model_handler)
    
    try:
        # --- 初始化 ChromaDB ---
        chroma_retriever_instance = ChromaDBRetriever(
            collection_name=chroma_collection_name,
            persist_directory=chroma_persist_dir,
            embedding_function=chroma_embedding_function
        )

        # --- 初始化 BM25 检索器 ---
        bm25_retriever_instance = None
        if bm25_index_dir and os.path.isdir(bm25_index_dir):
            try:
                bm25_retriever_instance = FileBM25Retriever(index_directory=bm25_index_dir)
            except Exception as e:
                api_logger.error(f"Failed to initialize FileBM25Retriever: {e}", exc_info=True)
                bm25_retriever_instance = None 
        else:
            api_logger.warning(f"BM25_INDEX_DIRECTORY not set or is not a valid directory. BM25 search will be disabled.")


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
                            parent_docs_map[parent_id] = {"metadata": metadata, "texts": []}
                        parent_docs_map[parent_id]["texts"].append(all_chunks_from_db['documents'][i])
                
                docs_to_store_in_docstore = [
                    LangchainDocument(
                        page_content="\n\n".join(sorted(data["texts"])), 
                        metadata={**data["metadata"], "doc_id": parent_id}
                    ) for parent_id, data in parent_docs_map.items()
                ]
                docstore.mset([(doc.metadata["doc_id"], doc) for doc in docs_to_store_in_docstore])
            else:
                api_logger.warning(f"ChromaDB collection '{chroma_collection_name}' is empty. Docstore will be empty.")

        except Exception as e:
            api_logger.error(f"Failed to build docstore during startup: {e}", exc_info=True)
            docstore = InMemoryStore() 
            
        # --- 硬件检测与策略决策 ---
        hw_manager = HardwareManager()
        tier = _get_hardware_tier(hw_manager)

        # 根据硬件等级决定是否启用再排序器
        enable_reranker_flag = (tier == HardwareTier.HIGH)

        # --- 初始化 FusionEngine (现在带有策略) ---
        fusion_engine_instance = FusionEngine(
            logger=api_logger, 
            enable_reranker=enable_reranker_flag
        )

        app.state.rag_context = RAGAppContext(
            chroma_retriever=chroma_retriever_instance,
            bm25_retriever=bm25_retriever_instance,
            docstore=docstore,
            gguf_embedding_resource=gguf_embed_resource,
            answer_cache=TTLCache(maxsize=100, ttl=900),
            llm_gbnf_instance=gbnf_llm,
            fusion_engine=fusion_engine_instance,
            hardware_tier=tier
        )
        
        asyncio.create_task(log_writer_task())
    except Exception as e:
        api_logger.critical(f"FATAL: Failed to initialize RAG components: {e}", exc_info=True)
        app.state.rag_context = None
    
    yield
    
    api_logger.info("--- RAG API Service: Cleaning up resources ---")
    
    if hasattr(app.state, 'rag_context') and app.state.rag_context and app.state.rag_context.gguf_embedding_resource:
        if hasattr(app.state.rag_context.gguf_embedding_resource, 'teardown_for_execution'):
            class FakeDagsterContextTeardown:
                def __init__(self, logger_instance):
                    self.log = logger_instance
            fake_dagster_context_teardown = FakeDagsterContextTeardown(api_logger)
            await asyncio.to_thread(app.state.rag_context.gguf_embedding_resource.teardown_for_execution, fake_dagster_context_teardown)
        else:
            api_logger.warning("GGUFEmbeddingResource does not have a teardown_for_execution method.")
    else:
        api_logger.warning("No RAGAppContext or GGUFEmbeddingResource found for teardown.")

    if hasattr(app.state, 'rag_context') and app.state.rag_context and app.state.rag_context.llm_gbnf_instance:
        app.state.rag_context.llm_gbnf_instance = None


def _fuse_results_rrf(
    vector_results: List[List[Dict[str, Any]]], 
    keyword_results: List[List[Dict[str, Any]]], 
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    使用倒数排序融合（Reciprocal Rank Fusion - RRF）来合并向量和关键词搜索的结果。
    
    Args:
        vector_results: 来自向量检索器的结果列表（每个子查询一个列表）。
        keyword_results: 来自关键词检索器的结果列表（每个子查询一个列表）。
        k: RRF算法中的排名常数，用于降低低排名结果的权重。

    Returns:
        一个融合、去重并按RRF分数重新排序的文档块列表。
    """
    # 步骤1: 将来自多个子查询的结果平铺成两个总的排名列表
    flat_vector_results = [chunk for sublist in vector_results for chunk in sublist]
    flat_keyword_results = [chunk for sublist in keyword_results for chunk in sublist]
    
    scores = {}
    all_docs_map = {}

    # 步骤2: 计算RRF分数
    # 处理向量搜索结果
    for rank, doc in enumerate(flat_vector_results):
        doc_id = doc.get("id")
        if not doc_id: continue
        scores[doc_id] = 1 / (k + rank + 1)
        all_docs_map[doc_id] = doc

    # 处理关键词搜索结果
    for rank, doc in enumerate(flat_keyword_results):
        doc_id = doc.get("id")
        if not doc_id: continue
        if doc_id not in scores:
            scores[doc_id] = 0
        scores[doc_id] += 1 / (k + rank + 1)
        all_docs_map.setdefault(doc_id, doc) 
    
    # 步骤3: 按RRF分数降序排序
    sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    # 步骤4: 构建最终的融合结果列表
    fused_results = [all_docs_map[doc_id] for doc_id in sorted_doc_ids]

    return fused_results


# --- FastAPI 应用实例 ---
app = FastAPI(
    title="Advanced RAG API Service with Manual Small-to-Big Retrieval",
    description="Provides API access to the RAG framework, now with manual small-to-big retrieval.",
    version="3.1.0",
    lifespan=lifespan
)


@app.post("/api/v1/rag/query", response_model=HybridRAGResponse)
async def query_rag_endpoint(request: Request, query_request: QueryRequest):
    """
    处理RAG查询请求，执行混合检索、文档重排和答案生成。

    Args:
        request: FastAPI请求对象。
        query_request: 包含用户查询和检索参数的请求体。

    Returns:
        HybridRAGResponse: 包含生成的答案和检索到的源文档。
    """
    start_time_total = datetime.now(timezone.utc)
    app_ctx: RAGAppContext = request.app.state.rag_context
    if not app_ctx or not app_ctx.llm_gbnf_instance or not app_ctx.fusion_engine:
        raise HTTPException(status_code=503, detail="RAG service or its core components are not initialized.")

    interaction_id_for_log = str(uuid.uuid4())
    exception_occurred: Optional[Exception] = None
    response_to_return: Optional[HybridRAGResponse] = None

    try:
        # 缓存逻辑
        cache_key = hashlib.md5(query_request.model_dump_json().encode('utf-8')).hexdigest()
        cached_response = app_ctx.answer_cache.get(cache_key)
        if cached_response is not None:
            return cached_response

        # 1. 查询规划与扩展
        query_plan = await generate_query_plan(app_ctx.llm_gbnf_instance, query_request.query)
        
        # --- 智能查询扩展策略 ---
        # 只有在查询计划没有生成具体的元数据过滤器时，才执行查询扩展。
        # 这可以防止在精确查找（如表格问答）时，通用扩展查询干扰检索结果。
        if not query_plan.metadata_filter:
            api_logger.info("No specific metadata filter found in plan. Performing query expansion.")
            sub_queries = await generate_expanded_queries(app_ctx.llm_gbnf_instance, query_request.query)
            unique_queries = list(dict.fromkeys([query_plan.query] + sub_queries))
        else:
            api_logger.info("Metadata filter found in plan. Skipping query expansion to ensure precision.")
            unique_queries = [query_plan.query]

            
        # --- 构建元数据过滤器 ---
        user_filter_conditions = query_request.filters.get("must", []) if query_request.filters else []
        llm_filter = query_plan.metadata_filter if query_plan else {}
        all_conditions = []
        for condition in user_filter_conditions:
            key = condition.get("key")
            match = condition.get("match")
            if key and match and "value" in match:
                all_conditions.append({key: {"$eq": match["value"]}})
        if llm_filter:
            if "$and" in llm_filter:
                all_conditions.extend(llm_filter["$and"])
            elif llm_filter:
                all_conditions.append(llm_filter)
        final_metadata_filter = {}
        if all_conditions:
            unique_conditions_as_strings = {json.dumps(c, sort_keys=True) for c in all_conditions}
            unique_conditions = [json.loads(s) for s in unique_conditions_as_strings]
            if len(unique_conditions) == 1:
                final_metadata_filter = unique_conditions[0]
            else:
                final_metadata_filter = {"$and": unique_conditions}

        # --- 并行执行向量和关键词检索 ---
        vector_retrieval_tasks = [
            app_ctx.chroma_retriever.retrieve(
                query_text=q,
                n_results=query_request.top_k_vector,
                where_filter=final_metadata_filter or None
            ) for q in unique_queries
        ]
        
        keyword_retrieval_tasks = []
        if app_ctx.bm25_retriever:
            keyword_retrieval_tasks = [
                app_ctx.bm25_retriever.retrieve(
                    query_text=q,
                    n_results=query_request.top_k_bm25
                ) for q in unique_queries
            ]
        else:
            pass

        # 等待所有检索任务完成
        all_retrieval_results = await asyncio.gather(*(vector_retrieval_tasks + keyword_retrieval_tasks))
        
        # 分离结果
        vector_results = all_retrieval_results[:len(vector_retrieval_tasks)]
        keyword_results = all_retrieval_results[len(vector_retrieval_tasks):]
        
        # --- 使用RRF融合结果 ---
        fused_child_chunks = _fuse_results_rrf(vector_results, keyword_results)
        
        # --- 从融合后的子块中提取父文档 ---
        all_parent_ids = {chunk['metadata']['parent_id'] for chunk in fused_child_chunks if 'parent_id' in chunk.get('metadata', {})}
        parent_docs = app_ctx.docstore.mget(list(all_parent_ids))
        valid_parent_docs = [doc for doc in parent_docs if doc]

        # --- 后续步骤（重排、生成答案等）保持不变，但使用融合后的结果 ---
        # Stage 3: Reranking
        reranked_docs = await app_ctx.fusion_engine.rerank_documents(
            query=query_request.query,
            documents=[RetrievedDocument(content=doc.page_content, metadata=doc.metadata, score=0.0, source_type="fused_retrieval") for doc in valid_parent_docs],
            top_n=5
        )

        # --- Stage 4: Step-by-Step Synthesis ---
        final_answer = NO_ANSWER_PHRASE_ANSWER_CLEAN
        failure_reason = ""
        
        if not reranked_docs:
            failure_reason = "知识库中未能找到任何与您问题相关的信息。"
        else:
            is_table_query_top_ranked = (reranked_docs and reranked_docs[0].metadata.get("paragraph_type") == "table")
            
            if is_table_query_top_ranked:
                table_doc = reranked_docs[0]
                try:
                    df = pd.read_csv(io.StringIO(table_doc.content), sep='|', skipinitialspace=True).dropna(axis=1, how='all').iloc[1:]
                    df.columns = [col.replace(" ", "").strip() for col in df.columns]
                    # 清洗所有单元格数据中的空格
                    for col in df.columns:
                        if df[col].dtype == 'object': 
                            df[col] = df[col].str.strip()

                    if len(df.columns) < 2:
                        raise ValueError("Table must have at least two columns for key-value lookup.")
                    
                    key_column_for_lookup = df.columns[1]
                    instruction = await generate_table_lookup_instruction(
                        llm_instance=app_ctx.llm_gbnf_instance,
                        user_query=query_request.query,
                        table_column_names=df.columns.tolist()
                    )

                    if instruction and "row_identifier" in instruction and "column_identifier" in instruction:
                        row_id_raw = instruction.get("row_identifier", "")
                        col_id_raw = instruction.get("column_identifier", "")

                        # 尝试完全匹配
                        result_series = df.loc[df[key_column_for_lookup] == row_id_raw, col_id_raw]

                        # 如果完全匹配失败，尝试忽略大小写和空格进行模糊匹配
                        if result_series.empty:
                            match_condition = df[key_column_for_lookup].str.strip().str.lower().str.contains(row_id_raw.strip().lower(), na=False)
                            result_series = df.loc[match_condition, col_id_raw]

                        if not result_series.empty:
                            value = result_series.iloc[0]
                            final_answer = f"根据查找到的表格信息，{row_id_raw}的{col_id_raw}是{value}。"
                        else:
                             failure_reason = f"模型指令无法执行：在表格的'{key_column_for_lookup}'列中未能找到行'{row_id_raw}'，或在表头中未能找到列'{col_id_raw}'。"
                    else:
                        failure_reason = "模型未能从问题中生成有效的表格查询指令。"
                except Exception as e_pandas:
                    failure_reason = f"处理表格数据时遇到代码错误: {e_pandas}"
                
                if failure_reason:
                    pass

            if final_answer == NO_ANSWER_PHRASE_ANSWER_CLEAN:
                summary_tasks = [generate_document_summary(app_ctx.llm_gbnf_instance, user_query=query_request.query, document_content=doc.content) for doc in reranked_docs]
                summaries = await asyncio.gather(*summary_tasks)
                
                relevant_summaries = []
                for doc, summary in zip(reranked_docs, summaries):
                    if summary:
                        filename = doc.metadata.get('filename', '未知文档')
                        relevant_summaries.append(f"根据文档《{filename}》的信息：{summary}")
                
                if not relevant_summaries:
                    failure_reason = "虽然检索到了相关文档，但无法从中提炼出与您问题直接相关的核心信息。"
                else:
                    fusion_context = "\n\n".join(relevant_summaries)
                    final_answer = await generate_answer_from_context(user_query=query_request.query, context_str=fusion_context, prompt_builder=lambda q, c: get_fusion_messages(q, c))
            
        if not final_answer or NO_ANSWER_PHRASE_ANSWER_CLEAN in final_answer:
            if not failure_reason: failure_reason = "根据检索到的上下文信息，无法直接回答您的问题。"
            suggestion = await generate_actionable_suggestion(app_ctx.llm_gbnf_instance, user_query=query_request.query, failure_reason=failure_reason)
            final_answer = f"{failure_reason} {suggestion}" if suggestion else failure_reason

        response_to_return = HybridRAGResponse(answer=final_answer, original_query=query_request.query, retrieved_sources=reranked_docs)
        if not failure_reason and NO_ANSWER_PHRASE_ANSWER_CLEAN not in final_answer:
            app_ctx.answer_cache[cache_key] = response_to_return
    except Exception as e:
        exception_occurred = e
        response_to_return = HybridRAGResponse(answer=f"An internal server error occurred: {e}", original_query=query_request.query, retrieved_sources=[])
    finally:
        processing_time_seconds = (datetime.now(timezone.utc) - start_time_total).total_seconds()
        log_data_for_finally = {
            "interaction_id": interaction_id_for_log, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "task_type": "rag_query_processing_v8_0_hybrid_search",
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
    uvicorn.run("zhz_rag.api.rag_api_service:app", host="0.0.0.0", port=8081, reload=False)

