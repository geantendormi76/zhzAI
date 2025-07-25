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
import re

# --- .env 文件加载 ---
_current_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root_dir = os.path.abspath(os.path.join(_current_file_dir, "..", ".."))
_dotenv_path = os.path.join(_project_root_dir, ".env")
if os.path.exists(_dotenv_path):
    load_dotenv(dotenv_path=_dotenv_path)
else:
    load_dotenv()

# --- 导入我们自己的模块 ---
from zhz_rag.config.pydantic_models import (
    QueryRequest,
    HybridRAGResponse,
    RetrievedDocument,
    PendingTaskSuggestion,
    AutoScheduledTaskConfirmation,
    UserIntent,                     # <--- 新增导入
    IntentType                      # <--- 新增导入
)
from zhz_rag.llm.llm_interface import (
    generate_answer_from_context,
    generate_query_plan,
    generate_table_lookup_instruction,
    generate_actionable_suggestion,
    generate_expanded_queries,
    generate_document_summary,
    extract_task_from_dialogue,
    classify_user_intent,           # <--- 新增导入
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
    V5.2: Handles RAG queries with a robust Table QA dispatch mechanism and result hydration.
    处理RAG查询请求，执行混合检索、文档重排和答案生成，并根据用户意图进行分派。

    Args:
        request: FastAPI请求对象。
        query_request: 包含用户查询和检索参数的请求体。

    Returns:
        HybridRAGResponse: 包含生成的答案和检索到的源文档，以及可能的行动建议。
    """
    start_time_total = datetime.now(timezone.utc)
    app_ctx: RAGAppContext = request.app.state.rag_context
    if not app_ctx or not app_ctx.llm_gbnf_instance:
        raise HTTPException(status_code=503, detail="RAG service or its core components are not initialized.")

    interaction_id_for_log = str(uuid.uuid4())
    user_intent: Optional[UserIntent] = None # 初始化用户意图

    try:
        # --- 1. 意图分类 ---
        # 首先对用户查询进行意图分类，判断是纯粹的任务创建、RAG问答还是混合意图
        user_intent = await classify_user_intent(app_ctx.llm_gbnf_instance, query_request.query)

        if not user_intent:
            # 如果意图分类失败，降级为标准的RAG查询流程
            api_logger.warning("Intent classification failed. Defaulting to RAG_QUERY intent.")
            user_intent = UserIntent(intent=IntentType.RAG_QUERY, reasoning="Classification failed, fallback.")

        # --- 2. 意图分派 ---
        
        # 场景 A: 纯任务创建
        # 如果意图是纯任务创建，则跳过所有RAG步骤，直接进行任务提取
        if user_intent.intent == IntentType.TASK_CREATION:
            api_logger.info("Intent classified as TASK_CREATION. Skipping RAG.")
            # 直接提取任务，不进行RAG
            task_info = await extract_task_from_dialogue(app_ctx.llm_gbnf_instance, query_request.query, "")
            if task_info and task_info.get("task_found"):
                # TODO: 未来这里应该调用task_manager API来实际创建任务
                # 目前，我们只返回一个确认信息
                return HybridRAGResponse(
                    original_query=query_request.query,
                    answer=f"好的，已为您记录待办事项：‘{task_info.get('title')}’，截止日期为 {task_info.get('due_date')}。",
                    retrieved_sources=[],
                    actionable_suggestion=None # 因为已经直接处理了
                )
            else:
                return HybridRAGResponse(
                    original_query=query_request.query,
                    answer="抱歉，我理解您想创建一个任务，但未能成功提取任务信息。",
                    retrieved_sources=[],
                    actionable_suggestion=None
                )

        # 场景 B & C: RAG问答 或 混合意图
        # 对于这两种意图，我们都需要执行完整的RAG流程
        api_logger.info(f"Intent classified as {user_intent.intent.value}. Executing full RAG pipeline.")
        
        final_answer = NO_ANSWER_PHRASE_ANSWER_CLEAN
        failure_reason = ""
        reranked_docs = []
        exception_occurred: Optional[Exception] = None
        response_to_return: Optional[HybridRAGResponse] = None

        # 缓存逻辑
        cache_key = hashlib.md5(query_request.model_dump_json().encode('utf-8')).hexdigest()
        if (cached_response := app_ctx.answer_cache.get(cache_key)) is not None:
            return cached_response

        # --- RAG 核心流程开始 ---
        # 1. 查询规划与扩展
        query_plan = await generate_query_plan(app_ctx.llm_gbnf_instance, query_request.query)
        
        # 智能查询扩展策略: 只有在查询计划没有生成具体的元数据过滤器时，才执行查询扩展。
        # 这可以防止在精确查找（如表格问答）时，通用扩展查询干扰检索结果。
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
            pass # BM25检索器未初始化，跳过关键词检索

        # 等待所有检索任务完成
        all_retrieval_results = await asyncio.gather(*(vector_retrieval_tasks + keyword_retrieval_tasks))
        
        # 分离结果
        vector_results_nested = all_retrieval_results[:len(vector_retrieval_tasks)]
        keyword_results_nested = all_retrieval_results[len(vector_retrieval_tasks):]
        
        # --- 【【【新增的核心修改：结果填充】】】 ---
        # 在融合前，为所有仅有ID的检索结果（主要是BM25）从ChromaDB中填充完整的content和metadata
        all_retrieved_ids = set()
        for sublist in keyword_results_nested:
            for chunk in sublist:
                all_retrieved_ids.add(chunk['id'])
        # 向量检索的结果也可能不完整，也需要收集ID进行填充
        for sublist in vector_results_nested:
            for chunk in sublist:
                all_retrieved_ids.add(chunk['id'])

        if all_retrieved_ids:
            api_logger.info(f"Hydrating {len(all_retrieved_ids)} unique chunk IDs from ChromaDB...")
            hydrated_docs_map = {}
            # 从ChromaDB批量获取所有需要的数据
            chroma_get_results = app_ctx.chroma_retriever._collection.get(
                ids=list(all_retrieved_ids),
                include=["metadatas", "documents"]
            )
            for i, doc_id in enumerate(chroma_get_results['ids']):
                hydrated_docs_map[doc_id] = {
                    "content": chroma_get_results['documents'][i],
                    "metadata": chroma_get_results['metadatas'][i]
                }
            
            # 填充 BM25 结果
            for sublist in keyword_results_nested:
                for chunk in sublist:
                    if chunk['id'] in hydrated_docs_map:
                        chunk.update(hydrated_docs_map[chunk['id']])
            
            # 填充 Vector 结果 (以防万一它也没有完整数据)
            for sublist in vector_results_nested:
                for chunk in sublist:
                    if chunk['id'] in hydrated_docs_map:
                        chunk.update(hydrated_docs_map[chunk['id']])
        # --- 【【【新增结束】】】 ---


        # --- 使用RRF融合结果 ---
        fused_child_chunks = _fuse_results_rrf(vector_results_nested, keyword_results_nested)
        
        # --- 智能问答分派 ---
        
        # 3.1 直接从融合后的子块中寻找表格
        table_chunk_candidate = None
        for chunk in fused_child_chunks:
            # 检查元数据中是否有 'table' 类型的标记
            if chunk.get('metadata', {}).get('paragraph_type') == 'table':
                table_chunk_candidate = chunk
                api_logger.info(f"Found a table chunk candidate with ID {chunk.get('id')} from fused results.")
                break # 找到第一个就优先处理

        # 3.2 如果找到表格块，则优先执行表格问答流程
        if table_chunk_candidate:
            api_logger.info("Prioritizing Table QA based on a directly retrieved table chunk.")
            # 将这个表格块作为我们唯一的上下文来源
            table_doc = RetrievedDocument(
                source_type=table_chunk_candidate.get("source_type", "table_chunk"),
                content=table_chunk_candidate.get("content", ""),
                score=table_chunk_candidate.get("score", 1.0), # 既然选中了，给个高分
                metadata=table_chunk_candidate.get("metadata", {})
            )
            reranked_docs = [table_doc]

            try:
                # 表格内容预处理 (逻辑不变)
                table_content_lines = table_doc.content.strip().split('\n')
                processed_lines = [line for line in table_content_lines if not re.match(r'^\s*\|?(:?-+:?\|)+(:?-+:?)?\s*$', line)]
                processed_table_content = "\n".join(processed_lines)
                
                api_logger.info(f"Preprocessed table content for pandas:\n{processed_table_content}")

                df = pd.read_csv(io.StringIO(processed_table_content), sep='|', skipinitialspace=True).dropna(axis=1, how='all')
                
                df.columns = [str(col).strip() for col in df.iloc[0]]
                df = df[1:].reset_index(drop=True)

                for col in df.columns:
                    if df[col].dtype == 'object': 
                        df[col] = df[col].str.strip()

                if len(df.columns) < 2: 
                    raise ValueError("Table for QA must have at least two columns after processing.")
                
                key_column_for_lookup = df.columns[0]
                api_logger.info(f"Using '{key_column_for_lookup}' as the key column for table lookup.")
                
                # --- 【【核心修复】】在进行字符串操作前，强制转换类型 ---
                # 确保用于模糊匹配的键列是字符串类型，避免AttributeError
                df[key_column_for_lookup] = df[key_column_for_lookup].astype(str)
                # --- 【【修复结束】】 ---
                
                instruction = await generate_table_lookup_instruction(
                    llm_instance=app_ctx.llm_gbnf_instance,
                    user_query=query_request.query,
                    table_column_names=df.columns.tolist()
                )

                if instruction and "row_identifier" in instruction and "column_identifier" in instruction:
                    row_id_raw = instruction.get("row_identifier", "")
                    col_id_raw = instruction.get("column_identifier", "")
                    
                    target_column = next((c for c in df.columns if col_id_raw.lower() in c.lower() or c.lower() in col_id_raw.lower()), None)
                    if not target_column: 
                        raise ValueError(f"Column '{col_id_raw}' not found in table.")

                    # 现在可以安全地使用 .str.contains
                    match_condition = df[key_column_for_lookup].str.contains(row_id_raw, case=False, na=False)
                    result_series = df.loc[match_condition, target_column]

                    if not result_series.empty:
                        value = result_series.iloc[0]
                        matched_row_name = df.loc[match_condition, key_column_for_lookup].iloc[0]
                        final_answer = f"根据查找到的表格信息，'{matched_row_name}'的'{target_column}'是'{value}'。"
                    else:
                        failure_reason = f"在表格的'{key_column_for_lookup}'列中未能找到与'{row_id_raw}'相关的行。"
                else:
                    failure_reason = "模型未能从问题中生成有效的表格查询指令。"
            except Exception as e_pandas:
                failure_reason = f"处理表格数据时遇到代码错误: {e_pandas}"
                api_logger.error(failure_reason, exc_info=True)
            
            if failure_reason: # 如果表格处理失败，清空答案，让后续流程处理
                final_answer = NO_ANSWER_PHRASE_ANSWER_CLEAN

        # 3.3 如果没有表格或表格问答失败，则执行通用文档问答流程
        if final_answer == NO_ANSWER_PHRASE_ANSWER_CLEAN:
            api_logger.info("No table answer generated. Proceeding with general document synthesis.")
            
            # 从融合后的子块中提取父文档
            all_parent_ids = {chunk['metadata']['parent_id'] for chunk in fused_child_chunks if 'parent_id' in chunk.get('metadata', {})}
            parent_docs = app_ctx.docstore.mget(list(all_parent_ids))
            valid_parent_docs = [doc for doc in parent_docs if doc]

            # 重排所有召回的父文档
            reranked_docs = await app_ctx.fusion_engine.rerank_documents(
                query=query_request.query,
                documents=[RetrievedDocument(content=doc.page_content, metadata=doc.metadata, score=0.0, source_type="fused_retrieval") for doc in valid_parent_docs],
                top_n=5
            )

            if not reranked_docs:
                failure_reason = "知识库中未能找到任何与您问题相关的信息。"
            else:
                summary_tasks = [generate_document_summary(app_ctx.llm_gbnf_instance, user_query=query_request.query, document_content=doc.content) for doc in reranked_docs]
                summaries = await asyncio.gather(*summary_tasks)
                
                relevant_summaries = [f"根据文档《{doc.metadata.get('filename', '未知文档')}》的信息：{summary}" for doc, summary in zip(reranked_docs, summaries) if summary]
                
                if not relevant_summaries:
                    failure_reason = "虽然检索到了相关文档，但无法从中提炼出与您问题直接相关的核心信息。"
                else:
                    fusion_context = "\n\n".join(relevant_summaries)
                    final_answer = await generate_answer_from_context(user_query=query_request.query, context_str=fusion_context, prompt_builder=lambda q, c: get_fusion_messages(q, c))

        # --- 4. 后续处理 (生成建议、提取任务等) ---
        if not final_answer or NO_ANSWER_PHRASE_ANSWER_CLEAN in final_answer:
            if not failure_reason: failure_reason = "根据检索到的上下文信息，无法直接回答您的问题。"
            suggestion = await generate_actionable_suggestion(app_ctx.llm_gbnf_instance, user_query=query_request.query, failure_reason=failure_reason)
            final_answer = f"{failure_reason} {suggestion}" if suggestion else failure_reason

        actionable_suggestion_response = None
        task_info = await extract_task_from_dialogue(app_ctx.llm_gbnf_instance, user_query=query_request.query, llm_answer=final_answer)
        if task_info and task_info.get("task_found"):
            title = task_info.get("title")
            due_date = task_info.get("due_date")
            reminder_offset = task_info.get("reminder_offset_minutes")

            if title and due_date:
                if reminder_offset is not None:
                    # 用户明确指定了提醒时间 -> 自动创建并返回确认信息
                    # TODO: 此处未来应调用 task_manager API 来实际创建任务
                    confirmation_msg = f"已为您自动创建任务“{title}”，并将在截止时间前 {reminder_offset} 分钟提醒您。"
                    actionable_suggestion_response = AutoScheduledTaskConfirmation(
                        title=title,
                        due_date=due_date,
                        reminder_offset_minutes=reminder_offset,
                        confirmation_message=confirmation_msg
                    )
                    api_logger.info(f"Generated AutoScheduledTaskConfirmation for the user.")
                else:
                    # 用户未指定提醒时间 -> 返回待确认建议
                    actionable_suggestion_response = PendingTaskSuggestion(
                        title=title,
                        due_date=due_date
                    )
                    api_logger.info(f"Generated PendingTaskSuggestion for the user.")

        # --- 4. 构建最终响应 ---
        response_to_return = HybridRAGResponse(
            answer=final_answer,
            original_query=query_request.query,
            retrieved_sources=reranked_docs,
            actionable_suggestion=actionable_suggestion_response
        )

        # 如果没有发生错误且答案有效，则将响应缓存起来
        if not failure_reason and NO_ANSWER_PHRASE_ANSWER_CLEAN not in final_answer:
            app_ctx.answer_cache[cache_key] = response_to_return

    except Exception as e:
        exception_occurred = e
        response_to_return = HybridRAGResponse(answer=f"An internal server error occurred: {e}", original_query=query_request.query, retrieved_sources=[])
    finally:
        processing_time_seconds = (datetime.now(timezone.utc) - start_time_total).total_seconds()
        log_data_for_finally = {
            "interaction_id": interaction_id_for_log,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "task_type": "rag_query_processing_v8_0_hybrid_search_with_intent_dispatch", # 更新任务类型以反映意图分派
            "original_user_query": query_request.query,
            "final_answer_from_llm": response_to_return.answer if response_to_return else "N/A",
            "final_context_docs_full": [doc.model_dump() for doc in response_to_return.retrieved_sources] if response_to_return else [],
            "retrieval_parameters": query_request.model_dump(),
            "processing_time_seconds": round(processing_time_seconds, 3),
            "user_intent_classified": user_intent.intent.value if user_intent else "N/A", # 添加意图分类结果
            "user_intent_reasoning": user_intent.reasoning if user_intent else "N/A"
        }
        if exception_occurred:
            log_data_for_finally["error_details"] = f"{type(exception_occurred).__name__}: {str(exception_occurred)}"
            log_data_for_finally["error_traceback"] = traceback.format_exc() # 确保 traceback 模块已导入
            
        await log_queue.put(log_data_for_finally)
        
        if exception_occurred:
            raise HTTPException(status_code=500, detail=str(exception_occurred))
        
        if response_to_return is None:
            response_to_return = HybridRAGResponse(answer="An unexpected error occurred during response generation.", original_query=query_request.query, retrieved_sources=[])
        
        return response_to_return
    
if __name__ == "__main__":
    uvicorn.run("zhz_rag.api.rag_api_service:app", host="0.0.0.0", port=8081, reload=False)

