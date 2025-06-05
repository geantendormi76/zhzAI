import os
import json
import asyncio
import traceback
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass, field # 确保导入 field
import time
import logging
import sys
import hashlib # <--- 添加
from datetime import datetime, timezone # <--- 添加
import uuid # <--- 添加


# MCP 框架导入
from mcp.server.fastmcp import FastMCP, Context

# --- 配置 rag_service 的专用日志 ---
_rag_service_py_dir = os.path.dirname(os.path.abspath(__file__))
_rag_service_log_file = os.path.join(_rag_service_py_dir, 'rag_service_debug.log')

rag_logger = logging.getLogger("RagServiceLogger")
rag_logger.setLevel(logging.DEBUG)
rag_logger.propagate = False

if rag_logger.hasHandlers():
    rag_logger.handlers.clear()

try:
    _file_handler = logging.FileHandler(_rag_service_log_file, mode='w')
    _file_handler.setLevel(logging.DEBUG)
    _formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _file_handler.setFormatter(_formatter)
    rag_logger.addHandler(_file_handler)
    rag_logger.info("--- RagServiceLogger configured to write to rag_service_debug.log ---")
except Exception as e:
    print(f"CRITICAL: Failed to configure RagServiceLogger: {e}")


# --- 从项目内部导入所有 RAG 模块 ---
from zhz_rag.config.pydantic_models import QueryRequest, HybridRAGResponse, RetrievedDocument
from zhz_rag.llm.sglang_wrapper import (
    generate_answer_from_context,
    generate_expanded_queries,
    generate_cypher_query,
    generate_clarification_question,
    generate_intent_classification,
    NO_ANSWER_PHRASE_ANSWER_CLEAN
)
from zhz_rag.core_rag.retrievers.chromadb_retriever import ChromaDBRetriever
from zhz_rag.core_rag.retrievers.file_bm25_retriever import FileBM25Retriever
from zhz_rag.core_rag.kg_retriever import KGRetriever
from zhz_rag.core_rag.fusion_engine import FusionEngine
from zhz_rag.utils.common_utils import log_interaction_data

from dotenv import load_dotenv

# 加载 .env 文件
# __file__ 是当前 rag_service.py 的路径: /home/zhz/zhz_agent/rag_service.py
# os.path.dirname(os.path.abspath(__file__)) 是 /home/zhz/zhz_agent 目录
# .env 文件与 rag_service.py 在同一个目录下 (zhz_agent 目录)
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, '.env')

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
    rag_logger.info(f"Loaded .env file from: {dotenv_path}")
else:
    rag_logger.warning(f".env file not found at {dotenv_path}, will rely on environment variables or defaults.")
    # 仍然尝试加载，因为python-dotenv的默认行为是查找当前工作目录和上级目录的.env
    load_dotenv()

# --- 应用上下文 Dataclass ---
@dataclass
class AppContext:
    # vector_retriever: VectorRetriever # 旧的
    chroma_retriever: Optional[ChromaDBRetriever] = None # 新的
    kg_retriever: Optional[KGRetriever] = None
    # bm25_retriever: BM25Retriever # 旧的
    file_bm25_retriever: Optional[FileBM25Retriever] = None # 新的
    fusion_engine: Optional[FusionEngine] = None
    # llm_generator: Optional[Any] = None # LLMGenerator在您的代码中没有被实例化并放入AppContext

# --- MCP 服务器生命周期管理 ---
@asynccontextmanager
async def app_lifespan_for_rag_service(server: FastMCP) -> AsyncIterator[AppContext]:
    rag_logger.info("--- RAG Service (FastMCP): 正在初始化 RAG 组件 (新版) ---")
    
    chroma_retriever_instance: Optional[ChromaDBRetriever] = None
    kg_retriever_instance: Optional[KGRetriever] = None
    file_bm25_retriever_instance: Optional[FileBM25Retriever] = None
    fusion_engine_instance: Optional[FusionEngine] = None

    # 初始化 ChromaDB Retriever
    try:
        # 这些路径和名称应该与Dagster流水线中配置的一致
        # 优先从环境变量读取，如果不存在则使用默认值（如果适用）
        chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "/home/zhz/dagster_home/chroma_data")
        chroma_collection_name = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
        embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH", "/home/zhz/models/bge-small-zh-v1.5")

        if not os.path.isdir(chroma_persist_dir):
                rag_logger.warning(f"ChromaDB persist directory '{chroma_persist_dir}' not found. Retrieval may fail or use an in-memory DB if ChromaDBRetriever handles this.")
        if not os.path.exists(embedding_model_path): # embedding_model_path 应该是目录
            rag_logger.warning(f"Embedding model path '{embedding_model_path}' not found. ChromaDBRetriever initialization might fail.")

        chroma_retriever_instance = ChromaDBRetriever(
            collection_name=chroma_collection_name,
            persist_directory=chroma_persist_dir,
            embedding_model_name_or_path=embedding_model_path
        )
        rag_logger.info("RAG Service: ChromaDBRetriever 初始化成功。")
    except Exception as e:
        rag_logger.error(f"RAG Service: ChromaDBRetriever 初始化失败: {e}", exc_info=True)
        # 不在此处抛出异常，允许服务在部分组件失败时仍能启动（如果设计如此）
    
    # 初始化 File BM25 Retriever
    try:
        bm25_index_dir = os.getenv("BM25_INDEX_DIRECTORY", "/home/zhz/dagster_home/bm25_index_data/")
        if not os.path.isdir(bm25_index_dir):
            rag_logger.warning(f"BM25 index directory '{bm25_index_dir}' not found. FileBM25Retriever initialization might fail.")
            
        file_bm25_retriever_instance = FileBM25Retriever(
            index_directory_path=bm25_index_dir
        )
        rag_logger.info("RAG Service: FileBM25Retriever 初始化成功。")
    except Exception as e:
        rag_logger.error(f"RAG Service: FileBM25Retriever 初始化失败: {e}", exc_info=True)

    # 初始化 KG Retriever
    try:
        # KGRetriever 内部会从环境变量读取NEO4J配置
        # generate_cypher_query 是从 zhz_agent.llm 导入的
        kg_retriever_instance = KGRetriever(llm_cypher_generator_func=generate_cypher_query)
        rag_logger.info("RAG Service: KGRetriever 初始化成功。")
    except Exception as e:
        rag_logger.error(f"RAG Service: KGRetriever 初始化失败: {e}", exc_info=True)
        if kg_retriever_instance and hasattr(kg_retriever_instance, 'close'): # 确保在失败前尝试关闭
            kg_retriever_instance.close()
            
    # 初始化 Fusion Engine
    try:
        fusion_engine_instance = FusionEngine(logger=rag_logger)
        rag_logger.info("RAG Service: FusionEngine 初始化成功。")
    except Exception as e:
        rag_logger.error(f"RAG Service: FusionEngine 初始化失败: {e}", exc_info=True)

    rag_logger.info("--- RAG Service (FastMCP): RAG 组件初始化尝试完成。---")

    ctx = AppContext(
        chroma_retriever=chroma_retriever_instance,
        kg_retriever=kg_retriever_instance,
        file_bm25_retriever=file_bm25_retriever_instance,
        fusion_engine=fusion_engine_instance
    )
    try:
        yield ctx
    finally:
        rag_logger.info("--- RAG Service (FastMCP): 正在清理资源 ---")
        if kg_retriever_instance: # 确保只在成功初始化后才调用close
            kg_retriever_instance.close() 
        rag_logger.info("--- RAG Service (FastMCP): 清理完成 ---")

# --- 初始化 FastMCP 应用 ---
rag_mcp_application = FastMCP(
    name="zhz_agent_rag_service", # 修改了服务名称以区分
    description="Upgraded Hybrid RAG 服务，使用持久化知识库。",
    lifespan=app_lifespan_for_rag_service,
)

@rag_mcp_application.tool()
async def query_rag_v2( # 重命名工具函数以避免与旧的混淆 (如果需要)
    ctx: Context,
    query: str, # 直接使用 query 作为输入，而不是 QueryRequest 对象
    top_k_vector: int = 3,
    top_k_kg: int = 2,
    top_k_bm25: int = 3,
    top_k_final: int = 3 # 最终融合后返回的文档数
) -> str: 
    rag_logger.info(f"\n--- RAG Service (query_rag_v2): 接收到查询: '{query}' ---")
    rag_logger.info(f"      Params: top_k_vector={top_k_vector}, top_k_kg={top_k_kg}, top_k_bm25={top_k_bm25}, top_k_final={top_k_final}")
    start_time_total = time.time()

    app_ctx: AppContext = ctx.request_context.lifespan_context
    response_payload = {} 
    original_query_for_response = query 
    final_json_output = ""
    # --- [新增日志变量] ---
    log_all_raw_retrievals_summary: List[Dict[str, Any]] = []
    log_final_context_docs_summary: List[Dict[str, Any]] = [] # This will store the full doc model dumps
    log_fused_context_text_for_llm_snippet: str = "N/A"
    log_final_answer_from_llm: str = "N/A"
    log_intent_classification_result: Optional[Dict[str, Any]] = None
    log_expanded_queries: Optional[List[str]] = None
    # --- [结束新增日志变量] ---

    try:
        # --- 1. LLM 驱动的意图分类和澄清触发 ---
        rag_logger.info(f"--- [TIME] 开始意图分类 at {time.time() - start_time_total:.2f}s ---")
        start_time_intent = time.time()
        intent_classification_result = await generate_intent_classification(query)
        log_intent_classification_result = intent_classification_result # <--- 记录日志
        rag_logger.info(f"--- [TIME] 结束意图分类, 耗时: {time.time() - start_time_intent:.2f}s. Result: {intent_classification_result}")

        if intent_classification_result.get("clarification_needed"):
            uncertainty_reason = intent_classification_result.get("reason", "查询可能存在歧义或信息不足。")
            clarification_question_text = await generate_clarification_question(query, uncertainty_reason)
            response_payload = {
                "status": "clarification_needed",
                "clarification_question": clarification_question_text,
                "original_query": original_query_for_response,
                "debug_info": {"uncertainty_reason": uncertainty_reason, "source": "intent_classification"}
            }
            rag_logger.info(f"--- 需要澄清，返回: {response_payload}")
            # final_json_output will be set before finally block

        else: 
             # --- 启用查询扩展 ---
            rag_logger.info(f"--- 查询清晰，无需澄清。将对原始查询 '{query}' 进行查询扩展 ---")
            start_time_expansion = time.time()
            expanded_queries = await generate_expanded_queries(query) # <--- 取消注释
            log_expanded_queries = expanded_queries # <--- 记录实际的扩展查询
            
            if not expanded_queries or query not in expanded_queries: # 确保原始查询一定在里面
                # 如果 generate_expanded_queries 返回空或不包含原始查询，至少处理原始查询
                if query not in (expanded_queries or []): # 处理 expanded_queries 可能为 None 的情况
                    expanded_queries = [query] + (expanded_queries or [])
                elif not expanded_queries:
                    expanded_queries = [query]

            rag_logger.info(f"--- 扩展后的查询列表 (共 {len(expanded_queries)} 个): {expanded_queries}. 耗时: {time.time() - start_time_expansion:.2f}s ---")
            
            all_raw_retrievals: List[RetrievedDocument] = []
            
            queries_to_process = expanded_queries # <--- 修改：现在处理所有扩展后的查询
            rag_logger.info(f"--- [TIME] 开始并行召回 for {len(queries_to_process)} queries at {time.time() - start_time_total:.2f}s ---")
            start_time_retrieval = time.time()

            for current_query_text in queries_to_process:
                rag_logger.info(f"Processing retrievals for query: '{current_query_text}'")
                
                # 向量检索 (ChromaDB)
                if app_ctx.chroma_retriever:
                    try:
                        chroma_docs_raw = app_ctx.chroma_retriever.retrieve(query_text=current_query_text, n_results=top_k_vector)
                        rag_logger.debug(f"   ChromaDB for '{current_query_text}' raw output: {chroma_docs_raw}") 
                        for doc_raw in chroma_docs_raw:
                            retrieved_doc = RetrievedDocument(
                                source_type="vector_chroma",
                                content=doc_raw.get("text", ""),
                                score=doc_raw.get("score", 0.0),
                                metadata={**doc_raw.get("metadata", {}), "original_query_part": current_query_text}
                            )
                            all_raw_retrievals.append(retrieved_doc)
                            log_all_raw_retrievals_summary.append(retrieved_doc.model_dump()) 
                        rag_logger.info(f"   ChromaDB for '{current_query_text}': found {len(chroma_docs_raw)} docs.")
                    except Exception as e_chroma:
                        rag_logger.error(f"   Error during ChromaDB retrieval for '{current_query_text}': {e_chroma}", exc_info=True)
                
                # 关键词检索 (BM25)
                if app_ctx.file_bm25_retriever:
                    try:
                        bm25_docs_raw = app_ctx.file_bm25_retriever.retrieve(query_text=current_query_text, n_results=top_k_bm25)
                        rag_logger.debug(f"   BM25 for '{current_query_text}' raw output (IDs and scores): {bm25_docs_raw}") 
                        for doc_raw_bm25 in bm25_docs_raw:
                            bm25_chunk_id = doc_raw_bm25.get("id")
                            text_content_for_bm25 = f"[BM25: Text for ID {bm25_chunk_id} pending]"
                            found_in_chroma = False
                            for existing_doc in all_raw_retrievals: 
                                if (existing_doc.metadata and (existing_doc.metadata.get("chunk_id") == bm25_chunk_id or existing_doc.metadata.get("id") == bm25_chunk_id)):
                                    text_content_for_bm25 = existing_doc.content
                                    found_in_chroma = True
                                    break
                            if not found_in_chroma and app_ctx.chroma_retriever and bm25_chunk_id: 
                                try:
                                    specific_chroma_doc = app_ctx.chroma_retriever._collection.get(ids=[bm25_chunk_id], include=["metadatas", "documents"]) # Also fetch documents for content
                                    if specific_chroma_doc:
                                        if specific_chroma_doc.get("documents") and specific_chroma_doc.get("documents")[0]:
                                            text_content_for_bm25 = specific_chroma_doc["documents"][0]
                                        elif specific_chroma_doc.get("metadatas") and specific_chroma_doc.get("metadatas")[0]: # Fallback to chunk_text in metadata
                                            text_content_for_bm25 = specific_chroma_doc["metadatas"][0].get("chunk_text", text_content_for_bm25)

                                except Exception as e_chroma_get:
                                    rag_logger.warning(f"   Failed to get text for BM25 ID {bm25_chunk_id} from Chroma: {e_chroma_get}")
                            
                            retrieved_doc = RetrievedDocument(
                                source_type="keyword_bm25s",
                                content=text_content_for_bm25,
                                score=doc_raw_bm25.get("score", 0.0),
                                metadata={"chunk_id": bm25_chunk_id, "original_query_part": current_query_text}
                            )
                            all_raw_retrievals.append(retrieved_doc)
                            log_all_raw_retrievals_summary.append(retrieved_doc.model_dump()) 
                        rag_logger.info(f"   BM25s for '{current_query_text}': found {len(bm25_docs_raw)} potential docs.")
                    except Exception as e_bm25:
                        rag_logger.error(f"   Error during BM25 retrieval for '{current_query_text}': {e_bm25}", exc_info=True)

                # 知识图谱检索
                if app_ctx.kg_retriever:
                    try:
                        rag_logger.info(f"   Performing KG retrieval for query: '{current_query_text}'")
                        kg_docs = await app_ctx.kg_retriever.retrieve_with_llm_cypher(query=current_query_text, top_k=top_k_kg)
                        rag_logger.debug(f"   KG for '{current_query_text}' raw output: {kg_docs}") 
                        for kg_doc_data in kg_docs: # kg_docs is List[Dict], needs conversion
                            retrieved_doc = RetrievedDocument(**kg_doc_data) # Convert dict to Pydantic model
                            if retrieved_doc.metadata:
                                retrieved_doc.metadata["original_query_part"] = current_query_text
                            else:
                                retrieved_doc.metadata = {"original_query_part": current_query_text}
                            all_raw_retrievals.append(retrieved_doc)
                            log_all_raw_retrievals_summary.append(retrieved_doc.model_dump()) 
                        rag_logger.info(f"   KG Retrieval for '{current_query_text}': found {len(kg_docs)} results.")
                    except Exception as e_kg:
                        rag_logger.error(f"   Error during KG retrieval for '{current_query_text}': {e_kg}", exc_info=True)
            
            rag_logger.info(f"--- [TIME] 结束所有召回, 耗时: {time.time() - start_time_retrieval:.2f}s ---")
            rag_logger.info(f"--- 总计从各路召回（所有查询处理后）的结果数: {len(all_raw_retrievals)} ---")
            if all_raw_retrievals:
                for i_doc, doc_retrieved in enumerate(all_raw_retrievals[:3]): # 日志只打印前3条摘要
                        rag_logger.debug(f"   Raw Doc {i_doc} (Summary): type={doc_retrieved.source_type}, score={doc_retrieved.score}, content='{str(doc_retrieved.content)[:50]}...'")

            if not all_raw_retrievals: 
                response_payload = {
                    "status": "success", 
                    "final_answer": "抱歉，根据您提供的查询，未能从知识库中找到相关信息。",
                    "original_query": original_query_for_response,
                    "retrieved_context_docs": [], 
                    "debug_info": {"message": "No documents retrieved from any source."}
                }
            else:
                rag_logger.info(f"--- [TIME] 开始结果融合与重排序 at {time.time() - start_time_total:.2f}s ---")
                start_time_fusion = time.time()
                final_context_docs: List[RetrievedDocument]
                if not app_ctx.fusion_engine:
                    rag_logger.error("FusionEngine not available! Skipping fusion and reranking.")
                    final_context_docs = sorted(all_raw_retrievals, key=lambda d: d.score if d.score is not None else -float('inf'), reverse=True)[:top_k_final]
                else:
                    final_context_docs = await app_ctx.fusion_engine.fuse_results(
                        all_raw_retrievals, 
                        original_query_for_response,
                        top_n_final=top_k_final
                    ) 
                log_final_context_docs_summary = [doc.model_dump() for doc in final_context_docs] 

                # --- 新增日志，检查 model_dump 的输出 ---
                if log_final_context_docs_summary:
                    rag_logger.info(f"DEBUG_MODEL_DUMP: First item of log_final_context_docs_summary (from model_dump()): {json.dumps(log_final_context_docs_summary[0], ensure_ascii=False, default=str)}")
                # --- 结束新增日志 ---

                rag_logger.info(f"--- [TIME] 结束结果融合与重排序, 耗时: {time.time() - start_time_fusion:.2f}s. Final context docs: {len(final_context_docs)} ---")
                if final_context_docs:
                    for i_fdoc, fdoc_retrieved in enumerate(final_context_docs[:3]): # 日志只打印前3条摘要
                        rag_logger.debug(f"   Fused Doc {i_fdoc} (Summary): type={fdoc_retrieved.source_type}, score={fdoc_retrieved.score}, content='{str(fdoc_retrieved.content)[:50]}...'")
                
                if not final_context_docs: 
                    fused_context_text_for_llm = "未在知识库中找到相关信息。"
                    final_answer_from_llm = "根据现有知识，未能找到您查询的相关信息。"
                    response_payload = {
                        "status": "success",
                        "final_answer": final_answer_from_llm,
                        "original_query": original_query_for_response,
                        "retrieved_context_docs": [], 
                        "debug_info": {"message": "No relevant context found after fusion."}
                    }
                else:
                    context_strings_for_llm = []
                    for doc in final_context_docs:
                        score_str = f"{doc.score:.4f}" if isinstance(doc.score, float) else str(doc.score if doc.score is not None else 'N/A')
                        context_strings_for_llm.append(
                            f"Source Type: {doc.source_type}, Score: {score_str}\nContent: {doc.content}"
                        )
                    fused_context_text_for_llm = "\n\n---\n\n".join(context_strings_for_llm)
                    log_fused_context_text_for_llm_snippet = fused_context_text_for_llm[:500] 

                    rag_logger.info(f"\n--- FUSED CONTEXT for LLM (length: {len(fused_context_text_for_llm)} chars) ---")
                    rag_logger.info(f"{fused_context_text_for_llm[:1000]}...") 
                    rag_logger.info(f"--- END OF FUSED CONTEXT ---\n")

                    rag_logger.info(f"--- [TIME] 开始最终答案生成 at {time.time() - start_time_total:.2f}s ---")
                    start_time_answer_gen = time.time()
                    final_answer_from_llm = await generate_answer_from_context(query, fused_context_text_for_llm)
                    log_final_answer_from_llm = final_answer_from_llm or "N/A" 
                    rag_logger.info(f"--- [TIME] 结束最终答案生成, 耗时: {time.time() - start_time_answer_gen:.2f}s ---")

                    if not final_answer_from_llm or final_answer_from_llm.strip() == NO_ANSWER_PHRASE_ANSWER_CLEAN:
                        final_answer_from_llm = "根据您提供的信息，我暂时无法给出明确的回答。"
                    
                    response_payload = {
                        "status": "success",
                        "final_answer": final_answer_from_llm,
                        "original_query": original_query_for_response,
                        "retrieved_context_docs": [doc.model_dump() for doc in final_context_docs], 
                        "debug_info": {"total_raw_retrievals_count": len(all_raw_retrievals)}
                    }

        final_json_output = json.dumps(response_payload, ensure_ascii=False)
        rag_logger.info(f"--- 'query_rag_v2' 逻辑执行完毕, 总耗时: {time.time() - start_time_total:.2f}s. ---")
        
    except Exception as e_main:
        rag_logger.error(f"RAG Service CRITICAL ERROR in 'query_rag_v2' (main try-except): {type(e_main).__name__} - {str(e_main)}", exc_info=True)
        user_query_for_err_log = original_query_for_response if 'original_query_for_response' in locals() and original_query_for_response else query
        response_payload = {
            "status": "error",
            "error_code": "RAG_SERVICE_INTERNAL_ERROR",
            "error_message": f"RAG服务内部发生未预期错误: {str(e_main)}",
            "original_query": user_query_for_err_log,
            "debug_info": {"exception_type": type(e_main).__name__}
        }
        final_json_output = json.dumps(response_payload, ensure_ascii=False)
    finally: 
        interaction_id_for_log = str(uuid.uuid4())
        current_app_version = "zhz_rag_mcp_service_0.2.1" 

        processed_final_context_docs_for_log = []
        temp_log_final_context_docs = locals().get('log_final_context_docs_summary') # 安全获取

        if temp_log_final_context_docs: # 如果 RAG 流程成功并且 final_context_docs 被处理了
            for doc_dict in temp_log_final_context_docs: # temp_log_final_context_docs 是 model_dump() 后的列表
                cleaned_doc = {}
                for key, value in doc_dict.items():
                    if isinstance(value, float) and (value != value or value == float('inf') or value == float('-inf')): 
                        cleaned_doc[key] = None 
                    else:
                        cleaned_doc[key] = value
                processed_final_context_docs_for_log.append(cleaned_doc)
        # 如果 temp_log_final_context_docs 为 None (例如澄清路径)，则 processed_final_context_docs_for_log 保持为 []

        full_log_entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "interaction_id": interaction_id_for_log,
            "original_query_interaction_id_ref": locals().get('original_query_interaction_id'), 
            "task_type": "rag_query_processing_full_log",
            "app_version": current_app_version,
            "original_user_query": locals().get('original_query_for_response', query), # query 总是有定义的
            "query_params": {
                "top_k_vector": top_k_vector, "top_k_kg": top_k_kg, 
                "top_k_bm25": top_k_bm25, "top_k_final": top_k_final
            },
            "intent_classification_result": locals().get('log_intent_classification_result'),
            "expanded_queries": locals().get('log_expanded_queries', []), # 默认为空列表
            "all_raw_retrievals_count": len(locals().get('log_all_raw_retrievals_summary', [])),
            "final_context_docs_count": len(processed_final_context_docs_for_log), # 使用清理后列表的长度
            "final_context_docs_summary": [ 
                {
                    "source_type": doc.get("source_type"), 
                    "score": doc.get("score"), 
                    "id": (doc.get("metadata",{}).get("chunk_id") or doc.get("metadata",{}).get("id")) if doc.get("metadata") else None, 
                    "content_preview": str(doc.get("content","N/A"))[:50]+"..."
                } 
                for doc in processed_final_context_docs_for_log[:5] # 使用清理后列表的摘要
            ], 
            "final_context_docs_full": processed_final_context_docs_for_log, # <--- 使用清理后的完整列表
            "fused_context_text_for_llm_snippet": locals().get('log_fused_context_text_for_llm_snippet', "N/A"),
            "final_answer_from_llm": locals().get('log_final_answer_from_llm', "N/A"),
            "final_response_payload_status": locals().get('response_payload', {}).get("status", "Unknown"),
            "total_processing_time_seconds": round(time.time() - start_time_total, 2) if 'start_time_total' in locals() else -1,
        }

        try:
            await log_interaction_data(full_log_entry) 
            rag_logger.info(f"Full RAG interaction log (ID: {interaction_id_for_log}) has been written.")
        except Exception as e_log_final:
            rag_logger.error(f"CRITICAL: Failed to write full RAG interaction log: {e_log_final}", exc_info=True)
        
        sys.stdout.flush(); sys.stderr.flush() 
    
    return final_json_output

if __name__ == "__main__":
    rag_logger.info("--- Starting RAG Service (FastMCP for mcpo via direct run) ---")
    rag_mcp_application.run()