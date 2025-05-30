# zhz_agent/rag_service.py

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
from zhz_agent.pydantic_models import QueryRequest, HybridRAGResponse, RetrievedDocument
from zhz_agent.llm import (
    generate_answer_from_context,
    generate_expanded_queries,
    generate_cypher_query,
    generate_clarification_question,
    generate_intent_classification,
    NO_ANSWER_PHRASE_ANSWER_CLEAN
)
from zhz_agent.chromadb_retriever import ChromaDBRetriever
from zhz_agent.file_bm25_retriever import FileBM25Retriever
from zhz_agent.kg import KGRetriever
from zhz_agent.fusion import FusionEngine
from zhz_agent.utils import log_interaction_data

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

# --- MCP 工具定义 ---
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
    rag_logger.info(f"    Params: top_k_vector={top_k_vector}, top_k_kg={top_k_kg}, top_k_bm25={top_k_bm25}, top_k_final={top_k_final}")
    start_time_total = time.time()

    app_ctx: AppContext = ctx.request_context.lifespan_context
    response_payload = {} 
    original_query_for_response = query 
    final_json_output = ""

    try:
        # --- 1. LLM 驱动的意图分类和澄清触发 (保持不变) ---
        rag_logger.info(f"--- [TIME] 开始意图分类 at {time.time() - start_time_total:.2f}s ---")
        start_time_intent = time.time()
        intent_classification_result = await generate_intent_classification(query)
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
            final_json_output = json.dumps(response_payload, ensure_ascii=False)
            sys.stdout.flush(); sys.stderr.flush()
            return final_json_output

        # --- 2. 查询扩展 (保持不变) ---
        rag_logger.info(f"--- 查询清晰，无需澄清。开始查询扩展 for: {query} ---")
        start_time_expansion = time.time()
        expanded_queries = await generate_expanded_queries(query) # 假设返回 List[str]
        rag_logger.info(f"--- 扩展查询列表 (共 {len(expanded_queries)} 个): {expanded_queries}. 耗时: {time.time() - start_time_expansion:.2f}s ---")
        
        all_raw_retrievals: List[RetrievedDocument] = []
        
        # --- 3. 并行多路召回 (使用新的检索器) ---
        # 我们将为原始查询和每个扩展查询都执行三路召回
        queries_to_process = [query] + expanded_queries 
        # 或者，如果觉得扩展查询过多，可以只用原始查询或选择性使用扩展查询
        # queries_to_process = [query] # 简化：仅使用原始查询进行召回

        rag_logger.info(f"--- [TIME] 开始并行召回 for {len(queries_to_process)} queries at {time.time() - start_time_total:.2f}s ---")
        start_time_retrieval = time.time()

        for current_query_text in queries_to_process:
            rag_logger.info(f"Processing retrievals for query: '{current_query_text}'")
            
            # 向量检索 (ChromaDB)
            if app_ctx.chroma_retriever:
                try:
                    chroma_docs_raw = app_ctx.chroma_retriever.retrieve(query_text=current_query_text, n_results=top_k_vector)
                    for doc_raw in chroma_docs_raw:
                        all_raw_retrievals.append(
                            RetrievedDocument(
                                source_type="vector_chroma",
                                content=doc_raw.get("text", ""),
                                score=doc_raw.get("score", 0.0),
                                metadata={**doc_raw.get("metadata", {}), "original_query_part": current_query_text} # 添加原始查询部分
                            )
                        )
                    rag_logger.info(f"  ChromaDB for '{current_query_text}': found {len(chroma_docs_raw)} docs.")
                except Exception as e_chroma:
                    rag_logger.error(f"  Error during ChromaDB retrieval for '{current_query_text}': {e_chroma}", exc_info=True)
            
            # 关键词检索 (BM25)
            if app_ctx.file_bm25_retriever:
                try:
                    bm25_docs_raw = app_ctx.file_bm25_retriever.retrieve(query_text=current_query_text, n_results=top_k_bm25)
                    # BM25只返回ID和分数，我们需要补充文本。
                    # 策略：尝试从已有的ChromaDB召回结果中匹配ID补充文本，或标记为待补充。
                    for doc_raw_bm25 in bm25_docs_raw:
                        bm25_chunk_id = doc_raw_bm25.get("id")
                        text_content_for_bm25 = f"[BM25: Text for ID {bm25_chunk_id} pending]"
                        # 简单的补充逻辑：
                        found_in_chroma = False
                        for chroma_doc in all_raw_retrievals: # 检查已有的（主要是chroma的）
                            if chroma_doc.metadata.get("chunk_id") == bm25_chunk_id or chroma_doc.metadata.get("id") == bm25_chunk_id : # ChromaDBRetriever的id是chunk_id
                                text_content_for_bm25 = chroma_doc.content
                                found_in_chroma = True
                                break
                        if not found_in_chroma and app_ctx.chroma_retriever: # 如果没找到，尝试从ChromaDB单独获取
                            try:
                                # 假设ChromaDB存储时ID就是chunk_id
                                # get()方法返回更完整的文档信息
                                specific_chroma_doc = app_ctx.chroma_retriever._collection.get(ids=[bm25_chunk_id], include=["metadatas"])
                                if specific_chroma_doc and specific_chroma_doc.get("metadatas") and specific_chroma_doc.get("metadatas")[0]:
                                    text_content_for_bm25 = specific_chroma_doc["metadatas"][0].get("chunk_text", text_content_for_bm25)
                            except Exception as e_chroma_get:
                                rag_logger.warning(f"  Failed to get text for BM25 ID {bm25_chunk_id} from Chroma: {e_chroma_get}")
                                
                        all_raw_retrievals.append(
                            RetrievedDocument(
                                source_type="keyword_bm25s",
                                content=text_content_for_bm25,
                                score=doc_raw_bm25.get("score", 0.0),
                                metadata={"chunk_id": bm25_chunk_id, "original_query_part": current_query_text}
                            )
                        )
                    rag_logger.info(f"  BM25s for '{current_query_text}': found {len(bm25_docs_raw)} potential docs.")
                except Exception as e_bm25:
                    rag_logger.error(f"  Error during BM25 retrieval for '{current_query_text}': {e_bm25}", exc_info=True)

            # 知识图谱检索 (Neo4j)
            if app_ctx.kg_retriever:
                try:
                    rag_logger.info(f"  Performing KG retrieval for query: '{current_query_text}'") # 添加日志
                    kg_docs = await app_ctx.kg_retriever.retrieve_with_llm_cypher(
                        query=current_query_text, # <--- 修改这里
                        top_k=top_k_kg
                    )
                    # retrieve_with_llm_cypher 已经返回 List[RetrievedDocument]
                    for kg_doc in kg_docs: # 添加原始查询部分到元数据
                        if kg_doc.metadata:
                            kg_doc.metadata["original_query_part"] = current_query_text
                        else:
                            kg_doc.metadata = {"original_query_part": current_query_text}
                    all_raw_retrievals.extend(kg_docs)
                    rag_logger.info(f"  KG Retrieval for '{current_query_text}': found {len(kg_docs)} results.")
                except Exception as e_kg:
                    rag_logger.error(f"  Error during KG retrieval for '{current_query_text}': {e_kg}", exc_info=True)
        
        rag_logger.info(f"--- [TIME] 结束所有召回, 耗时: {time.time() - start_time_retrieval:.2f}s ---")
        rag_logger.info(f"--- 总计从各路召回（所有查询处理后）的结果数: {len(all_raw_retrievals)} ---")
        for i_doc, doc_retrieved in enumerate(all_raw_retrievals[:10]): # 日志只打印前10条
            rag_logger.debug(f"  Raw Doc {i_doc}: type={doc_retrieved.source_type}, score={doc_retrieved.score}, content='{str(doc_retrieved.content)[:100]}...'")

        if not all_raw_retrievals: 
            # ... (无召回结果的处理，与您之前的代码类似) ...
            response_payload = {
                "status": "success", 
                "final_answer": "抱歉，根据您提供的查询，未能从知识库中找到相关信息。",
                "original_query": original_query_for_response,
                "debug_info": {"message": "No documents retrieved from any source."}
            }
            final_json_output = json.dumps(response_payload, ensure_ascii=False)
            sys.stdout.flush(); sys.stderr.flush()
            return final_json_output

        # --- 4. 结果融合与重排序 (使用FusionEngine) ---
        rag_logger.info(f"--- [TIME] 开始结果融合与重排序 at {time.time() - start_time_total:.2f}s ---")
        start_time_fusion = time.time()
        if not app_ctx.fusion_engine:
            rag_logger.error("FusionEngine not available! Skipping fusion and reranking.")
            # 如果没有融合引擎，直接使用原始召回结果（可能需要截断和简单排序）
            # 这里简化处理：直接取 all_raw_retrievals，按分数初排（如果分数可比）
            # 或者只用向量检索结果
            # 为了演示，我们假设至少需要向量结果，或者返回错误
            final_context_docs = sorted(all_raw_retrievals, key=lambda d: d.score, reverse=True)[:top_k_final]
        else:
            # FusionEngine的 fuse_results 方法在您的代码中是异步的
            final_context_docs = await app_ctx.fusion_engine.fuse_results(
                all_raw_retrievals, 
                original_query_for_response, # 传递原始查询给融合引擎
                top_n_final=top_k_final # 传递最终需要的文档数
            ) 
        rag_logger.info(f"--- [TIME] 结束结果融合与重排序, 耗时: {time.time() - start_time_fusion:.2f}s. Final context docs: {len(final_context_docs)} ---")
        
        # --- 5. 准备上下文并生成答案 (与您之前的代码类似) ---
        # 注意：您的FusionEngine.fuse_results 返回的是融合后的文本字符串，而不是RetrievedDocument列表
        # 我们需要调整这里，或者调整FusionEngine使其返回RetrievedDocument列表
        # 假设FusionEngine返回的是RetrievedDocument列表 (需要修改FusionEngine)
        
        if not final_context_docs: # 如果融合后没有文档
            fused_context_text_for_llm = "未在知识库中找到相关信息。"
            final_answer_from_llm = "根据现有知识，未能找到您查询的相关信息。"
            response_payload = {
                "status": "success",
                "final_answer": final_answer_from_llm,
                "original_query": original_query_for_response,
                "debug_info": {"message": "No relevant context found after fusion."}
            }
        else:
            # 假设 final_context_docs 是 List[RetrievedDocument]
            context_strings_for_llm = [
                f"Source Type: {doc.source_type}, Score: {doc.score:.4f}\nContent: {doc.content}" 
                for doc in final_context_docs
            ]
            fused_context_text_for_llm = "\n\n---\n\n".join(context_strings_for_llm)

            rag_logger.info(f"\n--- FUSED CONTEXT for LLM (length: {len(fused_context_text_for_llm)} chars) ---")
            rag_logger.info(f"{fused_context_text_for_llm[:1000]}...") # 日志打印部分上下文
            rag_logger.info(f"--- END OF FUSED CONTEXT ---\n")

            rag_logger.info(f"--- [TIME] 开始最终答案生成 at {time.time() - start_time_total:.2f}s ---")
            start_time_answer_gen = time.time()
            final_answer_from_llm = await generate_answer_from_context(query, fused_context_text_for_llm)
            rag_logger.info(f"--- [TIME] 结束最终答案生成, 耗时: {time.time() - start_time_answer_gen:.2f}s ---")

            if not final_answer_from_llm or final_answer_from_llm.strip() == NO_ANSWER_PHRASE_ANSWER_CLEAN:
                final_answer_from_llm = "根据您提供的信息，我暂时无法给出明确的回答。"
            
            response_payload = {
                "status": "success",
                "final_answer": final_answer_from_llm,
                "original_query": original_query_for_response,
                "retrieved_context_docs": [doc.model_dump() for doc in final_context_docs], # 返回用于生成答案的文档
                "debug_info": {"total_raw_retrievals_count": len(all_raw_retrievals)}
            }

        # --- 添加顶层RAG交互日志记录 (新添加) ---
        if response_payload.get("status") == "success" and final_answer_from_llm and final_context_docs:
            try:
                context_content_for_hash = " ".join(sorted([doc.content for doc in final_context_docs]))
                context_hash = hashlib.md5(context_content_for_hash.encode('utf-8')).hexdigest()
                current_app_version = "0.1.0" # 假设的应用版本，后续可以从配置读取

                top_level_rag_log_data = {
                    # "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    # "interaction_id": str(uuid.uuid4()),
                    "task_type": "rag_query_processing_success", # 更具体的类型
                    "user_query": original_query_for_response,
                    "processed_llm_output": final_answer_from_llm,
                    "retrieved_context_hash": context_hash,
                    "retrieved_documents_summary": [
                        {"source": doc.source_type,
                         "score": doc.score,
                         "id": doc.metadata.get("chunk_id") if doc.metadata else doc.metadata.get("id") if doc.metadata else None, # 尝试获取chunk_id或id
                         "content_preview": doc.content[:50] + "..." if doc.content else ""} # 添加内容预览
                        for doc in final_context_docs
                    ],
                    "final_context_docs_count": len(final_context_docs),
                    "application_version": current_app_version
                }
                # rag_logger.info(f"TOP_LEVEL_RAG_SUCCESS_LOG: {json.dumps(top_level_rag_log_data, ensure_ascii=False)}")
                await log_interaction_data(top_level_rag_log_data)
            except Exception as e_log_rag:
                rag_logger.error(f"Error during top-level RAG success logging: {e_log_rag}", exc_info=True)
        elif response_payload.get("status") == "clarification_needed":
            try:
                current_app_version = "0.1.0"
                top_level_rag_log_data = {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "interaction_id": str(uuid.uuid4()),
                    "task_type": "rag_clarification_needed",
                    "user_query": original_query_for_response,
                    "clarification_question": response_payload.get("clarification_question"),
                    "uncertainty_reason": response_payload.get("debug_info", {}).get("uncertainty_reason"),
                    "application_version": current_app_version
                }
                # rag_logger.info(f"TOP_LEVEL_RAG_CLARIFICATION_LOG: {json.dumps(top_level_rag_log_data, ensure_ascii=False)}")
                await log_interaction_data(top_level_rag_log_data)
            except Exception as e_log_clarify:
                rag_logger.error(f"Error during top-level RAG clarification logging: {e_log_clarify}", exc_info=True)
        # --- 结束顶层RAG交互日志记录 ---

        final_json_output = json.dumps(response_payload, ensure_ascii=False)
        rag_logger.info(f"--- 'query_rag_v2' 成功执行完毕, 总耗时: {time.time() - start_time_total:.2f}s. 返回JSON响应 ---")
        
        sys.stdout.flush(); sys.stderr.flush() #确保在 try 块内，return 前
        return final_json_output

    except Exception as e_main:
        rag_logger.error(f"RAG Service CRITICAL ERROR in 'query_rag_v2' (main try-except): {type(e_main).__name__} - {str(e_main)}", exc_info=True)
        
        # 确保 original_query_for_response 在此作用域内有效
        # 如果 query_rag_v2 的参数就是 query，且 original_query_for_response 在 try 开始时被赋值为 query
        user_query_for_err_log = original_query_for_response if 'original_query_for_response' in locals() and original_query_for_response else query
        
        response_payload = {
            "status": "error",
            "error_code": "RAG_SERVICE_INTERNAL_ERROR",
            "error_message": f"RAG服务内部发生未预期错误: {str(e_main)}",
            "original_query": user_query_for_err_log,
            "debug_info": {"exception_type": type(e_main).__name__}
        }

        # --- 添加顶层RAG错误日志记录 ---
        try:
            current_app_version = "0.1.0"
            top_level_rag_error_log_data = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "interaction_id": str(uuid.uuid4()),
                "task_type": "rag_query_processing_error",
                "user_query": user_query_for_err_log,
                "error_message": str(e_main),
                "error_type": type(e_main).__name__,
                "traceback": traceback.format_exc(),
                "application_version": current_app_version
            }
            # rag_logger.info(f"TOP_LEVEL_RAG_ERROR_LOG: {json.dumps(top_level_rag_error_log_data, ensure_ascii=False)}")
            await log_interaction_data(top_level_rag_error_log_data)
        except Exception as e_log_err_inner:
            rag_logger.error(f"CRITICAL: Error during top-level RAG error logging itself: {e_log_err_inner}", exc_info=True)
        # --- 结束顶层RAG错误日志记录 ---

        final_json_output = json.dumps(response_payload, ensure_ascii=False)
    sys.stdout.flush(); sys.stderr.flush()
    return final_json_output

# --- 用于本地独立测试的 main 部分 ---
async def local_rag_test():
    rag_logger.info("--- Starting Local RAG Test ---")
    # 确保所有服务（ChromaDB数据存在, BM25索引文件存在, Neo4j运行, SGLang运行）都准备好
    
    # 模拟FastMCP的Context和AppContext
    class MockLifespanContext:
        def __init__(self):
            self.chroma_retriever = AppContext().chroma_retriever
            self.file_bm25_retriever = AppContext().file_bm25_retriever
            self.kg_retriever = AppContext().kg_retriever
            self.fusion_engine = AppContext().fusion_engine
            # self.llm_generator = AppContext().llm_generator # 不在AppContext中

    class MockRequestContext:
        def __init__(self):
            self.lifespan_context = MockLifespanContext()

    class MockContext: # 模拟FastMCP的Context
        def __init__(self):
            self.request_context = MockRequestContext()
            self.tool_name = "query_rag_v2" # 假设
            self.call_id = "local_test_call"
            # logger 可以设为 rag_logger
            # self.logger = rag_logger 
            # 但工具函数内部的 app_ctx: AppContext = ctx.request_context.lifespan_context
            # 我们需要确保 lifespan_context 正确填充了检索器

    # 实际的app_lifespan_for_rag_service 会在FastMCP启动时填充AppContext
    # 为了本地测试，我们需要手动模拟这个填充过程，或者直接使用全局初始化的检索器
    # 更简单的方式是，让 query_rag_v2 直接使用全局初始化的检索器（如果它们在模块级别）
    # 但FastMCP的推荐做法是通过lifespan管理上下文。

    # 为了本地测试能跑通，我们先假设app_lifespan_for_rag_service在模块加载时已执行
    # 并填充了全局的检索器实例（虽然这不是FastMCP的典型用法）
    # 或者，我们直接在测试函数内调用 app_lifespan_for_rag_service
    
    async with app_lifespan_for_rag_service(None) as app_context_instance: # 传入None作为server参数
        
        # 构建一个模拟的 request_context，它需要有一个 lifespan_context 属性
        class DummyRequestContext:
            def __init__(self, lifespan_ctx):
                self.lifespan_context = lifespan_ctx
        
        mock_req_context = DummyRequestContext(app_context_instance)

        # 创建 FastMCP Context 实例时，直接传入所有必需的参数
        mock_mcp_context = Context(
            request_context=mock_req_context, # <--- 在这里传入
            tool_name="query_rag_v2",
            call_id="local_test_call",
            logger=rag_logger # 使用我们已配置的 rag_logger
        )
        # 不再需要下面这行，因为它会导致 AttributeError
        # mock_mcp_context.request_context = DummyRequestContext() 
        # mock_mcp_context.request_context.lifespan_context = app_context_instance


    # --- 使用新的测试查询列表 ---
        test_queries = [
            # --- 基本的单实体、单关系查询 ---
            "张三在哪里工作？", 
            "项目Alpha的文档编写任务分配给了谁？", 
            "李四负责哪些任务？",
            "市场部有哪些员工？", 
            "列出所有类型为TASK的实体。",
            "查询所有在创新科技公司工作的员工。",

            # --- 测试对 :ExtractedEntity 和 label 属性的正确使用 ---
            "查找所有人员的名称。", 
            "有多少个组织类型的实体？",

            # --- 稍微复杂一点的查询 ---
            "王五既在A公司工作，又负责了项目B吗？", 
            "列出所有被分配了任务的员工。", 

            # --- 明确指定实体类型的问题 ---
            "名为“市场调研报告”的任务分配给了哪个PERSON？",
            "名为“战略规划部”的ORGANIZATION有哪些PERSON在里面工作？",

            # --- 可能会让Qwen困惑或生成不优查询的问题 ---
            "告诉我关于张三和他的工作单位的信息。", 
            "谁在谷歌工作，并且也负责了项目Alpha的文档编写任务？",

            # --- 测试Schema中不存在的关系或属性 ---
            "张三的年龄是多少？", 
            "项目Alpha的预算有多少？", 
            "李四和王五是同事吗？", 
            "谷歌公司是什么时候成立的？", 
        ]
    # --- 结束新的测试查询列表 ---
        # ... (后续的测试循环代码保持不变) ...
        for t_query in test_queries:
            print(f"\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print(f"EXECUTING LOCAL TEST FOR QUERY: {t_query}")
            print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
            response_str = await query_rag_v2(mock_mcp_context, query=t_query, top_k_final=2)
            
            print("\n--- RAG Service Local Test Response ---")
            try:
                response_data = json.loads(response_str)
                print(f"Query: {response_data.get('original_query')}")
                print(f"Answer: {response_data.get('final_answer')}")
                print(f"Status: {response_data.get('status')}")
                if response_data.get('status') == 'success' and response_data.get('retrieved_context_docs'):
                    print("\nRetrieved & Reranked Documents for Context:")
                    for i, doc_dict in enumerate(response_data.get('retrieved_context_docs', [])):
                        # 确保doc_dict是字典才解包
                        if isinstance(doc_dict, dict):
                            doc_obj = RetrievedDocument(**doc_dict) 
                            print(f"  Doc {i+1} (Source: {doc_obj.source_type}, Score: {doc_obj.score:.4f}):")
                            print(f"    Content: {doc_obj.content[:150]}...")
                        else:
                            print(f"  Doc {i+1}: (Unexpected format: {type(doc_dict)}) - {str(doc_dict)[:150]}...")
                elif response_data.get('status') == 'clarification_needed':
                     print(f"Clarification Question: {response_data.get('clarification_question')}")

            except json.JSONDecodeError:
                print("Error decoding JSON response from RAG service.")
                print(f"Raw response string: {response_str}")
            print("---------------------------------------\n")

if __name__ == "__main__":
    if os.getenv("RUN_RAG_SERVICE_LOCAL_TEST") == "true":
        asyncio.run(local_rag_test())
    else:
        rag_logger.info("--- Starting RAG Service (FastMCP for mcpo via direct run) ---")
        rag_mcp_application.run() # 这会启动FastAPI Uvicorn服务器