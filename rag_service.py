# zhz_agent/rag_service.py

import os
import json
import asyncio
import traceback
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
import time
import logging
import sys # <--- 确保导入 sys

# MCP 框架导入
from mcp.server.fastmcp import FastMCP, Context # 确保从 fastmcp 导入


# --- 配置 rag_service 的专用日志 ---
_rag_service_py_dir = os.path.dirname(os.path.abspath(__file__))
_rag_service_log_file = os.path.join(_rag_service_py_dir, 'rag_service_debug.log')

rag_logger = logging.getLogger("RagServiceLogger") # 给一个独特的名字
rag_logger.setLevel(logging.DEBUG)
rag_logger.propagate = False # 不传递给根记录器

if rag_logger.hasHandlers():
    rag_logger.handlers.clear()

try:
    _file_handler = logging.FileHandler(_rag_service_log_file, mode='w') # 每次覆盖
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
    generate_clarification_options, 
    NO_ANSWER_PHRASE_ANSWER_CLEAN
)
from zhz_agent.vector import VectorRetriever
from zhz_agent.kg import KGRetriever
from zhz_agent.fusion import FusionEngine # FusionEngine 现在接收 logger
from zhz_agent.bm25 import BM25Retriever


from dotenv import load_dotenv
load_dotenv()

# --- 应用上下文 Dataclass ---
@dataclass
class AppContext:
    vector_retriever: VectorRetriever
    kg_retriever: KGRetriever
    bm25_retriever: BM25Retriever
    fusion_engine: FusionEngine

# --- MCP 服务器生命周期管理 ---
@asynccontextmanager
async def app_lifespan_for_rag_service(server: FastMCP) -> AsyncIterator[AppContext]:
    rag_logger.info("--- RAG Service (FastMCP): 正在初始化 RAG 组件 ---")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")

    vector_retriever_instance: Optional[VectorRetriever] = None
    kg_retriever_instance: Optional[KGRetriever] = None
    bm25_retriever_instance: Optional[BM25Retriever] = None
    fusion_engine_instance: Optional[FusionEngine] = None

    try:
        vector_retriever_instance = VectorRetriever(data_path=data_dir)
        rag_logger.info("RAG Service: VectorRetriever 初始化成功。")
    except Exception as e:
        rag_logger.error(f"RAG Service: VectorRetriever 初始化失败: {e}", exc_info=True)
        raise RuntimeError(f"VectorRetriever 初始化失败: {e}") from e
    
    try:
        kg_retriever_instance = KGRetriever(data_path=data_dir, llm_cypher_generator_func=generate_cypher_query)
        rag_logger.info("RAG Service: KGRetriever 初始化成功。")
    except Exception as e:
        rag_logger.error(f"RAG Service: KGRetriever 初始化失败: {e}", exc_info=True)
        raise RuntimeError(f"KGRetriever 初始化失败: {e}") from e
        
    try:
        bm25_retriever_instance = BM25Retriever(data_path=data_dir)
        rag_logger.info("RAG Service: BM25Retriever 初始化成功。")
    except Exception as e:
        rag_logger.error(f"RAG Service: BM25Retriever 初始化失败: {e}", exc_info=True)
        raise RuntimeError(f"BM25Retriever 初始化失败: {e}") from e
    
    try:
        fusion_engine_instance = FusionEngine(logger=rag_logger) # 传递 logger
        rag_logger.info("RAG Service: FusionEngine 初始化成功 (已传入logger)。")
    except Exception as e:
        rag_logger.error(f"RAG Service: FusionEngine 初始化失败: {e}", exc_info=True)
        raise RuntimeError(f"FusionEngine 初始化失败: {e}") from e

    rag_logger.info("--- RAG Service (FastMCP): RAG 组件初始化完成。---")

    ctx = AppContext(
        vector_retriever=vector_retriever_instance,
        kg_retriever=kg_retriever_instance,
        bm25_retriever=bm25_retriever_instance,
        fusion_engine=fusion_engine_instance
    )
    try:
        yield ctx
    finally:
        rag_logger.info("--- RAG Service (FastMCP): 正在清理资源 ---")
        if kg_retriever_instance:
            kg_retriever_instance.close() 
        rag_logger.info("--- RAG Service (FastMCP): 清理完成 ---")

# --- 初始化 FastMCP 应用 ---
rag_mcp_application = FastMCP(
    name="zhz_agent_service", 
    description="Hybrid RAG 服务，提供多路召回、融合与答案生成功能。",
    lifespan=app_lifespan_for_rag_service,
)

# --- MCP 工具定义 ---
@rag_mcp_application.tool()
async def query_rag(
    ctx: Context,
    query: str,
    top_k_vector: int = 3,
    top_k_kg: int = 2,
    top_k_bm25: int = 3
) -> str: # FastMCP 工具函数期望返回一个字符串 (通常是JSON字符串)
    rag_logger.info(f"\n--- RAG Service: 接收到查询: '{query}' ---")
    rag_logger.info(f"    top_k_vector: {top_k_vector}, top_k_kg: {top_k_kg}, top_k_bm25: {top_k_bm25}")
    start_time_total = time.time()

    app_ctx: AppContext = ctx.request_context.lifespan_context
    response_payload = {} 
    original_query_for_response = query 
    final_json_output = "" # 初始化最终的JSON输出

    try:
        # --- LLM 驱动的意图分类和澄清触发 ---
        rag_logger.info(f"--- RAG Service: [TIME] 开始 LLM 驱动的意图分类 at {time.time() - start_time_total:.2f}s ---")
        start_time_intent = time.time() # 确保start_time_intent已定义
        intent_classification_result = await generate_intent_classification(query)
        rag_logger.info(f"--- RAG Service: [TIME] 结束 LLM 驱动的意图分类, 耗时: {time.time() - start_time_intent:.2f}s ---")
        rag_logger.info(f"--- RAG Service: 意图分类结果: {intent_classification_result}")

        if intent_classification_result.get("clarification_needed"):
            # rag_logger.info(...) # 使用 rag_logger 替代 print
            uncertainty_reason = intent_classification_result.get("reason", "查询可能存在歧义或信息不足。")
            clarification_question_text = await generate_clarification_question(query, uncertainty_reason)
            response_payload = {
                "status": "clarification_needed",
                "clarification_question": clarification_question_text,
                "original_query": original_query_for_response,
                "debug_info": {"uncertainty_reason": uncertainty_reason, "source": "intent_classification"}
            }
            rag_logger.info(f"--- RAG Service: 需要澄清，返回: {response_payload}")
            final_json_output = json.dumps(response_payload, ensure_ascii=False)
            # --- 在所有返回路径前添加 flush ---
            sys.stdout.flush()
            sys.stderr.flush()
            return final_json_output

        # --- 如果不需要澄清，则继续RAG流程 ---
        rag_logger.info(f"--- RAG Service: LLM 意图分类结果：查询清晰，无需澄清。---")
        rag_logger.info(f"--- RAG Service: [TIME] 开始查询扩展 for: {query} at {time.time() - start_time_total:.2f}s ---")
        start_time_expansion = time.time()
        expanded_queries = await generate_expanded_queries(query)
        rag_logger.info(f"--- RAG Service: 扩展查询列表 (共 {len(expanded_queries)} 个): {expanded_queries} ---")
        rag_logger.info(f"--- RAG Service: [TIME] 结束查询扩展, 耗时: {time.time() - start_time_expansion:.2f}s ---")
        
        all_raw_retrievals: List[RetrievedDocument] = []
        tasks = []
        task_names = [] 

        for i_eq, q_ext in enumerate(expanded_queries):
            rag_logger.debug(f"为扩展查询 '{q_ext}' (索引 {i_eq}) 创建召回任务...")
            tasks.append(app_ctx.vector_retriever.retrieve(q_ext, top_k_vector))
            task_names.append(f"VectorRetrieve_EQ{i_eq}")
            tasks.append(app_ctx.kg_retriever.retrieve(q_ext, top_k_kg))
            task_names.append(f"KGRetrieve_EQ{i_eq}")
            tasks.append(app_ctx.bm25_retriever.retrieve(q_ext, top_k_bm25))
            task_names.append(f"BM25Retrieve_EQ{i_eq}")
        
        rag_logger.info(f"--- RAG Service: [TIME] 开始并行召回 (共 {len(tasks)} 个任务) at {time.time() - start_time_total:.2f}s ---")
        start_time_retrieval = time.time()
        all_results_nested = await asyncio.gather(*tasks, return_exceptions=True)
        rag_logger.info(f"--- RAG Service: [TIME] 结束并行召回, 耗时: {time.time() - start_time_retrieval:.2f}s ---")
        
        kg_retrieval_error_message: Optional[str] = None
        rag_logger.debug(f"--- RAG Service: 开始处理 {len(all_results_nested)} 个召回任务的结果 ---")

        for i, result_item in enumerate(all_results_nested):
            task_name = task_names[i] 
            if isinstance(result_item, Exception):
                error_msg = f"RAG Service WARNING: 任务 '{task_name}' 执行失败: {type(result_item).__name__} - {str(result_item)}"
                rag_logger.error(error_msg, exc_info=True)
                if "KGRetrieve" in task_name: 
                    kg_retrieval_error_message = f"知识图谱检索任务 '{task_name}' 失败: {str(result_item)}"
            elif isinstance(result_item, list):
                rag_logger.debug(f"任务 '{task_name}' 成功返回 {len(result_item)} 个结果。内容预览 (最多前2项):")
                for doc_idx, doc_item in enumerate(result_item[:2]):
                    rag_logger.debug(f"  Doc {doc_idx}: source={doc_item.source_type}, content='{str(doc_item.content)[:100]}...'")
                all_raw_retrievals.extend(result_item)
            else:
                rag_logger.warning(f"RAG Service WARNING: 任务 '{task_name}' 返回了预料之外的结果类型: {type(result_item)}")
        
        rag_logger.info(f"RAG Service: 总计从各路召回有效合并的结果数: {len(all_raw_retrievals)}")
        
        rag_logger.debug(f"--- RAG Service: 准备送入 FusionEngine 的 all_raw_retrievals (共 {len(all_raw_retrievals)} 条) ---")
        for i_doc, doc_retrieved in enumerate(all_raw_retrievals):
            rag_logger.debug(f"  Doc {i_doc}: type={doc_retrieved.source_type}, score={doc_retrieved.score}, content='{str(doc_retrieved.content)[:150]}...'")

        if kg_retrieval_error_message and not all_raw_retrievals:
            response_payload = {
                "status": "error",
                "error_code": "KNOWLEDGE_GRAPH_FAILURE_NO_FALLBACK",
                "error_message": kg_retrieval_error_message,
                "original_query": original_query_for_response,
                "debug_info": {"error_source": "knowledge_graph_retrieval", "details": kg_retrieval_error_message}
            }
            rag_logger.error(f"--- RAG Service: 'query_rag' 因KG查询失败且无其他结果，提前返回错误 ---") # 使用 rag_logger
            final_json_output = json.dumps(response_payload, ensure_ascii=False)
            sys.stdout.flush()
            sys.stderr.flush()
            return final_json_output

        # rag_logger.info(...) # 使用 rag_logger 替代 print
        # print(f"RAG Service: 总计原始召回结果数 (可能包含来自失败检索的空列表): {len(all_raw_retrievals)}")

        if not all_raw_retrievals: 
            response_payload = {
                "status": "success", 
                "final_answer": "抱歉，根据您提供的查询，未能从知识库中找到相关信息。",
                "original_query": original_query_for_response,
                "debug_info": {
                    "message": "No documents retrieved from any source.",
                    "kg_retrieval_error": kg_retrieval_error_message if kg_retrieval_error_message else "N/A"
                }
            }
            final_json_output = json.dumps(response_payload, ensure_ascii=False)
            sys.stdout.flush()
            sys.stderr.flush()
            return final_json_output

        rag_logger.info(f"--- RAG Service: [TIME] 开始结果融合 at {time.time() - start_time_total:.2f}s ---")
        start_time_fusion = time.time()
        fused_context_text = await app_ctx.fusion_engine.fuse_results(all_raw_retrievals, query) 
        rag_logger.info(f"--- RAG Service: [TIME] 结束结果融合, 耗时: {time.time() - start_time_fusion:.2f}s ---")
        
        rag_logger.info(f"\n--- RAG Service: FUSED CONTEXT TEXT for generate_answer_from_context ---")
        rag_logger.info(f"{fused_context_text}") 
        rag_logger.info(f"--- END OF FUSED CONTEXT TEXT ---\n")
        
        if fused_context_text == "未在知识库中找到相关信息。":
            final_answer = "根据现有知识，未能找到您查询的相关信息。"
            response_payload = {
                "status": "success",
                "final_answer": final_answer,
                "original_query": original_query_for_response,
                "debug_info": {
                    "message": "No relevant context found after fusion.",
                    "total_raw_retrievals_count": len(all_raw_retrievals),
                    "kg_retrieval_error": kg_retrieval_error_message if kg_retrieval_error_message else "N/A"
                }
            }
        else:
            rag_logger.info(f"--- RAG Service: [TIME] 开始最终答案生成 at {time.time() - start_time_total:.2f}s ---") # 使用 rag_logger
            start_time_answer_gen = time.time()
            final_answer = await generate_answer_from_context(query, fused_context_text)
            rag_logger.info(f"--- RAG Service: [TIME] 结束最终答案生成, 耗时: {time.time() - start_time_answer_gen:.2f}s ---") # 使用 rag_logger

            if not final_answer or final_answer.strip() == NO_ANSWER_PHRASE_ANSWER_CLEAN:
                final_answer = "根据您提供的信息，我暂时无法给出明确的回答。"
            response_payload = {
                "status": "success",
                "final_answer": final_answer,
                "original_query": original_query_for_response,
                "debug_info": {
                    "total_raw_retrievals_count": len(all_raw_retrievals),
                    "kg_retrieval_error": kg_retrieval_error_message if kg_retrieval_error_message else "N/A"
                }
            }
        
        # --- 这里是正常执行完毕的返回路径 ---
        final_json_output = json.dumps(response_payload, ensure_ascii=False) # 赋值给 final_json_output
        rag_logger.info(f"--- RAG Service: 'query_rag' 工具成功执行完毕, 总耗时: {time.time() - start_time_total:.2f}s, 准备返回JSON响应 ---")
        rag_logger.info(f"--- RAG Service: 准备返回的最终JSON (成功): {final_json_output[:500]}...")
        
    except Exception as e:
        # rag_logger.error(...) # 使用 rag_logger 替代 print
        rag_logger.error(f"RAG Service CRITICAL ERROR in 'query_rag' tool: {type(e).__name__} - {str(e)}", exc_info=True)
        # traceback.print_exc() # logging 的 exc_info=True 会处理堆栈

        response_payload = { 
            "status": "error",
            "error_code": "RAG_SERVICE_INTERNAL_ERROR",
            "error_message": f"RAG服务内部发生未预期错误: {str(e)}",
            "original_query": original_query_for_response,
            "debug_info": {"exception_type": type(e).__name__, "traceback_snippet": traceback.format_exc(limit=1)}
        }
        final_json_output = json.dumps(response_payload, ensure_ascii=False) # 赋值给 final_json_output
        rag_logger.info(f"--- RAG Service: 'query_rag' 工具因异常结束, 总耗时: {time.time() - start_time_total:.2f}s, 准备返回错误JSON响应 ---")
        rag_logger.info(f"--- RAG Service: 准备返回的最终JSON (异常): {final_json_output[:500]}...")
    
    # --- 确保在函数末尾，所有路径都有一个明确的返回，并且之前有 flush ---
    sys.stdout.flush()
    sys.stderr.flush()
    return final_json_output

if __name__ == "__main__":
    # 这部分代码在使用 -m zhz_agent.rag_service 时不会被 mcpo 直接执行
    # mcpo 是通过导入并调用 rag_mcp_application 对象来工作的
    # 但如果想独立测试这个文件，可以保留
    rag_logger.info("--- 启动 RAG Service (FastMCP for mcpo) ---") # 使用 rag_logger
    rag_mcp_application.run()