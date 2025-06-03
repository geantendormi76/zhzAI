# zhz_rag/core_rag/kg_retriever.py
import os
import json
import kuzu
import pandas as pd
from typing import List, Dict, Any, Optional, Callable, Iterator # 确保 Iterator 已导入
import logging
from contextlib import contextmanager

# 导入您的Cypher生成函数和Schema描述
from zhz_rag.llm.sglang_wrapper import generate_cypher_query # 确保路径正确
from zhz_rag.config.constants import NEW_KG_SCHEMA_DESCRIPTION # 确保路径正确

# 日志配置
kg_logger = logging.getLogger(__name__) 
# 确保 kg_logger 的级别和处理器已在 zhz_rag_mcp_service 或其他主入口配置，
# 或者在这里为它单独配置 handler 和 formatter，例如：
if not kg_logger.hasHandlers():
    kg_logger.setLevel(logging.DEBUG) # 开发时可以设为 DEBUG
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    ch.setFormatter(formatter)
    kg_logger.addHandler(ch)
    kg_logger.propagate = False # 避免重复日志（如果根logger也配置了handler）
kg_logger.info("KGRetriever (KuzuDB) logger initialized/reconfirmed.")


class KGRetriever:
    KUZU_DB_PATH_ENV = os.getenv("KUZU_DB_PATH", "/home/zhz/zhz_agent/zhz_rag/stored_data/kuzu_default_db")

    def __init__(self, db_path: Optional[str] = None, llm_cypher_generator_func: Callable = generate_cypher_query):
        self.db_path = db_path if db_path else self.KUZU_DB_PATH_ENV
        self.llm_cypher_generator_func = llm_cypher_generator_func
        self._db: Optional[kuzu.Database] = None
        kg_logger.info(f"KGRetriever (KuzuDB) __init__ called. Attempting to connect to DB path: {self.db_path}")
        self._connect_to_kuzu()
        # 这条日志现在移到 _connect_to_kuzu 成功之后打印

    def _connect_to_kuzu(self):
        kg_logger.info(f"Attempting to load KuzuDB from path: {self.db_path}")
        try:
            if not os.path.exists(self.db_path):
                kg_logger.error(f"KuzuDB path does not exist: {self.db_path}. KGRetriever cannot connect.")
                self._db = None
                return # 明确返回
            
            # 对于检索，通常只读即可，除非有特殊写需求
            # 如果 mcpo 服务可能并发访问，需要考虑 KuzuDB 的并发处理能力和锁机制
            self._db = kuzu.Database(self.db_path, read_only=True) 
            kg_logger.info(f"Successfully loaded KuzuDB from {self.db_path}. KGRetriever (KuzuDB) initialized and connected.")
        except Exception as e:
            kg_logger.error(f"Failed to connect to KuzuDB at {self.db_path}: {e}", exc_info=True)
            self._db = None

    @contextmanager
    def _get_connection(self) -> Iterator[kuzu.Connection]:
        if not self._db:
            kg_logger.warning("KuzuDB database object is None in _get_connection. Attempting to reconnect...")
            self._connect_to_kuzu() # 尝试重新连接
            if not self._db: # 再次检查
                kg_logger.error("KuzuDB reconnection failed. Cannot get a connection.")
                raise ConnectionError("KuzuDB is not connected or failed to reconnect. Cannot get a connection.")
        
        conn = None # 初始化 conn
        try:
            conn = kuzu.Connection(self._db)
            kg_logger.debug("KuzuDB connection obtained.")
            yield conn
        except Exception as e_conn: # 捕获 kuzu.Connection() 可能的异常
            kg_logger.error(f"Failed to create KuzuDB connection object: {e_conn}", exc_info=True)
            raise ConnectionError(f"Failed to create KuzuDB connection: {e_conn}")
        finally:
            # Kuzu Connection 对象没有显式的 close() 方法。
            # 它通常在其关联的 Database 对象被销毁时或垃圾回收时关闭。
            kg_logger.debug("KuzuDB connection context manager exiting.")
            pass 

    def close(self):
        kg_logger.info(f"Closing KuzuDB for retriever using path: {self.db_path}")
        if self._db:
            # KuzuDB Database 对象在其 __del__ 方法中处理关闭和资源释放。
            # 显式删除引用有助于触发垃圾回收，但不保证立即关闭。
            # KuzuDB 没有显式的 db.close() 方法。
            del self._db
            self._db = None
            kg_logger.info("KuzuDB Database object dereferenced (closed).")

    def execute_cypher_query_sync(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self._db:
            kg_logger.error("KuzuDB not initialized in execute_cypher_query_sync. Cannot execute query.")
            return []
        
        kg_logger.info(f"--- Executing KuzuDB Cypher ---")
        kg_logger.info(f"Query: {query}")
        kg_logger.info(f"Params: {parameters if parameters else 'No parameters'}")
        
        results_list: List[Dict[str, Any]] = []
        try:
            with self._get_connection() as conn: # 使用上下文管理器获取连接
                prepared_statement = conn.prepare(query)
                # 注意：KuzuDB 的 execute 方法对参数的处理方式。
                # 如果 parameters 为 None 或空字典，应传递 None 或 {}。
                # 如果查询本身不包含参数占位符，传递参数字典可能会导致错误。
                # 我们需要确保Cypher查询中的参数占位符（如 $param）与parameters字典中的键匹配。
                
                actual_params = parameters if parameters else {} # 确保是字典
                query_result = conn.execute(prepared_statement, **actual_params) # 使用 ** 解包参数
                
                df = query_result.get_as_df()
                results_list = df.to_dict(orient='records')
                
                kg_logger.info(f"KuzuDB Cypher executed. Records count: {len(results_list)}")
                if results_list:
                    kg_logger.debug(f"First KuzuDB record (sample): {json.dumps(results_list[0], ensure_ascii=False, indent=2, default=str)}")
                else:
                    kg_logger.debug("KuzuDB query returned no records.")
        except RuntimeError as kuzu_runtime_error: # KuzuDB Python API 通常抛出 RuntimeError
             kg_logger.error(f"KuzuDB RuntimeError during Cypher execution: '{query}' with params: {parameters}. Error: {kuzu_runtime_error}", exc_info=True)
             # 可以考虑将 KuzuDB 的错误信息包装后向上抛出或返回
             # return [{"error": f"KuzuDB execution error: {kuzu_runtime_error}"}] 
        except ConnectionError as conn_err: # 如果 _get_connection 内部抛出
             kg_logger.error(f"KuzuDB ConnectionError during Cypher execution: {conn_err}", exc_info=True)
        except Exception as e:
            kg_logger.error(f"Unexpected error executing KuzuDB Cypher query: '{query}' with params: {parameters}. Error: {e}", exc_info=True)
        return results_list

    # ... ( _format_kuzu_record_for_retrieval 和 retrieve_with_llm_cypher 保持不变，
    # 但 retrieve_with_llm_cypher 内部对 execute_cypher_query_sync 的调用现在会经过新的错误处理) ...
    def _format_kuzu_record_for_retrieval(self, record_data: Dict[str, Any]) -> str:
        # ... (保持不变)
        parts = []
        for key, value in record_data.items():
            if isinstance(value, dict): 
                if 'label' in value and 'text' in value: 
                    parts.append(f"{key}({value['text']}:{value['label']})")
                elif '_label' in value and '_src' in value and '_dst' in value: 
                     parts.append(f"{key}(TYPE={value['_label']})")
                else:
                    value_str = json.dumps(value, ensure_ascii=False, default=str) 
                    if len(value_str) > 100: value_str = value_str[:100] + "..."
                    parts.append(f"{key}: {value_str}")
            elif value is not None:
                parts.append(f"{key}: {str(value)}")
        return " | ".join(parts) if parts else "No specific details found in this KuzuDB record."


    async def retrieve_with_llm_cypher(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        kg_logger.info(f"Starting KG retrieval (KuzuDB) with LLM-generated Cypher for query: '{query}', top_k: {top_k}")
        
        kg_logger.info(f"Calling LLM to generate Cypher query for KuzuDB...")
        # 将接收结果的变量名统一为 cypher_query_or_unable_msg
        cypher_query_or_unable_msg = await self.llm_cypher_generator_func(
            user_question=query,
            kg_schema_description=NEW_KG_SCHEMA_DESCRIPTION 
        )
        kg_logger.info(f"LLM generated Cypher query/message for KuzuDB:\n---\n{cypher_query_or_unable_msg}\n---")

        # --- [使用正确的变量名进行检查] ---
        if not cypher_query_or_unable_msg or cypher_query_or_unable_msg == "无法生成Cypher查询.":
            kg_logger.warning("LLM could not generate a valid Cypher query for KuzuDB or returned 'unable to generate' message.")
            return [] # 直接返回空列表

        # 如果是有效的 Cypher 查询，则继续
        cypher_to_execute = cypher_query_or_unable_msg # <--- 使用正确的变量名
        
        cypher_query_with_limit: str # 明确类型
        if "LIMIT" not in cypher_to_execute.upper(): # 保持大小写不敏感的检查
            cypher_query_with_limit = f"{cypher_to_execute} LIMIT {top_k}"
        else: 
            cypher_query_with_limit = cypher_to_execute
            kg_logger.info(f"Query already contains LIMIT, using as is: {cypher_query_with_limit}")

        results_from_kuzu = self.execute_cypher_query_sync(cypher_query_with_limit)
        
        retrieved_docs_for_rag: List[Dict[str, Any]] = []
        for record_dict in results_from_kuzu:
            content_str = self._format_kuzu_record_for_retrieval(record_dict)
            doc_data = {
                "source_type": "knowledge_graph_kuzu",
                "content": content_str,
                "score": 1.0, 
                # 使用 cypher_to_execute (即原始的、未加 LIMIT 的 Cypher) 进行记录
                "metadata": {"cypher_query": cypher_to_execute, "original_query": query, "raw_kuzu_record": record_dict}
            }
            retrieved_docs_for_rag.append(doc_data)

        kg_logger.info(f"Retrieved {len(retrieved_docs_for_rag)} documents from KuzuDB using LLM-generated Cypher.")
        return retrieved_docs_for_rag