# zhz_rag/core_rag/kg_retriever.py
import os
import json
import kuzu
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Callable, Iterator, TYPE_CHECKING
import logging
from contextlib import contextmanager
import hashlib # 确保导入 hashlib

# --- 从项目中导入必要的模块 ---
from zhz_rag.llm.llm_interface import extract_entities_for_kg_query
from zhz_rag.config.pydantic_models import ExtractedEntitiesAndRelationIntent, IdentifiedEntity
if TYPE_CHECKING:
    from zhz_rag.llm.local_model_handler import LocalModelHandler


# 日志配置
kg_logger = logging.getLogger(__name__)
if not kg_logger.hasHandlers():
    kg_logger.setLevel(logging.DEBUG) # 可以设置为 INFO 或 DEBUG
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    ch.setFormatter(formatter)
    kg_logger.addHandler(ch)
    kg_logger.propagate = False
kg_logger.info("KGRetriever (KuzuDB) logger initialized/reconfirmed.")


class KGRetriever:
    KUZU_DB_PATH_ENV = os.getenv("KUZU_DB_PATH", "/home/zhz/zhz_agent/zhz_rag/stored_data/kuzu_default_db")

    def __init__(self, db_path: Optional[str] = None, embedder: Optional['LocalModelHandler'] = None): # <--- 修改类型提示
        self.db_path = db_path if db_path else self.KUZU_DB_PATH_ENV
        self._db: Optional[kuzu.Database] = None
        self._embedder = embedder 
        kg_logger.info(f"KGRetriever (KuzuDB) __init__ called. DB path: {self.db_path}")
        if self._embedder:
            kg_logger.info(f"KGRetriever initialized with embedder: {type(self._embedder)}")
        else:
            kg_logger.warning("KGRetriever initialized WITHOUT an embedder. Vector search will not be available.")
        self._connect_to_kuzu() # <--- _connect_to_kuzu 会被调用

    def _connect_to_kuzu(self):
        kg_logger.info(f"Attempting to load KuzuDB from path: {self.db_path}")
        try:
            if not os.path.exists(self.db_path):
                kg_logger.error(f"KuzuDB path does not exist: {self.db_path}. KGRetriever cannot connect.")
                self._db = None
                return
            
            # --- 修改开始：尝试以读写模式打开 ---
            kg_logger.warning("KGRetriever: DIAGNOSTIC MODE - Opening KuzuDB in READ_WRITE mode.")
            self._db = kuzu.Database(self.db_path, read_only=False) 
            # --- 修改结束 ---
            kg_logger.info(f"Successfully loaded KuzuDB from {self.db_path} (read_only=False FOR DIAGNOSTICS).")

            with self._get_connection() as conn: # 使用 self._get_connection 来获取连接
                kg_logger.info("KGRetriever: Attempting to LOAD VECTOR extension for this session during _connect_to_kuzu...")
                conn.execute("LOAD VECTOR;")
                kg_logger.info("KGRetriever: VECTOR extension loaded successfully for this retriever instance.")

                # --- 添加开始：在连接和加载扩展后立即检查索引 ---
                kg_logger.info("KGRetriever: Verifying indexes immediately after connecting and loading vector extension...")
                show_indexes_result_init = conn.execute("CALL SHOW_INDEXES() RETURN *;")
                if show_indexes_result_init:
                    indexes_df_init = pd.DataFrame(show_indexes_result_init.get_as_df())
                    kg_logger.info(f"KGRetriever: Indexes visible during __init__ / _connect_to_kuzu:\n{indexes_df_init.to_string()}")
                    if not indexes_df_init.empty and 'index name' in indexes_df_init.columns and 'entity_embedding_idx' in indexes_df_init['index name'].tolist():
                        kg_logger.info("KGRetriever __init__: 'entity_embedding_idx' IS VISIBLE at initialization.")
                    else:
                        kg_logger.warning("KGRetriever __init__: 'entity_embedding_idx' IS NOT VISIBLE at initialization.")
                else:
                    kg_logger.warning("KGRetriever __init__: CALL SHOW_INDEXES() did not return a result during initialization.")
                # --- 添加结束 ---

        except Exception as e:
            kg_logger.error(f"Failed to connect to KuzuDB at {self.db_path} or load/verify extension during init: {e}", exc_info=True)
            self._db = None
            # 如果在初始化时就失败，可能需要让服务启动失败或进入降级模式
            raise ConnectionError(f"Failed to initialize KGRetriever's KuzuDB connection or vector setup: {e}") from e

    @contextmanager
    def _get_connection(self) -> Iterator[kuzu.Connection]:
        if not self._db:
            kg_logger.warning("KuzuDB database object is None in _get_connection. Attempting to reconnect...")
            self._connect_to_kuzu()
            if not self._db:
                kg_logger.error("KuzuDB reconnection failed. Cannot get a connection.")
                raise ConnectionError("KuzuDB is not connected or failed to reconnect. Cannot get a connection.")
        conn = None
        try:
            conn = kuzu.Connection(self._db)
            kg_logger.debug("KuzuDB connection obtained.")
            yield conn
        except Exception as e_conn:
            kg_logger.error(f"Failed to create KuzuDB connection object: {e_conn}", exc_info=True)
            raise ConnectionError(f"Failed to create KuzuDB connection: {e_conn}")
        finally:
            kg_logger.debug("KuzuDB connection context manager exiting.")
            pass # KuzuDB Connection 对象没有显式的 close() 方法，依赖其 __del__

    def close(self):
        kg_logger.info(f"Closing KuzuDB for retriever using path: {self.db_path}")
        if self._db:
            del self._db # 依赖 KuzuDB Database 的 __del__ 方法进行清理和锁释放
            self._db = None
            kg_logger.info("KuzuDB Database object dereferenced (closed).")

    def _execute_cypher_query_sync(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self._db:
            kg_logger.error("KuzuDB not initialized in _execute_cypher_query_sync. Cannot execute query.")
            return []
        kg_logger.info(f"--- Executing KuzuDB Cypher --- Query: {query}, Params: {parameters if parameters else 'No parameters'}")
        results_list: List[Dict[str, Any]] = []
        try:
            with self._get_connection() as conn:
                # --- 添加开始 ---
                is_vector_query = "QUERY_VECTOR_INDEX" in query.upper()
                if is_vector_query: 
                    try:
                        conn.execute("LOAD VECTOR;")
                        kg_logger.debug("LOAD VECTOR executed before QUERY_VECTOR_INDEX in _execute_cypher_query_sync.")
                        
                        # 打印当前可见的索引
                        show_indexes_result = conn.execute("CALL SHOW_INDEXES() RETURN *;")
                        indexes_df = pd.DataFrame(show_indexes_result.get_as_df())
                        kg_logger.info(f"KGRetriever: Indexes visible to current connection before vector query:\n{indexes_df.to_string()}")
                        if not indexes_df.empty and 'entity_embedding_idx' in indexes_df['index name'].tolist():
                            kg_logger.info("KGRetriever: 'entity_embedding_idx' IS VISIBLE to current connection.")
                        else:
                            kg_logger.warning("KGRetriever: 'entity_embedding_idx' IS NOT VISIBLE to current connection.")
                            
                    except Exception as e_load_vec_exec:
                        kg_logger.warning(f"Failed to execute LOAD VECTOR or SHOW_INDEXES in _execute_cypher_query_sync (continuing anyway): {e_load_vec_exec}")
                actual_params = parameters if parameters else {}
                query_result = conn.execute(query, parameters=actual_params)
                if hasattr(query_result, 'get_as_df'):
                    df = pd.DataFrame(query_result.get_as_df())
                    results_list = df.to_dict(orient='records')
                elif isinstance(query_result, list):
                    kg_logger.info("KuzuDB query did not return a DataFrame-convertible result, result is a list.")
                    # results_list = query_result # 如果列表内已经是dict
                else:
                    kg_logger.info(f"KuzuDB query did not return a DataFrame-convertible result. Type: {type(query_result)}")

                kg_logger.info(f"KuzuDB Cypher executed. Records count (from DataFrame): {len(results_list)}")
                if results_list: 
                    kg_logger.debug(f"First KuzuDB record (from DataFrame): {str(results_list[0])[:200]}")
                elif not hasattr(query_result, 'get_as_df'):
                     kg_logger.debug(f"KuzuDB query result (raw, not DataFrame): {str(query_result)[:200]}")
        except RuntimeError as kuzu_runtime_error:
             kg_logger.error(f"KuzuDB RuntimeError during Cypher execution: '{query}' with params: {parameters}. Error: {kuzu_runtime_error}", exc_info=True)
        except ConnectionError as conn_err:
             kg_logger.error(f"KuzuDB ConnectionError during Cypher execution: {conn_err}", exc_info=True)
        except TypeError as e_type:
            kg_logger.error(f"TypeError during KuzuDB Cypher execution: '{query}' with params: {parameters}. Error: {e_type}", exc_info=True)
        except Exception as e:
            kg_logger.error(f"Unexpected error executing KuzuDB Cypher query: '{query}' with params: {parameters}. Error: {e}", exc_info=True)
        return results_list

    def _format_kuzu_records_for_retrieval(self, records: List[Dict[str, Any]], query_context: str = "") -> List[Dict[str, Any]]:
        formatted_docs = []
        if not records:
            return formatted_docs

        for record_data in records:
            parts = []
            # 优先展示的、信息量大的key
            primary_info_keys = ['text', 'label', 'id_prop', 'name', 'related_text', 'related_label', 'relationship_type', 'source_node_text', '_score']
            
            for key in primary_info_keys:
                if key in record_data and record_data[key] is not None:
                    if key == '_score' and isinstance(record_data[key], float):
                        parts.append(f"Similarity Score: {record_data[key]:.4f}")
                    else:
                        parts.append(f"{key.replace('_', ' ').capitalize()}: {record_data[key]}")
            
            # 其他非核心但有用的信息
            additional_info_parts = []
            for key, value in record_data.items():
                if key not in primary_info_keys and value is not None:
                    if isinstance(value, dict): # 例如 KuzuDB 返回的复杂节点/关系对象
                        value_str = json.dumps(value, ensure_ascii=False, default=str)
                        if len(value_str) > 70: value_str = value_str[:70] + "..."
                        additional_info_parts.append(f"{key}: {value_str}")
                    else:
                        additional_info_parts.append(f"{key}: {str(value)}")
            
            if additional_info_parts:
                parts.append("Other Info: " + " | ".join(additional_info_parts))
            
            content_str = " | ".join(parts) if parts else "Retrieved graph data node/relation."
            
            # 确保元数据中也包含关键信息，方便调试和后续处理
            doc_metadata = {
                "original_user_query_for_kg": query_context,
                "kuzu_retrieved_id_prop": record_data.get("id_prop"),
                "kuzu_retrieved_text": record_data.get("text") or record_data.get("related_text"),
                "kuzu_retrieved_label": record_data.get("label") or record_data.get("related_label"),
                "kuzu_retrieved_relationship": record_data.get("relationship_type"),
                "kuzu_retrieved_score": record_data.get("_score")
            }
            # 清理元数据中的None值
            doc_metadata = {k: v for k, v in doc_metadata.items() if v is not None}

            doc_data = {
                "source_type": "knowledge_graph_kuzu",
                "content": content_str,
                "score": record_data.get('_score', 0.0), # 使用记录中的_score，如果没有则为0
                "metadata": doc_metadata
            }
            formatted_docs.append(doc_data)
        return formatted_docs

    async def retrieve(self, user_query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        kg_logger.info(f"Starting KG retrieval for query: '{user_query}', top_k: {top_k}")

        extracted_info: Optional[ExtractedEntitiesAndRelationIntent] = await extract_entities_for_kg_query(user_query)

        search_text_for_vector: str
        if extracted_info and extracted_info.entities and extracted_info.entities[0].text:
            primary_entity_text = extracted_info.entities[0].text
            search_text_for_vector = primary_entity_text
            kg_logger.info(f"LLM extracted: Entities: {[e.model_dump() for e in extracted_info.entities]}, Relation Hint: {extracted_info.relation_hint}")
            kg_logger.info(f"Using primary extracted entity text for vector search: '{search_text_for_vector}'")
        else:
            search_text_for_vector = user_query
            kg_logger.info(f"No primary entities extracted or entity text is empty. Using original user query for vector search: '{user_query}'")

        all_kuzu_records: List[Dict[str, Any]] = []
        
        # --- Strategy 1: KuzuDB Vector Index Search (Preferred) ---
        if self._embedder: # self._embedder 现在是 LocalModelHandler 实例
            try:
                kg_logger.info(f"Generating embedding for vector search text: '{search_text_for_vector}' using LocalModelHandler via KGRetriever.")
                query_vector_list = await self._embedder.embed_query(search_text_for_vector) # <--- 添加 await

                if not query_vector_list: 
                    raise ValueError(f"Embedding generation failed in KGRetriever for vector search text: '{search_text_for_vector}'")
                
                vector_search_query = """ 
                    CALL QUERY_VECTOR_INDEX('ExtractedEntity', 'entity_embedding_idx', $query_vector, $top_k_param)
                    YIELD node, distance
                    RETURN node.id_prop AS id_prop, node.text AS text, node.label AS label, distance AS _score
                """
                vector_params = {"query_vector": query_vector_list, "top_k_param": top_k}
                
                kg_logger.info(f"Executing KuzuDB vector search. Table: 'ExtractedEntity', Index: 'entity_embedding_idx', Top K: {top_k}")
                vector_results = self._execute_cypher_query_sync(vector_search_query, vector_params)
                
                if vector_results:
                    # 注意：返回的 _score 是 distance，越小越相似
                    all_kuzu_records.extend(vector_results)
                    kg_logger.info(f"Retrieved {len(vector_results)} records via KuzuDB vector search for text: '{search_text_for_vector}'")
                else:
                    kg_logger.info(f"No results from KuzuDB vector search for text: '{search_text_for_vector}'")

            except Exception as e_vector_search:
                kg_logger.error(f"Error during KuzuDB vector search: {e_vector_search}", exc_info=True)
        else:
            kg_logger.warning("Embedder not available in KGRetriever, skipping KuzuDB vector search.")

        # --- Strategy 2: Template-based Cypher (逻辑保持不变) ---
        # (这部分代码与您上一轮修改后的一致，此处省略以便聚焦向量检索部分) ...
        # 确保这里的 _score 也是越小越好，或者与向量搜索的 score 含义一致，或者在融合时能区分
        # 为了简化，我们之前给模板查询的 _score 是固定的正值，表示一个“置信度”而非距离
        # 这意味着在排序时需要特别处理，或者统一 score 的含义。
        # 暂时保持模板查询的 _score 为固定值，代表其“非向量搜索”的来源。
        if extracted_info and extracted_info.entities:
            for entity_info in extracted_info.entities:
                if not entity_info.text:
                    continue
                if entity_info.label: 
                    attr_query = """
                        MATCH (n:ExtractedEntity {text: $text, label: $label}) 
                        RETURN n.id_prop AS id_prop, n.text AS text, n.label AS label, 0.85 AS _score_template 
                        LIMIT 1 
                    """ # 将模板查询的score命名为_score_template以区分
                    attr_params = {"text": entity_info.text, "label": entity_info.label.upper()}
                    # ... (后续模板查询逻辑不变) ...
                    kg_logger.info(f"Executing template attribute query for: '{entity_info.text}' (Label: {entity_info.label.upper()})")
                    attr_results = self._execute_cypher_query_sync(attr_query, attr_params)
                    if attr_results:
                         # 将 _score_template 转换为 _score，并赋予一个表示“非距离”的值，例如负值或很大的正值
                         # 或者，我们可以在排序时单独处理。简单起见，我们先让它有一个不同的键名。
                         # 更新：为了统一排序，我们将模板结果的“分数”也视为“距离”，但赋予一个较大的固定值
                         # 以便向量搜索结果（距离小）能排在前面。
                        for res in attr_results: res['_score'] = res.pop('_score_template', 100.0) # 假设100是很大的距离
                        all_kuzu_records.extend(attr_results)

                    if extracted_info.relation_hint:
                        mapped_rel_type = None
                        relation_hint_upper = extracted_info.relation_hint.upper() # 确保比较时也用大写
                        entity_label_upper = entity_info.label.upper()

                        if "WORK" in relation_hint_upper and entity_label_upper == "PERSON":
                            mapped_rel_type = "WORKS_AT" # <--- 确保是大写
                        elif ("ASSIGN" in relation_hint_upper or "负责" in extracted_info.relation_hint) and \
                             (entity_label_upper == "TASK" or entity_label_upper == "PROJECT"): # 扩展适用范围
                            mapped_rel_type = "ASSIGNED_TO" # <--- 确保是大写

                        # --- 初始化变量 ---
                        neighbor_query = None
                        neighbor_params = None
                        # --- 结束初始化 ---

                        if mapped_rel_type:
                            neighbor_query = f"""
                            MATCH (src:ExtractedEntity {{text: $text, label: $label}})
                                  -[r:{mapped_rel_type}]-> 
                                  (tgt:ExtractedEntity)
                            RETURN tgt.id_prop AS id_prop, tgt.text AS related_text, tgt.label AS related_label, 
                                   label(r) AS relationship_type, src.text AS source_node_text, 101.0 AS _score LIMIT $limit_val""" # 原来的查询字符串
                            neighbor_params = {
                                "text": entity_info.text, 
                                "label": entity_label_upper,
                                "limit_val": top_k 
                            }
                            kg_logger.info(f"Executing template neighbor query for: '{entity_info.text}' via relation '{mapped_rel_type}' with limit_val={top_k}")
                            # --- 只有在 neighbor_query 被定义时才执行 ---
                            neighbor_results = self._execute_cypher_query_sync(neighbor_query, neighbor_params)
                            if neighbor_results:
                                all_kuzu_records.extend(neighbor_results)
                        else:
                            kg_logger.info(f"No valid relation type mapped for hint '{extracted_info.relation_hint}' and entity '{entity_info.text}'. Skipping neighbor query.")

        if not all_kuzu_records:
            kg_logger.info(f"No records retrieved from KuzuDB for query: '{user_query}' after all strategies.")
            return []

        # --- Deduplication and Formatting ---
        # 确保所有记录都有 _score，且含义一致（越小越好）
        for rec in all_kuzu_records:
            rec.setdefault("_score", 200.0) # 给没有分数的记录一个非常大的距离值

        # 排序：按 _score 升序 (距离越小越好)
        sorted_kuzu_records = sorted(all_kuzu_records, key=lambda x: x.get('_score', 200.0), reverse=False)
        
        formatted_docs = self._format_kuzu_records_for_retrieval(sorted_kuzu_records, user_query)

        final_unique_docs_map: Dict[str, Dict[str, Any]] = {}
        for doc_dict in formatted_docs:
            content_key = doc_dict.get("content", "")
            current_score = doc_dict.get("score", 200.0) # score 仍然是距离

            if content_key not in final_unique_docs_map:
                final_unique_docs_map[content_key] = doc_dict
            else:
                # 如果内容已存在，保留距离更小的（分数更优的）
                existing_score = final_unique_docs_map[content_key].get("score", 200.0)
                if current_score < existing_score:
                    final_unique_docs_map[content_key] = doc_dict
        
        final_results_list = list(final_unique_docs_map.values())
        # 再次根据score（距离）排序，因为字典转列表后顺序可能打乱
        final_results_sorted_by_score = sorted(final_results_list, key=lambda x: x.get('score', 200.0), reverse=False)

        kg_logger.info(f"Total {len(all_kuzu_records)} records initially retrieved, "
                       f"{len(sorted_kuzu_records)} after initial sort by distance, "
                       f"{len(formatted_docs)} after formatting, "
                       f"{len(final_results_sorted_by_score)} after final content deduplication & re-sort. "
                       f"Returning top {min(top_k, len(final_results_sorted_by_score))} docs.")
        
        return final_results_sorted_by_score[:top_k]