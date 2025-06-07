# zhz_rag/core_rag/kg_retriever.py
import os
import json
import kuzu
import pandas as pd
from typing import List, Dict, Any, Optional, Callable, Iterator
import logging
from contextlib import contextmanager

# --- 修改：导入新的实体提取函数和Pydantic模型 ---
from zhz_rag.llm.llm_interface import extract_entities_for_kg_query
from zhz_rag.config.pydantic_models import ExtractedEntitiesAndRelationIntent, IdentifiedEntity
# from zhz_rag.config.constants import NEW_KG_SCHEMA_DESCRIPTION # Schema现在主要在prompt中使用
# --- 结束修改 ---

# 日志配置
kg_logger = logging.getLogger(__name__)
if not kg_logger.hasHandlers():
    kg_logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    ch.setFormatter(formatter)
    kg_logger.addHandler(ch)
    kg_logger.propagate = False
kg_logger.info("KGRetriever (KuzuDB) logger initialized/reconfirmed.")


class KGRetriever:
    KUZU_DB_PATH_ENV = os.getenv("KUZU_DB_PATH", "/home/zhz/zhz_agent/zhz_rag/stored_data/kuzu_default_db")

    # 移除 llm_cypher_generator_func 参数，因为我们将使用新的实体提取流程
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path if db_path else self.KUZU_DB_PATH_ENV
        self._db: Optional[kuzu.Database] = None
        kg_logger.info(f"KGRetriever (KuzuDB) __init__ called. DB path: {self.db_path}")
        self._connect_to_kuzu()

    def _connect_to_kuzu(self):
        kg_logger.info(f"Attempting to load KuzuDB from path: {self.db_path}")
        try:
            if not os.path.exists(self.db_path):
                kg_logger.error(f"KuzuDB path does not exist: {self.db_path}. KGRetriever cannot connect.")
                self._db = None
                return
            # 考虑是否需要在配置文件中指定 read_only 模式
            self._db = kuzu.Database(self.db_path, read_only=True)
            kg_logger.info(f"Successfully loaded KuzuDB from {self.db_path}.")
        except Exception as e:
            kg_logger.error(f"Failed to connect to KuzuDB at {self.db_path}: {e}", exc_info=True)
            self._db = None

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
            pass

    def close(self):
        kg_logger.info(f"Closing KuzuDB for retriever using path: {self.db_path}")
        if self._db:
            del self._db
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
                actual_params = parameters if parameters else {}
            # 直接将查询字符串和参数字典传递给 execute
                query_result = conn.execute(query, parameters=actual_params)
                # KuzuDB QueryResult to_df() is deprecated, use get_as_df()
                if hasattr(query_result, 'get_as_df'):
                    df = pd.DataFrame(query_result.get_as_df())
                    results_list = df.to_dict(orient='records')
                elif isinstance(query_result, list): # 有些简单的执行可能直接返回列表
                    # 如果是列表，我们需要确定其结构是否是我们期望的 List[Dict]
                    # 为简单起见，如果不是DataFrame兼容的，我们先认为没有结构化结果返回用于进一步处理
                    kg_logger.info("KuzuDB query did not return a DataFrame-convertible result, result is a list.")
                    # 根据实际情况处理列表结果，或将其置空
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
        except ConnectionError as conn_err: # 如果 _get_connection 内部抛出
             kg_logger.error(f"KuzuDB ConnectionError during Cypher execution: {conn_err}", exc_info=True)
        except TypeError as e_type: # 捕获可能的TypeError，以便更详细地调试
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
            primary_info_keys = ['text', 'label', 'id_prop', 'name', 'related_text', 'related_label', 'relationship']
            for key in primary_info_keys:
                if key in record_data and record_data[key] is not None:
                    parts.append(f"{key.replace('_', ' ').capitalize()}: {record_data[key]}")
            
            for key, value in record_data.items():
                if key not in primary_info_keys and value is not None:
                    if isinstance(value, dict):
                        if '_label' in value and '_src' in value and '_dst' in value: 
                             parts.append(f"Relation Type: {value['_label']}")
                        else:
                             value_str = json.dumps(value, ensure_ascii=False, default=str)
                             if len(value_str) > 70: value_str = value_str[:70] + "..."
                             parts.append(f"{key}: {value_str}")
                    else:
                        parts.append(f"{key}: {str(value)}")
            
            content_str = " | ".join(parts) if parts else "Retrieved graph data node/relation."
            
            doc_data = {
                "source_type": "knowledge_graph_kuzu",
                "content": content_str,
                "score": record_data.get('_score', 0.85), # Default score if not from vector search
                "metadata": {
                    "original_user_query_for_kg": query_context,
                    "retrieved_kuzu_record_preview": {k: record_data[k] for k in primary_info_keys if k in record_data}
                }
            }
            formatted_docs.append(doc_data)
        return formatted_docs

    async def retrieve(self, user_query: str, top_k: int = 3) -> List[Dict[str, Any]]: # Renamed main retrieval method
        kg_logger.info(f"Starting KG retrieval with LLM-extracted entities for query: '{user_query}', top_k: {top_k}")

        extracted_info: Optional[ExtractedEntitiesAndRelationIntent] = await extract_entities_for_kg_query(user_query)

        if not extracted_info or not extracted_info.entities:
            kg_logger.warning(f"LLM did not extract any entities for query: '{user_query}'. KG retrieval cannot proceed.")
            return []
        
        kg_logger.info(f"LLM extracted: Entities: {[e.model_dump() for e in extracted_info.entities]}, Relation Hint: {extracted_info.relation_hint}")

        all_kuzu_records: List[Dict[str, Any]] = []

        for entity_info in extracted_info.entities:
            if not entity_info.text:
                continue

            # --- Strategy 1: KuzuDB Vector Index Search (Preferred) ---
            # TODO: Implement KuzuDB vector search logic here.
            # This requires:
            # 1. An embedding function (e.g., from SentenceTransformer) to convert entity_info.text to a vector.
            # 2. Knowing the name of your vector index in KuzuDB.
            # 3. Executing a KuzuDB CALL db.idx.lookup(...) query.
            # Example (conceptual):
            # query_vector = self.embedding_function(entity_info.text)
            # vector_search_query = "CALL db.idx.lookup('your_node_vector_index', $query_vec, $k) YIELD node, score RETURN node.text, node.label, node.id_prop, score"
            # params = {"query_vec": query_vector, "k": top_k}
            # vector_results = self._execute_cypher_query_sync(vector_search_query, params)
            # if vector_results:
            #     all_kuzu_records.extend(vector_results)
            #     kg_logger.info(f"Retrieved {len(vector_results)} records via vector search for entity: '{entity_info.text}'")
            
            # --- Strategy 2: Template-based Cypher (if entity label is known, or as fallback) ---
            # This part uses simple Cypher based on extracted entity text and label.
            if entity_info.label: # Only proceed if LLM provided a label
                # Template 1: Get attributes of the identified entity
                attr_query = "MATCH (n:ExtractedEntity {text: $text, label: $label}) RETURN n.text AS text, n.label AS label, n.id_prop AS id_prop, 1.0 AS _score LIMIT 1"
                attr_params = {"text": entity_info.text, "label": entity_info.label.upper()}
                kg_logger.info(f"Executing attribute query for: {entity_info.text} ({entity_info.label})")
                attr_results = self._execute_cypher_query_sync(attr_query, attr_params)
                if attr_results:
                    all_kuzu_records.extend(attr_results)

                # Template 2: If a relation_hint exists, try to find 1-hop neighbors
                if extracted_info.relation_hint:
                    # This mapping from relation_hint to actual Cypher relation type needs to be robust
                    # For MVP, we can try a generic neighbor search or map common hints
                    # Example: if relation_hint is "工作" and entity_info.label is "PERSON", map to "WORKS_AT"
                    mapped_rel_type = None
                    if "工作" in extracted_info.relation_hint and entity_info.label == "PERSON":
                        mapped_rel_type = "WorksAt"
                    elif "分配" in extracted_info.relation_hint and entity_info.label == "TASK":
                        mapped_rel_type = "AssignedTo"
                    
                    if mapped_rel_type:
                        neighbor_query = f"""
                            MATCH (src:ExtractedEntity {{text: $text, label: $label}})
                                  -[r:{mapped_rel_type}]->
                                  (tgt:ExtractedEntity)
                            RETURN tgt.text AS related_text, tgt.label AS related_label, label(r) as relationship, 0.9 AS _score 
                            LIMIT {top_k}
                        """
                        neighbor_params = {"text": entity_info.text, "label": entity_info.label.upper()}
                        kg_logger.info(f"Executing neighbor query for: {entity_info.text} via relation {mapped_rel_type}")
                        neighbor_results = self._execute_cypher_query_sync(neighbor_query, neighbor_params)
                        if neighbor_results:
                            all_kuzu_records.extend(neighbor_results)
            else:
                kg_logger.info(f"Skipping template-based Cypher for entity '{entity_info.text}' as label was not provided by LLM.")


        if not all_kuzu_records:
            kg_logger.info(f"No records retrieved from KuzuDB for query: '{user_query}' after all strategies.")
            return []

        # Deduplicate records (e.g., if vector search and template search return same nodes)
        # A simple way is to convert to DataFrame and drop duplicates based on id_prop or text+label
        if all_kuzu_records:
            try:
                df_records = pd.DataFrame(all_kuzu_records)
                # Assuming 'id_prop' is a unique identifier for nodes, or use a combination of text and label
                # We need to handle cases where 'id_prop' might not be present in all results (e.g., from neighbor queries)
                # For simplicity, let's try to deduplicate based on the 'content' we will generate
                
                # First, generate content for all, then deduplicate based on content
                temp_formatted_for_dedup = self._format_kuzu_records_for_retrieval(all_kuzu_records, user_query)
                unique_contents = {}
                deduplicated_records_formatted = []
                for doc_dict in temp_formatted_for_dedup:
                    if doc_dict["content"] not in unique_contents:
                        unique_contents[doc_dict["content"]] = True
                        deduplicated_records_formatted.append(doc_dict)
                
                kg_logger.info(f"Deduplicated KuzuDB results from {len(all_kuzu_records)} to {len(deduplicated_records_formatted)} records.")
                return deduplicated_records_formatted[:top_k] # Apply top_k after deduplication
            except Exception as e_dedup:
                kg_logger.error(f"Error during deduplication of KuzuDB results: {e_dedup}", exc_info=True)
                # Fallback to returning raw (potentially duplicated) records if deduplication fails
                return self._format_kuzu_records_for_retrieval(all_kuzu_records, user_query)[:top_k]


        return self._format_kuzu_records_for_retrieval(all_kuzu_records, user_query)[:top_k]