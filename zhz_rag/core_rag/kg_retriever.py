# 文件: zhz_rag/core_rag/kg_retriever.py
import os
import json
import duckdb
from typing import List, Dict, Any, Optional, Iterator, TYPE_CHECKING
import logging
from contextlib import contextmanager
import asyncio
from cachetools import TTLCache # <--- 添加这一行
if TYPE_CHECKING:
    from zhz_rag.llm.local_model_handler import LocalModelHandler

from zhz_rag.llm.llm_interface import extract_entities_for_kg_query
from zhz_rag.config.pydantic_models import ExtractedEntitiesAndRelationIntent
from zhz_rag.utils.common_utils import normalize_text_for_id
from zhz_rag.utils.interaction_logger import log_interaction_data # <--- 确保这行存在
import uuid # <--- 添加导入
from datetime import datetime, timezone # <--- 添加导入

# 日志配置
kg_logger = logging.getLogger(__name__)
if not kg_logger.hasHandlers():
    kg_logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    ch.setFormatter(formatter)
    kg_logger.addHandler(ch)
    kg_logger.propagate = False
kg_logger.info("KGRetriever (DuckDB) logger initialized/reconfirmed.")


class KGRetriever:
    # 使用 DuckDB 的环境变量或默认路径
    DUCKDB_KG_FILE_PATH_ENV = os.getenv(
        "DUCKDB_KG_FILE_PATH",
        os.path.join(os.getenv("ZHZ_AGENT_PROJECT_ROOT", "/home/zhz/zhz_agent"), "zhz_rag", "stored_data", "duckdb_knowledge_graph.db")
    )

    def __init__(self, db_file_path: Optional[str] = None, embedder: Optional['LocalModelHandler'] = None):
        self.db_file_path = db_file_path if db_file_path else self.DUCKDB_KG_FILE_PATH_ENV
        self._embedder = embedder
        self._retrieval_cache: TTLCache = TTLCache(maxsize=100, ttl=300) # <--- 修改这一行
        self._retrieval_cache_lock = asyncio.Lock()
        kg_logger.info(f"KGRetriever (DuckDB) initialized. DB file path set to: {self.db_file_path}")

        if not os.path.exists(self.db_file_path):
            kg_logger.warning(f"DuckDB database file not found at {self.db_file_path}. Retrieval operations will likely fail if the DB is not created by the Dagster pipeline first.")
        
        # 健康检查
        try:
            with self._get_duckdb_connection() as conn_test:
                result = conn_test.execute("SELECT 42;").fetchone()
                if result and result[0] == 42:
                    kg_logger.info("DuckDB connection test successful and VSS setup attempted in _get_duckdb_connection.")
                else:
                    kg_logger.warning("DuckDB connection test failed to return expected result.")
        except Exception as e_init_conn_test:
            kg_logger.error(f"Error during initial DuckDB connection test: {e_init_conn_test}", exc_info=True)
    
    @contextmanager
    def _get_duckdb_connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """
        建立并返回一个DuckDB连接的上下文管理器。
        """
        conn: Optional[duckdb.DuckDBPyConnection] = None
        kg_logger.debug(f"_get_duckdb_connection: Attempting to connect to DB at {self.db_file_path}")
        try:
            if not os.path.exists(self.db_file_path):
                raise FileNotFoundError(f"DuckDB file '{self.db_file_path}' does not exist when trying to open connection.")

            conn = duckdb.connect(database=self.db_file_path, read_only=False)
            kg_logger.debug(f"DuckDB Connection object created for path: {self.db_file_path} (read_only=False)")
            
            try:
                conn.execute("LOAD vss;")
                kg_logger.debug("DuckDB: VSS extension loaded on connection.")
            except Exception as e_vss_setup:
                kg_logger.warning(f"DuckDB: Failed to setup VSS extension on connect: {e_vss_setup}. This might be okay if already set or not needed for this operation.")
            yield conn
        except Exception as e_outer:
            kg_logger.error(f"Error in _get_duckdb_connection: {e_outer}", exc_info=True)
            raise
        finally:
            kg_logger.debug("_get_duckdb_connection: Exiting context.")
            if conn:
                kg_logger.debug("Closing DuckDB connection.")
                conn.close()

    def _execute_duckdb_sql_query_sync(self, query: str, parameters: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """
        执行DuckDB SQL查询并返回结果列表。
        （此版本不再包含通用日志记录逻辑）
        """
        kg_logger.info(f"--- Executing DuckDB SQL --- Query: {query.strip()}")
        if parameters:
            log_params = [
                str(p)[:100] + '...' if isinstance(p, list) and len(str(p)) > 100 else p
                for p in parameters
            ]
            kg_logger.info(f"Params: {log_params}")

        results_list: List[Dict[str, Any]] = []
        try:
            with self._get_duckdb_connection() as conn:
                prepared_statement = conn.execute(query, parameters)
                
                if prepared_statement.description:
                    column_names = [desc[0] for desc in prepared_statement.description]
                    raw_results = prepared_statement.fetchall()
                    
                    for row_tuple in raw_results:
                        results_list.append(dict(zip(column_names, row_tuple)))
                        
                kg_logger.info(f"DuckDB SQL executed. Records count: {len(results_list)}")
                if results_list: 
                    kg_logger.debug(f"First DuckDB record: {str(results_list[0])[:200]}")
                else:
                     kg_logger.debug("DuckDB SQL query returned 0 records.")
        except duckdb.Error as duckdb_err:
             kg_logger.error(f"DuckDB Error during SQL execution: '{query}'. Error: {duckdb_err}", exc_info=True)
        except Exception as e:
            kg_logger.error(f"Unexpected error executing DuckDB SQL query: '{query}'. Error: {e}", exc_info=True)
        return results_list
    
    def _format_duckdb_records_for_retrieval(
        self, 
        records: List[Dict[str, Any]], 
        query_context: str = "",
        source_type_prefix: str = "duckdb_kg"
    ) -> List[Dict[str, Any]]:
        formatted_docs = []
        if not records:
            return formatted_docs

        for record_data in records:
            content_parts = []
            entity_text = record_data.get("text") or record_data.get("target_text") or record_data.get("source_text")
            entity_label = record_data.get("label") or record_data.get("target_label") or record_data.get("source_label")
            relation_type = record_data.get("relation_type")
            
            if "source_text" in record_data and "target_text" in record_data and relation_type:
                content_parts = [
                    f"Source: {record_data['source_text']} ({record_data.get('source_label', 'Entity')})",
                    f"Relation: {relation_type}",
                    f"Target: {record_data['target_text']} ({record_data.get('target_label', 'Entity')})"
                ]
            elif entity_text:
                content_parts.append(f"Entity: {entity_text}")
                if entity_label:
                    content_parts.append(f"Label: {entity_label}")
            else:
                content_parts.append(f"Retrieved KG data: {json.dumps({k:v for k,v in record_data.items() if k != 'embedding'}, ensure_ascii=False, default=str)[:100]}")

            content_str = " | ".join(content_parts)
            
            doc_metadata = {
                "original_user_query_for_kg": query_context,
                "duckdb_retrieved_id_prop": record_data.get("id_prop") or record_data.get("source_id_prop"),
                "duckdb_retrieved_data": {k:v for k,v in record_data.items() if k != 'embedding'}
            }
            if record_data.get("_source_strategy"):
                doc_metadata["_source_strategy"] = record_data.get("_source_strategy")

            score_value = record_data.get("distance")
            if score_value is None:
                score_value = record_data.get("_score")

            if record_data.get("distance") is not None:
                similarity_score = 1.0 / (1.0 + float(score_value)) if score_value is not None else 0.5 
            elif isinstance(score_value, (int, float)):
                similarity_score = float(score_value)
            else:
                similarity_score = 0.5

            doc_data = {
                "source_type": source_type_prefix,
                "content": content_str,
                "score": similarity_score, 
                "metadata": {k: v for k, v in doc_metadata.items() if v is not None}
            }
            formatted_docs.append(doc_data)
        return formatted_docs

    async def retrieve(self, user_query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        kg_logger.info(f"Starting DuckDB KG retrieval for query: '{user_query}', top_k: {top_k}")

        # --- 更新: 使用 TTLCache 和异步锁进行缓存检查 ---
        cache_key = f"{user_query}_{top_k}"
        async with self._retrieval_cache_lock:
            cached_result = self._retrieval_cache.get(cache_key)

        if cached_result is not None:
            kg_logger.info(f"KG CACHE HIT for key: '{cache_key[:100]}...'")
            return cached_result
        
        kg_logger.info(f"KG CACHE MISS for key: '{cache_key[:100]}...'. Performing retrieval.")
        # --- 缓存检查结束 ---

        if not self._embedder:
            kg_logger.error("Embedder not configured for KGRetriever. Vector search will be skipped.")
        
        all_retrieved_records: List[Dict[str, Any]] = []
        processed_entity_ids = set()

        # 1. LLM提取实体和关系意图
        extracted_info: Optional[ExtractedEntitiesAndRelationIntent] = None
        try:
            extracted_info = await extract_entities_for_kg_query(user_query)
            if extracted_info:
                entities_log = [e.model_dump() for e in extracted_info.entities]
                relations_log = [r.model_dump() for r in extracted_info.relations]
                kg_logger.info(f"LLM extracted for KG: Entities: {entities_log}, Relations: {relations_log}")
            else:
                kg_logger.info("LLM did not extract specific entities/relations for KG query.")
        except Exception as e_extract:
            kg_logger.error(f"Error during entity/relation extraction for KG: {e_extract}", exc_info=True)

        # 2. 向量搜索 (可选, 作为补充)
        if self._embedder:
            try:
                kg_logger.info(f"Generating embedding for vector search text: '{user_query}'")
                query_vector_list = await self._embedder.embed_query(user_query)
                if query_vector_list:
                    vector_search_sql = "SELECT id_prop, text, label, list_distance(embedding, ?) AS distance FROM ExtractedEntity ORDER BY distance ASC LIMIT ?;"
                    vector_results = self._execute_duckdb_sql_query_sync(vector_search_sql, [query_vector_list, top_k])
                    if vector_results:
                        for rec in vector_results: rec["_source_strategy"] = "vector_search"
                        all_retrieved_records.extend(vector_results)
                        processed_entity_ids.update(rec.get("id_prop") for rec in vector_results)
                        kg_logger.info(f"Retrieved {len(vector_results)} records via DuckDB vector search.")
                else:
                    kg_logger.warning("Failed to generate query embedding for vector search.")
            except Exception as e_vec_search:
                kg_logger.error(f"Error during DuckDB vector search: {e_vec_search}", exc_info=True)
        
        # 3. 基于LLM提取的实体进行精确查找
        if extracted_info and extracted_info.entities:
            for entity_info in extracted_info.entities:
                entity_text_norm = normalize_text_for_id(entity_info.text)
                entity_label_norm = entity_info.label.upper()
                exact_entity_sql = "SELECT id_prop, text, label FROM ExtractedEntity WHERE text = ? AND label = ? LIMIT 1;"
                entity_lookup_results = self._execute_duckdb_sql_query_sync(exact_entity_sql, [entity_text_norm, entity_label_norm])
                for rec in entity_lookup_results:
                    if rec.get("id_prop") not in processed_entity_ids:
                        rec["_source_strategy"] = "exact_entity_match"
                        all_retrieved_records.append(rec)
                        processed_entity_ids.add(rec.get("id_prop"))

        # 4. 基于LLM提取的结构化关系进行验证和邻居查找
        if extracted_info and extracted_info.relations:
            kg_logger.info(f"Found {len(extracted_info.relations)} structured relations to process.")
            for rel_item in extracted_info.relations:
                try:
                    head_text_norm = normalize_text_for_id(rel_item.head_entity_text)
                    head_label_norm = rel_item.head_entity_label.upper()
                    tail_text_norm = normalize_text_for_id(rel_item.tail_entity_text)
                    tail_label_norm = rel_item.tail_entity_label.upper()
                    relation_type_norm = rel_item.relation_type.upper()

                    kg_logger.info(f"Processing relation: ({head_text_norm}:{head_label_norm})-[{relation_type_norm}]->({tail_text_norm}:{tail_label_norm})")

                    # 步骤 4a: 验证关系本身是否存在
                    relation_verification_sql = """
                    SELECT r.relation_type, s.id_prop AS source_id_prop, s.text AS source_text, s.label AS source_label, t.id_prop AS target_id_prop, t.text AS target_text, t.label AS target_label
                    FROM KGExtractionRelation r
                    JOIN ExtractedEntity s ON r.source_node_id_prop = s.id_prop
                    JOIN ExtractedEntity t ON r.target_node_id_prop = t.id_prop
                    WHERE s.text = ? AND s.label = ? AND t.text = ? AND t.label = ? AND r.relation_type = ? LIMIT 1;
                    """
                    relation_verification_params = [head_text_norm, head_label_norm, tail_text_norm, tail_label_norm, relation_type_norm]
                    
                    log_entry = {
                        "interaction_id": str(uuid.uuid4()), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "task_type": "kg_executed_query_for_eval", "user_query_for_task": user_query,
                        "generated_query_language": "SQL_DuckDB", "generated_query": relation_verification_sql.strip(),
                        "query_parameters": [str(p) for p in relation_verification_params],
                        "application_version": "kg_retriever_0.2_rel_verify"
                    }
                    try:
                        asyncio.create_task(log_interaction_data(log_entry))
                    except Exception as e_log:
                        kg_logger.error(f"Error queuing precise log for relation verification query: {e_log}", exc_info=True)

                    relation_verification_results = self._execute_duckdb_sql_query_sync(relation_verification_sql, relation_verification_params)
                    
                    if relation_verification_results:
                        kg_logger.info(f"    Successfully verified relation in KG: {relation_type_norm} between '{head_text_norm}' and '{tail_text_norm}'")
                        
                        for verified_rel_record in relation_verification_results:
                            head_entity_from_rel = {"id_prop": verified_rel_record.get("source_id_prop"), "text": verified_rel_record.get("source_text"), "label": verified_rel_record.get("source_label"), "_source_strategy": f"verified_relation_head_{relation_type_norm}"}
                            if head_entity_from_rel.get("id_prop") not in processed_entity_ids:
                                all_retrieved_records.append(head_entity_from_rel)
                                processed_entity_ids.add(head_entity_from_rel.get("id_prop"))
                                kg_logger.info(f"      Added head entity '{head_entity_from_rel.get('text')}' from verified relation to results.")
                            tail_entity_from_rel = {"id_prop": verified_rel_record.get("target_id_prop"), "text": verified_rel_record.get("target_text"), "label": verified_rel_record.get("target_label"), "_source_strategy": f"verified_relation_tail_{relation_type_norm}"}
                            if tail_entity_from_rel.get("id_prop") not in processed_entity_ids:
                                all_retrieved_records.append(tail_entity_from_rel)
                                processed_entity_ids.add(tail_entity_from_rel.get("id_prop"))
                                kg_logger.info(f"      Added tail entity '{tail_entity_from_rel.get('text')}' from verified relation to results.")
                        
                        if head_text_norm and head_label_norm and relation_type_norm:
                            find_other_tails_sql = """
                            SELECT t.id_prop, t.text, t.label, r.relation_type
                            FROM ExtractedEntity h
                            JOIN KGExtractionRelation r ON h.id_prop = r.source_node_id_prop
                            JOIN ExtractedEntity t ON r.target_node_id_prop = t.id_prop
                            WHERE h.text = ? AND h.label = ? AND r.relation_type = ? AND t.text != ?
                            LIMIT ?;
                            """
                            find_other_tails_params = [head_text_norm, head_label_norm, relation_type_norm, tail_text_norm, top_k]
                            other_tails_results = self._execute_duckdb_sql_query_sync(find_other_tails_sql, find_other_tails_params)
                            for rec in other_tails_results:
                                if rec.get("id_prop") not in processed_entity_ids:
                                    rec["_source_strategy"] = f"neighbor_tail_for_{relation_type_norm}"
                                    all_retrieved_records.append(rec)
                                    processed_entity_ids.add(rec.get("id_prop"))
                                    kg_logger.info(f"        Added neighbor tail entity '{rec.get('text')}' to results.")

                        if tail_text_norm and tail_label_norm and relation_type_norm:
                            find_other_heads_sql = """
                            SELECT h.id_prop, h.text, h.label, r.relation_type
                            FROM ExtractedEntity t
                            JOIN KGExtractionRelation r ON t.id_prop = r.target_node_id_prop
                            JOIN ExtractedEntity h ON r.source_node_id_prop = h.id_prop
                            WHERE t.text = ? AND t.label = ? AND r.relation_type = ? AND h.text != ?
                            LIMIT ?;
                            """
                            find_other_heads_params = [tail_text_norm, tail_label_norm, relation_type_norm, head_text_norm, top_k]
                            other_heads_results = self._execute_duckdb_sql_query_sync(find_other_heads_sql, find_other_heads_params)
                            for rec in other_heads_results:
                                if rec.get("id_prop") not in processed_entity_ids:
                                    rec["_source_strategy"] = f"neighbor_head_for_{relation_type_norm}"
                                    all_retrieved_records.append(rec)
                                    processed_entity_ids.add(rec.get("id_prop"))
                                    kg_logger.info(f"        Added neighbor head entity '{rec.get('text')}' to results.")
                    else:
                        kg_logger.info(f"    Relation {relation_type_norm} between '{head_text_norm}' and '{tail_text_norm}' not found in KG via exact match.")

                except Exception as e_rel_proc:
                    kg_logger.error(f"Error processing structured relation item {rel_item.model_dump_json()}: {e_rel_proc}", exc_info=True)

        if not all_retrieved_records:
            kg_logger.info(f"No records retrieved from DuckDB KG for query: '{user_query}' after all strategies.")
            return []

        unique_records = []
        seen_ids = set()
        for record in all_retrieved_records:
            record_id = record.get("id_prop") or record.get("source_id_prop")
            if record_id and record_id in seen_ids:
                continue
            unique_records.append(record)
            if record_id:
                seen_ids.add(record_id)
        
        formatted_docs = self._format_duckdb_records_for_retrieval(unique_records, user_query, "duckdb_kg")
        
        # --- 更新: 存储到 TTLCache ---
        async with self._retrieval_cache_lock:
            self._retrieval_cache[cache_key] = formatted_docs
        kg_logger.info(f"KG CACHED {len(formatted_docs)} results for key: '{cache_key[:100]}...'")
        # --- 缓存存储结束 ---

        kg_logger.info(f"KGRetriever (DuckDB) retrieve method finished. Returning {len(formatted_docs)} formatted documents for fusion.")
        return formatted_docs

    def close(self):
        kg_logger.info(f"KGRetriever (DuckDB).close() called. (No persistent DB object to close in this version as connections are per-method).")
        pass
