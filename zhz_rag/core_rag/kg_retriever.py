# 文件: zhz_rag/core_rag/kg_retriever.py
import os
import json
import duckdb
from typing import List, Dict, Any, Optional, Iterator, TYPE_CHECKING
import logging
from contextlib import contextmanager

if TYPE_CHECKING:
    from zhz_rag.llm.local_model_handler import LocalModelHandler

from zhz_rag.llm.llm_interface import extract_entities_for_kg_query
from zhz_rag.config.pydantic_models import ExtractedEntitiesAndRelationIntent
from zhz_rag.utils.common_utils import normalize_text_for_id

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

            # 连接时允许写入，以解决潜在的WAL文件重放问题
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
        """
        kg_logger.info(f"--- Executing DuckDB SQL --- Query: {query.strip()}")
        if parameters:
            # 对列表类型的参数进行摘要打印，避免日志过长
            log_params = [
                str(p)[:100] + '...' if isinstance(p, list) and len(str(p)) > 100 else p
                for p in parameters
            ]
            kg_logger.info(f"Params: {log_params}")

        results_list: List[Dict[str, Any]] = []
        try:
            with self._get_duckdb_connection() as conn:
                prepared_statement = conn.execute(query, parameters)
                
                column_names = [desc[0] for desc in prepared_statement.description]
                
                raw_results = prepared_statement.fetchall()
                
                for row_tuple in raw_results:
                    results_list.append(dict(zip(column_names, row_tuple)))
                    
                kg_logger.info(f"DuckDB SQL executed. Records count: {len(results_list)}")
                if results_list: 
                    kg_logger.debug(f"First DuckDB record: {str(results_list[0])[:200]}")
                elif raw_results is not None:
                     kg_logger.debug("DuckDB SQL query returned 0 records.")
                else:
                     kg_logger.debug("DuckDB SQL query result was None or unexpected after fetchall.")

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
                # 格式化关系记录
                content_parts = [
                    f"Source: {record_data['source_text']} ({record_data.get('source_label', 'Entity')})",
                    f"Relation: {relation_type}",
                    f"Target: {record_data['target_text']} ({record_data.get('target_label', 'Entity')})"
                ]
            elif entity_text:
                # 格式化单个实体记录
                content_parts.append(f"Entity: {entity_text}")
                if entity_label:
                    content_parts.append(f"Label: {entity_label}")
            else:
                # 通用回退格式
                content_parts.append(f"Retrieved KG data: {json.dumps({k:v for k,v in record_data.items() if k != 'embedding'}, ensure_ascii=False, default=str)[:100]}")

            content_str = " | ".join(content_parts)
            
            doc_metadata = {
                "original_user_query_for_kg": query_context,
                "duckdb_retrieved_id_prop": record_data.get("id_prop") or record_data.get("source_id_prop"),
                "duckdb_retrieved_data": {k:v for k,v in record_data.items() if k != 'embedding'}
            }
            # 添加来源策略到元数据中
            if record_data.get("_source_strategy"):
                doc_metadata["_source_strategy"] = record_data.get("_source_strategy")

            score_value = record_data.get("distance")
            if score_value is None:
                score_value = record_data.get("_score")

            if record_data.get("distance") is not None:
                # 将距离转换为相似度分数 (0-1)，距离越小分数越高
                similarity_score = 1.0 / (1.0 + float(score_value)) if score_value is not None else 0.5 
            elif isinstance(score_value, (int, float)):
                similarity_score = float(score_value)
            else:
                # 对于非评分的查找，给一个默认分数
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
                    vector_search_sql = """
                    SELECT id_prop, text, label, list_distance(embedding, ?) AS distance
                    FROM ExtractedEntity
                    ORDER BY distance ASC
                    LIMIT ?;
                    """
                    kg_logger.info(f"Executing DuckDB vector search. Top K: {top_k}")
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
                kg_logger.info(f"Executing exact entity lookup for: text='{entity_text_norm}', label='{entity_label_norm}'")
                entity_lookup_results = self._execute_duckdb_sql_query_sync(exact_entity_sql, [entity_text_norm, entity_label_norm])
                
                for rec in entity_lookup_results:
                    if rec.get("id_prop") not in processed_entity_ids:
                        rec["_source_strategy"] = "exact_entity_match"
                        all_retrieved_records.append(rec)
                        processed_entity_ids.add(rec.get("id_prop"))

        # 4. 【核心修改】基于LLM提取的结构化关系进行精确查询
        if extracted_info and extracted_info.relations:
            kg_logger.info(f"Found {len(extracted_info.relations)} structured relations to process.")
            for rel_item in extracted_info.relations:
                try:
                    head_text = normalize_text_for_id(rel_item.head_entity_text)
                    head_label = rel_item.head_entity_label.upper()
                    tail_text = normalize_text_for_id(rel_item.tail_entity_text)
                    tail_label = rel_item.tail_entity_label.upper()
                    rel_type = rel_item.relation_type.upper()

                    kg_logger.info(f"Processing relation: ({head_text}:{head_label})-[{rel_type}]->({tail_text}:{tail_label})")

                    relation_query_sql = """
                    SELECT 
                        r.relation_type,
                        s.id_prop AS source_id_prop, s.text AS source_text, s.label AS source_label,
                        t.id_prop AS target_id_prop, t.text AS target_text, t.label AS target_label
                    FROM KGExtractionRelation r
                    JOIN ExtractedEntity s ON r.source_node_id_prop = s.id_prop
                    JOIN ExtractedEntity t ON r.target_node_id_prop = t.id_prop
                    WHERE s.text = ? AND s.label = ?
                      AND t.text = ? AND t.label = ?
                      AND r.relation_type = ?
                    LIMIT 1;
                    """
                    params = [head_text, head_label, tail_text, tail_label, rel_type]
                    
                    relation_results = self._execute_duckdb_sql_query_sync(relation_query_sql, params)
                    
                    if relation_results:
                        kg_logger.info(f"Successfully verified relation in KG: {rel_type}")
                        for rec in relation_results:
                            rec["_source_strategy"] = f"structured_relation_match_{rel_type}"
                            # 关系查询返回的结果更相关，即使实体已经存在，也值得加入
                            all_retrieved_records.append(rec)
                    else:
                        kg_logger.info(f"Relation not found in KG via exact match. Consider broadening search.")

                except Exception as e_rel_proc:
                    kg_logger.error(f"Error processing structured relation item {rel_item.model_dump_json()}: {e_rel_proc}", exc_info=True)

        if not all_retrieved_records:
            kg_logger.info(f"No records retrieved from DuckDB KG for query: '{user_query}' after all strategies.")
            return []

        # 去重，因为多种策略可能找到相同的结果
        unique_records = []
        seen_ids = set()
        for record in all_retrieved_records:
            record_id = record.get("id_prop") or record.get("source_id_prop")
            if record_id and record_id in seen_ids:
                continue
            unique_records.append(record)
            if record_id:
                seen_ids.add(record_id)
        
        # 格式化最终输出
        formatted_docs = self._format_duckdb_records_for_retrieval(unique_records, user_query, "duckdb_kg")
        
        kg_logger.info(f"KGRetriever (DuckDB) retrieve method finished. Returning {len(formatted_docs)} formatted documents for fusion.")
        return formatted_docs

    def close(self):
        kg_logger.info(f"KGRetriever (DuckDB).close() called. (No persistent DB object to close in this version as connections are per-method).")
        pass
