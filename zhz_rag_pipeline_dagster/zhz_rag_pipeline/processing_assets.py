# 文件: zhz_rag_pipeline_dagster/zhz_rag_pipeline/processing_assets.py

import dagster as dg
from typing import List, Dict, Any, Optional
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
import pandas as pd
from zhz_rag.utils.common_utils import normalize_text_for_id
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.pydantic_models_dagster import (
    ChunkOutput,
    ParsedDocumentOutput,
    EmbeddingOutput,
    KGTripleSetOutput,
    ExtractedEntity,
    ExtractedRelation
)
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.resources import (
    GGUFEmbeddingResource,
    ChromaDBResource,
    DuckDBResource,
    LocalLLMAPIResource,
)
import jieba
import bm25s
import pickle
import numpy as np
import os
import time


class TextChunkerConfig(dg.Config):
    chunk_size: int = 500
    chunk_overlap: int = 50

@dg.asset(
    name="text_chunks",
    description="Cleans and chunks parsed documents into smaller text segments.",
    group_name="processing",
    deps=["parsed_documents"]
)
def clean_chunk_text_asset(
    context: dg.AssetExecutionContext,
    config: TextChunkerConfig,
    parsed_documents: List[ParsedDocumentOutput]
) -> List[ChunkOutput]:
    all_chunks: List[ChunkOutput] = []
    context.log.info(f"Received {len(parsed_documents)} parsed documents to clean and chunk.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    for parsed_doc in parsed_documents:
        doc_id_from_meta = parsed_doc.original_metadata.get("filename", str(uuid.uuid4()))
        context.log.info(f"Processing document: {doc_id_from_meta}")
        cleaned_text = parsed_doc.parsed_text.strip()
        if not cleaned_text:
            context.log.warning(f"Document {doc_id_from_meta} has no valid content, skipping.")
            continue
        try:
            chunks_text_list = text_splitter.split_text(cleaned_text)
            context.log.info(f"Document {doc_id_from_meta} split into {len(chunks_text_list)} chunks.")
            for i, chunk_text_content in enumerate(chunks_text_list):
                chunk_meta = parsed_doc.original_metadata.copy()
                chunk_meta.update({"chunk_number": i + 1, "total_chunks_for_doc": len(chunks_text_list)})
                all_chunks.append(ChunkOutput(chunk_text=chunk_text_content, source_document_id=doc_id_from_meta, chunk_metadata=chunk_meta))
        except Exception as e:
            context.log.error(f"Failed to chunk document {doc_id_from_meta}: {e}")
    context.add_output_metadata(metadata={"total_chunks_generated": len(all_chunks)})
    return all_chunks

@dg.asset(
    name="text_embeddings",
    description="Generates vector embeddings for text chunks.",
    group_name="processing",
    deps=["text_chunks"]
)
def generate_embeddings_asset(
    context: dg.AssetExecutionContext,
    text_chunks: List[ChunkOutput],
    embedder: GGUFEmbeddingResource
) -> List[EmbeddingOutput]:
    all_embeddings: List[EmbeddingOutput] = []
    if not text_chunks:
        return all_embeddings
    chunk_texts_to_encode = [chunk.chunk_text for chunk in text_chunks]
    vectors = embedder.encode(chunk_texts_to_encode)

    for i, chunk_input in enumerate(text_chunks):
        model_name_for_log = embedder.embedding_model_path
        all_embeddings.append(EmbeddingOutput(
            chunk_id=chunk_input.chunk_id,
            chunk_text=chunk_input.chunk_text,
            embedding_vector=vectors[i] if i < len(vectors) and vectors[i] else [0.0] * embedder.get_embedding_dimension(), # 提供默认值以防嵌入失败
            embedding_model_name=model_name_for_log,
            original_chunk_metadata=chunk_input.chunk_metadata
        ))
    context.add_output_metadata(metadata={"total_embeddings_generated": len(all_embeddings)})
    return all_embeddings

@dg.asset(
    name="vector_store_embeddings",
    description="Stores text embeddings into a ChromaDB vector store.",
    group_name="indexing",
    deps=["text_embeddings"]
)
def vector_storage_asset(
    context: dg.AssetExecutionContext,
    text_embeddings: List[EmbeddingOutput],
    chroma_db: ChromaDBResource
) -> None:
    if not text_embeddings:
        context.log.warning("No embeddings received, nothing to store.")
        return
    ids_to_store = [emb.chunk_id for emb in text_embeddings]
    embeddings_to_store = [emb.embedding_vector for emb in text_embeddings]
    metadatas_to_store = [emb.original_chunk_metadata for emb in text_embeddings]
    for meta, emb in zip(metadatas_to_store, text_embeddings):
        meta["chunk_text"] = emb.chunk_text
    chroma_db.add_embeddings(ids=ids_to_store, embeddings=embeddings_to_store, metadatas=metadatas_to_store)
    context.add_output_metadata(metadata={"num_embeddings_stored": len(ids_to_store)})

class BM25IndexConfig(dg.Config):
    index_file_path: str = "/home/zhz/zhz_agent/zhz_rag/stored_data/bm25_index/"

@dg.asset(
    name="keyword_index",
    description="Builds and persists a BM25 keyword index from text chunks.",
    group_name="indexing",
    deps=["text_chunks"]
)
def keyword_index_asset(
    context: dg.AssetExecutionContext,
    config: BM25IndexConfig,
    text_chunks: List[ChunkOutput]
) -> None:
    if not text_chunks:
        context.log.warning("No text chunks received, skipping BM25 index building.")
        return
    corpus_texts = [chunk.chunk_text for chunk in text_chunks]
    document_ids = [chunk.chunk_id for chunk in text_chunks]
    corpus_tokenized_jieba = [list(jieba.cut_for_search(text)) for text in corpus_texts]
    bm25_model = bm25s.BM25()
    bm25_model.index(corpus_tokenized_jieba)
    index_directory = config.index_file_path
    os.makedirs(index_directory, exist_ok=True)
    bm25_model.save(index_directory)
    doc_ids_path = os.path.join(index_directory, "doc_ids.pkl")
    with open(doc_ids_path, 'wb') as f_out:
        pickle.dump(document_ids, f_out)
    context.add_output_metadata(metadata={"num_documents_indexed": len(corpus_texts), "index_directory_path": index_directory})


# --- KG Extraction 相关的配置和资产 ---

class KGExtractionConfig(dg.Config):
    # --- [核心修复] 使用双重大括号 {{ 和 }} 转义所有JSON示例中的大括号 ---
    extraction_prompt_template: str = (
        "你是一个信息抽取助手。请从以下提供的文本中抽取出所有的人名(PERSON)、组织机构名(ORGANIZATION)和任务(TASK)实体。\n"
        "同时，请抽取出以下两种关系：\n"
        "1. WORKS_AT (当一个人在一个组织工作时，例如：PERSON WORKS_AT ORGANIZATION)\n"
        "2. ASSIGNED_TO (当一个任务分配给一个人时，例如：TASK ASSIGNED_TO PERSON)\n\n"
        "请严格按照以下JSON格式进行输出，不要包含任何额外的解释或Markdown标记：\n"
        "{{\n"
        "  \"entities\": [\n"
        "    {{\"text\": \"实体1原文\", \"label\": \"实体1类型\"}},\n"
        "    ...\n"
        "  ],\n"
        "  \"relations\": [\n"
        "    {{\"head_entity_text\": \"头实体文本\", \"head_entity_label\": \"头实体类型\", \"relation_type\": \"关系类型\", \"tail_entity_text\": \"尾实体文本\", \"tail_entity_label\": \"尾实体类型\"}},\n"
        "    ...\n"
        "  ]\n"
        "}}\n"
        "如果文本中没有可抽取的实体或关系，请返回一个空的对应列表 (例如 {{\"entities\": [], \"relations\": []}})。\n\n"
        "文本：\n"
        "\"{text_to_extract}\"\n\n"
        "JSON输出："
    )
    # --- [修复结束] ---
    local_llm_model_name: str = "Qwen3-1.7B-GGUF_via_llama.cpp"

# --- [核心修复] 在这里添加缺失的常量定义 ---
DEFAULT_KG_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "提取到的实体原文"},
                    "label": {"type": "string", "description": "实体类型 (例如: PERSON, ORGANIZATION, TASK)"}
                },
                "required": ["text", "label"]
            },
            "description": "从文本中提取出的实体列表。"
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "head_entity_text": {"type": "string", "description": "头实体的文本"},
                    "head_entity_label": {"type": "string", "description": "头实体的类型 (例如: PERSON, TASK)"},
                    "relation_type": {"type": "string", "description": "关系类型 (例如: WORKS_AT, ASSIGNED_TO)"},
                    "tail_entity_text": {"type": "string", "description": "尾实体的文本"},
                    "tail_entity_label": {"type": "string", "description": "尾实体的类型 (例如: ORGANIZATION, PERSON)"}
                },
                "required": ["head_entity_text", "head_entity_label", "relation_type", "tail_entity_text", "tail_entity_label"]
            },
            "description": "从文本中提取出的关系三元组列表。"
        }
    },
    "required": ["entities", "relations"]
}
# --- [修复结束] ---

@dg.asset(
    name="kg_extractions",
    description="Extracts entities and relations from text chunks for knowledge graph construction.",
    group_name="kg_building",
    io_manager_key="pydantic_json_io_manager",
    deps=["text_chunks"]
)
async def kg_extraction_asset(
    context: dg.AssetExecutionContext,
    text_chunks: List[ChunkOutput],
    config: KGExtractionConfig,
    sglang_api: LocalLLMAPIResource
) -> List[KGTripleSetOutput]:
    all_kg_outputs: List[KGTripleSetOutput] = []
    if not text_chunks:
        return all_kg_outputs
    total_entities_count = 0
    total_relations_count = 0
    for i, chunk in enumerate(text_chunks):
        prompt = config.extraction_prompt_template.format(text_to_extract=chunk.chunk_text)
        try:
            structured_response = await sglang_api.generate_structured_output(prompt=prompt, json_schema=DEFAULT_KG_EXTRACTION_SCHEMA)
            entities_data = structured_response.get("entities", [])
            extracted_entities_list = [ExtractedEntity(text=normalize_text_for_id(e["text"]), label=e["label"].upper()) for e in entities_data if isinstance(e, dict)]
            relations_data = structured_response.get("relations", [])
            extracted_relations_list = [ExtractedRelation(head_entity_text=r['head_entity_text'], head_entity_label=r['head_entity_label'], relation_type=r['relation_type'], tail_entity_text=r['tail_entity_text'], tail_entity_label=r['tail_entity_label']) for r in relations_data if isinstance(r, dict)]
            total_entities_count += len(extracted_entities_list)
            total_relations_count += len(extracted_relations_list)
            all_kg_outputs.append(KGTripleSetOutput(
                chunk_id=chunk.chunk_id,
                extracted_entities=extracted_entities_list,
                extracted_relations=extracted_relations_list,
                extraction_model_name=config.local_llm_model_name,
                original_chunk_metadata=chunk.chunk_metadata
            ))
        except Exception as e:
            context.log.error(f"Failed KG extraction for chunk {chunk.chunk_id}: {e}", exc_info=True)
    context.add_output_metadata(metadata={"total_entities_extracted": total_entities_count, "total_relations_extracted": total_relations_count})
    return all_kg_outputs

# --- KuzuDB 构建资产链 ---

@dg.asset(
    name="duckdb_schema", # <--- 修改资产名称
    description="Creates the base schema (node and relation tables) in DuckDB.",
    group_name="kg_building",
    # deps=[kg_extraction_asset] # 保持依赖，确保在提取之后创建schema (逻辑上)
                                 # 虽然schema创建本身不直接使用提取结果，但流水线顺序上合理
)
def duckdb_schema_asset(context: dg.AssetExecutionContext, duckdb_kg: DuckDBResource, embedder: GGUFEmbeddingResource): # <--- 修改函数名和资源参数
    context.log.info("--- Starting DuckDB Schema Creation Asset ---")
    
    # 获取嵌入维度，与KuzuDB时类似
    EMBEDDING_DIM = embedder.get_embedding_dimension()
    if not EMBEDDING_DIM:
        raise ValueError("Could not determine embedding dimension from GGUFEmbeddingResource.")

    node_table_ddl = f"""
    CREATE TABLE IF NOT EXISTS ExtractedEntity (
        id_prop VARCHAR PRIMARY KEY,
        text VARCHAR,
        label VARCHAR,
        embedding FLOAT[{EMBEDDING_DIM}]
    );
    """

    relation_table_ddl = f"""
    CREATE TABLE IF NOT EXISTS KGExtractionRelation (
        relation_id VARCHAR PRIMARY KEY,
        source_node_id_prop VARCHAR,
        target_node_id_prop VARCHAR,
        relation_type VARCHAR
        -- Optional: FOREIGN KEY (source_node_id_prop) REFERENCES ExtractedEntity(id_prop),
        -- Optional: FOREIGN KEY (target_node_id_prop) REFERENCES ExtractedEntity(id_prop)
    );
    """
    # 也可以为关系表的 (source, target, type) 创建复合唯一索引或普通索引以加速查询
    relation_index_ddl = """
    CREATE INDEX IF NOT EXISTS idx_relation_source_target_type 
    ON KGExtractionRelation (source_node_id_prop, target_node_id_prop, relation_type);
    """
    
    ddl_commands = [node_table_ddl, relation_table_ddl, relation_index_ddl]

    try:
        with duckdb_kg.get_connection() as conn:
            context.log.info("Executing DuckDB DDL commands...")
            for command_idx, command in enumerate(ddl_commands):
                context.log.debug(f"Executing DDL {command_idx+1}:\n{command.strip()}")
                conn.execute(command)
            context.log.info("DuckDB Schema DDL commands executed successfully.")
    except Exception as e_ddl:
        context.log.error(f"Error during DuckDB schema creation: {e_ddl}", exc_info=True)
        raise
    context.log.info("--- DuckDB Schema Creation Asset Finished ---")


@dg.asset(
    name="duckdb_nodes", # <--- 修改资产名称
    description="Loads all unique extracted entities as nodes into DuckDB.",
    group_name="kg_building",
    deps=[duckdb_schema_asset, kg_extraction_asset] # <--- 修改依赖
)
def duckdb_nodes_asset(
    context: dg.AssetExecutionContext,
    kg_extractions: List[KGTripleSetOutput], # 来自 kg_extraction_asset 的输出
    duckdb_kg: DuckDBResource,               # <--- 修改资源参数
    embedder: GGUFEmbeddingResource          # 保持对 embedder 的依赖，用于生成嵌入
):
    context.log.info("--- Starting DuckDB Node Loading Asset (Using INSERT ON CONFLICT) ---")
    if not kg_extractions:
        context.log.warning("No KG extractions received. Skipping node loading.")
        return

    unique_nodes_data_for_insert: List[Dict[str, Any]] = []
    unique_nodes_keys = set() # 用于在Python层面去重，避免多次尝试插入相同实体

    for kg_set in kg_extractions:
        for entity in kg_set.extracted_entities:
            # 规范化文本和标签，用于生成唯一键和存储
            normalized_text = normalize_text_for_id(entity.text)
            normalized_label = entity.label.upper() # 确保标签大写
            
            # 为实体生成唯一ID (基于规范化文本和标签的哈希值)
            # 注意：如果同一个实体（相同文本和标签）在不同chunk中被提取，它们的id_prop会一样
            node_id_prop = hashlib.md5(f"{normalized_text}_{normalized_label}".encode('utf-8')).hexdigest()
            
            node_unique_key_for_py_dedup = (node_id_prop) # 使用id_prop进行Python层面的去重

            if node_unique_key_for_py_dedup not in unique_nodes_keys:
                unique_nodes_keys.add(node_unique_key_for_py_dedup)
                
                # 生成嵌入向量 (与KuzuDB时逻辑相同)
                embedding_vector_list = embedder.encode([normalized_text]) # embedder.encode期望一个列表
                final_embedding_for_db: List[float]

                if embedding_vector_list and embedding_vector_list[0] and \
                   isinstance(embedding_vector_list[0], list) and \
                   len(embedding_vector_list[0]) == embedder.get_embedding_dimension():
                    final_embedding_for_db = embedding_vector_list[0]
                else:
                    context.log.warning(f"Failed to generate valid embedding for node: {normalized_text} ({normalized_label}). Using zero vector. Embedding result: {embedding_vector_list}")
                    final_embedding_for_db = [0.0] * embedder.get_embedding_dimension()
                    
                unique_nodes_data_for_insert.append({
                    "id_prop": node_id_prop,
                    "text": normalized_text,
                    "label": normalized_label,
                    "embedding": final_embedding_for_db # DuckDB的FLOAT[]可以直接接受Python的List[float]
                })

    if not unique_nodes_data_for_insert:
        context.log.warning("No unique nodes found in extractions to load into DuckDB.")
        return

    nodes_processed_count = 0
    nodes_inserted_count = 0
    nodes_updated_count = 0

    upsert_sql = f"""
    INSERT INTO ExtractedEntity (id_prop, text, label, embedding)
    VALUES (?, ?, ?, ?)
    ON CONFLICT (id_prop) DO UPDATE SET
        text = excluded.text,
        label = excluded.label,
        embedding = excluded.embedding;
    """
    # excluded.column_name 用于引用试图插入但导致冲突的值

    try:
        with duckdb_kg.get_connection() as conn:
            context.log.info(f"Attempting to UPSERT {len(unique_nodes_data_for_insert)} unique nodes into DuckDB ExtractedEntity table...")
            
            # DuckDB 支持 executemany 用于批量操作，但对于 ON CONFLICT，逐条执行或构造大型 VALUES 列表可能更直接
            # 或者使用 pandas DataFrame + duckdb.register + CREATE TABLE AS / INSERT INTO SELECT
            # 这里为了清晰，我们先用循环执行，对于几千到几万个节点，性能尚可接受
            # 如果节点数量非常大 (几十万以上)，应考虑更优化的批量upsert策略

            for node_data_dict in unique_nodes_data_for_insert:
                params = (
                    node_data_dict["id_prop"],
                    node_data_dict["text"],
                    node_data_dict["label"],
                    node_data_dict["embedding"]
                )
                try:
                    # conn.execute() 对于 DML (如 INSERT, UPDATE) 不直接返回受影响的行数
                    # 但我们可以假设它成功了，除非抛出异常
                    conn.execute(upsert_sql, params)
                    # 无法直接判断是insert还是update，除非查询前后对比，这里简化处理
                    nodes_processed_count += 1 
                except Exception as e_upsert_item:
                    context.log.error(f"Error UPSERTING node with id_prop {node_data_dict.get('id_prop')} into DuckDB: {e_upsert_item}", exc_info=True)
            
            # 我们可以查一下表中的总行数来间接了解情况
            total_rows_after = conn.execute("SELECT COUNT(*) FROM ExtractedEntity").fetchone()[0]
            context.log.info(f"Successfully processed {nodes_processed_count} node upsert operations into DuckDB.")
            context.log.info(f"Total rows in ExtractedEntity table after upsert: {total_rows_after}")
            # 注意：这里的 nodes_processed_count 不直接等于插入或更新数，而是尝试操作的次数
            # 如果需要精确计数插入/更新，需要更复杂的逻辑或DuckDB特定功能

    except Exception as e_db_nodes:
        context.log.error(f"Error during DuckDB node loading: {e_db_nodes}", exc_info=True)
        raise
    
    context.add_output_metadata({
        "nodes_prepared_for_upsert": len(unique_nodes_data_for_insert),
        "nodes_processed_by_upsert_statement": nodes_processed_count,
    })
    context.log.info("--- DuckDB Node Loading Asset Finished ---")



@dg.asset(
    name="duckdb_relations", # <--- 修改资产名称
    description="Loads all extracted relationships into DuckDB.",
    group_name="kg_building",
    deps=[duckdb_nodes_asset] # <--- 修改依赖
)
def duckdb_relations_asset(
    context: dg.AssetExecutionContext, 
    kg_extractions: List[KGTripleSetOutput], # 来自 kg_extraction_asset
    duckdb_kg: DuckDBResource                # <--- 修改资源参数
):
    context.log.info("--- Starting DuckDB Relation Loading Asset ---")
    if not kg_extractions:
        context.log.warning("No KG extractions received. Skipping relation loading.")
        return

    relations_to_insert: List[Dict[str, str]] = []
    unique_relation_keys = set() # 用于在Python层面去重

    for kg_set in kg_extractions:
        for rel in kg_set.extracted_relations:
            # 从实体文本和标签生成源节点和目标节点的ID (与 duckdb_nodes_asset 中一致)
            source_node_text_norm = normalize_text_for_id(rel.head_entity_text)
            source_node_label_norm = rel.head_entity_label.upper()
            source_node_id = hashlib.md5(f"{source_node_text_norm}_{source_node_label_norm}".encode('utf-8')).hexdigest()

            target_node_text_norm = normalize_text_for_id(rel.tail_entity_text)
            target_node_label_norm = rel.tail_entity_label.upper()
            target_node_id = hashlib.md5(f"{target_node_text_norm}_{target_node_label_norm}".encode('utf-8')).hexdigest()
            
            relation_type_norm = rel.relation_type.upper()

            # 为关系本身生成一个唯一ID
            relation_unique_str = f"{source_node_id}_{relation_type_norm}_{target_node_id}"
            relation_id = hashlib.md5(relation_unique_str.encode('utf-8')).hexdigest()

            if relation_id not in unique_relation_keys:
                unique_relation_keys.add(relation_id)
                relations_to_insert.append({
                    "relation_id": relation_id,
                    "source_node_id_prop": source_node_id,
                    "target_node_id_prop": target_node_id,
                    "relation_type": relation_type_norm
                })
    
    if not relations_to_insert:
        context.log.warning("No unique relations found in extractions to load into DuckDB.")
        return

    relations_processed_count = 0
    
    # 使用 INSERT INTO ... ON CONFLICT DO NOTHING 来避免插入重复的关系 (基于 relation_id)
    insert_sql = """
    INSERT INTO KGExtractionRelation (relation_id, source_node_id_prop, target_node_id_prop, relation_type)
    VALUES (?, ?, ?, ?)
    ON CONFLICT (relation_id) DO NOTHING;
    """

    try:
        with duckdb_kg.get_connection() as conn:
            context.log.info(f"Attempting to INSERT {len(relations_to_insert)} unique relations into DuckDB KGExtractionRelation table...")
            
            for rel_data_dict in relations_to_insert:
                params = (
                    rel_data_dict["relation_id"],
                    rel_data_dict["source_node_id_prop"],
                    rel_data_dict["target_node_id_prop"],
                    rel_data_dict["relation_type"]
                )
                try:
                    conn.execute(insert_sql, params)
                    # DuckDB的execute对于INSERT ON CONFLICT DO NOTHING不直接返回是否插入
                    # 但我们可以假设它成功处理了（要么插入，要么忽略）
                    relations_processed_count += 1
                except Exception as e_insert_item:
                    context.log.error(f"Error INSERTING relation with id {rel_data_dict.get('relation_id')} into DuckDB: {e_insert_item}", exc_info=True)
            
            total_rels_after = conn.execute("SELECT COUNT(*) FROM KGExtractionRelation").fetchone()[0]
            context.log.info(f"Successfully processed {relations_processed_count} relation insert (ON CONFLICT DO NOTHING) operations.")
            context.log.info(f"Total rows in KGExtractionRelation table after inserts: {total_rels_after}")

    except Exception as e_db_rels:
        context.log.error(f"Error during DuckDB relation loading: {e_db_rels}", exc_info=True)
        raise
        
    context.add_output_metadata({
        "relations_prepared_for_insert": len(relations_to_insert),
        "relations_processed_by_insert_statement": relations_processed_count,
    })
    context.log.info("--- DuckDB Relation Loading Asset Finished ---")



@dg.asset(
    name="duckdb_vector_index", # <--- 修改资产名称
    description="Creates the HNSW vector index on the embedding column in DuckDB.",
    group_name="kg_building",
    deps=[duckdb_relations_asset]  # <--- 修改依赖
)
def duckdb_vector_index_asset(
    context: dg.AssetExecutionContext, 
    duckdb_kg: DuckDBResource # <--- 修改资源参数
):
    context.log.info("--- Starting DuckDB Vector Index Creation Asset ---")
    
    table_to_index = "ExtractedEntity"
    column_to_index = "embedding"
    # 索引名可以自定义，通常包含表名、列名和类型
    index_name = f"{table_to_index}_{column_to_index}_hnsw_idx"
    metric_type = "l2sq" # 欧氏距离的平方，与我们测试时一致

    # DuckDB 的 CREATE INDEX ... USING HNSW 语句
    # IF NOT EXISTS 确保了幂等性
    index_creation_sql = f"""
    CREATE INDEX IF NOT EXISTS {index_name} 
    ON {table_to_index} USING HNSW ({column_to_index}) 
    WITH (metric='{metric_type}');
    """

    try:
        with duckdb_kg.get_connection() as conn:
            # 在创建索引前，确保vss扩展已加载且持久化已开启 (虽然DuckDBResource的setup已做)
            try:
                conn.execute("LOAD vss;")
                conn.execute("SET hnsw_enable_experimental_persistence=true;")
                context.log.info("DuckDB: VSS extension loaded and HNSW persistence re-confirmed for index creation asset.")
            except Exception as e_vss_setup_idx:
                context.log.warning(f"DuckDB: Failed to re-confirm VSS setup for index asset: {e_vss_setup_idx}. "
                                     "Proceeding, assuming it was set by DuckDBResource.")

            context.log.info(f"Executing DuckDB vector index creation command:\n{index_creation_sql.strip()}")
            conn.execute(index_creation_sql)
            context.log.info(f"DuckDB vector index '{index_name}' creation command executed successfully (or index already existed).")

            # 验证索引是否已创建 (可选，但有助于确认)
            # DuckDB 中查看索引的命令可能不像 KuzuDB 那样直接，
            # 通常，如果 CREATE INDEX 没有报错，就表示成功了。
            # 我们可以尝试查询 pg_indexes 或 duckdb_indexes() (如果 DuckDB 版本支持)
            # 或者简单地依赖 CREATE INDEX IF NOT EXISTS 的行为。
            # 为了简化，我们这里先不加复杂的验证查询。
            # 如果需要验证，可以查询 PRAGMA show_indexes('ExtractedEntity'); 但解析其输出较麻烦。

    except Exception as e_index_asset:
        context.log.error(f"Error during DuckDB vector index creation: {e_index_asset}", exc_info=True)
        raise
    
    context.log.info("--- DuckDB Vector Index Creation Asset Finished ---")


# --- 更新 all_processing_assets 列表 ---
all_processing_assets = [
    clean_chunk_text_asset,
    generate_embeddings_asset,
    vector_storage_asset,
    keyword_index_asset,
    kg_extraction_asset,
    duckdb_schema_asset,
    duckdb_nodes_asset,
    duckdb_relations_asset,
    duckdb_vector_index_asset,
]