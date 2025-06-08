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
    SentenceTransformerResource,
    ChromaDBResource,
    LocalLLMAPIResource,
    KuzuDBReadWriteResource,
    KuzuDBReadOnlyResource
)
import jieba
import bm25s
import pickle
import numpy as np
import os
import time

# --- text_chunks, text_embeddings, vector_store_embeddings, keyword_index 资产定义保持不变 ---
# (为了保持代码完整，这里包含了所有资产的代码)

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
    embedder: SentenceTransformerResource
) -> List[EmbeddingOutput]:
    all_embeddings: List[EmbeddingOutput] = []
    if not text_chunks:
        return all_embeddings
    chunk_texts_to_encode = [chunk.chunk_text for chunk in text_chunks]
    vectors = embedder.encode(chunk_texts_to_encode)
    for i, chunk_input in enumerate(text_chunks):
        all_embeddings.append(EmbeddingOutput(
            chunk_id=chunk_input.chunk_id,
            chunk_text=chunk_input.chunk_text,
            embedding_vector=vectors[i],
            embedding_model_name=embedder.model_name_or_path,
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
    name="kuzu_schema",
    description="Creates the base schema (node and rel tables) in KuzuDB.",
    group_name="kg_building",
    deps=[kg_extraction_asset]
)
def kuzu_schema_asset(context: dg.AssetExecutionContext, kuzu_readwrite_db: KuzuDBReadWriteResource, embedder: SentenceTransformerResource):
    context.log.info("--- Starting KuzuDB Schema Creation Asset ---")
    with kuzu_readwrite_db.get_connection() as conn:
        EMBEDDING_DIM = embedder.get_embedding_dimension()
        conn.execute(
            f"CREATE NODE TABLE IF NOT EXISTS ExtractedEntity (id_prop STRING, text STRING, label STRING, embedding FLOAT[{EMBEDDING_DIM}], PRIMARY KEY (id_prop))"
        )
        # --- [核心修复] 将关系表名改为全大写，以匹配后续 MERGE 操作 ---
        conn.execute("CREATE REL TABLE IF NOT EXISTS WORKS_AT (FROM ExtractedEntity TO ExtractedEntity)")
        conn.execute("CREATE REL TABLE IF NOT EXISTS ASSIGNED_TO (FROM ExtractedEntity TO ExtractedEntity)")
        # --- [修复结束] ---
        context.log.info("KuzuDB Schema DDL commands executed successfully.")

@dg.asset(
    name="kuzu_nodes",
    description="Loads all unique extracted entities as nodes into KuzuDB.",
    group_name="kg_building",
    deps=["kuzu_schema", "kg_extractions"]
)
def kuzu_nodes_asset(context: dg.AssetExecutionContext, kg_extractions: List[KGTripleSetOutput], kuzu_readwrite_db: KuzuDBReadWriteResource, embedder: SentenceTransformerResource):
    context.log.info("--- Starting KuzuDB Node Loading Asset ---")
    if not kg_extractions:
        context.log.warning("No KG extractions received. Skipping node loading.")
        return
    unique_nodes = {}
    for kg_set in kg_extractions:
        for entity in kg_set.extracted_entities:
            node_key = (normalize_text_for_id(entity.text), entity.label.upper())
            if node_key not in unique_nodes:
                unique_nodes[node_key] = {
                    "id_prop": hashlib.md5(f"{node_key[0]}_{node_key[1]}".encode('utf-8')).hexdigest(),
                    "text": node_key[0],
                    "label": node_key[1],
                    "embedding": embedder.encode([node_key[0]])[0]
                }
    node_data = list(unique_nodes.values())
    if not node_data:
        context.log.warning("No unique nodes found in extractions to load.")
        return
    with kuzu_readwrite_db.get_connection() as conn:
        nodes_df = pd.DataFrame(node_data)
        context.log.info(f"Copying {len(nodes_df)} unique nodes into ExtractedEntity table...")
        conn.execute("COPY ExtractedEntity FROM nodes_df;")
        conn.execute("CHECKPOINT;")
        context.log.info("Successfully copied nodes and checkpointed.")

@dg.asset(
    name="kuzu_relations",
    description="Loads all extracted relationships into KuzuDB.",
    group_name="kg_building",
    deps=["kuzu_nodes"]
)
def kuzu_relations_asset(context: dg.AssetExecutionContext, kg_extractions: List[KGTripleSetOutput], kuzu_readwrite_db: KuzuDBReadWriteResource):
    context.log.info("--- Starting KuzuDB Relation Loading Asset ---")
    if not kg_extractions:
        context.log.warning("No KG extractions received. Skipping relation loading.")
        return
    relation_data = []
    for kg_set in kg_extractions:
        for rel in kg_set.extracted_relations:
            head_text_norm = normalize_text_for_id(rel.head_entity_text)
            head_label_norm = rel.head_entity_label.upper()
            tail_text_norm = normalize_text_for_id(rel.tail_entity_text)
            tail_label_norm = rel.tail_entity_label.upper()
            relation_data.append({
                "from_id": hashlib.md5(f"{head_text_norm}_{head_label_norm}".encode('utf-8')).hexdigest(),
                "to_id": hashlib.md5(f"{tail_text_norm}_{tail_label_norm}".encode('utf-8')).hexdigest(),
                "relation_type": rel.relation_type.upper()
            })
    if not relation_data:
        context.log.warning("No relation data found in extractions to load.")
        return
    with kuzu_readwrite_db.get_connection() as conn:
        all_rels_df = pd.DataFrame(relation_data)
        for rel_type in all_rels_df['relation_type'].unique():
            rels_df_for_merge = all_rels_df[all_rels_df['relation_type'] == rel_type]
            batch_params = rels_df_for_merge.to_dict(orient='records')
            if not batch_params: continue
            context.log.info(f"Attempting to MERGE {len(batch_params)} relations of type '{rel_type}'...")
            merge_query = f"""
                UNWIND $batch AS rel_item
                MATCH (from_node:ExtractedEntity {{id_prop: rel_item.from_id}})
                MATCH (to_node:ExtractedEntity {{id_prop: rel_item.to_id}})
                MERGE (from_node)-[:`{rel_type}`]->(to_node)
            """
            conn.execute(merge_query, {"batch": batch_params})
            context.log.info(f"Successfully merged relations of type '{rel_type}'.")
        conn.execute("CHECKPOINT;")
        context.log.info("Successfully merged all relations and checkpointed.")

@dg.asset(
    name="kuzu_vector_index",
    description="Creates the vector index on the ExtractedEntity table.",
    group_name="kg_building",
    deps=["kuzu_relations"]
)
def kuzu_vector_index_asset(context: dg.AssetExecutionContext, kuzu_readwrite_db: KuzuDBReadWriteResource):
    context.log.info("--- Starting KuzuDB Vector Index Creation Asset ---")
    with kuzu_readwrite_db.get_connection() as conn:
        try:
            conn.execute("LOAD VECTOR;")
            context.log.info("VECTOR extension loaded for index creation.")
        except Exception as e_load_vec:
            context.log.warning(f"Could not explicitly LOAD VECTOR (might be already loaded): {e_load_vec}")
        context.log.info("Executing vector index creation command...")
        conn.execute("CALL CREATE_VECTOR_INDEX('ExtractedEntity', 'entity_embedding_idx', 'embedding', metric := 'cosine')")
        context.log.info("Vector index creation command successfully executed.")

# --- 更新 all_processing_assets 列表 ---
all_processing_assets = [
    clean_chunk_text_asset,
    generate_embeddings_asset,
    vector_storage_asset,
    keyword_index_asset,
    kg_extraction_asset,
    kuzu_schema_asset,
    kuzu_nodes_asset,
    kuzu_relations_asset,
    kuzu_vector_index_asset,
]