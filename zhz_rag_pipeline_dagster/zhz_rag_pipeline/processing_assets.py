# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/zhz_rag_pipeline/processing_assets.py
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
    SGLangAPIResource,
    KuzuDBReadWriteResource,
    KuzuDBReadOnlyResource
)
import jieba
import bm25s
import pickle
import numpy as np
import os

class TextChunkerConfig(dg.Config):
    chunk_size: int = 500
    chunk_overlap: int = 50
    # separators: Optional[List[str]] = None # 可选的自定义分隔符

@dg.asset(
    name="text_chunks",
    description="Cleans and chunks parsed documents into smaller text segments.",
    group_name="processing", # 属于处理组
    # deps=["parsed_documents"] # <--- 删除或注释掉这一行
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
        # length_function=len, # 默认
        # add_start_index=True, # 如果需要块的起始索引
        # separators=config.separators if config.separators else None # 使用配置的分隔符
    )

    for parsed_doc in parsed_documents:
        # 使用原始文件名作为文档ID，如果不存在则生成UUID
        doc_id_from_meta = parsed_doc.original_metadata.get("filename")
        if not doc_id_from_meta:
            doc_id_from_meta = parsed_doc.original_metadata.get("document_path", str(uuid.uuid4()))
            context.log.warning(f"Filename not found in metadata for a document, using path or UUID: {doc_id_from_meta}")
        
        source_dir = parsed_doc.original_metadata.get('source_directory', 'Unknown_Source_Dir')
        context.log.info(f"Processing document: {doc_id_from_meta} (from {source_dir})")

        cleaned_text = parsed_doc.parsed_text.strip()

        if not cleaned_text or cleaned_text.startswith("[Unsupported file type:") or cleaned_text.startswith("[Error parsing document:"):
            context.log.warning(f"Document {doc_id_from_meta} has no valid content or was unsupported/errored in parsing, skipping chunking.")
            continue
        
        try:
            chunks_text_list = text_splitter.split_text(cleaned_text)
            context.log.info(f"Document {doc_id_from_meta} split into {len(chunks_text_list)} chunks.")

            for i, chunk_text_content in enumerate(chunks_text_list):
                chunk_meta = parsed_doc.original_metadata.copy() 
                chunk_meta.update({
                    "chunk_number": i + 1,
                    "total_chunks_for_doc": len(chunks_text_list),
                    "chunk_char_length": len(chunk_text_content),
                })

                chunk_output = ChunkOutput( # chunk_id 会在 ChunkOutput 的 __init__ 中自动生成
                    chunk_text=chunk_text_content,
                    source_document_id=doc_id_from_meta, # 使用从元数据获取的文档ID
                    chunk_metadata=chunk_meta
                )
                all_chunks.append(chunk_output)
        
        except Exception as e:
            context.log.error(f"Failed to chunk document {doc_id_from_meta}: {e}")
            
    if all_chunks:
        context.add_output_metadata(
            metadata={
                "total_chunks_generated": len(all_chunks),
                "first_chunk_doc_id": all_chunks[0].source_document_id if all_chunks else "N/A"
            }
        )
    else:
        context.log.warning("No chunks were generated from the input documents.")
        
    return all_chunks

# --- 新增：EmbeddingGenerationAgent (实现为 Asset) ---
@dg.asset(
    name="text_embeddings",
    description="Generates vector embeddings for text chunks using a SentenceTransformer model.",
    group_name="processing",
    # deps=["text_chunks"] # <--- 删除或注释掉这一行
)
def generate_embeddings_asset(
    context: dg.AssetExecutionContext,
    text_chunks: List[ChunkOutput], # 输入是上游资产的输出列表
    embedder: SentenceTransformerResource # <--- 注入我们定义的Resource
) -> List[EmbeddingOutput]:
    
    all_embeddings: List[EmbeddingOutput] = []
    context.log.info(f"Received {len(text_chunks)} text chunks to generate embeddings for.")

    if not text_chunks:
        context.log.warning("No text chunks received, skipping embedding generation.")
        return all_embeddings

    # 提取所有块的文本内容进行批量编码
    chunk_texts_to_encode = [chunk.chunk_text for chunk in text_chunks]
    
    try:
        context.log.info(f"Starting batch embedding generation for {len(chunk_texts_to_encode)} texts...")
        # 使用Resource的encode方法
        vectors = embedder.encode(chunk_texts_to_encode) 
        context.log.info(f"Successfully generated {len(vectors)} embedding vectors.")

        if len(vectors) != len(text_chunks):
            # 这是一个预期外的情况，应该记录严重错误
            context.log.error(f"Mismatch in number of chunks ({len(text_chunks)}) and generated vectors ({len(vectors)}). Aborting.")
            # 可以在这里抛出异常来使资产失败
            raise ValueError("Embedding generation resulted in a mismatched number of vectors.")

        for i, chunk_input in enumerate(text_chunks):
            embedding_output = EmbeddingOutput(
                chunk_id=chunk_input.chunk_id,
                chunk_text=chunk_input.chunk_text,
                embedding_vector=vectors[i],
                embedding_model_name=embedder.model_name_or_path, # 从Resource获取模型名
                original_chunk_metadata=chunk_input.chunk_metadata
            )
            all_embeddings.append(embedding_output)
        
        context.log.info(f"All {len(all_embeddings)} embeddings prepared.")

    except Exception as e:
        context.log.error(f"Failed to generate embeddings: {e}")
        # 根据策略，可以选择让资产失败，或者返回空列表/部分结果
        # 这里我们选择抛出异常，让资产运行失败，以便调查
        raise

    if all_embeddings:
        context.add_output_metadata(
            metadata={
                "total_embeddings_generated": len(all_embeddings),
                "embedding_model_used": embedder.model_name_or_path,
                "first_chunk_id_embedded": all_embeddings[0].chunk_id if all_embeddings else "N/A"
            }
        )
    return all_embeddings

# --- 新增：VectorStorageAgent (实现为 Asset) ---
@dg.asset(
    name="vector_store_embeddings",
    description="Stores text embeddings into a ChromaDB vector store.",
    group_name="indexing", # 新的分组
    # deps=["text_embeddings"] # 依赖上一个资产的name
)
def vector_storage_asset(
    context: dg.AssetExecutionContext,
    text_embeddings: List[EmbeddingOutput], # 输入是上游资产的输出列表
    chroma_db: ChromaDBResource # <--- 注入ChromaDB Resource
) -> None: # 这个资产通常只执行操作，不产生新的可传递数据资产，所以返回None
    
    context.log.info(f"Received {len(text_embeddings)} embeddings to store in ChromaDB.")

    if not text_embeddings:
        context.log.warning("No embeddings received, nothing to store.")
        # 可以在这里添加一个 AssetMaterialization 来记录这个空操作
        context.add_output_metadata(metadata={"num_embeddings_stored": 0, "status": "No data to store"})
        return

    ids_to_store: List[str] = []
    embeddings_to_store: List[List[float]] = []
    metadatas_to_store: List[Dict[str, Any]] = []

    for emb_output in text_embeddings: # emb_output 是 EmbeddingOutput 类型
        ids_to_store.append(emb_output.chunk_id) 
        embeddings_to_store.append(emb_output.embedding_vector)
        
        simple_metadata = {}
        for key, value in emb_output.original_chunk_metadata.items(): # 从原始块元数据开始
            if isinstance(value, (str, int, float, bool)):
                simple_metadata[key] = value
            else:
                simple_metadata[key] = str(value) 
        
        simple_metadata["chunk_text"] = emb_output.chunk_text # <--- 直接从emb_output获取

        metadatas_to_store.append(simple_metadata)

    try:
        chroma_db.add_embeddings(
            ids=ids_to_store,
            embeddings=embeddings_to_store,
            metadatas=metadatas_to_store
        )
        context.log.info(f"Successfully stored/updated {len(ids_to_store)} embeddings in ChromaDB.")
        
        # 记录物化信息
        context.add_output_metadata(
            metadata={
                "num_embeddings_stored": len(ids_to_store),
                "collection_name": chroma_db.collection_name,
                "status": "Success"
            }
        )
    except Exception as e:
        context.log.error(f"Failed to store embeddings in ChromaDB: {e}")
        context.add_output_metadata(
            metadata={
                "num_embeddings_stored": 0,
                "collection_name": chroma_db.collection_name,
                "status": f"Failed: {str(e)}"
            }
        )
        raise # 让资产失败


# --- 新增：定义 KeywordIndexAgent 的配置 Pydantic 模型 ---
class BM25IndexConfig(dg.Config):
    index_file_path: str = "/home/zhz/zhz_agent/zhz_rag/stored_data/bm25_index/"

# --- 修改：KeywordIndexAgent (实现为 Asset) ---
@dg.asset(
    name="keyword_index",
    description="Builds and persists a BM25 keyword index from text chunks.",
    group_name="indexing",
    # deps=["text_chunks"] 
)
def keyword_index_asset(
    context: dg.AssetExecutionContext,
    config: BM25IndexConfig,
    text_chunks: List[ChunkOutput] 
) -> None:
    
    context.log.info(f"Received {len(text_chunks)} text chunks to build BM25 index.")

    if not text_chunks:
        # ... (无数据处理不变) ...
        context.log.warning("No text chunks received, skipping BM25 index building.")
        context.add_output_metadata(metadata={"num_documents_indexed": 0, "status": "No data"})
        return

    corpus_texts: List[str] = [chunk.chunk_text for chunk in text_chunks]
    document_ids: List[str] = [chunk.chunk_id for chunk in text_chunks] # 我们仍然需要保存这个映射

    context.log.info("Tokenizing corpus using jieba...")
    # 使用jieba分词，但bm25s有自己的tokenize函数，可以接受自定义分词器，或者直接处理分词后的列表
    # 为了与bm25s的tokenize函数配合，我们可以先用jieba分好，再传给bm25s的tokenizer
    # 或者，如果bm25s的默认分词或其Tokenizer类能满足中文需求，可以直接用。
    # 为了简单且利用jieba，我们先分词
    corpus_tokenized_jieba = [list(jieba.cut_for_search(text)) for text in corpus_texts]
    context.log.info(f"Jieba tokenization complete. Example: {corpus_tokenized_jieba[0][:10] if corpus_tokenized_jieba else 'N/A'}")

    context.log.info("Initializing BM25s model and indexing corpus...")
    try:
        # 根据bm25s文档，先创建BM25对象，然后调用index方法
        bm25_model = bm25s.BM25() # 可以传入k1, b等参数
        # .index() 方法接受已经分词的语料库 (list of list of str)
        bm25_model.index(corpus_tokenized_jieba)
        context.log.info("BM25s model indexed successfully.")
    except Exception as e:
        context.log.error(f"Failed to initialize or index with BM25s model: {e}")
        raise

    index_directory = config.index_file_path # 这应该是一个目录路径

    if not os.path.exists(index_directory):
        os.makedirs(index_directory, exist_ok=True)
        context.log.info(f"Created directory for BM25 index: {index_directory}")

    try:
        context.log.info(f"Saving BM25 model to directory: {index_directory}")
        # 使用bm25s的save方法，它会将多个文件保存到该目录下
        # 它会自动保存词汇表 (vocab.index.json) 和其他必要文件
        bm25_model.save(
            index_directory,
            # 文件名参数是可选的，bm25s有默认文件名，例如：
            # data_name="data.csc.index.npy",
            # indices_name="indices.csc.index.npy",
            # indptr_name="indptr.csc.index.npy",
            # vocab_name="vocab.index.json", 
            # params_name="params.index.json"
        )
        context.log.info(f"BM25 model saved successfully to {index_directory}")

        # 单独保存我们的 document_ids 列表，因为bm25s内部索引是基于0,1,2...
        # 而我们需要映射回原始的chunk_id
        doc_ids_path = os.path.join(index_directory, "doc_ids.pkl") # 保持这个文件名
        with open(doc_ids_path, 'wb') as f_out:
            pickle.dump(document_ids, f_out)
        context.log.info(f"Document IDs saved successfully to {doc_ids_path}")
        
        context.add_output_metadata(
            metadata={
                "num_documents_indexed": len(corpus_texts),
                "index_directory_path": index_directory, # <--- 修改：现在是目录路径
                "status": "Success"
            }
        )
    except Exception as e:
        # ... (错误处理不变) ...
        context.log.error(f"Failed to save BM25 model or document IDs: {e}")
        context.add_output_metadata(
            metadata={
                "num_documents_indexed": 0,
                "index_directory_path": index_directory,
                "status": f"Failed to save index: {str(e)}"
            }
        )
        raise

# 知识图谱
DEFAULT_KG_EXTRACTION_SCHEMA = { # <--- 覆盖这里的整个字典
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
        "relations": { # <--- 新增/确保这部分存在且正确
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
    "required": ["entities", "relations"] # <--- 确保 "relations" 也在这里
}

class KGExtractionConfig(dg.Config):
    extraction_prompt_template: str = ( # <--- 覆盖这里的整个多行字符串
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
        "示例文本1：'项目Alpha的文档编写任务分配给了张三。张三在谷歌工作。'\n"
        "期望JSON输出1：\n"
        "{{\n"
        "  \"entities\": [\n"
        "    {{\"text\": \"项目Alpha的文档编写任务\", \"label\": \"TASK\"}},\n"
        "    {{\"text\": \"张三\", \"label\": \"PERSON\"}},\n"
        "    {{\"text\": \"谷歌\", \"label\": \"ORGANIZATION\"}}\n"
        "  ],\n"
        "  \"relations\": [\n"
        "    {{\"head_entity_text\": \"项目Alpha的文档编写任务\", \"head_entity_label\": \"TASK\", \"relation_type\": \"ASSIGNED_TO\", \"tail_entity_text\": \"张三\", \"tail_entity_label\": \"PERSON\"}},\n"
        "    {{\"head_entity_text\": \"张三\", \"head_entity_label\": \"PERSON\", \"relation_type\": \"WORKS_AT\", \"tail_entity_text\": \"谷歌\", \"tail_entity_label\": \"ORGANIZATION\"}}\n"
        "  ]\n"
        "}}\n\n"
        "文本：\n"
        "\"{text_to_extract}\"\n\n"
        "JSON输出："
    )
    sglang_model_name: str = "Qwen2.5-3B-Instruct_via_SGLang" # 保持不变

# --- 新增：KGExtractionAgent (实现为 Asset) ---
@dg.asset(
    name="kg_extractions",
    description="Extracts entities (and potentially relations) from text chunks for knowledge graph construction.",
    group_name="kg_building",
    io_manager_key="pydantic_json_io_manager" 
)
async def kg_extraction_asset(
    context: dg.AssetExecutionContext,
    text_chunks: List[ChunkOutput], 
    config: KGExtractionConfig,     
    sglang_api: SGLangAPIResource
) -> List[KGTripleSetOutput]: # 确保返回类型注解正确
    
    all_kg_outputs: List[KGTripleSetOutput] = []
    context.log.info(f"Received {len(text_chunks)} text chunks for KG extraction.")

    if not text_chunks:
        context.log.warning("No text chunks received, skipping KG extraction.")
        context.add_output_metadata(metadata={"num_chunks_processed": 0, "total_entities_extracted": 0, "total_relations_extracted": 0})
        return all_kg_outputs

    total_entities_count = 0
    total_relations_count = 0
    for i, chunk in enumerate(text_chunks):
        context.log.info(f"Extracting KG from chunk {i+1}/{len(text_chunks)} (ID: {chunk.chunk_id})")
        
        prompt = config.extraction_prompt_template.format(text_to_extract=chunk.chunk_text)
        
        try:
            structured_response = await sglang_api.generate_structured_output(
                prompt=prompt,
                json_schema=DEFAULT_KG_EXTRACTION_SCHEMA 
            )
            context.log.debug(f"SGLang structured response for chunk {chunk.chunk_id}: {structured_response}")

            entities_data = structured_response.get("entities", [])
            extracted_entities_list: List[ExtractedEntity] = []
            if isinstance(entities_data, list):
                for entity_dict in entities_data:
                    if isinstance(entity_dict, dict) and "text" in entity_dict and "label" in entity_dict:
                        # --- 规范化实体文本和标签 ---
                        normalized_entity_text = normalize_text_for_id(entity_dict["text"])
                        # 标签通常已经是大写或特定枚举值，但如果LLM可能返回不同大小写，也应规范化
                        normalized_entity_label = entity_dict["label"].upper() # 保持标签大写
                        extracted_entities_list.append(ExtractedEntity(
                            text=normalized_entity_text, # <--- 使用规范化后的文本
                            label=normalized_entity_label  # <--- 使用规范化/统一后的标签
                        ))
                    else:
                        context.log.warning(f"Skipping malformed entity data in chunk {chunk.chunk_id}: {entity_dict}")
            else:
                context.log.warning(f"'entities' field in SGLang response for chunk {chunk.chunk_id} is not a list: {entities_data}")
            
            total_entities_count += len(extracted_entities_list)

            relations_data = structured_response.get("relations", [])
            extracted_relations_list: List[ExtractedRelation] = []
            if isinstance(relations_data, list):
                for rel_dict in relations_data:
                    if (isinstance(rel_dict, dict) and
                        all(key in rel_dict for key in ["head_entity_text", "head_entity_label", 
                                                        "relation_type", "tail_entity_text", "tail_entity_label"])):
                        # --- 规范化关系中的实体文本和标签 ---
                        normalized_head_text = normalize_text_for_id(rel_dict["head_entity_text"])
                        normalized_head_label = rel_dict["head_entity_label"].upper()
                        normalized_tail_text = normalize_text_for_id(rel_dict["tail_entity_text"])
                        normalized_tail_label = rel_dict["tail_entity_label"].upper()
                        # 关系类型通常也应该是规范的
                        normalized_relation_type = rel_dict["relation_type"].upper()
                        extracted_relations_list.append(ExtractedRelation(
                            head_entity_text=normalized_head_text,
                            head_entity_label=normalized_head_label,
                            relation_type=normalized_relation_type,
                            tail_entity_text=normalized_tail_text,
                            tail_entity_label=normalized_tail_label
                        ))
                    else:
                        context.log.warning(f"Skipping malformed relation data in chunk {chunk.chunk_id}: {rel_dict}")
            else:
                context.log.warning(f"'relations' field in SGLang response for chunk {chunk.chunk_id} is not a list: {relations_data}")
            
            total_relations_count += len(extracted_relations_list)
            
            kg_output = KGTripleSetOutput(
                chunk_id=chunk.chunk_id,
                extracted_entities=extracted_entities_list,
                extracted_relations=extracted_relations_list,
                extraction_model_name=config.sglang_model_name,
                original_chunk_metadata=chunk.chunk_metadata
            )
            all_kg_outputs.append(kg_output)

        except Exception as e:
            context.log.error(f"Failed KG extraction for chunk {chunk.chunk_id}: {e}", exc_info=True) # 添加 exc_info=True
            all_kg_outputs.append(KGTripleSetOutput(
                chunk_id=chunk.chunk_id,
                extraction_model_name=config.sglang_model_name,
                original_chunk_metadata={"error": str(e), **chunk.chunk_metadata}
            ))
            
    context.add_output_metadata(
        metadata={
            "num_chunks_processed": len(text_chunks),
            "total_entities_extracted": total_entities_count,
            "total_relations_extracted": total_relations_count,
            "status": "Success" if len(all_kg_outputs) == len(text_chunks) else "Partial Success"
        }
    )
    return all_kg_outputs

# KuzuDB Concurrency Key (as suggested in the new strategy text file for write assets)
KUZU_WRITE_CONCURRENCY_KEY = "kuzu_write_access"

@dg.asset(
    name="kuzu_schema_initialized",
    description="Ensures KuzuDB is initialized and schema (tables) are ready, and performs DDL/Checkpoint.",
    group_name="kg_building",
    # deps=[], # 依赖通过参数自动推断，或者如果资源不作为参数传入，则不需要显式 deps
    tags={dg.MAX_RUNTIME_SECONDS_TAG: "300"} # KUZU_WRITE_CONCURRENCY_KEY 在 in_process 作业中意义不大
)
def kuzu_schema_initialized_asset(
    context: dg.AssetExecutionContext,
    kuzu_readwrite_db: KuzuDBReadWriteResource # 注入资源
) -> dg.Output[str]:
    db_path_used = kuzu_readwrite_db._resolved_db_path # 可以从资源获取路径信息
    context.log.info(f"Using KuzuDB at: {db_path_used}.")
    
    conn = kuzu_readwrite_db.get_connection() # <--- 从资源获取连接

    try:
        context.log.info("Executing DDL statements for schema creation...")
        schema_ddl_queries = [
            "CREATE NODE TABLE IF NOT EXISTS ExtractedEntity (id_prop STRING, text STRING, label STRING, PRIMARY KEY (id_prop))",
            "CREATE REL TABLE IF NOT EXISTS WorksAt (FROM ExtractedEntity TO ExtractedEntity)",
            "CREATE REL TABLE IF NOT EXISTS AssignedTo (FROM ExtractedEntity TO ExtractedEntity)"
        ]
        for ddl_query in schema_ddl_queries:
            context.log.debug(f"Executing DDL: {ddl_query}")
            conn.execute(ddl_query)
        context.log.info("DDL statements execution completed.")
        
        context.log.info("Executing manual CHECKPOINT.")
        conn.execute("CHECKPOINT;")
        context.log.info("Manual CHECKPOINT completed.")

        context.log.info("Verifying table existence after schema initialization...")
        node_table_names = conn._get_node_table_names()
        rel_tables_info = conn._get_rel_table_names()
        rel_table_names = [info['name'] for info in rel_tables_info]
        all_defined_tables = node_table_names + rel_table_names
        context.log.info(f"All defined tables in KuzuDB: {all_defined_tables}")
        
        required_tables = ["ExtractedEntity", "WorksAt", "AssignedTo"]
        all_found = True
        missing_tables = []
        for tbl in required_tables:
            if tbl not in all_defined_tables:
                all_found = False
                missing_tables.append(tbl)
        
        if all_found:
            context.log.info("Verification SUCCESS: All required tables found in KuzuDB.")
        else:
            raise dg.Failure(f"Schema verification failed. Missing tables: {', '.join(missing_tables)}")

    except Exception as e:
        context.log.error(f"Failed during KuzuDB schema DDL/CHECKPOINT/Verification: {e}", exc_info=True)
        raise dg.Failure(f"KuzuDB schema initialization/verification failed: {e}")
        
    return dg.Output("KuzuDB schema ensured, checkpointed, and verified.", metadata={"db_path": db_path_used, "defined_tables": all_defined_tables})

@dg.asset(
    name="kuzu_entity_nodes",
    description="Stores extracted entities as nodes in KuzuDB knowledge graph.",
    group_name="kg_building",
    # deps=[kuzu_schema_initialized_asset], # 通过参数推断
    tags={dg.MAX_RUNTIME_SECONDS_TAG: "600"}
)
def kuzu_entity_nodes_asset(
    context: dg.AssetExecutionContext,
    kg_extractions: List[KGTripleSetOutput],
    kuzu_schema_initialized: str, # 依赖上游资产的输出
    kuzu_readwrite_db: KuzuDBReadWriteResource # 注入资源
) -> None:
    context.log.info(f"Received {len(kg_extractions)} KG extraction sets to store entities in KuzuDB.")
    context.log.info(f"Upstream kuzu_schema_initialized_asset reported: {kuzu_schema_initialized}")

    if not kg_extractions:
        context.log.warning("No KG extractions received, nothing to store in KuzuDB.")
        context.add_output_metadata(metadata={"nodes_created_or_merged": 0, "status": "No data"})
        return
        
    dml_statements: List[tuple[str, Dict[str, Any]]] = []
    total_nodes_processed = 0

    for kg_output_set in kg_extractions:
        for entity in kg_output_set.extracted_entities:
            # 假设 entity.text 和 entity.label 已经是规范化过的
            text_for_id = entity.text 
            label_for_id = entity.label # 假设已经是大写
            id_prop_input_string = text_for_id + label_for_id # <--- 用于生成哈希的字符串
            entity_id_prop = hashlib.md5(id_prop_input_string.encode('utf-8')).hexdigest()  
            context.log.info(f"NODE CREATION: text='{entity.text}', label='{entity.label}', "
                             f"id_prop_input='{id_prop_input_string}', generated_id_prop='{entity_id_prop}'")
            query = """
                MERGE (e:ExtractedEntity {id_prop: $id_prop})
                ON CREATE SET e.text = $text, e.label = $label 
            """
            params = {
                "id_prop": entity_id_prop,
                "text": entity.text, 
                "label": entity.label
            }
            dml_statements.append((query, params))

    if not dml_statements:
        context.log.info("No valid entities found to store in KuzuDB after processing extractions.")
        context.add_output_metadata(metadata={"nodes_created_or_merged": 0, "status": "No entities to store"})
        return
    
    conn = kuzu_readwrite_db.get_connection() # <--- 从资源获取连接
    try:
        context.log.info(f"Executing {len(dml_statements)} MERGE operations for entities in KuzuDB...")
        executed_count = 0
        for query, params in dml_statements:
            context.log.debug(f"Executing DML: {query} with params: {params}")
            conn.execute(query, parameters=params)
            executed_count +=1
        context.log.info(f"Executed {executed_count} DML statements successfully for entities.")
        
        context.add_output_metadata(
            metadata={
                "nodes_created_or_merged": executed_count, # 使用实际执行数量
                "status": "Success"
            }
        )
    except Exception as e:
        context.log.error(f"Failed to store entities in KuzuDB: {e}", exc_info=True)
        raise dg.Failure(description=f"Failed to store entities in KuzuDB: {str(e)}")

@dg.asset(
    name="kuzu_entity_relations",
    description="Creates relationships in KuzuDB based on extracted KG data.",
    group_name="kg_building",
    deps=[kuzu_entity_nodes_asset.key], # 显式声明对 kuzu_entity_nodes_asset 的依赖
    tags={dg.MAX_RUNTIME_SECONDS_TAG: "600"}
)

def kuzu_entity_relations_asset(
    context: dg.AssetExecutionContext,
    kg_extractions: List[KGTripleSetOutput],
    kuzu_schema_initialized: str, # 依赖 kuzu_schema_initialized_asset 的输出
    kuzu_readwrite_db: KuzuDBReadWriteResource
) -> None:
    context.log.info(f"Received {len(kg_extractions)} KG extraction sets to create relations in KuzuDB.")
    context.log.info(f"Upstream kuzu_schema_initialized_asset reported: {kuzu_schema_initialized}")

    conn = kuzu_readwrite_db.get_connection()

    # --- Schema验证日志 ---
    try:
        context.log.info("Verifying table existence at the START of kuzu_entity_relations_asset...")
        node_tables_at_start = conn._get_node_table_names()
        rel_tables_info_at_start = conn._get_rel_table_names()
        rel_tables_at_start = [info['name'] for info in rel_tables_info_at_start]
        all_tables_at_start_actual_case = node_tables_at_start + rel_tables_at_start
        
        # 我们期望的表名（在DDL中定义的大小写）
        expected_relation_tables_in_schema = ["AssignedTo", "WorksAt"] 

        context.log.info(f"Node tables at start of relations asset: {node_tables_at_start}")
        context.log.info(f"Rel tables at start of relations asset (actual case): {rel_tables_at_start}")
        context.log.info(f"All tables at start of relations asset (actual case): {all_tables_at_start_actual_case}")

        for expected_table in expected_relation_tables_in_schema:
            # 检查时，我们将从数据库获取的表名与期望的表名进行比较
            # KuzuDB 返回的表名是区分大小写的，与 DDL 一致
            if expected_table not in all_tables_at_start_actual_case:
                context.log.error(f"CRITICAL: Expected table '{expected_table}' NOT FOUND at the very start of kuzu_entity_relations_asset on the shared connection!")
            else:
                context.log.info(f"Expected table '{expected_table}' IS PRESENT at the start of kuzu_entity_relations_asset.")
    except Exception as e_verify_start:
        context.log.error(f"Error verifying tables at start of kuzu_entity_relations_asset: {e_verify_start}")
    # --- Schema验证日志结束 ---

    if not kg_extractions:
        context.log.warning("No KG extractions received, nothing to store for relations.")
        context.add_output_metadata(metadata={"relations_created_or_merged": 0, "status": "No data"})
        return

    relations_to_create_params: List[Dict[str, Any]] = []
    relation_type_for_cypher_map: List[str] = [] # 用于存储 Cypher 查询中实际使用的关系表名

    for kg_output_set in kg_extractions:
        for rel_idx, rel in enumerate(kg_output_set.extracted_relations):
            context.log.info(f"Processing relation {rel_idx+1} from chunk {kg_output_set.chunk_id}: "
                             f"{rel.head_entity_text} -[{rel.relation_type}]-> {rel.tail_entity_text}")

            head_text_for_id = rel.head_entity_text
            head_label_for_id = rel.head_entity_label 
            tail_text_for_id = rel.tail_entity_text
            tail_label_for_id = rel.tail_entity_label

            head_id_prop_input_string = head_text_for_id + head_label_for_id
            head_entity_id_prop = hashlib.md5(head_id_prop_input_string.encode('utf-8')).hexdigest()
            
            tail_id_prop_input_string = tail_text_for_id + tail_label_for_id
            tail_entity_id_prop = hashlib.md5(tail_id_prop_input_string.encode('utf-8')).hexdigest()
            
            context.log.info(f"  Attempting to match Head: text='{rel.head_entity_text}', label='{rel.head_entity_label}', "
                             f"id_prop_input='{head_id_prop_input_string}', generated_head_id_prop='{head_entity_id_prop}'")
            context.log.info(f"  Attempting to match Tail: text='{rel.tail_entity_text}', label='{rel.tail_entity_label}', "
                             f"id_prop_input='{tail_id_prop_input_string}', generated_tail_id_prop='{tail_entity_id_prop}'")

            head_found = False
            tail_found = False
            try:
                head_check_result = conn.execute("MATCH (n:ExtractedEntity {id_prop: $id}) RETURN n.text, n.label", {"id": head_entity_id_prop})
                if head_check_result.has_next():
                    head_data = head_check_result.get_next()
                    context.log.info(f"    VERIFIED Head Node: id_prop='{head_entity_id_prop}', text='{head_data[0]}', label='{head_data[1]}'")
                    head_found = True
                else:
                    context.log.error(f"    VERIFICATION FAILED: Head Node with id_prop='{head_entity_id_prop}' (from text='{rel.head_entity_text}') NOT FOUND.")
                head_check_result.close()

                tail_check_result = conn.execute("MATCH (n:ExtractedEntity {id_prop: $id}) RETURN n.text, n.label", {"id": tail_entity_id_prop})
                if tail_check_result.has_next():
                    tail_data = tail_check_result.get_next()
                    context.log.info(f"    VERIFIED Tail Node: id_prop='{tail_entity_id_prop}', text='{tail_data[0]}', label='{tail_data[1]}'")
                    tail_found = True
                else:
                    context.log.error(f"    VERIFICATION FAILED: Tail Node with id_prop='{tail_entity_id_prop}' (from text='{rel.tail_entity_text}') NOT FOUND.")
                tail_check_result.close()
            except Exception as e_verify:
                context.log.error(f"    Error during node verification for relation: {e_verify}")

            if head_found and tail_found:
                # rel.relation_type 应该是从 kg_extraction_asset 传来的全大写形式，例如 "ASSIGNED_TO"
                relation_type_from_extraction = rel.relation_type 
                
                # 将其映射到 DDL 中定义的实际表名（首字母大写）
                actual_cypher_table_name = ""
                if relation_type_from_extraction == "ASSIGNED_TO":
                    actual_cypher_table_name = "AssignedTo"
                elif relation_type_from_extraction == "WORKS_AT":
                    actual_cypher_table_name = "WorksAt"
                
                if actual_cypher_table_name: # 如果是我们支持的关系类型
                    relations_to_create_params.append({
                        "head_id_prop": head_entity_id_prop,
                        "tail_id_prop": tail_entity_id_prop,
                    })
                    relation_type_for_cypher_map.append(actual_cypher_table_name)
                else:
                    context.log.warning(f"  Skipping relation due to unmapped original type: '{relation_type_from_extraction}'")
            else:
                context.log.warning(f"  Skipping relation creation for '{rel.head_entity_text} -[{rel.relation_type}]-> {rel.tail_entity_text}' because one or both nodes were not found with generated id_props.")
            
    if not relations_to_create_params:
        context.log.info("No valid relations to create after node verification and type mapping.")
        context.add_output_metadata(metadata={"relations_created_or_merged": 0, "status": "No valid relations after node/type check"})
        return
            
    executed_count = 0
    try:
        context.log.info(f"Attempting to execute {len(relations_to_create_params)} CREATE operations for relations in KuzuDB...")
        for i, params in enumerate(relations_to_create_params):
            rel_table_for_cypher = relation_type_for_cypher_map[i]
            query = f"""
                MATCH (h:ExtractedEntity {{id_prop: $head_id_prop}}), (t:ExtractedEntity {{id_prop: $tail_id_prop}})
                CREATE (h)-[r:{rel_table_for_cypher}]->(t)
            """
            context.log.debug(f"Executing DML for relation type '{rel_table_for_cypher}': {query} with params: {params}")
            conn.execute(query, parameters=params)
            executed_count += 1
        context.log.info(f"Successfully submitted {executed_count} DML statements for relations.")
        
        context.log.info("Final verification of relation counts after DML execution...")
        final_rel_counts = {}
        unique_relation_types_attempted = set(relation_type_for_cypher_map)
        for rel_type_to_check_in_cypher in unique_relation_types_attempted:
            try:
                count_query_result = conn.execute(f"MATCH ()-[r:{rel_type_to_check_in_cypher}]->() RETURN count(r) AS count")
                count_df = pd.DataFrame(count_query_result.get_as_df())
                actual_count_in_db_raw = count_df['count'].iloc[0] if not count_df.empty else 0
                actual_count_in_db = int(actual_count_in_db_raw) # <--- 修改这里：转换为Python int
                final_rel_counts[rel_type_to_check_in_cypher] = actual_count_in_db
                context.log.info(f"  Relation table '{rel_type_to_check_in_cypher}' final count in DB: {actual_count_in_db}")
            except Exception as e_count_final:
                context.log.error(f"  Error during final count for relation table '{rel_type_to_check_in_cypher}': {e_count_final}")
        context.add_output_metadata(
            metadata={
                "relations_attempted_creation": len(relations_to_create_params),
                "relations_dml_submitted_and_executed_by_kuzu": executed_count, # 更准确的描述
                "final_relation_counts_in_db": final_rel_counts,
                "status": "Success" if executed_count == len(relations_to_create_params) and executed_count > 0 else "Partial Success or Issues" if executed_count > 0 else "No relations created"
            }
        )
    except Exception as e:
        context.log.error(f"Failed during DML execution for relations: {e}", exc_info=True)
        raise dg.Failure(description=f"Failed to create relations in KuzuDB: {str(e)}")
    
# 确保 all_processing_assets 列表正确
all_processing_assets = [
    clean_chunk_text_asset,
    generate_embeddings_asset,
    vector_storage_asset,
    keyword_index_asset,
    kg_extraction_asset,
    kuzu_schema_initialized_asset,
    kuzu_entity_nodes_asset,
    kuzu_entity_relations_asset
]