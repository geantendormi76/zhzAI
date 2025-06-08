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

class TextChunkerConfig(dg.Config):
    chunk_size: int = 500
    chunk_overlap: int = 50
    # separators: Optional[List[str]] = None # 可选的自定义分隔符

@dg.asset(
    name="text_chunks",
    description="Cleans and chunks parsed documents into smaller text segments.",
    group_name="processing", # 属于处理组
    deps=["parsed_documents"] # <--- 删除或注释掉这一行
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
    sglang_api: LocalLLMAPIResource 
) -> List[KGTripleSetOutput]:
    
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
             # --- 核心修改：在这一步就进行规范化 ---
            entities_data = structured_response.get("entities", [])
            extracted_entities_list: List[ExtractedEntity] = []
            if isinstance(entities_data, list):
                for entity_dict in entities_data:
                    if isinstance(entity_dict, dict) and "text" in entity_dict and "label" in entity_dict:
                        # 对 LLM 返回的文本和标签立即进行规范化
                        normalized_text = normalize_text_for_id(entity_dict["text"])
                        normalized_label = entity_dict["label"].upper()
                        extracted_entities_list.append(ExtractedEntity(
                            text=normalized_text, 
                            label=normalized_label
                        ))
            
            relations_data = structured_response.get("relations", [])
            extracted_relations_list: List[ExtractedRelation] = []
            if isinstance(relations_data, list):
                for rel_dict in relations_data:
                    if (isinstance(rel_dict, dict) and
                        all(key in rel_dict for key in ["head_entity_text", "head_entity_label", 
                                                        "relation_type", "tail_entity_text", "tail_entity_label"])):
                        # 对关系中的所有文本和标签也进行规范化
                        extracted_relations_list.append(ExtractedRelation(
                            head_entity_text=normalize_text_for_id(rel_dict["head_entity_text"]),
                            head_entity_label=rel_dict["head_entity_label"].upper(),
                            relation_type=rel_dict["relation_type"].upper(),
                            tail_entity_text=normalize_text_for_id(rel_dict["tail_entity_text"]),
                            tail_entity_label=rel_dict["tail_entity_label"].upper()
                        ))
            # --- 修改结束 --- 
            total_entities_count += len(extracted_entities_list)
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

KUZU_WRITE_CONCURRENCY_KEY = "kuzu_write_access"


@dg.asset(
    name="kuzu_graph_construction",
    description="Creates schema, loads data, and creates vector index in KuzuDB.",
    group_name="kg_building",
    op_tags={"dagster/concurrency_key": "kuzu_write_pool"},
    deps=[kg_extraction_asset],
    tags={dg.MAX_RUNTIME_SECONDS_TAG: "900"}
)
def kuzu_graph_construction_asset(
    context: dg.AssetExecutionContext,
    kg_extractions: List[KGTripleSetOutput],
    kuzu_readwrite_db: KuzuDBReadWriteResource,
    embedder: SentenceTransformerResource
):
    context.log.info("--- Starting Unified KuzuDB Graph Construction Asset (UNWIND/MERGE Strategy) ---")
    if not kg_extractions:
        context.log.warning("No KG extractions received. Skipping.")
        return

    with kuzu_readwrite_db.get_connection() as conn:
        try:
            # 步骤 1: 创建 Schema (逻辑不变)
            context.log.info("Step 1: Creating Schema...")
            EMBEDDING_DIM = embedder.get_embedding_dimension()
            conn.execute(
                f"CREATE NODE TABLE IF NOT EXISTS ExtractedEntity (id_prop STRING, text STRING, label STRING, embedding FLOAT[{EMBEDDING_DIM}], PRIMARY KEY (id_prop))"
            )
            conn.execute("CREATE REL TABLE IF NOT EXISTS WorksAt (FROM ExtractedEntity TO ExtractedEntity)")
            conn.execute("CREATE REL TABLE IF NOT EXISTS AssignedTo (FROM ExtractedEntity TO ExtractedEntity)")
            context.log.info("Schema DDL commands executed.")

            # 步骤 2: 加载节点数据 (逻辑不变)
            context.log.info("Step 2: Preparing and loading node data...")
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
            if node_data:
                nodes_df = pd.DataFrame(node_data)
                context.log.info(f"Copying {len(nodes_df)} unique nodes into ExtractedEntity table...")
                conn.execute("COPY ExtractedEntity FROM nodes_df;")
                context.log.info(f"Successfully copied nodes.")
            else:
                context.log.warning("No unique nodes found in extractions to load.")

            # 步骤 3: 使用 UNWIND...MERGE... 加载关系数据
            context.log.info("Step 3: Preparing and loading relation data using UNWIND...MERGE...")
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

            if relation_data:
                all_rels_df = pd.DataFrame(relation_data)
                for rel_type in all_rels_df['relation_type'].unique():
                    rels_df_for_merge = all_rels_df[all_rels_df['relation_type'] == rel_type]
                    batch_params = rels_df_for_merge.to_dict(orient='records')
                    
                    if not batch_params:
                        continue
                        
                    context.log.info(f"Attempting to MERGE {len(batch_params)} relations of type '{rel_type}'...")
                    
                    merge_query = f"""
                        UNWIND $batch AS rel_item
                        MATCH (from_node:ExtractedEntity {{id_prop: rel_item.from_id}})
                        MATCH (to_node:ExtractedEntity {{id_prop: rel_item.to_id}})
                        MERGE (from_node)-[:`{rel_type}`]->(to_node)
                    """
                    context.log.debug(f"Executing relation merge query: {merge_query.strip()}")
                    
                    # --- [核心修复] ---
                    # 根据 TypeError，KuzuDB的execute方法期望参数
                    # 以一个字典的形式作为第二个位置参数传入，而不是作为关键字参数。
                    conn.execute(merge_query, {"batch": batch_params})
                    # --- [修复结束] ---

                    context.log.info(f"Successfully merged relations of type '{rel_type}'.")
            else:
                context.log.warning("No relation data found in extractions to load.")

            # 步骤 4: CHECKPOINT (逻辑不变)
            context.log.info("Step 4: Executing CHECKPOINT to persist all changes...")
            conn.execute("CHECKPOINT;")
            context.log.info("CHECKPOINT successful.")

            # 步骤 5: 创建向量索引 (逻辑不变)
            context.log.info("Step 5: Creating Vector Index (if not exists)...")
            try:
                conn.execute("LOAD VECTOR;")
                context.log.info("VECTOR extension loaded for index creation.")
            except Exception as e_load_vec:
                context.log.warning(f"Could not explicitly LOAD VECTOR (might be already loaded): {e_load_vec}")
            conn.execute("CALL CREATE_VECTOR_INDEX('ExtractedEntity', 'entity_embedding_idx', 'embedding', metric := 'cosine')")
            context.log.info("Vector index creation command successfully executed.")

        except Exception as e:
            raise dg.Failure(f"An error occurred within the KuzuDB asset: {e}") from e

    context.log.info("--- KuzuDB Graph Construction Asset Finished Successfully ---")

# (这个列表定义保持不变)
all_processing_assets = [
    clean_chunk_text_asset,
    generate_embeddings_asset,
    vector_storage_asset,
    keyword_index_asset,
    kg_extraction_asset,
    kuzu_graph_construction_asset
]