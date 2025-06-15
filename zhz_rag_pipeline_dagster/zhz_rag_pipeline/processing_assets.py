#  文件: zhz_rag_pipeline_dagster/zhz_rag_pipeline/processing_assets.py

import json
import asyncio
import re
import dagster as dg
from typing import List, Dict, Any, Optional, Union
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
    ExtractedRelation,
    # --- 添加导入我们需要的元素类型 ---
    TitleElement,
    NarrativeTextElement,
    ListItemElement,
    TableElement,
    CodeBlockElement,
    ImageElement,
    PageBreakElement,
    HeaderElement,
    FooterElement,
    DocumentElementMetadata
    # --- 结束添加 ---
)
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.resources import (
    GGUFEmbeddingResource,
    ChromaDBResource,
    DuckDBResource,
    LocalLLMAPIResource,
    SystemResource  # <--- 添加这一行以导入 SystemResource
)
import jieba
import bm25s
import pickle
import numpy as np
import os
from zhz_rag.utils.common_utils import normalize_text_for_id

_PYDANTIC_AVAILABLE = False
try:
    from .pydantic_models_dagster import ( # 使用相对导入
        ChunkOutput,
        ParsedDocumentOutput,
        EmbeddingOutput,
        KGTripleSetOutput,
        ExtractedEntity,
        ExtractedRelation,
        TitleElement,
        NarrativeTextElement,
        ListItemElement,
        TableElement,
        CodeBlockElement,
        ImageElement,
        PageBreakElement,
        HeaderElement,
        FooterElement,
        DocumentElementMetadata # <--- 确保这里导入了 DocumentElementMetadata
    )
    _PYDANTIC_AVAILABLE = True
    # 如果 Pydantic 可用，我们也可以直接从模型中获取 DocumentElementType
    # from .pydantic_models_dagster import DocumentElementType # 如果需要更精确的类型提示
except ImportError:
    # 定义占位符
    class BaseModel: pass
    class ChunkOutput(BaseModel): pass
    class ParsedDocumentOutput(BaseModel): pass
    class EmbeddingOutput(BaseModel): pass
    class KGTripleSetOutput(BaseModel): pass
    class ExtractedEntity(BaseModel): pass
    class ExtractedRelation(BaseModel): pass
    class TitleElement(BaseModel): pass
    class NarrativeTextElement(BaseModel): pass
    class ListItemElement(BaseModel): pass
    class TableElement(BaseModel): pass
    class CodeBlockElement(BaseModel): pass
    class ImageElement(BaseModel): pass
    class PageBreakElement(BaseModel): pass
    class HeaderElement(BaseModel): pass
    class FooterElement(BaseModel): pass
    class DocumentElementMetadata(BaseModel): pass # <--- 定义占位符
    DocumentElementType = Any # type: ignore
# --- 结束 Pydantic 模型导入 ---

import logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG) # <--- 确保是 DEBUG
    logger.info(f"Logger for {__name__} (processing_assets) configured with DEBUG level.")


class TextChunkerConfig(dg.Config):
    chunk_size: int = 1000 
    chunk_overlap: int = 100
    max_element_text_length_before_split: int = 1200 # 一个1200字符的段落如果语义连贯，可以考虑不切分。
    target_sentence_split_chunk_size: int = 600    # 略微增大子块的目标大小，使其包含更多上下文。
    sentence_split_chunk_overlap_sentences: int = 2  # 增加到2句重叠，以期在子块之间提供更好的语义连接。
    # --- 合并策略参数 ---
    min_chunk_length_to_avoid_merge: int = 250
    max_merged_chunk_size: int = 750


def split_text_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using a regex-based approach.
    Handles common sentence terminators and aims to preserve meaningful units.
    """
    if not text:
        return []
    # 改进的句子分割正则表达式，考虑了中英文句号、问号、感叹号
    # 并尝试处理省略号和一些特殊情况。
    # (?<=[。？！\.!\?]) 会匹配这些标点符号后面的位置 (lookbehind)
    # \s* 匹配标点后的任意空格
    # (?!$) 确保不是字符串末尾 (避免在末尾标点后产生空句子)
    # 对于中文，句号、问号、感叹号通常直接结束句子。
    # 对于英文，. ! ? 后面通常有空格或换行。
    
    # 一个更简单的版本，直接按标点分割，然后清理
    sentences = re.split(r'([。？！\.!\?])', text)
    result = []
    current_sentence = ""
    for i in range(0, len(sentences), 2):
        part = sentences[i]
        terminator = sentences[i+1] if i+1 < len(sentences) else ""
        current_sentence = part + terminator
        if current_sentence.strip():
            result.append(current_sentence.strip())
    
    # 如果上面的分割不理想，可以尝试更复杂的，但这个简单版本通常够用
    # 例如：
    # sentences = re.split(r'(?<=[。？！\.!\?])\s*', text)
    # sentences = [s.strip() for s in sentences if s.strip()]
    return result if result else [text] # 如果无法分割，返回原文本作为一个句子



def split_markdown_table_by_rows(
    markdown_table_text: str,
    target_chunk_size: int,
    max_chunk_size: int,
    context: Optional[dg.AssetExecutionContext] = None
) -> List[Dict[str, Any]]: # <--- 修改返回类型
    """
    Splits a Markdown table string by its data rows.
    Returns a list of dictionaries, each containing 'text', 'start_row_index', and 'end_row_index'.
    Indices are 0-based relative to the start of data_rows.
    """
    sub_chunks_data: List[Dict[str, Any]] = [] # <--- 修改变量名和类型
    lines = markdown_table_text.strip().split('\n')
    
    if len(lines) < 2: # 至少需要表头和分隔行
        if context: context.log.warning(f"Markdown table has less than 2 lines ({len(lines)}). Cannot process for row splitting. Returning as single chunk.")
        # 对于无法按预期处理的表格，返回原始文本及无效的行索引
        return [{"text": markdown_table_text, "start_row_index": -1, "end_row_index": -1}]


    header_row = lines[0]
    separator_row = lines[1]
    data_rows = lines[2:]

    if not data_rows:
        if context: context.log.warning("Markdown table has no data rows. Returning header and separator as single chunk.")
        return [{"text": f"{header_row}\n{separator_row}", "start_row_index": -1, "end_row_index": -1}]

    if not (separator_row.strip().startswith('|') and separator_row.strip().endswith('|') and '-' in separator_row):
        if context: context.log.warning(f"Markdown table separator row looks non-standard: '{separator_row}'. Row splitting might be unreliable.")
        # 即使分隔符不标准，也尝试继续，但标记行索引为不可靠
        # 或者直接返回整个表作为一个块，标记行索引无效
        # 为简单起见，我们先尝试，但如果这个情况频繁且导致问题，应返回整个表
        # return [{"text": markdown_table_text, "start_row_index": -2, "end_row_index": -2}] # -2 表示分隔符问题

    current_sub_chunk_lines = [header_row, separator_row]
    current_sub_chunk_char_count = len(header_row) + 1 + len(separator_row) + 1
    current_sub_chunk_start_row_idx = 0 # 记录当前子块数据行的起始索引

    for i, data_row in enumerate(data_rows):
        row_len = len(data_row)
        if not data_row.strip():
            if i == current_sub_chunk_start_row_idx and len(current_sub_chunk_lines) == 2 : # 如果是新块的第一个数据行就是空的，则递增起始行索引
                current_sub_chunk_start_row_idx +=1
            continue # 跳过空数据行

        # 条件：1. 当前块只有表头；2. 加入新行不超过目标大小；3. 当前块只有表头且加入第一行数据即使超目标大小但不超最大大小
        should_add_to_current = (
            len(current_sub_chunk_lines) == 2 or
            (current_sub_chunk_char_count + row_len + 1 <= target_chunk_size) or
            (len(current_sub_chunk_lines) == 2 and (current_sub_chunk_char_count + row_len + 1 > target_chunk_size) and (current_sub_chunk_char_count + row_len + 1 <= max_chunk_size))
        )

        if should_add_to_current and (current_sub_chunk_char_count + row_len + 1 <= max_chunk_size):
            current_sub_chunk_lines.append(data_row)
            current_sub_chunk_char_count += row_len + 1
        else:
            # 完成当前子块
            if len(current_sub_chunk_lines) > 2: # 确保至少有一行数据
                sub_chunks_data.append({
                    "text": "\n".join(current_sub_chunk_lines),
                    "start_row_index": current_sub_chunk_start_row_idx,
                    "end_row_index": i - 1 # 上一行是这个块的结束行
                })
                if context: context.log.debug(f"  Table sub-chunk created: rows {current_sub_chunk_start_row_idx}-{i-1}")

            # 开始新的子块
            current_sub_chunk_lines = [header_row, separator_row, data_row] # 当前行是新块的第一行数据
            current_sub_chunk_char_count = len(header_row) + 1 + len(separator_row) + 1 + len(data_row) + 1
            current_sub_chunk_start_row_idx = i
            
            # 如果当前行（作为新块的第一行）自己就超长了（非常罕见，除非max_chunk_size很小）
            if current_sub_chunk_char_count > max_chunk_size:
                if context: context.log.warning(f"  Single data row (original index {i}) with header exceeds max_chunk_size. Creating chunk for this row anyway.")
                # 即使超长，也为这一行创建一个块
                sub_chunks_data.append({
                    "text": "\n".join(current_sub_chunk_lines), # 包含表头、分隔符和这一超长行
                    "start_row_index": i,
                    "end_row_index": i 
                })
                if context: context.log.debug(f"  Table sub-chunk created for single very long row: {i}-{i}")
                # 重置，准备下一个数据行（如果有的话）
                current_sub_chunk_lines = [header_row, separator_row]
                current_sub_chunk_char_count = len(header_row) + 1 + len(separator_row) + 1
                current_sub_chunk_start_row_idx = i + 1


    # 添加最后一个正在构建的子块
    if len(current_sub_chunk_lines) > 2:
        sub_chunks_data.append({
            "text": "\n".join(current_sub_chunk_lines),
            "start_row_index": current_sub_chunk_start_row_idx,
            "end_row_index": len(data_rows) - 1 # 最后的数据行
        })
        if context: context.log.debug(f"  Table sub-chunk created (last): rows {current_sub_chunk_start_row_idx}-{len(data_rows)-1}")

    if not sub_chunks_data and markdown_table_text:
        if context: context.log.warning("Markdown table splitting by row resulted in no sub-chunks. Returning original table as one.")
        return [{"text": markdown_table_text, "start_row_index": 0, "end_row_index": len(data_rows) -1 if data_rows else -1}]
        
    return sub_chunks_data



def split_code_block_by_blank_lines(
    code_text: str,
    target_chunk_size: int, # 复用配置，但对于代码块，这个更像是一个上限指导
    max_chunk_size: int,    # 作为硬上限
    context: Optional[dg.AssetExecutionContext] = None
) -> List[str]:
    """
    Splits a code block string by blank lines (one or more empty lines).
    Tries to keep resulting chunks from exceeding max_chunk_size.
    """
    if not code_text.strip():
        return []

    # 使用正则表达式匹配一个或多个连续的空行作为分隔符
    # \n\s*\n 匹配一个换行符，后跟零或多个空白字符，再跟一个换行符
    potential_splits = re.split(r'(\n\s*\n)', code_text) # 保留分隔符以便后续处理
    
    sub_chunks = []
    current_chunk_lines = []
    current_chunk_char_count = 0

    # 第一个块总是从头开始
    if potential_splits:
        first_part = potential_splits.pop(0).strip()
        if first_part:
            current_chunk_lines.append(first_part)
            current_chunk_char_count += len(first_part)

    while potential_splits:
        delimiter = potential_splits.pop(0) # 这是分隔符 \n\s*\n
        if not potential_splits: # 没有更多内容了
            if delimiter.strip(): # 如果分隔符本身不是纯空白，也算内容
                 current_chunk_lines.append(delimiter.rstrip()) # 保留末尾的换行
                 current_chunk_char_count += len(delimiter.rstrip())
            break 
        
        next_part = potential_splits.pop(0).strip()
        if not next_part: # 如果下一个部分是空的，只处理分隔符
            if delimiter.strip():
                current_chunk_lines.append(delimiter.rstrip())
                current_chunk_char_count += len(delimiter.rstrip())
            continue

        # 检查加入 delimiter 和 next_part 是否会超长
        # 对于代码，我们通常希望在逻辑断点（空行）处分割，即使块较小
        # 但如果单个由空行分隔的块本身就超过 max_chunk_size，则需要进一步处理（目前简单截断或接受）
        
        # 简化逻辑：如果当前块非空，并且加入下一个部分（包括分隔的空行）会超过目标大小，
        # 或者严格超过最大大小，则结束当前块。
        # 这里的分隔符（空行）本身也应该被视为块的一部分，或者作为块的自然结束。

        # 更简单的策略：每个由 re.split 分割出来的非空部分（即代码段）自成一块
        # 如果代码段本身过长，则接受它，或者未来再细分
        if current_chunk_lines: # 如果当前块有内容
            # 检查如果加上 next_part 是否会超长（这里可以简化，因为空行分割通常意味着逻辑单元）
            # 我们先假设每个由空行分割的块都是一个独立的单元
            sub_chunks.append("\n".join(current_chunk_lines))
            if context: context.log.debug(f"  Code sub-chunk created (blank line split), len: {current_chunk_char_count}")
            current_chunk_lines = []
            current_chunk_char_count = 0
        
        if next_part: # 开始新的块
            current_chunk_lines.append(next_part)
            current_chunk_char_count += len(next_part)

    # 添加最后一个正在构建的子块
    if current_chunk_lines:
        sub_chunks.append("\n".join(current_chunk_lines))
        if context: context.log.debug(f"  Code sub-chunk created (blank line split, last), len: {current_chunk_char_count}")

    if not sub_chunks and code_text: # 如果完全没分割出任何东西（例如代码没有空行）
        if context: context.log.warning("Code block splitting by blank lines resulted in no sub-chunks. Returning original code block.")
        # 对于这种情况，我们可能需要一个字符分割器作为最终回退
        # 但为了简单起见，我们先返回原始代码块
        # 如果原始代码块 > max_chunk_size，它仍然会是一个大块
        if len(code_text) > max_chunk_size:
            if context: context.log.warning(f"  Original code block (len: {len(code_text)}) exceeds max_chunk_size ({max_chunk_size}) and was not split by blank lines. Consider character splitting as fallback.")
            # 这里可以插入 RecursiveCharacterTextSplitter 逻辑
            # from langchain_text_splitters import RecursiveCharacterTextSplitter
            # char_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=0, separators=["\n", " ", ""])
            # sub_chunks = char_splitter.split_text(code_text)
            # if context: context.log.info(f"    Fallback: Code block character split into {len(sub_chunks)} parts.")
            # return sub_chunks
        return [code_text] # 暂时返回原块

    # 过滤掉完全是空字符串的块 (re.split 可能产生)
    final_sub_chunks = [chunk for chunk in sub_chunks if chunk.strip()]
    return final_sub_chunks if final_sub_chunks else [code_text]



@dg.asset(
    name="text_chunks",
    description="Cleans/chunks documents. Splits long elements, merges short ones, enriches with contextual metadata.",
    group_name="processing",
    deps=["parsed_documents"]
)
def clean_chunk_text_asset(
    context: dg.AssetExecutionContext,
    config: TextChunkerConfig,
    parsed_documents: List[ParsedDocumentOutput]
) -> List[ChunkOutput]:
    initial_chunks: List[ChunkOutput] = []
    context.log.info(f"Received {len(parsed_documents)} parsed documents for initial chunking.")
    context.log.info(f"Initial chunking config: MaxElemLen={config.max_element_text_length_before_split}, "
                     f"TargetSubChunkSize={config.target_sentence_split_chunk_size}, "
                     f"SentenceOverlap={config.sentence_split_chunk_overlap_sentences}")

    for doc_idx, parsed_doc in enumerate(parsed_documents):
        doc_id_from_meta = parsed_doc.original_metadata.get("filename", f"doc_{doc_idx}_{str(uuid.uuid4())}")
        context.log.info(f"Processing document for initial chunks: {doc_id_from_meta}")

        current_title_hierarchy: Dict[int, str] = {}

        if not parsed_doc.elements:
            context.log.warning(f"Document {doc_id_from_meta} has no structured elements.")
            if parsed_doc.parsed_text and parsed_doc.parsed_text.strip():
                context.log.info(f"  Fallback: Document {doc_id_from_meta} has no elements, but has parsed_text. "
                                 f"Applying RecursiveCharacterTextSplitter to parsed_text.")
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                text_splitter_fallback = RecursiveCharacterTextSplitter(
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                    length_function=len
                )
                try:
                    chunks_text_list = text_splitter_fallback.split_text(parsed_doc.parsed_text.strip())
                    for i, chunk_text_content in enumerate(chunks_text_list):
                        chunk_meta = parsed_doc.original_metadata.copy()
                        chunk_meta.update({
                            "chunk_number_in_doc": len(initial_chunks) + 1,
                            "sub_chunk_sequence": i + 1,
                            "total_sub_chunks": len(chunks_text_list),
                            "chunking_strategy": "fallback_recursive_char_split_on_parsed_text",
                            "original_element_text_length": len(parsed_doc.parsed_text.strip()),
                            "title_hierarchy": {},
                            "is_merged_chunk": False,
                        })
                        initial_chunks.append(ChunkOutput(
                            chunk_text=chunk_text_content,
                            source_document_id=doc_id_from_meta,
                            chunk_metadata=chunk_meta
                        ))
                except Exception as e_fallback_split:
                    context.log.error(f"  Error during fallback splitting for {doc_id_from_meta}: {e_fallback_split}")
            continue

        doc_internal_chunk_counter = 0

        for element_idx, element in enumerate(parsed_doc.elements):
            element_text_content = ""
            element_type_str = "unknown"
            source_element_metadata_from_element_obj = {}

            if isinstance(element, TitleElement) or (isinstance(element, dict) and element.get("element_type") == "title"):
                element_text_content = element.text if hasattr(element, 'text') else element.get("text", "")
                element_type_str = "title"
                title_level = element.level if hasattr(element, 'level') else element.get("level", 1)
                source_element_metadata_from_element_obj["level"] = title_level
                
                keys_to_remove = [lvl for lvl in current_title_hierarchy if lvl >= title_level]
                for key in keys_to_remove:
                    del current_title_hierarchy[key]
                current_title_hierarchy[title_level] = element_text_content.strip()
                logger.debug(f"  Updated title hierarchy for {doc_id_from_meta}: {current_title_hierarchy}")

            elif isinstance(element, NarrativeTextElement) or (isinstance(element, dict) and element.get("element_type") == "narrative_text"):
                element_text_content = element.text if hasattr(element, 'text') else element.get("text", "")
                element_type_str = "narrative_text"
            elif isinstance(element, ListItemElement) or (isinstance(element, dict) and element.get("element_type") == "list_item"):
                element_text_content = element.text if hasattr(element, 'text') else element.get("text", "")
                element_type_str = "list_item"
                source_element_metadata_from_element_obj["item_level"] = element.level if hasattr(element, 'level') else element.get("level")
                source_element_metadata_from_element_obj["ordered"] = element.ordered if hasattr(element, 'ordered') else element.get("ordered")
                source_element_metadata_from_element_obj["item_number"] = element.item_number if hasattr(element, 'item_number') else element.get("item_number")
            elif isinstance(element, TableElement) or (isinstance(element, dict) and element.get("element_type") == "table"):
                if hasattr(element, 'markdown_representation') and element.markdown_representation:
                    element_text_content = element.markdown_representation
                elif hasattr(element, 'text_representation') and element.text_representation:
                    element_text_content = element.text_representation
                elif isinstance(element, dict):
                    element_text_content = element.get("markdown_representation") or element.get("text_representation", "")
                element_type_str = "table"
                source_element_metadata_from_element_obj["caption"] = element.caption if hasattr(element, 'caption') else element.get("caption")
            elif isinstance(element, CodeBlockElement) or (isinstance(element, dict) and element.get("element_type") == "code_block"):
                element_text_content = element.code if hasattr(element, 'code') else element.get("code", "")
                element_type_str = "code_block"
                source_element_metadata_from_element_obj["language"] = element.language if hasattr(element, 'language') else element.get("language")
            elif isinstance(element, PageBreakElement) or (isinstance(element, dict) and element.get("element_type") == "page_break"):
                element_type_str = "page_break"

            element_text_content = element_text_content.strip() if element_text_content else ""

            if not element_text_content and element_type_str != "page_break":
                context.log.debug(f"  Element {element_idx} (type: {element_type_str}) in {doc_id_from_meta} has no text content, skipping chunk creation.")
                continue
            
            base_chunk_meta = parsed_doc.original_metadata.copy()
            base_chunk_meta.update({
                "source_element_index": element_idx,
                "source_element_type": element_type_str,
                "title_hierarchy": current_title_hierarchy.copy(),
                "is_merged_chunk": False,
            })
            if source_element_metadata_from_element_obj:
                base_chunk_meta.update(source_element_metadata_from_element_obj)
            
            element_specific_meta = None
            if hasattr(element, 'metadata') and element.metadata:
                if _PYDANTIC_AVAILABLE and isinstance(element.metadata, DocumentElementMetadata):
                    element_specific_meta = element.metadata.model_dump(exclude_none=True)
                elif isinstance(element.metadata, dict):
                    element_specific_meta = element.metadata
            elif isinstance(element, dict) and element.get("metadata"):
                element_specific_meta = element.get("metadata")
            if element_specific_meta:
                base_chunk_meta.update({f"el_{k}": v for k, v in element_specific_meta.items()})

            if len(element_text_content) > config.max_element_text_length_before_split:
                sub_chunks_created_count = 0
                
                if element_type_str == "table" and hasattr(element, 'markdown_representation') and element.markdown_representation:
                    context.log.info(f"  Table Element {element_idx} (markdown_len: {len(element.markdown_representation)}) in {doc_id_from_meta} is too long. Attempting row-based splitting.")
                    table_sub_chunks_data = split_markdown_table_by_rows(
                        element.markdown_representation,
                        config.target_sentence_split_chunk_size,
                        config.max_merged_chunk_size,
                        context
                    )
                    
                    if table_sub_chunks_data and \
                       (len(table_sub_chunks_data) > 1 or
                       (len(table_sub_chunks_data) == 1 and table_sub_chunks_data[0]["text"] != element.markdown_representation) or
                       (len(table_sub_chunks_data) == 1 and table_sub_chunks_data[0]["start_row_index"] != -1)):
                        
                        context.log.info(f"    Successfully split/processed table into {len(table_sub_chunks_data)} sub-chunk(s) by row.")
                        for sub_idx, chunk_data_dict in enumerate(table_sub_chunks_data):
                            doc_internal_chunk_counter += 1
                            chunk_meta = base_chunk_meta.copy()
                            chunk_meta.update({
                                "chunk_number_in_doc": doc_internal_chunk_counter,
                                "source_element_type": "table_row_chunk",
                                "chunking_strategy": "table_split_by_row_v1",
                                "is_split_from_long_element": True,
                                "original_element_text_length": len(element.markdown_representation),
                                "sub_chunk_sequence_in_element": sub_idx + 1,
                                "total_sub_chunks_from_element": len(table_sub_chunks_data),
                                "table_original_start_row": chunk_data_dict["start_row_index"],
                                "table_original_end_row": chunk_data_dict["end_row_index"],
                            })
                            initial_chunks.append(ChunkOutput(chunk_text=chunk_data_dict["text"], source_document_id=doc_id_from_meta, chunk_metadata=chunk_meta))
                            sub_chunks_created_count += 1
                        if sub_chunks_created_count > 0: continue
                    else:
                        context.log.warning(f"    Row-based splitting of table (idx {element_idx}) did not result in valid sub-chunks or was not applicable. Falling back to sentence splitting for the entire table markdown.")

                elif element_type_str == "code_block":
                    context.log.info(f"  Code Block Element {element_idx} (len: {len(element_text_content)}) in {doc_id_from_meta} is too long. Attempting blank line splitting.")
                    code_sub_chunks = split_code_block_by_blank_lines(
                        element_text_content,
                        config.target_sentence_split_chunk_size,
                        config.max_merged_chunk_size,
                        context
                    )
                    if code_sub_chunks and (len(code_sub_chunks) > 1 or code_sub_chunks[0] != element_text_content):
                        context.log.info(f"    Successfully split code block into {len(code_sub_chunks)} sub-chunks by blank lines.")
                        for sub_idx, sub_chunk_text in enumerate(code_sub_chunks):
                            if not sub_chunk_text.strip(): continue
                            doc_internal_chunk_counter += 1
                            chunk_meta = base_chunk_meta.copy()
                            chunk_meta.update({
                                "chunk_number_in_doc": doc_internal_chunk_counter,
                                "source_element_type": "code_fragment",
                                "chunking_strategy": "code_split_by_blank_line_v1",
                                "is_split_from_long_element": True,
                                "original_element_text_length": len(element_text_content),
                                "sub_chunk_sequence_in_element": sub_idx + 1,
                                "total_sub_chunks_from_element": len(code_sub_chunks),
                            })
                            initial_chunks.append(ChunkOutput(chunk_text=sub_chunk_text, source_document_id=doc_id_from_meta, chunk_metadata=chunk_meta))
                            sub_chunks_created_count += 1
                        if sub_chunks_created_count > 0: continue
                    else:
                        context.log.warning(f"    Blank line splitting of code block (idx {element_idx}) did not result in multiple chunks or failed. Falling back to sentence/char splitting for the code content.")

                context.log.info(f"  Element {element_idx} (type: {element_type_str}, len: {len(element_text_content)}) in {doc_id_from_meta} is too long. Attempting sentence splitting (or fallback).")
                sentences = split_text_into_sentences(element_text_content)
                if not sentences:
                    context.log.warning(f"    Could not split element {element_idx} into sentences. Using RecursiveCharacterTextSplitter as fallback.")
                    from langchain_text_splitters import RecursiveCharacterTextSplitter
                    char_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=config.chunk_size,
                        chunk_overlap=config.chunk_overlap,
                        length_function=len
                    )
                    sub_chunks_texts = char_splitter.split_text(element_text_content)
                    for sub_idx, sub_chunk_text in enumerate(sub_chunks_texts):
                        doc_internal_chunk_counter += 1
                        chunk_meta = base_chunk_meta.copy()
                        chunk_meta.update({
                            "chunk_number_in_doc": doc_internal_chunk_counter,
                            "chunking_strategy": "element_char_split_due_to_no_sentence_split_v1",
                            "is_split_from_long_element": True,
                            "original_element_text_length": len(element_text_content),
                            "sub_chunk_sequence_in_element": sub_idx + 1,
                            "total_sub_chunks_from_element": len(sub_chunks_texts),
                        })
                        initial_chunks.append(ChunkOutput(chunk_text=sub_chunk_text, source_document_id=doc_id_from_meta, chunk_metadata=chunk_meta))
                        context.log.info(f"    Created sub-chunk {sub_idx+1}/{len(sub_chunks_texts)} (char_split) from element {element_idx} for {doc_id_from_meta}")
                else:
                    current_sub_chunk_sentences = []
                    current_sub_chunk_length = 0
                    sub_chunks_generated_for_element = []
                    for i, sentence in enumerate(sentences):
                        sentence_len = len(sentence)
                        if not current_sub_chunk_sentences or \
                           (current_sub_chunk_length + sentence_len + (1 if current_sub_chunk_sentences else 0) <= config.target_sentence_split_chunk_size):
                            current_sub_chunk_sentences.append(sentence)
                            current_sub_chunk_length += sentence_len + (1 if len(current_sub_chunk_sentences) > 1 else 0)
                        else:
                            sub_chunks_generated_for_element.append(" ".join(current_sub_chunk_sentences))
                            overlap_count = config.sentence_split_chunk_overlap_sentences
                            start_index_for_new_chunk = max(0, len(current_sub_chunk_sentences) - overlap_count)
                            current_sub_chunk_sentences = current_sub_chunk_sentences[start_index_for_new_chunk:]
                            current_sub_chunk_sentences.append(sentence)
                            current_sub_chunk_length = len(" ".join(current_sub_chunk_sentences))
                    if current_sub_chunk_sentences:
                        sub_chunks_generated_for_element.append(" ".join(current_sub_chunk_sentences))

                    for sub_idx, sub_chunk_text in enumerate(sub_chunks_generated_for_element):
                        doc_internal_chunk_counter += 1
                        chunk_meta = base_chunk_meta.copy()
                        chunk_meta.update({
                            "chunk_number_in_doc": doc_internal_chunk_counter,
                            "chunking_strategy": "element_split_by_sentence_v1",
                            "is_split_from_long_element": True,
                            "original_element_text_length": len(element_text_content),
                            "sub_chunk_sequence_in_element": sub_idx + 1,
                            "total_sub_chunks_from_element": len(sub_chunks_generated_for_element),
                        })
                        initial_chunks.append(ChunkOutput(chunk_text=sub_chunk_text, source_document_id=doc_id_from_meta, chunk_metadata=chunk_meta))
                        context.log.info(f"    Created sub-chunk {sub_idx+1}/{len(sub_chunks_generated_for_element)} (sentence_split) from element {element_idx} for {doc_id_from_meta}")
            
            else:
                doc_internal_chunk_counter += 1
                chunk_meta = base_chunk_meta.copy()
                chunk_meta.update({
                    "chunk_number_in_doc": doc_internal_chunk_counter,
                    "chunking_strategy": "element_as_chunk_v1",
                    "is_split_from_long_element": False,
                    "original_element_text_length": len(element_text_content),
                })
                initial_chunks.append(ChunkOutput(chunk_text=element_text_content, source_document_id=doc_id_from_meta, chunk_metadata=chunk_meta))
                context.log.info(f"  Created chunk from element {element_idx} (type: {element_type_str}, len: {len(element_text_content)}) in {doc_id_from_meta} (not split).")

    if not initial_chunks:
        context.log.info("No initial chunks to process for merging.")
        return []

    context.log.info(f"Starting short chunk merging process. Initial chunks: {len(initial_chunks)}")
    context.log.info(f"Merging config: MinLenToAvoidMerge={config.min_chunk_length_to_avoid_merge}, MaxMergedSize={config.max_merged_chunk_size}")
    
    merged_chunks: List[ChunkOutput] = []
    i = 0
    while i < len(initial_chunks):
        current_chunk = initial_chunks[i]
        
        merged_this_iteration = False

        if len(current_chunk.chunk_text) < config.min_chunk_length_to_avoid_merge or \
           current_chunk.chunk_metadata.get("source_element_type") == "title":

            chunks_to_merge = [current_chunk]
            current_merged_text = current_chunk.chunk_text
            current_merged_len = len(current_merged_text)
            
            j = i + 1
            if j < len(initial_chunks):
                next_chunk = initial_chunks[j]
                
                can_merge = False
                merge_reason = ""
                potential_merged_text = ""

                is_current_short_title = (
                    current_chunk.chunk_metadata.get("source_element_type") == "title" and
                    len(current_chunk.chunk_text) < config.min_chunk_length_to_avoid_merge
                )
                is_next_short_narrative = (
                    next_chunk.chunk_metadata.get("source_element_type") == "narrative_text" and
                    len(next_chunk.chunk_text) < config.min_chunk_length_to_avoid_merge
                )

                if is_current_short_title and is_next_short_narrative and \
                   current_chunk.source_document_id == next_chunk.source_document_id and \
                   current_chunk.chunk_metadata.get("source_element_index", -2) + 1 == next_chunk.chunk_metadata.get("source_element_index", -1):
                    
                    temp_merged_text = current_chunk.chunk_text + "\n\n" + next_chunk.chunk_text
                    if len(temp_merged_text) <= config.max_merged_chunk_size:
                        can_merge = True
                        merge_reason = "title_narrative"
                        potential_merged_text = temp_merged_text
                        chunks_to_merge.append(next_chunk)

                if not can_merge:
                    if current_chunk.chunk_metadata.get("is_split_from_long_element") and \
                       next_chunk.chunk_metadata.get("is_split_from_long_element") and \
                       current_chunk.chunk_metadata.get("source_document_id") == next_chunk.chunk_metadata.get("source_document_id") and \
                       current_chunk.chunk_metadata.get("source_element_index") == next_chunk.chunk_metadata.get("source_element_index"):
                        
                        separator = "\n" if current_chunk.chunk_metadata.get("source_element_type") == "list_item" else " "
                        temp_merged_text = current_merged_text + separator + next_chunk.chunk_text
                        if len(temp_merged_text) <= config.max_merged_chunk_size:
                            can_merge = True
                            merge_reason = "same_parent_split"
                            potential_merged_text = temp_merged_text
                            chunks_to_merge.append(next_chunk)
                    
                    elif not current_chunk.chunk_metadata.get("is_split_from_long_element") and \
                         not next_chunk.chunk_metadata.get("is_split_from_long_element") and \
                         current_chunk.chunk_metadata.get("source_document_id") == next_chunk.chunk_metadata.get("source_document_id") and \
                         current_chunk.chunk_metadata.get("source_element_type") == next_chunk.chunk_metadata.get("source_element_type") and \
                         current_chunk.chunk_metadata.get("source_element_index", -2) + 1 == next_chunk.chunk_metadata.get("source_element_index", -1):
                         
                        separator = "\n" if current_chunk.chunk_metadata.get("source_element_type") == "list_item" else " "
                        temp_merged_text = current_merged_text + separator + next_chunk.chunk_text
                        if len(temp_merged_text) <= config.max_merged_chunk_size:
                            can_merge = True
                            merge_reason = "adjacent_same_type_short"
                            potential_merged_text = temp_merged_text
                            chunks_to_merge.append(next_chunk)

                if can_merge:
                    first_merged_chunk = chunks_to_merge[0]
                    second_merged_chunk = chunks_to_merge[1]
                    
                    merged_meta = first_merged_chunk.chunk_metadata.copy() 
                    final_merge_reason = merge_reason if merge_reason else "unknown_merge_rule"
                    merged_meta["chunking_strategy"] = f"merged_{final_merge_reason}_v1"
                    merged_meta["is_merged_chunk"] = True
                    merged_meta["original_chunk_ids_merged"] = [c.chunk_id for c in chunks_to_merge]
                    merged_meta["original_texts_lengths_merged"] = [len(c.chunk_text) for c in chunks_to_merge]
                    
                    if merge_reason == "title_narrative":
                        merged_meta["source_element_type"] = "narrative_text"
                        merged_meta["source_element_index_range"] = [
                            first_merged_chunk.chunk_metadata.get("source_element_index"),
                            second_merged_chunk.chunk_metadata.get("source_element_index")
                        ]
                        merged_meta["title_hierarchy"] = first_merged_chunk.chunk_metadata.get("title_hierarchy", {}) 
                    elif merge_reason == "adjacent_same_type_short":
                        merged_meta["source_element_index_range"] = [
                            first_merged_chunk.chunk_metadata.get("source_element_index"),
                            second_merged_chunk.chunk_metadata.get("source_element_index")
                        ]
                    elif merge_reason == "same_parent_split":
                        merged_meta["sub_chunk_sequence_in_element_range"] = [
                            first_merged_chunk.chunk_metadata.get("sub_chunk_sequence_in_element"),
                            second_merged_chunk.chunk_metadata.get("sub_chunk_sequence_in_element")
                        ]
                    
                    merged_chunk_output = ChunkOutput(
                        chunk_text=potential_merged_text,
                        source_document_id=first_merged_chunk.source_document_id,
                        chunk_metadata=merged_meta
                    )
                    merged_chunks.append(merged_chunk_output)
                    context.log.info(f"  Merged 2 chunks (indices {i} and {i+1}) into one. New len: {len(potential_merged_text)}. Strategy: {merged_meta['chunking_strategy']}.")
                    i += 2
                    merged_this_iteration = True
            
            if not merged_this_iteration:
                merged_chunks.append(current_chunk)
                i += 1
        else:
            merged_chunks.append(current_chunk)
            i += 1
            
    context.log.info(f"Merging process finished. Total chunks after merging: {len(merged_chunks)}")

    context.add_output_metadata(metadata={"total_chunks_generated_after_merge": len(merged_chunks)})
    return merged_chunks


# 在 processing_assets.py 中

@dg.asset(
    name="text_embeddings",
    description="Generates vector embeddings for text chunks.",
    group_name="processing",
    deps=["text_chunks"]
)
def generate_embeddings_asset(
    context: dg.AssetExecutionContext,
    text_chunks: List[ChunkOutput], # <--- 这是 text_chunks 资产的输出
    embedder: GGUFEmbeddingResource
) -> List[EmbeddingOutput]:
    # +++ 新增打印语句 +++
    context.log.info(f"generate_embeddings_asset: Received {len(text_chunks)} text_chunks.")
    if text_chunks:
        context.log.info(f"generate_embeddings_asset: First chunk text (first 100 chars): '{text_chunks[0].chunk_text[:100]}'")
        context.log.info(f"generate_embeddings_asset: First chunk metadata: {text_chunks[0].chunk_metadata}")
    # +++ 结束新增打印语句 +++

    all_embeddings: List[EmbeddingOutput] = []
    if not text_chunks:
        context.log.warning("generate_embeddings_asset: No text chunks received, returning empty list.") # 添加一个明确的警告
        return all_embeddings
    
    # --- 确保 chunk_texts_to_encode 不为空才调用 embedder.encode ---
    chunk_texts_to_encode = [chunk.chunk_text for chunk in text_chunks if chunk.chunk_text and chunk.chunk_text.strip()]
    
    if not chunk_texts_to_encode:
        context.log.warning("generate_embeddings_asset: All received text chunks are empty or whitespace after filtering. Returning empty list.")
        # 即使原始 text_chunks 非空，但如果所有 chunk_text 都无效，也应该返回空 embedding 列表
        # 并且要确保下游知道期望的 EmbeddingOutput 数量可能是0
        return all_embeddings # 返回空列表是正确的

    vectors = embedder.encode(chunk_texts_to_encode)

    # --- 确保正确地将嵌入结果映射回原始的 text_chunks 列表（如果数量可能不一致）---
    # 当前的逻辑是假设 vectors 和 chunk_texts_to_encode 一一对应，并且 text_chunks 的顺序与 chunk_texts_to_encode 过滤前的顺序相关
    # 如果 chunk_texts_to_encode 进行了过滤，这里的循环需要更小心
    
    # 一个更安全的映射方式是，只为那些实际被编码的文本块创建 EmbeddingOutput
    # 但这要求下游能处理 EmbeddingOutput 列表长度可能小于 ChunkOutput 列表长度的情况，
    # 或者，我们应该为那些被过滤掉的 chunk 也创建一个带有零向量的 EmbeddingOutput。
    # 我们之前的 LocalModelHandler 修改是为了处理单个空文本，现在这里是资产层面的。

    # 保持与 LocalModelHandler 类似的健壮性：为所有传入的 text_chunks 生成 EmbeddingOutput，
    # 如果其文本为空或嵌入失败，则使用零向量。

    embedding_map = {text: vec for text, vec in zip(chunk_texts_to_encode, vectors)}

    for i, chunk_input in enumerate(text_chunks):
        model_name_for_log = embedder.embedding_model_path
        embedding_vector_for_chunk = [0.0] * embedder.get_embedding_dimension() # 默认为零向量

        if chunk_input.chunk_text and chunk_input.chunk_text.strip() and chunk_input.chunk_text in embedding_map:
            embedding_vector_for_chunk = embedding_map[chunk_input.chunk_text]
        elif chunk_input.chunk_text and chunk_input.chunk_text.strip(): 
            # 文本有效但没有在 embedding_map 中找到 (可能因为 embedder.encode 内部的某些问题)
            context.log.warning(f"generate_embeddings_asset: Valid chunk text for chunk_id {chunk_input.chunk_id} was not found in embedding_map. Using zero vector.")
        else: # 文本本身就是空的
            context.log.info(f"generate_embeddings_asset: Chunk_id {chunk_input.chunk_id} has empty text. Using zero vector.")


        all_embeddings.append(EmbeddingOutput(
            chunk_id=chunk_input.chunk_id,
            chunk_text=chunk_input.chunk_text, # 存储原始文本，即使它是空的
            embedding_vector=embedding_vector_for_chunk,
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
    text_embeddings: List[EmbeddingOutput], # 这应该是来自 generate_embeddings_asset 的输出
    chroma_db: ChromaDBResource # 这是注入的 ChromaDBResource 实例
) -> None:
    if not text_embeddings:
        context.log.warning("vector_storage_asset: No embeddings received, nothing to store in ChromaDB.")
        context.add_output_metadata(metadata={"num_embeddings_stored": 0})
        return

    ids_to_store = [emb.chunk_id for emb in text_embeddings]
    embeddings_to_store = [emb.embedding_vector for emb in text_embeddings]
    documents_to_store = [emb.chunk_text for emb in text_embeddings] # 获取真实的文本内容
    
    cleaned_metadatas: List[Dict[str, Any]] = []
    for i, emb_output in enumerate(text_embeddings):
        # 确保原始元数据存在且是字典
        original_meta = emb_output.original_chunk_metadata if isinstance(emb_output.original_chunk_metadata, dict) else {}
        meta = original_meta.copy() # 使用副本
        
        # 确保 chunk_text 也加入到元数据中（如果需要的话，或者仅依赖 documents 参数）
        # ChromaDB 的 documents 参数用于存储主要文本，元数据用于附加信息
        # 但为了调试和可能的回退，在元数据中也保留一份 chunk_text 是有益的
        meta["chunk_text_in_meta"] = str(emb_output.chunk_text) if emb_output.chunk_text is not None else "[TEXT IS NULL]"

        cleaned_meta_item: Dict[str, Any] = {}
        for key, value in meta.items():
            if isinstance(value, dict):
                if key == "title_hierarchy" and not value: 
                    cleaned_meta_item[key] = "None"
                    context.log.debug(f"Metadata for chunk {emb_output.chunk_id}: Replaced empty title_hierarchy dict with 'None' string.")
                else:
                    try:
                        cleaned_meta_item[key] = json.dumps(value, ensure_ascii=False)
                    except TypeError:
                        cleaned_meta_item[key] = str(value)
                        context.log.warning(f"Metadata for chunk {emb_output.chunk_id}: Could not JSON serialize dict for key '{key}', used str(). Value: {str(value)[:100]}...")
            elif isinstance(value, list):
                try:
                    cleaned_meta_item[key] = json.dumps(value, ensure_ascii=False)
                except TypeError:
                    cleaned_meta_item[key] = str(value)
                    context.log.warning(f"Metadata for chunk {emb_output.chunk_id}: Could not JSON serialize list for key '{key}', used str(). Value: {str(value)[:100]}...")
            elif value is None:
                cleaned_meta_item[key] = "" 
            else: 
                cleaned_meta_item[key] = value
        cleaned_metadatas.append(cleaned_meta_item)

    context.log.info(f"vector_storage_asset: Preparing to add/update {len(ids_to_store)} items to ChromaDB collection '{chroma_db.collection_name}'.")
    if ids_to_store:
        context.log.info(f"vector_storage_asset: Sample ID to store: {ids_to_store[0]}")
        # 确保 documents_to_store 也有对应内容，并且不是 None
        sample_doc_text = "[EMPTY DOCUMENT]"
        if documents_to_store and documents_to_store[0] is not None:
            sample_doc_text = str(documents_to_store[0])[:100] # 显示前100字符
        elif documents_to_store and documents_to_store[0] is None:
            sample_doc_text = "[DOCUMENT IS NULL]"
        context.log.info(f"vector_storage_asset: Sample document to store (from documents_to_store, first 100 chars): '{sample_doc_text}'")
        
        sample_meta_text = "[NO METADATA]"
        if cleaned_metadatas:
            sample_meta_text = str(cleaned_metadatas[0])[:200] # 显示元数据摘要
        context.log.info(f"vector_storage_asset: Sample cleaned metadata for first item: {sample_meta_text}")

    try:
        chroma_db.add_embeddings(
            ids=ids_to_store, 
            embeddings=embeddings_to_store, 
            documents=documents_to_store, # 传递真实的文本内容给ChromaDB的documents字段
            metadatas=cleaned_metadatas
        )
        # 尝试获取并记录操作后的集合计数
        # 注意: chroma_db._collection 可能是私有属性，直接访问不推荐，但为了调试可以尝试
        # 更好的方式是 ChromaDBResource 提供一个 get_collection_count() 方法
        collection_count_after_add = -1 # 默认值
        try:
            if chroma_db._collection: # 确保 _collection 不是 None
                 collection_count_after_add = chroma_db._collection.count()
        except Exception as e_count:
            context.log.warning(f"vector_storage_asset: Could not get collection count after add: {e_count}")

        context.add_output_metadata(metadata={"num_embeddings_stored": len(ids_to_store), "collection_count_after_add": collection_count_after_add})
        context.log.info(f"vector_storage_asset: Successfully called add_embeddings. Stored {len(ids_to_store)} items. Collection count now: {collection_count_after_add}")
    except Exception as e_chroma_add:
        context.log.error(f"vector_storage_asset: Failed to add embeddings to ChromaDB: {e_chroma_add}", exc_info=True)
        raise

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
    config: BM25IndexConfig, # 确保 BM25IndexConfig 在文件某处已定义
    text_chunks: List[ChunkOutput]
) -> None:
    if not text_chunks:
        context.log.warning("keyword_index_asset: No text chunks received, skipping BM25 index building.")
        context.add_output_metadata(metadata={"num_documents_indexed": 0, "index_directory_path": config.index_file_path})
        return

    # --- 新增：检查并记录空文本块 ---
    valid_chunks_for_indexing: List[ChunkOutput] = []
    for idx, chunk in enumerate(text_chunks):
        if chunk.chunk_text and chunk.chunk_text.strip():
            valid_chunks_for_indexing.append(chunk)
        else:
            context.log.warning(f"keyword_index_asset: Chunk {idx} (ID: {chunk.chunk_id}) has empty or whitespace-only text. Skipping for BM25 indexing.")
    
    if not valid_chunks_for_indexing:
        context.log.warning("keyword_index_asset: All received text chunks have empty or whitespace-only text after filtering. Skipping BM25 index building.")
        context.add_output_metadata(metadata={"num_documents_indexed": 0, "index_directory_path": config.index_file_path})
        return
    # --- 结束新增 ---

    # 使用过滤后的有效块
    corpus_texts = [chunk.chunk_text for chunk in valid_chunks_for_indexing]
    document_ids = [chunk.chunk_id for chunk in valid_chunks_for_indexing] # 确保ID与有效文本对应

    context.log.info(f"keyword_index_asset: Preparing to index {len(corpus_texts)} valid text chunks for BM25.")
    if corpus_texts: # 仅在有数据时打印样本
        context.log.info(f"keyword_index_asset: Sample document ID for BM25: {document_ids[0]}")
        context.log.info(f"keyword_index_asset: Sample document text for BM25 (first 50 chars): '{str(corpus_texts[0])[:50]}'")

    try:
        corpus_tokenized_jieba = [list(jieba.cut_for_search(text)) for text in corpus_texts]
        context.log.info(f"keyword_index_asset: Tokenized {len(corpus_tokenized_jieba)} texts for BM25.")
        
        bm25_model = bm25s.BM25() # 使用默认参数初始化
        context.log.info("keyword_index_asset: BM25 model initialized.")
        
        bm25_model.index(corpus_tokenized_jieba)
        indexed_doc_count = len(bm25_model.doc_freqs) if hasattr(bm25_model, 'doc_freqs') and bm25_model.doc_freqs is not None else len(corpus_tokenized_jieba)
        context.log.info(f"keyword_index_asset: BM25 model indexing complete for {indexed_doc_count} documents.")
        
        index_directory = config.index_file_path
        context.log.info(f"keyword_index_asset: BM25 index will be saved to directory: {index_directory}")
        os.makedirs(index_directory, exist_ok=True)
        
        bm25_model.save(index_directory) 
        context.log.info(f"keyword_index_asset: bm25_model.save('{index_directory}') called.")
        
        doc_ids_path = os.path.join(index_directory, "doc_ids.pkl")
        with open(doc_ids_path, 'wb') as f_out:
            pickle.dump(document_ids, f_out)
        context.log.info(f"keyword_index_asset: doc_ids.pkl saved to {doc_ids_path} with {len(document_ids)} IDs.")
        
        # 验证文件是否真的创建了
        expected_params_file = os.path.join(index_directory, "params.index.json") # bm25s 保存时会创建这个
        if os.path.exists(expected_params_file) and os.path.exists(doc_ids_path):
            context.log.info(f"keyword_index_asset: Verified that BM25 index files (e.g., params.index.json, doc_ids.pkl) exist in {index_directory}.")
        else:
            context.log.error(f"keyword_index_asset: BM25 index files (e.g., params.index.json or doc_ids.pkl) NOT FOUND in {index_directory} after save operations!")
            context.log.error(f"keyword_index_asset: Check - params.index.json exists: {os.path.exists(expected_params_file)}")
            context.log.error(f"keyword_index_asset: Check - doc_ids.pkl exists: {os.path.exists(doc_ids_path)}")
            # 如果文件未找到，可能需要抛出异常以使资产失败
            # raise FileNotFoundError(f"BM25 index files not found in {index_directory} after save.")

        context.add_output_metadata(
            metadata={
                "num_documents_indexed": len(corpus_texts), 
                "index_directory_path": index_directory,
                "bm25_corpus_size_actual": indexed_doc_count
            }
        )
        context.log.info("keyword_index_asset: BM25 indexing and saving completed successfully.")
    except Exception as e_bm25:
        context.log.error(f"keyword_index_asset: Error during BM25 indexing or saving: {e_bm25}", exc_info=True)
        raise

# --- KG Extraction 相关的配置和资产 ---

from zhz_rag.llm.rag_prompts import KG_EXTRACTION_SINGLE_CHUNK_PROMPT_TEMPLATE_V1, KG_EXTRACTION_BATCH_PROMPT_TEMPLATE_V1 # 导入常量

class KGExtractionConfig(dg.Config):
    extraction_prompt_template: str = KG_EXTRACTION_SINGLE_CHUNK_PROMPT_TEMPLATE_V1
    local_llm_model_name: str = "Qwen3-1.7B-GGUF_via_llama.cpp"

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


@dg.asset(
    name="kg_extractions",
    description="Extracts entities and relations from text chunks for knowledge graph construction.",
    group_name="kg_building",
    io_manager_key="pydantic_json_io_manager",
    deps=["text_chunks"]
)
async def kg_extraction_asset(
    context: dg.AssetExecutionContext, # Pylance 提示 dg.AssetExecutionContext 未定义 "SystemResource"
    text_chunks: List[ChunkOutput],
    config: KGExtractionConfig,
    LocalLLM_api: LocalLLMAPIResource,
    system_info: SystemResource  # <--- 我们添加了 system_info
) -> List[KGTripleSetOutput]:
    all_kg_outputs: List[KGTripleSetOutput] = []
    if not text_chunks:
        context.log.info("No text chunks received for KG extraction, skipping.")
        return all_kg_outputs

    total_input_chunks = len(text_chunks)
    total_entities_extracted_overall = 0
    total_relations_extracted_overall = 0
    successfully_processed_chunks_count = 0
    
    # 并发控制参数
    recommended_concurrency = system_info.get_recommended_concurrent_tasks(task_type="kg_extraction_llm")
    CONCURRENT_REQUESTS_LIMIT = max(1, recommended_concurrency) # 直接使用HAL推荐，但至少为1
    context.log.info(f"HAL recommended concurrency for 'kg_extraction_llm': {recommended_concurrency}. Effective limit set to: {CONCURRENT_REQUESTS_LIMIT}")
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)


    async def extract_kg_for_chunk(chunk: ChunkOutput) -> Optional[KGTripleSetOutput]:
        async with semaphore:
            # 使用单个chunk的prompt模板
            prompt = config.extraction_prompt_template.format(text_to_extract=chunk.chunk_text)
            try:
                context.log.debug(f"Starting KG extraction for chunk_id: {chunk.chunk_id}, Text (start): {chunk.chunk_text[:100]}...")
                structured_response = await LocalLLM_api.generate_structured_output(
                    prompt=prompt, 
                    json_schema=DEFAULT_KG_EXTRACTION_SCHEMA # 使用单个对象的schema
                )
                
                # 确保 structured_response 是字典类型
                if not isinstance(structured_response, dict):
                    context.log.error(f"Failed KG extraction for chunk {chunk.chunk_id}: LLM response was not a dict. Got: {type(structured_response)}. Response: {str(structured_response)[:200]}")
                    return None

                entities_data = structured_response.get("entities", [])
                extracted_entities_list = [
                    ExtractedEntity(text=normalize_text_for_id(e.get("text","")), label=e.get("label","UNKNOWN").upper())
                    for e in entities_data if isinstance(e, dict)
                ]
                
                relations_data = structured_response.get("relations", [])
                extracted_relations_list = [
                    ExtractedRelation(
                        head_entity_text=r.get('head_entity_text',""), 
                        head_entity_label=r.get('head_entity_label',"UNKNOWN").upper(), 
                        relation_type=r.get('relation_type',"UNKNOWN").upper(), 
                        tail_entity_text=r.get('tail_entity_text',""), 
                        tail_entity_label=r.get('tail_entity_label',"UNKNOWN").upper()
                    ) 
                    for r in relations_data if isinstance(r, dict) and 
                                               r.get('head_entity_text') and r.get('head_entity_label') and
                                               r.get('relation_type') and r.get('tail_entity_text') and
                                               r.get('tail_entity_label')
                ]
                
                context.log.debug(f"Finished KG extraction for chunk_id: {chunk.chunk_id}. Entities: {len(extracted_entities_list)}, Relations: {len(extracted_relations_list)}")
                return KGTripleSetOutput(
                    chunk_id=chunk.chunk_id,
                    extracted_entities=extracted_entities_list,
                    extracted_relations=extracted_relations_list,
                    extraction_model_name=config.local_llm_model_name,
                    original_chunk_metadata=chunk.chunk_metadata
                )
            except Exception as e:
                context.log.error(f"Failed KG extraction for chunk {chunk.chunk_id}: {e}", exc_info=True)
                return None 

    context.log.info(f"Starting KG extraction for {total_input_chunks} chunks with concurrency limit: {CONCURRENT_REQUESTS_LIMIT}.")
    
    tasks = [extract_kg_for_chunk(chunk) for chunk in text_chunks]
    
    results = await asyncio.gather(*tasks)
    
    context.log.info(f"Finished all KG extraction tasks. Received {len(results)} results (including potential None for failures).")

    for result_item in results:
        if result_item and isinstance(result_item, KGTripleSetOutput):
            all_kg_outputs.append(result_item)
            total_entities_extracted_overall += len(result_item.extracted_entities)
            total_relations_extracted_overall += len(result_item.extracted_relations)
            successfully_processed_chunks_count +=1
        elif result_item is None:
            context.log.warning("A KG extraction task failed and returned None.")
            
    context.log.info(f"KG extraction complete. Successfully processed {successfully_processed_chunks_count} out of {total_input_chunks} chunks.")
    context.add_output_metadata(
        metadata={
            "total_chunks_input_to_kg": total_input_chunks, # 恢复为 total_input_chunks
            "chunks_successfully_extracted_kg": successfully_processed_chunks_count,
            "total_entities_extracted": total_entities_extracted_overall, 
            "total_relations_extracted": total_relations_extracted_overall
            # 移除了批处理相关的元数据 "total_batches_processed", "batch_size_configured"
        }
    )
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