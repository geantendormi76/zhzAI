# zhz_rag_pipeline/ingestion_assets.py
import dagster as dg
import os
from typing import List, Dict, Any, Union, Optional
from datetime import datetime, timezone

# --- 修改：导入分发器并设置Pydantic可用性标志 ---
# 尝试导入Pydantic模型，并设置一个标志，以便在模型不可用时代码可以优雅地降级。
try:
    from .pydantic_models_dagster import LoadedDocumentOutput, ParsedDocumentOutput, NarrativeTextElement
    _PYDANTIC_AVAILABLE = True
except ImportError:
    LoadedDocumentOutput = dict  # type: ignore
    ParsedDocumentOutput = dict  # type: ignore
    NarrativeTextElement = dict  # type: ignore
    _PYDANTIC_AVAILABLE = False

from .parsers import dispatch_parsing # <--- 修改导入路径
# --- 修改结束 ---

class LoadDocumentsConfig(dg.Config):
    documents_directory: str = "/home/zhz/zhz_agent/data/raw_documents/" # 更新后的原始文档目录
    allowed_extensions: List[str] = [".txt", ".md", ".docx", ".pdf", ".xlsx", ".html", ".htm"] # 扩大允许范围以测试所有解析器

@dg.asset(
    name="raw_documents",
    description="Loads raw documents from a specified directory.",
    group_name="ingestion"
)
def load_documents_asset(
    context: dg.AssetExecutionContext, 
    config: LoadDocumentsConfig
) -> List[LoadedDocumentOutput]:  # type: ignore
    
    loaded_docs: List[LoadedDocumentOutput] = []  # type: ignore
    target_directory = config.documents_directory
    allowed_exts = tuple(config.allowed_extensions) 

    context.log.info(f"Scanning directory: {target_directory} for files with extensions: {allowed_exts}")

    if not os.path.isdir(target_directory):
        context.log.error(f"Directory not found: {target_directory}")
        return loaded_docs

    for filename in os.listdir(target_directory):
        file_path = os.path.join(target_directory, filename)
        if os.path.isfile(file_path):
            file_name_lower = filename.lower()
            file_extension = os.path.splitext(file_name_lower)[1]

            if file_extension in allowed_exts:
                context.log.info(f"Found matching file: {file_path}")
                try:
                    file_stat = os.stat(file_path)
                    
                    # --- VITAL FIX: Do not pass raw_content ---
                    # The downstream parser will handle reading the file from the path.
                    doc_output_data = {
                        "document_path": file_path,
                        "file_type": file_extension,
                        # raw_content is intentionally omitted
                        "metadata": {
                            "filename": filename,
                            "source_directory": target_directory,
                            "size_bytes": file_stat.st_size,
                            "creation_time_utc": datetime.fromtimestamp(file_stat.st_ctime, tz=timezone.utc).isoformat(),
                            "modified_time_utc": datetime.fromtimestamp(file_stat.st_mtime, tz=timezone.utc).isoformat()
                        }
                    }

                    if _PYDANTIC_AVAILABLE:
                        loaded_docs.append(LoadedDocumentOutput(**doc_output_data))
                    else:
                        loaded_docs.append(doc_output_data)

                    context.log.info(f"Successfully created LoadedDocumentOutput for: {file_path}")
                except Exception as e:
                    context.log.error(f"Failed to process file {file_path}: {e}")
            else:
                context.log.debug(f"Skipping file with non-allowed extension: {file_path}")
        else:
            context.log.debug(f"Skipping non-file item: {file_path}")
            
    if not loaded_docs:
        context.log.warning(f"No matching documents found in {target_directory}")

    if loaded_docs:
        first_doc_path = loaded_docs[0].document_path if _PYDANTIC_AVAILABLE and loaded_docs else "N/A"
        context.add_output_metadata(
            metadata={
                "num_documents_loaded": len(loaded_docs),
                "first_document_path": first_doc_path
            }
        )
    return loaded_docs



@dg.asset(
    name="parsed_documents",
    description="Parses loaded documents into text and extracts basic structure using a dispatcher.",
    group_name="ingestion"
)
def parse_document_asset(
    context: dg.AssetExecutionContext,
    raw_documents: List[LoadedDocumentOutput] # type: ignore
) -> List[ParsedDocumentOutput]: # type: ignore
    
    parsed_docs_output_list: List[ParsedDocumentOutput] = [] # type: ignore
    context.log.info(f"Received {len(raw_documents)} documents to parse.")

    for doc_input in raw_documents:
        doc_path = doc_input.document_path
        file_ext = doc_input.file_type.lower()
        original_metadata = doc_input.metadata.copy()
        original_metadata["source_file_path"] = doc_path

        context.log.info(f"Attempting to parse document: {doc_path} (Type: {file_ext})")

        try:
            # --- VITAL REFACTOR ---
            # 直接将文件路径传递给解析器，不再处理字节内容
            # 对于文本文件，解析器内部自己会用 'rt' 模式读取
            # 对于二进制文件(pdf, docx, xlsx)，解析器会用 'rb' 模式或相应库读取
            parsed_output = dispatch_parsing(file_ext, doc_path, original_metadata)

            if not parsed_output:
                context.log.warning(f"Parser for '{file_ext}' returned no output for {doc_path}. Creating a fallback.")
                fallback_text = f"[Content Not Parsed by Specific Parser: {doc_path}]"
                elements = [NarrativeTextElement(text=fallback_text)]
                parsed_output = ParsedDocumentOutput(
                    parsed_text=fallback_text,
                    elements=elements,
                    original_metadata=original_metadata
                )

            # 确保输出总是 Pydantic 模型
            if isinstance(parsed_output, dict):
                parsed_output = ParsedDocumentOutput(**parsed_output)
            
            parsed_docs_output_list.append(parsed_output)
            context.log.info(f"Successfully processed: {doc_path}")

        except Exception as e:
            context.log.error(f"Critical error during parsing asset for {doc_path}: {e}", exc_info=True)
            error_text = f"[Critical Parsing Exception for {doc_path}: {str(e)}]"
            elements = [NarrativeTextElement(text=error_text)]
            error_output = ParsedDocumentOutput(
                parsed_text=error_text,
                elements=elements,
                original_metadata=original_metadata
            )
            parsed_docs_output_list.append(error_output)

    if parsed_docs_output_list:
        context.add_output_metadata(
            metadata={
                "num_documents_processed_for_parsing": len(raw_documents),
                "num_parsed_document_outputs_generated": len(parsed_docs_output_list),
            }
        )
    return parsed_docs_output_list



all_ingestion_assets = [load_documents_asset, parse_document_asset]