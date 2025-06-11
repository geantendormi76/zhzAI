# zhz_rag_pipeline/ingestion_assets.py
import dagster as dg
import os
from typing import List, Dict, Any, Union, Optional

# 从我们新建的pydantic模型文件中导入
from .pydantic_models_dagster import LoadedDocumentOutput, ParsedDocumentOutput
from .document_parsers import (
    parse_markdown_to_structured_output,
    parse_docx_to_structured_output,
    parse_pdf_to_structured_output,
    parse_xlsx_to_structured_output,
    parse_html_to_structured_output 
)


class LoadDocumentsConfig(dg.Config):
    documents_directory: str = "/home/zhz/zhz_agent/data/raw_documents/" # 更新后的原始文档目录
    allowed_extensions: List[str] = [".txt"]

@dg.asset(
    name="raw_documents",
    description="Loads raw documents from a specified directory.",
    group_name="ingestion" # 给资产分组
)
def load_documents_asset(
    context: dg.AssetExecutionContext, 
    config: LoadDocumentsConfig
) -> List[LoadedDocumentOutput]:
    
    loaded_docs: List[LoadedDocumentOutput] = []
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
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    doc_output = LoadedDocumentOutput(
                        document_path=file_path,
                        file_type=file_extension,
                        raw_content=content,
                        metadata={
                            "filename": filename,
                            "source_directory": target_directory,
                            "size_bytes": os.path.getsize(file_path)
                        }
                    )
                    loaded_docs.append(doc_output)
                    context.log.info(f"Successfully loaded and created output for: {file_path}")
                except Exception as e:
                    context.log.error(f"Failed to read or process file {file_path}: {e}")
            else:
                context.log.debug(f"Skipping file with non-allowed extension: {file_path}")
        else:
            context.log.debug(f"Skipping non-file item: {file_path}")
            
    if not loaded_docs:
        context.log.warning(f"No matching documents found in {target_directory}")

    if loaded_docs:
        context.add_output_metadata(
            metadata={
                "num_documents_loaded": len(loaded_docs),
                "first_document_path": loaded_docs[0].document_path if loaded_docs else "N/A"
            }
        )
    return loaded_docs


@dg.asset(
    name="parsed_documents",
    description="Parses loaded documents into text and extracts basic structure.",
    group_name="ingestion",
    io_manager_key="pydantic_json_io_manager" # <--- 建议为这个资产使用PydanticListJsonIOManager
                                             # 因为它输出 List[ParsedDocumentOutput]
)
def parse_document_asset(
    context: dg.AssetExecutionContext, 
    raw_documents: List[LoadedDocumentOutput] 
) -> List[ParsedDocumentOutput]: # <--- 确认返回类型是 List[ParsedDocumentOutput]
    
    parsed_docs_output_list: List[ParsedDocumentOutput] = []
    context.log.info(f"Received {len(raw_documents)} documents to parse.")

    for doc_input in raw_documents:
        context.log.info(f"Attempting to parse document: {doc_input.document_path} (Type: {doc_input.file_type})")
        
        parsed_output: Optional[ParsedDocumentOutput] = None # Or Optional[dict] if Pydantic fails
        file_ext = doc_input.file_type.lower()
        file_path = doc_input.document_path # For parsers that need path
        
        # Ensure raw_content is string for text-based parsers
        content_str = ""
        if isinstance(doc_input.raw_content, bytes):
            try:
                content_str = doc_input.raw_content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    content_str = doc_input.raw_content.decode('gbk') # Common fallback for Chinese text
                except UnicodeDecodeError:
                    context.log.error(f"Could not decode content for {doc_input.document_path}. Skipping content-based parsing.")
                    content_str = f"[Unparsable Content: {doc_input.document_path}]"
        elif isinstance(doc_input.raw_content, str):
            content_str = doc_input.raw_content
        
        # Prepare metadata to pass to parser functions
        current_original_metadata = doc_input.metadata.copy()
        current_original_metadata["source_file_path"] = file_path # Ensure path is in metadata

        try:
            if file_ext == ".md":
                parsed_output = parse_markdown_to_structured_output(content_str, current_original_metadata)
            elif file_ext == ".docx":
                parsed_output = parse_docx_to_structured_output(file_path, current_original_metadata)
            elif file_ext == ".pdf":
                parsed_output = parse_pdf_to_structured_output(file_path, current_original_metadata)
            elif file_ext == ".xlsx":
                parsed_output = parse_xlsx_to_structured_output(file_path, current_original_metadata)
            elif file_ext in [".html", ".htm"]:
                parsed_output = parse_html_to_structured_output(content_str, current_original_metadata)
            else: # Fallback for .txt and other unknown types
                context.log.warning(f"Unsupported file type '{file_ext}' for structured parsing, treating as plain text: {file_path}")
                # For plain text, elements list will contain a single NarrativeTextElement
                from .pydantic_models_dagster import NarrativeTextElement # Local import to avoid top-level issues
                elements = [NarrativeTextElement(text=content_str)]
                parsed_output = ParsedDocumentOutput(
                    parsed_text=content_str,
                    elements=elements, # type: ignore
                    original_metadata=current_original_metadata
                )

            if parsed_output:
                # Ensure parsed_output is indeed a ParsedDocumentOutput if Pydantic models were loaded
                # If not (e.g., due to ImportError in document_parsers), it might be a dict.
                # The io_manager will handle Pydantic models. If it's a dict, PickledObject IO manager will work.
                parsed_docs_output_list.append(parsed_output) # type: ignore
                context.log.info(f"Successfully parsed (or created placeholder for): {file_path}")
            else:
                context.log.error(f"Parser returned None for {file_path}. Adding a fallback error entry.")
                from .pydantic_models_dagster import NarrativeTextElement
                elements = [NarrativeTextElement(text=f"[Parsing Error for {file_path}]")]
                parsed_docs_output_list.append(ParsedDocumentOutput(
                    parsed_text=f"[Parsing Error for {file_path}]",
                    elements=elements, # type: ignore
                    original_metadata=current_original_metadata
                ))

        except Exception as e_parse_asset:
            context.log.error(f"Critical error during parsing asset for {file_path}: {e_parse_asset}", exc_info=True)
            from .pydantic_models_dagster import NarrativeTextElement
            elements = [NarrativeTextElement(text=f"[Critical Parsing Exception for {file_path}: {str(e_parse_asset)}]")]
            parsed_docs_output_list.append(ParsedDocumentOutput(
                parsed_text=f"[Critical Parsing Exception for {file_path}: {str(e_parse_asset)}]",
                elements=elements, # type: ignore
                original_metadata=current_original_metadata
            ))
            
    if parsed_docs_output_list:
        context.add_output_metadata(
            metadata={
                "num_documents_processed_for_parsing": len(raw_documents),
                "num_parsed_document_outputs_generated": len(parsed_docs_output_list),
            }
        )
    return parsed_docs_output_list
all_ingestion_assets = [load_documents_asset, parse_document_asset]