# zhz_rag_pipeline/ingestion_assets.py
import dagster as dg
import os
from typing import List, Dict, Any, Union

# 从我们新建的pydantic模型文件中导入
from .pydantic_models_dagster import LoadedDocumentOutput, ParsedDocumentOutput

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
    group_name="ingestion", # 也属于摄入组
    deps=[load_documents_asset] 
)
def parse_document_asset(
    context: dg.AssetExecutionContext, 
    raw_documents: List[LoadedDocumentOutput] 
) -> List[ParsedDocumentOutput]: 
    
    parsed_docs: List[ParsedDocumentOutput] = []
    context.log.info(f"Received {len(raw_documents)} documents to parse.")

    for doc_input in raw_documents:
        context.log.info(f"Parsing document: {doc_input.document_path} (Type: {doc_input.file_type})")
        parsed_text_content = ""
        
        try:
            if doc_input.file_type == ".txt":
                if isinstance(doc_input.raw_content, bytes):
                    parsed_text_content = doc_input.raw_content.decode('utf-8')
                elif isinstance(doc_input.raw_content, str):
                    parsed_text_content = doc_input.raw_content
                else:
                    # 抛出更具体的错误或记录并跳过
                    context.log.error(f"Unexpected raw_content type for .txt file: {type(doc_input.raw_content)} in {doc_input.document_path}")
                    parsed_text_content = f"[Error: Unexpected content type {type(doc_input.raw_content)}]"

            # TODO: Add parsers for other file types like .pdf, .docx here
            # elif doc_input.file_type == ".pdf":
            #     parsed_text_content = "[PDF parsing not yet implemented]"
            #     context.log.warning(f"PDF parsing not yet implemented for {doc_input.document_path}")
            else:
                parsed_text_content = f"[Unsupported file type: {doc_input.file_type}]"
                context.log.warning(f"Unsupported file type '{doc_input.file_type}' for parsing: {doc_input.document_path}")

            parsed_output = ParsedDocumentOutput(
                parsed_text=parsed_text_content,
                # document_structure is None by default
                original_metadata=doc_input.metadata 
            )
            parsed_docs.append(parsed_output)
            context.log.info(f"Successfully (or with placeholder) parsed: {doc_input.document_path}")

        except Exception as e:
            context.log.error(f"Failed to parse document {doc_input.document_path}: {e}")
            parsed_output = ParsedDocumentOutput(
                parsed_text=f"[Error parsing document: {str(e)}]",
                original_metadata=doc_input.metadata
            )
            parsed_docs.append(parsed_output)

    if parsed_docs:
        context.add_output_metadata(
            metadata={
                "num_documents_parsed": len(parsed_docs),
                "first_parsed_doc_filename": parsed_docs[0].original_metadata.get("filename", "N/A") if parsed_docs else "N/A"
            }
        )
    return parsed_docs

all_ingestion_assets = [load_documents_asset, parse_document_asset]