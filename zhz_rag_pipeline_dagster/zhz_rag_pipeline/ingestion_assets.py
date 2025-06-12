# zhz_rag_pipeline/ingestion_assets.py
import dagster as dg
import os
from typing import List, Dict, Any, Union, Optional

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
    group_name="ingestion" # 给资产分组
)
def load_documents_asset(
    context: dg.AssetExecutionContext, 
    config: LoadDocumentsConfig
) -> List[Any]: 
    
    loaded_docs: List[Any] = [] 
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
                    # 对于需要路径的解析器（如docx, pdf），我们可能不需要立即读取内容
                    # 但为了简单起见并支持文本解析器，我们仍然读取
                    with open(file_path, 'rb') as f: # 以二进制模式读取以更好地处理不同编码
                        raw_bytes = f.read()
                    
                    # 在parse_document_asset中进行更智能的解码
                    doc_output_data = {
                        "document_path": file_path,
                        "file_type": file_extension,
                        "raw_content": raw_bytes, # 传递原始字节
                        "metadata": {
                            "filename": filename,
                            "source_directory": target_directory,
                            "size_bytes": os.path.getsize(file_path)
                        }
                    }

                    if _PYDANTIC_AVAILABLE:
                        loaded_docs.append(LoadedDocumentOutput(**doc_output_data))
                    else:
                        loaded_docs.append(doc_output_data) # type: ignore

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
        first_doc_path = loaded_docs[0].document_path if _PYDANTIC_AVAILABLE else loaded_docs[0]['document_path']
        context.add_output_metadata(
            metadata={
                "num_documents_loaded": len(loaded_docs),
                "first_document_path": first_doc_path if loaded_docs else "N/A"
            }
        )
    return loaded_docs


@dg.asset(
    name="parsed_documents",
    description="Parses loaded documents into text and extracts basic structure using a dispatcher.",
    group_name="ingestion",
    io_manager_key="pydantic_json_io_manager"
)
def parse_document_asset(
    context: dg.AssetExecutionContext, 
    raw_documents: List[Any]  # <--- 修改: 参数类型注解改为 List[Any]
) -> List[Any]: # <--- 修改: 返回类型注解改为 List[Any]
    
    parsed_docs_output_list: List[Any] = [] # <--- 修改: 变量类型注解改为 List[Any]
    context.log.info(f"Received {len(raw_documents)} documents to parse.")

    for doc_input_any in raw_documents: # <--- 修改变量名以反映其类型是 Any
        # Pydantic和字典访问的兼容性处理
        # 我们需要先判断 doc_input_any 的实际类型
        doc_path: str
        file_ext: str
        raw_content: Union[str, bytes] # 保持 Union 类型
        original_metadata: Dict[str, Any]

        if _PYDANTIC_AVAILABLE and isinstance(doc_input_any, LoadedDocumentOutput):
            doc_path = doc_input_any.document_path
            file_ext = doc_input_any.file_type.lower()
            raw_content = doc_input_any.raw_content
            original_metadata = doc_input_any.metadata.copy()
        elif isinstance(doc_input_any, dict): # 如果是字典 (Pydantic不可用时的回退)
            doc_path = doc_input_any.get('document_path', '')
            file_ext = doc_input_any.get('file_type', '').lower()
            raw_content = doc_input_any.get('raw_content', b'') # 默认为空字节串
            original_metadata = doc_input_any.get('metadata', {}).copy()
        else:
            context.log.error(f"Skipping document with unexpected type: {type(doc_input_any)}")
            continue
        context.log.info(f"Attempting to parse document: {doc_path} (Type: {file_ext})")
        
        parsed_output: Optional[Any] = None
        
        content_str = ""
        if isinstance(raw_content, bytes):
            try:
                content_str = raw_content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    content_str = raw_content.decode('gbk')
                except UnicodeDecodeError:
                    context.log.error(f"Could not decode content for {doc_path}. Skipping content-based parsing.")
                    content_str = f"[Unparsable Content: {doc_path}]"
        elif isinstance(raw_content, str):
            content_str = raw_content
        
        current_original_metadata = original_metadata
        current_original_metadata["source_file_path"] = doc_path

        try:
            # --- 修改：使用分发器 ---
            # 决定是传递路径还是内容字符串
            # 对于md, html, txt，我们传递content_str
            # 对于docx, pdf, xlsx，我们传递doc_path
            input_for_parser = content_str if file_ext in [".md", ".html", ".htm", ".txt"] else doc_path
            
            parsed_output = dispatch_parsing(file_ext, input_for_parser, current_original_metadata)
            # --- 修改结束 ---

            if not parsed_output and file_ext != ".txt": # 如果特定解析器失败，但不是txt，尝试纯文本
                context.log.warning(f"Specific parser for '{file_ext}' failed or returned None for {doc_path}. Falling back to plain text.")
                if _PYDANTIC_AVAILABLE:
                    elements = [NarrativeTextElement(text=content_str)]
                    parsed_output = ParsedDocumentOutput(
                        parsed_text=content_str, elements=elements, original_metadata=current_original_metadata
                    )
                else:
                    elements = [{"element_type":"narrative_text", "text":content_str}]
                    parsed_output = {"parsed_text":content_str, "elements":elements, "original_metadata":current_original_metadata}

            if file_ext == ".txt" and not parsed_output: # TXT 的特定回退逻辑
                context.log.info(f"Treating .txt file as plain text: {doc_path}")
                if _PYDANTIC_AVAILABLE:
                    elements = [NarrativeTextElement(text=content_str)]
                    parsed_output = ParsedDocumentOutput(
                        parsed_text=content_str, elements=elements, original_metadata=current_original_metadata
                    )
                else:
                    elements = [{"element_type":"narrative_text", "text":content_str}]
                    parsed_output = {"parsed_text":content_str, "elements":elements, "original_metadata":current_original_metadata}

            if parsed_output:
                # 确保 parsed_output 是 ParsedDocumentOutput 实例或可接受的字典
                if _PYDANTIC_AVAILABLE and not isinstance(parsed_output, ParsedDocumentOutput) and isinstance(parsed_output, dict):
                    try:
                        # 尝试将来自解析器回退的字典转换为Pydantic模型
                        parsed_output = ParsedDocumentOutput(**parsed_output)
                    except Exception as e_cast:
                        context.log.error(f"Failed to cast parsed_output dict to Pydantic model for {doc_path}: {e_cast}")
                        # 回退到错误表示
                        elements = [NarrativeTextElement(text=f"[Casting Error for {doc_path}]")] if _PYDANTIC_AVAILABLE else [{"element_type":"narrative_text", "text":f"[Casting Error for {doc_path}]"}]
                        parsed_output = ParsedDocumentOutput(parsed_text=f"[Casting Error for {doc_path}]", elements=elements, original_metadata=current_original_metadata) if _PYDANTIC_AVAILABLE else {"parsed_text":f"[Casting Error for {doc_path}]", "elements":elements, "original_metadata":current_original_metadata}
                
                parsed_docs_output_list.append(parsed_output) # type: ignore
                context.log.info(f"Successfully parsed (or created fallback for): {doc_path}")

        except Exception as e_parse_asset:
            context.log.error(f"Critical error during parsing asset for {doc_path}: {e_parse_asset}", exc_info=True)
            error_text = f"[Critical Parsing Exception for {doc_path}: {str(e_parse_asset)}]"
            if _PYDANTIC_AVAILABLE:
                elements = [NarrativeTextElement(text=error_text)]
                error_output = ParsedDocumentOutput(
                    parsed_text=error_text,
                    elements=elements, # type: ignore
                    original_metadata=current_original_metadata
                )
            else:
                elements = [{"element_type": "narrative_text", "text": error_text}]
                error_output = {
                    "parsed_text": error_text,
                    "elements": elements,
                    "original_metadata": current_original_metadata
                }
            parsed_docs_output_list.append(error_output) # type: ignore
            
    if parsed_docs_output_list:
        context.add_output_metadata(
            metadata={
                "num_documents_processed_for_parsing": len(raw_documents),
                "num_parsed_document_outputs_generated": len(parsed_docs_output_list),
            }
        )
    return parsed_docs_output_list

all_ingestion_assets = [load_documents_asset, parse_document_asset]