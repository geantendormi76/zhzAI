# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/zhz_rag_pipeline/parsers/pdf_parser.py
import os
import logging
from typing import List, Dict, Any, Optional, Union
import re

try:
    import fitz  # PyMuPDF
    _PYMUPDF_AVAILABLE = True
    logging.info("Successfully imported PyMuPDF (fitz) for PDF parsing.")
except ImportError:
    logging.error("PyMuPDF (fitz) not found. PDF parsing will not be available.")
    _PYMUPDF_AVAILABLE = False
    # 占位符，以防 fitz 未安装时代码尝试引用它
    class fitz: 
        class Document: pass
        class Page: pass
        class Rect: pass 

_PYDANTIC_MODELS_AVAILABLE_PDF = False
try:
    from ..pydantic_models_dagster import (
        ParsedDocumentOutput, DocumentElementType, NarrativeTextElement,
        DocumentElementMetadata, PageBreakElement
    )
    _PYDANTIC_MODELS_AVAILABLE_PDF = True
except ImportError:
    class BaseModel: pass
    class DocumentElementMetadata(BaseModel): page_number: Optional[int] = None
    class ParsedDocumentOutput(BaseModel): parsed_text: str; elements: list; original_metadata: dict; summary: Optional[str] = None
    class NarrativeTextElement(BaseModel): element_type:str="narrative_text"; text:str; metadata: Optional[DocumentElementMetadata] = None
    class PageBreakElement(BaseModel): element_type:str="page_break"; metadata: Optional[DocumentElementMetadata] = None
    DocumentElementType = Any

logger = logging.getLogger(__name__)

def _create_pdf_element_metadata(page_number: Optional[int] = None) -> Optional[Union[DocumentElementMetadata, Dict[str, Any]]]:
    meta_data_dict: Dict[str, Any] = {}
    if page_number is not None:
        meta_data_dict['page_number'] = page_number
    
    if not meta_data_dict:
        return None

    if _PYDANTIC_MODELS_AVAILABLE_PDF:
        return DocumentElementMetadata(**meta_data_dict)
    else:
        return meta_data_dict

def parse_pdf_to_structured_output(
    file_path: str, 
    original_metadata: Dict[str, Any]
) -> Optional[Union[ParsedDocumentOutput, Dict[str, Any]]]:
    logger.info(f"Attempting to parse PDF file: {file_path} using PyMuPDF (fitz)")
    if not _PYMUPDF_AVAILABLE:
        logger.error("PyMuPDF (fitz) is not available. PDF parsing cannot proceed.")
        return None

    elements: List[Any] = []
    full_text_parts: List[str] = []

    try:
        doc = fitz.open(file_path)
        logger.info(f"PyMuPDF opened PDF. Pages: {doc.page_count}")

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            
            # 使用 "dict" 模式提取文本块，并按阅读顺序排序
            page_content_blocks = page.get_text("dict", sort=True).get("blocks", [])
            
            page_text_collected = []
            
            if page_content_blocks:
                for block in page_content_blocks:
                    if block['type'] == 0: # 0 表示文本块
                        block_text_lines = []
                        for line in block.get("lines", []):
                            line_content = "".join([span.get("text", "") for span in line.get("spans", [])])
                            block_text_lines.append(line_content)
                        block_text_content = "\n".join(block_text_lines).strip()
                        if block_text_content:
                            page_text_collected.append(block_text_content)
            
            if page_text_collected:
                # 将整页的文本作为一个 NarrativeTextElement
                page_full_text = "\n\n".join(page_text_collected) # 用双换行符分隔来自不同块的文本
                element_metadata = _create_pdf_element_metadata(page_number=page_num + 1)
                if _PYDANTIC_MODELS_AVAILABLE_PDF:
                    elements.append(NarrativeTextElement(text=page_full_text, metadata=element_metadata)) # type: ignore
                else:
                    elements.append({"element_type": "narrative_text", "text": page_full_text, "metadata": element_metadata})
                full_text_parts.append(page_full_text)
            
            # 在每页（除了最后一页）之后添加一个 PageBreakElement
            if page_num < doc.page_count - 1:
                if _PYDANTIC_MODELS_AVAILABLE_PDF:
                    elements.append(PageBreakElement(metadata=_create_pdf_element_metadata(page_number=page_num + 1))) # type: ignore
                else:
                    elements.append({"element_type": "page_break", "metadata": _create_pdf_element_metadata(page_number=page_num + 1)})


        doc.close()
        
        linear_text = "\n\n--- Page Break ---\n\n".join(full_text_parts) # 用特殊标记分隔页面文本
        linear_text = re.sub(r'\n{3,}', '\n\n', linear_text).strip()

        if _PYDANTIC_MODELS_AVAILABLE_PDF:
            return ParsedDocumentOutput(
                parsed_text=linear_text,
                elements=elements, # type: ignore
                original_metadata=original_metadata
            )
        else:
            return {
                "parsed_text": linear_text,
                "elements": elements,
                "original_metadata": original_metadata
            }

    except FileNotFoundError:
        logger.error(f"PDF file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error parsing PDF file {file_path} with PyMuPDF: {e}", exc_info=True)
        return None