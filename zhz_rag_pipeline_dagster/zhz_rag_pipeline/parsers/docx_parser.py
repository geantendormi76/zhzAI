# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/zhz_rag_pipeline/parsers/docx_parser.py
import os
import logging
from typing import List, Dict, Any, Optional, Union
import re

# --- 依赖导入与可用性检查 ---
try:
    from unstructured.partition.docx import partition_docx
    from unstructured.documents.elements import (
        Element as UnstructuredElement,
        Text, 
        NarrativeText,
        Title,
        ListItem,
        Table,
        Image as UnstructuredImage, 
        Header as UnstructuredHeader, 
        Footer as UnstructuredFooter, 
        Address,
        EmailAddress,
        FigureCaption,
        PageBreak as UnstructuredPageBreak, 
        CodeSnippet
    )
    _UNSTRUCTURED_AVAILABLE_DOCX = True
    logging.info("Successfully imported Unstructured for DOCX parsing.")
except ImportError as e_unstructured:
    logging.error(f"Failed to import Unstructured for DOCX: {e_unstructured}. DOCX parsing will have limited functionality.")
    _UNSTRUCTURED_AVAILABLE_DOCX = False
    # 创建占位符类以避免后续 NameError
    class UnstructuredElement: pass
    class Text: pass                 # type: ignore
    class NarrativeText: pass        # type: ignore
    class Title: pass                # type: ignore
    class ListItem: pass             # type: ignore
    class Table: pass                # type: ignore
    class UnstructuredImage: pass    # type: ignore
    class UnstructuredHeader: pass   # type: ignore
    class UnstructuredFooter: pass   # type: ignore
    class Address: pass              # type: ignore
    class EmailAddress: pass         # type: ignore
    class FigureCaption: pass        # type: ignore
    class UnstructuredPageBreak: pass# type: ignore
    class CodeSnippet: pass          # type: ignore

try:
    from markdownify import markdownify as md # type: ignore
    _MARKDOWNIFY_AVAILABLE = True
except ImportError:
    logging.warning("markdownify library not found. HTML table to Markdown conversion will be skipped.")
    _MARKDOWNIFY_AVAILABLE = False
    def md(html_content: str) -> str: # Fallback
        return f"[Markdownify not available. HTML content: {html_content[:100]}...]"

_PYDANTIC_MODELS_AVAILABLE_DOCX = False
try:
    from ..pydantic_models_dagster import (
        ParsedDocumentOutput, DocumentElementType, TitleElement, NarrativeTextElement,
        ListItemElement, TableElement, CodeBlockElement, PageBreakElement, ImageElement,
        HeaderElement, FooterElement, DocumentElementMetadata
    )
    _PYDANTIC_MODELS_AVAILABLE_DOCX = True
except ImportError:
    class BaseModel: pass
    class DocumentElementMetadata(BaseModel): page_number: Optional[int] = None
    class ParsedDocumentOutput(BaseModel): parsed_text: str; elements: list; original_metadata: dict; summary: Optional[str] = None
    class TitleElement(BaseModel): element_type:str="title"; text:str; level:int; metadata: Optional[DocumentElementMetadata] = None
    class NarrativeTextElement(BaseModel): element_type:str="narrative_text"; text:str; metadata: Optional[DocumentElementMetadata] = None
    class ListItemElement(BaseModel): element_type:str="list_item"; text:str; level:int=0; ordered:bool=False; item_number:Optional[Union[int, str]]=None; metadata: Optional[DocumentElementMetadata] = None
    class TableElement(BaseModel): element_type:str="table"; markdown_representation:Optional[str]=None; html_representation:Optional[str]=None; caption:Optional[str]=None; metadata: Optional[DocumentElementMetadata] = None
    class CodeBlockElement(BaseModel): element_type:str="code_block"; code:str; language:Optional[str]=None; metadata: Optional[DocumentElementMetadata] = None
    class PageBreakElement(BaseModel): element_type:str="page_break"; metadata: Optional[DocumentElementMetadata] = None
    class ImageElement(BaseModel): element_type:str="image"; alt_text:Optional[str]=None; caption:Optional[str]=None; metadata: Optional[DocumentElementMetadata] = None
    class HeaderElement(BaseModel): element_type:str="header"; text:str; metadata: Optional[DocumentElementMetadata] = None
    class FooterElement(BaseModel): element_type:str="footer"; text:str; metadata: Optional[DocumentElementMetadata] = None
    DocumentElementType = Any

logger = logging.getLogger(__name__)

# --- 辅助函数 ---
def _create_doc_element_metadata(unstructured_element: UnstructuredElement) -> Optional[Union[DocumentElementMetadata, Dict[str, Any]]]:
    if not hasattr(unstructured_element, 'metadata'):
        return None
        
    meta_data_dict: Dict[str, Any] = {}
    if hasattr(unstructured_element.metadata, 'page_number') and unstructured_element.metadata.page_number is not None:
        meta_data_dict['page_number'] = unstructured_element.metadata.page_number
    
    if hasattr(unstructured_element.metadata, 'filename'):
        meta_data_dict['source_filename'] = unstructured_element.metadata.filename
    if hasattr(unstructured_element.metadata, 'filetype'):
        meta_data_dict['source_filetype'] = unstructured_element.metadata.filetype

    if not meta_data_dict:
        return None

    if _PYDANTIC_MODELS_AVAILABLE_DOCX:
        return DocumentElementMetadata(**meta_data_dict)
    else:
        return meta_data_dict

def _convert_unstructured_elements_to_custom(
    unstructured_elements: List[UnstructuredElement]
) -> List[Any]:
    custom_elements: List[Any] = []
    
    if not _UNSTRUCTURED_AVAILABLE_DOCX:
        logger.warning("Unstructured library is not available. Cannot perform detailed element conversion for DOCX.")
        # 作为回退，我们可以尝试将每个元素的文本提取为 NarrativeTextElement
        for el_idx, el_fallback in enumerate(unstructured_elements):
            fallback_text = getattr(el_fallback, 'text', f"[Unstructured not available - Element {el_idx+1}]").strip()
            if fallback_text:
                if _PYDANTIC_MODELS_AVAILABLE_DOCX:
                    custom_elements.append(NarrativeTextElement(text=fallback_text))
                else:
                    custom_elements.append({"element_type": "narrative_text", "text": fallback_text})
        return custom_elements

    # 如果 Unstructured 可用，则执行之前的详细转换逻辑
    for el in unstructured_elements:
        element_metadata = _create_doc_element_metadata(el)
        el_text = el.text.strip() if hasattr(el, 'text') and el.text else ""
        custom_el: Optional[Any] = None

        # --- 这里的 isinstance 检查现在是安全的，因为 _UNSTRUCTURED_AVAILABLE_DOCX 为 True ---
        if isinstance(el, Title):
            level = el.metadata.category_depth if hasattr(el.metadata, 'category_depth') and el.metadata.category_depth is not None else 1
            if el_text:
                if _PYDANTIC_MODELS_AVAILABLE_DOCX: custom_el = TitleElement(text=el_text, level=level, metadata=element_metadata) # type: ignore
                else: custom_el = {"element_type": "title", "text": el_text, "level": level, "metadata": element_metadata}
        
        elif isinstance(el, ListItem):
            level = el.metadata.category_depth if hasattr(el.metadata, 'category_depth') and el.metadata.category_depth is not None else 0
            if el_text:
                if _PYDANTIC_MODELS_AVAILABLE_DOCX: custom_el = ListItemElement(text=el_text, level=level, metadata=element_metadata) # type: ignore
                else: custom_el = {"element_type": "list_item", "text": el_text, "level": level, "metadata": element_metadata}

        elif isinstance(el, Table):
            html_table = el.metadata.text_as_html if hasattr(el.metadata, 'text_as_html') else None
            md_table = None
            if html_table and _MARKDOWNIFY_AVAILABLE:
                try: md_table = md(html_table)
                except Exception as e_md: logger.warning(f"Failed to convert HTML table to Markdown: {e_md}. HTML: {html_table[:100]}")
            
            caption = el.metadata.filename if hasattr(el.metadata, 'filename') else "Table"
            if _PYDANTIC_MODELS_AVAILABLE_DOCX: custom_el = TableElement(markdown_representation=md_table, html_representation=html_table, caption=caption, metadata=element_metadata) # type: ignore
            else: custom_el = {"element_type": "table", "markdown_representation": md_table, "html_representation": html_table, "caption": caption, "metadata": element_metadata}

        elif isinstance(el, (NarrativeText, Text, Address, EmailAddress, FigureCaption)):
            if el_text:
                if _PYDANTIC_MODELS_AVAILABLE_DOCX: custom_el = NarrativeTextElement(text=el_text, metadata=element_metadata) # type: ignore
                else: custom_el = {"element_type": "narrative_text", "text": el_text, "metadata": element_metadata}
        
        elif isinstance(el, UnstructuredHeader):
            if el_text:
                if _PYDANTIC_MODELS_AVAILABLE_DOCX: custom_el = HeaderElement(text=el_text, metadata=element_metadata) # type: ignore
                else: custom_el = {"element_type": "header", "text": el_text, "metadata": element_metadata}
        
        elif isinstance(el, UnstructuredFooter):
            if el_text:
                if _PYDANTIC_MODELS_AVAILABLE_DOCX: custom_el = FooterElement(text=el_text, metadata=element_metadata) # type: ignore
                else: custom_el = {"element_type": "footer", "text": el_text, "metadata": element_metadata}

        elif isinstance(el, UnstructuredImage):
            alt_text = el_text if el_text else (el.metadata.filename if hasattr(el.metadata, 'filename') else "Image")
            if _PYDANTIC_MODELS_AVAILABLE_DOCX: custom_el = ImageElement(alt_text=alt_text, metadata=element_metadata) # type: ignore
            else: custom_el = {"element_type": "image", "alt_text": alt_text, "metadata": element_metadata}

        elif isinstance(el, UnstructuredPageBreak):
            if _PYDANTIC_MODELS_AVAILABLE_DOCX: custom_el = PageBreakElement(metadata=element_metadata) # type: ignore
            else: custom_el = {"element_type": "page_break", "metadata": element_metadata}
        
        elif isinstance(el, CodeSnippet):
            if el_text:
                if _PYDANTIC_MODELS_AVAILABLE_DOCX: custom_el = CodeBlockElement(code=el_text, metadata=element_metadata) # type: ignore
                else: custom_el = {"element_type": "code_block", "code": el_text, "metadata": element_metadata}

        else: 
            if el_text:
                logger.debug(f"Unhandled Unstructured element type: {type(el).__name__}. Treating as NarrativeText. Text: {el_text[:50]}")
                if _PYDANTIC_MODELS_AVAILABLE_DOCX: custom_el = NarrativeTextElement(text=el_text, metadata=element_metadata) # type: ignore
                else: custom_el = {"element_type": "narrative_text", "text": el_text, "_unstructured_type": type(el).__name__, "metadata": element_metadata}
        
        if custom_el:
            custom_elements.append(custom_el)
    return custom_elements

def _generate_linear_text_from_custom_elements(elements: List[Any]) -> str:
    text_parts = []
    for el_data_any in elements:
        el_data: Dict[str, Any] = {}
        if _PYDANTIC_MODELS_AVAILABLE_DOCX and hasattr(el_data_any, 'model_dump'):
            el_data = el_data_any.model_dump(exclude_none=True)
        elif isinstance(el_data_any, dict):
            el_data = el_data_any
        else:
            continue

        el_type = el_data.get("element_type")
        text_content = el_data.get("text", "")
        
        current_element_text = ""
        if el_type == "title":
            current_element_text = f"\n{'#' * el_data.get('level',1)} {text_content}\n"
        elif el_type == "narrative_text":
            current_element_text = text_content + "\n"
        elif el_type == "list_item":
            prefix = f"{el_data.get('item_number', '')}. " if el_data.get('ordered') and el_data.get('item_number') else "- "
            indent = "  " * el_data.get('level', 0)
            current_element_text = f"{indent}{prefix}{text_content}\n"
        elif el_type == "table":
            caption = el_data.get('caption', 'Unnamed Table')
            md_repr = el_data.get('markdown_representation')
            if md_repr: current_element_text = f"\n[Table: {caption}]\n{md_repr}\n"
            elif el_data.get('html_representation'): current_element_text = f"\n[Table (HTML): {caption}]\n{el_data.get('html_representation')[:200]}...\n"
        elif el_type == "code_block":
            lang = el_data.get('language', "") or ""
            code_content = el_data.get('code', "")
            current_element_text = f"\n```{lang}\n{code_content}\n```\n"
        elif el_type == "page_break":
            current_element_text = "\n---\n"
        elif el_type == "header" or el_type == "footer":
            current_element_text = f"\n[{el_type.capitalize()}]: {text_content}\n"
        elif el_type == "image":
            alt_text = el_data.get('alt_text', 'Image')
            current_element_text = f"\n[Image: {alt_text}]\n"
        
        if current_element_text:
            text_parts.append(current_element_text)

    full_text = "".join(text_parts)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text).strip() # Clean up excessive newlines
    return full_text

# --- 主解析函数 ---
def parse_docx_to_structured_output(
    file_path: str, 
    original_metadata: Dict[str, Any]
) -> Optional[Union[ParsedDocumentOutput, Dict[str, Any]]]:
    logger.info(f"Attempting to parse DOCX file: {file_path} using Unstructured")
    if not _UNSTRUCTURED_AVAILABLE_DOCX:
        logger.error("Unstructured library is not available. DOCX parsing cannot proceed.")
        return None
    try:
        unstructured_elements = partition_docx(
            filename=file_path, 
            strategy="fast", 
            infer_table_structure=True
        )
        logger.info(f"Unstructured partitioned DOCX. Found {len(unstructured_elements)} elements.")

        custom_elements = _convert_unstructured_elements_to_custom(unstructured_elements)
        logger.info(f"Converted Unstructured elements to {len(custom_elements)} custom elements.")
        
        linear_text = _generate_linear_text_from_custom_elements(custom_elements)
        logger.info(f"Generated linear text from custom elements (length: {len(linear_text)}). Preview: {linear_text[:200]}")

        if _PYDANTIC_MODELS_AVAILABLE_DOCX:
            return ParsedDocumentOutput(
                parsed_text=linear_text,
                elements=custom_elements, # type: ignore
                original_metadata=original_metadata
            )
        else:
            return {
                "parsed_text": linear_text,
                "elements": custom_elements,
                "original_metadata": original_metadata
            }
    except FileNotFoundError:
        logger.error(f"DOCX file not found: {file_path}")
        return None
    except ImportError as ie:
        logger.error(f"ImportError during DOCX parsing with Unstructured: {ie}.")
        return None
    except Exception as e:
        logger.error(f"Error parsing DOCX file {file_path} with Unstructured: {e}", exc_info=True)
        return None