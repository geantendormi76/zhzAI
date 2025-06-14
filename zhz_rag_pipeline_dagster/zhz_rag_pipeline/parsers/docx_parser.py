# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/zhz_rag_pipeline/parsers/docx_parser.py

import os
from typing import List, Dict, Any, Optional, Union
import re

# --- 添加：为当前模块的 logger 进行基本配置 ---
import logging
logger = logging.getLogger(__name__)
if not logger.handlers: # 避免重复添加 handler (如果模块被多次导入)
    handler = logging.StreamHandler() # 输出到 stderr，通常会被 Dagster 捕获
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # 设置希望看到的最低日志级别 (INFO, DEBUG等)
    # logger.propagate = False # 可以考虑设置，防止日志向上传播到根logger导致重复打印，但通常 Dagster 会处理好
logger.info(f"Logger for {__name__} configured in docx_parser.py.") # 确认配置生效
# --- 结束添加 ---

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
    class Text: pass                  # type: ignore
    class NarrativeText: pass          # type: ignore
    class Title: pass                  # type: ignore
    class ListItem: pass               # type: ignore
    class Table: pass                  # type: ignore
    class UnstructuredImage: pass      # type: ignore
    class UnstructuredHeader: pass     # type: ignore
    class UnstructuredFooter: pass     # type: ignore
    class Address: pass                # type: ignore
    class EmailAddress: pass           # type: ignore
    class FigureCaption: pass          # type: ignore
    class UnstructuredPageBreak: pass  # type: ignore
    class CodeSnippet: pass            # type: ignore

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
    class TableElement(BaseModel): # <--- 修改此行
        element_type:str="table"; 
        markdown_representation:Optional[str]=None; 
        html_representation:Optional[str]=None; 
        text_representation:Optional[str]=None; # <--- 添加此字段
        caption:Optional[str]=None; 
        metadata: Optional[DocumentElementMetadata] = None
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
    unstructured_elements: List[UnstructuredElement], 
    doc_path_for_log: str # 添加一个参数用于日志记录
) -> List[Any]:
    custom_elements: List[Any] = []
    
    file_basename_for_log = os.path.basename(doc_path_for_log)

    # --- 使用 print 进行强制调试 ---
    logger.info(f"DOCX Parser ({file_basename_for_log}): _convert_unstructured_elements_to_custom received {len(unstructured_elements)} elements from unstructured.")
    if not unstructured_elements:
        logger.warning(f"DOCX Parser ({file_basename_for_log}): Unstructured returned an empty list of elements. No custom elements will be generated by this function initially.")
    
    if not _UNSTRUCTURED_AVAILABLE_DOCX:
        logger.warning(f"DOCX Parser ({file_basename_for_log}): Unstructured library is not available (should have been caught earlier).")
        # 作为回退，我们可以尝试将每个元素的文本提取为 NarrativeTextElement (如果 unstructured_elements 非空但 _UNSTRUCTURED_AVAILABLE_DOCX 意外为 False)
        for el_idx, el_fallback in enumerate(unstructured_elements):
            fallback_text = getattr(el_fallback, 'text', f"[Unstructured not fully available - Element {el_idx+1} in {file_basename_for_log}]").strip()
            if fallback_text:
                if _PYDANTIC_MODELS_AVAILABLE_DOCX:
                    custom_elements.append(NarrativeTextElement(text=fallback_text))
                else:
                    custom_elements.append({"element_type": "narrative_text", "text": fallback_text})
        return custom_elements

    for el_idx, el in enumerate(unstructured_elements):
        el_type_name = type(el).__name__
        el_id_str = getattr(el, 'id', 'N/A')
        el_text_preview = getattr(el, 'text', '')[:50].strip().replace('\n', ' ') if getattr(el, 'text', '') else "[NO TEXT]"
        # --- 修改日志级别 ---
        logger.debug( # <--- 从 info 修改为 debug
            f"DOCX Parser ({file_basename_for_log}): Processing unstructured element index {el_idx}, "
            f"Type: {el_type_name}, ID: {el_id_str}, Text Preview: '{el_text_preview}'"
        )
        
        # 打印 el.metadata.text_as_html 的预览（如果存在）
        html_preview_from_meta = getattr(el.metadata, 'text_as_html', None) if hasattr(el, 'metadata') else None
        if html_preview_from_meta:
            logger.debug( # <--- 从 info 修改为 debug
                f"  └─ ({file_basename_for_log}) Unstructured Element (idx {el_idx}, type {el_type_name}) has text_as_html (len: {len(html_preview_from_meta)}). Preview: {html_preview_from_meta[:70]}"
            )
        # --- 结束修改 ---
        
        element_metadata = _create_doc_element_metadata(el)
        el_text = el.text.strip() if hasattr(el, 'text') and el.text else ""
        custom_el: Optional[Any] = None

        if isinstance(el, Title):
            level = el.metadata.category_depth if hasattr(el.metadata, 'category_depth') and el.metadata.category_depth is not None else 1
            if el_text:
                logger.debug(f"DOCX Parser ({file_basename_for_log}): Element index {el_idx} is Title (level {level}).")
                if _PYDANTIC_MODELS_AVAILABLE_DOCX: custom_el = TitleElement(text=el_text, level=level, metadata=element_metadata)
                else: custom_el = {"element_type": "title", "text": el_text, "level": level, "metadata": element_metadata}
        
        elif isinstance(el, ListItem):
            level = el.metadata.category_depth if hasattr(el.metadata, 'category_depth') and el.metadata.category_depth is not None else 0
            # 尝试从元数据获取更精确的列表信息 (unstructured 可能提供)
            # item_number 和 ordered 的逻辑可以根据 unstructured 的实际输出来完善
            if el_text:
                logger.debug(f"DOCX Parser ({file_basename_for_log}): Element index {el_idx} is ListItem (level {level}).")
                if _PYDANTIC_MODELS_AVAILABLE_DOCX: custom_el = ListItemElement(text=el_text, level=level, metadata=element_metadata)
                else: custom_el = {"element_type": "list_item", "text": el_text, "level": level, "metadata": element_metadata}

        elif isinstance(el, Table):
            logger.info(f"DOCX Parser ({file_basename_for_log}): Element index {el_idx} IS an unstructured.documents.elements.Table object.")
            html_table = el.metadata.text_as_html if hasattr(el.metadata, 'text_as_html') else None
            
            if html_table:
                logger.info(f"DOCX Parser ({file_basename_for_log}): Found HTML table content for Table element (idx {el_idx}). Length: {len(html_table)}. Preview: {html_table[:150]}")
            else:
                logger.warning(f"DOCX Parser ({file_basename_for_log}): No HTML table content (el.metadata.text_as_html) found for Table element (idx {el_idx}) from unstructured. Element ID: {el.id if hasattr(el, 'id') else 'N/A'}")

            md_table = None
            if html_table and _MARKDOWNIFY_AVAILABLE:
                try: 
                    md_table = md(html_table)
                    logger.info(f"DOCX Parser ({file_basename_for_log}): Successfully converted HTML table (idx {el_idx}) to Markdown. MD Length: {len(md_table) if md_table else 0}")
                except Exception as e_md: 
                    logger.warning(f"DOCX Parser ({file_basename_for_log}): Failed to convert HTML table (idx {el_idx}) to Markdown: {e_md}. HTML: {html_table[:100]}")
            
            raw_table_text_fallback = el.text.strip() if hasattr(el, 'text') and el.text else None
            caption_text = None
            if hasattr(el.metadata, 'table_captions') and el.metadata.table_captions:
                    caption_obj = el.metadata.table_captions[0]
                    if hasattr(caption_obj, 'text'):
                            caption_text = caption_obj.text
            
            if not caption_text and hasattr(el.metadata, 'filename'): # Redundant if filename is always doc_path_for_log
                    caption_text = f"Table from {file_basename_for_log}" # Use basename
            final_caption = caption_text if caption_text else "Table"

            final_md_table = md_table
            final_html_table = html_table
            final_text_representation = None

            if not final_md_table and not final_html_table and raw_table_text_fallback:
                logger.info(f"DOCX Parser ({file_basename_for_log}): Table (idx {el_idx}) has no HTML/MD rep, but has raw text from unstructured: '{raw_table_text_fallback[:100]}...' Using it as text_representation.")
                final_text_representation = raw_table_text_fallback
            elif not final_md_table and not final_html_table and not raw_table_text_fallback:
                    logger.warning(f"DOCX Parser ({file_basename_for_log}): Table (idx {el_idx}) has no HTML, Markdown, or raw text representation from unstructured.")


            if _PYDANTIC_MODELS_AVAILABLE_DOCX: 
                custom_el = TableElement(
                    markdown_representation=final_md_table, 
                    html_representation=final_html_table, 
                    text_representation=final_text_representation,
                    caption=final_caption, 
                    metadata=element_metadata
                )
            else: 
                custom_el = {
                    "element_type": "table", 
                    "markdown_representation": final_md_table, 
                    "html_representation": final_html_table, 
                    "text_representation": final_text_representation,
                    "caption": final_caption, 
                    "metadata": element_metadata
                }
        
        elif isinstance(el, (NarrativeText, Text, Address, EmailAddress, FigureCaption)):
            if el_text:
                logger.debug(f"DOCX Parser ({file_basename_for_log}): Element index {el_idx} is NarrativeText/Text like.")
                if _PYDANTIC_MODELS_AVAILABLE_DOCX: custom_el = NarrativeTextElement(text=el_text, metadata=element_metadata)
                else: custom_el = {"element_type": "narrative_text", "text": el_text, "metadata": element_metadata}
        
        elif isinstance(el, UnstructuredHeader):
            if el_text:
                logger.debug(f"DOCX Parser ({file_basename_for_log}): Element index {el_idx} is Header.")
                if _PYDANTIC_MODELS_AVAILABLE_DOCX: custom_el = HeaderElement(text=el_text, metadata=element_metadata)
                else: custom_el = {"element_type": "header", "text": el_text, "metadata": element_metadata}
        
        elif isinstance(el, UnstructuredFooter):
            if el_text:
                logger.debug(f"DOCX Parser ({file_basename_for_log}): Element index {el_idx} is Footer.")
                if _PYDANTIC_MODELS_AVAILABLE_DOCX: custom_el = FooterElement(text=el_text, metadata=element_metadata)
                else: custom_el = {"element_type": "footer", "text": el_text, "metadata": element_metadata}

        elif isinstance(el, UnstructuredImage):
            alt_text = el_text if el_text else (el.metadata.filename if hasattr(el.metadata, 'filename') else "Image")
            logger.debug(f"DOCX Parser ({file_basename_for_log}): Element index {el_idx} is Image. Alt text: {alt_text}")
            if _PYDANTIC_MODELS_AVAILABLE_DOCX: custom_el = ImageElement(alt_text=alt_text, metadata=element_metadata)
            else: custom_el = {"element_type": "image", "alt_text": alt_text, "metadata": element_metadata}

        elif isinstance(el, UnstructuredPageBreak):
            logger.debug(f"DOCX Parser ({file_basename_for_log}): Element index {el_idx} is PageBreak.")
            if _PYDANTIC_MODELS_AVAILABLE_DOCX: custom_el = PageBreakElement(metadata=element_metadata)
            else: custom_el = {"element_type": "page_break", "metadata": element_metadata}
        
        elif isinstance(el, CodeSnippet):
            if el_text:
                logger.debug(f"DOCX Parser ({file_basename_for_log}): Element index {el_idx} is CodeSnippet.")
                if _PYDANTIC_MODELS_AVAILABLE_DOCX: custom_el = CodeBlockElement(code=el_text, metadata=element_metadata) # language can be inferred later if needed
                else: custom_el = {"element_type": "code_block", "code": el_text, "metadata": element_metadata}

        else: 
            # This is the catch-all for any other Unstructured element type
            # or if an element doesn't have text but we still want to represent it (though usually skipped if no text)
            if el_text: # Only create an element if there's text
                logger.warning(f"DOCX Parser ({file_basename_for_log}): Unhandled Unstructured element type: {el_type_name} at index {el_idx}. Treating as NarrativeText. Text: {el_text[:50]}")
                if _PYDANTIC_MODELS_AVAILABLE_DOCX: custom_el = NarrativeTextElement(text=el_text, metadata=element_metadata)
                else: custom_el = {"element_type": "narrative_text", "text": el_text, "_unstructured_type": el_type_name, "metadata": element_metadata}
            elif el_type_name != "CompositeElement": # CompositeElement often has no direct text but contains other elements
                    logger.debug(f"DOCX Parser ({file_basename_for_log}): Skipping Unstructured element type: {el_type_name} at index {el_idx} due to no text content.")

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
    file_basename_for_log = os.path.basename(file_path)
    
    # --- 使用 print 进行强制调试 ---
    logger.info(f"DOCX Parser: Attempting to parse DOCX file: {file_basename_for_log} using Unstructured")
    
    if not _UNSTRUCTURED_AVAILABLE_DOCX:
        logger.error(f"DOCX Parser ({file_basename_for_log}): Unstructured library is not available. DOCX parsing cannot proceed.")
        return None
    try:
        unstructured_elements = partition_docx(
            filename=file_path, 
            strategy="fast", 
            infer_table_structure=True,
        )
        logger.info(f"DOCX Parser ({file_basename_for_log}): Unstructured partitioned DOCX. Found {len(unstructured_elements)} raw elements from partition_docx.")

        custom_elements = _convert_unstructured_elements_to_custom(unstructured_elements, file_path)
        logger.info(f"DOCX Parser ({file_basename_for_log}): _convert_unstructured_elements_to_custom created {len(custom_elements)} custom elements.")
        
        linear_text = _generate_linear_text_from_custom_elements(custom_elements)
        logger.info(f"DOCX Parser ({file_basename_for_log}): Generated linear text (len: {len(linear_text)}). Preview: {linear_text[:100].replace(chr(10), ' ')}")

        if not custom_elements and not linear_text.strip() and unstructured_elements:
            logger.warning(f"DOCX Parser ({file_basename_for_log}): partition_docx returned {len(unstructured_elements)} elements, "
                            "but no custom elements or linear text were generated. This might indicate all elements were skipped "
                            "or had no text content suitable for conversion.")
        elif not custom_elements and not linear_text.strip() and not unstructured_elements:
                logger.warning(f"DOCX Parser ({file_basename_for_log}): partition_docx returned 0 elements, "
                                "and no custom elements or linear text were generated.")

        if _PYDANTIC_MODELS_AVAILABLE_DOCX:
            return ParsedDocumentOutput(
                parsed_text=linear_text,
                elements=custom_elements, 
                original_metadata=original_metadata
            )
        else:
            return {
                "parsed_text": linear_text,
                "elements": custom_elements,
                "original_metadata": original_metadata
            }
            
    except FileNotFoundError:
        logger.error(f"DOCX Parser ({file_basename_for_log}): File not found: {file_path}")
        return None
    except ImportError as ie:
        logger.error(f"DOCX Parser ({file_basename_for_log}): ImportError during DOCX parsing with Unstructured: {ie}.")
        return None
    except Exception as e:
        logger.error(f"DOCX Parser ({file_basename_for_log}): Critical error parsing DOCX file: {e}", exc_info=True)
        error_message = f"[ERROR PARSING DOCX: {file_basename_for_log} - {type(e).__name__}: {str(e)}]"
        if _PYDANTIC_MODELS_AVAILABLE_DOCX:
            return ParsedDocumentOutput(
                parsed_text=error_message,
                elements=[],
                original_metadata=original_metadata
            )
        else:
            return {
                "parsed_text": error_message,
                "elements": [],
                "original_metadata": original_metadata
            }