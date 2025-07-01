# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/zhz_rag_pipeline/parsers/md_parser.py
import os
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
import logging
from typing import List, Dict, Any, Optional, Union
import re

# --- Pydantic 模型导入和占位符定义 ---
_PARSER_PYDANTIC_AVAILABLE = False
try:
    from ..pydantic_models_dagster import (
        ParsedDocumentOutput, DocumentElementType, TitleElement, NarrativeTextElement,
        ListItemElement, TableElement, CodeBlockElement, PageBreakElement,
        DocumentElementMetadata
    )
    _PARSER_PYDANTIC_AVAILABLE = True
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
    DocumentElementType = Any

logger = logging.getLogger(__name__)

# --- 辅助函数 ---
def _get_node_text_content(node: SyntaxTreeNode, exclude_lists_and_tables: bool = False) -> str:
    if node.type == "text":
        return node.content
    if node.type == "softbreak":
        return " "
    if node.type == "hardbreak":
        return "\n"
    if node.type == "code_inline":
        return f"`{node.content}`"
    
    if exclude_lists_and_tables and node.type in ["bullet_list", "ordered_list", "table"]:
        return ""

    content = ""
    if node.children:
        for child in node.children:
            content += _get_node_text_content(child, exclude_lists_and_tables)
    return content

def _convert_table_node_to_markdown(table_node: SyntaxTreeNode) -> str:
    md_rows = []
    
    thead_node = next((child for child in table_node.children if child.type == 'thead'), None)
    tbody_node = next((child for child in table_node.children if child.type == 'tbody'), None)

    header_texts = []
    if thead_node:
        tr_node_header = next((child for child in thead_node.children if child.type == 'tr'), None)
        if tr_node_header:
            header_texts = [_get_node_text_content(cell).strip() for cell in tr_node_header.children if cell.type == 'th']
    elif tbody_node: 
        first_row_in_tbody = next((child for child in tbody_node.children if child.type == 'tr'), None)
        if first_row_in_tbody and all(cell.type == 'th' for cell in first_row_in_tbody.children):
             header_texts = [_get_node_text_content(cell).strip() for cell in first_row_in_tbody.children]

    if header_texts:
        md_rows.append("| " + " | ".join(header_texts) + " |")
        md_rows.append("| " + " | ".join(["---"] * len(header_texts)) + " |")

    rows_container = tbody_node if tbody_node else table_node 
    
    first_row_in_container_is_header = False
    if not header_texts and rows_container: # 只有在没有thead且容器存在时，才检查第一行是否是表头
        first_tr = next((child for child in rows_container.children if child.type == 'tr'), None)
        if first_tr and all(cell.type == 'th' for cell in first_tr.children):
            # 如果第一行全是th，作为表头处理
            header_texts_from_body = [_get_node_text_content(cell).strip() for cell in first_tr.children]
            if header_texts_from_body:
                md_rows.append("| " + " | ".join(header_texts_from_body) + " |")
                md_rows.append("| " + " | ".join(["---"] * len(header_texts_from_body)) + " |")
                first_row_in_container_is_header = True

    if rows_container: # 确保 rows_container 存在
        for row_idx, tr_node in enumerate(child for child in rows_container.children if child.type == 'tr'):
            # 如果第一行已经被作为表头处理了，则跳过它
            if first_row_in_container_is_header and row_idx == 0:
                continue
            
            # 如果已经通过 thead 处理了表头，那么 tbody/table 下的所有 tr 都应视为数据行
            # 如果没有通过 thead 处理表头，并且当前行也不是被推断为表头的 tbody 第一行，那么它也是数据行
            cell_texts = [_get_node_text_content(cell).strip() for cell in tr_node.children if cell.type == 'td']
            if cell_texts or len(tr_node.children) > 0 : 
                md_rows.append("| " + " | ".join(cell_texts) + " |")
            
    return "\n".join(md_rows)

# --- 主转换函数 ---
def _convert_md_tree_to_elements(root_node: SyntaxTreeNode) -> List[Any]: 
    elements: List[Any] = []
    
    def _process_node_recursive(node: SyntaxTreeNode, current_semantic_level: int = 0, list_ctx: Optional[Dict] = None):
        nonlocal elements
        current_metadata = None 

        node_type = node.type
        
        if node_type == "heading":
            level = int(node.tag[1:])
            text = _get_node_text_content(node).strip()
            if text or node.children: 
                if _PARSER_PYDANTIC_AVAILABLE: elements.append(TitleElement(text=text, level=level, metadata=current_metadata))
                else: elements.append({"element_type": "title", "text": text, "level": level, "metadata": current_metadata})
        
        elif node_type == "paragraph":
            text = _get_node_text_content(node).strip()
            if text:
                if _PARSER_PYDANTIC_AVAILABLE: elements.append(NarrativeTextElement(text=text, metadata=current_metadata))
                else: elements.append({"element_type": "narrative_text", "text": text, "metadata": current_metadata})

        elif node_type == "bullet_list" or node_type == "ordered_list":
            is_ordered_list = (node_type == "ordered_list")
            child_list_ctx = {
                "ordered": is_ordered_list,
                "start_num": int(node.attrs.get("start", 1)) if node.attrs and is_ordered_list else 1,
                "item_idx_in_list": 0 
            }
            for child_node in node.children:
                if child_node.type == "list_item":
                    _process_node_recursive(child_node, current_semantic_level + 1, child_list_ctx)
        
        elif node_type == "list_item":
            item_text = _get_node_text_content(node, exclude_lists_and_tables=True).strip()
            
            if item_text and list_ctx: 
                display_level = current_semantic_level - 1 
                item_number_str = None
                if list_ctx["ordered"]:
                    item_number_str = str(list_ctx["start_num"] + list_ctx["item_idx_in_list"])
                    list_ctx["item_idx_in_list"] += 1
                else: 
                    item_number_str = node.markup if node.markup else "-" 

                if _PARSER_PYDANTIC_AVAILABLE:
                    elements.append(ListItemElement(
                        text=item_text, level=display_level, 
                        ordered=list_ctx["ordered"], 
                        item_number=item_number_str, metadata=current_metadata
                    ))
                else:
                    elements.append({
                        "element_type": "list_item", "text": item_text, 
                        "level": display_level, "ordered": list_ctx["ordered"], 
                        "item_number": item_number_str, "metadata": current_metadata
                    })
            
            for child_node in node.children:
                if child_node.type in ["bullet_list", "ordered_list"]:
                    _process_node_recursive(child_node, current_semantic_level, None) # Pass current_semantic_level for nested list

        elif node_type == "table":
            md_table_representation = _convert_table_node_to_markdown(node)
            if md_table_representation:
                if _PARSER_PYDANTIC_AVAILABLE: elements.append(TableElement(markdown_representation=md_table_representation, metadata=current_metadata))
                else: elements.append({"element_type": "table", "markdown_representation": md_table_representation, "metadata": current_metadata})

        elif node_type == "fence" or node_type == "code_block":
            code_content = node.content.strip('\n') 
            lang = node.info.strip() if node.info else None
            if _PARSER_PYDANTIC_AVAILABLE: elements.append(CodeBlockElement(code=code_content, language=lang, metadata=current_metadata))
            else: elements.append({"element_type": "code_block", "code": code_content, "language": lang, "metadata": current_metadata})

        elif node_type == "hr":
            if _PARSER_PYDANTIC_AVAILABLE: elements.append(PageBreakElement(metadata=current_metadata))
            else: elements.append({"element_type": "page_break", "metadata": current_metadata})

        elif node_type == "blockquote":
            text = _get_node_text_content(node).strip()
            if text:
                if _PARSER_PYDANTIC_AVAILABLE: elements.append(NarrativeTextElement(text=text, metadata=current_metadata))
                else: elements.append({"element_type": "narrative_text", "text": text, "_is_blockquote": True, "metadata": current_metadata})
        
        elif node.children and node_type not in ["list_item", "heading", "paragraph", "table", "fence", "code_block", "blockquote", "hr", "bullet_list", "ordered_list"]: # Avoid re-processing children of already handled types
             for child in node.children:
                _process_node_recursive(child, current_semantic_level, list_ctx) # Pass context along

    _process_node_recursive(root_node) 
    return elements

def _generate_parsed_text_from_elements_internal(elements: List[Any]) -> str:
    text_parts = []
    for el_data_any in elements:
        el_data = {}
        if _PARSER_PYDANTIC_AVAILABLE and hasattr(el_data_any, 'model_dump'): el_data = el_data_any.model_dump()
        elif isinstance(el_data_any, dict): el_data = el_data_any
        else: continue
        el_type = el_data.get("element_type")
        if el_type == "title": text_parts.append(f"\n{'#' * el_data.get('level',1)} {el_data.get('text','')}\n")
        elif el_type == "narrative_text": text_parts.append(el_data.get('text','') + "\n")
        elif el_type == "list_item":
            item_num_display = str(el_data.get('item_number','-')) 
            prefix = f"{item_num_display}. " if el_data.get('ordered') else f"{item_num_display} "
            indent = "  " * el_data.get('level',0)
            text_parts.append(f"{indent}{prefix}{el_data.get('text','')}\n")
        elif el_type == "table":
            caption_text = str(el_data.get('caption')) if el_data.get('caption') is not None else 'Unnamed Table'
            if el_data.get('markdown_representation'): 
                text_parts.append(f"\n[Table: {caption_text}]\n{el_data['markdown_representation']}\n")
            elif el_data.get('text_representation'): 
                text_parts.append(f"\n[Table: {caption_text}]\n{el_data['text_representation']}\n")
        elif el_type == "code_block":
            lang = el_data.get('language', "") or ""
            text_parts.append(f"\n```{lang}\n{el_data.get('code','')}\n```\n")
        elif el_type == "page_break": text_parts.append("\n---\n")
        if el_type not in ['list_item'] and text_parts and (not text_parts[-1].endswith("\n\n") and not text_parts[-1].endswith("\n---\n\n") ) : 
             text_parts.append("\n") 
             
    raw_text = "".join(text_parts)
    cleaned_text = raw_text.strip()
    cleaned_text = cleaned_text.replace('\r\n', '\n') 
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text) 
    return cleaned_text

def parse_markdown_to_structured_output(md_content_str: str, original_metadata: Dict[str, Any]) -> Optional[Union[ParsedDocumentOutput, Dict[str, Any]]]:
    logger.info(f"Parsing Markdown content (length: {len(md_content_str)} chars) using markdown-it-py with SyntaxTreeNode...")
    try:
        md_parser = MarkdownIt("commonmark", {'linkify': True}).enable("table")
        tokens = md_parser.parse(md_content_str)
        
        root_syntax_node = SyntaxTreeNode(tokens)
        structured_elements = _convert_md_tree_to_elements(root_syntax_node) 

        linear_text = _generate_parsed_text_from_elements_internal(structured_elements)

        if _PARSER_PYDANTIC_AVAILABLE:
            return ParsedDocumentOutput(
                parsed_text=linear_text,
                elements=structured_elements, 
                original_metadata=original_metadata
            )
        else:
            return {
                "parsed_text": linear_text,
                "elements": structured_elements,
                "original_metadata": original_metadata
            }
    except Exception as e:
        logger.error(f"Error in parse_markdown_to_structured_output: {e}", exc_info=True)
        return None

# --- Placeholder for other parsers (保持不变) ---
def parse_docx_to_structured_output(file_path: str, original_metadata: Dict[str, Any]) -> Optional[Union[ParsedDocumentOutput, Dict[str, Any]]]:
    logger.info(f"DOCX parser placeholder for: {file_path}") 
    text = f"[DOCX content of {os.path.basename(file_path)}]"
    if _PARSER_PYDANTIC_AVAILABLE: return ParsedDocumentOutput(parsed_text=text, elements=[NarrativeTextElement(text=text)], original_metadata=original_metadata) 
    return {"parsed_text": text, "elements": [{"element_type":"narrative_text", "text":text}], "original_metadata": original_metadata}

def parse_pdf_to_structured_output(file_path: str, original_metadata: Dict[str, Any]) -> Optional[Union[ParsedDocumentOutput, Dict[str, Any]]]:
    logger.info(f"PDF parser placeholder for: {file_path}") 
    text = f"[PDF content of {os.path.basename(file_path)}]"
    if _PARSER_PYDANTIC_AVAILABLE: return ParsedDocumentOutput(parsed_text=text, elements=[NarrativeTextElement(text=text)], original_metadata=original_metadata) 
    return {"parsed_text": text, "elements": [{"element_type":"narrative_text", "text":text}], "original_metadata": original_metadata}

def parse_xlsx_to_structured_output(file_path: str, original_metadata: Dict[str, Any]) -> Optional[Union[ParsedDocumentOutput, Dict[str, Any]]]:
    logger.info(f"XLSX parser placeholder for: {file_path}") 
    text = f"[XLSX content of {os.path.basename(file_path)}]"
    if _PARSER_PYDANTIC_AVAILABLE: return ParsedDocumentOutput(parsed_text=text, elements=[NarrativeTextElement(text=text)], original_metadata=original_metadata) 
    return {"parsed_text": text, "elements": [{"element_type":"narrative_text", "text":text}], "original_metadata": original_metadata}
        
def parse_html_to_structured_output(html_content_str: str, original_metadata: Dict[str, Any]) -> Optional[Union[ParsedDocumentOutput, Dict[str, Any]]]:
    logger.info(f"HTML parser placeholder for content length: {len(html_content_str)}") 
    text = f"[HTML content snippet: {html_content_str[:100]}]"
    if _PARSER_PYDANTIC_AVAILABLE: return ParsedDocumentOutput(parsed_text=text, elements=[NarrativeTextElement(text=text)], original_metadata=original_metadata) 
    return {"parsed_text": text, "elements": [{"element_type":"narrative_text", "text":text}], "original_metadata": original_metadata}

def parse_txt_to_structured_output(txt_content_str: str, original_metadata: Dict[str, Any]) -> Optional[Union[ParsedDocumentOutput, Dict[str, Any]]]:
    logger.info(f"TXT parser for content length: {len(txt_content_str)}") 
    if _PARSER_PYDANTIC_AVAILABLE:
        return ParsedDocumentOutput(
            parsed_text=txt_content_str, 
            elements=[NarrativeTextElement(text=txt_content_str)], 
            original_metadata=original_metadata
        ) 
    return {
        "parsed_text": txt_content_str, 
        "elements": [{"element_type":"narrative_text", "text":txt_content_str}], 
        "original_metadata": original_metadata
    }