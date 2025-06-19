# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/zhz_rag_pipeline/document_parsers.py
import os
from markdown_it import MarkdownIt
import logging
from typing import List, Dict, Any, Optional, Union, Literal 

# --- 添加 Unstructured 的导入 ---
try:
    from unstructured.partition.docx import partition_docx
    from unstructured.documents.elements import Element as UnstructuredElement, Title, NarrativeText, ListItem, Table, Image, Header, Footer
    _UNSTRUCTURED_AVAILABLE = True
    print("INFO (document_parsers.py): Successfully imported Unstructured for DOCX.")
except ImportError:
    print("WARNING (document_parsers.py): Unstructured library not found. DOCX parsing will be a placeholder.")
    _UNSTRUCTURED_AVAILABLE = False
    class UnstructuredElement: pass # Dummy
# --- 结束添加 ---

# --- 导入我们定义的Pydantic模型 ---
# 假设这个文件和 pydantic_models_dagster.py 在同一个包下或能通过PYTHONPATH找到
try:
    from .pydantic_models_dagster import ( # 使用相对导入
        ParsedDocumentOutput,
        DocumentElementType, 
        TitleElement,
        NarrativeTextElement,
        ListItemElement,
        TableElement,
        CodeBlockElement,
        PageBreakElement,
        DocumentElementMetadata 
    )
    _PYDANTIC_MODELS_AVAILABLE_PARSERS = True
except ImportError:
    print("WARNING (document_parsers.py): Could not import Pydantic models. Using fallback Any/dict.")
    _PYDANTIC_MODELS_AVAILABLE_PARSERS = False
    class BaseModel: pass
    class DocumentElementMetadata(BaseModel): page_number: Optional[int] = None
    class ParsedDocumentOutput(BaseModel): parsed_text: str; elements: list; original_metadata: dict; summary: Optional[str] = None
    class TitleElement(BaseModel): element_type:str="title"; text:str; level:int; metadata: Optional[DocumentElementMetadata] = None
    class NarrativeTextElement(BaseModel): element_type:str="narrative_text"; text:str; metadata: Optional[DocumentElementMetadata] = None
    class ListItemElement(BaseModel): element_type:str="list_item"; text:str; level:int=0; ordered:bool=False; item_number:Optional[Union[int, str]]=None; metadata: Optional[DocumentElementMetadata] = None
    class TableElement(BaseModel): element_type:str="table"; markdown_representation:Optional[str]=None; metadata: Optional[DocumentElementMetadata] = None
    class CodeBlockElement(BaseModel): element_type:str="code_block"; code:str; language:Optional[str]=None; metadata: Optional[DocumentElementMetadata] = None
    class PageBreakElement(BaseModel): element_type:str="page_break"; metadata: Optional[DocumentElementMetadata] = None
    DocumentElementType = Any


logger = logging.getLogger(__name__) # 每个模块用自己的logger

# --- Markdown 解析逻辑 (从 poc_md_markdown_it.py 迁移并封装) ---



def _get_text_from_md_inline(inline_tokens: Optional[List[Any]]) -> str:
    # (这里是 get_text_from_inline_tokens 函数的完整代码)
    text_content = ""
    if inline_tokens is None: return ""
    for token in inline_tokens:
        if token.type == 'text':
            text_content += token.content
        elif token.type == 'code_inline':
            text_content += f"`{token.content}`"
        elif token.type == 'softbreak':
            text_content += ' ' 
        elif token.type == 'hardbreak':
            text_content += '\n'
        elif token.children: 
            text_content += _get_text_from_md_inline(token.children)
    return text_content

def _convert_md_tokens_to_elements_internal(tokens: list) -> List[Any]:
    # (这里是 convert_md_tokens_to_elements 函数的完整代码，但将其重命名为内部函数)
    # (并确保它在 _PYDANTIC_MODELS_AVAILABLE_PARSERS 为True时创建Pydantic实例，否则创建字典)
    elements: List[Any] = []
    idx = 0
    list_level_stack = [] 

    while idx < len(tokens):
        token = tokens[idx]

        if token.type == 'heading_open':
            level = int(token.tag[1:])
            idx_content = idx + 1
            text = ""
            if idx_content < len(tokens) and tokens[idx_content].type == 'inline':
                text = _get_text_from_md_inline(tokens[idx_content].children).strip()
            if _PYDANTIC_MODELS_AVAILABLE_PARSERS: elements.append(TitleElement(text=text, level=level))
            else: elements.append({"element_type": "title", "text": text, "level": level})
            idx = idx_content + 2 
            continue

        elif token.type == 'paragraph_open':
            is_list_item_para = False
            if list_level_stack and token.level >= list_level_stack[-1]["level"]:
                pass 
            if not is_list_item_para or not list_level_stack: 
                idx_content = idx + 1
                text = ""
                if idx_content < len(tokens) and tokens[idx_content].type == 'inline':
                    text = _get_text_from_md_inline(tokens[idx_content].children).strip()
                if text:
                    if _PYDANTIC_MODELS_AVAILABLE_PARSERS: elements.append(NarrativeTextElement(text=text))
                    else: elements.append({"element_type": "narrative_text", "text": text})
            idx = idx + 2 
            if idx < len(tokens) and tokens[idx-1].type == 'inline': 
                idx +=1 
            continue
        
        elif token.type == 'bullet_list_open':
            list_level_stack.append({"ordered": False, "level": token.level})
            idx += 1
            continue
        elif token.type == 'ordered_list_open':
            start_num = token.attrs.get('start', 1)
            list_level_stack.append({"ordered": True, "current_num": start_num, "level": token.level})
            idx += 1
            continue
        
        elif token.type == 'list_item_open':
            item_text = ""
            li_level = token.level
            next_token_idx = idx + 1
            if next_token_idx < len(tokens):
                next_token = tokens[next_token_idx]
                if next_token.type == 'paragraph_open' and next_token.level == li_level + 1 :
                    inline_idx = next_token_idx + 1
                    if inline_idx < len(tokens) and tokens[inline_idx].type == 'inline':
                        item_text = _get_text_from_md_inline(tokens[inline_idx].children).strip()
                elif next_token.type == 'inline' and next_token.level == li_level +1 :
                    item_text = _get_text_from_md_inline(next_token.children).strip()
            
            if list_level_stack:
                list_info = list_level_stack[-1]
                item_num_val = None
                if list_info["ordered"]:
                    item_num_val = list_info["current_num"]
                    list_info["current_num"] += 1
                
                if _PYDANTIC_MODELS_AVAILABLE_PARSERS:
                    elements.append(ListItemElement(
                        text=item_text, level=token.level, ordered=list_info["ordered"],
                        item_number=str(item_num_val) if item_num_val is not None else None))
                else:
                    elements.append({"element_type": "list_item", "text": item_text, "level":token.level, 
                                     "ordered":list_info["ordered"], "item_number":str(item_num_val) if item_num_val is not None else None})

            temp_idx = idx + 1; nesting_count = 0
            while temp_idx < len(tokens):
                if tokens[temp_idx].type == 'list_item_open' and tokens[temp_idx].level == li_level:
                    if nesting_count == 0: idx = temp_idx; break
                if tokens[temp_idx].type == 'list_item_open': nesting_count +=1
                if tokens[temp_idx].type == 'list_item_close':
                    if nesting_count == 0 and tokens[temp_idx].level == li_level: idx = temp_idx + 1; break
                    nesting_count -=1
                temp_idx += 1
            else: idx = temp_idx
            continue

        elif token.type in ['bullet_list_close', 'ordered_list_close']:
            if list_level_stack: list_level_stack.pop()
            idx += 1
            continue

        elif token.type == 'table_open':
            header_content = []; body_rows_cells = []; current_row_cells = []; in_thead = False
            temp_idx = idx + 1
            while temp_idx < len(tokens) and tokens[temp_idx].type != 'table_close':
                t_token = tokens[temp_idx]
                if t_token.type == 'thead_open': in_thead = True
                elif t_token.type == 'thead_close': in_thead = False
                elif t_token.type == 'tr_open': current_row_cells = []
                elif t_token.type in ['th_open', 'td_open']:
                    content_idx = temp_idx + 1
                    if content_idx < len(tokens) and tokens[content_idx].type == 'inline':
                        current_row_cells.append(_get_text_from_md_inline(tokens[content_idx].children).strip())
                elif t_token.type == 'tr_close':
                    if current_row_cells:
                        if in_thead or (not header_content and not body_rows_cells): header_content.append(list(current_row_cells))
                        else: body_rows_cells.append(list(current_row_cells))
                temp_idx += 1
            md_table_str = ""
            if header_content:
                md_table_str += "| " + " | ".join(header_content[0]) + " |\n"
                md_table_str += "| " + " | ".join(["---"] * len(header_content[0])) + " |\n"
            for row_data_list in body_rows_cells: md_table_str += "| " + " | ".join(row_data_list) + " |\n"
            if _PYDANTIC_MODELS_AVAILABLE_PARSERS: elements.append(TableElement(markdown_representation=md_table_str.strip()))
            else: elements.append({"element_type": "table", "markdown_representation": md_table_str.strip()})
            idx = temp_idx + 1 
            continue

        elif token.type == 'fence' or token.type == 'code_block':
            code_content = token.content.strip(); lang = token.info.strip() if token.info else None
            if _PYDANTIC_MODELS_AVAILABLE_PARSERS: elements.append(CodeBlockElement(code=code_content, language=lang))
            else: elements.append({"element_type": "code_block", "code": code_content, "language": lang})
            idx += 1
            continue
        
        elif token.type == 'hr':
            if _PYDANTIC_MODELS_AVAILABLE_PARSERS: elements.append(PageBreakElement())
            else: elements.append({"element_type": "page_break"})
            idx += 1
            continue
        
        elif token.type == 'blockquote_open':
            blockquote_text_parts = []; temp_idx = idx + 1; start_level = token.level
            while temp_idx < len(tokens):
                bq_token = tokens[temp_idx]
                if bq_token.type == 'blockquote_close' and bq_token.level == start_level: idx = temp_idx; break
                if bq_token.type == 'paragraph_open':
                    para_content_idx = temp_idx + 1
                    if para_content_idx < len(tokens) and tokens[para_content_idx].type == 'inline':
                        blockquote_text_parts.append(_get_text_from_md_inline(tokens[para_content_idx].children).strip())
                    temp_idx = para_content_idx + 1 
                    if temp_idx < len(tokens) and tokens[temp_idx].type == 'paragraph_close': temp_idx +=1
                    else: temp_idx -=1
                temp_idx +=1
            else: idx = temp_idx
            if blockquote_text_parts:
                full_text = "\n".join(blockquote_text_parts)
                if _PYDANTIC_MODELS_AVAILABLE_PARSERS: elements.append(NarrativeTextElement(text=full_text)) 
                else: elements.append({"element_type": "narrative_text", "text": full_text, "_is_blockquote": True})
            idx +=1
            continue
        idx += 1 
    return elements

def _generate_parsed_text_from_elements_internal(elements: List[Any]) -> str:
    # (这里是 generate_parsed_text_from_elements 函数的完整代码)
    # (确保它在 _PYDANTIC_MODELS_AVAILABLE_PARSERS 为True时能处理Pydantic实例，否则处理字典)
    text_parts = []
    for el_data_any in elements:
        el_data = {}
        if _PYDANTIC_MODELS_AVAILABLE_PARSERS and hasattr(el_data_any, 'model_dump'):
            el_data = el_data_any.model_dump() 
        elif isinstance(el_data_any, dict):
            el_data = el_data_any
        else: continue

        el_type = el_data.get("element_type")
        if el_type == "title": text_parts.append(f"\n{'#' * el_data.get('level',1)} {el_data.get('text','')}\n")
        elif el_type == "narrative_text": text_parts.append(el_data.get('text','') + "\n")
        elif el_type == "list_item":
            prefix = f"{el_data.get('item_number','')}. " if el_data.get('ordered') and el_data.get('item_number') else "- "
            indent = "  " * el_data.get('level',0)
            text_parts.append(f"{indent}{prefix}{el_data.get('text','')}\n")
        elif el_type == "table":
            if el_data.get('markdown_representation'): text_parts.append(f"\n[Table: {el_data.get('caption','Unnamed Table')}]\n{el_data['markdown_representation']}\n")
            elif el_data.get('text_representation'): text_parts.append(f"\n[Table: {el_data.get('caption','Unnamed Table')}]\n{el_data['text_representation']}\n")
        elif el_type == "code_block":
            lang = el_data.get('language', "") or ""
            text_parts.append(f"\n```{lang}\n{el_data.get('code','')}\n```\n")
        elif el_type == "page_break": text_parts.append("\n---\n")
        text_parts.append("\n") 
    return "".join(text_parts).strip().replace("\n\n\n", "\n\n").replace("\n\n\n", "\n\n")

def _convert_unstructured_to_pydantic(elements: List[UnstructuredElement]) -> List[DocumentElementType]:  # type: ignore
    """Converts a list of Unstructured elements to our internal Pydantic models."""
    pydantic_elements = []
    for el in elements:
        meta = DocumentElementMetadata(page_number=getattr(el.metadata, 'page_number', None))
        
        if isinstance(el, Title):
            pydantic_elements.append(TitleElement(text=el.text, level=getattr(el.metadata, 'category_depth', 1), metadata=meta))
        elif isinstance(el, NarrativeText):
            pydantic_elements.append(NarrativeTextElement(text=el.text, metadata=meta))
        elif isinstance(el, ListItem):
            pydantic_elements.append(ListItemElement(text=el.text, metadata=meta))
        elif isinstance(el, Table):
            # Unstructured v0.12+ has built-in markdown conversion
            pydantic_elements.append(TableElement(markdown_representation=getattr(el, 'text_as_html', str(el)), metadata=meta))
        elif isinstance(el, (Header, Footer, Image)):
             # We can choose to ignore headers, footers, and images for now
             continue
        else:
            # Fallback for any other element types
            if el.text.strip():
                 pydantic_elements.append(NarrativeTextElement(text=el.text, metadata=meta))
                 
    return pydantic_elements


def parse_markdown_to_structured_output(md_content_str: str, original_metadata: Dict[str, Any]) -> Optional[ParsedDocumentOutput]:
    """
    Top-level function to parse markdown string and return ParsedDocumentOutput.
    """
    logger.info(f"Parsing Markdown content (length: {len(md_content_str)} chars)...")
    try:
        md_parser = MarkdownIt("commonmark").enable("table") # Removed "breaks":True based on last log
        tokens = md_parser.parse(md_content_str)
        
        structured_elements = _convert_md_tokens_to_elements_internal(tokens)
        linear_text = _generate_parsed_text_from_elements_internal(structured_elements)

        if _PYDANTIC_MODELS_AVAILABLE_PARSERS:
            return ParsedDocumentOutput(
                parsed_text=linear_text,
                elements=structured_elements,
                original_metadata=original_metadata
            )
        else: # Fallback if Pydantic models aren't available (e.g. PoC context)
            return {
                 "parsed_text": linear_text,
                 "elements": structured_elements,
                 "original_metadata": original_metadata
            } # type: ignore 
    except Exception as e:
        logger.error(f"Error in parse_markdown_to_structured_output: {e}", exc_info=True)
        return None

# --- Placeholder for other parsers ---
def parse_docx_to_structured_output(file_path: str, original_metadata: Dict[str, Any]) -> Optional[ParsedDocumentOutput]:
    """
    Parses a DOCX file using Unstructured.io, extracts rich metadata,
    and converts elements to our internal Pydantic models.
    """
    if not _UNSTRUCTURED_AVAILABLE:
        logger.error("Unstructured library is not available. Cannot parse DOCX files.")
        return None
        
    logger.info(f"Parsing DOCX with Unstructured: {file_path}")
    
    try:
        # 使用 Unstructured 解析文档，设置策略以获取更干净的数据
        unstructured_elements = partition_docx(
            filename=file_path,
            strategy="hi_res",  # 使用高分辨率策略以更好地处理布局
            infer_table_structure=True, # 开启表格结构推断
        )
        
        # --- 关键：合并 Unstructured 提取的元数据 ---
        # partition_docx 返回的第一个元素通常包含文档级别的元数据
        doc_level_meta = {}
        if unstructured_elements:
             # Unstructured v0.12+ 将元数据附加到每个元素上
             # 我们从第一个元素获取通用元数据
             doc_level_meta = unstructured_elements[0].metadata.to_dict()

        # 将 Unstructured 的元数据与我们传入的原始元数据合并
        # Unstructured 的元数据优先级更高，因为它更具体
        combined_metadata = {**original_metadata, **doc_level_meta}
        # 移除一些不需要的内部键
        combined_metadata.pop('parent_id', None)
        combined_metadata.pop('category_depth', None)

        # 将 Unstructured 元素转换为我们自己的 Pydantic 模型
        pydantic_elements = _convert_unstructured_to_pydantic(unstructured_elements)
        
        if not pydantic_elements:
            logger.warning(f"No content elements were extracted from DOCX file: {file_path}")
            return None

        # 从转换后的元素生成线性文本表示
        linear_text = _generate_parsed_text_from_elements_internal(pydantic_elements)

        if _PYDANTIC_MODELS_AVAILABLE_PARSERS:
            return ParsedDocumentOutput(
                parsed_text=linear_text,
                elements=pydantic_elements,
                original_metadata=combined_metadata # <--- 使用合并后的元数据
            )
        else:
            # Fallback for non-pydantic environment (should not happen in production)
            return {
                "parsed_text": linear_text,
                "elements": pydantic_elements,
                "original_metadata": combined_metadata
            } # type: ignore
            
    except Exception as e:
        logger.error(f"Error parsing DOCX file '{file_path}' with Unstructured: {e}", exc_info=True)
        return None
    
def parse_pdf_to_structured_output(file_path: str, original_metadata: Dict[str, Any]) -> Optional[ParsedDocumentOutput]:
    logger.info(f"Parsing PDF: {file_path} (Not yet fully implemented in document_parsers.py)")
    # Here you would integrate the PyMuPDF logic from your PoC
    text_content = f"[Placeholder: PDF content for {os.path.basename(file_path)}]"
    elements = []
    if _PYDANTIC_MODELS_AVAILABLE_PARSERS:
        elements.append(NarrativeTextElement(text=text_content))
        return ParsedDocumentOutput(parsed_text=text_content, elements=elements, original_metadata=original_metadata)
    else:
        elements.append({"element_type":"narrative_text", "text":text_content})
        return {"parsed_text":text_content, "elements":elements, "original_metadata":original_metadata} # type: ignore

def parse_xlsx_to_structured_output(file_path: str, original_metadata: Dict[str, Any]) -> Optional[ParsedDocumentOutput]:
    logger.info(f"Parsing XLSX: {file_path} (Not yet fully implemented in document_parsers.py)")
    # Here you would integrate the pandas logic from your PoC
    text_content = f"[Placeholder: XLSX content for {os.path.basename(file_path)}]"
    elements = []
    if _PYDANTIC_MODELS_AVAILABLE_PARSERS:
        elements.append(NarrativeTextElement(text=text_content)) # Or TableElement
        return ParsedDocumentOutput(parsed_text=text_content, elements=elements, original_metadata=original_metadata)
    else:
        elements.append({"element_type":"narrative_text", "text":text_content})
        return {"parsed_text":text_content, "elements":elements, "original_metadata":original_metadata} # type: ignore
        
def parse_html_to_structured_output(html_content_str: str, original_metadata: Dict[str, Any]) -> Optional[ParsedDocumentOutput]:
    logger.info(f"Parsing HTML content (length: {len(html_content_str)} chars) (Not yet fully implemented in document_parsers.py)")
    # Here you would integrate the BeautifulSoup logic from your PoC
    text_content = f"[Placeholder: HTML content snippet {html_content_str[:100]}]"
    elements = []
    if _PYDANTIC_MODELS_AVAILABLE_PARSERS:
        elements.append(NarrativeTextElement(text=text_content))
        return ParsedDocumentOutput(parsed_text=text_content, elements=elements, original_metadata=original_metadata)
    else:
        elements.append({"element_type":"narrative_text", "text":text_content})
        return {"parsed_text":text_content, "elements":elements, "original_metadata":original_metadata} # type: ignore