import os
import sys
# --- 动态添加项目根目录的上层到sys.path ---
SCRIPT_DIR_MD = os.path.dirname(os.path.abspath(__file__))
POC_ROOT_MD = os.path.dirname(SCRIPT_DIR_MD)
PROJECT_ROOT_MD_GUESS = os.path.dirname(POC_ROOT_MD)

if PROJECT_ROOT_MD_GUESS not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_MD_GUESS)
    # print(f"DEBUG: Added to sys.path for Pydantic model import: {PROJECT_ROOT_MD_GUESS}")
# --- 动态添加结束 ---

from markdown_it import MarkdownIt
import logging
import json
from typing import List, Dict, Any, Optional, Union, Literal

_PYDANTIC_MODELS_AVAILABLE = False
try:
    from zhz_rag_pipeline_dagster.zhz_rag_pipeline.pydantic_models_dagster import (
        ParsedDocumentOutput,
        DocumentElementType,
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
    )
    _PYDANTIC_MODELS_AVAILABLE = True
    print("INFO: Successfully imported Pydantic models from zhz_rag_pipeline_dagster.")
except ImportError as e:
    print(f"WARNING: Could not import Pydantic models due to: {e}. Using fallback Any/dict types for PoC.")
    class BaseModel: pass
    class DocumentElementMetadata(BaseModel): page_number: Optional[int] = None; source_coordinates: Optional[Dict[str, float]] = None; custom_properties: Optional[Dict[str, Any]] = None
    class ParsedDocumentOutput(BaseModel): parsed_text: str; elements: list; original_metadata: dict; summary: Optional[str] = None
    class TitleElement(BaseModel): element_type:str="title"; text:str; level:int; metadata: Optional[DocumentElementMetadata] = None
    class NarrativeTextElement(BaseModel): element_type:str="narrative_text"; text:str; metadata: Optional[DocumentElementMetadata] = None
    class ListItemElement(BaseModel): element_type:str="list_item"; text:str; level:int=0; ordered:bool=False; item_number:Optional[Union[int, str]]=None; metadata: Optional[DocumentElementMetadata] = None
    class TableElement(BaseModel): element_type:str="table"; markdown_representation:Optional[str]=None; text_representation:Optional[str]=None; html_representation:Optional[str]=None; caption:Optional[str]=None; metadata: Optional[DocumentElementMetadata] = None
    class CodeBlockElement(BaseModel): element_type:str="code_block"; code:str; language:Optional[str]=None; metadata: Optional[DocumentElementMetadata] = None
    class ImageElement(BaseModel): element_type:str="image"; alt_text:Optional[str]=None; caption:Optional[str]=None; metadata: Optional[DocumentElementMetadata] = None
    class PageBreakElement(BaseModel): element_type:str="page_break"; metadata: Optional[DocumentElementMetadata] = None
    class HeaderElement(BaseModel): element_type:str="header"; text:str; metadata: Optional[DocumentElementMetadata] = None
    class FooterElement(BaseModel): element_type:str="footer"; text:str; metadata: Optional[DocumentElementMetadata] = None
    DocumentElementType = Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

SAMPLE_DOCS_DIR = os.path.join(SCRIPT_DIR_MD, "sample_docs") # Use absolute path for sample_docs

def get_text_from_inline_tokens(inline_tokens: Optional[List[Any]]) -> str:
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
            text_content += get_text_from_inline_tokens(token.children)
    return text_content

def convert_md_tokens_to_elements(tokens: list) -> List[Any]:
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
                text = get_text_from_inline_tokens(tokens[idx_content].children).strip()
            if _PYDANTIC_MODELS_AVAILABLE:
                elements.append(TitleElement(text=text, level=level))
            else:
                elements.append({"element_type": "title", "text": text, "level": level})
            idx = idx_content + 2 # Skip inline and heading_close
            continue

        elif token.type == 'paragraph_open':
            is_list_item_para = False
            if list_level_stack and token.level >= list_level_stack[-1]["level"]:
                pass

            if not is_list_item_para or not list_level_stack:
                idx_content = idx + 1
                text = ""
                if idx_content < len(tokens) and tokens[idx_content].type == 'inline':
                    text = get_text_from_inline_tokens(tokens[idx_content].children).strip()
                if text:
                    if _PYDANTIC_MODELS_AVAILABLE:
                        elements.append(NarrativeTextElement(text=text))
                    else:
                        elements.append({"element_type": "narrative_text", "text": text})
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

            # --- 增强: 兼容标准列表和紧凑列表 ---
            next_token_idx = idx + 1
            if next_token_idx < len(tokens):
                next_token = tokens[next_token_idx]
                # 标准列表: list_item_open -> paragraph_open -> inline
                if next_token.type == 'paragraph_open' and next_token.level == li_level + 1:
                    inline_idx = next_token_idx + 1
                    if inline_idx < len(tokens) and tokens[inline_idx].type == 'inline':
                        item_text = get_text_from_inline_tokens(tokens[inline_idx].children).strip()
                # 紧凑列表: list_item_open -> inline
                elif next_token.type == 'inline' and next_token.level == li_level + 1:
                    item_text = get_text_from_inline_tokens(next_token.children).strip()

            if list_level_stack:
                list_info = list_level_stack[-1]
                item_num_val = None
                if list_info["ordered"]:
                    item_num_val = list_info["current_num"]
                    list_info["current_num"] += 1

                if _PYDANTIC_MODELS_AVAILABLE:
                    elements.append(ListItemElement(
                        text=item_text,
                        level=token.level, # --- 修正: 使用 list_item_open token 的 level ---
                        ordered=list_info["ordered"],
                        item_number=str(item_num_val) if item_num_val is not None else None
                    ))
                else:
                    elements.append({
                        "element_type": "list_item", "text": item_text,
                        "level": token.level, # --- 修正: 使用 list_item_open token 的 level ---
                        "ordered": list_info["ordered"],
                        "item_number": str(item_num_val) if item_num_val is not None else None
                    })

            temp_idx = idx + 1
            nesting_count = 0
            while temp_idx < len(tokens):
                if tokens[temp_idx].type == 'list_item_open' and tokens[temp_idx].level == li_level:
                    if nesting_count == 0:
                        idx = temp_idx
                        break
                if tokens[temp_idx].type == 'list_item_open': nesting_count +=1
                if tokens[temp_idx].type == 'list_item_close':
                    if nesting_count == 0 and tokens[temp_idx].level == li_level:
                        idx = temp_idx + 1
                        break
                    nesting_count -=1
                temp_idx += 1
            else:
                idx = temp_idx
            continue

        elif token.type in ['bullet_list_close', 'ordered_list_close']:
            if list_level_stack:
                list_level_stack.pop()
            idx += 1
            continue

        elif token.type == 'table_open':
            header_cells = []
            body_rows_cells = []
            current_row_cells = []
            in_thead = False

            temp_idx = idx + 1
            while temp_idx < len(tokens) and tokens[temp_idx].type != 'table_close':
                t_token = tokens[temp_idx]
                if t_token.type == 'thead_open': in_thead = True
                elif t_token.type == 'thead_close': in_thead = False
                elif t_token.type == 'tr_open': current_row_cells = []
                elif t_token.type in ['th_open', 'td_open']:
                    content_idx = temp_idx + 1
                    if content_idx < len(tokens) and tokens[content_idx].type == 'inline':
                        current_row_cells.append(get_text_from_inline_tokens(tokens[content_idx].children).strip())
                elif t_token.type == 'tr_close':
                    if current_row_cells:
                        if in_thead:
                            header_cells.append(list(current_row_cells))
                        else:
                            body_rows_cells.append(list(current_row_cells))
                temp_idx += 1

            md_table_str = ""
            if header_cells:
                md_table_str += "| " + " | ".join(header_cells[0]) + " |\n"
                md_table_str += "| " + " | ".join(["---"] * len(header_cells[0])) + " |\n"
            for row_data_list in body_rows_cells:
                md_table_str += "| " + " | ".join(row_data_list) + " |\n"

            if _PYDANTIC_MODELS_AVAILABLE:
                elements.append(TableElement(markdown_representation=md_table_str.strip()))
            else:
                elements.append({"element_type": "table", "markdown_representation": md_table_str.strip()})
            idx = temp_idx + 1
            continue

        elif token.type == 'fence' or token.type == 'code_block':
            code_content = token.content.strip()
            lang = token.info.strip() if token.info else None
            if _PYDANTIC_MODELS_AVAILABLE:
                elements.append(CodeBlockElement(code=code_content, language=lang))
            else:
                elements.append({"element_type": "code_block", "code": code_content, "language": lang})
            idx += 1
            continue

        elif token.type == 'hr':
            if _PYDANTIC_MODELS_AVAILABLE:
                elements.append(PageBreakElement())
            else:
                elements.append({"element_type": "page_break"})
            idx += 1
            continue

        # --- 添加并优化: Blockquote 处理 ---
        elif token.type == 'blockquote_open':
            blockquote_text_parts = []
            temp_idx = idx + 1
            start_level = token.level

            while temp_idx < len(tokens):
                bq_token = tokens[temp_idx]
                if bq_token.type == 'blockquote_close' and bq_token.level == start_level:
                    idx = temp_idx
                    break

                if bq_token.type == 'paragraph_open':
                    para_content_idx = temp_idx + 1
                    if para_content_idx < len(tokens) and tokens[para_content_idx].type == 'inline':
                        blockquote_text_parts.append(get_text_from_inline_tokens(tokens[para_content_idx].children).strip())
                    
                    temp_idx = para_content_idx + 1
                    if temp_idx < len(tokens) and tokens[temp_idx].type == 'paragraph_close':
                        temp_idx +=1
                    else:
                        temp_idx -=1
                        
                temp_idx += 1
            else:
                idx = temp_idx

            if blockquote_text_parts:
                full_text = "\n".join(blockquote_text_parts)
                if _PYDANTIC_MODELS_AVAILABLE:
                    elements.append(NarrativeTextElement(text=full_text))
                else:
                    elements.append({"element_type": "narrative_text", "text": full_text, "_is_blockquote": True})
            idx +=1
            continue

        idx += 1
    return elements

def generate_parsed_text_from_elements(elements: List[Any]) -> str:
    text_parts = []
    for el_data_any in elements:
        el_data = {}
        if _PYDANTIC_MODELS_AVAILABLE and hasattr(el_data_any, 'model_dump'):
            el_data = el_data_any.model_dump()
        elif isinstance(el_data_any, dict):
            el_data = el_data_any
        else:
            logger.warning(f"Skipping unknown element type in generate_parsed_text: {type(el_data_any)}")
            continue

        el_type = el_data.get("element_type")
        if el_type == "title":
            text_parts.append(f"\n{'#' * el_data.get('level',1)} {el_data.get('text','')}\n")
        elif el_type == "narrative_text":
            text_parts.append(el_data.get('text','') + "\n")
        elif el_type == "list_item":
            prefix = f"{el_data.get('item_number','')}. " if el_data.get('ordered') and el_data.get('item_number') else "- "
            indent = "  " * el_data.get('level',0)
            text_parts.append(f"{indent}{prefix}{el_data.get('text','')}\n")
        elif el_type == "table":
            if el_data.get('markdown_representation'):
                text_parts.append(f"\n[Table:]\n{el_data['markdown_representation']}\n")
            elif el_data.get('text_representation'):
                text_parts.append(f"\n[Table:]\n{el_data['text_representation']}\n")
        elif el_type == "code_block":
            lang = el_data.get('language', "") or ""
            text_parts.append(f"\n```{lang}\n{el_data.get('code','')}\n```\n")
        elif el_type == "page_break":
            text_parts.append("\n---\n")
        text_parts.append("\n")

    return "".join(text_parts).strip().replace("\n\n\n", "\n\n").replace("\n\n\n", "\n\n")

def analyze_md_with_markdown_it(file_path: str) -> Optional[Any]:
    logger.info(f"\n--- Analyzing MD with markdown-it-py: {file_path} ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        md = MarkdownIt("commonmark").enable("table")
        tokens = md.parse(md_content)

        structured_elements = convert_md_tokens_to_elements(tokens)

        logger.info(f"\n[Converted Pydantic Elements/Dicts (Found {len(structured_elements)})]")
        for i, element_obj in enumerate(structured_elements):
            if _PYDANTIC_MODELS_AVAILABLE and hasattr(element_obj, 'model_dump_json'):
                logger.info(f"  Element {i+1}: {element_obj.model_dump_json(indent=2, exclude_none=True)}")
            else:
                logger.info(f"  Element {i+1} (raw dict): {json.dumps(element_obj, indent=2, ensure_ascii=False)}")

        linear_text = generate_parsed_text_from_elements(structured_elements)
        logger.info("\n[Generated Linear Parsed Text]")
        logger.info(linear_text)

        if _PYDANTIC_MODELS_AVAILABLE:
            doc_output = ParsedDocumentOutput(
                parsed_text=linear_text,
                elements=structured_elements,
                original_metadata={"filename": os.path.basename(file_path), "source_path": file_path}
            )
            return doc_output
        else:
            logger.warning("Returning raw dictionary as Pydantic models for ParsedDocumentOutput are not available.")
            return {
                "parsed_text": linear_text,
                "elements": structured_elements,
                "original_metadata": {"filename": os.path.basename(file_path), "source_path": file_path}
            }
    except Exception as e:
        logger.error(f"Error processing file {file_path} with markdown-it-py: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    if not os.path.isdir(SAMPLE_DOCS_DIR):
        logger.error(f"Sample documents directory '{SAMPLE_DOCS_DIR}' not found relative to script. Full path expected: {os.path.abspath(SAMPLE_DOCS_DIR)}")
    else:
        md_files = sorted([f for f in os.listdir(SAMPLE_DOCS_DIR)
                           if f.lower().endswith(".md") and not f.startswith("~$")])
        if not md_files:
            logger.warning(f"No .md files found in '{SAMPLE_DOCS_DIR}'.")
        else:
            for filename in md_files:
                file_path = os.path.join(SAMPLE_DOCS_DIR, filename)
                parsed_doc_result = analyze_md_with_markdown_it(file_path)
                if parsed_doc_result:
                    if _PYDANTIC_MODELS_AVAILABLE:
                        logger.info(f"\nSuccessfully created ParsedDocumentOutput for {filename}")
                    else:
                        logger.info(f"\nSuccessfully created fallback dictionary output for {filename}")
                else:
                    logger.error(f"Failed to create ParsedDocumentOutput for {filename}")