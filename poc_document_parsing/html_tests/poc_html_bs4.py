import os
import sys
from bs4 import BeautifulSoup, NavigableString, Tag
import logging
import json
from typing import List, Dict, Any, Optional, Union, Literal, Set

# --- 动态添加项目根目录的上层到sys.path ---
SCRIPT_DIR_HTML = os.path.dirname(os.path.abspath(__file__))
POC_ROOT_HTML = os.path.dirname(SCRIPT_DIR_HTML)
PROJECT_ROOT_HTML_GUESS = os.path.dirname(POC_ROOT_HTML) 

if PROJECT_ROOT_HTML_GUESS not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_HTML_GUESS)
# --- 动态添加结束 ---

_PYDANTIC_MODELS_AVAILABLE_HTML = False
try:
    from zhz_rag_pipeline_dagster.zhz_rag_pipeline.pydantic_models_dagster import (
        ParsedDocumentOutput, DocumentElementType, TitleElement, NarrativeTextElement,
        ListItemElement, TableElement, CodeBlockElement, PageBreakElement,
        DocumentElementMetadata
    )
    _PYDANTIC_MODELS_AVAILABLE_HTML = True
    print("INFO: Successfully imported Pydantic models for HTML PoC.")
except ImportError as e:
    print(f"WARNING: Could not import Pydantic models for HTML PoC due to: {e}. Using fallback Any/dict types.")
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


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

SAMPLE_DOCS_DIR = os.path.join(SCRIPT_DIR_HTML, "sample_docs")

# Tags to typically ignore for main content extraction
IGNORE_TAGS = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'meta', 'link', 'button', 'input']
# Tags that define a semantic block but we want to process their children
CONTAINER_TAGS = ['div', 'section', 'article', 'main', 'body']


def table_to_markdown(table_tag: Tag) -> str:
    """Converts a BeautifulSoup table Tag to a Markdown string, respecting thead and tbody."""
    md_rows = []
    
    # Process header
    header_row = table_tag.find('thead')
    if not header_row:
        header_row = table_tag # fallback to table if no thead
        
    header_cells = header_row.find_all(['th', 'td']) # th is preferred
    if header_cells:
        header_texts = [cell.get_text(separator=' ', strip=True) for cell in header_cells]
        md_rows.append("| " + " | ".join(header_texts) + " |")
        md_rows.append("| " + " | ".join(["---"] * len(header_texts)) + " |")

    # Process body
    body = table_tag.find('tbody')
    if not body:
        body = table_tag # fallback to table if no tbody

    for row in body.find_all('tr'):
        # Avoid re-processing header if it was part of the main table tag
        if row.find('th') and md_rows:
            continue
            
        cell_texts = [cell.get_text(separator=' ', strip=True) for cell in row.find_all('td')]
        if cell_texts:
            md_rows.append("| " + " | ".join(cell_texts) + " |")
            
    return "\n".join(md_rows)

def convert_html_to_elements(soup: BeautifulSoup) -> List[Any]:
    """
    Converts a BeautifulSoup object to a list of structured elements using a recursive approach.
    """
    elements: List[Any] = []
    processed_tags: Set[Tag] = set()

    def _process_tag(tag: Tag, current_level: int = 0):
        """Recursively processes a tag and its children."""
        if tag in processed_tags or tag.name in IGNORE_TAGS:
            return

        is_block_element = True
        
        # --- Handle specific element types ---
        if tag.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            level = int(tag.name[1:])
            text = tag.get_text(strip=True)
            if text:
                if _PYDANTIC_MODELS_AVAILABLE_HTML: elements.append(TitleElement(text=text, level=level))
                else: elements.append({"element_type": "title", "text": text, "level": level})
        
        elif tag.name == 'p':
            text = tag.get_text(strip=True)
            if text:
                if _PYDANTIC_MODELS_AVAILABLE_HTML: elements.append(NarrativeTextElement(text=text))
                else: elements.append({"element_type": "narrative_text", "text": text})

        elif tag.name in ['ul', 'ol']:
            ordered = tag.name == 'ol'
            start_num = int(tag.get('start', '1'))
            # Process direct li children of this list
            for i, li in enumerate(tag.find_all('li', recursive=False)):
                # Extract text that is a direct child of li, excluding nested list text
                li_text = ''.join(c.strip() for c in li.find_all(string=True, recursive=False))

                if li_text:
                    item_num_str = str(start_num + i) if ordered else None
                    if _PYDANTIC_MODELS_AVAILABLE_HTML: elements.append(ListItemElement(text=li_text, level=current_level, ordered=ordered, item_number=item_num_str))
                    else: elements.append({"element_type": "list_item", "text": li_text, "level": current_level, "ordered": ordered, "item_number": item_num_str})
                
                # Recursively process the li tag to find nested lists
                _process_tag(li, current_level + 1)
                processed_tags.add(li) # Mark li as processed

        elif tag.name == 'table':
            md_table = table_to_markdown(tag)
            caption_tag = tag.find('caption')
            caption_text = caption_tag.get_text(strip=True) if caption_tag else None
            if _PYDANTIC_MODELS_AVAILABLE_HTML: elements.append(TableElement(markdown_representation=md_table, html_representation=str(tag), caption=caption_text))
            else: elements.append({"element_type": "table", "markdown_representation": md_table, "html_representation": str(tag), "caption": caption_text})

        elif tag.name == 'pre':
            code_tag = tag.find('code')
            code_text, lang = "", None
            if code_tag:
                code_text = code_tag.get_text()
                lang_class = code_tag.get('class', [])
                if lang_class:
                    lang = next((cls.split('language-')[-1] for cls in lang_class if cls.startswith('language-')), None)
            else:
                code_text = tag.get_text()
            
            if code_text.strip():
                if _PYDANTIC_MODELS_AVAILABLE_HTML: elements.append(CodeBlockElement(code=code_text.strip('\n'), language=lang))
                else: elements.append({"element_type": "code_block", "code": code_text.strip('\n'), "language": lang})

        elif tag.name == 'blockquote':
            text = tag.get_text(strip=True)
            if text:
                if _PYDANTIC_MODELS_AVAILABLE_HTML: elements.append(NarrativeTextElement(text=text))
                else: elements.append({"element_type": "narrative_text", "text": text, "_is_blockquote": True})

        elif tag.name == 'hr':
            if _PYDANTIC_MODELS_AVAILABLE_HTML: elements.append(PageBreakElement())
            else: elements.append({"element_type": "page_break"})
        
        else:
            is_block_element = False

        # --- Recursion logic ---
        processed_tags.add(tag)
        if not is_block_element or tag.name in CONTAINER_TAGS or tag.name == 'li':
            for child in tag.children:
                if isinstance(child, Tag):
                    _process_tag(child, current_level)

    # --- Main execution flow for conversion ---
    main_content_area = soup.find('article') or soup.find('main') or soup.body
    if not main_content_area:
        logger.warning("Could not find <article>, <main>, or <body> tag in HTML document.")
        return elements

    for ignore_tag_name in IGNORE_TAGS:
        for tag_to_remove in main_content_area.find_all(ignore_tag_name):
            tag_to_remove.decompose()

    # Start the recursive processing from the main content area
    _process_tag(main_content_area)
    
    return elements

def generate_parsed_text_from_html_elements(elements: List[Any]) -> str:
    text_parts = []
    for el_data_any in elements:
        el_data = {}
        if _PYDANTIC_MODELS_AVAILABLE_HTML and hasattr(el_data_any, 'model_dump'):
            el_data = el_data_any.model_dump()
        elif isinstance(el_data_any, dict):
            el_data = el_data_any
        else:
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
                text_parts.append(f"\n[Table: {el_data.get('caption','Unnamed Table')}]\n{el_data['markdown_representation']}\n")
        elif el_type == "code_block":
            lang = el_data.get('language', "") or ""
            text_parts.append(f"\n```{lang}\n{el_data.get('code','')}\n```\n")
        elif el_type == "page_break":
            text_parts.append("\n---\n")
        text_parts.append("\n")
    return "".join(text_parts).strip().replace("\n\n\n", "\n\n")

def analyze_html_with_bs4(file_path: str) -> Optional[Any]:
    logger.info(f"\n--- Analyzing HTML with BeautifulSoup4: {file_path} ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, "lxml")

        structured_elements = convert_html_to_elements(soup)
        
        logger.info(f"\n[Converted Pydantic Elements/Dicts (Found {len(structured_elements)})]")
        for i, element_obj in enumerate(structured_elements):
            if _PYDANTIC_MODELS_AVAILABLE_HTML and hasattr(element_obj, 'model_dump_json'):
                logger.info(f"  Element {i+1}: {element_obj.model_dump_json(indent=2, exclude_none=True)}")
            else: 
                logger.info(f"  Element {i+1} (raw dict): {json.dumps(element_obj, indent=2, ensure_ascii=False)}")

        linear_text = generate_parsed_text_from_html_elements(structured_elements)
        logger.info("\n[Generated Linear Parsed Text from HTML Elements]")
        logger.info(linear_text)
        
        if _PYDANTIC_MODELS_AVAILABLE_HTML:
            doc_output = ParsedDocumentOutput(
                parsed_text=linear_text,
                elements=structured_elements,
                original_metadata={"filename": os.path.basename(file_path), "source_path": file_path}
            )
            return doc_output
        else:
            logger.warning("Returning raw dictionary for HTML as Pydantic models are not available.")
            return {
                "parsed_text": linear_text,
                "elements": structured_elements,
                "original_metadata": {"filename": os.path.basename(file_path), "source_path": file_path}
            }

    except Exception as e:
        logger.error(f"Error processing file {file_path} with BeautifulSoup4: {e}", exc_info=True)
        return None

if __name__ == "__main__":
    if not os.path.isdir(SAMPLE_DOCS_DIR):
        logger.error(f"Sample documents directory '{SAMPLE_DOCS_DIR}' not found relative to script. Full path expected: {os.path.abspath(SAMPLE_DOCS_DIR)}")
    else:
        html_files = sorted([f for f in os.listdir(SAMPLE_DOCS_DIR) 
                             if (f.lower().endswith(".html") or f.lower().endswith(".htm")) and not f.startswith("~$")])
        if not html_files:
            logger.warning(f"No .html or .htm files found in '{SAMPLE_DOCS_DIR}'.")
        else:
            for filename in html_files:
                file_path = os.path.join(SAMPLE_DOCS_DIR, filename)
                parsed_doc_result = analyze_html_with_bs4(file_path)
                if parsed_doc_result:
                    if _PYDANTIC_MODELS_AVAILABLE_HTML:
                        logger.info(f"\nSuccessfully created ParsedDocumentOutput for {filename}")
                    else:
                        logger.info(f"\nSuccessfully created fallback dictionary output for {filename}")
                else:
                    logger.error(f"Failed to create ParsedDocumentOutput for {filename}")