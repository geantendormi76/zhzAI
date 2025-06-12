# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/zhz_rag_pipeline/parsers/html_parser.py
import os
import logging
from typing import List, Dict, Any, Optional, Union, Set
import re

try:
    from bs4 import BeautifulSoup, Tag, NavigableString
    _BS4_AVAILABLE = True
    logging.info("Successfully imported BeautifulSoup4 for HTML parsing.")
except ImportError:
    logging.error("BeautifulSoup4 (bs4) not found. HTML parsing will not be available.")
    _BS4_AVAILABLE = False
    class BeautifulSoup: pass # Placeholder
    class Tag: pass
    class NavigableString: pass


_PYDANTIC_MODELS_AVAILABLE_HTML = False
try:
    from ..pydantic_models_dagster import (
        ParsedDocumentOutput, DocumentElementType, TitleElement, NarrativeTextElement,
        ListItemElement, TableElement, CodeBlockElement, PageBreakElement,
        DocumentElementMetadata
    )
    _PYDANTIC_MODELS_AVAILABLE_HTML = True
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

# Tags to typically ignore for main content extraction
IGNORE_TAGS_HTML = ['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'meta', 'link', 'button', 'input', 'noscript', 'iframe', 'canvas', 'svg', 'path']
# Tags that define a semantic block but we want to process their children
CONTAINER_TAGS_HTML = ['div', 'section', 'article', 'main', 'body', 'figure', 'figcaption', 'details', 'summary']


def _table_to_markdown(table_tag: Tag) -> str:
    """Converts a BeautifulSoup table Tag to a Markdown string."""
    md_rows = []
    header_processed = False
    
    # Process header (thead)
    thead = table_tag.find('thead')
    if thead:
        header_rows_tags = thead.find_all('tr')
        for hr_tag in header_rows_tags:
            header_cells = hr_tag.find_all(['th', 'td'])
            if header_cells:
                header_texts = [cell.get_text(separator=' ', strip=True) for cell in header_cells]
                md_rows.append("| " + " | ".join(header_texts) + " |")
                if not header_processed: # Add separator only after the first header row group
                    md_rows.append("| " + " | ".join(["---"] * len(header_texts)) + " |")
                    header_processed = True
    
    # Process body (tbody or direct tr in table)
    tbody = table_tag.find('tbody')
    if not tbody: # If no tbody, look for tr directly under table
        rows_to_process = table_tag.find_all('tr', recursive=False)
    else:
        rows_to_process = tbody.find_all('tr')
        
    for row_tag in rows_to_process:
        # Skip if this row was already processed as part of thead (if thead was missing)
        if not header_processed and row_tag.find('th'):
            header_cells = row_tag.find_all(['th', 'td'])
            header_texts = [cell.get_text(separator=' ', strip=True) for cell in header_cells]
            md_rows.append("| " + " | ".join(header_texts) + " |")
            md_rows.append("| " + " | ".join(["---"] * len(header_texts)) + " |")
            header_processed = True
            continue
        
        cell_texts = [cell.get_text(separator=' ', strip=True) for cell in row_tag.find_all('td')]
        if cell_texts: # Only add row if it has content
            md_rows.append("| " + " | ".join(cell_texts) + " |")
            
    return "\n".join(md_rows)

def _convert_html_tag_to_elements_recursive(tag: Tag, elements_list: List[Any], processed_tags: Set[Tag], current_list_level: int = 0):
    """
    Recursively processes a BeautifulSoup Tag and its children to extract structured elements.
    Modifies elements_list in place.
    """
    if tag in processed_tags or not isinstance(tag, Tag) or tag.name in IGNORE_TAGS_HTML:
        return

    tag_name = tag.name.lower()
    element_metadata = None # Placeholder for now, can be enhanced to include source line numbers etc.
    
    created_element = False

    if tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
        level = int(tag_name[1:])
        text = tag.get_text(strip=True)
        if text:
            if _PYDANTIC_MODELS_AVAILABLE_HTML: elements_list.append(TitleElement(text=text, level=level, metadata=element_metadata))
            else: elements_list.append({"element_type": "title", "text": text, "level": level, "metadata": element_metadata})
            created_element = True
    
    elif tag_name == 'p':
        text = tag.get_text(strip=True)
        if text:
            if _PYDANTIC_MODELS_AVAILABLE_HTML: elements_list.append(NarrativeTextElement(text=text, metadata=element_metadata))
            else: elements_list.append({"element_type": "narrative_text", "text": text, "metadata": element_metadata})
            created_element = True

    elif tag_name in ['ul', 'ol']:
        ordered = tag_name == 'ol'
        start_num = int(tag.get('start', '1')) if ordered else 1
        
        # Iterate over direct children that are <li>
        direct_li_children = [child for child in tag.children if isinstance(child, Tag) and child.name == 'li']
        for i, li_tag in enumerate(direct_li_children):
            if li_tag in processed_tags: continue
            
            # Extract text directly under <li>, excluding text from nested lists
            li_text_parts = []
            for content_child in li_tag.contents:
                if isinstance(content_child, NavigableString):
                    stripped_text = content_child.strip()
                    if stripped_text: li_text_parts.append(stripped_text)
                elif isinstance(content_child, Tag) and content_child.name not in ['ul', 'ol']: # Get text from non-list children
                    li_text_parts.append(content_child.get_text(strip=True))
            
            final_li_text = " ".join(li_text_parts).strip()

            if final_li_text:
                item_num_str = str(start_num + i) if ordered else None
                if _PYDANTIC_MODELS_AVAILABLE_HTML: elements_list.append(ListItemElement(text=final_li_text, level=current_list_level, ordered=ordered, item_number=item_num_str, metadata=element_metadata))
                else: elements_list.append({"element_type": "list_item", "text": final_li_text, "level": current_list_level, "ordered": ordered, "item_number": item_num_str, "metadata": element_metadata})
            
            processed_tags.add(li_tag) # Mark <li> as processed for its direct text
            # Recursively process children of this <li> for nested lists or other elements
            for child_of_li in li_tag.children:
                if isinstance(child_of_li, Tag):
                     _convert_html_tag_to_elements_recursive(child_of_li, elements_list, processed_tags, current_list_level + 1)
        created_element = True # The list itself is an element boundary

    elif tag_name == 'table':
        md_table = _table_to_markdown(tag)
        caption_tag = tag.find('caption')
        caption_text = caption_tag.get_text(strip=True) if caption_tag else None
        if md_table or caption_text : # Only add if table has content or caption
            if _PYDANTIC_MODELS_AVAILABLE_HTML: elements_list.append(TableElement(markdown_representation=md_table, html_representation=str(tag), caption=caption_text, metadata=element_metadata))
            else: elements_list.append({"element_type": "table", "markdown_representation": md_table, "html_representation": str(tag), "caption": caption_text, "metadata": element_metadata})
        created_element = True

    elif tag_name == 'pre':
        code_tag = tag.find('code')
        code_text, lang = "", None
        if code_tag:
            code_text = code_tag.get_text() # Keep original spacing and newlines
            lang_class = code_tag.get('class', [])
            if lang_class: lang = next((cls.split('language-')[-1] for cls in lang_class if cls.startswith('language-')), None)
        else:
            code_text = tag.get_text()
        
        if code_text.strip(): # Check if there's actual code
            if _PYDANTIC_MODELS_AVAILABLE_HTML: elements_list.append(CodeBlockElement(code=code_text.strip('\n'), language=lang, metadata=element_metadata))
            else: elements_list.append({"element_type": "code_block", "code": code_text.strip('\n'), "language": lang, "metadata": element_metadata})
        created_element = True

    elif tag_name == 'blockquote':
        text = tag.get_text(strip=True)
        if text:
            if _PYDANTIC_MODELS_AVAILABLE_HTML: elements_list.append(NarrativeTextElement(text=text, metadata=element_metadata))
            else: elements_list.append({"element_type": "narrative_text", "text": text, "_is_blockquote": True, "metadata": element_metadata})
        created_element = True

    elif tag_name == 'hr':
        if _PYDANTIC_MODELS_AVAILABLE_HTML: elements_list.append(PageBreakElement(metadata=element_metadata))
        else: elements_list.append({"element_type": "page_break", "metadata": element_metadata})
        created_element = True
    
    processed_tags.add(tag)
    # If the tag itself wasn't a specific block element we handled, or it's a known container,
    # process its children.
    if not created_element or tag_name in CONTAINER_TAGS_HTML:
        for child in tag.children:
            if isinstance(child, Tag):
                _convert_html_tag_to_elements_recursive(child, elements_list, processed_tags, current_list_level)
            elif isinstance(child, NavigableString): # Handle loose text not in <p> etc.
                loose_text = child.strip()
                if loose_text and tag_name not in ['ul', 'ol']: # Avoid adding list item text twice
                    if _PYDANTIC_MODELS_AVAILABLE_HTML: elements_list.append(NarrativeTextElement(text=loose_text, metadata=element_metadata))
                    else: elements_list.append({"element_type": "narrative_text", "text": loose_text, "_is_loose_text": True, "metadata": element_metadata})

def _generate_linear_text_from_html_elements(elements: List[Any]) -> str:
    # This function is identical to the one in docx_parser.py, can be refactored to common_utils later.
    text_parts = []
    for el_data_any in elements:
        el_data: Dict[str, Any] = {}
        if _PYDANTIC_MODELS_AVAILABLE_HTML and hasattr(el_data_any, 'model_dump'):
            el_data = el_data_any.model_dump(exclude_none=True)
        elif isinstance(el_data_any, dict):
            el_data = el_data_any
        else: continue

        el_type = el_data.get("element_type")
        text_content = el_data.get("text", "")
        current_element_text = ""
        if el_type == "title": current_element_text = f"\n{'#' * el_data.get('level',1)} {text_content}\n"
        elif el_type == "narrative_text": current_element_text = text_content + "\n"
        elif el_type == "list_item":
            prefix = f"{el_data.get('item_number', '')}. " if el_data.get('ordered') and el_data.get('item_number') else "- "
            indent = "  " * el_data.get('level', 0)
            current_element_text = f"{indent}{prefix}{text_content}\n"
        elif el_type == "table":
            caption = el_data.get('caption','Unnamed Table')
            md_repr = el_data.get('markdown_representation')
            if md_repr: current_element_text = f"\n[Table: {caption}]\n{md_repr}\n"
        elif el_type == "code_block":
            lang = el_data.get('language', "") or ""
            code_content = el_data.get('code', "")
            current_element_text = f"\n```{lang}\n{code_content}\n```\n"
        elif el_type == "page_break": current_element_text = "\n---\n"
        if current_element_text: text_parts.append(current_element_text)
    full_text = "".join(text_parts)
    return re.sub(r'\n{3,}', '\n\n', full_text).strip()

def parse_html_to_structured_output(
    html_content_str: str, 
    original_metadata: Dict[str, Any]
) -> Optional[Union[ParsedDocumentOutput, Dict[str, Any]]]:
    logger.info(f"Attempting to parse HTML content (length: {len(html_content_str)} chars) using BeautifulSoup4")
    if not _BS4_AVAILABLE:
        logger.error("BeautifulSoup4 (bs4) is not available. HTML parsing cannot proceed.")
        return None

    elements: List[Any] = []
    try:
        # Try lxml first, then html.parser
        try:
            soup = BeautifulSoup(html_content_str, "lxml")
        except Exception: # Fallback if lxml is not installed or fails
            logger.warning("lxml parser not available or failed, falling back to html.parser for HTML.")
            soup = BeautifulSoup(html_content_str, "html.parser")

        # Attempt to find the main content area
        main_content_area = soup.find('article') or soup.find('main') or soup.body
        if not main_content_area:
            logger.warning("Could not find <article>, <main>, or <body> tag. Parsing entire document if possible.")
            main_content_area = soup # Fallback to entire soup object

        # Remove ignored tags before processing
        for ignore_tag_name in IGNORE_TAGS_HTML:
            for tag_to_remove in main_content_area.find_all(ignore_tag_name):
                tag_to_remove.decompose()
        
        processed_tags_set: Set[Tag] = set()
        _convert_html_tag_to_elements_recursive(main_content_area, elements, processed_tags_set)
        
        logger.info(f"Converted HTML to {len(elements)} custom elements.")
        
        linear_text = _generate_linear_text_from_html_elements(elements)
        logger.info(f"Generated linear text from HTML elements (length: {len(linear_text)}). Preview: {linear_text[:200]}")

        if _PYDANTIC_MODELS_AVAILABLE_HTML:
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
    except Exception as e:
        logger.error(f"Error parsing HTML content with BeautifulSoup4: {e}", exc_info=True)
        return None