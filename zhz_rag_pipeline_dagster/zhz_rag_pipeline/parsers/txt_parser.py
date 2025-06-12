# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/zhz_rag_pipeline/parsers/txt_parser.py
import os
import logging
from typing import Dict, Any, Optional, Union

_PYDANTIC_MODELS_AVAILABLE_TXT = False
try:
    from ..pydantic_models_dagster import (
        ParsedDocumentOutput, NarrativeTextElement,
        DocumentElementMetadata
    )
    _PYDANTIC_MODELS_AVAILABLE_TXT = True
except ImportError:
    class BaseModel: pass
    class DocumentElementMetadata(BaseModel): page_number: Optional[int] = None # Not really applicable for txt
    class ParsedDocumentOutput(BaseModel): parsed_text: str; elements: list; original_metadata: dict; summary: Optional[str] = None
    class NarrativeTextElement(BaseModel): element_type:str="narrative_text"; text:str; metadata: Optional[DocumentElementMetadata] = None
    # DocumentElementType is not strictly needed here as we only create NarrativeTextElement

logger = logging.getLogger(__name__)

def parse_txt_to_structured_output(
    txt_content_str: str, # For .txt, we expect the content string directly
    original_metadata: Dict[str, Any]
) -> Optional[Union[ParsedDocumentOutput, Dict[str, Any]]]:
    logger.info(f"Attempting to parse TXT content (length: {len(txt_content_str)} chars)")

    # For .txt files, the entire content is treated as a single narrative text block.
    # No complex structure is assumed or extracted.
    
    elements = []
    element_metadata = None # No specific sub-element metadata for a single block txt file

    if _PYDANTIC_MODELS_AVAILABLE_TXT:
        elements.append(NarrativeTextElement(text=txt_content_str, metadata=element_metadata)) # type: ignore
        doc_output = ParsedDocumentOutput(
            parsed_text=txt_content_str,
            elements=elements, # type: ignore
            original_metadata=original_metadata
        )
    else:
        elements.append({"element_type": "narrative_text", "text": txt_content_str, "metadata": element_metadata})
        doc_output = {
            "parsed_text": txt_content_str,
            "elements": elements,
            "original_metadata": original_metadata
        }
    
    logger.info(f"Successfully processed TXT content into a single element.")
    return doc_output