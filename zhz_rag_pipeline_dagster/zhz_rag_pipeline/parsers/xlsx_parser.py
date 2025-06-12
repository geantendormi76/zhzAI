# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/zhz_rag_pipeline/parsers/xlsx_parser.py
import os
import logging
from typing import List, Dict, Any, Optional, Union

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
    logging.info("Successfully imported pandas for XLSX parsing.")
except ImportError:
    logging.error("pandas library not found. XLSX parsing will not be available.")
    _PANDAS_AVAILABLE = False
    class pd: # Placeholder
        @staticmethod
        def read_excel(*args, **kwargs): raise NotImplementedError("pandas not available")
        class DataFrame:
            def to_markdown(self, *args, **kwargs): raise NotImplementedError("pandas not available")
            def to_string(self, *args, **kwargs): raise NotImplementedError("pandas not available")
            @property
            def empty(self): return True

_PYDANTIC_MODELS_AVAILABLE_XLSX = False
try:
    from ..pydantic_models_dagster import (
        ParsedDocumentOutput, DocumentElementType, TableElement,
        DocumentElementMetadata
    )
    _PYDANTIC_MODELS_AVAILABLE_XLSX = True
except ImportError:
    class BaseModel: pass
    class DocumentElementMetadata(BaseModel): page_number: Optional[int] = None # page_number for xlsx might be sheet index
    class ParsedDocumentOutput(BaseModel): parsed_text: str; elements: list; original_metadata: dict; summary: Optional[str] = None
    class TableElement(BaseModel): element_type:str="table"; markdown_representation:Optional[str]=None; html_representation:Optional[str]=None; caption:Optional[str]=None; metadata: Optional[DocumentElementMetadata] = None
    DocumentElementType = Any

logger = logging.getLogger(__name__)

def _create_xlsx_element_metadata(sheet_index: Optional[int] = None) -> Optional[Union[DocumentElementMetadata, Dict[str, Any]]]:
    meta_data_dict: Dict[str, Any] = {}
    if sheet_index is not None:
        # Using page_number to represent sheet index for consistency with other parsers
        meta_data_dict['page_number'] = sheet_index 
    
    if not meta_data_dict:
        return None

    if _PYDANTIC_MODELS_AVAILABLE_XLSX:
        return DocumentElementMetadata(**meta_data_dict)
    else:
        return meta_data_dict

def parse_xlsx_to_structured_output(
    file_path: str, 
    original_metadata: Dict[str, Any]
) -> Optional[Union[ParsedDocumentOutput, Dict[str, Any]]]:
    logger.info(f"Attempting to parse XLSX file: {file_path} using pandas")
    if not _PANDAS_AVAILABLE:
        logger.error("pandas library is not available. XLSX parsing cannot proceed.")
        return None

    elements: List[Any] = []
    all_sheets_text_parts: List[str] = []

    try:
        # Read all sheets into a dictionary of DataFrames
        # Setting header=0 to use the first row as column names.
        # If a sheet has no header, pandas will infer column names like 0, 1, 2...
        xls = pd.read_excel(file_path, sheet_name=None, header=0) 
        logger.info(f"Pandas opened XLSX. Sheets: {list(xls.keys())}")

        for sheet_idx, (sheet_name, df) in enumerate(xls.items()):
            if df.empty:
                logger.info(f"Sheet '{sheet_name}' (index {sheet_idx}) is empty. Skipping.")
                continue

            sheet_md_representation = None
            sheet_text_representation = None
            
            try:
                # Attempt to convert DataFrame to Markdown
                sheet_md_representation = df.to_markdown(index=False)
                all_sheets_text_parts.append(f"Sheet: {sheet_name}\n{sheet_md_representation}")
                logger.info(f"Successfully converted sheet '{sheet_name}' to Markdown.")
            except Exception as e_markdown:
                logger.warning(f"Failed to convert sheet '{sheet_name}' to Markdown (tabulate library might be missing): {e_markdown}. Falling back to string representation.")
                try:
                    sheet_text_representation = df.to_string(index=False)
                    all_sheets_text_parts.append(f"Sheet: {sheet_name}\n{sheet_text_representation}")
                    logger.info(f"Successfully converted sheet '{sheet_name}' to string representation.")
                except Exception as e_string:
                    logger.error(f"Failed to convert sheet '{sheet_name}' even to string: {e_string}")
                    all_sheets_text_parts.append(f"Sheet: {sheet_name}\n[Error converting sheet to text]")

            element_metadata = _create_xlsx_element_metadata(sheet_index=sheet_idx)
            if _PYDANTIC_MODELS_AVAILABLE_XLSX:
                table_el = TableElement(
                    markdown_representation=sheet_md_representation,
                    # html_representation can be df.to_html(index=False) if needed later
                    caption=str(sheet_name), # Ensure caption is string
                    metadata=element_metadata # type: ignore
                )
                elements.append(table_el)
            else:
                elements.append({
                    "element_type": "table",
                    "markdown_representation": sheet_md_representation,
                    "caption": str(sheet_name),
                    "metadata": element_metadata
                })
        
        linear_text = "\n\n\n".join(all_sheets_text_parts).strip() # Use more newlines to separate sheets

        if _PYDANTIC_MODELS_AVAILABLE_XLSX:
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
        logger.error(f"XLSX file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error parsing XLSX file {file_path} with pandas: {e}", exc_info=True)
        # Fallback: try to read as raw text if pandas fails catastrophically
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f_raw:
                raw_content = f_raw.read(5000) # Limit raw read
            error_text = f"[Failed to parse XLSX with pandas. Raw content preview (first 5000 chars)]:\n{raw_content}"
            if _PYDANTIC_MODELS_AVAILABLE_XLSX:
                return ParsedDocumentOutput(parsed_text=error_text, elements=[], original_metadata=original_metadata) # type: ignore
            else:
                return {"parsed_text": error_text, "elements": [], "original_metadata": original_metadata}
        except Exception as e_raw:
            logger.error(f"Failed to even read XLSX as raw text after pandas error: {e_raw}")
            return None