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

    try:
        # Read all sheets into a dictionary of DataFrames
        xls = pd.read_excel(file_path, sheet_name=None, header=0) 
        logger.info(f"Pandas opened XLSX. Sheets: {list(xls.keys())}")

        for sheet_idx, (sheet_name, df) in enumerate(xls.items()):
            if df.empty:
                logger.info(f"Sheet '{sheet_name}' (index {sheet_idx}) is empty. Skipping.")
                continue

            # -- 核心修改点：将每个sheet作为一个独立的TableElement --
            try:
                # 优先使用Markdown格式，因为它对LLM最友好
                # `tabulate` 库是 to_markdown 的一个依赖，需要确保已安装
                md_representation = df.to_markdown(index=False)
                
                # 创建一个描述性的标题，包含文件名和工作表名
                table_caption = f"Content from sheet '{sheet_name}' in file '{os.path.basename(file_path)}'."

                element_metadata = _create_xlsx_element_metadata(sheet_index=sheet_idx)

                if _PYDANTIC_MODELS_AVAILABLE_XLSX:
                    table_el = TableElement(
                        markdown_representation=md_representation,
                        caption=table_caption,
                        metadata=element_metadata # type: ignore
                    )
                    elements.append(table_el)
                else:
                    elements.append({
                        "element_type": "table",
                        "markdown_representation": md_representation,
                        "caption": table_caption,
                        "metadata": element_metadata
                    })
                logger.info(f"Successfully created a TableElement for sheet '{sheet_name}'.")

            except Exception as e_markdown:
                logger.error(f"Failed to convert sheet '{sheet_name}' to Markdown. It will be skipped. Error: {e_markdown}. Ensure 'tabulate' is installed (`pip install tabulate`).")
                # 如果转换失败，我们可以选择跳过这个sheet或记录一个错误元素
                continue # 这里选择跳过
        
        # -- 核心修改点：不再生成一个巨大的、拼接的 parsed_text --
        # 我们只返回一个包含所有独立表格元素的列表
        if _PYDANTIC_MODELS_AVAILABLE_XLSX:
            return ParsedDocumentOutput(
                parsed_text="",  # 主文本字段留空
                elements=elements, # type: ignore
                original_metadata=original_metadata,
                summary=f"Parsed {len(elements)} sheets as structured tables from {os.path.basename(file_path)}."
            )
        else:
            return {
                "parsed_text": "",
                "elements": elements,
                "original_metadata": original_metadata,
                "summary": f"Parsed {len(elements)} sheets as structured tables from {os.path.basename(file_path)}."
            }

    except FileNotFoundError:
        logger.error(f"XLSX file not found: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error parsing XLSX file {file_path} with pandas: {e}", exc_info=True)
        return None