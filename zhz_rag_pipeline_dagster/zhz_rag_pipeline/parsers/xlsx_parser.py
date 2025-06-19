# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/zhz_rag_pipeline/parsers/xlsx_parser.py
import os
import logging
from typing import List, Dict, Any, Optional, Union
import pandas as pd

# 确保安装了 'pandas', 'openpyxl', 'tabulate'
# pip install pandas openpyxl tabulate

# --- Pydantic 模型导入和占位符定义 (保持不变) ---
_PYDANTIC_MODELS_AVAILABLE_XLSX = False
try:
    from ..pydantic_models_dagster import (
        ParsedDocumentOutput, TableElement, DocumentElementMetadata
    )
    _PYDANTIC_MODELS_AVAILABLE_XLSX = True
except ImportError:
    class BaseModel: pass
    class DocumentElementMetadata(BaseModel): page_number: Optional[int] = None
    class ParsedDocumentOutput(BaseModel): parsed_text: str; elements: list; original_metadata: dict; summary: Optional[str] = None
    class TableElement(BaseModel): element_type: str = "table"; markdown_representation: Optional[str] = None; html_representation: Optional[str] = None; caption: Optional[str] = None; metadata: Optional[DocumentElementMetadata] = None

logger = logging.getLogger(__name__)

# --- 辅助函数 (保持不变) ---
def _create_xlsx_element_metadata(sheet_index: int, table_index_in_sheet: int) -> Optional[Union[DocumentElementMetadata, Dict[str, Any]]]:
    meta_data_dict: Dict[str, Any] = {
        'page_number': sheet_index, # 使用 page_number 表示工作表索引
        'custom_properties': {'table_index_in_sheet': table_index_in_sheet}
    }
    if _PYDANTIC_MODELS_AVAILABLE_XLSX:
        return DocumentElementMetadata(**meta_data_dict)
    else:
        return meta_data_dict

# --- 主解析函数 (全新版本) ---
def parse_xlsx_to_structured_output(
    file_path: str,
    original_metadata: Dict[str, Any]
) -> Optional[Union[ParsedDocumentOutput, Dict[str, Any]]]:
    """
    V2: 解析XLSX文件，智能识别并提取每个工作表内的多个独立表格。
    """
    logger.info(f"Attempting to parse XLSX file with multi-table support: {file_path}")
    
    try:
        xls = pd.ExcelFile(file_path)
    except Exception as e:
        logger.error(f"Failed to open Excel file {file_path}: {e}", exc_info=True)
        return None

    all_elements: List[Any] = []
    
    for sheet_idx, sheet_name in enumerate(xls.sheet_names):
        try:
            # 读取整个工作表，不指定表头，以便我们手动查找
            df_full = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            
            if df_full.empty:
                logger.info(f"Sheet '{sheet_name}' is empty. Skipping.")
                continue

            # 通过查找空行来识别表格块
            is_row_empty = df_full.isnull().all(axis=1)
            # 获取空行的索引
            empty_row_indices = is_row_empty[is_row_empty].index.tolist()
            
            table_blocks: List[pd.DataFrame] = []
            last_split_index = -1
            
            for empty_idx in empty_row_indices:
                block_df = df_full.iloc[last_split_index + 1 : empty_idx].dropna(how='all')
                if not block_df.empty:
                    table_blocks.append(block_df)
                last_split_index = empty_idx
            
            # 添加最后一个块（从最后一个空行到末尾）
            final_block_df = df_full.iloc[last_split_index + 1 :].dropna(how='all')
            if not final_block_df.empty:
                table_blocks.append(final_block_df)

            logger.info(f"Sheet '{sheet_name}' was split into {len(table_blocks)} potential table blocks.")

            for table_idx, block_df in enumerate(table_blocks):
                # 将第一行作为表头
                header = block_df.iloc[0]
                table_data = block_df[1:]
                table_data.columns = header
                
                md_representation = table_data.to_markdown(index=False)
                table_caption = f"内容来自文件 '{os.path.basename(file_path)}' 的工作表 '{sheet_name}' (表格 {table_idx + 1})"
                
                element_metadata = _create_xlsx_element_metadata(
                    sheet_index=sheet_idx, 
                    table_index_in_sheet=table_idx
                )

                if _PYDANTIC_MODELS_AVAILABLE_XLSX:
                    table_el = TableElement(
                        markdown_representation=md_representation,
                        caption=table_caption,
                        metadata=element_metadata
                    )
                    all_elements.append(table_el)
                else:
                    all_elements.append({
                        "element_type": "table",
                        "markdown_representation": md_representation,
                        "caption": table_caption,
                        "metadata": element_metadata
                    })
                logger.info(f"  Successfully created TableElement for table {table_idx+1} in sheet '{sheet_name}'.")
                logger.debug(f"    - Table {table_idx+1} content preview: {md_representation[:200].replace(chr(10), ' ')}...")


        except Exception as e_sheet:
            logger.error(f"Failed to process sheet '{sheet_name}' in {file_path}: {e_sheet}", exc_info=True)
            continue
    
    if _PYDANTIC_MODELS_AVAILABLE_XLSX:
        return ParsedDocumentOutput(
            parsed_text="",
            elements=all_elements,
            original_metadata=original_metadata,
            summary=f"从文件 '{os.path.basename(file_path)}' 中解析了 {len(all_elements)} 个独立的表格。"
        )
    else:
        return {
            "parsed_text": "",
            "elements": all_elements,
            "original_metadata": original_metadata,
            "summary": f"从文件 '{os.path.basename(file_path)}' 中解析了 {len(all_elements)} 个独立的表格。"
        }