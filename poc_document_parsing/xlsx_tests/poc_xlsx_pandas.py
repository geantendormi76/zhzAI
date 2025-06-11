import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SAMPLE_DOCS_DIR = "sample_docs"

def analyze_xlsx_with_pandas(file_path):
    logger.info(f"\n--- Analyzing XLSX with pandas: {file_path} ---")
    try:
        # pandas.read_excel() 可以读取所有工作表，如果 sheet_name=None
        # 或者指定特定的工作表名称或索引
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        logger.info(f"  Workbook '{os.path.basename(file_path)}' loaded. Sheet names: {sheet_names}")

        for sheet_name in sheet_names:
            logger.info(f"\n  --- Processing Sheet: {sheet_name} ---")
            try:
                # header=0 表示第一行为表头，尝试推断
                # 对于可能有多个表格或复杂表头的sheet，可能需要更复杂的读取逻辑
                # 或者先读取整个sheet，再进行后处理
                df = pd.read_excel(xls, sheet_name=sheet_name, header=0) 
                
                # 打印原始DataFrame信息 (少量行)
                logger.info(f"  DataFrame shape: {df.shape}")
                logger.info(f"  DataFrame columns: {df.columns.tolist()}")
                logger.info(f"  DataFrame head:\n{df.head().to_string()}")

                # 策略1: 将DataFrame转换为Markdown格式的文本
                # pandas 自带 to_markdown() 方法 (需要 tabulate 库)
                try:
                    markdown_text = df.to_markdown(index=False) # index=False 避免写入DataFrame的索引
                    logger.info("\n  [Content as Markdown (via pandas.to_markdown)]")
                    logger.info(f"Sheet: {sheet_name}\n{markdown_text}")
                except Exception as e_md:
                    logger.warning(f"    Could not convert DataFrame to Markdown (is 'tabulate' library installed?): {e_md}")
                    # 如果 to_markdown 失败，可以尝试 to_string 作为备选
                    string_representation = df.to_string(index=False)
                    logger.info("\n  [Content as String (via pandas.to_string)]")
                    logger.info(f"Sheet: {sheet_name}\n{string_representation}")


                # 策略2: 每行转换为描述性文本 (基于DataFrame)
                descriptive_text = f"Sheet: {sheet_name}\n"
                if not df.empty:
                    headers = df.columns.tolist()
                    for row_idx, row_series in df.iterrows():
                        descriptive_text += f"Row {row_idx + 1}: " # pandas iterrows 索引可能不从0开始，但通常是
                        parts = []
                        for col_name in headers:
                            cell_value = str(row_series[col_name] or "").strip()
                            parts.append(f"'{col_name}' is '{cell_value}'")
                        descriptive_text += "; ".join(parts) + ".\n"
                
                logger.info("\n  [Content as Descriptive Text (per row, from DataFrame)]")
                logger.info(descriptive_text)

            except Exception as e_sheet:
                logger.error(f"    Error processing sheet '{sheet_name}': {e_sheet}", exc_info=True)
                # 尝试读取为纯文本块，如果作为DataFrame读取失败
                try:
                    df_raw = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                    raw_text_content = "\n".join(["\t".join(map(str, row)) for index, row in df_raw.iterrows()])
                    logger.info(f"    Raw text content from sheet '{sheet_name}' (fallback):\n{raw_text_content[:500]}...")
                except Exception as e_raw_read:
                     logger.error(f"    Failed to read sheet '{sheet_name}' even as raw data: {e_raw_read}")


    except Exception as e:
        logger.error(f"Error processing file {file_path} with pandas: {e}", exc_info=True)

if __name__ == "__main__":
    if not os.path.isdir(SAMPLE_DOCS_DIR):
        logger.error(f"Sample documents directory '{SAMPLE_DOCS_DIR}' not found.")
    else:
        xlsx_files = sorted([f for f in os.listdir(SAMPLE_DOCS_DIR) if f.lower().endswith(".xlsx") and not f.startswith("~$")])
        if not xlsx_files:
            logger.warning(f"No .xlsx files found in '{SAMPLE_DOCS_DIR}'.")
        else:
            for filename in xlsx_files:
                file_path = os.path.join(SAMPLE_DOCS_DIR, filename)
                analyze_xlsx_with_pandas(file_path)