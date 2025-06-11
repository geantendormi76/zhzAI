import os
# 使用通用的 partition 函数，它能自动检测文件类型
from unstructured.partition.auto import partition
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SAMPLE_DOCS_DIR = "sample_docs"

def analyze_pdf_with_unstructured(file_path):
    logger.info(f"\n--- Analyzing PDF with Unstructured: {file_path} ---")
    try:
        # 使用通用的 partition 函数，让它自动检测类型
        # strategy="hi_res" 尝试更高质量的解析
        # infer_table_structure=True 尝试更好地解析表格
        # pdf_infer_table_structure=True 是专门针对PDF的表格推断（较新版本Unstructured可能推荐这个）
        elements = partition(
            filename=file_path, 
            strategy="fast", # <--- 修改为 "fast" 策略
            # infer_table_structure=True, # "fast" 策略下，表格推断可能有限或不同
            # pdf_infer_table_structure=True # 同上
        )

        logger.info(f"\n[Extracted Elements (Found {len(elements)} elements)]")
        for i, element in enumerate(elements):
            element_category = type(element).__name__
            text_preview = str(element.text)[:150] + ('...' if len(str(element.text)) > 150 else '')
            
            logger.info(f"  Element {i+1}: Category='{element_category}', Text='{text_preview}'")
            
            if hasattr(element, 'metadata'):
                meta_to_log = {
                    "source_filename": getattr(element.metadata, 'filename', 'N/A'),
                    "page_number": getattr(element.metadata, 'page_number', None),
                    "parent_id": getattr(element.metadata, 'parent_id', None),
                    "category_depth": getattr(element.metadata, 'category_depth', None),
                    "languages": getattr(element.metadata, 'languages', [])
                }
                if element_category == "Table" and hasattr(element.metadata, "text_as_html") and element.metadata.text_as_html:
                    meta_to_log["table_html_preview"] = str(element.metadata.text_as_html)[:200] + "..."
                
                logger.info(f"    Metadata: {meta_to_log}")
            else:
                logger.info(f"    Metadata: Not available for this element type ({element_category})")

    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
    except ImportError as ie:
        logger.error(f"ImportError processing {file_path} with Unstructured: {ie}. Did you install 'unstructured[pdf]' and its dependencies?")
    except Exception as e:
        logger.error(f"Error processing file {file_path} with Unstructured: {e}", exc_info=True)

if __name__ == "__main__":
    if not os.path.isdir(SAMPLE_DOCS_DIR):
        logger.error(f"Sample documents directory '{SAMPLE_DOCS_DIR}' not found.")
    else:
        pdf_files = sorted([f for f in os.listdir(SAMPLE_DOCS_DIR) if f.lower().endswith(".pdf") and not f.startswith("~$")])
        if not pdf_files:
            logger.warning(f"No .pdf files found in '{SAMPLE_DOCS_DIR}'.")
        else:
            for filename in pdf_files:
                file_path = os.path.join(SAMPLE_DOCS_DIR, filename)
                analyze_pdf_with_unstructured(file_path)