import os
from unstructured.partition.docx import partition_docx
# from unstructured.partition.auto import partition # 备选
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SAMPLE_DOCS_DIR = "sample_docs"

def analyze_docx_with_unstructured(file_path):
    logger.info(f"\n--- Analyzing DOCX with Unstructured: {file_path} ---")
    try:
        # 使用 partition_docx 直接处理 docx 文件
        # strategy="hi_res" 可能需要额外的依赖或配置，如果遇到问题可以先尝试去掉
        # infer_table_structure=True 尝试更好地解析表格
        elements = partition_docx(filename=file_path, strategy="hi_res", infer_table_structure=True)
        
        # 或者，如果上面的 partition_docx 仍然有问题，可以尝试更通用的 partition
        # from unstructured.partition.auto import partition
        # elements = partition(filename=file_path, strategy="hi_res", infer_table_structure=True)


        logger.info(f"\n[Extracted Elements (Found {len(elements)} elements)]")
        for i, element in enumerate(elements):
            # --- 修改：直接使用 element 的类型名作为其类别 ---
            element_category = type(element).__name__ 
            # --- 修改结束 ---
            
            text_preview = str(element.text)[:150] + ('...' if len(str(element.text)) > 150 else '')
            
            # --- 修改：日志中打印 element_category ---
            logger.info(f"  Element {i+1}: Category='{element_category}', Text='{text_preview}'")
            # --- 修改结束 ---
            
            if hasattr(element, 'metadata'):
                meta_to_log = {
                    "source_filename": getattr(element.metadata, 'filename', 'N/A'), # 使用 getattr 更安全
                    "page_number": getattr(element.metadata, 'page_number', None),
                    # "element_id": getattr(element.metadata, 'id', None), # Unstructured的id可能不是我们需要的
                    "parent_id": getattr(element.metadata, 'parent_id', None) 
                }
                
                # 尝试获取更具体的元数据，不同元素类型可能不同
                if hasattr(element.metadata, 'text_as_html') and element.metadata.text_as_html:
                    meta_to_log["table_html_preview"] = str(element.metadata.text_as_html)[:200] + "..."
                if hasattr(element.metadata, 'category_depth'): # 例如 Title 元素可能有
                    meta_to_log["category_depth"] = element.metadata.category_depth
                if hasattr(element.metadata, 'languages'):
                     meta_to_log["languages"] = element.metadata.languages
                
                # 打印所有可用的元数据键，帮助我们了解实际有哪些信息
                # logger.debug(f"    Available metadata keys: {vars(element.metadata).keys()}")

                logger.info(f"    Metadata: {meta_to_log}")
            else:
                logger.info(f"    Metadata: Not available for this element type ({element_category})")

    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
    except ImportError as ie:
        logger.error(f"ImportError processing {file_path} with Unstructured: {ie}. Did you install 'unstructured[docx]' and its dependencies (like pandoc)?")
    except Exception as e:
        logger.error(f"Error processing file {file_path} with Unstructured: {e}", exc_info=True)

if __name__ == "__main__":
    if not os.path.isdir(SAMPLE_DOCS_DIR):
        logger.error(f"Sample documents directory '{SAMPLE_DOCS_DIR}' not found.")
    else:
        docx_files = sorted([
        f for f in os.listdir(SAMPLE_DOCS_DIR) 
        if f.lower().endswith(".docx") and not f.startswith("~$")
    ])
        if not docx_files:
            logger.warning(f"No .docx files found in '{SAMPLE_DOCS_DIR}'.")
        else:
            for filename in docx_files:
                file_path = os.path.join(SAMPLE_DOCS_DIR, filename)
                analyze_docx_with_unstructured(file_path)