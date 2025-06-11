import os
import fitz  # PyMuPDF
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SAMPLE_DOCS_DIR = "sample_docs"

def analyze_pdf_with_pymupdf(file_path):
    logger.info(f"\n--- Analyzing PDF with PyMuPDF: {file_path} ---")
    try:
        doc = fitz.open(file_path)
        logger.info(f"  Document '{os.path.basename(file_path)}': {doc.page_count} page(s).")

        full_text = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            
            # 尝试提取纯文本 (默认提取顺序)
            text_simple = page.get_text("text")
            full_text.append(f"\n--- Page {page_num + 1} (Simple Text Extraction) ---\n{text_simple.strip()}")

            # 尝试按阅读顺序提取文本块 (更智能)
            # blocks = page.get_text("blocks", sort=True) # sort=True 尝试按阅读顺序
            blocks = page.get_text("dict", sort=True)["blocks"] # 获取更详细的块信息
            
            page_block_texts = []
            if blocks:
                logger.info(f"  Page {page_num + 1} - Extracted {len(blocks)} text blocks (sorted):")
                for i, block in enumerate(blocks):
                    if block['type'] == 0: # 0 indicates a text block
                        block_text_lines = []
                        for line in block.get("lines", []):
                            line_content = "".join([span.get("text", "") for span in line.get("spans", [])])
                            block_text_lines.append(line_content)
                        block_text = "\n".join(block_text_lines)
                        logger.info(f"    Block {i} (bbox: {block.get('bbox')}): \"{block_text[:100].replace(chr(10), ' ')}{'...' if len(block_text) > 100 else ''}\"")
                        page_block_texts.append(block_text)
            
            full_text.append(f"\n--- Page {page_num + 1} (Sorted Blocks Extraction) ---\n" + "\n".join(page_block_texts))


            # 尝试查找表格 (PyMuPDF 的原生表格查找能力有限，但可以尝试)
            # 对于纯文本PDF，表格通常需要基于文本的启发式方法或更高级的工具
            tables = page.find_tables()
            if tables.tables: # tables 是一个 TableFinder 对象
                logger.info(f"  Page {page_num + 1} - Found {len(tables.tables)} potential table(s):")
                for table_idx, tab in enumerate(tables): # 迭代找到的表
                    logger.info(f"    Table {table_idx+1} (bbox: {tab.bbox}):")
                    # tab.extract() 返回一个列表的列表
                    extracted_table_data = tab.extract()
                    if extracted_table_data:
                        for row_idx, row_data in enumerate(extracted_table_data):
                            logger.info(f"      Row {row_idx+1}: {row_data}")
                    else:
                        logger.info("      Could not extract data from this table object.")
            else:
                logger.info(f"  Page {page_num + 1} - No tables automatically detected by PyMuPDF's find_tables().")


        logger.info("\n[Full Extracted Text (Concatenated from sorted blocks)]")
        # 打印部分合并后的文本以供概览
        concatenated_text_from_blocks = "".join(full_text) # 简单合并，实际分块时会更细致
        logger.info(concatenated_text_from_blocks[:1000] + ('...' if len(concatenated_text_from_blocks) > 1000 else ''))

        doc.close()
    except Exception as e:
        logger.error(f"Error processing file {file_path} with PyMuPDF: {e}", exc_info=True)

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
                analyze_pdf_with_pymupdf(file_path)