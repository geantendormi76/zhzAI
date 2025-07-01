# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/zhz_rag_pipeline/parsers/__init__.py
import logging # 添加 logging 导入
from typing import Callable, Dict, Any, Optional, Union

# 尝试导入 Pydantic 模型，如果失败，则类型别名使用 Any
try:
    from ..pydantic_models_dagster import ParsedDocumentOutput
    _ParserOutputType = Optional[ParsedDocumentOutput]
except ImportError:
    _ParserOutputType = Optional[Any] # Fallback

# 定义一个类型别名，表示解析函数的签名
# 输入可以是路径(str)或内容(str/bytes)，元数据字典，返回Pydantic模型或字典
ParserFunction = Callable[[Union[str, bytes], Dict[str, Any]], _ParserOutputType]

# 从各个解析器模块导入主解析函数
from .md_parser import parse_markdown_to_structured_output
from .docx_parser import parse_docx_to_structured_output
from .pdf_parser import parse_pdf_to_structured_output
from .xlsx_parser import parse_xlsx_to_structured_output
from .html_parser import parse_html_to_structured_output
from .txt_parser import parse_txt_to_structured_output

logger = logging.getLogger(__name__) # 添加 logger 实例

# 创建一个解析器注册表 (合并自 parser_dispatcher.py)
PARSER_REGISTRY: Dict[str, ParserFunction] = {
    ".md": parse_markdown_to_structured_output,
    ".docx": parse_docx_to_structured_output,
    ".pdf": parse_pdf_to_structured_output,
    ".xlsx": parse_xlsx_to_structured_output,
    ".html": parse_html_to_structured_output,
    ".htm": parse_html_to_structured_output,  # Alias for html
    ".txt": parse_txt_to_structured_output,
}

def dispatch_parsing( # 合并自 parser_dispatcher.py
    file_extension: str,
    content_or_path: Union[str, bytes], # 确保这里是 Union[str, bytes]
    original_metadata: Dict[str, Any]
) -> Optional[Any]: # 返回 Optional[Any] 以匹配下游期望
    parser_func = PARSER_REGISTRY.get(file_extension.lower())
    if parser_func:
        try:
            # 调用相应的解析函数
            # txt_parser 和 md_parser, html_parser 期望 content_str
            # docx_parser, pdf_parser, xlsx_parser 期望 file_path
            # content_or_path 变量在 ingestion_assets.py 中已经根据 file_ext 做了区分
            return parser_func(content_or_path, original_metadata)
        except Exception as e:
            logger.error(f"Error calling parser for '{file_extension}' on '{original_metadata.get('source_file_path', 'N/A')}': {e}", exc_info=True)
            return None # 解析失败返回 None
    else:
        logger.warning(f"No specific parser registered for file type '{file_extension}'.")
        # 尝试一个通用的纯文本提取作为最终回退（如果适用且有实现）
        # 或者直接返回None
        return None

def get_parser(file_extension: str) -> Optional[ParserFunction]: # 保留此函数以防其他地方用到
    return PARSER_REGISTRY.get(file_extension.lower())

__all__ = [
    "parse_markdown_to_structured_output",
    "parse_docx_to_structured_output",
    "parse_pdf_to_structured_output",
    "parse_xlsx_to_structured_output",
    "parse_html_to_structured_output",
    "parse_txt_to_structured_output",
    "get_parser", # 保留
    "dispatch_parsing", # 新增导出
    "PARSER_REGISTRY",
    "ParserFunction"
]