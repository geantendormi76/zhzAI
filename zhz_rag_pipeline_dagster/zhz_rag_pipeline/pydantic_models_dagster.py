# zhz_rag_pipeline/pydantic_models_dagster.py
from typing import List, Dict, Any, Union, Optional, Literal
# --- 修改：从 pydantic 导入 BaseModel 和 Field ---
from pydantic import BaseModel, Field
# --- 修改结束 ---
import uuid
# from typing import List # 这行是多余的，因为上面已经从 typing 导入了 List

class LoadedDocumentOutput(BaseModel):
    document_path: str
    file_type: str
    raw_content: Union[str, bytes]
    metadata: Dict[str, Any]

# --- 修改：在 ParsedDocumentOutput 定义之前定义其依赖的 Element 类型 ---
class DocumentElementMetadata(BaseModel):
    """通用元数据，可附加到任何文档元素上"""
    page_number: Optional[int] = None
    source_coordinates: Optional[Dict[str, float]] = None # 例如，PDF中的bbox
    custom_properties: Optional[Dict[str, Any]] = None # 其他特定于元素的属性

class TitleElement(BaseModel):
    element_type: Literal["title"] = "title"
    text: str
    level: int # 例如 1 代表 H1, 2 代表 H2
    metadata: Optional[DocumentElementMetadata] = None

class NarrativeTextElement(BaseModel): # 普通段落文本
    element_type: Literal["narrative_text"] = "narrative_text"
    text: str
    metadata: Optional[DocumentElementMetadata] = None

class ListItemElement(BaseModel):
    element_type: Literal["list_item"] = "list_item"
    text: str
    level: int = 0 # 列表嵌套层级，0代表顶层列表项
    ordered: bool = False # True代表有序列表项, False代表无序
    item_number: Optional[Union[int, str]] = None # 例如 "1", "a", "*"
    metadata: Optional[DocumentElementMetadata] = None

class TableElement(BaseModel):
    element_type: Literal["table"] = "table"
    text_representation: Optional[str] = None 
    markdown_representation: Optional[str] = None
    html_representation: Optional[str] = None
    caption: Optional[str] = None
    metadata: Optional[DocumentElementMetadata] = None

class CodeBlockElement(BaseModel):
    element_type: Literal["code_block"] = "code_block"
    code: str
    language: Optional[str] = None
    metadata: Optional[DocumentElementMetadata] = None

class ImageElement(BaseModel): 
    element_type: Literal["image"] = "image"
    alt_text: Optional[str] = None
    caption: Optional[str] = None
    metadata: Optional[DocumentElementMetadata] = None

class PageBreakElement(BaseModel):
    element_type: Literal["page_break"] = "page_break"
    metadata: Optional[DocumentElementMetadata] = None
    
class HeaderElement(BaseModel):
    element_type: Literal["header"] = "header"
    text: str
    metadata: Optional[DocumentElementMetadata] = None

class FooterElement(BaseModel):
    element_type: Literal["footer"] = "footer"
    text: str
    metadata: Optional[DocumentElementMetadata] = None

DocumentElementType = Union[
    TitleElement, 
    NarrativeTextElement, 
    ListItemElement, 
    TableElement, 
    CodeBlockElement,
    ImageElement,
    PageBreakElement,
    HeaderElement,
    FooterElement
]

class ParsedDocumentOutput(BaseModel):
    parsed_text: str = Field(description="文档内容的线性化纯文本表示，尽可能保留语义。") 
    elements: List[DocumentElementType] = Field(default_factory=list, description="从文档中解析出的结构化元素列表。")
    original_metadata: Dict[str, Any] = Field(description="关于原始文档的元数据，如文件名、路径、大小等。")
    summary: Optional[str] = None
# --- 已有模型 ---
class ChunkOutput(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4())) # 确保 Field 被导入
    chunk_text: str
    source_document_id: str 
    chunk_metadata: Dict[str, Any]

class EmbeddingOutput(BaseModel):
    chunk_id: str 
    chunk_text: str 
    embedding_vector: List[float]
    embedding_model_name: str 
    original_chunk_metadata: Dict[str, Any]

class ExtractedEntity(BaseModel):
    text: str 
    label: str 

class ExtractedRelation(BaseModel):
    head_entity_text: str
    head_entity_label: str
    relation_type: str
    tail_entity_text: str
    tail_entity_label: str

class KGTripleSetOutput(BaseModel):
    chunk_id: str 
    extracted_entities: List[ExtractedEntity] = Field(default_factory=list)
    extracted_relations: List[ExtractedRelation] = Field(default_factory=list) 
    extraction_model_name: str 
    original_chunk_metadata: Dict[str, Any]