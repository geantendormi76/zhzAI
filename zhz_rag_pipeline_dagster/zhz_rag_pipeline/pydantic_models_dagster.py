# zhz_rag_pipeline/pydantic_models_dagster.py
from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel
import uuid

class LoadedDocumentOutput(BaseModel):
    document_path: str
    file_type: str
    raw_content: Union[str, bytes]
    metadata: Dict[str, Any]

class ParsedDocumentOutput(BaseModel):
    parsed_text: str
    document_structure: Optional[Dict[str, Any]] = None
    original_metadata: Dict[str, Any]

class ChunkOutput(BaseModel):
    chunk_id: str = "" 
    chunk_text: str
    source_document_id: str 
    chunk_metadata: Dict[str, Any]

    def __init__(self, **data: Any):
        if 'chunk_id' not in data or not data['chunk_id']:
            data['chunk_id'] = str(uuid.uuid4())
        super().__init__(**data)

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
    head_entity_label: str # 例如 "PERSON"
    relation_type: str    # 例如 "WORKS_AT"
    tail_entity_text: str
    tail_entity_label: str # 例如 "ORGANIZATION"

class KGTripleSetOutput(BaseModel):
    chunk_id: str 
    extracted_entities: List[ExtractedEntity] = []
    extracted_relations: List[ExtractedRelation] = [] 
    extraction_model_name: str 
    original_chunk_metadata: Dict[str, Any]