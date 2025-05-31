# zhz_rag_pipeline/definitions.py
import dagster as dg

from .ingestion_assets import all_ingestion_assets
from .processing_assets import all_processing_assets
from .evaluation_assets import all_evaluation_assets
from .resources import (
    SentenceTransformerResource, SentenceTransformerResourceConfig,
    ChromaDBResource, ChromaDBResourceConfig,
    SGLangAPIResource, SGLangAPIResourceConfig,
    Neo4jResource, Neo4jResourceConfig,GeminiAPIResource, 
    GeminiAPIResourceConfig 
)

all_project_assets = all_ingestion_assets + all_processing_assets + all_evaluation_assets 

defs = dg.Definitions(
    assets=all_project_assets, 
    resources={
        "embedder": SentenceTransformerResource(
            model_name_or_path=SentenceTransformerResourceConfig().model_name_or_path
        ),
        "chroma_db": ChromaDBResource(
            collection_name=ChromaDBResourceConfig().collection_name,
            persist_directory=ChromaDBResourceConfig().persist_directory
        ),
        "sglang_api": SGLangAPIResource(
            api_url=SGLangAPIResourceConfig().api_url,
            default_temperature=SGLangAPIResourceConfig().default_temperature,
            default_max_new_tokens=SGLangAPIResourceConfig().default_max_new_tokens
        ),
        "neo4j_res": Neo4jResource(
            uri=Neo4jResourceConfig().uri,
            user=Neo4jResourceConfig().user,
            password=Neo4jResourceConfig().password,
            database=Neo4jResourceConfig().database
        ),
        "gemini_api": GeminiAPIResource( 
            model_name=GeminiAPIResourceConfig().model_name,
            proxy_url=GeminiAPIResourceConfig().proxy_url, 
            default_temperature=GeminiAPIResourceConfig().default_temperature,
            default_max_tokens=GeminiAPIResourceConfig().default_max_tokens
        )
    }
)