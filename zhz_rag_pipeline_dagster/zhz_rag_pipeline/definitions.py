import dagster as dg
import os
from dagster import Definitions

from zhz_rag_pipeline_dagster.zhz_rag_pipeline.ingestion_assets import all_ingestion_assets
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.processing_assets import all_processing_assets

from zhz_rag_pipeline_dagster.zhz_rag_pipeline.resources import (
    GGUFEmbeddingResource, GGUFEmbeddingResourceConfig,
    ChromaDBResource, ChromaDBResourceConfig,
    LocalLLMAPIResource, LocalLLMAPIResourceConfig,
    DuckDBResource,
    GeminiAPIResource, GeminiAPIResourceConfig,
    SystemResource
)
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.custom_io_managers import PydanticListJsonIOManager

all_defined_assets = all_ingestion_assets + all_processing_assets
pydantic_io_manager_instance = PydanticListJsonIOManager()

defs = Definitions(
    assets=all_defined_assets,
    resources={
        # --- START: 覆盖这个 embedder 定义 ---
        "embedder": GGUFEmbeddingResource(
            api_url=os.getenv("EMBEDDING_API_URL", "http://127.0.0.1:8089")
        ),
        # --- END: 覆盖结束 ---
        "chroma_db": ChromaDBResource(collection_name=ChromaDBResourceConfig().collection_name, persist_directory=ChromaDBResourceConfig().persist_directory),
        "LocalLLM_api": LocalLLMAPIResource(api_url=LocalLLMAPIResourceConfig().api_url, default_temperature=LocalLLMAPIResourceConfig().default_temperature, default_max_new_tokens=LocalLLMAPIResourceConfig().default_max_new_tokens),
        "duckdb_kg": DuckDBResource(),
        "gemini_api": GeminiAPIResource(model_name=GeminiAPIResourceConfig().model_name, proxy_url=GeminiAPIResourceConfig().proxy_url, default_temperature=GeminiAPIResourceConfig().default_temperature, default_max_tokens=GeminiAPIResourceConfig().default_max_tokens),
        "pydantic_json_io_manager": pydantic_io_manager_instance,
        "system_info": SystemResource()
    }
)