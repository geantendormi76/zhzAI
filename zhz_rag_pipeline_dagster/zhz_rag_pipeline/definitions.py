import dagster as dg
import os
from dagster import Definitions

from zhz_rag_pipeline_dagster.zhz_rag_pipeline.ingestion_assets import all_ingestion_assets
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.processing_assets import all_processing_assets

from zhz_rag_pipeline_dagster.zhz_rag_pipeline.resources import (
    GGUFEmbeddingResource,
    ChromaDBResource,
    LocalLLMAPIResource,
    DuckDBResource,
    GeminiAPIResource,
    SystemResource
)
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.custom_io_managers import PydanticListJsonIOManager

# --- 根据官方文档，将所有资产组合在一起 ---
all_assets = all_ingestion_assets + all_processing_assets

# --- 根据官方文档，定义一个包含所有资源的字典 ---
# Dagster 会自动为每个资产提供它所需要的资源
all_resources = {
    # IO 管理器，键名必须是 "io_manager" 才能被默认使用
    "io_manager": PydanticListJsonIOManager(),
    
    # 其他应用级资源
    "embedder": GGUFEmbeddingResource(
        api_url=os.getenv("EMBEDDING_API_URL", "http://127.0.0.1:8089")
    ),
    "chroma_db": ChromaDBResource(
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "zhz_rag_collection"),
        persist_directory=os.path.join(os.getenv("ZHZ_AGENT_PROJECT_ROOT", "/home/zhz/zhz_agent"), "zhz_rag", "stored_data", "chromadb_index")
    ),
    "LocalLLM_api": LocalLLMAPIResource(
        api_url="http://127.0.0.1:8088/v1/chat/completions",
        default_temperature=0.1,
        default_max_new_tokens=2048
    ),
    "duckdb_kg": DuckDBResource(
        db_file_path=os.path.join(os.getenv("ZHZ_AGENT_PROJECT_ROOT", "/home/zhz/zhz_agent"), "zhz_rag", "stored_data", "duckdb_knowledge_graph.db")
    ),
    "gemini_api": GeminiAPIResource(
        model_name="gemini/gemini-1.5-flash-latest",
        proxy_url=os.getenv("LITELLM_PROXY_URL"),
        default_temperature=0.1,
        default_max_tokens=2048
    ),
    "system_resource": SystemResource()
}

# --- 创建最终的、简洁的 Definitions 对象 ---
defs = Definitions(
    assets=all_assets,
    resources=all_resources
)