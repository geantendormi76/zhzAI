# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/zhz_rag_pipeline/definitions.py
import dagster as dg
import os

from dagster import (
    define_asset_job,
    Definitions,
    in_process_executor,
    AssetSelection
)

from zhz_rag_pipeline_dagster.zhz_rag_pipeline.ingestion_assets import all_ingestion_assets
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.processing_assets import all_processing_assets

from zhz_rag_pipeline_dagster.zhz_rag_pipeline.resources import (
    GGUFEmbeddingResource, GGUFEmbeddingResourceConfig,
    ChromaDBResource, ChromaDBResourceConfig,
    LocalLLMAPIResource, LocalLLMAPIResourceConfig,
    DuckDBResource,
    GeminiAPIResource, GeminiAPIResourceConfig
)
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.custom_io_managers import PydanticListJsonIOManager

all_defined_assets = all_ingestion_assets + all_processing_assets

# --- 修改作业名称 ---
# 将 kuzu_kg_write_job 重命名为 duckdb_kg_build_job
duckdb_kg_build_job = define_asset_job( # <--- 修改变量名
    name="duckdb_kg_build_job",         # <--- 修改作业的实际名称
    selection=AssetSelection.all(),
    executor_def=in_process_executor
)
# --- 修改结束 ---

pydantic_io_manager_instance = PydanticListJsonIOManager()

defs = Definitions(
    assets=all_defined_assets,
    jobs=[duckdb_kg_build_job], # <--- 在 jobs 列表中使用新的作业变量名
    resources={
        "embedder": GGUFEmbeddingResource(
            embedding_model_path=os.getenv("EMBEDDING_MODEL_PATH"),
            n_ctx=int(os.getenv("EMBEDDING_N_CTX", GGUFEmbeddingResourceConfig.model_fields['n_ctx'].default)),
            n_gpu_layers=int(os.getenv("EMBEDDING_N_GPU_LAYERS", GGUFEmbeddingResourceConfig.model_fields['n_gpu_layers'].default))
        ),
        "chroma_db": ChromaDBResource(collection_name=ChromaDBResourceConfig().collection_name, persist_directory=ChromaDBResourceConfig().persist_directory),
        "LocalLLM_api": LocalLLMAPIResource(api_url=LocalLLMAPIResourceConfig().api_url, default_temperature=LocalLLMAPIResourceConfig().default_temperature, default_max_new_tokens=LocalLLMAPIResourceConfig().default_max_new_tokens),
        "duckdb_kg": DuckDBResource(),
        "gemini_api": GeminiAPIResource(model_name=GeminiAPIResourceConfig().model_name, proxy_url=GeminiAPIResourceConfig().proxy_url, default_temperature=GeminiAPIResourceConfig().default_temperature, default_max_tokens=GeminiAPIResourceConfig().default_max_tokens),
        "pydantic_json_io_manager": pydantic_io_manager_instance,
    }
)