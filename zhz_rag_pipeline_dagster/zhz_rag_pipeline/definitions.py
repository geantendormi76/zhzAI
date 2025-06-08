# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/zhz_rag_pipeline/definitions.py
import dagster as dg
import os

from dagster import (
    define_asset_job,
    Definitions,
    in_process_executor,
    AssetSelection  # 保持导入以备后用
)

# 导入所有资产列表
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.ingestion_assets import all_ingestion_assets
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.processing_assets import all_processing_assets

# 导入所有资源和 IO 管理器
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.resources import (
    SentenceTransformerResource, SentenceTransformerResourceConfig,
    ChromaDBResource, ChromaDBResourceConfig,
    LocalLLMAPIResource, LocalLLMAPIResourceConfig,
    KuzuDBReadWriteResource,
    KuzuDBReadOnlyResource,
    GeminiAPIResource, GeminiAPIResourceConfig
)
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.custom_io_managers import PydanticListJsonIOManager

# 1. 定义一个包含项目中所有资产的列表
all_defined_assets = all_ingestion_assets + all_processing_assets

# 2. [最终的、最简单的作业定义]
#    直接选择所有资产。因为我们使用了 in_process_executor，
#    并且 KuzuDB 资源管理现在是正确的，所以即使选择了所有资产，
#    它们也会被串行执行，不会有并发问题。
#    当您只想运行 KuzuDB 相关的部分时，可以在 UI 中选择一个子集。
kuzu_kg_write_job = define_asset_job(
    name="kuzu_kg_write_job",
    selection=AssetSelection.all(), # 选择 all_defined_assets 中的所有资产
    executor_def=in_process_executor
)

# 3. 实例化 IO 管理器
pydantic_io_manager_instance = PydanticListJsonIOManager()

# 4. 将所有定义组合成最终的 Definitions 对象
defs = Definitions(
    assets=all_defined_assets,
    jobs=[kuzu_kg_write_job],
    resources={
        "embedder": SentenceTransformerResource(model_name_or_path=SentenceTransformerResourceConfig().model_name_or_path),
        "chroma_db": ChromaDBResource(collection_name=ChromaDBResourceConfig().collection_name, persist_directory=ChromaDBResourceConfig().persist_directory),
        "sglang_api": LocalLLMAPIResource(api_url=LocalLLMAPIResourceConfig().api_url, default_temperature=LocalLLMAPIResourceConfig().default_temperature, default_max_new_tokens=LocalLLMAPIResourceConfig().default_max_new_tokens),
        "kuzu_readwrite_db": KuzuDBReadWriteResource(),
        # kuzu_readonly_db 资源暂时没有被任何启用的资产使用，但保留它也无妨
        "kuzu_readonly_db": KuzuDBReadOnlyResource(),
        # gemini_api 资源也只被评估资产使用，理论上可以注释掉，但保留也无妨
        "gemini_api": GeminiAPIResource(model_name=GeminiAPIResourceConfig().model_name, proxy_url=GeminiAPIResourceConfig().proxy_url, default_temperature=GeminiAPIResourceConfig().default_temperature, default_max_tokens=GeminiAPIResourceConfig().default_max_tokens),
        "pydantic_json_io_manager": pydantic_io_manager_instance,
    }
)