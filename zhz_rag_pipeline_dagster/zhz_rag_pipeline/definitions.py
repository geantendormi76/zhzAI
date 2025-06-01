# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/zhz_rag_pipeline/definitions.py
import dagster as dg
import os

# 导入 in_process_executor
from dagster import define_asset_job, in_process_executor

from zhz_rag_pipeline_dagster.zhz_rag_pipeline.ingestion_assets import all_ingestion_assets
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.processing_assets import (
    all_processing_assets,
    # 单独导入 Kuzu 相关的资产，以便放入特定作业
    kuzu_schema_initialized_asset,
    kuzu_entity_nodes_asset,
    kuzu_entity_relations_asset,
    # 也需要 kg_extractions 作为上游
    kg_extraction_asset,
    # 以及 kg_extractions 的上游
    clean_chunk_text_asset# 假设 kg_extraction_asset 的上游是 text_chunks
)
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.evaluation_assets import all_evaluation_assets
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.resources import (
    SentenceTransformerResource, SentenceTransformerResourceConfig,
    ChromaDBResource, ChromaDBResourceConfig,
    SGLangAPIResource, SGLangAPIResourceConfig,
    KuzuDBReadWriteResource, 
    KuzuDBReadOnlyResource,
    GeminiAPIResource, GeminiAPIResourceConfig
)
from zhz_rag_pipeline_dagster.zhz_rag_pipeline.custom_io_managers import PydanticListJsonIOManager

# 将所有非 Kuzu 写操作的资产组合起来
# 注意：我们需要从 all_processing_assets 中移除 Kuzu 相关的写资产
# 或者更简单的方式是，明确列出哪些资产属于哪个作业或全局资产组
non_kuzu_write_processing_assets = [
    asset for asset in all_processing_assets 
    if asset.key not in [
        kuzu_schema_initialized_asset.key, 
        kuzu_entity_nodes_asset.key, 
        kuzu_entity_relations_asset.key
    ]
]
# kg_extraction_asset 虽然是 kg_building 组，但它不直接写 KuzuDB，而是产生供下游写入的数据
# text_chunks_asset 是 kg_extraction_asset 的上游，也需要包含

# 所有资产（用于默认加载，或者如果某些资产不属于特定作业）
# all_project_assets = all_ingestion_assets + non_kuzu_write_processing_assets + all_evaluation_assets
# 实际上，定义了作业后，我们主要通过作业来执行。
# 我们可以将所有资产都列在 Definitions 的 assets 中，然后作业通过 selection 选择。

all_defined_assets = all_ingestion_assets + all_processing_assets + all_evaluation_assets


# 定义 KuzuDB 写入作业
kuzu_kg_write_job = define_asset_job(
    name="kuzu_kg_write_job",
    selection=[ # 选择所有参与 KuzuDB 知识图谱构建的资产
        clean_chunk_text_asset, # kg_extraction_asset 的上游
        kg_extraction_asset,
        kuzu_schema_initialized_asset,
        kuzu_entity_nodes_asset,
        kuzu_entity_relations_asset
    ],
    executor_def=in_process_executor # <--- 使用单进程执行器
)

# IO Manager
pydantic_io_manager_instance = PydanticListJsonIOManager()

defs = dg.Definitions(
    assets=all_defined_assets, # 列出所有资产
    jobs=[kuzu_kg_write_job],  # <--- 添加作业
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
        "kuzu_readwrite_db": KuzuDBReadWriteResource(
            # 使用资源类中定义的默认值
            # clear_on_startup_for_testing=True # 测试时可以设为True，确保从干净状态开始
        ),
        "kuzu_readonly_db": KuzuDBReadOnlyResource(
            # 使用资源类中定义的默认值
        ),
        "gemini_api": GeminiAPIResource(
            model_name=GeminiAPIResourceConfig().model_name,
            proxy_url=GeminiAPIResourceConfig().proxy_url,
            default_temperature=GeminiAPIResourceConfig().default_temperature,
            default_max_tokens=GeminiAPIResourceConfig().default_max_tokens
        ),
        "pydantic_json_io_manager": pydantic_io_manager_instance,
        # 如果 KuzuDB 资源被其他非此作业的资产使用，确保它们使用 KuzuDBReadOnlyResource
    }
)