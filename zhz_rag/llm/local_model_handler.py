# zhz_rag/llm/local_model_handler.py
# 版本: 简化版，不再处理多进程，仅用于定义 LlamaCppEmbeddingFunction

import os
import sys
import logging
from typing import List, Optional, Any, Sequence
import numpy as np
from cachetools import TTLCache
import asyncio

# --- 日志配置 ---
handler_logger = logging.getLogger("LocalModelHandler") # 日志名可以保持
if not handler_logger.hasHandlers():
    handler_logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - PID:%(process)d - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    handler_logger.addHandler(stream_handler)
    handler_logger.propagate = False

# 为了避免与 core_rag 中的 LlamaCppEmbeddingFunction 混淆，
# 我们可以考虑重命名这个类，例如 MyModelHandlerForEmbeddings，
# 或者确保它的接口与 'LocalModelHandler' 类型提示兼容。
# 我们先保持类名，但确保它有 'embedding_model_path' 属性。
class LlamaCppEmbeddingFunction: # 这个类将作为 "model_handler"
    """
    一个与 Dagster 资源交互的嵌入函数包装器。
    它不直接创建模型，而是通过 GGUFEmbeddingResource 与工作进程通信。
    这个类将扮演 'LocalModelHandler' 的角色，被 core_rag 中的 LlamaCppEmbeddingFunction 使用。
    """
    def __init__(self, resource: Any, embedding_model_path_for_handler: Optional[str] = None): # resource 是 GGUFEmbeddingResource
        if resource is None:
            raise ValueError("GGUFEmbeddingResource is required.")
        self.resource = resource
        # CoreRetriever_LlamaCppEmbeddingFunction 期望 model_handler 有 embedding_model_path 属性
        self.embedding_model_path = embedding_model_path_for_handler if embedding_model_path_for_handler else os.getenv("EMBEDDING_MODEL_PATH")
        if not self.embedding_model_path:
             handler_logger.warning("embedding_model_path not provided to LocalModelHandler's LlamaCppEmbeddingFunction and not found in env. This might be an issue if downstream components expect it.")

        self._dimension: Optional[int] = None
        self._query_cache: TTLCache = TTLCache(maxsize=200, ttl=3600)
        self._cache_lock = asyncio.Lock()
        
        # 尝试在初始化时获取维度
        try:
            # 注意：resource.get_embedding_dimension() 是同步的，在异步 __init__ 中应小心
            # 但由于 __init__ 本身不是 async，这里直接调用是OK的
            # 如果 resource.get_embedding_dimension() 内部做了异步转同步，那也没问题
            dim = self.resource.get_embedding_dimension()
            if dim is not None:
                self._dimension = dim
                handler_logger.info(f"LocalModelHandler's LlamaCppEmbeddingFunction initialized. Dimension from resource: {self._dimension}")
            else:
                handler_logger.info("LocalModelHandler's LlamaCppEmbeddingFunction initialized. Dimension not immediately available from resource.")
        except Exception as e:
             handler_logger.warning(f"Error getting dimension from resource during init: {e}. Will fetch on first use.")


    async def __call__(self, input: Sequence[str]) -> List[List[float]]:
        if not input:
            return []
        handler_logger.info(f"LocalModelHandler's LlamaCppEmbeddingFunction: Generating embeddings for {len(input)} documents (via resource).")
        # 直接调用 Dagster 资源的 encode 方法
        # GGUFEmbeddingResource.encode 是同步包装的异步，所以 to_thread 是合适的
        embeddings = await asyncio.to_thread(self.resource.encode, list(input))
        return embeddings

    async def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return await self.__call__(list(texts))

    async def embed_query(self, text: str) -> List[float]:
        async with self._cache_lock:
            cached_result = self._query_cache.get(text)
        if cached_result is not None:
            handler_logger.info(f"Query Vector CACHE HIT for query (LocalModelHandler): '{text[:50]}...'")
            return cached_result
        
        handler_logger.info(f"Query Vector CACHE MISS for query (LocalModelHandler): '{text[:50]}...'. Generating new embedding.")
        embedding_list = await asyncio.to_thread(self.resource.encode, [text])
        
        if embedding_list and embedding_list[0]:
            embedding_vector = embedding_list[0]
            async with self._cache_lock:
                self._query_cache[text] = embedding_vector
            return embedding_vector
        
        dim = await self.get_dimension()
        return [0.0] * (dim or 1024) # Fallback to 1024 if dim is None

    def get_embedding_dimension(self) -> Optional[int]: # 改为同步，因为它在 __init__ 和 CoreRetriever 中被同步调用
        if self._dimension is None:
            try:
                dim_from_res = self.resource.get_embedding_dimension()
                if dim_from_res is not None:
                    self._dimension = dim_from_res
                    handler_logger.info(f"Dimension fetched from resource (sync): {self._dimension}")
            except Exception as e:
                handler_logger.error(f"Error fetching dimension from resource (sync): {e}")
        return self._dimension

    async def _get_embedding_dimension_from_worker_once(self) -> Optional[int]: # 辅助异步获取方法
         if self._dimension is None:
             try:
                 # GGUFEmbeddingResource.get_embedding_dimension() 是同步的
                 # 如果要在异步方法中调用，需要 to_thread
                 dim = await asyncio.to_thread(self.resource.get_embedding_dimension)
                 if dim is not None:
                     self._dimension = dim
                     handler_logger.info(f"Dimension fetched from resource (async helper): {self._dimension}")
             except Exception as e:
                 handler_logger.error(f"Error fetching dimension from resource (async helper): {e}")
         return self._dimension