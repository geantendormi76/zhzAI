# 文件: zhz_rag/core_rag/retrievers/embedding_functions.py

import logging
from typing import List, Dict, TYPE_CHECKING, Optional, Sequence
import numpy as np
from chromadb import Documents, Embeddings
import asyncio # 确保 asyncio 已导入
from cachetools import TTLCache

if TYPE_CHECKING:
    from zhz_rag.llm.local_model_handler import LocalModelHandler

logger = logging.getLogger(__name__)

def l2_normalize_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
    """对一批嵌入向量进行L2归一化。"""
    if not embeddings or not isinstance(embeddings, list):
        return []
    
    normalized_embeddings = []
    for emb_list in embeddings:
        if not emb_list or not isinstance(emb_list, list): 
            normalized_embeddings.append([])
            continue
        try:
            emb_array = np.array(emb_list, dtype=np.float32)
            norm = np.linalg.norm(emb_array)
            if norm == 0: 
                normalized_embeddings.append(emb_list) 
            else:
                normalized_embeddings.append((emb_array / norm).tolist())
        except Exception as e_norm:
            logger.error(f"Error during L2 normalization of an embedding: {e_norm}", exc_info=True)
            normalized_embeddings.append(emb_list) 
    return normalized_embeddings



class LlamaCppEmbeddingFunction:
    """
    一个与 LangChain 兼容的 ChromaDB 嵌入函数。
    V3: 恢复为原生异步，以解决同步/异步桥接导致的死锁问题。
    """
    def __init__(self, model_handler: 'LocalModelHandler'):
        if model_handler is None:
            raise ValueError("LocalModelHandler is required.")
        self.model_handler = model_handler
        self._dimension: Optional[int] = None
        self._query_cache: TTLCache = TTLCache(maxsize=200, ttl=3600)
        self._cache_lock = asyncio.Lock()
        
        try:
            self._dimension = self.model_handler.get_embedding_dimension()
            if self._dimension:
                 logger.info(f"LlamaCppEmbeddingFunction initialized. Dimension from handler: {self._dimension}")
        except Exception as e_dim_init:
            logger.warning(f"LlamaCppEmbeddingFunction: Error trying to get dimension during init: {e_dim_init}.")

    # --- 核心修改：恢复为 async def ---
    async def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        logger.info(f"LlamaCppEmbeddingFunction (ASYNC __call__): Generating embeddings for {len(input)} documents.")
        # 直接 await 异步方法
        return await self.model_handler.embed_documents(list(input))

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self.__call__(texts)

    async def embed_query(self, text: str) -> List[float]:
        async with self._cache_lock:
            cached_result = self._query_cache.get(text)
        
        if cached_result is not None:
            logger.info(f"Query Vector CACHE HIT for query: '{text[:50]}...'")
            return cached_result
        
        logger.info(f"Query Vector CACHE MISS for query: '{text[:50]}...'. Generating new embedding.")
        embedding_vector = await self.model_handler.embed_query(text)
        
        if embedding_vector:
            async with self._cache_lock:
                self._query_cache[text] = embedding_vector
        else:
            embedding_vector = [0.0] * (self._dimension or 1024)

        return embedding_vector