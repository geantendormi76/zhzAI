# 文件: zhz_rag/core_rag/retrievers/embedding_functions.py

import logging
from typing import List, Dict, TYPE_CHECKING, Optional, Sequence
import numpy as np
from chromadb import Documents, Embeddings
import asyncio # 确保 asyncio 已导入

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
    一个自定义的 ChromaDB 嵌入函数，使用 LocalModelHandler (llama.cpp) 生成嵌入。
    此类的方法现在设计为异步的，以正确桥接 LocalModelHandler 的异步嵌入方法。
    ChromaDB 0.4.x 及以上版本支持异步嵌入函数。
    【新增】此类包含一个简单的内存缓存，用于避免对相同的查询文本重复进行嵌入计算。
    """
    def __init__(self, model_handler: 'LocalModelHandler'):
        if model_handler is None:
            logger.error("LlamaCppEmbeddingFunction initialized with no model_handler.")
            raise ValueError("LocalModelHandler is required.")
        if not model_handler.embedding_model_path:
            logger.error("LlamaCppEmbeddingFunction initialized with a model_handler that does not have an embedding_model_path configured.")
            raise ValueError("LocalModelHandler must have an embedding_model_path configured to be used for embeddings.")
        self.model_handler = model_handler
        self._dimension: Optional[int] = None
        
        # --- 新增：初始化查询缓存和异步锁 ---
        self._query_cache: Dict[str, List[float]] = {}
        self._cache_lock = asyncio.Lock()
        
        try:
            self._dimension = self.model_handler.get_embedding_dimension()
            if self._dimension:
                 logger.info(f"LlamaCppEmbeddingFunction initialized. Dimension from handler: {self._dimension}")
            else:
                 logger.info("LlamaCppEmbeddingFunction initialized. Dimension not immediately available, will be fetched on first embedding or via async get_dimension.")
        except Exception as e_dim_init:
            logger.warning(f"LlamaCppEmbeddingFunction: Error trying to get dimension during init: {e_dim_init}. Will fetch on first use.")

    async def __call__(self, input: Documents) -> Embeddings:
        if not isinstance(input, list):
            logger.error(f"LlamaCppEmbeddingFunction received input of type {type(input)}, expected List[str].")
            if isinstance(input, str):
                processed_texts_for_handler = [input + "<|endoftext|>" if input and not input.endswith("<|endoftext|>") else input]
            else:
                try: 
                    num_items = len(input)
                    return [[] for _ in range(num_items)]
                except TypeError:
                    return [[]] 
        elif not input: 
            return [] 
        else:
            processed_texts_for_handler: List[str] = []
            for text_item in input:
                if isinstance(text_item, str):
                    if text_item and not text_item.endswith("<|endoftext|>"):
                        processed_texts_for_handler.append(text_item + "<|endoftext|>")
                    else:
                        processed_texts_for_handler.append(text_item) 
                else:
                    logger.warning(f"LlamaCppEmbeddingFunction received non-string item in input list: {type(text_item)}. Converting to string and adding <|endoftext|>.")
                    str_item = str(text_item)
                    processed_texts_for_handler.append(str_item + "<|endoftext|>" if str_item and not str_item.endswith("<|endoftext|>") else str_item)
        
        logger.info(f"LlamaCppEmbeddingFunction: Generating embeddings for {len(processed_texts_for_handler)} processed texts (async).")
        if processed_texts_for_handler:
            logger.debug(f"LlamaCppEmbeddingFunction: First processed text for embedding: '{processed_texts_for_handler[0][:150]}...'")

        try:
            raw_embeddings_list = await self.model_handler.embed_documents(processed_texts_for_handler)
            embeddings_list = raw_embeddings_list if raw_embeddings_list else []
            
            if self._dimension is None and embeddings_list and embeddings_list[0]:
                self._dimension = len(embeddings_list[0])
                logger.info(f"LlamaCppEmbeddingFunction: Dimension updated from embedding result: {self._dimension}")
            elif embeddings_list and embeddings_list[0] and self._dimension and len(embeddings_list[0]) != self._dimension: 
                logger.warning(f"LlamaCppEmbeddingFunction: Inconsistent embedding dimension detected! Expected {self._dimension}, got {len(embeddings_list[0])}.")
            
            return embeddings_list
        except Exception as e:
            logger.error(f"LlamaCppEmbeddingFunction: Error during async embedding generation: {e}", exc_info=True)
            return [([0.0] * (self._dimension or 1024)) if self._dimension else [] for _ in input]

    async def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return await self.__call__(input=list(texts))

    async def embed_query(self, text: str) -> List[float]:
        if not text:
            async with self._cache_lock:
                if self._dimension is None:
                    dim_from_handler = self.model_handler.get_embedding_dimension()
                    if dim_from_handler is None:
                        dim_from_handler = await self.model_handler._get_embedding_dimension_from_worker_once()
                    self._dimension = dim_from_handler
            return [0.0] * (self._dimension or 1024) if self._dimension else []
        
        # --- 缓存检查 (Cache Check) ---
        async with self._cache_lock:
            if text in self._query_cache:
                logger.info(f"Cache HIT for query: '{text[:50]}...'")
                return self._query_cache[text]
        
        logger.info(f"Cache MISS for query: '{text[:50]}...'. Generating new embedding.")
        
        # 调用模型生成向量
        embedding_vector = await self.model_handler.embed_query(text)

        # 缓存新生成的向量
        if embedding_vector:
            async with self._cache_lock:
                if self._dimension is None:
                    self._dimension = len(embedding_vector)
                    logger.info(f"LlamaCppEmbeddingFunction: Dimension updated from query embedding result: {self._dimension}")
                elif len(embedding_vector) != self._dimension:
                    logger.warning(f"LlamaCppEmbeddingFunction: Query embedding dimension mismatch! Expected {self._dimension}, got {len(embedding_vector)}. Not caching this result.")
                    return embedding_vector # 直接返回，但不缓存

                # 存储到缓存
                self._query_cache[text] = embedding_vector
                logger.info(f"Cached new embedding for query: '{text[:50]}...'")
            return embedding_vector
        else: # embed_query 返回了 None 或空列表
            logger.warning(f"Failed to generate embedding for query: '{text[:50]}...'. Returning empty list or zero vector.")
            async with self._cache_lock:
                if self._dimension is None:
                    dim_from_handler = await self.model_handler._get_embedding_dimension_from_worker_once()
                    self._dimension = dim_from_handler
            return [0.0] * (self._dimension or 1024) if self._dimension else []
    
    async def get_dimension(self) -> Optional[int]:
        if self._dimension is None:
            async with self._cache_lock:
                # 再次检查，防止在等待锁的过程中其他协程已经设置了维度
                if self._dimension is None:
                    dim_from_handler_sync = self.model_handler.get_embedding_dimension()
                    if dim_from_handler_sync is not None:
                        self._dimension = dim_from_handler_sync
                    else:
                        logger.info("LlamaCppEmbeddingFunction: Dimension not cached, attempting async fetch from worker.")
                        dim_from_worker = await self.model_handler._get_embedding_dimension_from_worker_once()
                        if dim_from_worker is not None:
                            self._dimension = dim_from_worker
                        else:
                            logger.error("LlamaCppEmbeddingFunction: Failed to get dimension asynchronously.")
                            return None
                    if self._dimension:
                        logger.info(f"LlamaCppEmbeddingFunction: Dimension fetched/updated: {self._dimension}")
        return self._dimension
