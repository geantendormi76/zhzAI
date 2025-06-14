# 文件: zhz_rag/core_rag/retrievers/embedding_functions.py

import logging
from typing import List, TYPE_CHECKING, Optional, Sequence
import numpy as np
from chromadb import Documents, Embeddings
import asyncio # 确保 asyncio 已导入

if TYPE_CHECKING:
    from zhz_rag.llm.local_model_handler import LocalModelHandler

logger = logging.getLogger(__name__)

# l2_normalize_embeddings 函数保持不变 (或者可以考虑也移到 LocalModelHandler 或 worker 中，如果只在那里用)
# ... (l2_normalize_embeddings 代码不变) ...
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
        try:
            self._dimension = self.model_handler.get_embedding_dimension() # get_embedding_dimension 是同步的
            if self._dimension:
                 logger.info(f"LlamaCppEmbeddingFunction initialized. Dimension from handler: {self._dimension}")
            else:
                 logger.info("LlamaCppEmbeddingFunction initialized. Dimension not immediately available, will be fetched on first embedding or via async get_dimension.")
        except Exception as e_dim_init:
            logger.warning(f"LlamaCppEmbeddingFunction: Error trying to get dimension during init: {e_dim_init}. Will fetch on first use.")

    # --- 删除 _run_async_in_new_loop 方法 ---

    async def __call__(self, input: Documents) -> Embeddings: # <--- 修改为 async def
        if not isinstance(input, list):
            logger.error(f"LlamaCppEmbeddingFunction received input of type {type(input)}, expected List[str].")
            # ... (处理非list输入的逻辑可以保持，但现在是异步上下文) ...
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
            # ... (processed_texts_for_handler 的构建逻辑保持不变) ...
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
            # --- 修改：直接 await LocalModelHandler 的异步方法 ---
            raw_embeddings_list = await self.model_handler.embed_documents(processed_texts_for_handler)
            
            if raw_embeddings_list:
                # L2归一化已在 LocalModelHandler 的子进程工作函数中完成，此处通常不再需要
                # embeddings_list = l2_normalize_embeddings(raw_embeddings_list) 
                embeddings_list = raw_embeddings_list # 假设 worker 返回已归一化的
            else:
                embeddings_list = []
            
            # --- 修改：维度获取和检查逻辑 ---
            if self._dimension is None and embeddings_list and embeddings_list[0]:
                self._dimension = len(embeddings_list[0]) # 从实际结果推断
                logger.info(f"LlamaCppEmbeddingFunction: Dimension updated from embedding result: {self._dimension}")
            elif embeddings_list and embeddings_list[0] and self._dimension and len(embeddings_list[0]) != self._dimension: 
                logger.warning(f"LlamaCppEmbeddingFunction: Inconsistent embedding dimension detected! "
                               f"Expected {self._dimension}, got {len(embeddings_list[0])}.")
            
            return embeddings_list
        except Exception as e:
            logger.error(f"LlamaCppEmbeddingFunction: Error during async embedding generation: {e}", exc_info=True)
            # 确保返回的列表长度与输入匹配，即使出错
            return [([0.0] * (self._dimension or 1024)) if self._dimension else [] for _ in input]


    async def embed_documents(self, texts: Sequence[str]) -> List[List[float]]: # <--- 修改为 async def
        # __call__ 已经是异步的了，可以直接调用
        return await self.__call__(input=list(texts))

    async def embed_query(self, text: str) -> List[float]: # <--- 修改为 async def
        if not text:
            # 尝试获取维度，如果未知
            if self._dimension is None:
                dim_from_handler = self.model_handler.get_embedding_dimension() # 同步获取
                if dim_from_handler is None: # 如果还是未知，尝试异步获取一次
                    dim_from_handler = await self.model_handler._get_embedding_dimension_from_worker_once()
                self._dimension = dim_from_handler
            return [0.0] * (self._dimension or 1024) if self._dimension else []
        
        # --- 修改：直接 await LocalModelHandler 的异步方法 ---
        embedding_result = await self.model_handler.embed_query(text)
        
        # --- 修改：维度获取和检查逻辑 ---
        if self._dimension is None and embedding_result:
            self._dimension = len(embedding_result)
            logger.info(f"LlamaCppEmbeddingFunction: Dimension updated from query embedding result: {self._dimension}")
        elif embedding_result and self._dimension and len(embedding_result) != self._dimension:
            logger.warning(f"LlamaCppEmbeddingFunction: Query embedding dimension mismatch! Expected {self._dimension}, got {len(embedding_result)}.")
            # 根据策略返回空或错误向量
            return [0.0] * (self._dimension or 1024) if self._dimension else []

        return embedding_result if embedding_result else ([0.0] * (self._dimension or 1024) if self._dimension else [])
    
    async def get_dimension(self) -> Optional[int]: # <--- 修改为 async def
        if self._dimension is None:
            # 尝试从 model_handler 同步获取
            dim_from_handler_sync = self.model_handler.get_embedding_dimension()
            if dim_from_handler_sync is not None:
                self._dimension = dim_from_handler_sync
            else:
                # 如果同步获取失败，则异步从 worker 获取
                logger.info("LlamaCppEmbeddingFunction: Dimension not cached, attempting async fetch from worker.")
                dim_from_worker = await self.model_handler._get_embedding_dimension_from_worker_once()
                if dim_from_worker is not None:
                    self._dimension = dim_from_worker
                else:
                    logger.error("LlamaCppEmbeddingFunction: Failed to get dimension asynchronously.")
                    return None # 或者抛出异常

            if self._dimension:
                logger.info(f"LlamaCppEmbeddingFunction: Dimension fetched/updated: {self._dimension}")
        return self._dimension