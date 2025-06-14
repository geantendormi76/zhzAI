# 文件: zhz_rag/llm/local_model_handler.py
import os
import logging
from typing import List, Optional, Dict, Any
import asyncio
import numpy as np
import multiprocessing
from functools import partial
from multiprocessing.pool import Pool

# 导入新的工作函数
from zhz_rag.llm.embedding_process_worker import embed_texts_in_subprocess, embed_query_in_subprocess

logger = logging.getLogger(__name__)

def l2_normalize_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
    """
    对一批嵌入向量进行L2归一化。
    注意：这个函数现在主要作为备用或验证，因为归一化逻辑已移至工作进程中。
    """
    if not embeddings or not isinstance(embeddings, list):
        return []
    normalized_embeddings = []
    for emb_list in embeddings:
        if not emb_list or not isinstance(emb_list, list) or not all(isinstance(x, (float, int)) for x in emb_list):
            logger.warning(f"L2_NORM (LMH): Skipping invalid or empty inner list: {emb_list}")
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
            logger.error(f"Error during L2 normalization of an embedding in LocalModelHandler: {e_norm}", exc_info=True)
            normalized_embeddings.append(emb_list) 
    return normalized_embeddings

class LocalModelHandler:
    """
    管理本地GGUF模型加载和调用的句柄。
    LLM模型在主进程中加载。
    嵌入模型的操作被委托给一个独立的子进程池，以避免多线程冲突和段错误。
    """
    _instance_count = 0 

    def __init__(
        self,
        llm_model_path: Optional[str] = None,
        embedding_model_path: Optional[str] = None,
        n_ctx_llm: int = 4096,
        n_gpu_layers_llm: int = 0,
        n_ctx_embed: int = 2048,
        n_gpu_layers_embed: int = 0,
        pooling_type_embed: int = 2,
        embedding_pool_size: Optional[int] = None
    ):
        LocalModelHandler._instance_count += 1
        self.instance_id = LocalModelHandler._instance_count
        logger.info(f"LMH Instance [{self.instance_id}] __init__ called.")

        # LLM 模型加载逻辑
        self.llm_model_path = llm_model_path
        self.n_ctx_llm = n_ctx_llm
        self.n_gpu_layers_llm = n_gpu_layers_llm
        self.llm_model: Optional[Any] = None
        
        if llm_model_path:
            try:
                from llama_cpp import Llama
                logger.info(f"LMH Instance [{self.instance_id}]: Loading LLM model from: {llm_model_path}")
                self.llm_model = Llama(
                    model_path=llm_model_path,
                    n_ctx=n_ctx_llm,
                    n_gpu_layers=n_gpu_layers_llm,
                    verbose=False
                )
                logger.info(f"LMH Instance [{self.instance_id}]: LLM model loaded successfully.")
            except Exception as e:
                logger.error(f"LMH Instance [{self.instance_id}]: Failed to load LLM model from {llm_model_path}: {e}", exc_info=True)
                self.llm_model = None

        # 保存嵌入模型配置，不再直接加载
        self.embedding_model_path = embedding_model_path
        self.n_ctx_embed = n_ctx_embed
        self.n_gpu_layers_embed = n_gpu_layers_embed
        self.pooling_type_embed = pooling_type_embed
        
        self._embedding_model_dimension: Optional[int] = None
        self._dimension_lock = asyncio.Lock()

        # 初始化进程池
        self._embedding_pool: Optional[Pool] = None
        self._pool_size = embedding_pool_size if embedding_pool_size else os.cpu_count() or 1
        
        if self.embedding_model_path:
            try:
                ctx = multiprocessing.get_context('spawn')
                self._embedding_pool = ctx.Pool(processes=self._pool_size)
                logger.info(f"LMH Instance [{self.instance_id}]: Embedding subprocess pool initialized with size {self._pool_size} and context 'spawn'.")
            except Exception as e_pool:
                logger.error(f"LMH Instance [{self.instance_id}]: Failed to initialize embedding subprocess pool: {e_pool}", exc_info=True)
                self._embedding_pool = None
        else:
            logger.warning(f"LMH Instance [{self.instance_id}]: No embedding_model_path provided. Embedding pool not created.")

        if not self.llm_model and not self.embedding_model_path:
            logger.warning(f"LMH Instance [{self.instance_id}]: Initialized without LLM and no path for embedding model.")

    async def _get_embedding_dimension_from_worker_once(self) -> Optional[int]:
        """
        通过启动一个临时工作进程来获取嵌入维度，并缓存结果。
        """
        if not self.embedding_model_path or not self._embedding_pool:
            logger.error(f"LMH Instance [{self.instance_id}]: Cannot get dimension, no model path or pool.")
            return None

        dummy_text = "dimension" 
        logger.info(f"LMH Instance [{self.instance_id}]: Attempting to derive embedding dimension via a dummy query in subprocess...")
        try:
            task_args = (
                dummy_text,
                self.embedding_model_path,
                self.n_ctx_embed,
                self.n_gpu_layers_embed,
                self.pooling_type_embed
            )
            loop = asyncio.get_running_loop()
            async_result = self._embedding_pool.apply_async(embed_query_in_subprocess, args=task_args)
            result_embedding = await loop.run_in_executor(None, async_result.get, 60)

            if result_embedding and isinstance(result_embedding, list) and len(result_embedding) > 0:
                dim = len(result_embedding)
                logger.info(f"LMH Instance [{self.instance_id}]: Successfully derived embedding dimension: {dim}")
                return dim
            else:
                logger.error(f"LMH Instance [{self.instance_id}]: Failed to derive dimension. Dummy query returned: {result_embedding}")
                return None
        except multiprocessing.TimeoutError:
            logger.error(f"LMH Instance [{self.instance_id}]: Timeout while trying to get dimension from worker.")
            return None
        except Exception as e_dim:
            logger.error(f"LMH Instance [{self.instance_id}]: Error deriving embedding dimension: {e_dim}", exc_info=True)
            return None

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._embedding_pool or not self.embedding_model_path:
            logger.error(f"LMH Instance [{self.instance_id}]: Embedding pool or model path not available. Cannot embed documents.")
            async with self._dimension_lock:
                if self._embedding_model_dimension is None:
                    logger.warning(f"LMH Instance [{self.instance_id}]: embed_documents - dimension unknown, attempting to get it first.")
                    temp_dim = await self._get_embedding_dimension_from_worker_once()
                    if temp_dim:
                        self._embedding_model_dimension = temp_dim
                    else:
                        logger.error(f"LMH Instance [{self.instance_id}]: embed_documents - Failed to get dimension, cannot proceed.")
                        return [[] for _ in texts]
            return [[0.0] * self._embedding_model_dimension if self._embedding_model_dimension else [] for _ in texts]

        if not texts:
            return []

        processed_texts = [(text + "<|endoftext|>" if text and not text.endswith("<|endoftext|>") else text) for text in texts]
        logger.info(f"LMH Instance [{self.instance_id}]: Submitting {len(processed_texts)} documents for embedding to subprocess pool...")

        try:
            task_func = partial(
                embed_texts_in_subprocess,
                embedding_model_path=self.embedding_model_path,
                n_ctx_embed=self.n_ctx_embed,
                n_gpu_layers_embed=self.n_gpu_layers_embed,
                pooling_type_embed=self.pooling_type_embed
            )
            
            loop = asyncio.get_running_loop()
            async_result = self._embedding_pool.apply_async(task_func, args=(processed_texts,))
            embeddings_list = await loop.run_in_executor(None, async_result.get, 300)

            if embeddings_list is None:
                raise RuntimeError("Embedding subprocess returned None, possibly due to timeout or unhandled error in worker.")

            logger.info(f"LMH Instance [{self.instance_id}]: Received {len(embeddings_list)} embeddings from subprocess.")
            return embeddings_list

        except multiprocessing.TimeoutError:
            logger.error(f"LMH Instance [{self.instance_id}]: Timeout embedding documents in subprocess.", exc_info=True)
            return [[0.0] * (self._embedding_model_dimension or 1024) for _ in texts]
        except Exception as e_async_embed_docs:
            logger.error(f"LMH Instance [{self.instance_id}]: Error submitting document embedding task to subprocess: {e_async_embed_docs}", exc_info=True)
            return [[0.0] * (self._embedding_model_dimension or 1024) for _ in texts]

    async def embed_query(self, text: str) -> List[float]:
        if not self._embedding_pool or not self.embedding_model_path:
            logger.error(f"LMH Instance [{self.instance_id}]: Embedding pool or model path not available. Cannot embed query.")
            async with self._dimension_lock:
                if self._embedding_model_dimension is None:
                    logger.warning(f"LMH Instance [{self.instance_id}]: embed_query - dimension unknown, attempting to get it first.")
                    temp_dim = await self._get_embedding_dimension_from_worker_once()
                    if temp_dim:
                        self._embedding_model_dimension = temp_dim
                    else:
                        logger.error(f"LMH Instance [{self.instance_id}]: embed_query - Failed to get dimension, cannot proceed.")
                        return []
            return [0.0] * self._embedding_model_dimension if self._embedding_model_dimension else []

        if not text: 
            return []

        processed_text = text + "<|endoftext|>" if not text.endswith("<|endoftext|>") else text
        logger.info(f"LMH Instance [{self.instance_id}]: Submitting query for embedding to subprocess pool (first 100): {processed_text[:100]}...")

        try:
            task_func = partial(
                embed_query_in_subprocess,
                embedding_model_path=self.embedding_model_path,
                n_ctx_embed=self.n_ctx_embed,
                n_gpu_layers_embed=self.n_gpu_layers_embed,
                pooling_type_embed=self.pooling_type_embed
            )
            loop = asyncio.get_running_loop()
            async_result = self._embedding_pool.apply_async(task_func, args=(processed_text,))
            embedding_vector = await loop.run_in_executor(None, async_result.get, 60)

            if embedding_vector is None:
                 raise RuntimeError("Embedding subprocess returned None for query, possibly due to timeout or unhandled error in worker.")

            logger.info(f"LMH Instance [{self.instance_id}]: Received query embedding from subprocess (len: {len(embedding_vector)}).")
            return embedding_vector
            
        except multiprocessing.TimeoutError:
            logger.error(f"LMH Instance [{self.instance_id}]: Timeout embedding query in subprocess.", exc_info=True)
            return [0.0] * (self._embedding_model_dimension or 1024)
        except Exception as e_async_embed_query:
            logger.error(f"LMH Instance [{self.instance_id}]: Error submitting query embedding task to subprocess: {e_async_embed_query}", exc_info=True)
            return [0.0] * (self._embedding_model_dimension or 1024)
    
    def get_embedding_dimension(self) -> Optional[int]:
        if self._embedding_model_dimension is None:
             logger.warning(f"LMH Instance [{self.instance_id}]: get_embedding_dimension() called but dimension is not yet known. "
                            "It should be fetched by calling an embedding method first or during initialization.")
        return self._embedding_model_dimension

    async def generate_text_with_local_llm(self, messages: List[Dict[str,str]], temperature: float = 0.1, max_tokens: int = 1024, stop: Optional[List[str]]=None) -> Optional[str]:
        if not self.llm_model:
            logger.error(f"LMH Instance [{self.instance_id}]: LLM model is not loaded. Cannot generate text.")
            return None
        
        logger.info(f"LMH Instance [{self.instance_id}]: Generating text with local LLM. Message count: {len(messages)}")
        
        def _blocking_llm_call():
            try:
                if not hasattr(self.llm_model, 'create_chat_completion'):
                    logger.error(f"LMH Instance [{self.instance_id}]: self.llm_model is not a valid Llama instance for generation.")
                    return None

                completion_params = {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if stop:
                    completion_params["stop"] = stop
                
                response = self.llm_model.create_chat_completion(**completion_params) 
                
                if response and response.get("choices") and response["choices"][0].get("message"):
                    content = response["choices"][0]["message"].get("content")
                    logger.info(f"LMH Instance [{self.instance_id}]: LLM generation successful (sync part). Output (first 100 chars): {str(content)[:100]}...")
                    return content
                else:
                    logger.warning(f"LMH Instance [{self.instance_id}]: LLM generation did not return expected content (sync part). Response: {response}")
                    return None
            except Exception as e_sync:
                logger.error(f"LMH Instance [{self.instance_id}]: Error during synchronous LLM call: {e_sync}", exc_info=True)
                return None

        try:
            generated_content = await asyncio.to_thread(_blocking_llm_call)
            return generated_content
        except Exception as e_async:
            logger.error(f"LMH Instance [{self.instance_id}]: Error in asyncio.to_thread for LLM call: {e_async}", exc_info=True)
            return None

    def close_embedding_pool(self):
        """Safely close and join the multiprocessing pool."""
        logger.info(f"LMH Instance [{self.instance_id}]: Attempting to close embedding pool...")
        if self._embedding_pool:
            try:
                self._embedding_pool.close()
                self._embedding_pool.join()
                self._embedding_pool = None
                logger.info(f"LMH Instance [{self.instance_id}]: Embedding pool closed and joined successfully.")
            except Exception as e_close:
                logger.error(f"LMH Instance [{self.instance_id}]: Error closing embedding pool: {e_close}", exc_info=True)
                if self._embedding_pool:
                    try:
                        logger.warning(f"LMH Instance [{self.instance_id}]: Pool close/join failed, attempting terminate.")
                        self._embedding_pool.terminate()
                        self._embedding_pool.join()
                    except Exception as e_term:
                        logger.error(f"LMH Instance [{self.instance_id}]: Error terminating embedding pool: {e_term}", exc_info=True)
                self._embedding_pool = None
        else:
            logger.info(f"LMH Instance [{self.instance_id}]: Embedding pool was not initialized or already closed.")

    def __del__(self):
        """Destructor to ensure the pool is closed when the object is garbage collected."""
        logger.info(f"LMH Instance [{self.instance_id}]: __del__ called. Ensuring embedding pool is closed.")
        self.close_embedding_pool()