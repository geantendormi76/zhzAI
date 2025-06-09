# 文件: zhz_rag/core_rag/retrievers/embedding_functions.py
import logging
from typing import List, TYPE_CHECKING, Optional, Sequence
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
    此类的方法设计为同步的，以符合 ChromaDB EmbeddingFunction 接口的期望。
    它内部会同步地运行 LocalModelHandler 中的异步嵌入方法。
    """
    def __init__(self, model_handler: 'LocalModelHandler'):
        if model_handler is None or model_handler.embedding_model is None:
            logger.error("LlamaCppEmbeddingFunction initialized with no model_handler or no embedding model in handler.")
            raise ValueError("LocalModelHandler with a loaded embedding model is required.")
        self.model_handler = model_handler
        self._dimension: Optional[int] = None 
        logger.info("LlamaCppEmbeddingFunction initialized.")

    def _run_async_in_new_loop(self, coro):
        """辅助函数，在当前线程中创建一个新事件循环来运行协程。"""
        # 这种方式适用于在一个不由asyncio管理的线程中调用异步代码（例如由to_thread启动的线程）
        try:
            return asyncio.run(coro)
        except RuntimeError as e:
            if "cannot run event loop while another loop is running" in str(e):
                logger.error("LlamaCppEmbeddingFunction: Attempted to run asyncio.run() from within an existing event loop. This is not allowed for this function's design.")
                # 这种情况下，调用者（例如ChromaDBRetriever.retrieve）应该已经确保此函数在正确的线程中运行
                # 或者 LocalModelHandler 的方法本身就需要调整为完全同步（但这会阻塞其调用者，如果调用者是异步的）
                # 鉴于 ChromaDB 的同步接口，这是一个设计上的权衡点。
                # 目前，我们假设此函数总是在一个可以安全创建新循环的线程中被调用。
            raise # 将原始错误重新抛出，以便上层能看到

    def __call__(self, input: Documents) -> Embeddings:
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
        
        logger.info(f"LlamaCppEmbeddingFunction: Generating embeddings for {len(processed_texts_for_handler)} processed texts.")
        if processed_texts_for_handler:
            logger.debug(f"LlamaCppEmbeddingFunction: First processed text for embedding: '{processed_texts_for_handler[0][:150]}...'")

        try:
            # 调用 LocalModelHandler 的异步 embed_documents 方法，并在此处同步运行它
            raw_embeddings_list = self._run_async_in_new_loop(
                self.model_handler.embed_documents(processed_texts_for_handler)
            )
            
            if raw_embeddings_list:
                logger.debug("LlamaCppEmbeddingFunction: Normalizing embeddings (L2).")
                embeddings_list = l2_normalize_embeddings(raw_embeddings_list) # l2_normalize_embeddings 是同步的
            else:
                embeddings_list = []
            
            if embeddings_list and self._dimension is None and embeddings_list[0]: 
                self._dimension = len(embeddings_list[0])
                logger.debug(f"LlamaCppEmbeddingFunction: Cached embedding dimension: {self._dimension}")
            elif embeddings_list and embeddings_list[0] and len(embeddings_list[0]) != self._dimension: 
                logger.warning(f"LlamaCppEmbeddingFunction: Inconsistent embedding dimension detected! "
                               f"Expected {self._dimension}, got {len(embeddings_list[0])}.")
            
            return embeddings_list
        except Exception as e:
            logger.error(f"LlamaCppEmbeddingFunction: Error during embedding generation: {e}", exc_info=True)
            return [[] for _ in input]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self.__call__(input=list(texts))

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return []
        # 调用 LocalModelHandler 的异步 embed_query 方法，并在此处同步运行它
        embedding_result = self._run_async_in_new_loop(
            self.model_handler.embed_query(text)
        )
        return embedding_result if embedding_result else []
    
    def get_dimension(self) -> Optional[int]:
        if self._dimension is None:
            self._dimension = self.model_handler.get_embedding_dimension()
            if self._dimension:
                logger.debug(f"LlamaCppEmbeddingFunction: Fetched dimension from model_handler: {self._dimension}")
        return self._dimension