# 文件: zhz_rag/llm/local_model_handler.py
import os
import logging
from typing import List, Optional, Dict
from llama_cpp import Llama
import asyncio
import numpy as np


logger = logging.getLogger(__name__) # 使用模块级logger
# 建议在调用此模块的顶层配置日志，例如在 rag_api_service.py

def l2_normalize_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
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
            logger.error(f"Error during L2 normalization of an embedding in LocalModelHandler: {e_norm}", exc_info=True)
            normalized_embeddings.append(emb_list)
    return normalized_embeddings


class LocalModelHandler:
    def __init__(
        self,
        llm_model_path: Optional[str] = None, # 用于文本生成的LLM模型路径
        embedding_model_path: Optional[str] = None, # 用于文本嵌入的GGUF模型路径
        n_ctx_llm: int = 4096,
        n_gpu_layers_llm: int = 0, # 默认为0，表示CPU
        n_ctx_embed: int = 2048, # 嵌入模型的上下文窗口
        n_gpu_layers_embed: int = 0 # 默认为0，表示CPU
    ):
        self.llm_model: Optional[Llama] = None
        self.embedding_model: Optional[Llama] = None
        self.embedding_model_dimension: Optional[int] = None # 用于存储嵌入维度

        if llm_model_path:
            try:
                logger.info(f"LocalModelHandler: Loading LLM model from: {llm_model_path}")
                self.llm_model = Llama(
                    model_path=llm_model_path,
                    n_ctx=n_ctx_llm,
                    n_gpu_layers=n_gpu_layers_llm,
                    verbose=False 
                )
                logger.info("LocalModelHandler: LLM model loaded successfully.")
            except Exception as e:
                logger.error(f"LocalModelHandler: Failed to load LLM model from {llm_model_path}: {e}", exc_info=True)

        if embedding_model_path:
            try:
                logger.info(f"LocalModelHandler: Loading embedding model from: {embedding_model_path}")
                self.embedding_model = Llama(
                    model_path=embedding_model_path,
                    n_ctx=n_ctx_embed,
                    n_gpu_layers=n_gpu_layers_embed,
                    embedding=True, # <--- 关键: 必须设置为 True for GGUF embedding models
                    verbose=False
                )
                logger.info("LocalModelHandler: Embedding model loaded successfully.")
                try:
                    test_string_for_dimension = "hello world" 
                    test_embedding_vector = self.embedding_model.embed(test_string_for_dimension)
                    determined_dimension = None
                    if hasattr(test_embedding_vector, 'shape') and len(test_embedding_vector.shape) == 1:
                        determined_dimension = test_embedding_vector.shape[0]
                    elif isinstance(test_embedding_vector, list):
                         determined_dimension = len(test_embedding_vector)
                    
                    # --- 修改开始：强制维度如果检测不正确 ---
                    if determined_dimension is None or determined_dimension < 100: # 假设维度小于100都是不合理的
                        logger.warning(f"LocalModelHandler: Detected embedding dimension as {determined_dimension}, which is too low. Forcing to 1024 for Qwen3-Embedding-0.6B.")
                        self.embedding_model_dimension = 1024
                    else:
                        self.embedding_model_dimension = determined_dimension
                    # --- 修改结束 ---
                    logger.info(f"LocalModelHandler: Final embedding dimension set to: {self.embedding_model_dimension}")
                except Exception as e_dim:
                    logger.warning(f"LocalModelHandler: Could not automatically determine embedding dimension: {e_dim}. It might need to be set manually or checked later.")
                    # Qwen3-Embedding-0.6B 的维度是 1024，可以作为后备
                    # self.embedding_model_dimension = 1024 
                    # logger.info(f"LocalModelHandler: Assuming embedding dimension 1024 for Qwen3-Embedding-0.6B.")


            except Exception as e:
                logger.error(f"LocalModelHandler: Failed to load embedding model from {embedding_model_path}: {e}", exc_info=True)
        
        if not self.llm_model and not self.embedding_model:
            logger.warning("LocalModelHandler initialized without any models loaded.")

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.embedding_model:
            logger.error("LocalModelHandler: Embedding model is not loaded. Cannot embed documents.")
            return [[] for _ in texts] 
        if not texts:
            return []
        
        processed_texts: List[str] = []
        for text_item in texts:
            if isinstance(text_item, str):
                if text_item and not text_item.endswith("<|endoftext|>"):
                    processed_texts.append(text_item + "<|endoftext|>")
                else:
                    processed_texts.append(text_item)
            else:
                logger.warning(f"LocalModelHandler received non-string item in texts list for embedding: {type(text_item)}.")
                str_item = str(text_item)
                processed_texts.append(str_item + "<|endoftext|>" if str_item and not str_item.endswith("<|endoftext|>") else str_item)

        logger.info(f"LocalModelHandler: Embedding {len(processed_texts)} documents (async via to_thread)...")
        
        def _blocking_embed_docs():
            try:
                # 调用 llama-cpp-python 的 embed 方法
                embedding_results_from_llama = self.embedding_model.embed(processed_texts)
                
                raw_embeddings_list_for_norm: List[List[float]] = []

                # --- 关键逻辑修改开始 ---
                if not embedding_results_from_llama:
                    logger.warning("LocalModelHandler: self.embedding_model.embed() returned None or empty.")
                    return [([0.0] * self.embedding_model_dimension) if self.embedding_model_dimension else [] for _ in processed_texts]

                # 检查返回的是否是单个向量 (当 processed_texts 只有一个元素时，某些版本的 llama-cpp-python 可能直接返回一个向量)
                if len(processed_texts) == 1 and isinstance(embedding_results_from_llama, (np.ndarray, list)) and \
                   ( (isinstance(embedding_results_from_llama, np.ndarray) and embedding_results_from_llama.ndim == 1) or \
                     (isinstance(embedding_results_from_llama, list) and all(isinstance(x, (float, int)) for x in embedding_results_from_llama)) ):
                    logger.debug("LocalModelHandler: embed() returned a single vector for single input text.")
                    single_embedding = embedding_results_from_llama.tolist() if isinstance(embedding_results_from_llama, np.ndarray) else embedding_results_from_llama
                    if len(single_embedding) == self.embedding_model_dimension:
                        raw_embeddings_list_for_norm.append(single_embedding)
                    else:
                        logger.warning(f"LocalModelHandler: Single embedding dimension mismatch. Expected {self.embedding_model_dimension}, got {len(single_embedding)}. Using zero vector.")
                        raw_embeddings_list_for_norm.append([0.0] * self.embedding_model_dimension if self.embedding_model_dimension else [])
                # 检查返回的是否是向量列表 (当 processed_texts 有多个元素时，或者单个元素也可能返回列表的列表)
                elif isinstance(embedding_results_from_llama, list) and all(isinstance(item, (np.ndarray, list)) for item in embedding_results_from_llama):
                    logger.debug("LocalModelHandler: embed() returned a list of vectors/arrays.")
                    for item_idx, item in enumerate(embedding_results_from_llama):
                        current_embedding: List[float] = []
                        if isinstance(item, np.ndarray) and item.ndim == 1:
                            current_embedding = item.tolist()
                        elif isinstance(item, list) and all(isinstance(x, (float, int)) for x in item):
                            current_embedding = item
                        else:
                            logger.warning(f"LocalModelHandler: Unexpected item type or structure at index {item_idx} in embedding results list: {type(item)}. Using zero vector.")
                        
                        if len(current_embedding) == self.embedding_model_dimension:
                            raw_embeddings_list_for_norm.append(current_embedding)
                        else:
                            logger.warning(f"LocalModelHandler: Embedding dimension mismatch for item {item_idx}. Expected {self.embedding_model_dimension}, got {len(current_embedding)}. Using zero vector.")
                            raw_embeddings_list_for_norm.append([0.0] * self.embedding_model_dimension if self.embedding_model_dimension else [])
                else:
                    logger.error(f"LocalModelHandler: Unexpected return type or structure from self.embedding_model.embed(): {type(embedding_results_from_llama)}. Cannot process. Value preview: {str(embedding_results_from_llama)[:200]}")
                    return [([0.0] * self.embedding_model_dimension) if self.embedding_model_dimension else [] for _ in processed_texts]

                normalized_embeddings_list = l2_normalize_embeddings(raw_embeddings_list_for_norm)
                logger.info(f"LocalModelHandler: Successfully embedded and normalized {len(normalized_embeddings_list)} documents (sync part).")
                return normalized_embeddings_list
            except Exception as e_sync_embed_docs:
                logger.error(f"LocalModelHandler: Error during synchronous document embedding: {e_sync_embed_docs}", exc_info=True)
                return [([0.0] * self.embedding_model_dimension) if self.embedding_model_dimension else [] for _ in processed_texts]

        try:
            return await asyncio.to_thread(_blocking_embed_docs)
        except Exception as e_async_embed_docs:
            logger.error(f"LocalModelHandler: Error in asyncio.to_thread for document embedding: {e_async_embed_docs}", exc_info=True)
            return [([0.0] * self.embedding_model_dimension) if self.embedding_model_dimension else [] for _ in texts]

    async def embed_query(self, text: str) -> List[float]:
        if not self.embedding_model:
            logger.error("LocalModelHandler: Embedding model is not loaded. Cannot embed query.")
            return []
        if not text: 
            return []

        processed_text = text + "<|endoftext|>" if not text.endswith("<|endoftext|>") else text
            
        logger.info(f"LocalModelHandler: Embedding query (first 100 chars): {processed_text[:100]}...")

        def _blocking_embed_query():
            try:
                embedding_result = self.embedding_model.embed(processed_text) # processed_text 是单个字符串
                
                # --- 修改开始：直接检查返回类型 ---
                if isinstance(embedding_result, np.ndarray):
                    # 如果是 numpy 数组，转换为 list
                    raw_embedding_list_for_norm = [embedding_result.tolist()]
                elif isinstance(embedding_result, list) and all(isinstance(x, (float, int)) for x in embedding_result):
                    # 如果直接是 List[float] 或 List[int]
                    raw_embedding_list_for_norm = [embedding_result]
                else:
                    logger.error(f"LocalModelHandler: Unexpected return type from self.embedding_model.embed(): {type(embedding_result)}. Value: {str(embedding_result)[:100]}")
                    return [] # 返回空列表表示失败
                # --- 修改结束 ---
                    
                normalized_embedding_list_of_list = l2_normalize_embeddings(raw_embedding_list_for_norm)
                final_embedding = normalized_embedding_list_of_list[0] if normalized_embedding_list_of_list and normalized_embedding_list_of_list[0] else []
                logger.info(f"LocalModelHandler: Successfully embedded and normalized query (sync part). Dimension: {len(final_embedding)}")
                return final_embedding
            except Exception as e_sync_embed_query:
                logger.error(f"LocalModelHandler: Error during synchronous query embedding: {e_sync_embed_query}", exc_info=True)
                return []
        try:
            return await asyncio.to_thread(_blocking_embed_query)
        except Exception as e_async_embed_query:
            logger.error(f"LocalModelHandler: Error in asyncio.to_thread for query embedding: {e_async_embed_query}", exc_info=True)
            return []


    def get_embedding_dimension(self) -> Optional[int]:
        """获取嵌入模型的输出维度。"""
        if self.embedding_model_dimension:
            return self.embedding_model_dimension
        
        # 如果初始化时未能获取，再次尝试
        if self.embedding_model:
            try:
                test_emb = self.embedding_model.embed("dimension_check")
                self.embedding_model_dimension = len(test_emb)
                logger.info(f"LocalModelHandler: Determined embedding dimension on demand: {self.embedding_model_dimension}")
                return self.embedding_model_dimension
            except Exception as e:
                logger.warning(f"LocalModelHandler: Could not determine embedding dimension on demand: {e}")
        return None # 如果还是无法确定

    async def generate_text_with_local_llm(self, messages: List[Dict[str,str]], temperature: float = 0.1, max_tokens: int = 1024, stop: Optional[List[str]]=None) -> Optional[str]:
        """
        使用本地加载的LLM模型生成文本。
        这个方法是异步的，因为它可能在 FastAPI 的异步端点中被调用。
        但 llama-cpp-python 的调用本身是同步阻塞的，所以我们用 asyncio.to_thread。
        """
        if not self.llm_model:
            logger.error("LocalModelHandler: LLM model is not loaded. Cannot generate text.")
            return None
        
        logger.info(f"LocalModelHandler: Generating text with local LLM. Message count: {len(messages)}")
        
        # llama-cpp-python 的调用是同步的，如果在异步函数中直接调用会阻塞事件循环
        # 所以我们使用 asyncio.to_thread 将其包装起来在单独的线程中运行
        def _blocking_llm_call():
            try:
                completion_params = {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if stop:
                    completion_params["stop"] = stop
                
                response = self.llm_model.create_chat_completion(**completion_params) # type: ignore
                
                if response and response.get("choices") and response["choices"][0].get("message"):
                    content = response["choices"][0]["message"].get("content")
                    logger.info(f"LocalModelHandler: LLM generation successful (sync part). Output (first 100 chars): {str(content)[:100]}...")
                    return content
                else:
                    logger.warning(f"LocalModelHandler: LLM generation did not return expected content (sync part). Response: {response}")
                    return None
            except Exception as e_sync:
                logger.error(f"LocalModelHandler: Error during synchronous LLM call: {e_sync}", exc_info=True)
                return None # 或者抛出异常由上层捕获

        try:
            generated_content = await asyncio.to_thread(_blocking_llm_call)
            return generated_content
        except Exception as e_async: # 捕获 to_thread 可能抛出的其他错误
            logger.error(f"LocalModelHandler: Error in asyncio.to_thread for LLM call: {e_async}", exc_info=True)
            return None