# 文件: zhz_rag/llm/local_model_handler.py
import os
import logging
from typing import List, Optional, Dict, Any
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
                    embedding_results_from_llama = self.embedding_model.embed(processed_texts) # processed_texts 是 List[str]
                    
                    processed_embeddings_for_norm: List[List[float]] = []

                    # --- 关键逻辑修改开始 ---
                    if not embedding_results_from_llama:
                        logger.warning("LocalModelHandler (_blocking_embed_docs): self.embedding_model.embed() returned None or empty for batch.")
                        return [([0.0] * self.embedding_model_dimension) if self.embedding_model_dimension else [] for _ in processed_texts]

                    if isinstance(embedding_results_from_llama, list):
                        for item_idx, item in enumerate(embedding_results_from_llama):
                            current_embedding_list: List[float] = []
                            if isinstance(item, np.ndarray):
                                if item.ndim == 1: # 预期的 NumPy 1D 数组
                                    current_embedding_list = item.tolist()
                                # llama.cpp embed(List[str]) 不应该返回多维数组给单个item，但以防万一
                                elif item.ndim == 2 and item.shape[0] == 1: 
                                    current_embedding_list = item[0].tolist()
                                else:
                                    logger.warning(f"LocalModelHandler (_blocking_embed_docs): Unexpected NumPy array shape for item {item_idx}: {item.shape}. Using zero vector.")
                            elif isinstance(item, list):
                                # 检查是否是 List[float]
                                if all(isinstance(x, (float, int)) for x in item):
                                    current_embedding_list = item
                                # 不应该出现 List[List[float]] for a single item from batch, 但以防万一
                                elif len(item) == 1 and isinstance(item[0], list) and all(isinstance(x, (float, int)) for x in item[0]):
                                    current_embedding_list = item[0]
                                else:
                                     logger.warning(f"LocalModelHandler (_blocking_embed_docs): Unexpected list structure for item {item_idx}. Got list, but not List[float]. Type of first element: {type(item[0]) if item else 'EmptyList'}. Using zero vector.")
                            else:
                                logger.warning(f"LocalModelHandler (_blocking_embed_docs): Unexpected item type at index {item_idx} in embedding results list: {type(item)}. Using zero vector.")
                            
                            # 维度检查和填充
                            if len(current_embedding_list) == self.embedding_model_dimension:
                                processed_embeddings_for_norm.append(current_embedding_list)
                            else:
                                logger.warning(f"LocalModelHandler (_blocking_embed_docs): Embedding dimension mismatch for item {item_idx}. Expected {self.embedding_model_dimension}, got {len(current_embedding_list)}. Using zero vector.")
                                processed_embeddings_for_norm.append([0.0] * self.embedding_model_dimension if self.embedding_model_dimension else [])
                    else:
                        logger.error(f"LocalModelHandler (_blocking_embed_docs): Unexpected return type from self.embedding_model.embed() for multiple texts: {type(embedding_results_from_llama)}. Expected List.")
                        return [([0.0] * self.embedding_model_dimension) if self.embedding_model_dimension else [] for _ in processed_texts]
                    # --- 关键逻辑修改结束 ---

                    normalized_embeddings_list = l2_normalize_embeddings(processed_embeddings_for_norm)
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

        def _blocking_embed_query(self) -> List[float]: # 添加 self
                try:
                    # processed_text 应该在调用此方法前已准备好
                    embedding_result = self.embedding_model.embed(processed_text) 
                    
                    single_embedding_list: List[float] = []

                    # --- 增强的类型和结构检查 ---
                    if isinstance(embedding_result, np.ndarray):
                        if embedding_result.ndim == 1:
                            single_embedding_list = embedding_result.tolist()
                        elif embedding_result.ndim == 2 and embedding_result.shape[0] == 1:
                            single_embedding_list = embedding_result[0].tolist()
                        else:
                            logger.error(f"LMH (_blocking_embed_query): Unexpected NumPy array shape: {embedding_result.shape} for '{processed_text[:50]}...'")
                    elif isinstance(embedding_result, list):
                        # 情况1: 直接是 List[float]
                        if all(isinstance(x, (float, int)) for x in embedding_result):
                            single_embedding_list = [float(x) for x in embedding_result] # 确保是float
                        # 情况2: 是 List[List[float]] 且内部只有一个列表 (GGUF对单个输入可能返回这个)
                        elif len(embedding_result) == 1 and isinstance(embedding_result[0], list) and \
                             all(isinstance(x, (float, int)) for x in embedding_result[0]):
                            single_embedding_list = [float(x) for x in embedding_result[0]] # 确保是float
                        else:
                            logger.error(f"LMH (_blocking_embed_query): Unexpected list structure from embed() for '{processed_text[:50]}...'. Got list, but not List[float] or List[List[float]] with one element. Type of first element: {type(embedding_result[0]) if embedding_result else 'EmptyList'}.")
                    else:
                        logger.error(f"LMH (_blocking_embed_query): Unexpected return type from embed() for '{processed_text[:50]}...': {type(embedding_result)}. Value: {str(embedding_result)[:100]}")
                    
                    # 维度检查和归一化 (如果成功提取了 embedding)
                    if single_embedding_list:
                        if len(single_embedding_list) == self.embedding_model_dimension:
                            # 对单个向量列表进行归一化 (l2_normalize_embeddings 期望 List[List[float]])
                            normalized_embedding_list_of_list = l2_normalize_embeddings([single_embedding_list])
                            final_embedding = normalized_embedding_list_of_list[0] if normalized_embedding_list_of_list and normalized_embedding_list_of_list[0] else []
                        else:
                            logger.warning(f"LMH (_blocking_embed_query): Embedding dimension mismatch for '{processed_text[:50]}...'. Expected {self.embedding_model_dimension}, got {len(single_embedding_list)}. Using zero vector.")
                            final_embedding = [0.0] * self.embedding_model_dimension if self.embedding_model_dimension else []
                    else: # 如果 single_embedding_list 为空 (表示提取或转换失败)
                        logger.warning(f"LMH (_blocking_embed_query): Failed to extract a valid single embedding for '{processed_text[:50]}...'. Using zero vector.")
                        final_embedding = [0.0] * self.embedding_model_dimension if self.embedding_model_dimension else []
                    
                    logger.info(f"LMH: Successfully processed query embedding (sync part). Dimension: {len(final_embedding)}")
                    return final_embedding
                except Exception as e_sync_embed_query:
                    logger.error(f"LMH: Error during synchronous query embedding for '{processed_text[:50]}...': {e_sync_embed_query}", exc_info=True)
                    return [0.0] * self.embedding_model_dimension if self.embedding_model_dimension else [] # 返回零向量


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