# 文件: zhz_rag/llm/local_model_handler.py
import os
import logging
from typing import List, Optional, Dict
from llama_cpp import Llama
import asyncio
import numpy as np
import ctypes

logger = logging.getLogger(__name__)

def l2_normalize_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
    if not embeddings or not isinstance(embeddings, list):
        return []
    normalized_embeddings = []
    for emb_list in embeddings:
        if not emb_list or not isinstance(emb_list, list) or not all(isinstance(x, (float, int)) for x in emb_list):
            logger.warning(f"L2_NORM: Skipping invalid or empty inner list: {emb_list}")
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
        llm_model_path: Optional[str] = None,
        embedding_model_path: Optional[str] = None,
        n_ctx_llm: int = 4096,
        n_gpu_layers_llm: int = 0,
        n_ctx_embed: int = 2048,
        n_gpu_layers_embed: int = 0,
        pooling_type_embed: int = 2 # 0 for NONE, 1 for MEAN, 2 for CLS. Defaulting to CLS.
    ):
        self.llm_model: Optional[Llama] = None
        self.embedding_model: Optional[Llama] = None
        self.embedding_model_dimension: Optional[int] = None
        self.pooling_type_embed = pooling_type_embed # Store it

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
                    embedding=True,
                    pooling_type=self.pooling_type_embed, # Use the stored integer value
                    verbose=False
                )
                logger.info(f"LocalModelHandler: Embedding model loaded successfully with pooling_type={self.pooling_type_embed}.")
                
                if self.embedding_model:
                    self.embedding_model_dimension = self.embedding_model.n_embd()
                    logger.info(f"LocalModelHandler: Embedding dimension obtained from model metadata (n_embd()): {self.embedding_model_dimension}")
                    if not self.embedding_model_dimension or self.embedding_model_dimension <= 0:
                        logger.error(f"LocalModelHandler: n_embd() returned invalid dimension {self.embedding_model_dimension}. This should not happen.")
                        raise ValueError(f"Invalid embedding dimension from model: {self.embedding_model_dimension}")
                else:
                    logger.error("LocalModelHandler: Embedding model loaded but instance is None, cannot get dimension.")
                    raise RuntimeError("Embedding model failed to instantiate correctly.")

            except Exception as e:
                logger.error(f"LocalModelHandler: Failed to load embedding model or get dimension from {embedding_model_path}: {e}", exc_info=True)
                self.embedding_model = None 
                self.embedding_model_dimension = None
        
        if not self.llm_model and not self.embedding_model:
            logger.warning("LocalModelHandler initialized without any models loaded.")

    def _blocking_embed_documents_internal(self, processed_texts_for_block: List[str]) -> List[List[float]]:
        if not self.embedding_model or not self.embedding_model_dimension:
            logger.error("LMH (_blocking_embed_docs): Embedding model or dimension not initialized.")
            return [[] for _ in processed_texts_for_block]

        default_zero_vector = [0.0] * self.embedding_model_dimension
        
        if not processed_texts_for_block:
            logger.info("LMH (_blocking_embed_docs): Received empty list of texts to embed.")
            return []

        logger.info(f"LMH (_blocking_embed_docs): Preparing to embed {len(processed_texts_for_block)} documents using create_embedding (batch).")

        # 1. 识别有效文本及其原始索引
        valid_texts_with_indices: List[tuple[int, str]] = []
        for i, text in enumerate(processed_texts_for_block):
            if text and text.strip(): # 确保文本非空且去除空白后仍有内容
                valid_texts_with_indices.append((i, text))
            else:
                logger.warning(f"LMH (_blocking_embed_docs): Input text at original index {i} is empty or invalid. Will use zero vector. Text: '{text}'")

        if not valid_texts_with_indices:
            logger.info("LMH (_blocking_embed_docs): No valid texts found after filtering. Returning all zero vectors.")
            return [list(default_zero_vector) for _ in processed_texts_for_block]

        valid_texts_to_embed = [text for _, text in valid_texts_with_indices]
        logger.info(f"LMH (_blocking_embed_docs): Embedding {len(valid_texts_to_embed)} valid texts out of {len(processed_texts_for_block)} total.")

        # 2. 对有效文本进行批量嵌入
        raw_embeddings_for_valid_texts: List[List[float]] = []
        try:
            response = self.embedding_model.create_embedding(input=valid_texts_to_embed)
            if response and "data" in response and isinstance(response["data"], list):
                embeddings_data = response["data"]
                # 假设 create_embedding 返回的顺序与输入 valid_texts_to_embed 的顺序一致
                # 并且其 "index" 字段是相对于 valid_texts_to_embed 列表的索引
                if len(embeddings_data) == len(valid_texts_to_embed):
                    for item_idx, item in enumerate(embeddings_data):
                        if isinstance(item, dict) and "embedding" in item:
                            # 确认 item["index"] 是否与 item_idx 一致，如果不是，需要更复杂的排序
                            # llama-cpp-python 通常按输入顺序返回，其内部索引也是如此
                            if item.get("index") != item_idx:
                                logger.warning(f"LMH (_blocking_embed_docs): Embedding response index mismatch. Expected {item_idx}, got {item.get('index')}. Assuming order is preserved.")

                            embedding_vector = item["embedding"]
                            if isinstance(embedding_vector, list) and \
                               all(isinstance(x, (float, int)) for x in embedding_vector) and \
                               len(embedding_vector) == self.embedding_model_dimension:
                                raw_embeddings_for_valid_texts.append([float(x) for x in embedding_vector])
                            else:
                                logger.warning(f"LMH (_blocking_embed_docs): Valid text at valid_idx {item_idx} got invalid embedding vector or dimension. Using zero vector. Vector: {str(embedding_vector)[:100]}")
                                raw_embeddings_for_valid_texts.append(list(default_zero_vector))
                        else:
                            logger.warning(f"LMH (_blocking_embed_docs): Invalid item format in embedding response for valid text at valid_idx {item_idx}. Using zero vector.")
                            raw_embeddings_for_valid_texts.append(list(default_zero_vector))
                else: # 返回的嵌入数量与有效文本数量不匹配
                    logger.error(f"LMH (_blocking_embed_docs): Mismatch between number of embeddings received ({len(embeddings_data)}) and number of valid texts sent ({len(valid_texts_to_embed)}). Using zero vectors for all valid texts.")
                    raw_embeddings_for_valid_texts = [list(default_zero_vector) for _ in valid_texts_to_embed]
            else: # create_embedding 返回无效响应
                logger.error(f"LMH (_blocking_embed_docs): Invalid or empty response from create_embedding for valid texts: {str(response)[:200]}. Using zero vectors for all valid texts.")
                raw_embeddings_for_valid_texts = [list(default_zero_vector) for _ in valid_texts_to_embed]
        except Exception as e_batch_embed: # 捕获批量嵌入过程中的任何异常
            logger.error(f"LMH (_blocking_embed_docs): Error during batch embedding of valid texts: {e_batch_embed}", exc_info=True)
            raw_embeddings_for_valid_texts = [list(default_zero_vector) for _ in valid_texts_to_embed]

        # 3. 构建最终的嵌入列表，保持原始顺序，并为无效文本填充零向量
        final_embeddings_ordered: List[List[float]] = [list(default_zero_vector) for _ in processed_texts_for_block]
        
        valid_embedding_idx = 0
        for original_idx, _ in valid_texts_with_indices:
            if valid_embedding_idx < len(raw_embeddings_for_valid_texts):
                final_embeddings_ordered[original_idx] = raw_embeddings_for_valid_texts[valid_embedding_idx]
                valid_embedding_idx += 1
            else: # 应该不会发生，因为上面已经处理了数量不匹配的情况
                logger.error(f"LMH (_blocking_embed_docs): Logic error in mapping valid embeddings back. Should not happen.")
                final_embeddings_ordered[original_idx] = list(default_zero_vector) # 安全回退
        
        if len(final_embeddings_ordered) != len(processed_texts_for_block):
            logger.critical(f"LMH (_blocking_embed_docs): Final ordered embeddings length ({len(final_embeddings_ordered)}) "
                            f"mismatches input texts length ({len(processed_texts_for_block)}). This is a bug. Re-padding.")
            # 尽力确保输出列表长度与输入一致
            padded_final_embeddings = [list(default_zero_vector)] * len(processed_texts_for_block)
            for i in range(min(len(final_embeddings_ordered), len(processed_texts_for_block))):
                if final_embeddings_ordered[i] and len(final_embeddings_ordered[i]) == self.embedding_model_dimension:
                    padded_final_embeddings[i] = final_embeddings_ordered[i]
            final_embeddings_ordered = padded_final_embeddings

        normalized_embeddings = l2_normalize_embeddings(final_embeddings_ordered)
        logger.info(f"LMH: Successfully processed and normalized {len(normalized_embeddings)} document embeddings (batch, with invalid text handling).")
        return normalized_embeddings

    def _blocking_embed_query_internal(self, processed_text: str) -> List[float]:
        if not self.embedding_model or not self.embedding_model_dimension:
            logger.error("LMH (_blocking_embed_query): Embedding model or dimension not initialized.")
            return []
        
        default_zero_vector = [0.0] * self.embedding_model_dimension
        try:
            # llama-cpp-python's embed() for a single string with pooling_type set
            # should return a List[float] directly.
            embedding_vector = self.embedding_model.embed(processed_text)
            
            current_embedding_list: List[float] = []
            if isinstance(embedding_vector, np.ndarray):
                if embedding_vector.ndim == 1:
                    current_embedding_list = embedding_vector.tolist()
                else:
                    logger.warning(f"LMH (_blocking_embed_query): Query embedding (np.ndarray) has unexpected shape {embedding_vector.shape}. Using zero vector.")
            elif isinstance(embedding_vector, list):
                if all(isinstance(x, (float, int)) for x in embedding_vector):
                    current_embedding_list = [float(x) for x in embedding_vector]
                # Removed the List[List[float]] check for single query as it's less likely with pooling
                else:
                    logger.warning(f"LMH (_blocking_embed_query): Query embedding (list) has unexpected inner types. Using zero vector.")
            else:
                logger.warning(f"LMH (_blocking_embed_query): Query embedding is not np.ndarray or list. Type: {type(embedding_vector)}. Using zero vector.")

            if len(current_embedding_list) == self.embedding_model_dimension:
                normalized_embedding_list_of_list = l2_normalize_embeddings([current_embedding_list])
                final_embedding = normalized_embedding_list_of_list[0] if normalized_embedding_list_of_list and normalized_embedding_list_of_list[0] else list(default_zero_vector)
                logger.info(f"LMH: Successfully processed query embedding (sync part). Dimension: {len(final_embedding)}")
                return final_embedding
            else:
                if current_embedding_list:
                    logger.warning(f"LMH (_blocking_embed_query): Query embedding dim mismatch after parsing. Expected {self.embedding_model_dimension}, got {len(current_embedding_list)}. Using zero vector.")
                else:
                    logger.warning(f"LMH (_blocking_embed_query): Failed to parse query embedding. Using zero vector.")
                return list(default_zero_vector)
        except Exception as e_sync_embed_query:
            logger.error(f"LMH: Error during synchronous query embedding for '{processed_text[:50]}...': {e_sync_embed_query}", exc_info=True)
            return list(default_zero_vector)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.embedding_model:
            logger.error("LocalModelHandler: Embedding model is not loaded. Cannot embed documents.")
            return [[] for _ in texts] 
        if not texts:
            return []

        processed_texts = [
            (text + "<|endoftext|>" if text and not text.endswith("<|endoftext|>") else text)
            for text in texts
        ]
        logger.info(f"LocalModelHandler: Embedding {len(processed_texts)} documents (async via to_thread)...")
        try:
            return await asyncio.to_thread(self._blocking_embed_documents_internal, processed_texts)
        except Exception as e_async_embed_docs:
            logger.error(f"LocalModelHandler: Error in asyncio.to_thread for document embedding: {e_async_embed_docs}", exc_info=True)
            return [[0.0] * self.embedding_model_dimension if self.embedding_model_dimension else [] for _ in texts]

    async def embed_query(self, text: str) -> List[float]:
        if not self.embedding_model:
            logger.error("LocalModelHandler: Embedding model is not loaded. Cannot embed query.")
            return []
        if not text: 
            return []

        processed_text_for_block = text + "<|endoftext|>" if not text.endswith("<|endoftext|>") else text
        logger.info(f"LocalModelHandler: Embedding query (async via to_thread, first 100 chars): {processed_text_for_block[:100]}...")
        try:
            return await asyncio.to_thread(self._blocking_embed_query_internal, processed_text_for_block)
        except Exception as e_async_embed_query:
            logger.error(f"LocalModelHandler: Error in asyncio.to_thread for query embedding: {e_async_embed_query}", exc_info=True)
            return [0.0] * self.embedding_model_dimension if self.embedding_model_dimension else []
    
    def get_embedding_dimension(self) -> Optional[int]:
        if not self.embedding_model_dimension:
             logger.warning(f"LocalModelHandler: get_embedding_dimension() called but dimension is not set. This should have been set during __init__.")
        return self.embedding_model_dimension

    async def generate_text_with_local_llm(self, messages: List[Dict[str,str]], temperature: float = 0.1, max_tokens: int = 1024, stop: Optional[List[str]]=None) -> Optional[str]:
        if not self.llm_model:
            logger.error("LocalModelHandler: LLM model is not loaded. Cannot generate text.")
            return None
        
        logger.info(f"LocalModelHandler: Generating text with local ·LLM. Message count: {len(messages)}")
        
        def _blocking_llm_call():
            try:
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
                    logger.info(f"LocalModelHandler: LLM generation successful (sync part). Output (first 100 chars): {str(content)[:100]}...")
                    return content
                else:
                    logger.warning(f"LocalModelHandler: LLM generation did not return expected content (sync part). Response: {response}")
                    return None
            except Exception as e_sync:
                logger.error(f"LocalModelHandler: Error during synchronous LLM call: {e_sync}", exc_info=True)
                return None

        try:
            generated_content = await asyncio.to_thread(_blocking_llm_call)
            return generated_content
        except Exception as e_async:
            logger.error(f"LocalModelHandler: Error in asyncio.to_thread for LLM call: {e_async}", exc_info=True)
            return None