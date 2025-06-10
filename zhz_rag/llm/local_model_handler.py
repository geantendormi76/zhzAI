# 文件: zhz_rag/llm/local_model_handler.py
import os
import logging
from typing import List, Optional, Dict
from llama_cpp import Llama
import asyncio
import numpy as np

logger = logging.getLogger(__name__)

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
        llm_model_path: Optional[str] = None,
        embedding_model_path: Optional[str] = None,
        n_ctx_llm: int = 4096,
        n_gpu_layers_llm: int = 0,
        n_ctx_embed: int = 2048,
        n_gpu_layers_embed: int = 0
    ):
        self.llm_model: Optional[Llama] = None
        self.embedding_model: Optional[Llama] = None
        self.embedding_model_dimension: Optional[int] = None

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
                    verbose=False
                )
                logger.info("LocalModelHandler: Embedding model loaded successfully.")
                try:
                    test_string_for_dimension = "hello world"
                    test_embedding_vector = self.embedding_model.embed(test_string_for_dimension)
                    determined_dimension = None
                    if hasattr(test_embedding_vector, 'shape') and len(test_embedding_vector.shape) == 1:
                        determined_dimension = test_embedding_vector.shape[0]
                    elif isinstance(test_embedding_vector, list) and all(isinstance(x, (float, int)) for x in test_embedding_vector):
                         determined_dimension = len(test_embedding_vector)
                    
                    if determined_dimension is None or determined_dimension < 100:
                        logger.warning(f"LocalModelHandler: Detected embedding dimension as {determined_dimension}, which is too low. Forcing to 1024 for Qwen3-Embedding-0.6B.")
                        self.embedding_model_dimension = 1024
                    else:
                        self.embedding_model_dimension = determined_dimension
                    logger.info(f"LocalModelHandler: Final embedding dimension set to: {self.embedding_model_dimension}")
                except Exception as e_dim:
                    logger.warning(f"LocalModelHandler: Could not automatically determine embedding dimension: {e_dim}. Forcing to 1024.")
                    self.embedding_model_dimension = 1024
                    logger.info(f"LocalModelHandler: Final embedding dimension set to: {self.embedding_model_dimension}")
            except Exception as e:
                logger.error(f"LocalModelHandler: Failed to load embedding model from {embedding_model_path}: {e}", exc_info=True)
        
        if not self.llm_model and not self.embedding_model:
            logger.warning("LocalModelHandler initialized without any models loaded.")

    def _blocking_embed_documents_internal(self, processed_texts: List[str]) -> List[List[float]]:
        """Synchronous internal method to embed multiple documents."""
        if not self.embedding_model: return [[] for _ in processed_texts]
        try:
            embedding_results_from_llama = self.embedding_model.embed(processed_texts)
            processed_embeddings_for_norm: List[List[float]] = []

            if not embedding_results_from_llama:
                logger.warning("LMH (_blocking_embed_docs_internal): embed() returned None or empty for batch.")
                return [([0.0] * self.embedding_model_dimension) if self.embedding_model_dimension else [] for _ in processed_texts]

            if isinstance(embedding_results_from_llama, list):
                if not embedding_results_from_llama:
                     return [([0.0] * self.embedding_model_dimension) if self.embedding_model_dimension else [] for _ in processed_texts]

                if all(isinstance(item, list) and all(isinstance(x, (float, int)) for x in item) for item in embedding_results_from_llama):
                    logger.debug("LMH (_blocking_embed_docs_internal): embed() returned List[List[float]].")
                    for item_idx, item_list_float in enumerate(embedding_results_from_llama):
                        if len(item_list_float) == self.embedding_model_dimension:
                            processed_embeddings_for_norm.append([float(x) for x in item_list_float])
                        else:
                            logger.warning(f"LMH (_blocking_embed_docs_internal): Dim mismatch for item {item_idx} (already List[float]). Expected {self.embedding_model_dimension}, got {len(item_list_float)}. Using zero vector.")
                            processed_embeddings_for_norm.append([0.0] * self.embedding_model_dimension if self.embedding_model_dimension else [])
                elif all(isinstance(item, np.ndarray) for item in embedding_results_from_llama):
                    logger.debug("LMH (_blocking_embed_docs_internal): embed() returned List[np.ndarray].")
                    for item_idx, item_np_array in enumerate(embedding_results_from_llama):
                        if item_np_array.ndim == 1 and len(item_np_array) == self.embedding_model_dimension:
                            processed_embeddings_for_norm.append(item_np_array.tolist())
                        else:
                            logger.warning(f"LMH (_blocking_embed_docs_internal): Dim mismatch or wrong ndim for item {item_idx} (np.ndarray). Expected 1D of {self.embedding_model_dimension}, got shape {item_np_array.shape}. Using zero vector.")
                            processed_embeddings_for_norm.append([0.0] * self.embedding_model_dimension if self.embedding_model_dimension else [])
                else:
                    logger.error(f"LMH (_blocking_embed_docs_internal): embed() returned a list, but its elements are not uniformly List[float] or np.ndarray. Type of first element: {type(embedding_results_from_llama[0]) if embedding_results_from_llama else 'EmptyList'}")
                    return [([0.0] * self.embedding_model_dimension) if self.embedding_model_dimension else [] for _ in processed_texts]
            elif isinstance(embedding_results_from_llama, np.ndarray) and len(processed_texts) == 1: # Should not happen if processed_texts is always a list
                logger.debug("LMH (_blocking_embed_docs_internal): embed() returned a single np.ndarray for a single input text in batch (unexpected).")
                if embedding_results_from_llama.ndim == 1 and len(embedding_results_from_llama) == self.embedding_model_dimension:
                    processed_embeddings_for_norm.append(embedding_results_from_llama.tolist())
                else:
                    logger.warning(f"LMH (_blocking_embed_docs_internal): Dim mismatch or wrong ndim for single np.ndarray. Expected 1D of {self.embedding_model_dimension}, got shape {embedding_results_from_llama.shape}. Using zero vector.")
                    processed_embeddings_for_norm.append([0.0] * self.embedding_model_dimension if self.embedding_model_dimension else [])
            else:
                logger.error(f"LMH (_blocking_embed_docs_internal): Unexpected return type from embed() for batch: {type(embedding_results_from_llama)}. Value: {str(embedding_results_from_llama)[:200]}")
                return [([0.0] * self.embedding_model_dimension) if self.embedding_model_dimension else [] for _ in processed_texts]

            normalized_embeddings_list = l2_normalize_embeddings(processed_embeddings_for_norm)
            logger.info(f"LMH: Successfully processed {len(normalized_embeddings_list)} document embeddings (sync part).")
            return normalized_embeddings_list
        except Exception as e_sync_embed_docs:
            logger.error(f"LMH: Error during synchronous document embedding: {e_sync_embed_docs}", exc_info=True)
            return [([0.0] * self.embedding_model_dimension) if self.embedding_model_dimension else [] for _ in processed_texts]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.embedding_model:
            logger.error("LocalModelHandler: Embedding model is not loaded. Cannot embed documents.")
            return [([0.0] * self.embedding_model_dimension) if self.embedding_model_dimension else [] for _ in texts]
        if not texts:
            return []
        
        processed_texts_for_block: List[str] = []
        for text_item in texts:
            if isinstance(text_item, str):
                if text_item and not text_item.endswith("<|endoftext|>"):
                    processed_texts_for_block.append(text_item + "<|endoftext|>")
                else:
                    processed_texts_for_block.append(text_item)
            else:
                logger.warning(f"LocalModelHandler received non-string item in texts list for embedding: {type(text_item)}.")
                str_item = str(text_item)
                processed_texts_for_block.append(str_item + "<|endoftext|>" if str_item and not str_item.endswith("<|endoftext|>") else str_item)

        logger.info(f"LocalModelHandler: Embedding {len(processed_texts_for_block)} documents (async via to_thread)...")
        try:
            return await asyncio.to_thread(self._blocking_embed_documents_internal, processed_texts_for_block)
        except Exception as e_async_embed_docs:
            logger.error(f"LocalModelHandler: Error in asyncio.to_thread for document embedding: {e_async_embed_docs}", exc_info=True)
            return [([0.0] * self.embedding_model_dimension) if self.embedding_model_dimension else [] for _ in texts]

    def _blocking_embed_query_internal(self, processed_text: str) -> List[float]:
        """Synchronous internal method to embed a single query."""
        if not self.embedding_model: return []
        try:
            embedding_result = self.embedding_model.embed(processed_text)
            single_embedding_list: List[float] = []

            if isinstance(embedding_result, np.ndarray):
                if embedding_result.ndim == 1:
                    single_embedding_list = embedding_result.tolist()
                elif embedding_result.ndim == 2 and embedding_result.shape[0] == 1:
                    single_embedding_list = embedding_result[0].tolist()
                else:
                    logger.error(f"LMH (_blocking_embed_query_internal): Unexpected NumPy array shape: {embedding_result.shape} for '{processed_text[:50]}...'")
            elif isinstance(embedding_result, list):
                if all(isinstance(x, (float, int)) for x in embedding_result):
                    single_embedding_list = [float(x) for x in embedding_result]
                elif len(embedding_result) == 1 and isinstance(embedding_result[0], list) and \
                     all(isinstance(x, (float, int)) for x in embedding_result[0]):
                    single_embedding_list = [float(x) for x in embedding_result[0]]
                else:
                    logger.error(f"LMH (_blocking_embed_query_internal): Unexpected list structure from embed() for '{processed_text[:50]}...'. Got list, but not List[float] or List[List[float]] with one element. Type of first element: {type(embedding_result[0]) if embedding_result else 'EmptyList'}.")
            else:
                logger.error(f"LMH (_blocking_embed_query_internal): Unexpected return type from embed() for '{processed_text[:50]}...': {type(embedding_result)}. Value: {str(embedding_result)[:100]}")
            
            if not single_embedding_list or (self.embedding_model_dimension and len(single_embedding_list) != self.embedding_model_dimension):
                logger.warning(f"LMH (_blocking_embed_query_internal): Embedding dimension mismatch or empty. Expected {self.embedding_model_dimension}, got {len(single_embedding_list) if single_embedding_list else 'empty'}. Using zero vector. Processed text: '{processed_text[:50]}...'")
                final_embedding = [0.0] * self.embedding_model_dimension if self.embedding_model_dimension else []
            else:
                normalized_embedding_list_of_list = l2_normalize_embeddings([single_embedding_list])
                final_embedding = normalized_embedding_list_of_list[0] if normalized_embedding_list_of_list and normalized_embedding_list_of_list[0] else []
            
            logger.info(f"LMH: Successfully processed query embedding (sync part). Dimension: {len(final_embedding)}")
            return final_embedding
        except Exception as e_sync_embed_query:
            logger.error(f"LMH: Error during synchronous query embedding for '{processed_text[:50]}...': {e_sync_embed_query}", exc_info=True)
            return [0.0] * self.embedding_model_dimension if self.embedding_model_dimension else []

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
        if self.embedding_model_dimension:
            return self.embedding_model_dimension
        if self.embedding_model:
            try:
                # Re-attempt to determine dimension if not set during init
                test_string_for_dimension = "dimension_check_get"
                # Need to call the synchronous internal version if this method is called from a sync context
                # Or, make get_embedding_dimension async too if it needs to run the async embed_query.
                # For simplicity, assuming it might be called from sync context, we'll call a blocking version.
                # This part is tricky if get_dimension itself is called from an async context without await.
                # Let's assume for now it's called from a context where running a blocking call is okay
                # or that self.embedding_model_dimension was set during __init__.
                
                # A safer direct call to the blocking embed for dimension check:
                processed_text = test_string_for_dimension + "<|endoftext|>"
                temp_embedding = self._blocking_embed_query_internal(processed_text)
                if temp_embedding:
                    self.embedding_model_dimension = len(temp_embedding)
                    logger.info(f"LocalModelHandler: Determined embedding dimension on demand via get_dimension: {self.embedding_model_dimension}")
                    return self.embedding_model_dimension
            except Exception as e:
                logger.warning(f"LocalModelHandler: Could not determine embedding dimension on demand in get_dimension: {e}")
        logger.warning(f"LocalModelHandler: get_embedding_dimension returning stored dimension: {self.embedding_model_dimension} (may be None or forced 1024)")
        return self.embedding_model_dimension


    async def generate_text_with_local_llm(self, messages: List[Dict[str,str]], temperature: float = 0.1, max_tokens: int = 1024, stop: Optional[List[str]]=None) -> Optional[str]:
        if not self.llm_model:
            logger.error("LocalModelHandler: LLM model is not loaded. Cannot generate text.")
            return None
        
        logger.info(f"LocalModelHandler: Generating text with local LLM. Message count: {len(messages)}")
        
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