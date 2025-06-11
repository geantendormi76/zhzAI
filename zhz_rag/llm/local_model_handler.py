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
        
        try:
            raw_embeddings_list = self.embedding_model.embed(processed_texts_for_block)

            if not isinstance(raw_embeddings_list, list) or len(raw_embeddings_list) != len(processed_texts_for_block):
                logger.error(f"LMH (_blocking_embed_docs): Expected a list of {len(processed_texts_for_block)} embeddings, "
                               f"but got {type(raw_embeddings_list)} with length {len(raw_embeddings_list) if isinstance(raw_embeddings_list, list) else 'N/A'}. Using zero vectors.")
                return [list(default_zero_vector) for _ in processed_texts_for_block]

            final_embeddings: List[List[float]] = []
            for i, emb_item in enumerate(raw_embeddings_list):
                current_embedding_list: List[float] = []
                if isinstance(emb_item, np.ndarray):
                    if emb_item.ndim == 1:
                        current_embedding_list = emb_item.tolist()
                    elif emb_item.ndim == 2 and emb_item.shape[0] == 1: 
                        current_embedding_list = emb_item[0].tolist()
                    else:
                        logger.warning(f"LMH (_blocking_embed_docs): Item {i} is an np.ndarray with unexpected shape {emb_item.shape}. Using zero vector.")
                elif isinstance(emb_item, list):
                    if len(emb_item) == 1 and isinstance(emb_item[0], list) and all(isinstance(x, (float, int)) for x in emb_item[0]):
                        current_embedding_list = [float(x) for x in emb_item[0]]
                    elif all(isinstance(x, (float, int)) for x in emb_item):
                        current_embedding_list = [float(x) for x in emb_item]
                    else:
                        logger.warning(f"LMH (_blocking_embed_docs): Item {i} is a list with unexpected inner types. Using zero vector.")
                else:
                    logger.warning(f"LMH (_blocking_embed_docs): Item {i} is not np.ndarray or list. Type: {type(emb_item)}. Using zero vector.")

                if len(current_embedding_list) == self.embedding_model_dimension:
                    final_embeddings.append(current_embedding_list)
                else:
                    if current_embedding_list: 
                        logger.warning(f"LMH (_blocking_embed_docs): Embedding for item {i} has incorrect dimension after parsing. Expected {self.embedding_model_dimension}, got {len(current_embedding_list)}. Using zero vector.")
                    else: 
                        logger.warning(f"LMH (_blocking_embed_docs): Failed to parse embedding for item {i}. Using zero vector.")
                    final_embeddings.append(list(default_zero_vector))
            
            normalized_embeddings = l2_normalize_embeddings(final_embeddings)
            logger.info(f"LMH: Successfully processed {len(normalized_embeddings)} document embeddings (sync part). First embedding dim: {len(normalized_embeddings[0]) if normalized_embeddings and normalized_embeddings[0] else 'N/A'}")
            return normalized_embeddings
        except Exception as e_sync_embed_docs:
            logger.error(f"LMH: Error during synchronous document embedding: {e_sync_embed_docs}", exc_info=True)
            return [list(default_zero_vector) for _ in processed_texts_for_block]

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