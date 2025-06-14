# 文件: zhz_rag/llm/embedding_process_worker.py

import os
import logging
from typing import List, Dict, Any, Optional
from llama_cpp import Llama
import numpy as np

# --- 全局变量，用于在子进程中缓存模型实例 ---
# 注意: 每个进程池中的工作进程会有自己的这个变量副本
_process_local_model_cache: Dict[str, Llama] = {}
_process_local_model_dimension_cache: Dict[str, int] = {}

# --- 日志配置 (与 LocalModelHandler 类似，但确保独立) ---
worker_logger = logging.getLogger("EmbeddingProcessWorker")
# 避免重复添加处理器，如果此模块被多次导入或以某种方式重新加载
if not worker_logger.hasHandlers():
    worker_logger.setLevel(logging.INFO) # 或者 DEBUG
    # 注意：在多进程环境中，日志输出到控制台可能交错。
    # 对于生产环境，可能需要更复杂的日志策略（如QueueHandler）。
    # 但对于调试，StreamHandler 也可以。
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - PID:%(process)d - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    worker_logger.addHandler(stream_handler)
    worker_logger.propagate = False


def l2_normalize_embeddings_worker(embeddings: List[List[float]]) -> List[List[float]]:
    if not embeddings or not isinstance(embeddings, list):
        return []
    normalized_embeddings = []
    for emb_list in embeddings:
        if not emb_list or not isinstance(emb_list, list) or not all(isinstance(x, (float, int)) for x in emb_list):
            worker_logger.warning(f"L2_NORM_WORKER: Skipping invalid or empty inner list: {emb_list}")
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
            worker_logger.error(f"Error during L2 normalization in EmbeddingProcessWorker: {e_norm}", exc_info=True)
            normalized_embeddings.append(emb_list)
    return normalized_embeddings


def _get_embedding_model_instance_in_worker(
    model_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    pooling_type: int
) -> Optional[Llama]:
    """
    在当前工作进程中获取或创建并缓存 Llama 嵌入模型实例。
    使用 model_path 作为缓存键。
    """
    global _process_local_model_cache
    global _process_local_model_dimension_cache

    if model_path in _process_local_model_cache:
        worker_logger.info(f"WORKER: Reusing cached embedding model for path: {model_path}")
        return _process_local_model_cache[model_path]

    worker_logger.info(f"WORKER: Attempting to load embedding model in process: {model_path}")
    try:
        model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            embedding=True,
            pooling_type=pooling_type,
            verbose=False # 在工作进程中减少冗余日志
        )
        dimension = model.n_embd()
        if not dimension or dimension <= 0:
            worker_logger.error(f"WORKER: Loaded model from {model_path} but got invalid dimension {dimension}.")
            return None
        
        _process_local_model_cache[model_path] = model
        _process_local_model_dimension_cache[model_path] = dimension
        worker_logger.info(f"WORKER: Embedding model loaded and cached for {model_path}. Dimension: {dimension}, Pooling: {pooling_type}")
        return model
    except Exception as e:
        worker_logger.error(f"WORKER: Failed to load embedding model in process for path {model_path}: {e}", exc_info=True)
        return None


def embed_texts_in_subprocess(
    texts: List[str],
    embedding_model_path: str,
    n_ctx_embed: int,
    n_gpu_layers_embed: int,
    pooling_type_embed: int
) -> List[List[float]]:
    """
    在子进程中执行批量文本嵌入。
    """
    worker_logger.info(f"WORKER: embed_texts_in_subprocess received {len(texts)} texts.")
    model = _get_embedding_model_instance_in_worker(
        embedding_model_path, n_ctx_embed, n_gpu_layers_embed, pooling_type_embed
    )
    if not model:
        worker_logger.error("WORKER: Failed to get model instance in subprocess. Returning empty embeddings.")
        return [[] for _ in texts]
    
    dimension = _process_local_model_dimension_cache.get(embedding_model_path)
    if not dimension: # 应该不会发生，因为 _get_embedding_model_instance_in_worker 会设置它
        worker_logger.error("WORKER: Model dimension not found in cache after model load. Critical error.")
        return [[] for _ in texts]

    default_zero_vector = [0.0] * dimension
    
    # 与 LocalModelHandler._blocking_embed_documents_internal 类似的处理逻辑
    valid_texts_with_indices: List[tuple[int, str]] = []
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_texts_with_indices.append((i, text))
        else:
            worker_logger.warning(f"WORKER: Input text at original index {i} is empty or invalid. Will use zero vector. Text: '{text}'")

    if not valid_texts_with_indices:
        return [list(default_zero_vector) for _ in texts]

    valid_texts_to_embed = [text for _, text in valid_texts_with_indices]
    raw_embeddings_for_valid_texts: List[List[float]] = []

    try:
        response = model.create_embedding(input=valid_texts_to_embed)
        # ... (此处省略与 LocalModelHandler._blocking_embed_documents_internal 中类似的详细的响应解析和错误处理逻辑)
        # 为了简洁，我们先做一个简化版的解析，假设一切顺利
        # 在实际应用中，需要复制 LocalModelHandler 中对 response 的完整健壮性检查

        if response and "data" in response and isinstance(response["data"], list):
            embeddings_data = response["data"]
            if len(embeddings_data) == len(valid_texts_to_embed):
                for item_idx, item in enumerate(embeddings_data):
                    if isinstance(item, dict) and "embedding" in item and \
                       isinstance(item["embedding"], list) and len(item["embedding"]) == dimension:
                        raw_embeddings_for_valid_texts.append([float(x) for x in item["embedding"]])
                    else:
                        worker_logger.warning(f"WORKER: Valid text at valid_idx {item_idx} got invalid embedding. Using zero vector.")
                        raw_embeddings_for_valid_texts.append(list(default_zero_vector))
            else:
                worker_logger.error(f"WORKER: Mismatch in num embeddings received. Using zero vectors.")
                raw_embeddings_for_valid_texts = [list(default_zero_vector) for _ in valid_texts_to_embed]
        else:
            worker_logger.error(f"WORKER: Invalid response from create_embedding. Using zero vectors.")
            raw_embeddings_for_valid_texts = [list(default_zero_vector) for _ in valid_texts_to_embed]
            
    except Exception as e_batch_embed:
        worker_logger.error(f"WORKER: Error during batch embedding: {e_batch_embed}", exc_info=True)
        raw_embeddings_for_valid_texts = [list(default_zero_vector) for _ in valid_texts_to_embed]

    final_embeddings_ordered: List[List[float]] = [list(default_zero_vector) for _ in texts]
    valid_embedding_idx = 0
    for original_idx, _ in valid_texts_with_indices:
        if valid_embedding_idx < len(raw_embeddings_for_valid_texts):
            final_embeddings_ordered[original_idx] = raw_embeddings_for_valid_texts[valid_embedding_idx]
            valid_embedding_idx += 1
        else:
            final_embeddings_ordered[original_idx] = list(default_zero_vector)
            
    normalized_embeddings = l2_normalize_embeddings_worker(final_embeddings_ordered)
    worker_logger.info(f"WORKER: Successfully processed and normalized {len(normalized_embeddings)} document embeddings in subprocess.")
    return normalized_embeddings


def embed_query_in_subprocess(
    text: str,
    embedding_model_path: str,
    n_ctx_embed: int,
    n_gpu_layers_embed: int,
    pooling_type_embed: int
) -> List[float]:
    """
    在子进程中执行单个查询文本嵌入。
    """
    worker_logger.info(f"WORKER: embed_query_in_subprocess received query (first 100): '{text[:100]}'")
    model = _get_embedding_model_instance_in_worker(
        embedding_model_path, n_ctx_embed, n_gpu_layers_embed, pooling_type_embed
    )
    if not model:
        worker_logger.error("WORKER: Failed to get model instance for query. Returning empty embedding.")
        return []
        
    dimension = _process_local_model_dimension_cache.get(embedding_model_path)
    if not dimension:
        worker_logger.error("WORKER: Model dimension not found in cache for query. Critical error.")
        return []
    
    default_zero_vector = [0.0] * dimension

    if not text or not text.strip():
        worker_logger.warning("WORKER: Received empty or invalid text for query embedding. Returning zero vector.")
        return list(default_zero_vector)

    try:
        # 使用 create_embedding 来保持与批量接口的一致性，即使是单个查询
        # 因为我们观察到 Llama.embed() 可能不稳定
        response = model.create_embedding(input=[text])
        if response and "data" in response and isinstance(response["data"], list) and len(response["data"]) == 1:
            item = response["data"][0]
            if isinstance(item, dict) and "embedding" in item and \
               isinstance(item["embedding"], list) and len(item["embedding"]) == dimension:
                embedding_vector = [float(x) for x in item["embedding"]]
                normalized_list_of_list = l2_normalize_embeddings_worker([embedding_vector])
                final_embedding = normalized_list_of_list[0] if normalized_list_of_list and normalized_list_of_list[0] else list(default_zero_vector)
                worker_logger.info(f"WORKER: Successfully processed query embedding in subprocess. Dimension: {len(final_embedding)}")
                return final_embedding
            else:
                worker_logger.warning(f"WORKER: Query embedding response invalid format/dim. Using zero vector.")
                return list(default_zero_vector)
        else:
            worker_logger.error(f"WORKER: Invalid or empty response from create_embedding for query. Using zero vector.")
            return list(default_zero_vector)
    except Exception as e_query_embed:
        worker_logger.error(f"WORKER: Error during query embedding: {e_query_embed}", exc_info=True)
        return list(default_zero_vector)