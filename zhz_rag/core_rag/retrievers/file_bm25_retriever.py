# 文件: zhz_rag/core_rag/retrievers/file_bm25_retriever.py

from typing import List, Dict, Any, Optional
import jieba
import bm25s
import pickle
import os
import logging
import numpy as np
import traceback
import asyncio
from cachetools import TTLCache # <--- 添加这一行

# 配置日志记录器
logger = logging.getLogger(__name__)

class FileBM25Retriever:
    def __init__(
        self,
        index_directory: str,
    ):
        """
        初始化 FileBM25Retriever。

        Args:
            index_directory (str): 存储 BM25 索引文件的目录路径。
        """
        self.index_directory_path = index_directory

        self._bm25_model: Optional[bm25s.BM25] = None
        self._doc_ids: Optional[List[str]] = None
        # --- 使用 TTLCache 初始化缓存 ---
        self._retrieval_cache: TTLCache = TTLCache(maxsize=100, ttl=300)
        self._initialize_retriever()

    def _initialize_retriever(self):
        """
        加载 BM25 模型和文档ID。
        """
        if not os.path.isdir(self.index_directory_path):
            logger.error(f"BM25 index directory not found at: {self.index_directory_path}")
            raise FileNotFoundError(f"BM25 index directory not found: {self.index_directory_path}")

        try:
            logger.info(f"Loading BM25 model from directory: {self.index_directory_path}")
            self._bm25_model = bm25s.BM25.load(
                self.index_directory_path,
                load_corpus=False,
            )
            
            if self._bm25_model is None:
                logger.error("Failed to load BM25 model (returned None).")
                raise ValueError("Failed to load BM25 model.")
            logger.info("BM25 model loaded successfully.")

            doc_ids_path = os.path.join(self.index_directory_path, "doc_ids.pkl")
            if not os.path.exists(doc_ids_path):
                logger.error(f"doc_ids.pkl not found in {self.index_directory_path}")
                raise FileNotFoundError(f"doc_ids.pkl not found in {self.index_directory_path}")
            
            with open(doc_ids_path, 'rb') as f_in:
                self._doc_ids = pickle.load(f_in)
            
            if self._doc_ids is None:
                logger.warning(f"doc_ids.pkl loaded, but it was empty or invalid.")
                self._doc_ids = []
            logger.info(f"Document IDs loaded successfully. Number of indexed documents: {len(self._doc_ids)}")

        except Exception as e:
            logger.error(f"Failed to load BM25 index or document IDs: {e}", exc_info=True)
            raise

    def retrieve(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        根据查询文本使用BM25检索相关的文档块ID和分数。

        Args:
            query_text (str): 用户查询的文本。
            n_results (int): 希望返回的最大结果数量。

        Returns:
            List[Dict[str, Any]]: 检索到的文档块列表。每个字典包含：
                                'id' (chunk_id),
                                'score' (BM25分数),
                                'source_type' (固定为 "keyword_bm25")
        """
        if self._bm25_model is None or self._doc_ids is None:
            logger.error("BM25 Retriever is not properly initialized.")
            return []

        # --- 更新: 使用 TTLCache 进行缓存检查 ---
        cache_key = f"{query_text}_{n_results}"
        # BM25是同步的，所以我们不需要异步锁，可以直接访问缓存
        cached_result = self._retrieval_cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"BM25 CACHE HIT for key: '{cache_key[:100]}...'")
            return cached_result
        logger.info(f"BM25 CACHE MISS for key: '{cache_key[:100]}...'. Performing retrieval.")
        # --- 缓存检查结束 ---

        if not self._doc_ids:
            logger.info("BM25 index is empty, no results to retrieve.")
            return []

        logger.info(f"Retrieving documents with BM25 for query: '{query_text[:100]}...' with n_results={n_results}")

        try:
            query_tokenized = list(jieba.cut_for_search(query_text))
            logger.debug(f"Tokenized query (jieba for BM25): {query_tokenized}")

            # 调用 bm25s 模型获取分数
            all_scores = self._bm25_model.get_scores(query_tokenized)
            
            actual_n_results = min(n_results, len(self._doc_ids))
            if actual_n_results <= 0:
                return []
            
            # 获取分数最高的n个结果的索引
            top_n_indices = np.argsort(all_scores)[-actual_n_results:][::-1]

            retrieved_docs = []
            for index in top_n_indices:
                if 0 <= index < len(self._doc_ids):
                    doc_id = self._doc_ids[index]
                    score = float(all_scores[index])
                    retrieved_docs.append({
                        "id": doc_id,
                        "score": score,
                        "source_type": "keyword_bm25"
                    })
                else:
                    logger.warning(f"BM25 retrieval: Index {index} out of bounds for doc_ids list (len: {len(self._doc_ids)}). Skipping.")

            # --- 更新: 存储到 TTLCache ---
            self._retrieval_cache[cache_key] = retrieved_docs
            logger.info(f"BM25 CACHED {len(retrieved_docs)} results for key: '{cache_key[:100]}...'")
            # --- 缓存存储结束 ---
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents using BM25.")
            return retrieved_docs

        except Exception as e:
            logger.error(f"Error during BM25 retrieval: {e}", exc_info=True)
            return []
        
