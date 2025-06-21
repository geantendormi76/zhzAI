# /home/zhz/zhz_agent/zhz_rag/core_rag/retrievers/file_bm25_retriever.py

from typing import List, Dict, Any, Optional
import jieba
import bm25s
import pickle
import os
import logging
import sys
import numpy as np
from cachetools import TTLCache
import asyncio

# --- 日志配置 (标准化) ---
bm25_logger = logging.getLogger("BM25Retriever")
bm25_logger.setLevel(os.getenv("BM25_LOG_LEVEL", "INFO").upper())
bm25_logger.propagate = False

if not bm25_logger.hasHandlers():
    _handler = logging.StreamHandler(sys.stdout)
    _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _handler.setFormatter(_formatter)
    bm25_logger.addHandler(_handler)


class FileBM25Retriever:
    """
    一个从文件中加载BM25索引并执行关键词检索的检索器。
    它依赖于由Dagster流水线生成的索引文件和doc_ids文件。
    V2: 优化为异步接口并增加线程安全缓存。
    """
    def __init__(self, index_directory: str):
        """
        初始化 FileBM25Retriever。

        Args:
            index_directory (str): 存储 BM25 索引文件和 doc_ids.pkl 的目录路径。
        """
        self.index_directory_path = index_directory
        self._bm25_model: Optional[bm25s.BM25] = None
        self._doc_ids: Optional[List[str]] = None
        # 使用TTLCache缓存检索结果，缓存100个查询，每个存活300秒（5分钟）
        self._retrieval_cache: TTLCache = TTLCache(maxsize=100, ttl=300)
        self._retrieval_cache_lock = asyncio.Lock() # 异步锁保护缓存读写

        self._initialize_retriever()

    def _initialize_retriever(self):
        """
        同步加载 BM25 模型和文档ID。这个方法在服务启动时调用。
        """
        if not os.path.isdir(self.index_directory_path):
            bm25_logger.error(f"BM25 index directory not found at: {self.index_directory_path}")
            raise FileNotFoundError(f"BM25 index directory not found: {self.index_directory_path}")

        try:
            bm25_logger.info(f"Loading BM25 model from directory: {self.index_directory_path}")
            # 从磁盘加载模型，不加载语料库本身以节省内存
            self._bm25_model = bm25s.BM25.load(
                self.index_directory_path,
                load_corpus=False,
            )
            
            if self._bm25_model is None:
                bm25_logger.error("Failed to load BM25 model (bm25s.BM25.load returned None).")
                raise ValueError("Failed to load BM25 model.")
            
            bm25_logger.info("BM25 model loaded successfully.")

            doc_ids_path = os.path.join(self.index_directory_path, "doc_ids.pkl")
            if not os.path.exists(doc_ids_path):
                bm25_logger.error(f"doc_ids.pkl not found in {self.index_directory_path}")
                raise FileNotFoundError(f"doc_ids.pkl not found in {doc_ids_path}")
            
            with open(doc_ids_path, 'rb') as f_in:
                self._doc_ids = pickle.load(f_in)
            
            if self._doc_ids is None:
                bm25_logger.warning(f"doc_ids.pkl loaded, but it was empty or invalid.")
                self._doc_ids = []
            
            bm25_logger.info(f"Document IDs for BM25 loaded successfully. Total indexed documents: {len(self._doc_ids)}")

        except Exception as e:
            bm25_logger.error(f"Failed to load BM25 index or document IDs: {e}", exc_info=True)
            raise

    async def retrieve(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        根据查询文本使用BM25检索相关的文档块ID和分数。
        这是一个异步包装的同步操作，并带有缓存。
        """
        if self._bm25_model is None or self._doc_ids is None:
            bm25_logger.error("BM25 Retriever is not properly initialized. Cannot retrieve.")
            return []

        cache_key = f"{query_text}_{n_results}"
        async with self._retrieval_cache_lock:
            cached_result = self._retrieval_cache.get(cache_key)
        
        if cached_result is not None:
            bm25_logger.info(f"BM25 CACHE HIT for key: '{cache_key[:100]}...'")
            return cached_result

        bm25_logger.info(f"BM25 CACHE MISS for key: '{cache_key[:100]}...'. Performing retrieval.")
        
        if not self._doc_ids:
            bm25_logger.warning("BM25 index is empty, no results to retrieve.")
            return []

        def _blocking_retrieve():
            try:
                query_tokenized = list(jieba.cut_for_search(query_text))
                bm25_logger.debug(f"Tokenized query for BM25: {query_tokenized}")

                all_scores = self._bm25_model.get_scores(query_tokenized)
                
                # 确保请求的结果数不超过索引中的文档总数
                actual_n_results = min(n_results, len(self._doc_ids))
                if actual_n_results <= 0:
                    return []
                
                # 获取分数最高的n个结果的索引
                top_n_indices = np.argsort(all_scores)[-actual_n_results:][::-1]

                retrieved_docs = []
                for index in top_n_indices:
                    # 再次检查索引有效性
                    if 0 <= index < len(self._doc_ids):
                        doc_id = self._doc_ids[index]
                        score = float(all_scores[index])
                        # 仅返回ID和分数，内容将在后续步骤中从docstore获取
                        retrieved_docs.append({
                            "id": doc_id,
                            "score": score,
                            "source_type": "keyword_bm25s"
                        })
                    else:
                        bm25_logger.warning(f"BM25 retrieval: Index {index} is out of bounds for doc_ids list (len: {len(self._doc_ids)}). Skipping.")
                
                return retrieved_docs

            except Exception as e_inner:
                bm25_logger.error(f"Error during blocking BM25 retrieval: {e_inner}", exc_info=True)
                return []

        retrieved_results = await asyncio.to_thread(_blocking_retrieve)
        
        async with self._retrieval_cache_lock:
            self._retrieval_cache[cache_key] = retrieved_results
        
        bm25_logger.info(f"Retrieved and cached {len(retrieved_results)} documents using BM25 for query: '{query_text[:50]}...'")
        return retrieved_results