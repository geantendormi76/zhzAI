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
        # 用于缓存检索结果的字典
        self._retrieval_cache: Dict[str, List[Dict[str, Any]]] = {}

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

        # --- 添加：缓存检查 ---
        cache_key = f"{query_text}_{n_results}"
        if cache_key in self._retrieval_cache:
            logger.info(f"BM25 CACHE HIT for key: '{cache_key[:100]}...'")
            return self._retrieval_cache[cache_key]
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

            # --- 添加：存储到缓存 ---
            self._retrieval_cache[cache_key] = retrieved_docs
            logger.info(f"BM25 CACHED {len(retrieved_docs)} results for key: '{cache_key[:100]}...'")
            # --- 缓存存储结束 ---
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents using BM25.")
            return retrieved_docs

        except Exception as e:
            logger.error(f"Error during BM25 retrieval: {e}", exc_info=True)
            return []

def main_sync_test_runner():
    """
    用于独立测试 FileBM25Retriever 的主函数。
    """
    logger.info("--- FileBM25Retriever Sync Test Runner ---")
    try:
        # 尝试从环境变量或默认路径获取索引目录
        bm25_index_dir = os.getenv("BM25_INDEX_DIRECTORY", "/home/zhz/zhz_agent/zhz_rag/stored_data/bm25_index")
        
        if not os.path.isdir(bm25_index_dir):
            # 如果默认路径不存在，尝试备用路径
            bm25_index_dir_dagster = "/home/zhz/dagster_home/bm25_index_data/"
            if os.path.isdir(bm25_index_dir_dagster):
                bm25_index_dir = bm25_index_dir_dagster
            else:
                logger.error(f"BM25 index directory for test not found at {bm25_index_dir} or {bm25_index_dir_dagster}")
                return
        
        logger.info(f"Using BM25 index directory for test: {bm25_index_dir}")
        retriever = FileBM25Retriever(index_directory=bm25_index_dir)
        
        test_query = "人工智能的应用有哪些？"
        
        # 第一次查询，应该未命中缓存
        print("\n--- First retrieval (should be CACHE MISS) ---")
        retrieved_results = retriever.retrieve(test_query, n_results=3)
        if retrieved_results:
            print(f"--- BM25 Retrieved Results for query: '{test_query}' ---")
            for i, doc in enumerate(retrieved_results):
                print(f"Result {i+1}: ID: {doc['id']}, Score: {doc['score']:.4f}, Source: {doc.get('source_type')}")
        else:
            print(f"\nNo results for query: '{test_query}'")
        
        # 第二次查询，应该命中缓存
        print("\n--- Second retrieval (should be CACHE HIT) ---")
        retrieved_results_cached = retriever.retrieve(test_query, n_results=3)
        if retrieved_results_cached:
            print(f"--- Cached BM25 Retrieved Results for query: '{test_query}' ---")
            for i, doc in enumerate(retrieved_results_cached):
                print(f"Result {i+1}: ID: {doc['id']}, Score: {doc['score']:.4f}, Source: {doc.get('source_type')}")
        else:
            print(f"\nNo results for cached query: '{test_query}'")


    except Exception as e:
        print(f"An error occurred during the BM25 test: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    # 配置顶层日志记录器，以便在独立运行时能看到日志输出
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main_sync_test_runner()
