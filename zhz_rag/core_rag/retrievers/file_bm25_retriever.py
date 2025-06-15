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
# 移除模块级的 basicConfig，应在应用入口配置
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class FileBM25Retriever: # <--- 定义 FileBM25Retriever 类
    def __init__(
        self,
        # --- 修改：参数名与 rag_api_service.py 中实例化时一致 ---
        index_directory: str, # 原为 index_directory_path
        # user_dict_path: Optional[str] = None # 如果需要用户词典，可以取消注释
    ):
        # --- 修改：使用传入的参数 ---
        self.index_directory_path = index_directory # 保存为实例变量
        # self.user_dict_path = user_dict_path

        self._bm25_model: Optional[bm25s.BM25] = None
        self._doc_ids: Optional[List[str]] = None 
        
        self._initialize_retriever()

    def _initialize_retriever(self):
        # if self.user_dict_path and os.path.exists(self.user_dict_path):
        #     jieba.load_userdict(self.user_dict_path)
        #     logger.info(f"Loaded jieba user dictionary from: {self.user_dict_path}")
        
        if not os.path.isdir(self.index_directory_path):
            logger.error(f"BM25 index directory not found at: {self.index_directory_path}")
            # 最好抛出异常，而不是让模型为空，这样在服务启动时就能发现问题
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
            logger.error(f"Failed to load BM25 index or document IDs: {e}", exc_info=True) # 添加 exc_info
            # 在初始化失败时也应该抛出异常，以便上层知道
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
            # 考虑是否应该尝试重新初始化，或者直接抛出异常/返回空
            # 如果初始化在 __init__ 中失败并抛出异常，这里就不太可能执行到
            return [] 
        
        if not self._doc_ids: 
            logger.info("BM25 index is empty, no results to retrieve.")
            return []

        logger.info(f"Retrieving documents with BM25 for query: '{query_text[:100]}...' with n_results={n_results}")

        try:
            query_tokenized = list(jieba.cut_for_search(query_text)) # 使用 jieba 分词
            logger.debug(f"Tokenized query (jieba for BM25): {query_tokenized}")

            all_scores = self._bm25_model.get_scores(query_tokenized)
            
            actual_n_results = min(n_results, len(self._doc_ids))
            if actual_n_results <= 0: # 如果请求0个结果或索引为空
                return []
            
            top_n_indices = np.argsort(all_scores)[-actual_n_results:][::-1]

            retrieved_docs = []
            for index in top_n_indices:
                # 确保 index 在 self._doc_ids 的有效范围内
                if 0 <= index < len(self._doc_ids):
                    doc_id = self._doc_ids[index]
                    score = float(all_scores[index])
                    retrieved_docs.append({
                        "id": doc_id,
                        "score": score,
                        "source_type": "keyword_bm25" # 添加 source_type
                    })
                else:
                    logger.warning(f"BM25 retrieval: Index {index} out of bounds for doc_ids list (len: {len(self._doc_ids)}). Skipping.")

            
            logger.info(f"Retrieved {len(retrieved_docs)} documents using BM25.")
            return retrieved_docs

        except Exception as e:
            logger.error(f"Error during BM25 retrieval: {e}", exc_info=True) # 添加 exc_info
            return []

# --- 将原来的 if __name__ == '__main__': 测试代码移到类定义之外 ---
async def main_bm25_test(): # 改为异步以匹配其他测试，或保持同步
    logger.info("--- FileBM25Retriever Test ---")
    
    try:
        # 在API服务启动时，index_directory_path 是从 rag_api_service.py 传递的
        # 这里为了独立测试，直接使用环境变量或硬编码路径
        bm25_index_dir = os.getenv("BM25_INDEX_DIRECTORY", "/home/zhz/zhz_agent/zhz_rag/stored_data/bm25_index")
        if not os.path.isdir(bm25_index_dir): # 确保测试前路径存在
             # 尝试使用Dagster中的路径作为备选 (如果环境变量BM25_INDEX_DIRECTORY未设置)
             bm25_index_dir_dagster = "/home/zhz/dagster_home/bm25_index_data/"
             if os.path.isdir(bm25_index_dir_dagster):
                 bm25_index_dir = bm25_index_dir_dagster
             else:
                 logger.error(f"BM25 index directory for test not found at {bm25_index_dir} or {bm25_index_dir_dagster}")
                 return

        logger.info(f"Using BM25 index directory for test: {bm25_index_dir}")
        retriever = FileBM25Retriever(index_directory=bm25_index_dir) # 传递参数
        
        test_query = "人工智能的应用有哪些？" 
        retrieved_results = retriever.retrieve(test_query, n_results=3) # retrieve 是同步的
        
        if retrieved_results:
            print(f"\n--- BM25 Retrieved Results for query: '{test_query}' ---")
            for i, doc in enumerate(retrieved_results):
                print(f"Result {i+1}:")
                print(f"  ID: {doc['id']}")
                print(f"  Score: {doc['score']:.4f}")
                print(f"  Source Type: {doc.get('source_type', 'N/A')}")
                print("-" * 20)
        else:
            print(f"\nNo results retrieved with BM25 for query: '{test_query}'")

    except Exception as e:
        print(f"An error occurred during the BM25 test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # 如果 main_bm25_test 是异步的，用 asyncio.run(main_bm25_test())
    # 如果是同步的，直接调用 main_bm25_test()
    # 当前 retrieve 是同步的，所以 main_bm25_test 也可以是同步的
    # 为了与其他测试代码的风格保持一致，我们也可以让它看起来像异步启动，但实际是同步执行
    # 或者直接让 main_bm25_test 是同步的：
    def run_sync_test():
        asyncio.run(main_bm25_test()) # 如果内部有异步操作，需要这样
                                     # 但当前 main_bm25_test 内部没有异步操作

    # 更简单的方式，如果 main_bm25_test 本身不需要是 async：
    # main_bm25_test() 
    # 但为了保持可以 await 的可能性，暂时保留 async def main_bm25_test
    # 并假设如果它内部有异步操作，会用 asyncio.run()
    # 实际上，由于 retrieve 是同步的，main_bm25_test 不需要是 async
    # 我们将其改为同步，以简化 __main__ 测试：

    def main_sync_test_runner():
        logger.info("--- FileBM25Retriever Sync Test Runner ---")
        try:
            bm25_index_dir = os.getenv("BM25_INDEX_DIRECTORY", "/home/zhz/zhz_agent/zhz_rag/stored_data/bm25_index")
            # ... (与上面 main_bm25_test 内部类似的路径检查和实例化) ...
            if not os.path.isdir(bm25_index_dir):
                 bm25_index_dir_dagster = "/home/zhz/dagster_home/bm25_index_data/"
                 if os.path.isdir(bm25_index_dir_dagster):
                     bm25_index_dir = bm25_index_dir_dagster
                 else:
                     logger.error(f"BM25 index directory for test not found at {bm25_index_dir} or {bm25_index_dir_dagster}")
                     return
            
            retriever = FileBM25Retriever(index_directory=bm25_index_dir)
            test_query = "人工智能的应用有哪些？" 
            retrieved_results = retriever.retrieve(test_query, n_results=3)
            if retrieved_results:
                print(f"\n--- BM25 Retrieved Results for query: '{test_query}' ---")
                for i, doc in enumerate(retrieved_results):
                    print(f"Result {i+1}: ID: {doc['id']}, Score: {doc['score']:.4f}, Source: {doc.get('source_type')}")
            else:
                print(f"\nNo results for query: '{test_query}'")
        except Exception as e:
            print(f"Error in BM25 test: {e}")
            traceback.print_exc()

    main_sync_test_runner()