from typing import List, Dict, Any, Optional, Tuple
import jieba
import bm25s # 我们确认使用 bm25s
import pickle
import os
import logging
import numpy as np

# 配置日志记录器
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class FileBM25Retriever:
    def __init__(
        self,
        index_directory_path: str = "/home/zhz/dagster_home/bm25_index_data/", # 与Dagster中配置一致
        # user_dict_path: Optional[str] = None 
    ):
        self.index_directory_path = index_directory_path
        # self.user_dict_path = user_dict_path

        self._bm25_model: Optional[bm25s.BM25] = None
        self._doc_ids: Optional[List[str]] = None 
        # self._tokenizer: Optional[bm25s.Tokenizer] = None # bm25s.tokenize 是一个函数，或者可以用Tokenizer类

        self._initialize_retriever()

    def _initialize_retriever(self):
        # if self.user_dict_path and os.path.exists(self.user_dict_path):
        #     # ... (加载jieba用户词典的逻辑不变) ...
        
        if not os.path.isdir(self.index_directory_path): # <--- 修改：检查目录是否存在
            logger.error(f"BM25 index directory not found at: {self.index_directory_path}")
            raise FileNotFoundError(f"BM25 index directory not found: {self.index_directory_path}")

        try:
            logger.info(f"Loading BM25 model from directory: {self.index_directory_path}")
            # 使用bm25s的load类方法
            # load_corpus=False 因为我们不期望在模型文件中包含原始语料库文本
            # mmap=False (默认) 先不使用内存映射，除非索引非常大
            self._bm25_model = bm25s.BM25.load(
                self.index_directory_path,
                load_corpus=False, # 通常我们不需要在这里加载原始语料库
                # mmap=False 
            )
            
            if self._bm25_model is None: # load 失败通常会抛异常，但以防万一
                logger.error("Failed to load BM25 model (returned None).")
                raise ValueError("Failed to load BM25 model.")
            logger.info("BM25 model loaded successfully.")

            # 单独加载 document_ids 列表
            doc_ids_path = os.path.join(self.index_directory_path, "doc_ids.pkl")
            if not os.path.exists(doc_ids_path):
                logger.error(f"doc_ids.pkl not found in {self.index_directory_path}")
                raise FileNotFoundError(f"doc_ids.pkl not found in {self.index_directory_path}")
            
            with open(doc_ids_path, 'rb') as f_in:
                self._doc_ids = pickle.load(f_in)
            
            if self._doc_ids is None: # pickle 加载空文件可能返回None
                 logger.warning(f"doc_ids.pkl loaded, but it was empty or invalid.")
                 self._doc_ids = [] # 设为空列表以避免后续错误
            logger.info(f"Document IDs loaded successfully. Number of indexed documents: {len(self._doc_ids)}")

        except Exception as e:
            logger.error(f"Failed to load BM25 index or document IDs: {e}")
            raise
            
    def retrieve(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        # ... (初始化检查和空索引检查不变) ...
        if self._bm25_model is None or self._doc_ids is None:
            logger.error("BM25 Retriever is not properly initialized.")
            # ... (尝试重新初始化或返回空的逻辑不变) ...
            return []
        
        if not self._doc_ids: 
            logger.info("BM25 index is empty, no results to retrieve.")
            return []

        logger.info(f"Retrieving documents with BM25 for query: '{query_text[:100]}...' with n_results={n_results}")

        try:
            # 1. 将查询文本分词
            # bm25s 有自己的 tokenize 函数，它会处理停用词、词干提取（如果配置了）并返回模型期望的格式
            # 我们需要确保查询时的分词方式与索引时一致。
            # 如果索引时用了jieba，查询时也应该用jieba。
            # bm25s.BM25 对象在加载后，其内部应该已经有了词汇表 (vocab_dict)，
            # 它的 get_scores 方法期望的是与词汇表ID对应的输入，或者它能自己处理分词后的文本列表。
            
            query_tokenized_jieba = list(jieba.cut_for_search(query_text))
            logger.debug(f"Tokenized query (jieba): {query_tokenized_jieba}")

            # 2. 使用BM25模型进行查询
            # bm25s 的 retrieve 方法可以直接返回文档索引和分数
            # 它接受分词后的查询 (list of str)
            # results 是文档索引 (numpy array), scores 是对应的分数 (numpy array)
            # 它们都是 (n_queries, k) 的形状，我们只有一个查询，所以是 (1, k)
            
            # 确保 k 不超过实际文档数
            actual_k = min(n_results, len(self._doc_ids))
            if actual_k == 0: # 如果索引中没有文档
                return []

            results_indices, results_scores = self._bm25_model.get_top_n(
                query_tokenized_jieba, 
                corpus=None, # 我们不需要在这里提供原始语料库，它返回的是索引
                n=actual_k
            )
            # get_top_n 返回的是一个列表（每个查询一个结果列表），我们只有一个查询
            # 每个结果列表中的元素是 (doc_index, score) 吗？还是直接是doc_index?
            # 查阅 bm25s 文档：get_top_n(query_tokens, documents, n=5)
            #   - query_tokens: list of tokens for the query.
            #   - documents: list of documents (list of tokens).
            #   - n: number of top documents to retrieve.
            # Returns: list of top n documents.
            # 这看起来是返回文档本身，不是我们想要的。

            # 让我们回到使用 get_scores 然后自己排序的方式，这更可控
            all_scores = self._bm25_model.get_scores(query_tokenized_jieba)
            
            top_n_indices = np.argsort(all_scores)[-actual_k:][::-1] # 降序取前N

            retrieved_docs = []
            for doc_corpus_index in top_n_indices:
                doc_id = self._doc_ids[doc_corpus_index] # 从0-based索引映射到我们的chunk_id
                score = float(all_scores[doc_corpus_index]) 
                retrieved_docs.append({
                    "id": doc_id,
                    "score": score
                })
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents using BM25.")
            return retrieved_docs

        except Exception as e:
            logger.error(f"Error during BM25 retrieval: {e}")
            return []
            
    def retrieve(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        根据查询文本使用BM25检索相关的文档块ID和分数。

        Args:
            query_text (str): 用户查询的文本。
            n_results (int): 希望返回的最大结果数量。

        Returns:
            List[Dict[str, Any]]: 检索到的文档块列表。每个字典包含：
                                   'id' (chunk_id), 
                                   'score' (BM25分数)
        """
        if self._bm25_model is None or self._doc_ids is None:
            logger.error("BM25 Retriever is not properly initialized.")
            try:
                self._initialize_retriever()
                if self._bm25_model is None or self._doc_ids is None:
                    return []
            except Exception as e:
                logger.error(f"Failed to re-initialize BM25 retriever during retrieve call: {e}")
                return []
        
        if not self._doc_ids: # 如果索引为空
            logger.info("BM25 index is empty, no results to retrieve.")
            return []

        logger.info(f"Retrieving documents with BM25 for query: '{query_text[:100]}...' with n_results={n_results}")

        try:
            # 1. 将查询文本分词
            query_tokenized = list(jieba.cut_for_search(query_text))
            logger.debug(f"Tokenized query: {query_tokenized}")

            # 2. 使用BM25模型进行查询
            # bm25s 的 get_scores 方法返回所有文档的分数
            # bm25s 的 get_batch_results (或类似名称) 可能更适合获取top-N
            # 我们需要查阅 bm25s 的API来获取top-N的文档索引和分数
            # 假设它有一个类似 get_top_n 的方法，或者我们需要自己处理 get_scores 的结果

            # 查阅 bm25s 文档，它通常使用 `bm25_model.get_scores(query_tokenized)` 得到所有分数
            # 然后我们需要自己排序并取top N
            # 或者，bm25s.BM25 可能有更直接的方法，例如 `search` 或 `query`
            # 经过快速查阅，bm25s 似乎没有直接的 top_n 方法，但其设计是为了快速计算所有分数。
            # `bm25_model.get_scores(query_tokenized)` 返回一个numpy数组，包含每个文档的分数。

            all_scores = self._bm25_model.get_scores(query_tokenized)
            
            # 获取分数最高的n_results个文档的索引
            # 注意：如果实际文档数少于n_results，则取实际数量
            actual_n_results = min(n_results, len(self._doc_ids))
            
            # 使用numpy的argsort来获取排序后的索引，然后取最后N个（因为argsort默认升序）
            # 或者取负数再取前N个
            top_n_indices = np.argsort(all_scores)[-actual_n_results:][::-1] # 降序取前N

            retrieved_docs = []
            for index in top_n_indices:
                doc_id = self._doc_ids[index]
                score = float(all_scores[index]) # 转换为Python float
                # 我们只返回ID和分数，文本内容由上层逻辑获取
                retrieved_docs.append({
                    "id": doc_id,
                    "score": score
                })
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents using BM25.")
            return retrieved_docs

        except Exception as e:
            logger.error(f"Error during BM25 retrieval: {e}")
            return []

if __name__ == '__main__':
    logger.info("--- FileBM25Retriever Test ---")
    
    # 确保您的BM25索引文件已通过Dagster流水线创建
    # (例如 /home/zhz/dagster_home/bm25_index/rag_bm25_index.pkl)
    try:
        retriever = FileBM25Retriever()
        
        # 测试查询 (与ChromaDBRetriever使用相同的查询，以便后续比较和融合)
        test_query = "人工智能的应用有哪些？" 
        
        retrieved_results = retriever.retrieve(test_query, n_results=3)
        
        if retrieved_results:
            print(f"\n--- BM25 Retrieved Results for query: '{test_query}' ---")
            for i, doc in enumerate(retrieved_results):
                print(f"Result {i+1}:")
                print(f"  ID: {doc['id']}")
                print(f"  Score: {doc['score']:.4f}")
                # 我们这里不获取文本，由rag_service负责
                print("-" * 20)
        else:
            print(f"\nNo results retrieved with BM25 for query: '{test_query}'")

    except Exception as e:
        print(f"An error occurred during the BM25 test: {e}")