# 文件: zhz_rag/core_rag/retrievers/chromadb_retriever.py

import asyncio
import json
from typing import List, Dict, Any, Optional
import chromadb
import logging
from .embedding_functions import LlamaCppEmbeddingFunction
from cachetools import TTLCache # <--- 添加这一行

# 配置ChromaDBRetriever的日志记录器
logger = logging.getLogger(__name__)
# 注意：在模块级别配置basicConfig可能会影响整个应用的日志行为。
# 通常建议在应用入口处统一配置。
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ChromaDBRetriever:
    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embedding_function: LlamaCppEmbeddingFunction
    ):
        """
        初始化ChromaDBRetriever。

        Args:
            collection_name (str): 要查询的ChromaDB集合名称。
            persist_directory (str): ChromaDB数据持久化的目录。
            embedding_function (LlamaCppEmbeddingFunction): 用于生成嵌入的函数实例。
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._embedding_function = embedding_function
        self._client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.Collection] = None
        self._dimension: Optional[int] = None

    # --- 新增: 初始化TTLCache召回结果缓存和异步锁 ---
        # maxsize=100: 最多缓存100个查询结果
        # ttl=300: 缓存条目存活300秒（5分钟）
        self._retrieval_cache: TTLCache = TTLCache(maxsize=100, ttl=300)
        self._retrieval_cache_lock = asyncio.Lock()

        # 初始化依然是同步的，在服务启动时执行
        self._initialize_retriever()

    def _initialize_retriever(self):
        """
        初始化ChromaDB客户端和集合。
        """
        try:
            logger.info(f"Initializing ChromaDB client from path: {self.persist_directory}")
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            
            logger.info(f"Getting or creating ChromaDB collection: {self.collection_name} using provided async embedding function.")
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_function 
            )

            if self._collection.count() == 0:
                logger.warning(f"ChromaDB collection '{self.collection_name}' is empty!")
            else:
                logger.info(f"ChromaDB collection '{self.collection_name}' loaded. Item count: {self._collection.count()}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client or collection: {e}", exc_info=True)
            raise

    async def retrieve(
        self, 
        query_text: str, 
        n_results: int = 5, 
        include_fields: Optional[List[str]] = None,
        where_filter: Optional[Dict[str, Any]] = None  # <--- 新增参数
    ) -> List[Dict[str, Any]]:
        if self._collection is None or self._embedding_function is None:
            logger.error("Retriever is not properly initialized.")
            return []

        # --- 更新缓存键以包含过滤器 ---
        cache_key_parts = [query_text, str(n_results)]
        if where_filter:
            # 将过滤器字典转换为稳定的字符串表示
            filter_str = json.dumps(where_filter, sort_keys=True)
            cache_key_parts.append(filter_str)
        cache_key = "_".join(cache_key_parts)
        
        async with self._retrieval_cache_lock:
            cached_result = self._retrieval_cache.get(cache_key)
        
        if cached_result is not None:
            logger.info(f"ChromaDB CACHE HIT for key with filter: '{cache_key[:100]}...'")
            return cached_result
        
        logger.info(f"ChromaDB CACHE MISS for key: '{cache_key[:100]}...'. Performing retrieval.")
        if where_filter:
            logger.info(f"Applying metadata filter: {where_filter}")

        logger.info(f"Retrieving documents for query: '{query_text[:100]}...' with n_results={n_results}")
        
        try:
            query_embedding = await self._embedding_function.embed_query(query_text)
            
            if not query_embedding: 
                logger.error(f"Failed to generate embedding for query: {query_text[:100]}")
                return []
            
            # +++ 新增日志 +++
            logger.info(f"ChromaDBRetriever: Query embedding for '{query_text[:50]}' (first 10 elements): {str(query_embedding[:10])}")
            logger.info(f"ChromaDBRetriever: Length of query embedding: {len(query_embedding)}")
            is_query_emb_all_zeros = all(v == 0.0 for v in query_embedding)
            logger.info(f"ChromaDBRetriever: Is query embedding all zeros: {is_query_emb_all_zeros}")
            # +++ 结束新增日志 +++

            def _blocking_query():
                include_fields_query = include_fields if include_fields is not None else ["metadatas", "documents", "distances"]
                # --- 核心修改：在查询时应用 where_filter ---
                return self._collection.query(
                    query_embeddings=[query_embedding], 
                    n_results=n_results,
                    include=include_fields_query,
                    where=where_filter  # <--- 应用过滤器
                )

            results = await asyncio.to_thread(_blocking_query)
            
            retrieved_docs = []
            if results and results.get("ids") and results.get("ids")[0]:
                ids_list = results["ids"][0]
                documents_list = results.get("documents", [[]])[0]
                metadatas_list = results.get("metadatas", [[]])[0] 
                distances_list = results.get("distances", [[]])[0]

                for i in range(len(ids_list)):
                    chunk_id = ids_list[i]
                    metadata = metadatas_list[i] if metadatas_list and i < len(metadatas_list) else {}
                    distance = distances_list[i] if distances_list and i < len(distances_list) else float('inf')
                    content = documents_list[i] if documents_list and i < len(documents_list) else metadata.get("chunk_text", "[Content not found]")
                    score = (1 - distance / 2.0) if distance != float('inf') and distance <= 2.0 else 0.0 

                    retrieved_docs.append({
                        "id": chunk_id,
                        "content": content,
                        "score": score,
                        "distance": distance, 
                        "metadata": metadata,
                        "source_type": "vector_chromadb"
                    })
                
                logger.info(f"Retrieved {len(retrieved_docs)} documents from ChromaDB (with filter: {where_filter is not None}).")
            else:
                logger.info("No documents retrieved from ChromaDB for the query.")

            async with self._retrieval_cache_lock:
                self._retrieval_cache[cache_key] = retrieved_docs
            logger.info(f"ChromaDB CACHED {len(retrieved_docs)} results for key: '{cache_key[:100]}...'")
            return retrieved_docs

        except Exception as e:
            logger.error(f"Error during ChromaDB retrieval: {e}", exc_info=True)
            return []
        
        
    async def get_texts_by_ids(self, ids: List[str]) -> Dict[str, str]:
        """
        (异步) 根据提供的ID列表从ChromaDB集合中获取文档的文本内容。
        """
        if not self._collection:
            logger.error("ChromaDBRetriever: Collection is not initialized.")
            return {id_val: "[Error: Collection not initialized]" for id_val in ids}
        
        if not ids:
            return {}
            
        logger.info(f"ChromaDBRetriever: Getting texts for {len(ids)} IDs.")
        
        def _blocking_get():
            return self._collection.get(ids=ids, include=["documents"])

        try:
            results = await asyncio.to_thread(_blocking_get)
            retrieved_ids = results.get("ids", [])
            retrieved_docs = results.get("documents", [])
            
            texts_map = {doc_id: doc_text for doc_id, doc_text in zip(retrieved_ids, retrieved_docs)}

            for doc_id in ids:
                if doc_id not in texts_map:
                    texts_map[doc_id] = f"[Content for chunk_id {doc_id} not found in ChromaDB]"
                    logger.warning(f"ChromaDBRetriever: Content for ID '{doc_id}' not found in get() result.")

            logger.info(f"ChromaDBRetriever: Returning texts_map with {len(texts_map)} entries.")
            return texts_map

        except Exception as e:
            logger.error(f"ChromaDBRetriever: Error during get_texts_by_ids: {e}", exc_info=True)
            return {id_val: f"[Error retrieving content for ID {id_val}]" for id_val in ids}
