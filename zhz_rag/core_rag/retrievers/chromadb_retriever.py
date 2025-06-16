# 文件: zhz_rag/core_rag/retrievers/chromadb_retriever.py

import asyncio
from typing import List, Dict, Any, Optional
import chromadb
import logging
from .embedding_functions import LlamaCppEmbeddingFunction

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

        # --- 新增: 初始化召回结果缓存和异步锁 ---
        self._retrieval_cache: Dict[str, List[Dict[str, Any]]] = {}
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

    async def retrieve(self, query_text: str, n_results: int = 5, include_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]: 
        if self._collection is None or self._embedding_function is None:
            logger.error("Retriever is not properly initialized.")
            return []

        # --- 新增: 缓存检查 ---
        cache_key = f"{query_text}_{n_results}" # 使用查询文本和n_results作为联合键
        async with self._retrieval_cache_lock:
            if cache_key in self._retrieval_cache:
                logger.info(f"ChromaDB CACHE HIT for key: '{cache_key[:100]}...'")
                return self._retrieval_cache[cache_key]
        logger.info(f"ChromaDB CACHE MISS for key: '{cache_key[:100]}...'. Performing retrieval.")
        # --- 缓存检查结束 ---

        logger.info(f"Retrieving documents for query: '{query_text[:100]}...' with n_results={n_results}")
        
        try:
            # 1. (异步) 将查询文本向量化
            logger.debug(f"Calling _embedding_function.embed_query with query: '{query_text[:50]}...'")
            query_embedding = await self._embedding_function.embed_query(query_text)
            
            if not query_embedding: 
                logger.error(f"Failed to generate embedding for query: {query_text[:100]}")
                return []
            
            # 2. (异步) 在ChromaDB中查询
            def _blocking_query():
                include_fields_query = include_fields if include_fields is not None else ["metadatas", "documents", "distances"]
                return self._collection.query(
                    query_embeddings=[query_embedding], 
                    n_results=n_results,
                    include=include_fields_query 
                )

            results = await asyncio.to_thread(_blocking_query)
            
            # 3. 处理并格式化结果
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
                
                logger.info(f"Retrieved {len(retrieved_docs)} documents from ChromaDB.")
            else:
                logger.info("No documents retrieved from ChromaDB for the query.")

            # --- 新增: 存储到缓存 ---
            async with self._retrieval_cache_lock:
                self._retrieval_cache[cache_key] = retrieved_docs
                logger.info(f"ChromaDB CACHED {len(retrieved_docs)} results for key: '{cache_key[:100]}...'")
            # --- 缓存存储结束 ---
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

# 简单的本地测试代码
async def main_test():
    logger.info("--- ChromaDBRetriever Async Test ---")
    
    try:
        from zhz_rag.llm.local_model_handler import LocalModelHandler
        from dotenv import load_dotenv
        import os
        load_dotenv()
        
        # 假设 LocalModelHandler 已经正确配置并可以实例化
        model_handler = LocalModelHandler(
            embedding_model_path=os.getenv("EMBEDDING_MODEL_PATH"),
            n_gpu_layers_embed=int(os.getenv("EMBEDDING_N_GPU_LAYERS", 0))
        )
        
        embed_fn = LlamaCppEmbeddingFunction(model_handler)
        
        retriever = ChromaDBRetriever(
            collection_name=os.getenv("CHROMA_COLLECTION_NAME", "rag_documents"),
            persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY"),
            embedding_function=embed_fn
        )
        
        test_query = "人工智能的应用有哪些？"
        
        print("\n--- First Retrieval ---")
        retrieved_results = await retriever.retrieve(test_query, n_results=3)
        
        if retrieved_results:
            print(f"Retrieved {len(retrieved_results)} results.")
        else:
            print("No results retrieved.")

        print("\n--- Second Retrieval (should hit cache) ---")
        retrieved_results_cached = await retriever.retrieve(test_query, n_results=3)

        if retrieved_results_cached:
            print(f"Retrieved {len(retrieved_results_cached)} results from cache.")
        else:
            print("No results retrieved from cache.")
        
        # 关闭模型处理器中的进程池
        model_handler.close_embedding_pool()
            
    except Exception as e:
        print(f"An error occurred during the test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    asyncio.run(main_test())
