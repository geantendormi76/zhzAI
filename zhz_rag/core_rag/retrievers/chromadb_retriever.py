import asyncio
from typing import List, Dict, Any, Optional
import chromadb
import logging
from .embedding_functions import LlamaCppEmbeddingFunction

# 配置ChromaDBRetriever的日志记录器
logger = logging.getLogger(__name__)
# 注意：在模块级别配置basicConfig可能会影响整个应用的日志行为。
# 通常建议在应用入口处统一配置。此处暂时保留，但需注意。
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
        self._dimension: Optional[int] = None # 在这里也缓存一下维度

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
            # ChromaDB 应该能处理异步的 embedding_function
            # 注意: 此处传递的 embedding_function 必须符合 chromadb.api.types.EmbeddingFunction 协议
            # 我们的 LlamaCppEmbeddingFunction 实现了 __call__ 方法，是兼容的。
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

    # --- 修改: 将 retrieve 方法改造为异步方法 ---
    async def retrieve(self, query_text: str, n_results: int = 5, include_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]: 
        if self._collection is None or self._embedding_function is None:
            logger.error("Retriever is not properly initialized.")
            return []

        logger.info(f"Retrieving documents for query: '{query_text[:100]}...' with n_results={n_results}")
        
        try:
            # 1. (异步) 将查询文本向量化
            logger.debug(f"Calling _embedding_function.embed_query with query: '{query_text[:50]}...'")
            # --- 修改: 使用 await 来调用异步的 embed_query ---
            query_embedding = await self._embedding_function.embed_query(query_text)
            
            if not query_embedding: 
                logger.error(f"Failed to generate embedding for query: {query_text[:100]}")
                return []
            
            # 2. (异步) 在ChromaDB中查询
            # self._collection.query 是一个阻塞的I/O操作，需要用 asyncio.to_thread 包装
            def _blocking_query():
                include_fields_query = include_fields if include_fields is not None else ["metadatas", "documents", "distances"]
                return self._collection.query(
                    query_embeddings=[query_embedding], 
                    n_results=n_results,
                    include=include_fields_query 
                )

            # --- 修改: 使用 asyncio.to_thread 运行阻塞的IO调用 ---
            results = await asyncio.to_thread(_blocking_query)
            
            # 3. 处理并格式化结果
            retrieved_docs = []
            if results and results.get("ids") and results.get("ids")[0]:
                ids_list = results["ids"][0]
                # --- 修改: 现在默认请求了 documents 字段 ---
                documents_list = results.get("documents", [[]])[0]
                metadatas_list = results.get("metadatas", [[]])[0] 
                distances_list = results.get("distances", [[]])[0]

                for i in range(len(ids_list)):
                    chunk_id = ids_list[i]
                    metadata = metadatas_list[i] if metadatas_list and i < len(metadatas_list) else {}
                    distance = distances_list[i] if distances_list and i < len(distances_list) else float('inf')
                    # --- 修改: 直接从 results['documents'] 获取文本，更可靠 ---
                    content = documents_list[i] if documents_list and i < len(documents_list) else metadata.get("chunk_text", "[Content not found]")

                    # 分数计算逻辑保持不变
                    score = (1 - distance / 2.0) if distance != float('inf') and distance <= 2.0 else 0.0 

                    retrieved_docs.append({
                        "id": chunk_id,
                        "content": content,
                        "score": score,
                        "distance": distance, 
                        "metadata": metadata,
                        "source_type": "vector_chromadb" # 明确来源
                    })
                
                logger.info(f"Retrieved {len(retrieved_docs)} documents from ChromaDB.")
            else:
                logger.info("No documents retrieved from ChromaDB for the query.")

            return retrieved_docs

        except Exception as e:
            logger.error(f"Error during ChromaDB retrieval: {e}", exc_info=True)
            return []
            
    # --- 修改: 将 get_texts_by_ids 方法改造为异步方法 ---
    async def get_texts_by_ids(self, ids: List[str]) -> Dict[str, str]:
        """
        (异步) 根据提供的ID列表从ChromaDB集合中获取文档的文本内容。

        Args:
            ids (List[str]): 要获取文本的文档ID列表。

        Returns:
            Dict[str, str]: 一个字典，键是文档ID，值是对应的文本内容。
        """
        if not self._collection:
            logger.error("ChromaDBRetriever: Collection is not initialized.")
            return {id_val: "[Error: Collection not initialized]" for id_val in ids}
        
        if not ids:
            return {}
            
        logger.info(f"ChromaDBRetriever: Getting texts for {len(ids)} IDs.")
        
        def _blocking_get():
            # ChromaDB的get方法是阻塞的I/O操作
            return self._collection.get(ids=ids, include=["documents"]) # 只请求 documents

        try:
            # --- 修改: 使用 asyncio.to_thread 运行阻塞的IO调用 ---
            results = await asyncio.to_thread(_blocking_get)
            
            retrieved_ids = results.get("ids", [])
            retrieved_docs = results.get("documents", [])
            
            texts_map = {doc_id: doc_text for doc_id, doc_text in zip(retrieved_ids, retrieved_docs)}

            # 为未找到的ID填充提示信息
            for doc_id in ids:
                if doc_id not in texts_map:
                    texts_map[doc_id] = f"[Content for chunk_id {doc_id} not found in ChromaDB]"
                    logger.warning(f"ChromaDBRetriever: Content for ID '{doc_id}' not found in get() result.")

            logger.info(f"ChromaDBRetriever: Returning texts_map with {len(texts_map)} entries.")
            return texts_map

        except Exception as e:
            logger.error(f"ChromaDBRetriever: Error during get_texts_by_ids: {e}", exc_info=True)
            return {id_val: f"[Error retrieving content for ID {id_val}]" for id_val in ids}

# 简单的本地测试代码，现在需要使用 asyncio.run() 来执行
async def main_test():
    logger.info("--- ChromaDBRetriever Async Test ---")
    
    # 假设 LocalModelHandler 已经正确配置并可以实例化
    # 您需要根据您的项目结构调整这部分
    try:
        from zhz_rag.llm.local_model_handler import LocalModelHandler
        from dotenv import load_dotenv
        import os
        load_dotenv()
        
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
        
        retrieved_results = await retriever.retrieve(test_query, n_results=3)
        
        if retrieved_results:
            print(f"\n--- Retrieved Results for query: '{test_query}' ---")
            for i, doc in enumerate(retrieved_results):
                print(f"Result {i+1}:")
                print(f"  ID: {doc.get('id')}")
                print(f"  Content (first 100 chars): {str(doc.get('content'))[:100]}...")
                print(f"  Score: {doc.get('score'):.4f} (Distance: {doc.get('distance'):.4f})")
                print(f"  Metadata: {doc.get('metadata')}")
                print("-" * 20)
        else:
            print(f"\nNo results retrieved for query: '{test_query}'")

        # --- 新增：关闭进程池 ---
        model_handler.close_embedding_pool()
            
    except Exception as e:
        print(f"An error occurred during the test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # --- 修改: 使用 asyncio.run 来执行异步的 main_test ---
    asyncio.run(main_test())
