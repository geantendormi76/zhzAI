from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
import logging
from .embedding_functions import LlamaCppEmbeddingFunction

# 配置ChromaDBRetriever的日志记录器
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ChromaDBRetriever:
    def __init__(
        self,
        collection_name: str, # 改为必需参数
        persist_directory: str, # 改为必需参数
        embedding_function: LlamaCppEmbeddingFunction # <--- 新增参数
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
        self._embedding_function = embedding_function # <--- 使用新参数
        self._client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.Collection] = None

        self._initialize_retriever()

    def _initialize_retriever(self):
        """
        初始化ChromaDB客户端和集合。嵌入模型由传入的 embedding_function 处理。
        """
        try:
            logger.info(f"Initializing ChromaDB client from path: {self.persist_directory}")
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            
            logger.info(f"Getting or creating ChromaDB collection: {self.collection_name} using provided embedding function.")
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_function # <--- 传递嵌入函数

            )

            if self._collection.count() == 0:
                logger.warning(f"ChromaDB collection '{self.collection_name}' is empty!")
            else:
                logger.info(f"ChromaDB collection '{self.collection_name}' loaded. Item count: {self._collection.count()}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client or collection: {e}", exc_info=True)
            raise
            
    def retrieve(self, query_text: str, n_results: int = 5, include_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        根据查询文本从ChromaDB检索相似的文档块。

        Args:
            query_text (str): 用户查询的文本。
            n_results (int): 希望返回的最大结果数量。
            include_fields (Optional[List[str]]): 希望从ChromaDB返回的字段列表，
                                               例如 ["metadatas", "documents", "distances"]。
                                               如果为None，则ChromaDB通常会返回默认字段。
                                               我们存储时，chunk_text在metadatas里。

        Returns:
            List[Dict[str, Any]]: 检索到的文档块列表。每个字典通常包含：
                                   'id' (chunk_id), 
                                   'text' (chunk_text from metadata), 
                                   'metadata' (原始元数据),
                                   'distance' (相似度距离，越小越相似)
                                   具体结构取决于ChromaDB的返回和我们的处理。
        """
        if self._collection is None or self._embedding_function is None:
            logger.error("Retriever is not properly initialized (collection or embedding_function is None).")
            # 尝试重新初始化，或者直接返回错误/空列表
            try:
                self._initialize_retriever()
                if self._collection is None or self._embedding_function is None: # 再次检查
                    return []
            except Exception as e_reinit:
                logger.error(f"Failed to re-initialize retriever during retrieve call: {e_reinit}")
                return []

        logger.info(f"Retrieving documents for query: '{query_text[:100]}...' with n_results={n_results}")
        
        try:
            # 1. 将查询文本向量化 (现在通过 LlamaCppEmbeddingFunction)
            logger.debug(f"Calling _embedding_function.embed_query with query: '{query_text[:50]}...'")
            query_embedding = self._embedding_function.embed_query(query_text) # <--- 确认这里有 (query_text)
            
            if not query_embedding: 
                logger.error(f"Failed to generate embedding for query: {query_text[:100]}")
                return [] # 如果嵌入失败，直接返回空列表
            
            # 2. 在ChromaDB中查询 (include_fields_query 的逻辑不变)
            if include_fields is None:
                include_fields_query = ["metadatas", "distances"] 
            else:
                include_fields_query = include_fields
            results = self._collection.query(
                query_embeddings=[query_embedding], 
                n_results=n_results,
                include=include_fields_query 
            )
            # 3. 处理并格式化结果
            retrieved_docs = []
            if results and results.get("ids") and results.get("ids")[0]:
                ids_list = results["ids"][0]
                metadatas_list = results.get("metadatas", [[]])[0] 
                distances_list = results.get("distances", [[]])[0] 

                for i in range(len(ids_list)):
                    # ... (提取 chunk_id, metadata, distance 的代码不变) ...
                    chunk_id = ids_list[i]
                    metadata = metadatas_list[i] if metadatas_list and i < len(metadatas_list) else {}
                    distance = distances_list[i] if distances_list and i < len(distances_list) else float('inf')
                    chunk_text_content = metadata.get("chunk_text", "[Chunk text not found in metadata]")

                    # 计算相似度分数
                    # 如果ChromaDB使用cosine距离 (范围0-2, 0表示最相似)
                    # 相似度 = 1 - (distance / 2)  => 范围 0-1, 1表示最相似
                    # 或者直接用 cosine_similarity = 1 - distance (如果distance是 1-cos_sim)
                    # ChromaDB的cosine距离是 sqrt(2-2*cos_sim) 的平方，即 2-2*cos_sim (如果向量已归一化)
                    # 所以 cos_sim = 1 - distance / 2
                    # 我们希望分数越大越好
                    score = (1 - distance / 2.0) if distance != float('inf') and distance <=2.0 else 0.0 
                    # 确保分数在合理范围，如果distance意外地大于2，则score为0

                    retrieved_docs.append({
                        "id": chunk_id,
                        "text": chunk_text_content,
                        "score": score, # <--- 更新了分数计算
                        "distance": distance, 
                        "metadata": metadata 
                    })
                
                logger.info(f"Retrieved {len(retrieved_docs)} documents from ChromaDB.")
            else:
                logger.info("No documents retrieved from ChromaDB for the query.")

            return retrieved_docs

        except Exception as e:
            logger.error(f"Error during ChromaDB retrieval: {e}")
            return []
        
    def get_texts_by_ids(self, ids: List[str]) -> Dict[str, str]:
        """
        根据提供的ID列表从ChromaDB集合中获取文档的文本内容。

        Args:
            ids (List[str]): 要获取文本的文档ID列表。

        Returns:
            Dict[str, str]: 一个字典，键是文档ID，值是对应的文本内容。
                            如果某个ID未找到或其元数据中没有文本，则对应的值可能不存在或为特定提示。
        """
        if not self._collection:
            logger.error("ChromaDBRetriever: Collection is not initialized. Cannot get texts by IDs.")
            return {id_val: "[Error: Collection not initialized]" for id_val in ids}
        
        if not ids:
            return {}
            
        logger.info(f"ChromaDBRetriever: Getting texts for {len(ids)} IDs.")
        
        texts_map: Dict[str, str] = {}
        try:
            # ChromaDB的get方法可以接收ids列表，并指定包含哪些部分
            # 我们需要元数据中的 'chunk_text'
            results = self._collection.get(
                ids=ids,
                include=["metadatas"] # 我们只需要元数据来提取 chunk_text
            )
            
            retrieved_ids = results.get("ids", [])
            retrieved_metadatas = results.get("metadatas", [])
            
            if retrieved_ids and retrieved_metadatas and len(retrieved_ids) == len(retrieved_metadatas):
                for i in range(len(retrieved_ids)):
                    doc_id = retrieved_ids[i]
                    metadata = retrieved_metadatas[i]
                    if metadata and "chunk_text" in metadata:
                        texts_map[doc_id] = metadata["chunk_text"]
                    else:
                        texts_map[doc_id] = f"[Content for chunk_id {doc_id} not found in metadata]"
                        logger.warning(f"ChromaDBRetriever: Metadata or 'chunk_text' not found for ID: {doc_id}")
            else:
                logger.warning(f"ChromaDBRetriever: Mismatch in lengths or empty results from _collection.get(). IDs asked: {len(ids)}, IDs got: {len(retrieved_ids)}")
                # 为所有请求的ID设置一个未找到的默认值
                for id_val in ids:
                    if id_val not in texts_map: # 避免覆盖已可能通过其他方式找到的
                         texts_map[id_val] = f"[Content for chunk_id {id_val} not found after ChromaDB get()]"

        except Exception as e:
            logger.error(f"ChromaDBRetriever: Error during get_texts_by_ids: {e}", exc_info=True)
            # 出错时，为所有请求的ID设置错误提示
            for id_val in ids:
                 texts_map[id_val] = f"[Error retrieving content for ID {id_val}]"
        
        logger.info(f"ChromaDBRetriever: Returning texts_map with {len(texts_map)} entries.")
        return texts_map

if __name__ == '__main__':
    # 简单的测试代码
    logger.info("--- ChromaDBRetriever Test ---")
    
    # 确保您的ChromaDB数据库中已经有通过Dagster流水线存入的数据
    # 并且模型路径正确
    try:
        retriever = ChromaDBRetriever()
        
        # 测试查询
        test_query = "人工智能的应用有哪些？" 
        # 根据您doc1.txt "这是第一个测试文档，关于人工智能和机器学习。" 应该能召回一些
        
        retrieved_results = retriever.retrieve(test_query, n_results=3)
        
        if retrieved_results:
            print(f"\n--- Retrieved Results for query: '{test_query}' ---")
            for i, doc in enumerate(retrieved_results):
                print(f"Result {i+1}:")
                print(f"  ID: {doc['id']}")
                print(f"  Text (first 100 chars): {doc['text'][:100]}...")
                print(f"  Score: {doc['score']:.4f} (Distance: {doc['distance']:.4f})")
                print(f"  Metadata: {doc['metadata']}")
                print("-" * 20)
        else:
            print(f"\nNo results retrieved for query: '{test_query}'")
            
        # 测试一个可能没有结果的查询
        # test_query_no_results = "恐龙为什么会灭绝？"
        # retrieved_no_results = retriever.retrieve(test_query_no_results, n_results=3)
        # if not retrieved_no_results:
        #     print(f"\nCorrectly retrieved no results for query: '{test_query_no_results}'")

    except Exception as e:
        print(f"An error occurred during the test: {e}")
