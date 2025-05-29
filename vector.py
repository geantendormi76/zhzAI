# zhz_agent/vector.py
import json
import os
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- [修改] 导入 Pydantic 模型 -> 改为绝对导入 ---
from zhz_agent.pydantic_models import RetrievedDocument

class VectorRetriever:
    def __init__(self, data_path: str):
        """
        初始化向量检索器。
        
        参数:
            data_path (str): 包含文档的目录路径。
        """
        self.data_path = data_path
        self.documents: List[Dict[str, Any]] = []  # 存储加载的文档列表
        self.vectorizer: Optional[TfidfVectorizer] = None  # TF-IDF向量化器实例
        self.document_vectors: Optional[np.ndarray] = None  # 文档的向量表示
        self._load_documents()  # 加载文档并初始化向量化器

    def _load_documents(self):
        """
        从JSON文件加载文档并初始化TF-IDF向量化器。
        """
        file_path = os.path.join(self.data_path, "sample_documents.json")
        if not os.path.exists(file_path):
            print(f"错误: 模拟向量文档文件不存在: {file_path}")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        if not self.documents:
            print("警告: 模拟向量文档文件为空。")
            return

        corpus = [doc['content'] for doc in self.documents]
        
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.document_vectors = self.vectorizer.fit_transform(corpus)
        print(f"VectorRetriever: 加载了 {len(self.documents)} 个文档，TF-IDF模型已训练。")

    async def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        """
        根据查询文本进行语义检索。
        
        参数:
            query (str): 用户输入的查询文本。
            top_k (int): 返回的最相关文档数量，默认为3。
        
        返回:
            List[RetrievedDocument]: 包含最相关文档的列表。
        """
        if not self.vectorizer or self.document_vectors is None:
            print("VectorRetriever未初始化或没有文档。")
            return []

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1] # 降序排列，取top_k
        
        retrieved_results: List[RetrievedDocument] = []
        for idx in top_indices:
            doc = self.documents[idx]
            score = float(similarities[idx]) # 将numpy float转换为Python float
            
            if score >= 0.0: # 暂时设为0，确保能返回结果
                retrieved_results.append(
                    RetrievedDocument(
                        source_type="vector_search",
                        content=doc['content'],
                        score=score,
                        metadata=doc.get('metadata', {})
                    )
                )
        print(f"VectorRetriever: 检索到 {len(retrieved_results)} 个结果。")
        return retrieved_results