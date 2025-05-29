# zhz_agent/bm25.py

import json
import os
from typing import List, Dict, Any, Optional
import jieba # 用于中文分词
from bm25s import BM25 # 导入bm25s库

# --- [修改] 从项目内部导入pydantic_models -> 改为绝对导入 ---
from zhz_agent.pydantic_models import RetrievedDocument

class BM25Retriever:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.documents: List[Dict[str, Any]] = [] # 存储原始文档
        self.tokenized_corpus: List[List[str]] = [] # 存储分词后的文档
        self.bm25_model: Optional[BM25] = None # BM25模型实例
        self._load_documents_and_build_index()

    def _load_documents_and_build_index(self):
        """
        从JSON文件加载文档，进行分词，并构建BM25索引。
        """
        file_path = os.path.join(self.data_path, "sample_documents.json")
        if not os.path.exists(file_path):
            print(f"错误: 模拟BM25文档文件不存在: {file_path}")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)

        if not self.documents:
            print("警告: 模拟BM25文档文件为空。")
            return

        # 提取所有文档内容用于分词和索引
        corpus_texts = [doc['content'] for doc in self.documents]

        print("BM25Retriever: 正在对文档进行分词并构建索引...")
        # 使用jieba进行中文分词
        self.tokenized_corpus = [list(jieba.cut(text)) for text in corpus_texts]

        # 初始化BM25模型并构建索引
        self.bm25_model = BM25()
        self.bm25_model.index(self.tokenized_corpus)

        print(f"BM25Retriever: 加载了 {len(self.documents)} 个文档，BM25索引已构建。")

    async def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        """
        根据查询文本进行关键词检索。
        """
        if not self.bm25_model or not self.documents:
            print("BM25Retriever未初始化或没有文档。")
            return []

        # 对查询进行分词
        query_tokens_single = list(jieba.cut(query)) # 单个查询的分词结果
        query_tokens_batch = [query_tokens_single] # <--- 将单个查询封装成列表的列表

        # 执行BM25检索
        # bm25s的retrieve方法返回文档索引和对应的BM25分数
        doc_indices, doc_scores = self.bm25_model.retrieve(query_tokens_batch, k=top_k, return_as="tuple")

        retrieved_results: List[RetrievedDocument] = []

        # bm25s.retrieve 返回的是一个元组 (doc_indices_batch, doc_scores_batch)
        # 即使是单个查询，它们也是列表的列表，所以我们取第一个元素
        if doc_indices is not None and doc_indices.shape[0] > 0 and doc_indices[0].size > 0: # 检查是否有结果
            # doc_indices[0] 是第一个查询的结果索引数组
            # doc_scores[0] 是第一个查询的结果分数数组

            for i in range(doc_indices[0].size): # 遍历第一个查询的结果
                doc_idx = doc_indices[0][i]
                score = float(doc_scores[0][i]) # 将numpy float转换为Python float

                # 过滤掉得分过低的（BM25分数可能为负，或非常接近0）
                # 这里可以根据实际情况调整阈值，但BM25通常不设严格阈值，而是取top_k
                # 如果score为负，通常表示不相关，可以过滤掉
                if score > 0:
                    doc = self.documents[doc_idx]
                    retrieved_results.append(
                        RetrievedDocument(
                            source_type="keyword_bm25",
                            content=doc['content'],
                            score=score,
                            metadata=doc.get('metadata', {})
                        )
                    )
        print(f"BM25Retriever: 检索到 {len(retrieved_results)} 个BM25结果。")
        return retrieved_results

# 示例用法 (可以在文件末尾添加，用于独立测试)
# async def main():
#     bm25_retriever = BM25Retriever(data_path="data")
#     results = await bm25_retriever.retrieve("公司最新人力资源政策", top_k=2)
#     for res in results:
#         print(f"得分: {res.score:.4f}, 来源: {res.source_type}, 内容: {res.content[:50]}...")
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())