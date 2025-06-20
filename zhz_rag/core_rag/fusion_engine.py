import os
import hashlib
import jieba
from typing import List, Dict, Any, Optional
import logging
import asyncio
from sentence_transformers import CrossEncoder

from zhz_rag.config.pydantic_models import RetrievedDocument

class FusionEngine:
    def __init__(self, logger: Optional[logging.Logger] = None, rrf_k: int = 60):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger("FusionEngineLogger")
            if not self.logger.hasHandlers():
                self.logger.setLevel(logging.INFO)
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        
        self.rrf_k = rrf_k
        self.cross_encoder: Optional[CrossEncoder] = None # 明确类型
        
        # --- 在初始化时就调用加载 ---
        self._initialize_reranker()

    def _initialize_reranker(self):
        """
        严格从本地路径初始化Cross-Encoder模型。
        如果本地路径不存在，则禁用再排序功能。
        """
        # --- 使用您指定的本地模型路径 ---
        # 模型名称/ID
        model_name = "BAAI/bge-reranker-base"
        # 构建本地路径
        local_model_path = os.path.join(os.path.expanduser("~"), "models", model_name)

        self.logger.info(f"Attempting to load reranker model from local path: {local_model_path}")

        if not os.path.isdir(local_model_path):
            self.logger.error(f"Reranker model directory not found at: '{local_model_path}'. Reranking will be disabled.")
            self.cross_encoder = None
            return

        try:
            # 直接将本地路径传递给CrossEncoder
            self.cross_encoder = CrossEncoder(local_model_path, max_length=512)
            self.logger.info(f"Cross-Encoder model loaded successfully from '{local_model_path}'.")
        except Exception as e:
            self.logger.error(f"Failed to load Cross-Encoder model from '{local_model_path}': {e}", exc_info=True)
            self.cross_encoder = None
            
    async def rerank_documents(
        self,
        query: str,
        documents: List[RetrievedDocument],
        top_n: int = 5
    ) -> List[RetrievedDocument]:
        """
        使用Cross-Encoder模型对文档列表进行再排序，返回相关性最高的top_n个文档。
        """
        if self.cross_encoder is None:
            self.logger.warning("Reranker is not available, returning top_n documents without reranking.")
            return documents[:top_n]

        if not documents:
            return []

        if not query:
            self.logger.warning("Reranking query is empty. Returning original documents.")
            return documents
        
        self.logger.info(f"Reranking {len(documents)} documents for query: '{query[:50]}...'")

        model_input_pairs = [[query, doc.content] for doc in documents]
        
        def _predict():
            # 使用 try-except 包装 predict 调用，增加鲁棒性
            try:
                return self.cross_encoder.predict(model_input_pairs, show_progress_bar=False)
            except Exception as e:
                self.logger.error(f"Error during Cross-Encoder prediction: {e}", exc_info=True)
                return [] # 返回空列表表示预测失败
            
        scores = await asyncio.to_thread(_predict)

        if len(scores) != len(documents):
            self.logger.error("Reranking failed: number of scores does not match number of documents.")
            return documents[:top_n] # 失败时返回原始文档
        
        for doc, score in zip(documents, scores):
            doc.score = float(score)

        reranked_docs = sorted(documents, key=lambda d: d.score or -1.0, reverse=True)
        
        if reranked_docs:
            self.logger.info(f"Reranking complete. Top score: {reranked_docs[0].score:.4f}")
        
        return reranked_docs[:top_n]

    # ( _apply_rrf 和 fuse_results_with_rrf 方法保持不变 )
    def _apply_rrf(self, all_docs: List[RetrievedDocument]) -> List[RetrievedDocument]:
        if not all_docs:
            return []
        docs_by_source: Dict[str, List[RetrievedDocument]] = {}
        for doc in all_docs:
            source_type = doc.source_type or "unknown_source"
            if source_type not in docs_by_source:
                docs_by_source[source_type] = []
            docs_by_source[source_type].append(doc)

        doc_scores: Dict[str, float] = {}
        doc_objects: Dict[str, RetrievedDocument] = {}

        for source_type, docs_list in docs_by_source.items():
            sorted_docs = sorted(docs_list, key=lambda d: d.score if d.score is not None else -1, reverse=True)
            for rank, doc in enumerate(sorted_docs, 1):
                content_hash = hashlib.md5(doc.content.encode('utf-8')).hexdigest()
                if content_hash not in doc_scores:
                    doc_scores[content_hash] = 0.0
                    doc_objects[content_hash] = doc
                doc_scores[content_hash] += 1.0 / (self.rrf_k + rank)
        fused_results = []
        for content_hash, rrf_score in doc_scores.items():
            doc_obj = doc_objects[content_hash]
            doc_obj.score = rrf_score
            fused_results.append(doc_obj)
        fused_results.sort(key=lambda d: d.score or 0.0, reverse=True)
        return fused_results

    async def fuse_results_with_rrf(
        self,
        all_raw_retrievals: List[RetrievedDocument],
        top_n_final: int = 3
    ) -> List[RetrievedDocument]:
        self.logger.info(f"Fusing {len(all_raw_retrievals)} raw documents using RRF.")
        fused_and_ranked_results = self._apply_rrf(all_raw_retrievals)
        return fused_and_ranked_results[:top_n_final]