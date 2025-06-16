import hashlib
import jieba
import torch
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any, Optional
import logging
import os

# 从项目内部导入pydantic_models
from zhz_rag.config.pydantic_models import RetrievedDocument

class FusionEngine:
    def __init__(self, logger: Optional[logging.Logger] = None, use_rrf: bool = True, rrf_k: int = 60):
        """
        初始化融合引擎。

        Args:
            logger (Optional[logging.Logger]): 日志记录器实例。
            use_rrf (bool): 是否使用RRF（倒数排序融合）代替CrossEncoder进行排序。默认为 True。
            rrf_k (int): RRF算法中的平滑常数k。默认为 60。
        """
        self.reranker_model_path_from_env = os.getenv("RERANKER_MODEL_PATH")
        if not self.reranker_model_path_from_env:
            default_fallback_path = "/home/zhz/models/Qwen3-Reranker-0.6B-seq-cls"
            logging.error(f"RERANKER_MODEL_PATH not found in environment variables! Falling back to default: {default_fallback_path}")
            self.reranker_model_path_from_env = default_fallback_path

        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger("FusionEngineLogger")
            if not self.logger.hasHandlers():
                self.logger.setLevel(logging.INFO)
                self.logger.info("FusionEngine initialized with its own basic logger (no handlers configured by default).")
            else:
                self.logger.info("FusionEngine initialized, re-using existing logger configuration for FusionEngineLogger.")

        self.reranker_tokenizer: Optional[AutoTokenizer] = None
        self.reranker_model: Optional[AutoModelForSequenceClassification] = None
        self.reranker_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.use_rrf = use_rrf
        self.rrf_k = rrf_k
        
        self._load_reranker_model()

    def _load_reranker_model(self):
        """
        加载用于重排序的Cross-Encoder模型。
        """
        self.logger.info(f"FusionEngine: Loading reranker model from: {self.reranker_model_path_from_env} to {self.reranker_device}...")
        if not os.path.isdir(self.reranker_model_path_from_env):
            _error_msg_model_path = f"Error: Reranker model local path does not exist or is not a directory: {self.reranker_model_path_from_env}."
            self.logger.error(_error_msg_model_path)
            self.reranker_model = None
            self.reranker_tokenizer = None
            return

        try:
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_path_from_env, trust_remote_code=True)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_path_from_env, trust_remote_code=True)
            self.reranker_model.to(self.reranker_device)

            if self.reranker_device == 'cuda' and hasattr(self.reranker_model, 'half'):
                self.reranker_model.half()
                self.logger.info("FusionEngine: Reranker model loaded to GPU and using FP16.")
            else:
                self.logger.info(f"FusionEngine: Reranker model loaded to {self.reranker_device}.")
            
            self.reranker_model.eval()
            self.logger.info("FusionEngine: Reranker model loading successful!")
        except Exception as e:
            self.logger.error(f"Error: Reranker model loading failed: {e}", exc_info=True)
            self.reranker_tokenizer = None
            self.reranker_model = None

    def _rerank_documents_sync(self, query: str, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
        使用Cross-Encoder模型对文档进行精细重排序（同步执行）。
        """
        if not self.reranker_model or not self.reranker_tokenizer:
            self.logger.warning("FusionEngine: Reranker model not loaded. Cannot perform fine-grained reranking. Returning documents sorted by original score.")
            return sorted(documents, key=lambda doc: doc.score if doc.score is not None else -float('inf'), reverse=True)

        if not documents:
            self.logger.info("FusionEngine: No documents to rerank.")
            return []

        pairs = []
        valid_documents_for_reranking = []
        for doc in documents:
            if isinstance(doc.content, str):
                pairs.append([query, doc.content])
                valid_documents_for_reranking.append(doc)
            else:
                self.logger.warning(f"FusionEngine: Document with non-string content skipped for reranking. ID: {doc.metadata.get('chunk_id', 'N/A')}, Type: {type(doc.content)}")
        
        if not pairs:
            self.logger.info("FusionEngine: No valid document pairs for reranking after content check.")
            return []

        self.logger.info(f"FusionEngine: Reranking {len(valid_documents_for_reranking)} documents with CrossEncoder...")
        
        try:
            with torch.no_grad():
                inputs = self.reranker_tokenizer(
                    pairs, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt', 
                    max_length=512
                ).to(self.reranker_device)
                
                logits = self.reranker_model(**inputs).logits
                scores = logits.view(-1).float().cpu().numpy()

            for i, doc in enumerate(valid_documents_for_reranking):
                doc.score = float(scores[i])

            reranked_docs = sorted(valid_documents_for_reranking, key=lambda doc: doc.score, reverse=True)
            self.logger.info(f"FusionEngine: Reranking complete. {len(reranked_docs)} documents sorted.")
            return reranked_docs
        except Exception as e_rerank_detail:
            self.logger.error(f"FusionEngine: Detailed error during reranking with CrossEncoder: {e_rerank_detail}", exc_info=True)
            return sorted(valid_documents_for_reranking, key=lambda d: d.score if d.score is not None else -float('inf'), reverse=True)

    def _apply_rrf(self, query: str, all_retrieved_docs: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """
        应用倒数排序融合（RRF）算法对文档进行排序。
        """
        if not all_retrieved_docs:
            return []

        self.logger.info(f"Applying RRF to {len(all_retrieved_docs)} documents with k={self.rrf_k}.")

        doc_scores_by_id: Dict[str, Dict[str, Any]] = {} 
        doc_objects_by_id: Dict[str, RetrievedDocument] = {}

        # 按召回源对文档进行分组
        docs_by_source: Dict[str, List[RetrievedDocument]] = {}
        for doc in all_retrieved_docs:
            source_type = doc.source_type or "unknown"
            if source_type not in docs_by_source:
                docs_by_source[source_type] = []
            docs_by_source[source_type].append(doc)

        # 为每个召回源的列表计算RRF分数贡献
        for source_type, docs_list in docs_by_source.items():
            # 假设分数越高排名越靠前
            sorted_docs = sorted(docs_list, key=lambda d: d.score if d.score is not None else -float('inf'), reverse=True)
            for rank, doc in enumerate(sorted_docs, 1): # rank 从1开始
                content_hash = hashlib.md5(doc.content.encode('utf-8')).hexdigest()
                if content_hash not in doc_objects_by_id:
                    doc_objects_by_id[content_hash] = doc 
                    doc_scores_by_id[content_hash] = {"rrf_score": 0.0}
                
                # 累加RRF分数
                doc_scores_by_id[content_hash]["rrf_score"] += 1.0 / (self.rrf_k + rank)

        # 将RRF分数更新回文档对象并排序
        final_rrf_results = []
        for content_hash, data in doc_scores_by_id.items():
            doc_obj = doc_objects_by_id[content_hash]
            doc_obj.score = data["rrf_score"]
            final_rrf_results.append(doc_obj)
        
        final_rrf_results.sort(key=lambda d: d.score if d.score is not None else 0.0, reverse=True)
        self.logger.info(f"RRF processing complete. Returning {len(final_rrf_results)} documents sorted by RRF score.")
        return final_rrf_results

    def _tokenize_text(self, text: str) -> set[str]:
        if not isinstance(text, str):
            self.logger.warning(f"FusionEngine: _tokenize_text received non-string input: {type(text)}. Returning empty set.")
            return set()
        return set(jieba.cut(text))

    def _calculate_jaccard_similarity(self, query_tokens: set[str], doc_tokens: set[str]) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        intersection = len(query_tokens.intersection(doc_tokens))
        union = len(query_tokens.union(doc_tokens))
        return intersection / union if union > 0 else 0.0
        
    async def fuse_results(
        self,
        all_raw_retrievals: List[RetrievedDocument],
        user_query: str,
        top_n_final: int = 3
    ) -> List[RetrievedDocument]:
        self.logger.info(f"FusionEngine: Fusing {len(all_raw_retrievals)} raw retrieved documents for query: '{user_query}'. Target top_n_final: {top_n_final}")

        if not all_raw_retrievals:
            self.logger.info("FusionEngine: No documents to fuse.")
            return []

        # 1. 去重 (基于内容的哈希值)
        unique_docs_map: Dict[str, RetrievedDocument] = {}
        for doc in all_raw_retrievals:
            if not isinstance(doc.content, str) or not doc.content.strip():
                self.logger.debug(f"FusionEngine: Skipping doc with invalid content for hashing: {doc.metadata.get('chunk_id', 'N/A') if doc.metadata else 'N/A'}")
                continue
            content_hash = hashlib.md5(doc.content.encode('utf-8')).hexdigest()
            if content_hash not in unique_docs_map:
                unique_docs_map[content_hash] = doc
            else:
                if doc.score is not None and unique_docs_map[content_hash].score is not None:
                    if doc.score > unique_docs_map[content_hash].score: # type: ignore
                        unique_docs_map[content_hash] = doc
                elif doc.score is not None:
                    unique_docs_map[content_hash] = doc
                self.logger.debug(f"FusionEngine: Duplicate content hash found. Doc with score {doc.score} vs existing {unique_docs_map[content_hash].score}. Content: {doc.content[:50]}...")
        
        unique_docs = list(unique_docs_map.values())
        self.logger.info(f"FusionEngine: After deduplication (content hash): {len(unique_docs)} documents.")

        if not unique_docs:
            return []

        # 2. 初步筛选 (基于长度和Jaccard相似度)
        JACCARD_THRESHOLD = 0.02
        MIN_DOC_LENGTH_CHARS_KG = 10 
        MIN_DOC_LENGTH_CHARS_OTHER = 10
        MAX_DOC_LENGTH_CHARS = 1500

        query_tokens_set = self._tokenize_text(user_query)
        screened_results: List[RetrievedDocument] = []
        
        self.logger.info(f"FusionEngine: Starting light screening for {len(unique_docs)} unique documents. Query tokens: {query_tokens_set if query_tokens_set else 'N/A'}")
        for doc_idx, doc in enumerate(unique_docs):
            doc_content_str = str(doc.content)
            doc_length = len(doc_content_str)
            doc_id_for_log = doc.metadata.get("chunk_id", doc.metadata.get("id", f"doc_idx_{doc_idx}")) if doc.metadata else f"doc_idx_{doc_idx}"
            
            # 长度筛选
            min_len_chars = MIN_DOC_LENGTH_CHARS_KG if doc.source_type in ["knowledge_graph", "duckdb_kg"] else MIN_DOC_LENGTH_CHARS_OTHER

            if not (min_len_chars <= doc_length <= MAX_DOC_LENGTH_CHARS):
                self.logger.info(f"  Screening REJECT (Length): DocID: {doc_id_for_log}, Length: {doc_length}, Expected: [{min_len_chars}-{MAX_DOC_LENGTH_CHARS}], Type: {doc.source_type}. Content: '{doc_content_str[:100].replace(chr(10), ' ')}...'")
                continue

            # Jaccard相似度筛选
            jaccard_sim = -1.0 
            apply_jaccard_filter = True

            if doc.source_type in ["duckdb_kg", "knowledge_graph"]:
                apply_jaccard_filter = False
                self.logger.info(f"  Screening INFO (Jaccard - KG Skip): DocID: {doc_id_for_log}, Type: {doc.source_type}. Skipping Jaccard filter.")
            
            if apply_jaccard_filter and query_tokens_set:
                doc_tokens_set = self._tokenize_text(doc_content_str)
                jaccard_sim = self._calculate_jaccard_similarity(query_tokens_set, doc_tokens_set)
                
                if jaccard_sim < JACCARD_THRESHOLD:
                    self.logger.info(f"  Screening REJECT (Jaccard - Non-KG): DocID: {doc_id_for_log}, Similarity: {jaccard_sim:.4f} < {JACCARD_THRESHOLD}. Doc Tokens: {doc_tokens_set if len(doc_tokens_set) < 20 else str(list(doc_tokens_set)[:20])+'...'} Content: '{doc_content_str[:100].replace(chr(10), ' ')}...'")
                    continue
            elif apply_jaccard_filter and not query_tokens_set:
                self.logger.info(f"  Screening SKIP (Jaccard - Empty Query Tokens): DocID: {doc_id_for_log}")

            screened_results.append(doc)
            jaccard_display = f"{jaccard_sim:.4f}" if jaccard_sim != -1.0 else "N/A"
            self.logger.info(f"  Screening PASS: DocID: {doc_id_for_log}, Length: {doc_length}, Jaccard: {jaccard_display}, Type: {doc.source_type}. Content: '{doc_content_str[:100].replace(chr(10), ' ')}...'")
            
        self.logger.info(f"FusionEngine: After light screening: {len(screened_results)} documents remain.")
        
        if not screened_results:
            self.logger.info("FusionEngine: No documents remain after light screening. Returning empty list.")
            return []
        
        fused_and_ranked_results: List[RetrievedDocument]
        if self.use_rrf:
            self.logger.info("FusionEngine: Using RRF for fusion and ranking.")
            fused_and_ranked_results = await asyncio.to_thread(
                self._apply_rrf,
                query=user_query,
                all_retrieved_docs=screened_results
            )
        else:
            self.logger.info("FusionEngine: Using CrossEncoder for reranking.")
            # --- 新增日志 ---
            self.logger.info(f"FusionEngine: Documents going into CrossEncoder ({len(screened_results)} items):")
            for doc_idx, doc_to_rerank in enumerate(screened_results):
                self.logger.info(f"  ToRerank[{doc_idx}]: ID={doc_to_rerank.metadata.get('chunk_id', 'N/A') if doc_to_rerank.metadata else 'N/A'}, Score={doc_to_rerank.score}, Source={doc_to_rerank.source_type}, Content='{str(doc_to_rerank.content)[:100].replace(chr(10), ' ')}...'")
            # --- 日志结束 ---
            fused_and_ranked_results = await asyncio.to_thread(
                self._rerank_documents_sync,
                query=user_query,
                documents=screened_results 
            )

        self.logger.info(f"FusionEngine: After fusion/reranking: {len(fused_and_ranked_results)} documents.")
        for i_doc, doc_reranked in enumerate(fused_and_ranked_results[:top_n_final+5]):
            self.logger.debug(f"   Ranked Doc {i_doc}: type={doc_reranked.source_type}, new_score={doc_reranked.score:.4f}, content='{str(doc_reranked.content)[:100]}...'")

        # 4. 根据 top_n_final 截取最终结果
        final_output_documents = fused_and_ranked_results[:top_n_final]

        self.logger.info(f"FusionEngine: Returning final top {len(final_output_documents)} documents.")
        return final_output_documents
