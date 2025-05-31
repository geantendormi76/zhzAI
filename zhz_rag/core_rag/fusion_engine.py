# zhz_agent/fusion.py
import hashlib
import jieba
import torch
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any, Optional
import logging 
import os

# 从项目内部导入pydantic_models
from zhz_agent.pydantic_models import RetrievedDocument

class FusionEngine:
    _current_script_path = os.path.abspath(__file__)
    _script_directory = os.path.dirname(_current_script_path)

    # 确保您的模型路径指向正确的位置，如果不在 local_models/bge-reranker-base
    LOCAL_RERANKER_MODEL_PATH = os.getenv(
        "RERANKER_MODEL_PATH", 
        "/home/zhz/models/bge-reranker-base" # <--- 直接指定新的、统一管理后的模型路径
    )

    def __init__(self, logger: Optional[logging.Logger] = None):
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
        self._load_reranker_model()

    def _load_reranker_model(self):
        self.logger.info(f"FusionEngine: Loading reranker model from: {self.LOCAL_RERANKER_MODEL_PATH} to {self.reranker_device}...")
        
        if not os.path.isdir(self.LOCAL_RERANKER_MODEL_PATH): # 检查是否是目录
            _error_msg_model_path = f"Error: Reranker model local path does not exist or is not a directory: {self.LOCAL_RERANKER_MODEL_PATH}."
            self.logger.error(_error_msg_model_path)
            # 在实际应用中，这里可能应该抛出异常，或者让服务无法启动
            # 为了测试，我们先允许模型为空，后续调用会检查
            self.reranker_model = None
            self.reranker_tokenizer = None
            return # 提前返回

        try:
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.LOCAL_RERANKER_MODEL_PATH)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(self.LOCAL_RERANKER_MODEL_PATH)
            self.reranker_model.to(self.reranker_device)

            if self.reranker_device == 'cuda' and hasattr(self.reranker_model, 'half'): # 检查是否有half方法
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
        if not self.reranker_model or not self.reranker_tokenizer:
            self.logger.warning("FusionEngine: Reranker model not loaded. Cannot perform fine-grained reranking. Returning documents as is (or after basic sort if any).")
            # 可以选择返回原始顺序，或者按原始分数（如果可比）排序
            return sorted(documents, key=lambda doc: doc.score if doc.score is not None else -float('inf'), reverse=True)


        if not documents:
            self.logger.info("FusionEngine: No documents to rerank.")
            return []

        # 确保文档内容是字符串
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
                    max_length=512 # BGE Reranker通常是512
                ).to(self.reranker_device)
                
                logits = self.reranker_model(**inputs).logits
                # CrossEncoder通常直接输出一个分数，而不是需要sigmoid/softmax
                # BGE-Reranker输出的是logit，可以直接用作分数，或者通过sigmoid转为概率（但不必要）
                scores = logits.view(-1).float().cpu().numpy()

            for i, doc in enumerate(valid_documents_for_reranking):
                doc.score = float(scores[i]) # 更新文档的score为reranker的打分

            reranked_docs = sorted(valid_documents_for_reranking, key=lambda doc: doc.score, reverse=True)
            self.logger.info(f"FusionEngine: Reranking complete. {len(reranked_docs)} documents sorted.")
            return reranked_docs
        except Exception as e_rerank_detail:
            self.logger.error(f"FusionEngine: Detailed error during reranking with CrossEncoder: {e_rerank_detail}", exc_info=True)
            # 如果重排序失败，返回按原始分数（如果可比）排序的文档，或者简单返回valid_documents_for_reranking
            return sorted(valid_documents_for_reranking, key=lambda d: d.score if d.score is not None else -float('inf'), reverse=True)


    def _tokenize_text(self, text: str) -> set[str]:
        if not isinstance(text, str): # 添加类型检查
            self.logger.warning(f"FusionEngine: _tokenize_text received non-string input: {type(text)}. Returning empty set.")
            return set()
        return set(jieba.cut(text))

    def _calculate_jaccard_similarity(self, query_tokens: set[str], doc_tokens: set[str]) -> float:
        if not query_tokens or not doc_tokens: # 处理空集合的情况
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
                # 如果内容重复，可以保留分数较高的一个（如果分数可比且来自不同召回源）
                # 这里简化处理，保留第一个遇到的，或者可以根据source_type和score进行更复杂的选择
                if doc.score is not None and unique_docs_map[content_hash].score is not None:
                    if doc.score > unique_docs_map[content_hash].score: # type: ignore
                        unique_docs_map[content_hash] = doc # 保留分数更高的
                elif doc.score is not None: # 当前文档有分数，已存的没有
                     unique_docs_map[content_hash] = doc
                self.logger.debug(f"FusionEngine: Duplicate content hash found. Doc with score {doc.score} vs existing {unique_docs_map[content_hash].score}. Content: {doc.content[:50]}...")
        
        unique_docs = list(unique_docs_map.values())
        self.logger.info(f"FusionEngine: After deduplication (content hash): {len(unique_docs)} documents.")

        if not unique_docs:
            return []

        # 2. 初步筛选 (基于长度和Jaccard相似度)
        # 定义阈值
        JACCARD_THRESHOLD = 0.05  # Jaccard相似度阈值，低于此则可能被过滤
        MIN_DOC_LENGTH_CHARS_KG = 10    # 知识图谱结果的最小字符长度
        MIN_DOC_LENGTH_CHARS_OTHER = 10 # 其他来源（向量、BM25）的最小字符长度
        # Reranker (如BGE-Reranker) 通常处理的token上限是512。
        # 一个中文字符大致对应1-3个token，英文单词大致对应1个token。
        # 为安全起见，可以设置一个字符上限，例如 1000-1500 字符，避免超长输入给Reranker。
        # 如果Reranker的tokenizer有max_length参数，它会自动截断，但预先过滤可以减少不必要的处理。
        MAX_DOC_LENGTH_CHARS = 1500 # 文档的最大字符长度，防止过长输入给reranker

        query_tokens_set = self._tokenize_text(user_query)
        screened_results: List[RetrievedDocument] = []
        
        self.logger.info(f"FusionEngine: Starting light screening for {len(unique_docs)} unique documents.")
        for doc_idx, doc in enumerate(unique_docs):
            doc_content_str = str(doc.content) # 确保是字符串
            doc_length = len(doc_content_str)
            
            # 长度筛选
            min_len_chars = MIN_DOC_LENGTH_CHARS_KG if doc.source_type == "knowledge_graph" else MIN_DOC_LENGTH_CHARS_OTHER
            if not (min_len_chars <= doc_length <= MAX_DOC_LENGTH_CHARS):
                self.logger.debug(f"  Screening: Doc {doc_idx} (ID: {doc.metadata.get('chunk_id', 'N/A') if doc.metadata else 'N/A'}) failed length check. Length: {doc_length}, Expected: [{min_len_chars}-{MAX_DOC_LENGTH_CHARS}], Type: {doc.source_type}. Content: '{doc_content_str[:50]}...'")
                continue

            # Jaccard相似度筛选 (可选，如果query_tokens_set为空则跳过)
            if query_tokens_set: # 只有当查询分词结果非空时才进行Jaccard计算
                doc_tokens_set = self._tokenize_text(doc_content_str)
                if not doc_tokens_set: # 如果文档分词结果为空，Jaccard为0
                    jaccard_sim = 0.0
                else:
                    jaccard_sim = self._calculate_jaccard_similarity(query_tokens_set, doc_tokens_set)
                
                if jaccard_sim < JACCARD_THRESHOLD:
                    self.logger.debug(f"  Screening: Doc {doc_idx} (ID: {doc.metadata.get('chunk_id', 'N/A') if doc.metadata else 'N/A'}) failed Jaccard check. Similarity: {jaccard_sim:.4f} < {JACCARD_THRESHOLD}. Content: '{doc_content_str[:50]}...'")
                    continue
            else:
                self.logger.debug(f"  Screening: Doc {doc_idx} (ID: {doc.metadata.get('chunk_id', 'N/A') if doc.metadata else 'N/A'}) - Query tokens empty, skipping Jaccard check.")


            screened_results.append(doc)
            self.logger.debug(f"  Screening: Doc {doc_idx} (ID: {doc.metadata.get('chunk_id', 'N/A') if doc.metadata else 'N/A'}) passed light screening.")
        
        self.logger.info(f"FusionEngine: After light screening: {len(screened_results)} documents remain.")
        
        if not screened_results:
            self.logger.info("FusionEngine: No documents remain after light screening. Returning empty list.")
            return []
        
        # 如果初筛后文档数量仍然很多，可以考虑再根据原始分数进行一次粗排和截断
        # 例如，如果 screened_results 数量 > top_n_final * 10，则取分数最高的前 top_n_final * 10 个
        # 这需要确保原始分数具有一定的可比性，或者对不同来源的分数进行大致的归一化
        # 当前我们先不加这一步，假设上游召回和初步筛选已将数量控制在合理范围
        docs_for_reranking = screened_results

        # 3. 使用Cross-Encoder进行精细重排序
        # _rerank_documents_sync 是同步函数，在异步函数中调用需要用 asyncio.to_thread
        final_fused_and_reranked_results = await asyncio.to_thread(
            self._rerank_documents_sync,
            query=user_query,
            documents=docs_for_reranking # 使用筛选后的文档
        )

        self.logger.info(f"FusionEngine: After reranking: {len(final_fused_and_reranked_results)} documents.")
        for i_doc, doc_reranked in enumerate(final_fused_and_reranked_results[:top_n_final+5]): # 日志多打几条看看分数
            self.logger.debug(f"  Reranked Doc {i_doc}: type={doc_reranked.source_type}, new_score={doc_reranked.score:.4f}, content='{str(doc_reranked.content)[:100]}...'")

        # 4. 根据 top_n_final 截取最终结果
        final_output_documents = final_fused_and_reranked_results[:top_n_final]

        self.logger.info(f"FusionEngine: Returning final top {len(final_output_documents)} documents.")
        return final_output_documents