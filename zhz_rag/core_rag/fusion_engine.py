import hashlib
import jieba
from typing import List, Dict, Any, Optional
import logging
import asyncio # 确保存在，未来可能用到

from zhz_rag.config.pydantic_models import RetrievedDocument

class FusionEngine:
    def __init__(self, logger: Optional[logging.Logger] = None, rrf_k: int = 60):
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger("FusionEngineLogger")
            # (如果需要，可以在这里添加基本的日志配置)
        
        self.rrf_k = rrf_k
        self.logger.info(f"FusionEngine initialized. Strategy: RRF with k={self.rrf_k}.")

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

    def _apply_rrf(self, all_docs: List[RetrievedDocument]) -> List[RetrievedDocument]:
        if not all_docs:
            return []

        # 1. 按召回源对文档进行分组，并记录其原始排名
        docs_by_source: Dict[str, List[RetrievedDocument]] = {}
        for doc in all_docs:
            source_type = doc.source_type or "unknown_source"
            if source_type not in docs_by_source:
                docs_by_source[source_type] = []
            docs_by_source[source_type].append(doc)

        # 2. 计算每个文档的RRF分数
        doc_scores: Dict[str, float] = {}
        doc_objects: Dict[str, RetrievedDocument] = {}

        for source_type, docs_list in docs_by_source.items():
            # 按原始分数降序排序，分数越高排名越靠前
            sorted_docs = sorted(docs_list, key=lambda d: d.score if d.score is not None else -float('inf'), reverse=True)
            
            for rank, doc in enumerate(sorted_docs, 1): # rank 从1开始
                # 使用内容的哈希值作为文档的唯一标识符
                content_hash = hashlib.md5(doc.content.encode('utf-8')).hexdigest()
                
                if content_hash not in doc_scores:
                    doc_scores[content_hash] = 0.0
                    doc_objects[content_hash] = doc
                
                # 累加RRF分数
                doc_scores[content_hash] += 1.0 / (self.rrf_k + rank)

        # 3. 将RRF分数更新到文档对象中
        fused_results = []
        for content_hash, rrf_score in doc_scores.items():
            doc_obj = doc_objects[content_hash]
            doc_obj.score = rrf_score  # 使用RRF分数覆盖原始分数
            fused_results.append(doc_obj)
        
        # 4. 根据RRF分数对最终结果进行排序
        fused_results.sort(key=lambda d: d.score or 0.0, reverse=True)
        
        self.logger.info(f"RRF processing complete. Returning {len(fused_results)} unique documents sorted by RRF score.")
        return fused_results

    async def fuse_results(
        self,
        all_raw_retrievals: List[RetrievedDocument],
        user_query: str,
        top_n_final: int = 3
    ) -> List[RetrievedDocument]:
        self.logger.info(f"FusionEngine: Fusing {len(all_raw_retrievals)} raw documents for query: '{user_query}' using RRF.")

        if not all_raw_retrievals:
            return []

        # 1. 初步筛选 (Light Screening) - 对非KG来源的文档进行Jaccard相似度过滤
        JACCARD_THRESHOLD = 0.02
        query_tokens = self._tokenize_text(user_query)
        screened_docs: List[RetrievedDocument] = []
        
        for doc in all_raw_retrievals:
            # KG召回的结果通常是实体或关系，Jaccard相似度不适用，应直接通过
            if doc.source_type in ["knowledge_graph", "duckdb_kg"]:
                screened_docs.append(doc)
                continue

            # 对于其他来源，进行Jaccard相似度检查
            if query_tokens:
                doc_tokens = self._tokenize_text(doc.content)
                similarity = self._calculate_jaccard_similarity(query_tokens, doc_tokens)
                if similarity >= JACCARD_THRESHOLD:
                    screened_docs.append(doc)
                else:
                    self.logger.debug(f"Screening REJECT (Jaccard): Doc from {doc.source_type}, Sim: {similarity:.4f} < {JACCARD_THRESHOLD}. Content: '{doc.content[:80]}...'")
            else:
                # 如果查询为空，则不进行Jaccard过滤
                screened_docs.append(doc)
        
        self.logger.info(f"After light screening, {len(screened_docs)} documents remain for RRF.")

        if not screened_docs:
            return []

        # 2. 去重并应用RRF融合
        # RRF算法天然地处理了来自不同源的相同内容，因为它会累加分数到同一个content_hash上
        # 我们只需将所有筛选后的文档传入即可
        fused_and_ranked_results = self._apply_rrf(screened_docs)

        self.logger.info(f"Fusion engine returning final top {top_n_final} documents after RRF.")
        return fused_and_ranked_results[:top_n_final]
    def __init__(self, logger: Optional[logging.Logger] = None, rrf_k: int = 60): # 移除了 use_rrf
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger("FusionEngineLogger")
            if not self.logger.hasHandlers(): # 基本的日志配置
                self.logger.setLevel(logging.INFO)
                ch = logging.StreamHandler() # 默认输出到控制台
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)
                # self.logger.propagate = False # 通常不需要，除非您有特定的父logger行为
                self.logger.info("FusionEngine initialized with its own basic logger.")

        self.rrf_k = rrf_k
        self.logger.info(f"FusionEngine initialized. Default strategy: RRF with k={self.rrf_k}.")


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

    def _apply_rrf(self, query: str, all_screened_docs: List[RetrievedDocument]) -> List[RetrievedDocument]: # query参数保留，可能未来RRF变种会用到
        if not all_screened_docs:
            return []

        self.logger.info(f"Applying RRF to {len(all_screened_docs)} screened documents with k={self.rrf_k}.")

        doc_scores_by_content_hash: Dict[str, Dict[str, Any]] = {} # 使用 content_hash 作为主键


        # 1. 按召回源对文档进行分组，并记录其原始排名
        docs_by_source: Dict[str, List[RetrievedDocument]] = {}
        for doc in all_screened_docs:
            source_type = doc.source_type or "unknown_source" # 处理 source_type 可能为 None 的情况
            if source_type not in docs_by_source:
                docs_by_source[source_type] = []
            docs_by_source[source_type].append(doc)

        for source_type, docs_list in docs_by_source.items():
            # 假设分数越高排名越靠前，如果分数相同则按出现顺序 (Python的sorted是稳定的)
            sorted_docs_from_source = sorted(docs_list, key=lambda d: d.score if d.score is not None else -float('inf'), reverse=True)
            for rank, doc_from_source in enumerate(sorted_docs_from_source, 1): # rank 从1开始
                content_hash = hashlib.md5(doc_from_source.content.encode('utf-8')).hexdigest()
                
                if content_hash not in doc_scores_by_content_hash:
                    # 存储第一个遇到的具有此内容的文档对象，并初始化RRF分数
                    doc_scores_by_content_hash[content_hash] = {
                        "rrf_score": 0.0,
                        "doc_object": doc_from_source # 存储原始文档对象
                    }
                
                # 累加RRF分数
                doc_scores_by_content_hash[content_hash]["rrf_score"] += 1.0 / (self.rrf_k + rank)

        # 2. 将RRF分数更新回文档对象并收集结果
        final_rrf_results = []
        for content_hash, data in doc_scores_by_content_hash.items():
            doc_obj_to_update = data["doc_object"]
            doc_obj_to_update.score = data["rrf_score"] # 用RRF分数覆盖原始分数
            final_rrf_results.append(doc_obj_to_update)
        
        # 3. 根据RRF分数对最终结果进行排序
        final_rrf_results.sort(key=lambda d: d.score if d.score is not None else 0.0, reverse=True) # 确保score为None时有默认值
        
        self.logger.info(f"RRF processing complete. Returning {len(final_rrf_results)} documents sorted by RRF score.")
        return final_rrf_results
        
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

        # 1. 去重 (基于内容的哈希值) - 逻辑保持不变
        unique_docs_map: Dict[str, RetrievedDocument] = {}
        for doc in all_raw_retrievals:
            if not isinstance(doc.content, str) or not doc.content.strip():
                self.logger.debug(f"FusionEngine: Skipping doc with invalid content for hashing: {doc.metadata.get('chunk_id', 'N/A') if doc.metadata else 'N/A'}")
                continue
            content_hash = hashlib.md5(doc.content.encode('utf-8')).hexdigest()
            if content_hash not in unique_docs_map:
                unique_docs_map[content_hash] = doc
            else:
                # 保留分数较高的一个（如果分数可比）
                if doc.score is not None and unique_docs_map[content_hash].score is not None:
                    if doc.score > unique_docs_map[content_hash].score: # type: ignore
                        unique_docs_map[content_hash] = doc
                elif doc.score is not None: # 当前文档有分数，已存的没有
                       unique_docs_map[content_hash] = doc
        
        unique_docs = list(unique_docs_map.values())
        self.logger.info(f"FusionEngine: After deduplication (content hash): {len(unique_docs)} documents.")

        if not unique_docs:
            return []

        # 2. 初步筛选 (基于长度和Jaccard相似度) - 逻辑保持不变，除了日志中的jaccard_display
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
            
            min_len_chars_for_current_doc = MIN_DOC_LENGTH_CHARS_KG if doc.source_type in ["knowledge_graph", "duckdb_kg"] else MIN_DOC_LENGTH_CHARS_OTHER
            if not (min_len_chars_for_current_doc <= doc_length <= MAX_DOC_LENGTH_CHARS):
                self.logger.info(f"  Screening REJECT (Length): DocID: {doc_id_for_log}, Length: {doc_length}, Expected: [{min_len_chars_for_current_doc}-{MAX_DOC_LENGTH_CHARS}], Type: {doc.source_type}. Content: '{doc_content_str[:100].replace(chr(10), ' ')}...'")
                continue

            jaccard_sim = -1.0 
            apply_jaccard_filter = True
            if doc.source_type in ["duckdb_kg", "knowledge_graph"]:
                apply_jaccard_filter = False
                self.logger.info(f"  Screening INFO (Jaccard - KG Skip): DocID: {doc_id_for_log}, Type: {doc.source_type}. Skipping Jaccard filter.")
            
            if apply_jaccard_filter and query_tokens_set:
                doc_tokens_set = self._tokenize_text(doc_content_str)
                jaccard_sim = self._calculate_jaccard_similarity(query_tokens_set, doc_tokens_set)
                if jaccard_sim < JACCARD_THRESHOLD:
                    self.logger.info(f"  Screening REJECT (Jaccard - Non-KG): DocID: {doc_id_for_log}, Similarity: {jaccard_sim:.4f} < {JACCARD_THRESHOLD}. Doc Tokens: {list(doc_tokens_set)[:20] if doc_tokens_set else 'N/A'} Content: '{doc_content_str[:100].replace(chr(10), ' ')}...'")
                    continue
            elif apply_jaccard_filter and not query_tokens_set:
                self.logger.info(f"  Screening SKIP (Jaccard - Empty Query Tokens): DocID: {doc_id_for_log}")
            
            jaccard_display = f"{jaccard_sim:.4f}" if jaccard_sim != -1.0 else "N/A (KG or no query tokens)" # 更清晰的日志
            screened_results.append(doc)
            self.logger.info(f"  Screening PASS: DocID: {doc_id_for_log}, Length: {doc_length}, Jaccard: {jaccard_display}, Type: {doc.source_type}. Content: '{doc_content_str[:100].replace(chr(10), ' ')}...'")
            
        self.logger.info(f"FusionEngine: After light screening: {len(screened_results)} documents remain.")
        
        if not screened_results:
            self.logger.info("FusionEngine: No documents remain after light screening. Returning empty list.")
            return []
        
        # --- 直接使用 RRF ---
        self.logger.info("FusionEngine: Using RRF for fusion and ranking.")
        fused_and_ranked_results = await asyncio.to_thread(
            self._apply_rrf, # 直接调用 _apply_rrf
            query=user_query, # query 参数仍然传递，以备未来RRF变种可能需要
            all_screened_docs=screened_results
        )
        # --- RRF 调用结束 ---

        self.logger.info(f"FusionEngine: After RRF processing: {len(fused_and_ranked_results)} documents.") # 更新日志
        for i_doc, doc_ranked in enumerate(fused_and_ranked_results[:top_n_final+5]): # 更新变量名
            self.logger.debug(f"   RRF Ranked Doc {i_doc}: Source={doc_ranked.source_type}, RRF_Score={doc_ranked.score:.4f}, Content='{str(doc_ranked.content)[:100]}...'")

        final_output_documents = fused_and_ranked_results[:top_n_final]
        self.logger.info(f"FusionEngine: Returning final top {len(final_output_documents)} documents after RRF.")
        return final_output_documents