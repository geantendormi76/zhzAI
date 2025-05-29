# zhz_agent/fusion.py
import hashlib
import jieba
import torch
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Any, Optional
import logging # <--- 新增导入 logging

# 从项目内部导入pydantic_models -> 改为绝对导入
from zhz_agent.pydantic_models import RetrievedDocument
import os

# 获取一个logger实例，可以与rag_service.py中的logger同名或不同名，
# 但如果希望输出到同一个文件，需要在rag_service.py中获取这个logger并配置handler
# 为了简单起见，这里我们让 FusionEngine 接收一个 logger 对象
# 如果没有传递logger，它会自己创建一个基本的控制台logger (或不记录详细步骤)

class FusionEngine:
    _current_script_path = os.path.abspath(__file__)
    _script_directory = os.path.dirname(_current_script_path)
    LOCAL_RERANKER_MODEL_PATH = os.path.join(_script_directory, "local_models", "bge-reranker-base")

    def __init__(self, logger: Optional[logging.Logger] = None): # <--- 修改构造函数以接收logger
        if logger:
            self.logger = logger
        else:
            # 如果没有提供logger，创建一个默认的，或者决定不记录详细步骤
            self.logger = logging.getLogger("FusionEngineLogger")
            if not self.logger.hasHandlers(): # 防止重复添加handler
                self.logger.setLevel(logging.INFO) # 或 DEBUG
                # console_handler = logging.StreamHandler()
                # console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
                # self.logger.addHandler(console_handler)
                # self.logger.propagate = False # 可选，如果不想让它也输出到根logger
                self.logger.info("FusionEngine initialized with its own basic logger.")
            else:
                self.logger.info("FusionEngine initialized, re-using existing logger configuration for FusionEngineLogger.")


        self.reranker_tokenizer: Optional[AutoTokenizer] = None
        self.reranker_model: Optional[AutoModelForSequenceClassification] = None
        self.reranker_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_reranker_model()

    def _load_reranker_model(self):
        self.logger.info(f"FusionEngine: 正在加载重排序模型从本地路径: {self.LOCAL_RERANKER_MODEL_PATH} 到 {self.reranker_device}...")
        
        if not os.path.exists(self.LOCAL_RERANKER_MODEL_PATH):
            _error_msg_model_path = f"错误：重排序模型本地路径不存在: {self.LOCAL_RERANKER_MODEL_PATH}。请先运行 download_reranker_model.py 下载模型。"
            self.logger.error(_error_msg_model_path)
            raise RuntimeError(_error_msg_model_path)

        try:
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(self.LOCAL_RERANKER_MODEL_PATH)
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(self.LOCAL_RERANKER_MODEL_PATH)
            self.reranker_model.to(self.reranker_device)

            if self.reranker_device == 'cuda':
                self.reranker_model.half()
                self.logger.info("FusionEngine: 重排序模型已加载到GPU并使用FP16。")
            else:
                self.logger.info("FusionEngine: 重排序模型已加载到CPU。")
            
            self.reranker_model.eval()
            self.logger.info("FusionEngine: 重排序模型加载成功！")
        except Exception as e:
            self.logger.error(f"错误: 重排序模型加载失败: {e}", exc_info=True)
            self.reranker_tokenizer = None
            self.reranker_model = None

    def _rerank_documents_sync(self, query: str, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        if not self.reranker_model or not self.reranker_tokenizer:
            self.logger.warning("FusionEngine: 重排序模型未加载。无法进行精细重排序。返回原始文档。")
            return documents

        if not documents:
            self.logger.info("FusionEngine: 没有文档可供精细重排序。")
            return []

        pairs = [[query, doc.content] for doc in documents]
        self.logger.info(f"FusionEngine: 正在对 {len(documents)} 个文档进行精细重排序...")
        
        with torch.no_grad():
            inputs = self.reranker_tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            ).to(self.reranker_device)
            scores = self.reranker_model(**inputs).logits.view(-1).float().cpu().numpy()

        for i, doc in enumerate(documents):
            doc.score = float(scores[i])

        reranked_docs = sorted(documents, key=lambda doc: doc.score, reverse=True)
        self.logger.info(f"FusionEngine: 精细重排序后得到 {len(reranked_docs)} 个结果。")
        return reranked_docs

    def _tokenize_text(self, text: str) -> set[str]:
        return set(jieba.cut(text))

    def _calculate_jaccard_similarity(self, query_tokens: set[str], doc_tokens: set[str]) -> float:
        intersection = len(query_tokens.intersection(doc_tokens))
        union = len(query_tokens.union(doc_tokens))
        return intersection / union if union > 0 else 0.0
        
    async def fuse_results(self, 
                           all_raw_retrievals: List[RetrievedDocument],
                           user_query: str) -> str:
        

        self.logger.info(f"FusionEngine: 正在融合总计 {len(all_raw_retrievals)} 个原始召回结果。")
        
        combined_results: List[RetrievedDocument] = []
        seen_content_hashes = set()

        for doc in all_raw_retrievals:
            content_hash = hashlib.md5(doc.content.encode('utf-8')).hexdigest()
            if content_hash not in seen_content_hashes:
                combined_results.append(doc)
                seen_content_hashes.add(content_hash)
            else:
                self.logger.debug(f"FusionEngine: 发现重复内容，已跳过: {doc.content[:50]}...")

        combined_results.sort(key=lambda doc: (doc.score if doc.score is not None else -1.0, len(doc.content)), reverse=True)
        
        self.logger.debug(f"FusionEngine: 去重和初步排序后得到 {len(combined_results)} 个结果。")
        for i_doc, doc_fused in enumerate(combined_results):
            self.logger.debug(f"  Combined Doc {i_doc}: type={doc_fused.source_type}, score={doc_fused.score}, content='{str(doc_fused.content)[:100]}...'")

        # --- 修改筛选参数 ---
        JACCARD_THRESHOLD = 0.05 
        # 对于知识图谱这类精确但可能简短的结果，可以将最小长度设得很小，或者针对不同source_type设置不同阈值
        # 简单起见，我们先统一降低 MIN_DOC_LENGTH_CHARS
        MIN_DOC_LENGTH_CHARS_KG = 5    # 知识图谱结果的最小长度可以非常短
        MIN_DOC_LENGTH_CHARS_OTHER = 30 # 其他来源文档的最小长度可以保持稍大一些 (原为50)
        MAX_DOC_LENGTH_CHARS = 1000 

        query_tokens_set = self._tokenize_text(user_query)
        
        screened_results: List[RetrievedDocument] = []
        for doc in combined_results:
            current_min_doc_length = MIN_DOC_LENGTH_CHARS_OTHER
            if doc.source_type == "knowledge_graph":
                current_min_doc_length = MIN_DOC_LENGTH_CHARS_KG
            
            if not (current_min_doc_length <= len(doc.content) <= MAX_DOC_LENGTH_CHARS):
                self.logger.debug(f"FusionEngine: 过滤掉长度不符的文档 (长度: {len(doc.content)}, 要求范围: [{current_min_doc_length}-{MAX_DOC_LENGTH_CHARS}]): type={doc.source_type}, content='{doc.content[:50]}...'")
                continue

            # Jaccard相似度过滤可以考虑只对非KG结果应用，或者对KG结果使用不同阈值
            # 为简单起见，暂时对所有类型应用相同Jaccard阈值，但要注意KG结果通常与自然语言查询的词汇重叠度不高
            if doc.source_type != "knowledge_graph": # 或者可以给KG一个更低的Jaccard阈值
                doc_tokens_set = self._tokenize_text(doc.content)
                jaccard_score = self._calculate_jaccard_similarity(query_tokens_set, doc_tokens_set)
                if jaccard_score < JACCARD_THRESHOLD:
                    self.logger.debug(f"FusionEngine: 过滤掉Jaccard相似度过低的文档 (Jaccard: {jaccard_score:.2f}): type={doc.source_type}, content='{doc.content[:50]}...'")
                    continue
            else: # 对于KG结果，我们可以选择跳过Jaccard检查，或者使用一个非常低的阈值
                self.logger.debug(f"FusionEngine: 知识图谱结果 '{doc.content[:50]}...' 跳过Jaccard相似度检查或使用默认通过。")

            screened_results.append(doc)
        
        self.logger.debug(f"FusionEngine: 轻量级初筛后得到 {len(screened_results)} 个结果。")
        for i_doc, doc_screened in enumerate(screened_results):
            self.logger.debug(f"  Screened Doc {i_doc}: type={doc_screened.source_type}, score={doc_screened.score}, jaccard_with_query_approx={self._calculate_jaccard_similarity(query_tokens_set, self._tokenize_text(doc_screened.content)):.2f}, content='{str(doc_screened.content)[:100]}...'")
        
        final_fused_and_reranked_results = await asyncio.to_thread(
            self._rerank_documents_sync,
            query=user_query, 
            documents=screened_results
        )
        
        self.logger.debug(f"FusionEngine: 精细重排序后得到 {len(final_fused_and_reranked_results)} 个结果。")
        for i_doc, doc_reranked in enumerate(final_fused_and_reranked_results):
            self.logger.debug(f"  Reranked Doc {i_doc}: type={doc_reranked.source_type}, score={doc_reranked.score:.4f}, content='{str(doc_reranked.content)[:100]}...'")        
        context_parts = []
        # --- 新增：优先处理KG结果，或者给它们特殊标记 ---
        # 我们可以先将KG结果和其他结果分开，然后有策略地组合
        kg_docs = [doc for doc in final_fused_and_reranked_results if doc.source_type == "knowledge_graph"]
        other_docs = [doc for doc in final_fused_and_reranked_results if doc.source_type != "knowledge_graph"]

        # 优先将KG结果放在前面，或者给它们更强的引导词
        doc_counter = 1
        for doc in kg_docs:
            # 可以考虑增强KG结果的提示，例如：
            context_parts.append(f"【知识图谱精确信息】文档 {doc_counter} (来源: {doc.source_type}, 原始得分: {doc.score:.4f}):\n{doc.content}\n")
            doc_counter += 1
        
        for doc in other_docs:
            context_parts.append(f"【相关上下文】文档 {doc_counter} (来源: {doc.source_type}, 重排序得分: {doc.score:.4f}):\n{doc.content}\n")
            doc_counter += 1
        
        if not context_parts:
            self.logger.info("FusionEngine: 最终未找到相关信息可供生成上下文。")
            return "未在知识库中找到相关信息。"
        
        final_context = "\n".join(context_parts)
        self.logger.info(f"FusionEngine: 最终生成的上下文文本长度: {len(final_context)}。")
        return final_context