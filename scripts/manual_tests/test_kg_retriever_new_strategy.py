# /home/zhz/zhz_agent/scripts/manual_tests/test_kg_retriever_new_strategy.py
import asyncio
import os
import sys
import logging
import json # 确保导入
from typing import List, Dict, Any, Optional, Callable, Iterator

# --- 配置项目根目录到 sys.path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- 结束 sys.path 配置 ---

try:
    from zhz_rag.core_rag.kg_retriever import KGRetriever
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure 'zhz_rag' package is discoverable and all dependencies are installed.")
    sys.exit(1)

# 加载环境变量
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# --- 日志配置 ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- 测试用例定义 ---
test_cases = [
    {
        "name": "TC1: 查询实体属性 (张三)",
        "user_query": "我想知道张三的详细信息。",
        "top_k": 2,
        "expected_keywords_in_content": ["张三", "PERSON"]
    },
    {
        "name": "TC2: 查询实体属性 (项目Alpha的文档编写任务)",
        "user_query": "项目Alpha的文档编写任务的具体内容是什么？",
        "top_k": 2,
        "expected_keywords_in_content": ["项目alpha的文档编写任务", "TASK"] # 假设LLM会小写化
    },
    {
        "name": "TC3: 查询关系 (张三在哪里工作)",
        "user_query": "张三在哪个公司工作？",
        "top_k": 3,
        "expected_keywords_in_content": ["张三", "谷歌", "WORKS_AT"] # 假设图谱中有张三在谷歌工作的数据
    },
    {
        "name": "TC4: 查询关系 (项目Alpha分配给谁)",
        "user_query": "项目Alpha的文档编写任务是分配给谁的？",
        "top_k": 3,
        "expected_keywords_in_content": ["项目alpha的文档编写任务", "张三", "ASSIGNED_TO"] # 假设分配给了张三
    },
    {
        "name": "TC5: 模糊查询，LLM可能提取不出精确实体用于模板",
        "user_query": "关于AI在项目管理中的应用有哪些讨论？",
        "top_k": 3,
        "expected_keywords_in_content": [] # 期望返回空，因为LLM可能无法提取精确实体给模板
                                          # 如果实现了向量检索，这里可能会有结果
    },
    {
        "name": "TC6: 与图谱无关的问题",
        "user_query": "马德里的天气怎么样？",
        "top_k": 3,
        "expected_keywords_in_content": [] # 期望返回空结果
    },
    {
        "name": "TC7: 查询实体属性 (谷歌公司)",
        "user_query": "谷歌公司的信息？",
        "top_k": 2,
        "expected_keywords_in_content": ["谷歌", "ORGANIZATION"] # 期望LLM能识别"谷歌公司"为"谷歌"且类型为ORGANIZATION
    }
]

async def run_tests():
    logger.info("--- Starting New KGRetriever Strategy Tests ---")
    if not os.getenv("SGLANG_API_URL"):
        os.environ["SGLANG_API_URL"] = "http://localhost:8088/v1/chat/completions"
        logger.info(f"SGLANG_API_URL not set, defaulting to: {os.environ['SGLANG_API_URL']}")

    kg_retriever: Optional[KGRetriever] = None
    successful_tests = 0
    failed_tests = 0
    
    try:
        kg_retriever = KGRetriever()
        logger.info("KGRetriever instance created.")

        for i, case in enumerate(test_cases):
            logger.info(f"\n--- Test Case {i+1}: {case['name']} ---")
            logger.info(f"User Query: {case['user_query']}")
            logger.info(f"Expected Keywords in Content: {case['expected_keywords_in_content']}")
            
            retrieved_docs = await kg_retriever.retrieve(
                user_query=case['user_query'],
                top_k=case['top_k']
            )
            
            logger.info(f"\nKGRetriever returned {len(retrieved_docs)} documents:")
            
            case_passed = False
            if not retrieved_docs and not case["expected_keywords_in_content"]:
                logger.info("  RESULT: PASSED (Correctly returned no documents)")
                case_passed = True
            elif retrieved_docs and not case["expected_keywords_in_content"]: # 期望空，但返回了内容
                logger.error("  RESULT: FAILED (Expected no documents, but got some)")
            elif not retrieved_docs and case["expected_keywords_in_content"]: # 期望有内容，但返回空
                logger.error("  RESULT: FAILED (Expected documents with keywords, but got no documents)")
            else: # 期望有内容，也返回了内容，检查关键词
                found_all_keywords_in_at_least_one_doc = False
                if retrieved_docs: # 确保列表不为空
                    # 检查是否至少有一个文档包含了所有期望的关键词
                    # （更宽松的检查，因为一个查询可能返回多个相关片段，不一定每个片段都包含所有关键词）
                    for doc in retrieved_docs:
                        doc_content_lower = str(doc.get('content', "")).lower()
                        current_doc_has_all = True
                        for keyword in case["expected_keywords_in_content"]:
                            if keyword.lower() not in doc_content_lower:
                                current_doc_has_all = False
                                break
                        if current_doc_has_all:
                            found_all_keywords_in_at_least_one_doc = True
                            break 
                    
                    if found_all_keywords_in_at_least_one_doc:
                        logger.info("  RESULT: PASSED (At least one retrieved document contains all expected keywords)")
                        case_passed = True
                    else:
                        logger.error("  RESULT: FAILED (No single retrieved document contains all expected keywords)")
                else: # retrieved_docs 为空，但期望有关键词，上面已经处理
                    pass

            if case_passed:
                successful_tests +=1
            else:
                failed_tests +=1

            # 打印检索到的文档详情（无论通过与否）
            if retrieved_docs:
                for doc_idx, doc in enumerate(retrieved_docs):
                    logger.info(f"  Document {doc_idx + 1}:")
                    logger.info(f"    Source Type: {doc.get('source_type')}")
                    logger.info(f"    Score: {doc.get('score')}")
                    logger.info(f"    Content: {doc.get('content')}")
                    logger.info(f"    Metadata: {json.dumps(doc.get('metadata', {}), ensure_ascii=False, indent=2)}")
            
            logger.info("--- End of Test Case ---")

    except Exception as e_main:
        logger.error(f"Error during KGRetriever test execution: {e_main}", exc_info=True)
    finally:
        if kg_retriever and hasattr(kg_retriever, 'close'):
            kg_retriever.close()
            logger.info("KGRetriever closed.")
        
        logger.info(f"\n--- KGRetriever Test Summary ---")
        logger.info(f"Total tests: {len(test_cases)}")
        logger.info(f"Passed: {successful_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info("--- All New KGRetriever Strategy Tests Finished ---")

if __name__ == "__main__":
    asyncio.run(run_tests())