# /home/zhz/zhz_agent/scripts/manual_tests/test_clarification_generation.py
import asyncio
import os
import sys

# --- 配置项目根目录到 sys.path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- 结束 sys.path 配置 ---

try:
    from zhz_rag.llm.sglang_wrapper import generate_clarification_question
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure 'zhz_rag' package is discoverable and all dependencies are installed.")
    sys.exit(1)

# 加载环境变量
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# --- 测试用例定义 ---
test_cases = [
    {
        "name": "TC1: Excel操作意图模糊",
        "original_query": "帮我看看销售数据。",
        "uncertainty_reason": "用户没有说明具体想对销售数据做什么操作，例如是汇总、筛选还是查找特定记录。",
        "expected_behavior_description": "应询问用户想进行的具体操作类型，例如：请问您希望对销售数据进行哪种类型的分析？是想看汇总、筛选特定条件，还是查找具体记录呢？"
    },
    {
        "name": "TC2: RAG查询对象不明确",
        "original_query": "告诉我关于那个项目的最新情况。",
        "uncertainty_reason": "用户提到的“那个项目”指代不明，系统中可能存在多个项目。",
        "expected_behavior_description": "应询问用户具体指的是哪个项目，例如：请问您具体指的是哪个项目呢？如果您能提供项目名称或更详细的描述，我可以更好地帮您查找。"
    },
    {
        "name": "TC3: Web搜索范围过大",
        "original_query": "查一下人工智能。",
        "uncertainty_reason": "“人工智能”是一个非常宽泛的主题，直接搜索结果可能过多且不聚焦。",
        "expected_behavior_description": "应引导用户缩小搜索范围，例如：您想了解人工智能的哪个方面呢？是定义、最新进展、应用领域还是其他具体内容？"
    },
    {
        "name": "TC4: 缺少关键信息 (例如，RAG需要公司名)",
        "original_query": "我们公司的年假政策是怎么样的？",
        "uncertainty_reason": "缺少公司名称，无法定位到具体的年假政策文档。",
        "expected_behavior_description": "应直接询问公司名称，例如：请问您的公司全称是什么？"
    },
    {
        "name": "TC5: 简单的、不一定需要澄清的模糊指令 (看LLM如何判断)",
        "original_query": "处理一下这个文件。",
        "uncertainty_reason": "用户没有说明要对文件进行何种处理，也没有指明是哪个文件。",
        "expected_behavior_description": "期望LLM能识别出信息严重不足，并提出澄清，例如：请问您想对哪个文件进行什么操作呢？"
    }
]

async def run_tests():
    print("--- Starting Clarification Question Generation Tests ---")
    # 确保本地LLM服务 (local_llm_service.py) 正在运行
    if not os.getenv("SGLANG_API_URL"):
        os.environ["SGLANG_API_URL"] = "http://localhost:8088/v1/chat/completions"
        print(f"SGLANG_API_URL not set, defaulting to: {os.environ['SGLANG_API_URL']}")

    for i, case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {case['name']} ---")
        print(f"Original Query: {case['original_query']}")
        print(f"Uncertainty Reason: {case['uncertainty_reason']}")
        print(f"Expected Behavior Description: {case['expected_behavior_description']}")
        
        clarification_question = await generate_clarification_question(
            original_query=case['original_query'],
            uncertainty_reason=case['uncertainty_reason']
        )
        
        print(f"\nLLM Generated Clarification Question:\n{clarification_question}")
        print("--- End of Test Case ---")

if __name__ == "__main__":
    asyncio.run(run_tests())
    print("\n--- All Clarification Question Generation Tests Finished ---")