# /home/zhz/zhz_agent/scripts/manual_tests/test_answer_generation.py
import asyncio
import os
import sys

# --- 配置项目根目录到 sys.path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- 结束 sys.path 配置 ---

try:
    from zhz_rag.llm.sglang_wrapper import generate_answer_from_context, NO_ANSWER_PHRASE_ANSWER_CLEAN
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure 'zhz_rag' package is discoverable and all dependencies are installed.")
    sys.exit(1)

# 加载环境变量 (如果 sglang_wrapper.py 或其调用的函数依赖环境变量)
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# --- 测试用例定义 ---
test_cases = [
    {
        "name": "TC1: 完全基于上下文回答",
        "user_query": "项目Alpha的最新进展是什么？",
        "context_str": "项目Alpha目前已完成需求分析阶段，正在进行系统设计。下周将开始编码工作。负责人是张三。",
        "expected_behavior_description": "应直接从上下文中提取信息回答，例如：项目Alpha已完成需求分析，正在系统设计，下周开始编码，负责人是张三。"
    },
    {
        "name": "TC2: 上下文信息不足 - 完全无法回答",
        "user_query": "项目Beta的预算是多少？",
        "context_str": "项目Alpha目前已完成需求分析阶段。项目Gamma的预算是50万。",
        "expected_behavior_description": f"应返回标准无法回答短语：'{NO_ANSWER_PHRASE_ANSWER_CLEAN}'"
    },
    {
        "name": "TC3: 上下文信息部分相关 - 部分回答",
        "user_query": "请介绍项目Alpha的进展和风险。",
        "context_str": "项目Alpha目前已完成需求分析阶段，正在进行系统设计。关于风险，暂无明确提及。",
        "expected_behavior_description": "应回答项目Alpha的进展，并指出上下文中未提及风险信息。例如：项目Alpha已完成需求分析，正在系统设计。关于您提到的风险，提供的上下文中未包含相关信息。"
    },
    {
        "name": "TC4: 引用来源 (模拟上下文包含来源标记)",
        "user_query": "公司的年假政策有多少天？",
        "context_str": "来源文档A第3段：公司规定员工每年享有15天带薪年假。\n来源文档B第1节：所有年假申请需提前两周提交。",
        "expected_behavior_description": "应回答15天，并可能尝试引用来源。例如：根据文档A第3段，公司员工每年享有15天带薪年假。"
    },
    {
        "name": "TC5: 上下文信息重复，应综合回答",
        "user_query": "项目Alpha的负责人是谁？",
        "context_str": "根据会议记录，项目Alpha的负责人是李明。\n项目Alpha由李明负责领导。",
        "expected_behavior_description": "应综合信息，只回答一次负责人是李明，避免重复。"
    },
    {
        "name": "TC6: 避免引入外部知识",
        "user_query": "太阳的中心温度是多少？",
        "context_str": "地球是一个行星，围绕太阳公转。", # 上下文与问题无关
        "expected_behavior_description": f"应返回标准无法回答短语：'{NO_ANSWER_PHRASE_ANSWER_CLEAN}'，而不是从LLM自身知识回答。"
    },
    {
        "name": "TC7: 答案风格 - 列表格式 (如果适用)",
        "user_query": "项目Alpha的主要里程碑有哪些？",
        "context_str": "项目Alpha的关键里程碑包括：完成需求文档、通过设计评审、首个版本发布。",
        "expected_behavior_description": "答案应简洁，并可能使用列表格式呈现。例如：项目Alpha的主要里程碑包括：\n- 完成需求文档\n- 通过设计评审\n- 首个版本发布"
    }
]

async def run_tests():
    print("--- Starting Answer Generation Tests ---")
    # 确保本地LLM服务 (local_llm_service.py) 正在运行
    # sglang_wrapper.py 中的 call_llm_via_openai_api_local_only 会调用它
    if not os.getenv("SGLANG_API_URL"):
        os.environ["SGLANG_API_URL"] = "http://localhost:8088/v1/chat/completions"
        print(f"SGLANG_API_URL not set, defaulting to: {os.environ['SGLANG_API_URL']}")

    for i, case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {case['name']} ---")
        print(f"User Query: {case['user_query']}")
        print(f"Context:\n{case['context_str']}")
        print(f"Expected Behavior: {case['expected_behavior_description']}")
        
        generated_answer = await generate_answer_from_context(
            user_query=case['user_query'],
            context_str=case['context_str']
        )
        
        print(f"\nLLM Generated Answer:\n{generated_answer}")
        print("--- End of Test Case ---")

if __name__ == "__main__":
    asyncio.run(run_tests())
    print("\n--- All Answer Generation Tests Finished ---")