# /home/zhz/zhz_agent/zhz_rag/llm/__init__.py

# 从 sglang_wrapper.py 导入新的函数名
from .sglang_wrapper import call_llm_via_openai_api

# 为了兼容可能仍然在其他地方使用旧名称 'call_sglang_llm' 的导入
# 我们可以创建一个别名，让 call_sglang_llm 指向新的函数
call_sglang_llm = call_llm_via_openai_api

# 导出其他 sglang_wrapper.py 中定义的、且希望从 zhz_rag.llm 直接导入的函数
from .sglang_wrapper import (
    generate_answer_from_context,
    generate_cypher_query, # 这个函数内部应该使用 call_llm_via_openai_api
    generate_expanded_queries,
    generate_intent_classification,
    generate_clarification_question,
    generate_clarification_options,
    NO_ANSWER_PHRASE_ANSWER_CLEAN,
    NO_ANSWER_PHRASE_KG_CLEAN
)

# 也可以选择性地从 custom_crewai_llms.py 导出
# from .custom_crewai_llms import CustomGeminiLLM, CustomSGLangLLM # 如果需要的话

# 你可以通过 __all__ 变量来明确指定当执行 from zhz_rag.llm import * 时会导入哪些名称
# 但通常显式导入更好
# __all__ = [
#     "call_llm_via_openai_api",
#     "call_sglang_llm", # 别名
#     "generate_answer_from_context",
#     "generate_cypher_query",
#     # ... 其他导出的名称
# ]