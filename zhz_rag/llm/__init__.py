# /home/zhz/zhz_agent/zhz_rag/llm/__init__.py

from .llm_interface import call_llm_via_openai_api_local_only # 导入新函数

call_sglang_llm = call_llm_via_openai_api_local_only # 别名指向新函数

from .llm_interface import (
    generate_answer_from_context,
    # generate_cypher_query, # 已停用
    generate_expanded_queries,
    generate_intent_classification, # 这个现在直接用 litellm 调用 Gemini
    generate_clarification_question,
    generate_clarification_options,
    NO_ANSWER_PHRASE_ANSWER_CLEAN,
    # NO_ANSWER_PHRASE_KG_CLEAN # 已停用
)