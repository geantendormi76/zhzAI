# zhz_agent/llm.py (renamed to llm_interface.py as per typical module naming)

import pandas as pd
from cachetools import TTLCache
import os
import httpx  # 用于异步HTTP请求
import json  # 用于处理JSON数据
import asyncio  # 用于 asyncio.to_thread
from typing import List, Dict, Any, Optional, Union, Callable 
from dotenv import load_dotenv
import traceback  # Ensure traceback is imported
from pydantic import ValidationError
from zhz_rag.utils.interaction_logger import log_interaction_data # 导入修复后的健壮日志函数
from zhz_rag.config.constants import NEW_KG_SCHEMA_DESCRIPTION # <--- 确保导入这个常量

from zhz_rag.config.pydantic_models import ExtractedEntitiesAndRelationIntent, RagQueryPlan, UserIntent # <--- 新增导入
# 提示词导入
from llama_cpp import Llama, LlamaGrammar 
from zhz_rag.llm.rag_prompts import (
    get_answer_generation_messages,
    get_clarification_question_messages,
    get_query_expansion_messages,
    get_suggestion_generation_messages,
    get_fusion_messages,
    get_document_summary_messages,
    get_task_extraction_messages,
    V2_PLANNING_PROMPT_TEMPLATE,
    V2_PLANNING_GBNF_SCHEMA,
    get_intent_classification_messages,   
    INTENT_CLASSIFICATION_GBNF_SCHEMA   
)
import logging
import re
import uuid  # 用于生成 interaction_id
from datetime import datetime, timezone  # 用于生成时间戳
import litellm # <--- 确保这个导入存在

# --- 全局 GBNF 模型实例管理 (性能优化) ---
_llm_gbnf_instance: Optional[Llama] = None
_llm_gbnf_instance_lock = asyncio.Lock()

def _get_gbnf_llm_instance() -> Llama:
    """
    获取一个单例的、线程安全的 Llama GBNF 模型实例。
    在第一次调用时初始化。
    """
    global _llm_gbnf_instance
    if _llm_gbnf_instance is None:
        model_path_from_env = os.getenv("LOCAL_LLM_GGUF_MODEL_PATH")
        if not model_path_from_env or not os.path.exists(model_path_from_env):
            llm_py_logger.critical(f"关键错误: GBNF 调用的 LLM 模型路径未设置或无效: {model_path_from_env}")
            raise ValueError("LOCAL_LLM_GGUF_MODEL_PATH 环境变量未配置或路径无效。")
        
        llm_py_logger.info(f"--- 正在加载 GBNF LLM 模型，请稍候... Path: {model_path_from_env} ---")
        _llm_gbnf_instance = Llama(
            model_path=model_path_from_env,
            n_gpu_layers=int(os.getenv("LLM_N_GPU_LAYERS", 0)),
            n_ctx=int(os.getenv("LLM_N_CTX", 4096)),
            verbose=False
        )
        llm_py_logger.info("--- GBNF LLM 模型加载成功 ---")
    return _llm_gbnf_instance
# --- 全局 GBNF 模型实例管理结束 ---

load_dotenv()  # 确保加载.env文件

_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_INTERACTION_LOGS_DIR = os.path.join(_LLM_DIR, '..', '..', 'stored_data', 'rag_interaction_logs')


if not os.path.exists(RAG_INTERACTION_LOGS_DIR):
    try:
        os.makedirs(RAG_INTERACTION_LOGS_DIR)
    except Exception:
        pass

def get_llm_log_filepath() -> str:
    """获取当前LLM交互日志文件的完整路径，按天分割。"""
    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return os.path.join(RAG_INTERACTION_LOGS_DIR, f"rag_interactions_{today_str}.jsonl")

async def log_llm_interaction_to_jsonl(interaction_data: Dict[str, Any]):
    """
    将单条LLM交互数据异步追加到JSONL文件中。
    (This function might be part of what log_interaction_data uses, or an alternative logger. Keeping for completeness from original llm.py)
    """
    filepath = get_llm_log_filepath()
    try:
        def _write_sync():
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(interaction_data, ensure_ascii=False) + "\n")
        await asyncio.to_thread(_write_sync)
        llm_py_logger.debug(f"Successfully logged LLM interaction to {filepath}")
    except Exception as e:
        llm_py_logger.error(f"Failed to log LLM interaction to {filepath}: {e}", exc_info=True)

llm_py_logger = logging.getLogger("LLMUtilsLogger")
llm_py_logger.setLevel(os.getenv("LLM_LOG_LEVEL", "INFO").upper())

if not llm_py_logger.hasHandlers():
    _llm_console_handler = logging.StreamHandler()
    _llm_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _llm_console_handler.setFormatter(_llm_formatter)
    llm_py_logger.addHandler(_llm_console_handler)
    llm_py_logger.propagate = False

llm_py_logger.info("--- LLMUtilsLogger configured ---")

NO_ANSWER_PHRASE_ANSWER_CLEAN = "根据目前提供的资料，我无法找到关于您问题的明确信息。"
NO_ANSWER_PHRASE_KG_CLEAN = "从知识图谱中未找到直接相关信息。"
UNIQUE_STOP_TOKEN = "<|im_endofunable|>"
NO_ANSWER_PHRASE_ANSWER_WITH_STOP_TOKEN = f"{NO_ANSWER_PHRASE_ANSWER_CLEAN}{UNIQUE_STOP_TOKEN}"
# NO_ANSWER_PHRASE_KG_WITH_STOP_TOKEN = f"{NO_ANSWER_PHRASE_KG_CLEAN}{UNIQUE_STOP_TOKEN}"

# # Placeholder for the schema description. Replace with actual schema.
# NEW_KG_SCHEMA_DESCRIPTION = """
# {
#   "node_labels": ["Person", "Project", "Task", "Document", "Region", "SalesAmount", "Product"],
#   "relationship_types": ["WORKS_ON", "ASSIGNED_TO", "HAS_DOCUMENT", "HAS_SALES_AMOUNT", "RELATED_TO"],
#   "node_properties": {
#     "Person": [{"property": "name", "type": "STRING"}, {"property": "role", "type": "STRING"}],
#     "Project": [{"property": "name", "type": "STRING"}, {"property": "status", "type": "STRING"}],
#     "Task": [{"property": "name", "type": "STRING"}, {"property": "status", "type": "STRING"}, {"property": "priority", "type": "STRING"}],
#     "Document": [{"property": "id", "type": "STRING"}, {"property": "title", "type": "STRING"}, {"property": "type", "type": "STRING"}],
#     "Region": [{"property": "name", "type": "STRING"}],
#     "SalesAmount": [{"property": "period", "type": "STRING"}, {"property": "numeric_amount", "type": "FLOAT"}, {"property": "unit", "type": "STRING"}],
#     "Product": [{"property": "name", "type": "STRING"}, {"property": "category", "type": "STRING"}]
#   },
#   "relationship_properties": {},
#   "output_format_guidance": {
#     "description": "Your response MUST be a JSON object with two fields: 'status' and 'query'.",
#     "status_field": {
#       "description": "The 'status' field can be one of two values: 'success' or 'unable_to_generate'.",
#       "success": "If you can generate a Cypher query, status should be 'success'.",
#       "unable_to_generate": "If you cannot generate a Cypher query based on the question and schema, status should be 'unable_to_generate'."
#     },
#     "query_field": {
#       "description": "The 'query' field contains the Cypher query as a string if status is 'success'.",
#       "success_example": "MATCH (n) RETURN n LIMIT 1",
#       "unable_to_generate_example": "无法生成Cypher查询."
#     }
#   },
#   "examples": [
#     {
#       "User Question": "Who is task 'FixBug123' assigned to?",
#       "Your EXACT Response": {
#         "status": "success",
#         "query": "MATCH (t:Task {name: 'FixBug123'})<-[:ASSIGNED_TO]-(p:Person) RETURN p.name AS assignedTo"
#       }
#     },
#     {
#       "User Question": "What is the color of the sky?",
#       "Your EXACT Response": {
#         "status": "unable_to_generate",
#         "query": "无法生成Cypher查询."
#       }
#     }
#   ]
# }
# """

LLM_API_URL = os.getenv("SGLANG_API_URL", "http://localhost:8088/v1/chat/completions")

async def call_llm_via_openai_api_local_only( # 改个名字以示区分
    prompt: Union[str, List[Dict[str, str]]], # prompt 可以是字符串或消息列表
    temperature: float = 0.2,
    max_new_tokens: Optional[int] = 1024,
    stop_sequences: Optional[List[str]] = None,
    task_type: str = "unknown_local_llm_call",
    user_query_for_log: Optional[str] = None,
    model_name_for_log: str = "local_qwen_via_openai_api_compat",
    application_version_for_log: str = "0.1.0_local_compat"
) -> Optional[str]:
    llm_py_logger.info(f"Calling LOCAL LLM ({model_name_for_log}) for task: {task_type}, Target API: {LLM_API_URL}")

    current_messages: List[Dict[str, str]]
    original_prompt_for_log: str

    if isinstance(prompt, str): # 假设旧的SGLang风格的prompt字符串
        original_prompt_for_log = prompt
        # 尝试从SGLang格式转换为OpenAI messages格式
        # 这个转换逻辑需要根据您SGLang prompt的具体格式来定
        # 一个简化的例子，可能需要调整：
        current_messages = []
        # 简单的假设：如果prompt以<|im_start|>system开头，则提取system和user部分
        if prompt.startswith("<|im_start|>system"):
            parts = prompt.split("<|im_start|>")
            for part in parts:
                if not part.strip(): continue
                role_content = part.split("<|im_end|>")[0].strip()
                if "\n" in role_content:
                    role, content = role_content.split("\n", 1)
                    current_messages.append({"role": role.strip().lower(), "content": content.strip()})
        if not current_messages: # 如果转换失败或不是SGLang格式，则认为是单个user消息
            current_messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        current_messages = prompt
        original_prompt_for_log = "Messages list provided directly."
    else:
        llm_py_logger.error(f"Invalid 'prompt' argument type: {type(prompt)}")
        return None

    payload = {
        "model": model_name_for_log, # 这个model名会被本地服务忽略，但符合OpenAI格式
        "messages": current_messages,
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }
    if stop_sequences:
        payload["stop"] = stop_sequences

    headers = {"Content-Type": "application/json"}
    llm_parameters_for_log = {k:v for k,v in payload.items() if k not in ['messages', 'model']} # model已在顶层记录
    raw_llm_output_text = None
    error_info = None

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(LLM_API_URL, json=payload, headers=headers) # LLM_API_URL 指向本地服务
            response.raise_for_status()
            response_json = response.json()
            if response_json.get("choices") and response_json["choices"][0].get("message"):
                raw_llm_output_text = response_json["choices"][0]["message"].get("content", "")
            else:
                raw_llm_output_text = "[[LLM_RESPONSE_MALFORMED_CHOICES_OR_MESSAGE_LOCAL]]"
            llm_py_logger.info(f"FULL Local LLM Raw Output for task '{task_type}': >>>{raw_llm_output_text}<<<")

    except Exception as e:
        llm_py_logger.error(f"Error calling local LLM service: {e}", exc_info=True)
        error_info = str(e)
        # 确保记录错误
        log_error_data = {
            "interaction_id": str(uuid.uuid4()), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "task_type": task_type + "_local_error", "user_query_for_task": user_query_for_log,
            "llm_input_messages": current_messages,
            "llm_input_original_prompt_if_string": original_prompt_for_log if isinstance(prompt, str) else None,
            "llm_parameters": llm_parameters_for_log,
            "raw_llm_output": f"Error: {error_info}. Partial raw output: {str(raw_llm_output_text)[:200] if raw_llm_output_text else 'N/A'}",
            "error_details": traceback.format_exc(), "application_version": application_version_for_log
        }
        await log_interaction_data(log_error_data)
        return None # 出错时返回None

    # 记录成功的调用
    log_success_data = {
        "interaction_id": str(uuid.uuid4()), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_type": task_type, "user_query_for_task": user_query_for_log,
        "llm_input_messages": current_messages,
        "llm_input_original_prompt_if_string": original_prompt_for_log if isinstance(prompt, str) else None,
        "llm_parameters": llm_parameters_for_log,
        "raw_llm_output": raw_llm_output_text, "application_version": application_version_for_log
    }
    await log_interaction_data(log_success_data)
    return raw_llm_output_text

# async def generate_cypher_query(user_question: str) -> Optional[str]: # kg_schema_description 参数可以移除了，因为它已包含在新的prompt函数中
#     llm_py_logger.info(f"Attempting to generate Cypher query (template-based) for: '{user_question}' via local service.")

#     messages_for_llm = get_cypher_generation_messages_with_templates(user_question)

#     cypher_stop_sequences = ['<|im_end|>', '```'] # 如果输出包含markdown的json块

#     llm_response_json_str = await call_llm_via_openai_api_local_only( 
#         prompt=messages_for_llm,
#         temperature=0.0, # 对于精确的JSON和Cypher生成，温度设为0
#         max_new_tokens=1024, # 允许足够的空间输出JSON和Cypher
#         stop_sequences=cypher_stop_sequences,
#         task_type="cypher_generation_template_based_local_service",
#         user_query_for_log=user_question,
#         model_name_for_log="qwen3_gguf_cypher_template_local"
#     )

#     if not llm_response_json_str:
#         llm_py_logger.warning(f"LLM call for Cypher (template-based) returned None or empty. User question: '{user_question}'")
#         return json.dumps({"status": "unable_to_generate", "query": "无法生成Cypher查询."}) # 始终返回JSON字符串

#     cleaned_json_str = llm_response_json_str.strip()
#     if cleaned_json_str.startswith("```json"):
#         cleaned_json_str = cleaned_json_str[len("```json"):].strip()
#     if cleaned_json_str.endswith("```"):
#         cleaned_json_str = cleaned_json_str[:-len("```")].strip()

#     try:

#         parsed_for_validation = json.loads(cleaned_json_str)
#         if isinstance(parsed_for_validation, dict) and \
#            "status" in parsed_for_validation and \
#            "query" in parsed_for_validation:
#             llm_py_logger.info(f"LLM returned valid JSON for Cypher (template-based): {cleaned_json_str}")
#             return cleaned_json_str
#         else:
#             llm_py_logger.warning(f"LLM output for Cypher (template-based) was JSON but not expected structure: {cleaned_json_str}")
#             return json.dumps({"status": "unable_to_generate", "query": "LLM输出JSON结构错误."})
#     except json.JSONDecodeError:
#         llm_py_logger.error(f"Failed to parse JSON response for Cypher (template-based): '{cleaned_json_str}'", exc_info=True)
#         # 如果不是有效的JSON，但包含"MATCH"，可能LLM直接输出了Cypher，尝试包装它
#         if "MATCH" in cleaned_json_str.upper() or "RETURN" in cleaned_json_str.upper():
#              llm_py_logger.warning("LLM output for Cypher (template-based) was not JSON but looks like Cypher, wrapping it.")
#              return json.dumps({"status": "success", "query": cleaned_json_str})
#         return json.dumps({"status": "unable_to_generate", "query": "LLM输出非JSON格式."})

async def generate_answer_from_context(
    user_query: str,
    context_str: str,
    prompt_builder: Optional[Callable[[str, str], List[Dict[str, str]]]] = None
) -> Optional[str]:
    """
    Generates an answer from context.
    V2: Can accept a dynamic prompt builder for specialized tasks like Table-QA.
    """
    from .rag_prompts import get_answer_generation_messages # 默认的 prompt builder

    # 如果没有提供特定的 prompt_builder，就使用默认的通用版本
    if prompt_builder is None:
        prompt_builder = get_answer_generation_messages
    
    llm_py_logger.info(f"Generating answer for query: '{user_query[:50]}...' using prompt builder: {prompt_builder.__name__}")
    
    # 使用 prompt_builder 来构建 messages
    messages = prompt_builder(user_query, context_str)
    
    final_answer = await call_llm_via_openai_api_local_only(
        prompt=messages, # <--- 将 messages 列表传递给 prompt 参数
        task_type=f"answer_generation_using_{prompt_builder.__name__}",
        model_name_for_log="qwen3_gguf_answer_gen_v2"
        # 其他参数如 temperature, max_tokens 会使用该函数的默认值
    )
    
    if final_answer and final_answer != NO_ANSWER_PHRASE_ANSWER_CLEAN:
        return final_answer
    elif final_answer: # 如果是 "无法找到信息"
        return final_answer
    else: # 如果返回None或空字符串
        llm_py_logger.warning("Answer generation returned None or empty string. Falling back to default no-answer phrase.")
        return NO_ANSWER_PHRASE_ANSWER_CLEAN

# async def generate_simulated_kg_query_response(user_query: str, kg_schema_description: str, kg_data_summary_for_prompt: str) -> Optional[str]:
#     prompt_str = f"""<|im_start|>system
# 你是一个知识图谱查询助手。你的任务是根据用户提出的问题、知识图谱Schema描述和图谱中的数据摘要，直接抽取出与问题最相关的1-2个事实片段作为答案。
# 只输出事实片段，不要解释，不要生成Cypher语句，不要包含任何额外对话或标记。
# 如果找不到直接相关的事实，请**直接且完整地**回答：“{NO_ANSWER_PHRASE_KG_WITH_STOP_TOKEN}”<|im_end|>
# <|im_start|>user
# 知识图谱Schema描述:
# {kg_schema_description}

# 知识图谱数据摘要: 
# {kg_data_summary_for_prompt}

# 用户问题: {user_query}<|im_end|>
# <|im_start|>assistant
# """
#     stop_sequences = ["<|im_end|>", UNIQUE_STOP_TOKEN]
#     return await call_llm_via_openai_api_local_only(
#         prompt=prompt_str,
#         temperature=0.5,
#         max_new_tokens=256,
#         stop_sequences=stop_sequences,
#         task_type="simulated_kg_query_response",
#         user_query_for_log=user_query
#     )


async def generate_clarification_question(original_query: str, uncertainty_reason: str) -> Optional[str]:
    llm_py_logger.info(f"调用LLM API生成澄清问题。原始查询: '{original_query}', 原因: '{uncertainty_reason}'")
    messages_for_llm = get_clarification_question_messages(original_query, uncertainty_reason)

    clarification_question_raw = await call_llm_via_openai_api_local_only(
        prompt=messages_for_llm,
        temperature=0.5,
        max_new_tokens=128,
        stop_sequences=['<|im_end|>'], # 对于Qwen系列，<|im_end|> 是一个常见的结束标记
        task_type="clarification_question_generation",
        user_query_for_log=original_query
    )
    
    if not clarification_question_raw or not clarification_question_raw.strip():
        llm_py_logger.warning("LLM未能生成澄清问题，返回默认提示。")
        return "抱歉，我不太理解您的意思，请您再具体说明一下。"  
    cleaned_question_from_llm = clarification_question_raw.strip()
    llm_py_logger.debug(f"LLM原始澄清输出 (清理后): '{cleaned_question_from_llm}'")
    potential_lines = cleaned_question_from_llm.splitlines()
    
    final_extracted_question = None

    for line in reversed(potential_lines):
        line_stripped = line.strip()
        if not line_stripped: # 跳过空行
            continue
        if line_stripped.endswith("？") or line_stripped.endswith("?"):
            if not (line_stripped.startswith("好的，") or \
                    line_stripped.startswith("首先，") or \
                    line_stripped.startswith("因此，") or \
                    line_stripped.startswith("所以，") or \
                    line_stripped.startswith("根据这个原因，") or \
                    "我需要生成一个" in line_stripped or \
                    "可能的澄清问题是" in line_stripped or \
                    "澄清问题应该是" in line_stripped or \
                    "接下来，" in line_stripped):
                final_extracted_question = line_stripped
                llm_py_logger.info(f"通过行分割和问号结尾提取到澄清问题: '{final_extracted_question}'")
                break 
        elif any(line_stripped.startswith(prefix) for prefix in ["请问您", "您对", "您具体指的是"]):
            final_extracted_question = line_stripped
            llm_py_logger.info(f"通过行分割和特定前缀提取到澄清问题: '{final_extracted_question}'")
            break

    if final_extracted_question:
        llm_py_logger.info(f"LLM成功生成并提取到最终澄清问题: {final_extracted_question}")
        return final_extracted_question
    else:
        potential_sentences = re.split(r'(?<=[。？！?])\s*', cleaned_question_from_llm)
        for sentence in reversed(potential_sentences):
            sentence_stripped = sentence.strip()
            if not sentence_stripped:
                continue
            if sentence_stripped.endswith("？") or sentence_stripped.endswith("?") or \
               any(sentence_stripped.startswith(prefix) for prefix in ["请问您", "您是想", "您具体指的是", "关于您提到的"]):
                if not (sentence_stripped.startswith("好的，") or \
                        sentence_stripped.startswith("首先，") or \
                        "我需要生成一个" in sentence_stripped or \
                        "可能的澄清问题是" in sentence_stripped): # 避免选择思考过程
                    final_extracted_question = sentence_stripped
                    llm_py_logger.info(f"通过句子分割和启发式规则提取到澄清问题: '{final_extracted_question}'")
                    break
        
        if final_extracted_question:
            llm_py_logger.info(f"LLM成功生成并提取到最终澄清问题 (后备逻辑): {final_extracted_question}")
            return final_extracted_question
        else:
            llm_py_logger.warning(f"未能通过所有启发式规则从LLM输出中提取明确的澄清问句。原始输出为: '{cleaned_question_from_llm}'。将返回默认澄清。")

            if len(cleaned_question_from_llm) < 70 and (cleaned_question_from_llm.endswith("？") or cleaned_question_from_llm.endswith("?")): # 70是个经验值
                 llm_py_logger.info(f"原始输出较短且以问号结尾，将其作为澄清问题返回: '{cleaned_question_from_llm}'")
                 return cleaned_question_from_llm
            return "抱歉，我不太理解您的意思，请您再具体说明一下。"


async def generate_clarification_options(original_query: str, uncertainty_reason: str) -> List[str]:
    prompt_str = f"""<|im_start|>system
你是一个智能助手，擅长根据用户查询的模糊性提供具体的澄清选项。
你的任务是根据用户原始查询和系统检测到的不确定性原因，生成3-5个具体的、可供用户选择的澄清选项。
每个选项都应该是一个简洁的短语或问题，帮助用户明确其意图。
你的回答必须是一个JSON数组（列表），其中每个元素是一个字符串（澄清选项）。
只输出JSON数组，不要包含任何其他解释、对话标记或代码块。

示例:
用户查询: "帮我预定机票。"
不确定性原因: "缺少出发城市、目的地、日期等信息。"
助手:
[
  "请问您想从哪个城市出发？",
  "请问您的目的地是哪里？",
  "请问您希望在哪一天出行？",
  "您有偏好的航空公司或舱位等级吗？"
]<|im_end|>
<|im_start|>user
用户原始查询: {original_query}
不确定性原因: {uncertainty_reason}

请生成澄清选项:<|im_end|>
<|im_start|>assistant
"""
    stop_sequences = ["<|im_end|>"]
    llm_py_logger.info(f"调用LLM API生成澄清选项 (Prompt长度: {len(prompt_str)} 字符)...")
    llm_output = await call_llm_via_openai_api_local_only(
        prompt=prompt_str,
        temperature=0.7,
        max_new_tokens=256,
        stop_sequences=stop_sequences,
        task_type="clarification_options_generation",
        user_query_for_log=original_query
    )

    options = []
    if llm_output:
        try:
            json_str = llm_output.strip()
            if json_str.startswith("```json"):
                json_str = json_str[len("```json"):].strip()
            if json_str.endswith("```"):
                json_str = json_str[:-len("```")].strip()
            
            parsed_options = json.loads(json_str)
            if isinstance(parsed_options, list) and all(isinstance(o, str) for o in parsed_options):
                options = parsed_options
                llm_py_logger.info(f"LLM成功生成 {len(options)} 个澄清选项。")
            else:
                llm_py_logger.warning(f"LLM生成的澄清选项JSON格式不符合预期 (不是字符串列表): {llm_output[:200]}...")
        except json.JSONDecodeError as e:
            llm_py_logger.error(f"解析LLM澄清选项JSON失败: {e}. 原始输出: {llm_output[:200]}...", exc_info=True)
        except Exception as e:
            llm_py_logger.error(f"处理LLM澄清选项时发生未知错误: {e}. 原始输出: {llm_output[:200]}...", exc_info=True)
    else:
        llm_py_logger.warning("LLM未能生成澄清选项。")
    
    if not options:
        options.append("请提供更多详细信息。")
    
    return options


INTENT_CLASSIFICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "clarification_needed": {"type": "boolean"},
        "reason": {"type": "string"}
    },
    "required": ["clarification_needed", "reason"]
}

async def generate_intent_classification(user_query: str) -> Dict[str, Any]:
    llm_py_logger.info(f"Generating intent classification for query: '{user_query[:100]}...' using Gemini.")
    
    # 针对Gemini优化的Prompt，强调直接输出JSON
    system_prompt_for_intent = f"""你是一个智能意图分类器。你的任务是分析用户查询，判断该查询是否清晰明确，或者是否存在歧义、信息不足导致需要进一步澄清。
如果查询包含具体的命名实体（如人名“张三”、项目名“项目X”、产品名“新产品A”等），并且问题是关于这些实体的特定信息（例如“张三的职位是什么？”、“项目X的截止日期是哪天？”、“新产品A的功能有哪些？”），则通常认为查询是清晰的，不需要澄清。
只有当查询缺少定位关键信息所必需的核心实体，或者询问的范围过于宽泛无法直接操作时，才需要澄清。

如果查询需要澄清，请说明原因。
你的【唯一输出】必须是一个严格符合以下结构的JSON对象，不要包含任何其他文本、解释或markdown标记:
{{
  "clarification_needed": true/false,
  "reason": "如果需要澄清，请简要说明原因；如果不需要，则为空字符串。"
}}

示例1 (需要澄清 - 信息不足):
用户查询: "帮我预定明天去上海的机票。"
助手 JSON 输出:
{{
  "clarification_needed": true,
  "reason": "缺少出发城市、具体时间（上午/下午/晚上）、舱位等级等信息。"
}}

示例2 (不需要澄清 - 清晰):
用户查询: "公司最新的销售额报告在哪里可以找到？"
助手 JSON 输出:
{{
  "clarification_needed": false,
  "reason": ""
}}
"""
    
    messages_for_gemini = [
        {"role": "system", "content": system_prompt_for_intent},
        {"role": "user", "content": f"用户查询: {user_query}"}
    ]

    # 从环境变量获取Gemini配置
    gemini_model_name = os.getenv("CLOUD_LLM_MODEL_NAME_FOR_LITELLM", "gemini/gemini-1.5-flash-latest")
    gemini_api_key = os.getenv("GEMINI_API_KEY") # 或者 GOOGLE_API_KEY
    proxy_url = os.getenv("LITELLM_PROXY_URL")

    litellm_params: Dict[str, Any] = {
        "model": gemini_model_name,
        "messages": messages_for_gemini,
        "api_key": gemini_api_key,
        "temperature": 0.1, 
        "max_tokens": 256,  # 意图分类的JSON输出通常较短
        # "response_format": {"type": "json_object"} # LiteLLM的Gemini集成可能尚不支持此参数，暂时注释
    }
    if proxy_url:
        # LiteLLM 的 proxy 参数期望一个字典，或者直接是一个字符串URL (取决于LiteLLM版本和具体实现)
        # 为保险起见，我们按文档常见的字典格式提供
        litellm_params["proxy"] = {
            "http": proxy_url,
            "https": proxy_url,
        }
        # 或者，如果您的LiteLLM版本支持直接传递字符串URL作为代理：
        # litellm_params["api_base"] = proxy_url # 这会将代理用于所有请求，如果Gemini也通过此代理
        # litellm_params["base_url"] = proxy_url # 有些版本用 base_url
        # 更通用的方式是设置环境变量 HTTP_PROXY 和 HTTPS_PROXY，LiteLLM通常会读取它们
        # 但为了显式，我们这里尝试通过参数传递给litellm.acompletion

    llm_py_logger.info(f"Calling Gemini (via LiteLLM) for intent classification. Model: {gemini_model_name}")
    debug_params = {k:v for k,v in litellm_params.items() if k not in ['messages', 'api_key']}
    llm_py_logger.debug(f"LiteLLM params for intent (excluding messages & api_key): {debug_params}")
    
    raw_gemini_output_text = None
    error_info_intent = None
    parsed_result_dict: Optional[Dict[str, Any]] = None # 用于存储最终解析结果

    try:
        response = await litellm.acompletion(**litellm_params)
        if response and response.choices and response.choices[0].message and response.choices[0].message.content:
            raw_gemini_output_text = response.choices[0].message.content.strip()
            llm_py_logger.info(f"Gemini intent classification raw output: {raw_gemini_output_text[:300]}...")
            
            # 尝试解析JSON (与之前的提取逻辑类似)
            json_str_candidate = raw_gemini_output_text
            # 1. 尝试从 markdown block 中提取
            markdown_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", json_str_candidate, re.DOTALL)
            if markdown_match:
                json_str_candidate = markdown_match.group(1)
                llm_py_logger.debug(f"Extracted JSON candidate from markdown: {json_str_candidate[:200]}...")
            
            # 2. 如果没有markdown，或者提取后仍然不是纯JSON，尝试直接解析或查找第一个 '{' 和最后一个 '}'
            try:
                parsed_result_dict = json.loads(json_str_candidate)
            except json.JSONDecodeError: # 如果直接解析失败
                first_brace = json_str_candidate.find('{')
                last_brace = json_str_candidate.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    json_str_candidate = json_str_candidate[first_brace : last_brace+1]
                    llm_py_logger.debug(f"Extracted JSON candidate by braces: {json_str_candidate[:200]}...")
                    try:
                        parsed_result_dict = json.loads(json_str_candidate)
                    except json.JSONDecodeError as e_json_brace:
                        error_info_intent = f"Failed to decode JSON from Gemini intent (braces): {e_json_brace}"
                        llm_py_logger.error(error_info_intent, exc_info=True)
                else: # 没有找到有效的花括号对
                    error_info_intent = "No valid JSON object found in Gemini intent output."
                    llm_py_logger.error(error_info_intent + f" Raw: {raw_gemini_output_text[:200]}")
            
            # 验证解析后的JSON结构
            if parsed_result_dict and isinstance(parsed_result_dict, dict) and \
               "clarification_needed" in parsed_result_dict and \
               "reason" in parsed_result_dict:
                llm_py_logger.info(f"Gemini successfully classified intent: {parsed_result_dict}")
                # 记录成功的调用
                log_data_intent = {
                    "interaction_id": str(uuid.uuid4()), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "task_type": "intent_classification_gemini", "user_query_for_task": user_query,
                    "llm_input_messages": messages_for_gemini, "llm_parameters": {k:v for k,v in litellm_params.items() if k not in ['messages', 'api_key', 'proxy']},
                    "raw_llm_output": raw_gemini_output_text, "application_version": "0.1.0_intent_gemini"
                }
                await log_interaction_data(log_data_intent)
                return parsed_result_dict
            else: # 解析成功但结构不对
                if parsed_result_dict: # 避免对None调用get
                    error_info_intent = f"Gemini intent output JSON structure mismatch. Parsed: {parsed_result_dict}"
                else: # parsed_result_dict 为 None (例如，花括号提取失败后)
                    error_info_intent = "Gemini intent output JSON structure mismatch (parsed_result_dict is None)."
                llm_py_logger.warning(error_info_intent)
        else: # response.choices[0].message.content 为空或不存在
            error_info_intent = "Gemini intent call returned empty or malformed response content."
            llm_py_logger.error(f"{error_info_intent} Full response object: {response}")

    except Exception as e_gemini_call:
        error_info_intent = f"Error calling Gemini for intent: {e_gemini_call}"
        llm_py_logger.error(error_info_intent, exc_info=True)

    # 如果执行到这里，说明出错了或者没有得到期望的JSON
    log_error_data_intent = {
        "interaction_id": str(uuid.uuid4()), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_type": "intent_classification_gemini_error", "user_query_for_task": user_query,
        "llm_input_messages": messages_for_gemini, "llm_parameters": {k:v for k,v in litellm_params.items() if k not in ['messages', 'api_key', 'proxy']},
        "raw_llm_output": raw_gemini_output_text or "N/A", "error_details": error_info_intent,
        "application_version": "0.1.0_intent_gemini"
    }
    await log_interaction_data(log_error_data_intent)
    
    llm_py_logger.warning(f"Gemini failed to generate valid intent classification, defaulting to no clarification needed. Error: {error_info_intent or 'Unknown reason'}")
    return {"clarification_needed": False, "reason": f"Intent classification by Gemini failed: {error_info_intent or 'Unknown reason'}"}

# --- 新增：用于提取实体和关系意图的函数 ---
# async def extract_entities_for_kg_query(user_question: str) -> Optional[ExtractedEntitiesAndRelationIntent]:
#     llm_py_logger.info(f"Attempting to extract entities and relation intent for KG query (with GBNF) from: '{user_question}'")

#     # --- 使用您在 test_gbnf_extraction.py 中验证成功的 One-Shot Prompt 构建逻辑 ---
#     one_shot_example = """
# --- 示例 ---
# 输入文本: "Alice在ACME公司担任工程师。"
# 输出JSON:
# {
#   "entities": [
#     {"text": "Alice", "label": "PERSON"},
#     {"text": "ACME公司", "label": "ORGANIZATION"},
#     {"text": "工程师", "label": "TASK"}
#   ],
#   "relations": [
#     {"head_entity_text": "Alice", "head_entity_label": "PERSON", "relation_type": "WORKS_AT", "tail_entity_text": "ACME公司", "tail_entity_label": "ORGANIZATION"}
#   ]
# }
# --- 任务开始 ---"""
    
#     # system_content 部分与您的测试脚本保持一致
#     system_content_for_prompt = (
#         f"你是一个严格的JSON知识图谱提取器。请根据用户提供的文本，严格按照示例格式，生成一个包含'entities'和'relations'的JSON对象。\n"
#         f"{one_shot_example}"
#     )

#     # user_content 部分也与您的测试脚本保持一致
#     user_content_for_prompt = (
#         f"输入文本: \"{user_question}\"\n" # 注意：这里用的是 user_question，而不是固定的 sample_text_to_extract
#         f"输出JSON:\n"
#     )

#     full_prompt_for_extraction = (
#         f"<|im_start|>system\n{system_content_for_prompt}<|im_end|>\n"
#         f"<|im_start|>user\n{user_content_for_prompt}<|im_end|>\n"
#         f"<|im_start|>assistant\n"
#     )
#     # --- Prompt 构建结束 ---

#     llm_response_str = await call_local_llm_with_gbnf(
#         full_prompt=full_prompt_for_extraction,
#         grammar_str=KG_EXTRACTION_GBNF_STRING, # 使用我们定义的GBNF字符串
#         temperature=0.1,
#         max_tokens=1024, # 与您的测试脚本一致
#         repeat_penalty=1.2, # 与您的测试脚本一致
#         stop_sequences=["<|im_end|>"], # Qwen的停止标记
#         task_type="kg_entity_relation_extraction_gbnf",
#         user_query_for_log=user_question,
#         model_name_for_log="qwen3_gguf_kg_ext_gbnf"
#     )

#     if not llm_response_str:
#         llm_py_logger.warning(f"LLM call for KG entity/relation extraction (GBNF) returned None or empty. User question: '{user_question}'")
#         return None

    # GBNF应该确保输出是有效的JSON，所以我们可以直接尝试解析
    try:
        # .strip() 以防万一有额外的空白被GBNF的 space 规则匹配但未被移除
        parsed_data = json.loads(llm_response_str.strip())
        extracted_info = ExtractedEntitiesAndRelationIntent(**parsed_data)
        llm_py_logger.info(f"Successfully parsed Pydantic model from GBNF LLM output: {extracted_info.model_dump_json(indent=2)}")
        return extracted_info
    except json.JSONDecodeError as e_json:
        llm_py_logger.error(f"Failed to decode JSON from GBNF LLM output: '{llm_response_str}'. Error: {e_json}", exc_info=True)
        return None
    except Exception as e_pydantic: # Catch Pydantic validation errors
        llm_py_logger.error(f"Failed to validate Pydantic model from GBNF LLM JSON: '{llm_response_str}'. Error: {e_pydantic}", exc_info=True)
        return None
    

async def call_local_llm_with_gbnf(
    llm_instance: Llama, # <--- 接受一个已加载的实例作为参数
    full_prompt: str,
    grammar_str: str,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    repeat_penalty: float = 1.2,
    stop_sequences: Optional[List[str]] = None,
    task_type: str = "gbnf_constrained_generation",
    user_query_for_log: Optional[str] = None,
    model_name_for_log: str = "local_qwen_gguf_gbnf",
    application_version_for_log: str = "0.1.0_gbnf"
) -> Optional[str]:
    llm_py_logger.info(f"Calling LOCAL LLM with GBNF for task: {task_type}. Prompt length: {len(full_prompt)}")

    raw_llm_output_text = None
    error_info = None
    
    try:
        compiled_grammar = LlamaGrammar.from_string(grammar_str)
        
        # 不再需要获取实例，直接使用传入的 llm_instance
        if llm_instance is None:
            raise ValueError("LLM GBNF instance is not available in the application context.")

        def _blocking_llm_call():
            response = llm_instance.create_completion(
                prompt=full_prompt,
                grammar=compiled_grammar,
                temperature=temperature,
                max_tokens=max_tokens,
                repeat_penalty=repeat_penalty,
                stop=stop_sequences or []
            )
            return response['choices'][0]['text']

        raw_llm_output_text = await asyncio.to_thread(_blocking_llm_call)
        llm_py_logger.info(f"GBNF Call: Raw LLM Output for task '{task_type}': >>>{raw_llm_output_text}<<<")

    except Exception as e:
        llm_py_logger.error(f"Error calling local LLM service with GBNF: {e}", exc_info=True)
        error_info = str(e)
        log_error_data = {
            "interaction_id": str(uuid.uuid4()), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "task_type": task_type + "_error", "user_query_for_task": user_query_for_log,
            "llm_input_prompt": full_prompt[:500] + "...",
            "llm_parameters": {"temperature": temperature, "max_tokens": max_tokens, "repeat_penalty": repeat_penalty, "stop": stop_sequences},
            "raw_llm_output": f"Error: {error_info}. Partial raw output: {str(raw_llm_output_text)[:200] if raw_llm_output_text else 'N/A'}",
            "error_details": traceback.format_exc(), "application_version": application_version_for_log
        }
        await log_interaction_data(log_error_data)
        return None

    log_success_data = {
        "interaction_id": str(uuid.uuid4()), "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_type": task_type, "user_query_for_task": user_query_for_log,
        "llm_input_prompt": full_prompt[:500] + "...",
        "llm_parameters": {"temperature": temperature, "max_tokens": max_tokens, "repeat_penalty": repeat_penalty, "stop": stop_sequences, "grammar_used": True},
        "raw_llm_output": raw_llm_output_text, "application_version": application_version_for_log
    }
    await log_interaction_data(log_success_data)
    return raw_llm_output_text



async def classify_user_intent(llm_instance: Llama, user_query: str) -> Optional[UserIntent]:
    """
    V4.2: Classifies the user's intent using a dedicated prompt and GBNF schema.
    """
    llm_py_logger.info(f"Classifying intent for query: '{user_query[:100]}...'")

    messages = get_intent_classification_messages(user_query)
    full_prompt = "".join([f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages])
    full_prompt += "<|im_start|>assistant\n"

    llm_response_str = await call_local_llm_with_gbnf(
        llm_instance=llm_instance,
        full_prompt=full_prompt,
        grammar_str=INTENT_CLASSIFICATION_GBNF_SCHEMA,
        temperature=0.0,
        max_tokens=256,
        task_type="intent_classification",
        user_query_for_log=user_query
    )

    if not llm_response_str:
        llm_py_logger.warning("LLM call for intent classification returned no response.")
        return None

    try:
        parsed_data = json.loads(llm_response_str.strip())
        intent_obj = UserIntent(**parsed_data)
        llm_py_logger.info(f"Successfully classified intent: {intent_obj.intent.value}")
        return intent_obj
    except (json.JSONDecodeError, ValidationError) as e:
        llm_py_logger.error(f"Failed to parse or validate intent classification JSON. Error: {e}. Raw: '{llm_response_str}'", exc_info=True)
        return None




async def generate_query_plan(llm_instance: Llama, user_query: str) -> Optional[RagQueryPlan]:
    """
    Analyzes the user query and generates a structured plan for retrieval,
    containing a core query string and a metadata filter.
    Uses GBNF for reliable JSON output.
    """
    # 导入我们新定义的 Prompt, GBNF Schema 和 Pydantic 模型
    from .rag_prompts import V2_PLANNING_PROMPT_TEMPLATE, V2_PLANNING_GBNF_SCHEMA
    from ..config.pydantic_models import RagQueryPlan

    llm_py_logger.info(f"Generating RAG query plan for: '{user_query[:100]}...'")

    # 1. 使用新的模板准备 Prompt
    full_prompt = V2_PLANNING_PROMPT_TEMPLATE.format(user_query=user_query)

    # 2. 调用带有 GBNF 约束的本地 LLM
    llm_response_str = await call_local_llm_with_gbnf(
        llm_instance=llm_instance,
        full_prompt=full_prompt,
        grammar_str=V2_PLANNING_GBNF_SCHEMA,
        temperature=0.0,  # 对于精确的JSON生成，使用零温度
        max_tokens=512,  # 为过滤器和查询提供足够的空间
        task_type="rag_query_planning", # 更新任务类型
        user_query_for_log=user_query,
        model_name_for_log="qwen3_gguf_rag_planner"
    )

    if not llm_response_str:
        llm_py_logger.warning("LLM query planner call returned no response. Falling back to simple query.")
        return RagQueryPlan(query=user_query, metadata_filter={})

    # 3. 解析结果并返回 Pydantic 对象
    try:
        # 清理LLM可能返回的 markdown 代码块标记
        cleaned_response = llm_response_str.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[len("```json"):].strip()
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3].strip()
            
        parsed_data = json.loads(cleaned_response)
        query_plan = RagQueryPlan(**parsed_data)
        
        llm_py_logger.info(f"Successfully generated query plan. Query: '{query_plan.query}', Filter: {query_plan.metadata_filter}")
        return query_plan
        
    except (json.JSONDecodeError, TypeError, ValidationError) as e:
        llm_py_logger.error(f"Failed to parse or validate LLM query plan. Error: {e}. Raw output: '{llm_response_str}'. Falling back to simple query.", exc_info=True)
        # 即使解析失败，也要保证RAG流程能继续，返回一个基础的计划
        return RagQueryPlan(query=user_query, metadata_filter={})
    

# --- V4 - Table QA Instruction Generation (Validated by PoC) ---

TABLE_QA_INSTRUCTION_PROMPT_TEMPLATE = """
# 指令
你是一个专门从用户问题中提取表格查询指令的AI。你的唯一任务是分析【用户问题】和【表格列名】，然后输出一个包含`row_identifier`和`column_identifier`的JSON对象。

## 规则
- `row_identifier`: 必须是用户问题中提到的、用于定位**某一行**的具体名称或值。
- `column_identifier`: 必须是用户问题中提到的、用于定位**某一列**的列名。

## 表格信息
【表格列名】: {column_names}

## 用户问题
【用户问题】: "{user_query}"

## 输出要求
请严格按照以下格式输出一个JSON对象，不要包含任何其他文字或解释。
```json
{{
  "row_identifier": "string, 用户问题中提到的具体行名",
  "column_identifier": "string, 用户问题中提到的具体列名"
}}
示例
用户问题: "笔记本电脑A的库存是多少？"
你的JSON输出:
{{
  "row_identifier": "笔记本电脑A",
  "column_identifier": "库存"
}}
你的JSON输出:
"""
# 使用主项目中已验证过的、最健壮的通用JSON GBNF Schema
from .rag_prompts import V2_PLANNING_GBNF_SCHEMA as TABLE_QA_INSTRUCTION_GBNF_SCHEMA


async def generate_table_lookup_instruction(llm_instance: Llama, user_query: str, table_column_names: List[str]) -> Optional[Dict[str, str]]:
    """
    Uses LLM to generate a structured instruction for table lookup, based on the user query and table columns.
    Returns a dictionary like {"row_identifier": "...", "column_identifier": "..."}.
    """
    llm_py_logger.info(f"Generating table lookup instruction for query: '{user_query}'")
    
    prompt = TABLE_QA_INSTRUCTION_PROMPT_TEMPLATE.format(
        column_names=", ".join(table_column_names),
        user_query=user_query
    )

    llm_response_str = await call_local_llm_with_gbnf(
        llm_instance=llm_instance,
        full_prompt=prompt,
        grammar_str=TABLE_QA_INSTRUCTION_GBNF_SCHEMA,
        temperature=0.0,
        max_tokens=256,
        task_type="table_qa_instruction_generation",
        user_query_for_log=user_query
    )
    if not llm_response_str:
        llm_py_logger.warning("LLM call for table instruction generation returned None.")
        return None
    try:
        instruction_json = json.loads(llm_response_str)
        if "row_identifier" in instruction_json and "column_identifier" in instruction_json:
            llm_py_logger.info(f"Successfully generated table lookup instruction: {instruction_json}")
            return instruction_json
        else:
            llm_py_logger.warning(f"Generated JSON is missing required keys: {instruction_json}")
            return None
    except json.JSONDecodeError:
        llm_py_logger.error(f"Failed to decode JSON from LLM for table instruction: {llm_response_str}")
        return None
    

async def generate_actionable_suggestion(llm_instance: Llama, user_query: str, failure_reason: str) -> Optional[str]:
    """
    Generates actionable suggestions for the user when the RAG system fails to find a direct answer.
    """
    llm_py_logger.info(f"Generating actionable suggestion for query: '{user_query}' due to: {failure_reason}")
    
    messages = get_suggestion_generation_messages(user_query, failure_reason)
    
    # 注意：这个函数生成的是自然语言，不需要 GBNF 约束，所以我们使用通用的 OpenAI API 格式调用。
    # 这里我们假设 `call_llm_via_openai_api_local_only` 内部最终会使用一个不需要 GBNF 的 Llama 实例，
    # 或者它是一个指向不同 LLM 服务的调用。为了保持当前架构，我们暂时不修改它。
    suggestion = await call_llm_via_openai_api_local_only(
        prompt=messages,
        temperature=0.7,
        max_new_tokens=512,
        task_type="suggestion_generation",
        user_query_for_log=user_query,
        model_name_for_log="qwen3_gguf_suggestion_gen"
    )

    if suggestion:
        cleaned_suggestion = re.sub(r"^(好的，|当然，|这里有一些建议：)\s*", "", suggestion.strip(), flags=re.IGNORECASE)
        return cleaned_suggestion
    
    return "您可以尝试换个问法，或检查相关文档是否已在知识库中。"


async def generate_expanded_queries(llm_instance: Llama, original_query: str) -> List[str]:
    """
    V3: Expands a single user query into multiple related sub-queries to enhance retrieval coverage.
    Uses a dedicated, stricter GBNF schema and improved parsing logic.
    """
    llm_py_logger.info(f"Attempting to generate expanded queries for: '{original_query}'")
    
    # --- 修改: 导入新的 Prompt 函数和 GBNF Schema ---
    from .rag_prompts import get_query_expansion_messages, QUERY_EXPANSION_GBNF_SCHEMA 

    messages = get_query_expansion_messages(original_query)
    
    full_prompt = "".join([f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages])
    full_prompt += "<|im_start|>assistant\n"

    llm_response_str = await call_local_llm_with_gbnf(
        llm_instance=llm_instance,
        full_prompt=full_prompt,
        grammar_str=QUERY_EXPANSION_GBNF_SCHEMA, # <--- 修改: 使用新的、更严格的 GBNF Schema
        temperature=0.6,
        max_tokens=1024,
        task_type="query_expansion_gbnf_v2", # 更新任务类型
        user_query_for_log=original_query
    )

    expanded_queries = []
    if llm_response_str:
        cleaned_response = llm_response_str.strip()
        try:
            parsed_data = json.loads(cleaned_response)
            
            # --- 修改: 从 {"queries": [...]} 结构中提取列表 ---
            if isinstance(parsed_data, dict) and "queries" in parsed_data and isinstance(parsed_data["queries"], list):
                potential_queries = parsed_data["queries"]
                expanded_queries = [q for q in potential_queries if isinstance(q, str) and q.strip()]
                llm_py_logger.info(f"Successfully extracted {len(expanded_queries)} queries from LLM response.")
            else:
                llm_py_logger.warning(f"LLM response for expansion was valid JSON but not the expected structure: {parsed_data}")

        except (json.JSONDecodeError, TypeError) as e:
            llm_py_logger.error(f"Failed to decode or process JSON for query expansion: '{cleaned_response}'. Error: {e}")

    # 确保原始查询总是第一个
    final_queries = [original_query]
    for q in expanded_queries:
        if q.strip() and q.strip() not in final_queries:
            final_queries.append(q.strip())
    
    llm_py_logger.info(f"Final list of queries ({len(final_queries)} total, deduplicated): {final_queries}")
    return final_queries


async def generate_document_summary(llm_instance: Llama, user_query: str, document_content: str) -> str:
    """
    Generates a single, concise summary sentence for a document based on the user query.
    """
    llm_py_logger.info(f"Generating summary for document based on query: '{user_query[:50]}...'")
    
    messages = get_document_summary_messages(user_query, document_content)
    
    # 摘要任务是自然语言生成，不需要 GBNF
    summary = await call_llm_via_openai_api_local_only(
        prompt=messages,
        temperature=0.1, # 摘要需要精确，低温度
        max_new_tokens=150, # 限制输出长度
        stop_sequences=["\n", "。", "."], # 一句话结束就停止
        task_type="document_summary_generation",
        user_query_for_log=user_query
    )

    if summary and summary.strip().lower() != "irrelevant":
        llm_py_logger.info("Successfully generated a relevant summary.")
        return summary.strip()
    
    # --- 核心降级逻辑 ---
    llm_py_logger.warning("Summary generation failed or returned irrelevant. "
                        "Falling back to using a snippet of the original document content.")
    # 返回原始文档内容的前N个字符作为后备摘要
    return document_content[:500] + "..."



async def extract_task_from_dialogue(
    llm_instance: Llama,
    user_query: str,
    llm_answer: str
) -> Optional[Dict[str, Any]]:
    """
    Analyzes a user query and the system's answer to extract a potential actionable task.
    Uses GBNF for reliable JSON output.

    Args:
        llm_instance: An initialized Llama model instance for GBNF-constrained generation.
        user_query: The original query from the user.
        llm_answer: The final answer generated by the RAG system.

    Returns:
        A dictionary containing the task information (e.g., {"task_found": true, "title": ..., "due_date": ...})
        or None if an error occurs.
    """
    llm_py_logger.info(f"Attempting to extract actionable task from dialogue...")

    # 1. 使用 prompt 模板构建完整的 prompt
    messages = get_task_extraction_messages(user_query, llm_answer)
    # 将 messages 列表转换为单个字符串 prompt
    full_prompt = "".join([f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages])
    full_prompt += "<|im_start|>assistant\n"

    # 2. 调用 GBNF 约束的 LLM
    llm_response_str = await call_local_llm_with_gbnf(
        llm_instance=llm_instance,
        full_prompt=full_prompt,
        grammar_str=V2_PLANNING_GBNF_SCHEMA,  # <--- 核心修正：使用正确的、已导入的Schema
        temperature=0.0,
        max_tokens=256,
        task_type="task_extraction_from_dialogue",
        user_query_for_log=user_query
    )

    if not llm_response_str:
        llm_py_logger.warning("LLM call for task extraction returned no response.")
        return None

    # 3. 解析 JSON 结果
    try:
        task_data = json.loads(llm_response_str.strip())
        if isinstance(task_data, dict) and "task_found" in task_data:
            llm_py_logger.info(f"Task extraction result: {task_data}")
            return task_data
        else:
            llm_py_logger.warning(f"Task extraction JSON is malformed: {task_data}")
            return None
    except json.JSONDecodeError:
        llm_py_logger.error(f"Failed to decode JSON from task extraction LLM output: '{llm_response_str}'")
        return None