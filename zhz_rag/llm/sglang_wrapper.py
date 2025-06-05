# zhz_agent/llm.py (renamed to sglang_wrapper.py as per typical module naming)
# or more accurately, this is the content for sglang_wrapper.py based on the inputs

import os
import httpx  # 用于异步HTTP请求
import json  # 用于处理JSON数据
import asyncio  # 用于 asyncio.to_thread
from typing import List, Dict, Any, Optional, Union # Added Union
from dotenv import load_dotenv
import traceback  # Ensure traceback is imported
from zhz_rag.utils.common_utils import log_interaction_data # 导入通用日志函数
from zhz_rag.config.constants import NEW_KG_SCHEMA_DESCRIPTION # <--- 确保导入这个常量
from zhz_rag.utils.common_utils import log_interaction_data # 导入通用日志函数
import logging
import re
import uuid  # 用于生成 interaction_id
from datetime import datetime, timezone  # 用于生成时间戳
import litellm # <--- 确保这个导入存在

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
NO_ANSWER_PHRASE_KG_WITH_STOP_TOKEN = f"{NO_ANSWER_PHRASE_KG_CLEAN}{UNIQUE_STOP_TOKEN}"

# Placeholder for the schema description. Replace with actual schema.
NEW_KG_SCHEMA_DESCRIPTION = """
{
  "node_labels": ["Person", "Project", "Task", "Document", "Region", "SalesAmount", "Product"],
  "relationship_types": ["WORKS_ON", "ASSIGNED_TO", "HAS_DOCUMENT", "HAS_SALES_AMOUNT", "RELATED_TO"],
  "node_properties": {
    "Person": [{"property": "name", "type": "STRING"}, {"property": "role", "type": "STRING"}],
    "Project": [{"property": "name", "type": "STRING"}, {"property": "status", "type": "STRING"}],
    "Task": [{"property": "name", "type": "STRING"}, {"property": "status", "type": "STRING"}, {"property": "priority", "type": "STRING"}],
    "Document": [{"property": "id", "type": "STRING"}, {"property": "title", "type": "STRING"}, {"property": "type", "type": "STRING"}],
    "Region": [{"property": "name", "type": "STRING"}],
    "SalesAmount": [{"property": "period", "type": "STRING"}, {"property": "numeric_amount", "type": "FLOAT"}, {"property": "unit", "type": "STRING"}],
    "Product": [{"property": "name", "type": "STRING"}, {"property": "category", "type": "STRING"}]
  },
  "relationship_properties": {},
  "output_format_guidance": {
    "description": "Your response MUST be a JSON object with two fields: 'status' and 'query'.",
    "status_field": {
      "description": "The 'status' field can be one of two values: 'success' or 'unable_to_generate'.",
      "success": "If you can generate a Cypher query, status should be 'success'.",
      "unable_to_generate": "If you cannot generate a Cypher query based on the question and schema, status should be 'unable_to_generate'."
    },
    "query_field": {
      "description": "The 'query' field contains the Cypher query as a string if status is 'success'.",
      "success_example": "MATCH (n) RETURN n LIMIT 1",
      "unable_to_generate_example": "无法生成Cypher查询."
    }
  },
  "examples": [
    {
      "User Question": "Who is task 'FixBug123' assigned to?",
      "Your EXACT Response": {
        "status": "success",
        "query": "MATCH (t:Task {name: 'FixBug123'})<-[:ASSIGNED_TO]-(p:Person) RETURN p.name AS assignedTo"
      }
    },
    {
      "User Question": "What is the color of the sky?",
      "Your EXACT Response": {
        "status": "unable_to_generate",
        "query": "无法生成Cypher查询."
      }
    }
  ]
}
"""

LLM_API_URL = os.getenv("SGLANG_API_URL", "http://localhost:8088/v1/chat/completions")

# 添加回这个函数，但它现在只被 generate_answer_from_context, generate_cypher_query 等调用
# 并且目标是本地的 OpenAI 兼容服务
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
            llm_py_logger.info(f"Local LLM Raw Output: {str(raw_llm_output_text)[:200]}...")
    # ... (省略错误处理和日志记录，与您之前的 call_llm_via_openai_api 类似) ...
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

async def generate_cypher_query(user_question: str, kg_schema_description: str = NEW_KG_SCHEMA_DESCRIPTION) -> Optional[str]:
    llm_py_logger.info(f"Attempting to generate Cypher query for: '{user_question}' via local service with GBNF + post-processing.")
    
    system_prompt_for_json_cypher = kg_schema_description 
    messages_for_llm = [
        {"role": "system", "content": system_prompt_for_json_cypher},
        {"role": "user", "content": f"用户问题: {user_question}"} 
    ]
    cypher_stop_sequences = ['<|im_end|>', '无法生成Cypher查询.', '```'] # 添加 '```' 以防模型生成 Markdown 后想继续

    llm_response_json_str = await call_llm_via_openai_api_local_only( 
        prompt=messages_for_llm,
        temperature=0.0, 
        max_new_tokens=1024, 
        stop_sequences=cypher_stop_sequences, # <--- 使用定义的 stop_sequences
        task_type="cypher_generation_final_attempt_local_service",
        user_query_for_log=user_question,
        model_name_for_log="qwen3_gguf_cypher_final_local"
    )

    if not llm_response_json_str:
        llm_py_logger.warning(f"LLM call for Cypher (local_final) returned None or empty. User question: '{user_question}'")
        return "无法生成Cypher查询."

    try:
        parsed_response = json.loads(llm_response_json_str)
        
        status = parsed_response.get("status")
        query_content = parsed_response.get("query")

        # Log the received JSON from local_llm_service
        log_data_received = {
            "interaction_id": str(uuid.uuid4()),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "task_type": "cypher_json_received_from_local_service",
            "user_query_for_task": user_question,
            "raw_json_from_local_service": llm_response_json_str,
            "parsed_status_from_local_service": status,
            "parsed_query_from_local_service": query_content
        }
        await log_interaction_data(log_data_received)

        if status == "success" and isinstance(query_content, str) and query_content.strip():
            llm_py_logger.info(f"Successfully extracted Cypher from local service JSON: {query_content}")
            return query_content.strip()
        elif status == "unable_to_generate" and query_content == "无法生成Cypher查询.":
            llm_py_logger.info(f"Local service indicated 'unable_to_generate' for: '{user_question}'")
            return "无法生成Cypher查询."
        else: # Should not happen if local_llm_service.py works as designed
            llm_py_logger.warning(f"Unexpected JSON structure from local_llm_service. Status: {status}, Query: {query_content}. Defaulting to 'unable'.")
            return "无法生成Cypher查询."
            
    except json.JSONDecodeError:
        llm_py_logger.error(f"Failed to parse JSON response from local_llm_service: '{llm_response_json_str}'", exc_info=True)
        return "无法生成Cypher查询."
    except Exception as e:
        llm_py_logger.error(f"Error processing response from local_llm_service: {e}", exc_info=True)
        return "无法生成Cypher查询."

async def generate_answer_from_context(user_query: str, context_str: str) -> Optional[str]: # context 参数名改为 context_str
    llm_py_logger.info(f"Generating answer for query: '{user_query[:100]}...' using provided context.")
    
    system_prompt_for_answer = f"""
你是一个非常严谨的AI问答助手。你的任务是根据【上下文信息】回答【用户问题】。

**核心指令：**

1.  **严格基于上下文：** 你的回答【必须只能】使用【上下文信息】中明确提供的文字。禁止进行任何形式的推断、联想、猜测或引入外部知识。
2.  **逐句核对：** 对于用户问题中的每一个信息点或子问题，你都必须在上下文中找到【直接对应的证据】才能回答。
3.  **明确关联性：** 如果用户问题试图关联上下文中的不同信息片段（例如，A是否与B有关），你必须在上下文中找到【明确陈述这种关联性的直接证据】。如果上下文中分别提到了A和B，但没有明确说明它们之间的关系，则视为无法关联。
4.  **处理无法回答的部分：**
    *   如果【上下文信息】完全不包含回答【用户问题】的任何相关信息，或者无法找到任何直接证据，请【只回答】：“{NO_ANSWER_PHRASE_ANSWER_CLEAN}”
    *   如果【用户问题】包含多个子问题，而【上下文信息】只能回答其中的一部分：
        *   请只回答你能找到直接证据的部分。
        *   对于上下文中没有直接证据支持的其他子问题，请明确指出：“关于[某子问题]，上下文中未提供明确信息。”
        *   不要对未提供信息的部分进行猜测或尝试回答。
5.  **简洁明了：** 回答要直接、简洁。

请严格遵守以上指令。
"""
    # 构造 messages 列表
    messages_for_llm = [
        {"role": "system", "content": system_prompt_for_answer},
        {"role": "user", "content": f"用户问题: {user_query}\n\n上下文信息:\n{context_str}"}
    ]

    raw_answer = await call_llm_via_openai_api_local_only(
        prompt=messages_for_llm, # <--- 传递 messages 列表
        temperature=0.05,
        max_new_tokens=1024, 
        stop_sequences=['<|im_end|>', UNIQUE_STOP_TOKEN], # 可以保留，以防万一
        task_type="answer_generation_from_context",
        user_query_for_log=user_query,
        model_name_for_log="qwen3_gguf_answer_gen"
    )
    
    if raw_answer and raw_answer.strip() and \
       raw_answer.strip() != "[[LLM_RESPONSE_MALFORMED_CHOICES_OR_MESSAGE]]" and \
       raw_answer.strip() != "[[CONTENT_NOT_FOUND]]":
        
        # local_llm_service.py 中的 post_process_llm_output 应该已经处理了 <think>
        # 但如果模型仍然可能输出 "根据目前提供的资料..." 之外的内容，
        # 而我们期望严格匹配，这里可以再加一层检查。
        # 对于答案生成，通常不需要像Cypher那样严格的后处理。
        final_answer = raw_answer.strip()
        if final_answer == NO_ANSWER_PHRASE_ANSWER_CLEAN:
            llm_py_logger.info("LLM indicated unable to answer from context.")
        return final_answer
    else:
        llm_py_logger.warning(f"Answer generation returned None, empty, or placeholder. Query: {user_query}")
        return NO_ANSWER_PHRASE_ANSWER_CLEAN # Fallback

async def generate_simulated_kg_query_response(user_query: str, kg_schema_description: str, kg_data_summary_for_prompt: str) -> Optional[str]:
    prompt_str = f"""<|im_start|>system
你是一个知识图谱查询助手。你的任务是根据用户提出的问题、知识图谱Schema描述和图谱中的数据摘要，直接抽取出与问题最相关的1-2个事实片段作为答案。
只输出事实片段，不要解释，不要生成Cypher语句，不要包含任何额外对话或标记。
如果找不到直接相关的事实，请**直接且完整地**回答：“{NO_ANSWER_PHRASE_KG_WITH_STOP_TOKEN}”<|im_end|>
<|im_start|>user
知识图谱Schema描述:
{kg_schema_description}

知识图谱数据摘要: 
{kg_data_summary_for_prompt}

用户问题: {user_query}<|im_end|>
<|im_start|>assistant
"""
    stop_sequences = ["<|im_end|>", UNIQUE_STOP_TOKEN]
    return await call_llm_via_openai_api_local_only(
        prompt=prompt_str,
        temperature=0.5,
        max_new_tokens=256,
        stop_sequences=stop_sequences,
        task_type="simulated_kg_query_response",
        user_query_for_log=user_query
    )

async def generate_expanded_queries(original_query: str) -> List[str]:
    prompt_str = f"""<|im_start|>system
你是一个专家查询分析师。根据用户提供的查询，生成3个不同但相关的子问题，以探索原始查询的不同方面。这些子问题将用于检索更全面的信息。
你的回答必须是一个JSON数组（列表），其中每个元素是一个字符串（子问题）。
只输出JSON数组，不要包含任何其他解释、对话标记或代码块。

示例:
用户查询: "公司年度财务报告和未来一年的预算规划"
助手:
[
  "公司最近的年度财务报告总结是什么？",
  "未来一年的详细预算规划有哪些主要构成？",
  "对比往年，公司财务状况有何显著变化？"
]<|im_end|>
<|im_start|>user
原始查询: {original_query}<|im_end|>
<|im_start|>assistant
"""
    stop_sequences = ["<|im_end|>"]
    
    llm_py_logger.info(f"调用LLM API进行查询扩展 (Prompt长度: {len(prompt_str)} 字符)...")
    llm_output = await call_llm_via_openai_api_local_only(
        prompt=prompt_str,
        temperature=0.7,
        max_new_tokens=512,
        stop_sequences=stop_sequences,
        task_type="query_expansion",
        user_query_for_log=original_query
    )
    expanded_queries = []
    if llm_output:
        try:
            json_str = llm_output.strip()
            if json_str.startswith("```json"):
                json_str = json_str[len("```json"):].strip()
            if json_str.endswith("```"):
                json_str = json_str[:-len("```")].strip()
            
            parsed_queries = json.loads(json_str)
            if isinstance(parsed_queries, list) and all(isinstance(q, str) for q in parsed_queries):
                expanded_queries = parsed_queries
                llm_py_logger.info(f"LLM成功生成 {len(expanded_queries)} 个扩展查询。")
            else:
                llm_py_logger.warning(f"LLM生成的扩展查询JSON格式不符合预期 (不是字符串列表): {llm_output[:200]}...")
        except json.JSONDecodeError as e:
            llm_py_logger.error(f"解析LLM扩展查询JSON失败: {e}. 原始输出: {llm_output[:200]}...", exc_info=True)
        except Exception as e:
            llm_py_logger.error(f"处理LLM扩展查询时发生未知错误: {e}. 原始输出: {llm_output[:200]}...", exc_info=True)
    else:
        llm_py_logger.warning("LLM未能生成扩展查询。")

    # Always include the original query
    if original_query not in expanded_queries:
        expanded_queries.append(original_query)
    return expanded_queries


async def generate_clarification_question(original_query: str, uncertainty_reason: str) -> Optional[str]:
    prompt_str = f"""<|im_start|>system
你是一个智能助手，擅长在理解用户查询时识别歧义并请求澄清。
你的任务是根据用户原始查询和系统检测到的不确定性原因，生成一个简洁、明确的澄清问题。
澄清问题应该帮助用户选择正确的意图，或者提供更多必要的信息。
只输出澄清问题，不要包含任何额外解释、对话标记或代码块。<|im_end|>
<|im_start|>user
用户原始查询: {original_query}
不确定性原因: {uncertainty_reason}

请生成一个澄清问题:<|im_end|>
<|im_start|>assistant
"""
    stop_sequences = ["<|im_end|>"]
    llm_py_logger.info(f"调用LLM API生成澄清问题 (Prompt长度: {len(prompt_str)} 字符)...")
    clarification_question = await call_llm_via_openai_api_local_only(
        prompt=prompt_str,
        temperature=0.5,
        max_new_tokens=128,
        stop_sequences=stop_sequences,
        task_type="clarification_question_generation",
        user_query_for_log=original_query
    )
    if not clarification_question or clarification_question.strip() == "":
        llm_py_logger.warning("LLM未能生成澄清问题，返回默认提示。")
        return "抱歉，我不太理解您的意思，请您再具体说明一下。"
    llm_py_logger.info(f"LLM成功生成澄清问题: {clarification_question.strip()}")
    return clarification_question.strip()

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