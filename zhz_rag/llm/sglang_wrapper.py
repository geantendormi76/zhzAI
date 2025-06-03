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

async def call_llm_via_openai_api(
    prompt: Union[str, List[Dict[str, str]]],
    temperature: float = 0.2,
    max_new_tokens: Optional[int] = 1024,
    stop_sequences: Optional[List[str]] = None,
    task_type: str = "unknown_llm_call",
    user_query_for_log: Optional[str] = None,
    model_name_for_log: str = "qwen3_gguf_via_openai_api", # Updated default model name
    application_version_for_log: str = "0.1.0"
) -> Optional[str]:

    llm_py_logger.info(f"Attempting to call LLM. Task: {task_type}, Target API: {LLM_API_URL}")

    current_messages: List[Dict[str, str]]
    original_prompt_for_log: str # For logging the original SGLang-style prompt if applicable

    if isinstance(prompt, str):
        original_prompt_for_log = prompt
        llm_py_logger.warning(f"call_llm_via_openai_api received a string prompt for task '{task_type}'. Attempting basic conversion to OpenAI messages. This is deprecated and may not be optimal.")
        current_messages = []
        if "<|im_start|>system" in original_prompt_for_log:
            parts = original_prompt_for_log.split("<|im_start|>")
            for part_content in parts:
                if not part_content.strip():
                    continue
                # Strip <|im_end|> and then content
                role_content_pair = part_content.split("<|im_end|>")[0].strip()
                if "\n" in role_content_pair: # Expects "role\ncontent"
                    role, message_content = role_content_pair.split("\n", 1)
                    current_messages.append({"role": role.strip().lower(), "content": message_content.strip()})
                else: # Fallback if no explicit role, assume user, or handle simple SGLang role prefix
                    # This part needs careful mapping from SGLang role prefixes if used without newline
                    # For now, a simple split for "system content" or "user content" without newline might be too naive.
                    # The provided split logic was `role, message_content = role_content_pair.split("\n", 1)`
                    # If it's just "system: message" or "system message" this is harder.
                    # The logic from the txt:
                    #   role_content_pair = part_content.split("<|im_end|>")[0].strip()
                    #   if "\n" in role_content_pair:
                    #       role, message_content = role_content_pair.split("\n", 1)
                    #       current_messages.append({"role": role.strip().lower(), "content": message_content.strip()})
                    #   else: # 可能是只有内容的user message
                    #       current_messages.append({"role": "user", "content": role_content_pair.strip()})
                    # This seems reasonable. Let's ensure role extraction.
                    # A more robust way for SGLang prompts:
                    # Check for "system\n", "user\n", "assistant\n" explicitly.
                    temp_role_content = role_content_pair.strip()
                    role_found = False
                    for r in ["system", "user", "assistant"]:
                        if temp_role_content.lower().startswith(r): # covers "system\ncontent" or "system content"
                            # check if role is followed by newline or space
                            if len(temp_role_content) > len(r) and (temp_role_content[len(r)] == '\n' or temp_role_content[len(r)] == ' ' or temp_role_content[len(r)] == ':'):
                                message_content = temp_role_content[len(r):].lstrip(' \n:')
                                current_messages.append({"role": r, "content": message_content.strip()})
                                role_found = True
                                break
                    if not role_found: # Default to user if no role prefix detected or if it's just content
                         current_messages.append({"role": "user", "content": temp_role_content})


            if not current_messages or (current_messages and current_messages[-1]["role"] == "assistant"):
                 current_messages.append({"role": "user", "content": "Continue."}) # Ensure last message is user if needed
        else: # If not SGLang format with <|im_start|>system, treat the whole string as a user message
            current_messages = [{"role": "user", "content": original_prompt_for_log}]
        llm_py_logger.debug(f"  Converted string prompt to messages: {json.dumps(current_messages, ensure_ascii=False, indent=2)}")
    elif isinstance(prompt, list):
        current_messages = prompt
        original_prompt_for_log = "Messages list provided directly."
        llm_py_logger.debug(f"  Received messages list directly: {json.dumps(current_messages, ensure_ascii=False, indent=2)}")
    else:
        llm_py_logger.error(f"Invalid 'prompt' argument type for call_llm_via_openai_api: {type(prompt)}")
        return None

    payload = {
        "model": model_name_for_log,
        "messages": current_messages,
        "temperature": temperature,
        "max_tokens": max_new_tokens,
    }
    if stop_sequences:
        payload["stop"] = stop_sequences

    headers = {"Content-Type": "application/json"}
    
    llm_parameters_for_log = {
        "model": model_name_for_log,
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "stop_sequences": stop_sequences
    }
    raw_llm_output_text = None

    try:
        llm_py_logger.info(f"Sending request to LLM API: {LLM_API_URL}")
        llm_py_logger.debug(f"Payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(LLM_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            response_json = response.json()
            
            if response_json.get("choices") and \
               isinstance(response_json["choices"], list) and \
               len(response_json["choices"]) > 0 and \
               response_json["choices"][0].get("message"):
                raw_llm_output_text = response_json["choices"][0]["message"].get("content")
                if raw_llm_output_text is None: # content might be null
                    raw_llm_output_text = "" 
            else:
                raw_llm_output_text = "[[LLM_RESPONSE_MALFORMED_CHOICES_OR_MESSAGE]]"
            
            llm_py_logger.info(f"LLM Raw Output (from API): {str(raw_llm_output_text)[:500]}...")

            interaction_log_data = {
                "interaction_id": str(uuid.uuid4()),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "task_type": task_type,
                "user_query_for_task": user_query_for_log,
                "llm_input_messages": current_messages,
                "llm_input_original_prompt_if_string": original_prompt_for_log if isinstance(prompt, str) else None,
                "llm_parameters": llm_parameters_for_log,
                "raw_llm_output": raw_llm_output_text,
                "application_version": application_version_for_log
            }
            await log_interaction_data(interaction_log_data)
            return raw_llm_output_text

    except httpx.HTTPStatusError as e:
        llm_py_logger.error(f"HTTPStatusError calling LLM API: {e}. Response: {e.response.text[:500]}", exc_info=True)
        error_info = f"HTTPStatusError: {e.response.status_code} - {e.response.text[:200]}"
    except httpx.RequestError as e:
        llm_py_logger.error(f"RequestError calling LLM API: {e}", exc_info=True)
        error_info = f"RequestError: {str(e)}"
    except json.JSONDecodeError as e:
        response_text_for_debug = "N/A"
        if 'response' in locals() and hasattr(response, 'text'):
            response_text_for_debug = response.text[:500]
        llm_py_logger.error(f"JSONDecodeError from LLM API: {e}. Raw response: {response_text_for_debug}", exc_info=True)
        error_info = f"JSONDecodeError: {str(e)}"
    except Exception as e:
        llm_py_logger.error(f"Unknown error in call_llm_via_openai_api: {type(e).__name__} - {e}", exc_info=True)
        error_info = f"Unknown error: {type(e).__name__} - {str(e)}"

    error_log_data = {
        "interaction_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_type": task_type + "_error",
        "user_query_for_task": user_query_for_log,
        "llm_input_messages": current_messages if 'current_messages' in locals() else [{"role":"system", "content": "Error: messages not formed"}],
        "llm_input_original_prompt_if_string": original_prompt_for_log if 'original_prompt_for_log' in locals() and isinstance(prompt, str) else None,
        "llm_parameters": llm_parameters_for_log,
        "raw_llm_output": f"Error: {error_info}. Partial raw output: {str(raw_llm_output_text)[:200] if raw_llm_output_text else 'N/A'}",
        "error_details": traceback.format_exc(),
        "application_version": application_version_for_log
    }
    await log_interaction_data(error_log_data)
    return None

async def generate_cypher_query(user_question: str, kg_schema_description: str = NEW_KG_SCHEMA_DESCRIPTION) -> Optional[str]:
    llm_py_logger.info(f"Attempting to generate Cypher query for: '{user_question}' via local service with GBNF + post-processing.")
    
    system_prompt_for_json_cypher = kg_schema_description 
    messages_for_llm = [
        {"role": "system", "content": system_prompt_for_json_cypher},
        {"role": "user", "content": f"用户问题: {user_question}"} 
    ]
    cypher_stop_sequences = ['<|im_end|>', '无法生成Cypher查询.', '```'] # 添加 '```' 以防模型生成 Markdown 后想继续

    llm_response_json_str = await call_llm_via_openai_api( 
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
你是一个AI问答助手。你的任务是根据【上下文信息】回答【用户问题】。

**核心指令：**

1.  **尝试直接回答：** 请首先仔细阅读【上下文信息】，如果其中包含能直接回答【用户问题】的内容，请用上下文中的信息直接、简洁地回答。
2.  **忠实原文：** 你的回答必须严格基于【上下文信息】，禁止加入任何外部知识或个人观点。
3.  **如果无法回答：** 如果你分析了【上下文信息】后，确认其中确实没有能回答【用户问题】的明确信息，那么请只回答以下这句话：
 "{NO_ANSWER_PHRASE_ANSWER_CLEAN}"
 不要添加任何其他解释、建议或反问。

请直接给出答案，或者只给出上述那句固定的“无法找到信息”的回复。
"""
    # 构造 messages 列表
    messages_for_llm = [
        {"role": "system", "content": system_prompt_for_answer},
        {"role": "user", "content": f"用户问题: {user_query}\n\n上下文信息:\n{context_str}"}
    ]

    raw_answer = await call_llm_via_openai_api(
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
    return await call_llm_via_openai_api(
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
    llm_output = await call_llm_via_openai_api(
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
    clarification_question = await call_llm_via_openai_api(
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
    llm_output = await call_llm_via_openai_api(
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
    llm_py_logger.info(f"Generating intent classification for query: '{user_query[:100]}...'")
    prompt_str = f"""<|im_start|>system
你是一个智能意图分类器。你的任务是分析用户查询，判断该查询是否清晰明确，或者是否存在歧义、信息不足导致需要进一步澄清。
如果查询包含具体的命名实体（如人名“张三”、项目名“项目X”、产品名“新产品A”等），并且问题是关于这些实体的特定信息（例如“张三的职位是什么？”、“项目X的截止日期是哪天？”、“新产品A的功能有哪些？”），则通常认为查询是清晰的，不需要澄清。
只有当查询缺少定位关键信息所必需的核心实体，或者询问的范围过于宽泛无法直接操作时，才需要澄清。

如果查询需要澄清，请说明原因。
你的回答必须是一个JSON对象，包含两个字段：
1. "clarification_needed": 布尔值，如果需要澄清则为 true，否则为 false。
2. "reason": 字符串，如果需要澄清，请简要说明原因；如果不需要，则为空字符串。

示例1 (需要澄清 - 信息不足):
用户查询: "帮我预定明天去上海的机票。"
助手:
{{
  "clarification_needed": true,
  "reason": "缺少出发城市、具体时间（上午/下午/晚上）、舱位等级等信息。"
}}

示例2 (不需要澄清 - 清晰):
用户查询: "公司最新的销售额报告在哪里可以找到？"
助手:
{{
  "clarification_needed": false,
  "reason": ""
}}

示例3 (需要澄清 - 实体不明确):
用户查询: "关于项目进展的文档。"
助手:
{{
  "clarification_needed": true,
  "reason": "项目名称不明确，文档类型（报告、计划、会议纪要等）不明确。"
}}

示例4 (不需要澄清 - 包含具体实体和明确问题):
用户查询: "张三参与了哪个项目？"
助手:
{{
  "clarification_needed": false,
  "reason": ""
}}

示例5 (不需要澄清 - 包含具体实体和明确问题):
用户查询: "华东区域2024年第一季度的销售额是多少？"
助手:
{{
  "clarification_needed": false,
  "reason": ""
}}

示例6 (需要澄清 - “公司”指默认上下文，但其余部分仍模糊):
用户查询: "公司的政策"
助手:
{{
  "clarification_needed": true,
  "reason": "未能明确指出是关于哪方面的公司政策（例如：人力资源、IT安全、财务等）。"
}}<|im_end|>
<|im_start|>user
用户查询: {user_query}<|im_end|>
<|im_start|>assistant
"""
    stop_sequences = ["<|im_end|>"]
    llm_py_logger.info(f"调用LLM API进行意图分类 (Prompt长度: {len(prompt_str)} 字符)...")

    llm_output = await call_llm_via_openai_api(
        prompt=prompt_str,
        temperature=0.01,
        max_new_tokens=256,
        stop_sequences=stop_sequences,
        task_type="intent_classification",
        user_query_for_log=user_query
    )

    if llm_output:
        try:
            json_str = llm_output.strip()
            if json_str.startswith("```json"):
                json_str = json_str[len("```json"):].strip()
            if json_str.endswith("```"):
                json_str = json_str[:-len("```")].strip()

            parsed_result = json.loads(json_str)
            if isinstance(parsed_result, dict) and \
               "clarification_needed" in parsed_result and \
               "reason" in parsed_result:
                llm_py_logger.info(f"LLM成功进行意图分类: {parsed_result}")
                return parsed_result
            else:
                llm_py_logger.warning(f"LLM生成的意图分类JSON格式不符合预期: {llm_output[:200]}...")
        except json.JSONDecodeError as e:
            llm_py_logger.error(f"解析LLM意图分类JSON失败: {e}. 原始输出: {llm_output[:200]}...", exc_info=True)
        except Exception as e:
            llm_py_logger.error(f"处理LLM意图分类时发生未知错误: {e}. 原始输出: {llm_output[:200]}...", exc_info=True)

    llm_py_logger.warning("LLM未能生成有效的意图分类结果，默认不需澄清。")
    return {"clarification_needed": False, "reason": "LLM分类失败或无结果。"}