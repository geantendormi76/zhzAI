# zhz_agent_project/llm.py
import os
import httpx # 用于异步HTTP请求
import json # 用于处理JSON数据
import asyncio # 用于 asyncio.to_thread
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import traceback # Ensure traceback is imported
from zhz_agent.utils import log_interaction_data # <--- 添加: 导入通用日志函数
import logging
import uuid # <--- 添加: 用于生成 interaction_id
from datetime import datetime, timezone # <--- 添加: 用于生成时间戳
load_dotenv() # 确保加载.env文件

# --- 日志文件配置 (新添加) ---
RAG_EVAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rag_eval_data')
# 确保目录存在，如果不存在则创建
if not os.path.exists(RAG_EVAL_DATA_DIR):
    try:
        os.makedirs(RAG_EVAL_DATA_DIR)
        print(f"Successfully created directory: {RAG_EVAL_DATA_DIR}")
    except Exception as e:
        print(f"Error creating directory {RAG_EVAL_DATA_DIR}: {e}. Please create it manually.")

def get_llm_log_filepath() -> str:
    """获取当前LLM交互日志文件的完整路径，按天分割。"""
    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return os.path.join(RAG_EVAL_DATA_DIR, f"llm_interactions_{today_str}.jsonl")

async def log_llm_interaction_to_jsonl(interaction_data: Dict[str, Any]):
    """
    将单条LLM交互数据异步追加到JSONL文件中。
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
# --- 结束日志文件配置 ---


# --- 为 llm.py 配置一个logger ---
llm_py_logger = logging.getLogger("LLMUtilsLogger") # 给一个独特的名字
llm_py_logger.setLevel(logging.INFO) # 可以设置为 INFO 或 DEBUG

# 防止重复添加handler，如果这个模块被多次导入或初始化
if not llm_py_logger.hasHandlers():
    _llm_console_handler = logging.StreamHandler() # 输出到控制台
    _llm_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s')
    _llm_console_handler.setFormatter(_llm_formatter)
    llm_py_logger.addHandler(_llm_console_handler)
    llm_py_logger.propagate = False # 通常不希望 utils 模块的日志传播到根

llm_py_logger.info("--- LLMUtilsLogger configured ---")
# --- 结束logger配置 ---


# --- 定义LLM在无法回答时应该输出的精确短语 (不含停止标记) ---
NO_ANSWER_PHRASE_ANSWER_CLEAN = "根据提供的信息，无法回答该问题。"
NO_ANSWER_PHRASE_KG_CLEAN = "从知识图谱中未找到直接相关信息。"

# --- 定义通用的唯一停止标记 ---
UNIQUE_STOP_TOKEN = "<|im_endofunable|>" # 报告建议的独特停止标记

# --- 定义LLM在Prompt中被要求输出的，包含特殊停止标记的短语 ---
NO_ANSWER_PHRASE_ANSWER_WITH_STOP_TOKEN = f"{NO_ANSWER_PHRASE_ANSWER_CLEAN}{UNIQUE_STOP_TOKEN}"
NO_ANSWER_PHRASE_KG_WITH_STOP_TOKEN = f"{NO_ANSWER_PHRASE_KG_CLEAN}{UNIQUE_STOP_TOKEN}"

# --- 配置 SGLang LLM API ---
SGLANG_API_URL = os.getenv("SGLANG_API_URL") 
# 确保SGLang服务正在运行，并且这个URL是正确的

async def call_sglang_llm(prompt: str,
                          temperature: float = 0.2,
                          max_new_tokens: Optional[int] = 1024,
                          stop_sequences: Optional[List[str]] = None,
                          # --- 添加以下参数用于日志记录 (添加) ---
                          task_type: str = "unknown", # 例如: "cypher_generation", "answer_generation"
                          user_query_for_log: Optional[str] = None, # 顶层用户查询
                          model_name_for_log: str = "qwen2.5-3b-instruct", # 假设默认
                          application_version_for_log: str = "0.1.0" # 假设版本
                          ) -> Optional[str]:

    llm_py_logger.debug(f"Attempting to call SGLang LLM. Task: {task_type}, Prompt (first 100 chars): {prompt[:100]}...") 

    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": stop_sequences if stop_sequences else [] 
        }
    }
    headers = {"Content-Type": "application/json"}
    # --- 为日志准备 llm_parameters (添加) ---
    llm_parameters_for_log = {
        "model": model_name_for_log,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "stop_sequences": stop_sequences if stop_sequences else []
    }
    # --- 结束日志参数准备 ---
    raw_llm_output_text = None # <--- (添加) 初始化变量
    processed_llm_output_text = None # <--- (添加) 初始化变量

    try:
        print(f"[LLM_DEBUG] Payload constructed. SGLANG_API_URL: {SGLANG_API_URL}")
        async with httpx.AsyncClient(timeout=120.0) as client: 
            print("[LLM_DEBUG] httpx.AsyncClient created. About to POST...")
            response = await client.post(SGLANG_API_URL, json=payload, headers=headers)
            print(f"[LLM_DEBUG] POST request sent. Status code: {response.status_code}")
            
            response.raise_for_status()
            print("[LLM_DEBUG] response.raise_for_status() passed.")

            response_json = response.json()
            print("[LLM_DEBUG] response.json() successful.")

            raw_llm_output_text = response_json.get("text", "[[TEXT_FIELD_NOT_FOUND]]").strip() # <--- (修改) 赋值给 raw_llm_output_text
            print(f"\n<<<<<<<<<< SGLANG LLM INPUT PROMPT START (call_sglang_llm) >>>>>>>>>>\n{prompt}\n<<<<<<<<<< SGLANG LLM INPUT PROMPT END >>>>>>>>>>\n")
            print(f"\n>>>>>>>>>> SGLANG LLM RAW OUTPUT TEXT START (call_sglang_llm) >>>>>>>>>>\n{raw_llm_output_text}\n>>>>>>>>>> SGLANG LLM RAW OUTPUT TEXT END >>>>>>>>>>\n")
            llm_py_logger.debug(f"Full SGLang raw response JSON: {response.text}")

            meta_info = response_json.get("meta_info", {})
            finish_reason = meta_info.get("finish_reason", {})
            print(f"[LLM_DEBUG] Meta info: {meta_info}, Finish reason: {finish_reason}")

            if finish_reason.get("type") == "stop" and finish_reason.get("matched") == UNIQUE_STOP_TOKEN:
                processed_llm_output_text = raw_llm_output_text.split(UNIQUE_STOP_TOKEN)[0].strip() if raw_llm_output_text and raw_llm_output_text != "[[TEXT_FIELD_NOT_FOUND]]" else NO_ANSWER_PHRASE_ANSWER_CLEAN # <--- (修改)
            else:
                processed_llm_output_text = raw_llm_output_text if raw_llm_output_text != "[[TEXT_FIELD_NOT_FOUND]]" else None # <--- (修改)

            # --- 日志记录 (修改为调用通用函数) ---
            interaction_log_data = {
                # timestamp_utc 和 interaction_id 会由 log_interaction_data 自动添加 (如果未提供)
                "task_type": task_type, # task_type 已经作为参数传入 call_sglang_llm
                "user_query": user_query_for_log,
                "llm_input_prompt": prompt,
                "llm_parameters": llm_parameters_for_log,
                "raw_llm_output": raw_llm_output_text,
                "processed_llm_output": processed_llm_output_text,
                "application_version": application_version_for_log
            }
            await log_interaction_data(interaction_log_data) # <--- 修改: 调用通用函数
            # --- 结束日志记录 ---

            return processed_llm_output_text # <--- (修改) 返回处理后的文本

    except httpx.HTTPStatusError as e: # 更具体的HTTP错误
        print(f"[LLM_DEBUG] httpx.HTTPStatusError: {e}")
        print(f"[LLM_DEBUG] Response content causing status error: {e.response.text[:500]}")
        traceback.print_exc()
        return None
    except httpx.RequestError as e:
        print(f"[LLM_DEBUG] httpx.RequestError: {e}")
        traceback.print_exc()
        return None
    except json.JSONDecodeError as e:
        response_text_for_debug = "N/A"
        if 'response' in locals() and hasattr(response, 'text'):
            response_text_for_debug = response.text[:500]
        print(f"[LLM_DEBUG] json.JSONDecodeError: {e}. Raw response text: {response_text_for_debug}")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"[LLM_DEBUG] Unknown error in call_sglang_llm: {type(e).__name__} - {e}")
        traceback.print_exc()
        # --- 即使异常也要尝试记录 (修改为调用通用函数) ---
        error_log_data = { # <--- (修改) 组装数据字典
            "task_type": task_type,
            "user_query": user_query_for_log,
            "llm_input_prompt": prompt if 'prompt' in locals() else "Prompt not available",
            "llm_parameters": llm_parameters_for_log if 'llm_parameters_for_log' in locals() else {},
            "raw_llm_output": f"Error: {type(e).__name__} - {str(e)}. Partial raw output: {raw_llm_output_text if 'raw_llm_output_text' in locals() and raw_llm_output_text else 'N/A'}",
            "processed_llm_output": None,
            "error_details": traceback.format_exc(), # 添加错误详情
            "application_version": application_version_for_log
        }
        await log_interaction_data(error_log_data) # <--- 修改: 调用通用函数
        # --- 结束异常时的日志记录 ---
        return None
    
async def generate_answer_from_context(user_query: str, context: str) -> Optional[str]:
    
    prompt = f"""<|im_start|>system
你是一个精确且忠实的问答助手。你的任务是严格根据用户问题和下面提供的【上下文信息】，生成一个准确、简洁、直接的答案。

**【回答核心准则 - 必须严格遵守！】**

1.  **【忠实于上下文 - 最高优先级】**: 你的所有回答都**必须完全基于**提供的【上下文信息】。**绝对不允许**依赖任何外部知识、个人观点或进行任何形式的推测。如果上下文中没有明确支持的信息，就必须指出信息未知或无法回答。

2.  **【优先采纳精确信息】**: 如果上下文中包含以“【知识图谱精确信息】”标记的片段，这代表了高置信度的结构化事实。请**优先直接采纳这些信息**来回答用户问题的对应部分。忽略该片段的原始得分。

3.  **【处理组合问题/多方面问题】**:
    *   仔细分析用户问题，识别其中是否包含多个子问题或期望获取多个方面的信息（例如，通过“和”、“以及”连接，或一个问句包含多个疑问点）。
    *   对于每个识别出的子问题或信息点，请在【上下文信息】中独立查找答案。
    *   将找到的各个方面的答案清晰地组织起来。如果某个方面的信息在上下文中找不到，**必须明确指出该方面信息未知或未提供**，例如：“关于[问题的某个方面]，上下文中未提供具体信息。” **不要因为部分信息缺失而放弃回答其他能找到信息的部分。**

4.  **【处理数值与计算】**:
    *   如果用户问题需要对上下文中的数字进行计算（如求和、求差、平均等），并且上下文中包含可用于计算的明确数字（注意识别数字和单位），请执行计算并在答案中清晰展示计算结果和原始数据。
    *   例如，如果上下文说“A产品销售额50万元，B产品销售额30万元”，用户问“AB产品总销售额”，你应该回答“A产品销售额为50万元，B产品销售额为30万元，总销售额为80万元。”
    *   如果上下文中的数字不清晰、单位不一致难以换算，或者进行计算的条件不足，请列出原始数据并说明无法精确计算。

5.  **【信息不足的最终判断】**: 如果综合运用以上规则后，【上下文信息】中**完全没有**与用户问题相关的任何信息，或者完全无法回答用户问题的任何一个方面，则**必须直接且完整地**输出：“{NO_ANSWER_PHRASE_ANSWER_WITH_STOP_TOKEN}”

**【输出要求】**: 你的回答必须是纯文本，直接给出答案本身，语气应客观、专业。避免使用对话标记或不必要的解释性语句，除非是为了说明信息来源或缺失情况。
<|im_end|>
<|im_start|>user
用户问题: {user_query}

上下文信息:
{context}
<|im_end|>
<|im_start|>assistant
"""
    stop_sequences = ["<|im_end|>", UNIQUE_STOP_TOKEN] 
    # --- 修改调用 call_sglang_llm 以传递日志参数 (修改) ---
    return await call_sglang_llm(
        prompt,
        temperature=0.05,
        max_new_tokens=512,
        stop_sequences=stop_sequences,
        task_type="answer_generation", # <--- 添加
        user_query_for_log=user_query, # <--- 添加
        # model_name_for_log 和 application_version_for_log 可以使用 call_sglang_llm 的默认值
    )

async def generate_simulated_kg_query_response(user_query: str, kg_schema_description: str, kg_data_summary_for_prompt: str) -> Optional[str]:
    """
    模拟Text-to-Cypher和知识图谱查询。
    这里我们不实际生成Cypher，而是让LLM直接根据问题和KG描述生成“事实片段”。
    """
    prompt = f"""<|im_start|>system
你是一个知识图谱查询助手。你的任务是根据用户提出的问题、知识图谱Schema描述和图谱中的数据摘要，直接抽取出与问题最相关的1-2个事实片段作为答案。
只输出事实片段，不要解释，不要生成Cypher语句，不要包含任何额外对话或标记。
如果找不到直接相关的事实，请**直接且完整地**回答：“{NO_ANSWER_PHRASE_KG_WITH_STOP_TOKEN}”
<|im_end|>
<|im_start|>user
知识图谱Schema描述:
{kg_schema_description}

知识图谱数据摘要: 
{kg_data_summary_for_prompt}

用户问题: {user_query}
<|im_end|>
<|im_start|>assistant
"""
    stop_sequences = ["<|im_end|>", UNIQUE_STOP_TOKEN]
    # --- 修改调用 call_sglang_llm (修改) ---
    return await call_sglang_llm(
        prompt,
        temperature=0.5,
        max_new_tokens=256,
        stop_sequences=stop_sequences,
        task_type="simulated_kg_query_response", # <--- 添加
        user_query_for_log=user_query # <--- 添加
    )

async def generate_expanded_queries(original_query: str) -> List[str]:
    """
    利用LLM从用户原始查询生成多个语义相关但表述各异的子问题或扩展查询。
    """
    prompt = f"""<|im_start|>system
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
]
<|im_end|>
<|im_start|>user
原始查询: {original_query}
<|im_end|>
<|im_start|>assistant
"""
    stop_sequences = ["<|im_end|>"]
    
    print(f"调用SGLang LLM API进行查询扩展 (Prompt长度: {len(prompt)} 字符)...")
    # --- 修改调用 call_sglang_llm (修改) ---
    llm_output = await call_sglang_llm(
        prompt,
        temperature=0.7,
        max_new_tokens=512,
        stop_sequences=stop_sequences,
        task_type="query_expansion", # <--- 添加
        user_query_for_log=original_query # <--- 添加
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
                print(f"LLM成功生成 {len(expanded_queries)} 个扩展查询。")
            else:
                print(f"LLM生成的扩展查询JSON格式不符合预期 (不是字符串列表): {llm_output[:200]}...")
        except json.JSONDecodeError as e:
            print(f"解析LLM扩展查询JSON失败: {e}. 原始输出: {llm_output[:200]}...")
        except Exception as e:
            print(f"处理LLM扩展查询时发生未知错误: {e}. 原始输出: {llm_output[:200]}...")
    else:
        print("LLM未能生成扩展查询。")

    expanded_queries.append(original_query) 
    return expanded_queries

async def generate_cypher_query(user_question: str, kg_schema_description: str) -> Optional[str]:
    """
    利用LLM将自然语言问题转换为Neo4j Cypher查询语句。
    """
    prompt = f"""<|im_start|>system
你是顶级Neo4j Cypher查询生成专家。你的任务是根据用户问题和严格提供的【知识图谱Schema】，生成一个【语法正确】、【逻辑合理】且【高效】的Cypher查询。

**【核心指令与约束 - 必须严格遵守！】**

1.  **【Schema绝对绑定 - 最高优先级】**:
    *   你生成的Cypher查询中所有用到的【节点标签】、【关系类型】、【属性名称】及其对应的【数据类型】，都**必须严格存在于**下面提供的 "知识图谱Schema描述" 中。
    *   在构建查询的每一步，都要反复与Schema核对。**严禁臆断、猜测或使用任何Schema中未明确定义的元素。**
    *   **属性名称的大小写和确切拼写必须与Schema完全一致。**
    *   **关系类型的名称和方向必须与Schema完全一致。** 例如，如果Schema定义为 `(Person)-[:WORKS_ON]->(Project)`，则查询中不能是 `(Project)-[:WORKS_ON]->(Person)`，除非Schema中也定义了反向关系。

2.  **【纯净输出格式 - 严格要求】**:
    *   如果能生成有效查询，你的回答**必须只包含纯粹的Cypher查询语句本身**。
    *   如果根据问题和Schema无法生成有效的Cypher查询（例如，问题超出了Schema表达能力，或问题本身逻辑不通），则**必须只输出固定的短语：“无法生成Cypher查询。”**
    *   **绝对禁止**在有效的Cypher语句前后添加任何前缀（如“Cypher查询: ”）、后缀、解释、注释、markdown标记（如 ```cypher ```）或任何其他多余的文本。

3.  **【属性与值的使用】**:
    *   当在`WHERE`子句中对属性进行匹配时，确保值的类型与Schema中定义的属性类型一致。例如，如果`name`是字符串，则匹配 `name: '张三'`；如果`year`是数字，则匹配 `year: 2023`。
    *   对于数值计算（如`SUM`, `AVG`），**必须**使用Schema中明确指定的数字类型属性（例如，`SalesAmount`节点的 `numeric_amount`）。

4.  **【查询构建逻辑指引】**:
    *   **实体识别**: 准确识别用户问题中的核心实体及其在Schema中对应的节点标签和属性。
    *   **关系路径**: 基于问题和Schema构建清晰的`MATCH`路径。
    *   **条件过滤**: 使用`WHERE`子句添加必要的过滤条件。
    *   **结果返回**: 使用`RETURN`子句指定需要返回的信息，并用`AS`为返回的列指定清晰、合法的别名（字母或下划线开头）。
    *   **多步查询**: 对于需要关联多个信息点的问题，合理使用`WITH`传递中间结果。
    *   **聚合**: 如需统计或汇总，正确使用`COUNT()`, `SUM()`, `COLLECT()`等聚合函数。

**【知识图谱Schema描述】**:
{kg_schema_description}

**【查询示例 - 严格基于上述Schema】**:

*   用户问题: "张三参与了哪个项目？"
    Cypher查询: MATCH (p:Person {{name: '张三'}})-[:WORKS_ON]->(proj:Project) RETURN proj.name AS projectName

*   用户问题: "华东区域2024年第一季度的销售额是多少？"
    Cypher查询: MATCH (r:Region {{name: '华东'}})-[:HAS_SALES_AMOUNT]->(sa:SalesAmount {{period: '2024年第一季度'}}) RETURN sa.numeric_amount AS salesAmount, sa.unit AS salesUnit

*   用户问题: "查询所有产品的名称。"
    Cypher查询: MATCH (prod:Product) RETURN prod.name AS productName

*   用户问题: "项目X有哪些人参与？"
    Cypher查询: MATCH (p:Person)-[:WORKS_ON]->(proj:Project {{name: '项目X'}}) RETURN p.name AS participantName

*   用户问题: "2024年第一季度所有区域的总销售额是多少？"
    Cypher查询: MATCH (r:Region)-[:HAS_SALES_AMOUNT]->(sa:SalesAmount {{period: '2024年第一季度'}}) RETURN sum(sa.numeric_amount) AS totalSales, sa.unit AS commonUnit LIMIT 1 
    (此查询假设所有相关销售额的单位是相同的，并取第一个出现的单位作为代表)

*   用户问题: "与新产品A相关的文档ID是什么？"
    Cypher查询: MATCH (p:Product {{name: '新产品A'}})-[:RELATED_TO]->(d:Document) RETURN d.id AS documentId

*   用户问题: "公司CEO是谁？" (假设Schema中没有CEO信息)
    Cypher查询: 无法生成Cypher查询。

现在，请根据以下用户问题和上述Schema及规则生成Cypher查询。
<|im_end|>
<|im_start|>user
用户问题: {user_question}
<|im_end|>
<|im_start|>assistant
"""
    # ... (后续的LLM调用和后处理逻辑与您当前版本一致) ...
    stop_sequences = ["<|im_end|>", "无法生成Cypher查询。"] 
    
    # --- 修改调用 call_sglang_llm (修改) ---
    cypher_query = await call_sglang_llm(
        prompt,
        temperature=0.0,
        max_new_tokens=400,
        stop_sequences=stop_sequences,
        task_type="cypher_generation", # <--- 添加
        user_query_for_log=user_question # <--- 添加
        # 可以考虑将 kg_schema_description 的版本号或哈希值也记录下来
        # 例如，在 interaction_log_data 中添加 "kg_schema_version": "v4_final"
    )
    
    if not cypher_query or cypher_query.strip() == "" or cypher_query.strip().lower() == "无法生成cypher查询。" or "无法生成cypher查询" in cypher_query.strip().lower():
        llm_py_logger.warning(f"LLM未能生成有效Cypher查询或明确表示无法生成，原始输出: '{cypher_query}'") # <--- 使用 llm_py_logger
        return "无法生成Cypher查询。" 
    
    processed_query = cypher_query.strip()
    prefixes_to_remove = ["cypher查询:", "cypher query:", "```cypher", "```sql", "```"]
    for prefix in prefixes_to_remove:
        if processed_query.lower().startswith(prefix.lower()):
            processed_query = processed_query[len(prefix):].strip()
    
    if processed_query.endswith("```"):
        processed_query = processed_query[:-len("```")].strip()

    if not processed_query:
        llm_py_logger.warning(f"LLM生成的Cypher查询在移除常见前缀/后缀后为空，原始输出: '{cypher_query}'") # <--- 使用 llm_py_logger

    llm_py_logger.info(f"LLM成功生成Cypher查询 (后处理后): {processed_query}")
    return processed_query

async def generate_clarification_question(original_query: str, uncertainty_reason: str) -> Optional[str]:
    """
    利用LLM根据用户原始查询和不确定性原因，生成一个具体的澄清问题。
    """
    prompt = f"""<|im_start|>system
你是一个智能助手，擅长在理解用户查询时识别歧义并请求澄清。
你的任务是根据用户原始查询和系统检测到的不确定性原因，生成一个简洁、明确的澄清问题。
澄清问题应该帮助用户选择正确的意图，或者提供更多必要的信息。
只输出澄清问题，不要包含任何额外解释、对话标记或代码块。
<|im_end|>
<|im_start|>user
用户原始查询: {original_query}
不确定性原因: {uncertainty_reason}

请生成一个澄清问题:
<|im_end|>
<|im_start|>assistant
"""
    stop_sequences = ["<|im_end|>"]
    print(f"调用SGLang LLM API生成澄清问题 (Prompt长度: {len(prompt)} 字符)...")
    # --- 修改调用 call_sglang_llm (修改) ---
    clarification_question = await call_sglang_llm(
        prompt,
        temperature=0.5,
        max_new_tokens=128,
        stop_sequences=stop_sequences,
        task_type="clarification_question_generation", # <--- 添加
        user_query_for_log=original_query # <--- 添加
    )
    if not clarification_question or clarification_question.strip() == "":
        print("LLM未能生成澄清问题，返回默认提示。")
        return "抱歉，我不太理解您的意思，请您再具体说明一下。"
    print(f"LLM成功生成澄清问题: {clarification_question}")
    return clarification_question

async def generate_clarification_options(original_query: str, uncertainty_reason: str) -> List[str]:
    """
    利用LLM根据用户原始查询和不确定性原因，生成多个具体的澄清选项。
    """
    prompt = f"""<|im_start|>system
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
]
<|im_end|>
<|im_start|>user
用户原始查询: {original_query}
不确定性原因: {uncertainty_reason}

请生成澄清选项:
<|im_end|>
<|im_start|>assistant
"""
    stop_sequences = ["<|im_end|>"]
    print(f"调用SGLang LLM API生成澄清选项 (Prompt长度: {len(prompt)} 字符)...")
    # --- 修改调用 call_sglang_llm (修改) ---
    llm_output = await call_sglang_llm(
        prompt,
        temperature=0.7,
        max_new_tokens=256,
        stop_sequences=stop_sequences,
        task_type="clarification_options_generation", # <--- 添加
        user_query_for_log=original_query # <--- 添加
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
                print(f"LLM成功生成 {len(options)} 个澄清选项。")
            else:
                print(f"LLM生成的澄清选项JSON格式不符合预期 (不是字符串列表): {llm_output[:200]}...")
        except json.JSONDecodeError as e:
            print(f"解析LLM澄清选项JSON失败: {e}. 原始输出: {llm_output[:200]}...")
        except Exception as e:
            print(f"处理LLM澄清选项时发生未知错误: {e}. 原始输出: {llm_output[:200]}...")
    else:
        print("LLM未能生成澄清选项。")
    
    if not options:
        options.append("请提供更多详细信息。")
    
    return options

async def generate_intent_classification(user_query: str) -> Dict[str, Any]:
    """
    利用LLM对用户查询进行意图分类，判断是否需要澄清。
    返回一个字典，包含 'clarification_needed' (bool) 和 'reason' (str)。
    """
    prompt = f"""<|im_start|>system
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
}}
<|im_end|>
<|im_start|>user
用户查询: {user_query}
<|im_end|>
<|im_start|>assistant
"""
    stop_sequences = ["<|im_end|>"]
    print(f"调用SGLang LLM API进行意图分类 (Prompt长度: {len(prompt)} 字符)...")

    # --- 修改调用 call_sglang_llm (修改) ---
    llm_output = await call_sglang_llm(
        prompt,
        temperature=0.1,
        max_new_tokens=256,
        stop_sequences=stop_sequences,
        task_type="intent_classification", # <--- 添加
        user_query_for_log=user_query # <--- 添加
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
                print(f"LLM成功进行意图分类: {parsed_result}")
                return parsed_result
            else:
                print(f"LLM生成的意图分类JSON格式不符合预期: {llm_output[:200]}...")
        except json.JSONDecodeError as e:
            print(f"解析LLM意图分类JSON失败: {e}. 原始输出: {llm_output[:200]}...")
        except Exception as e:
            print(f"处理LLM意图分类时发生未知错误: {e}. 原始输出: {llm_output[:200]}...")

    print("LLM未能生成有效的意图分类结果，默认不需澄清。")
    return {"clarification_needed": False, "reason": "LLM分类失败或无结果。"}
