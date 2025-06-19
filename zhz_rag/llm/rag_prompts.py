# /home/zhz/zhz_rag/llm/rag_prompts.py
from typing import List, Dict, Any
from zhz_rag.config.constants import NEW_KG_SCHEMA_DESCRIPTION


# 可以将 NO_ANSWER_PHRASE_ANSWER_CLEAN 也移到这里，或者从 constants.py 导入
NO_ANSWER_PHRASE_ANSWER_CLEAN = "根据目前提供的资料，我无法找到关于您问题的明确信息。" # 保持与 llm_interface.py 一致

def get_answer_generation_messages(user_query: str, context_str: str) -> List[Dict[str, str]]:
    """
    构建用于从上下文中生成答案的LLM输入messages。
    V2: 优化了无法回答时的响应，使其更具建设性。
    """
    system_prompt_for_answer = f"""
你是一个非常严谨、客观且专业的AI问答助手。你的核心任务是根据【上下文信息】回答【用户问题】。

**核心指令与行为准则：**

1.  **【绝对忠实于上下文】**: 你的回答【必须且只能】使用【上下文信息】中明确提供的文字和事实。严禁进行任何形式的推断、联想、猜测或引入外部知识。

2.  **【处理无法回答的情况】**:
    *   **如果上下文信息充足**: 请直接、简洁地回答用户问题。
    *   **如果上下文信息不足以回答**:
        *   **第一步**: 明确告知用户无法找到信息。请使用这个固定的句子开头：“{NO_ANSWER_PHRASE_ANSWER_CLEAN}”。
        *   **第二步**: 在此基础上，尝试分析用户问题的意图，并给出一句简短、有帮助的建议，引导用户进行下一步操作。
        *   **示例1**: 如果用户询问特定文件的信息但未找到，你可以建议：“您或许可以检查文件名是否正确，或确认该文件是否已在知识库中。”
        *   **示例2**: 如果用户询问一个需要特定知识但未找到答案的问题，你可以建议：“您可能需要查阅相关的专业文档或联系相关领域的专家。”
        *   **最终输出**: 将第一步和第二步合并成一个流畅的回答。例如：“根据目前提供的资料，我无法找到关于您问题的明确信息。您或许可以检查文件名是否正确，或确认该文件是否已在知识库中。”

3.  **【答案风格：专业、简洁】**:
    *   直接针对用户问题，避免不必要的寒暄。
    *   语言表达专业、客观。

/no_think

请严格遵守以上指令，以最高的准确性和忠实度来完成回答。
"""
    messages = [
        {"role": "system", "content": system_prompt_for_answer},
        {"role": "user", "content": f"用户问题: {user_query}\n\n上下文信息:\n{context_str}"}
    ]
    return messages


def get_clarification_question_messages(original_query: str, uncertainty_reason: str) -> List[Dict[str, str]]:
    """
    构建用于生成澄清问题的LLM输入messages。
    """
    system_prompt_for_clarification = f"""你的【唯一任务】是根据用户提供的【用户原始查询】和【不确定性原因】，生成一个【简洁、明确、友好且直接的澄清问句】。
    

**【严格的输出要求】**
*   你的【最终且唯一】的输出【必须】是这个澄清问句本身。
*   【绝对禁止】输出任何思考过程、解释、前缀、后缀或任何与澄清问句无关的文字。
*   澄清问句本身不应包含用户的原始查询或不确定性原因的复述。
/no_think


**示例：**

<example>
  <user_original_query>帮我查查天气</user_original_query>
  <uncertainty_reason>缺少地点信息</uncertainty_reason>
  <assistant_clarification_question>请问您想查询哪个城市的天气呢？</assistant_clarification_question>
</example>

<example>
  <user_original_query>分析一下销售数据</user_original_query>
  <uncertainty_reason>用户没有说明具体想对销售数据做什么操作，例如是汇总、筛选还是查找特定记录。</uncertainty_reason>
  <assistant_clarification_question>请问您希望对销售数据进行哪种具体操作，例如汇总统计、筛选特定条件，还是查找某些记录？</assistant_clarification_question>
</example>

<example>
  <user_original_query>给我推荐一些关于人工智能的书籍</user_original_query>
  <uncertainty_reason>用户没有说明偏好的人工智能子领域或书籍类型（入门/进阶/技术/哲学等）。</uncertainty_reason>
  <assistant_clarification_question>您对人工智能的哪个子领域或什么类型的书籍（如入门、技术实践、哲学探讨等）更感兴趣？</assistant_clarification_question>
</example>

<example>
  <user_original_query>我们公司的年假政策是怎么样的？</user_original_query>
  <uncertainty_reason>缺少公司名称，无法定位到具体的年假政策文档。</uncertainty_reason>
  <assistant_clarification_question>请问您的公司全称是什么？</assistant_clarification_question>
</example>

<example>
  <user_original_query>处理一下这个文件。</user_original_query>
  <uncertainty_reason>用户没有说明要对文件进行何种处理，也没有指明是哪个文件。</uncertainty_reason>
  <assistant_clarification_question>请问您希望对哪个文件进行什么具体操作呢？</assistant_clarification_question>
</example>
"""
    user_content = f"""用户原始查询: {original_query}
不确定性原因: {uncertainty_reason}

你应该输出的澄清问句:""" # 改为“澄清问句”

    messages = [
        {"role": "system", "content": system_prompt_for_clarification},
        {"role": "user", "content": user_content}
    ]
    return messages

# --- 精简的Cypher模板定义 (只保留一个核心模板) ---
SIMPLIFIED_CYPHER_TEMPLATES = [
    {
        "id": "template_find_entity_attributes_by_text_label",
        "description": "根据提供的实体文本和实体标签，查找该实体的所有基本属性。",
        "template": "MATCH (n:ExtractedEntity {{text: $entity_text, label: $entity_label}}) RETURN n.text AS text, n.label AS label, n.id_prop AS id_prop LIMIT 1",
        "params_needed": ["entity_text", "entity_label"]
    }
]

def get_cypher_generation_messages_with_templates(user_question: str) -> List[Dict[str, str]]: # 函数名保持一致
    """
    构建用于（基于【单个指定模板】）生成Cypher查询的LLM输入messages。
    这个版本用于测试模型对单个模板的参数提取能力。
    """
    
    # 在这个测试版本中，我们假设总是使用第一个（也是唯一一个）模板
    selected_template = SIMPLIFIED_CYPHER_TEMPLATES[0]
    
    template_description_for_prompt = f"""你将使用以下Cypher查询模板：
Template ID: {selected_template['id']}
Description: {selected_template['description']}
Cypher Structure: {selected_template['template']}
Parameters Needed: {', '.join(selected_template['params_needed'])}
"""

    system_prompt_for_cypher = f"""你是一个精确的参数提取助手。你的任务是根据用户问题，为下面提供的【唯一Cypher查询模板】提取参数，并构建一个Cypher查询。

**【图谱Schema核心部分参考】**
(你主要关注 `:ExtractedEntity` 节点及其属性: `text`, `label`, `id_prop`。其中 `label` 的常见值是 "PERSON", "ORGANIZATION", "TASK"。)
{NEW_KG_SCHEMA_DESCRIPTION} 
# ^^^ Schema描述已包含输出JSON格式 {{"status": "success/unable_to_generate", "query": "..."}} 的指导，请严格遵循该JSON输出格式。

**【当前需要填充的Cypher查询模板】**
{template_description_for_prompt}

**【你的任务与输出要求】**
1.  仔细分析【用户问题】，理解其核心查询意图。
2.  判断该意图是否与提供的【当前需要填充的Cypher查询模板】描述相符。
3.  如果相符：
    a.  从【用户问题】中提取填充该模板所需的所有【Parameters Needed】。确保参数值与Schema中的实体文本和标签格式相符（例如，标签应为大写 "PERSON", "ORGANIZATION", "TASK"）。
    b.  将提取的参数值替换到模板的Cypher语句中（例如，`$entity_text` 替换为提取到的实体名）。
    c.  最终输出一个JSON对象，格式为：`{{"status": "success", "query": "填充好参数的Cypher语句"}}`。
4.  如果不相符（例如，用户问题意图与模板描述不符，或无法从问题中提取到模板所需的所有关键参数）：
    a.  最终输出一个JSON对象，格式为：`{{"status": "unable_to_generate", "query": "无法生成Cypher查询."}}`。
5.  【绝对禁止】输出任何除了上述指定JSON对象之外的文本、解释或思考过程。


**【处理示例】**
<example>
  <user_question>我想知道张三的详细信息。</user_question>
  <assistant_output_json>{{
    "status": "success",
    "query": "MATCH (n:ExtractedEntity {{text: '张三', label: 'PERSON'}}) RETURN n.text AS text, n.label AS label, n.id_prop AS id_prop LIMIT 1"
  }}</assistant_output_json>
</example>
<example>
  <user_question>项目Alpha的文档编写任务是什么？</user_question>
  <assistant_output_json>{{
    "status": "success",
    "query": "MATCH (n:ExtractedEntity {{text: '项目alpha的文档编写任务', label: 'TASK'}}) RETURN n.text AS text, n.label AS label, n.id_prop AS id_prop LIMIT 1"
  }}</assistant_output_json>
</example>
<example>
  <user_question>法国的首都是哪里？</user_question>
  <assistant_output_json>{{
    "status": "unable_to_generate",
    "query": "无法生成Cypher查询."
  }}</assistant_output_json>
</example>
"""
    user_content = f"用户问题: {user_question}"

    messages = [
        {"role": "system", "content": system_prompt_for_cypher},
        {"role": "user", "content": user_content}
    ]
    return messages

# --- 新增：实体与关系意图提取的提示词生成函数 ---
def get_entity_relation_extraction_messages(user_question: str) -> List[Dict[str, str]]:
    """
    构建用于从用户查询中提取核心实体和关系意图的LLM输入messages。
    目标是输出一个符合 ExtractedEntitiesAndRelationIntent Pydantic 模型结构的纯净JSON对象。
    这个版本的Prompt极度强调JSON输出格式。
    """
    import re
    match = re.search(r'label\s*:\s*STRING\s*\(实体类型。\s*允许的值\s*:\s*("([^"]+)"(?:,\s*"([^"]+)")*)\)', NEW_KG_SCHEMA_DESCRIPTION)
    allowed_entity_labels_str = "PERSON, ORGANIZATION, TASK, DOCUMENT, PROJECT, REGION, PRODUCT, OTHER"
    if match:
        labels_group = match.group(1)
        extracted_labels = re.findall(r'"([^"]+)"', labels_group)
        if extracted_labels:
            allowed_entity_labels_str = ", ".join(extracted_labels)
            if "OTHER" not in extracted_labels:
                 allowed_entity_labels_str += ", OTHER"

    # --- V3 "最最严格" Prompt ---
    system_prompt_for_entity_extraction = f"""<|im_start|>system
USER_QUERY_TO_PROCESS:
{user_question}

TASK: Analyze USER_QUERY_TO_PROCESS. Output ONLY a valid JSON object.
NO EXPLANATIONS. NO EXTRA TEXT. NO MARKDOWN. JUST JSON.

JSON_OUTPUT_SCHEMA:
{{
  "entities": [
    {{"text": "string, extracted entity text from USER_QUERY_TO_PROCESS", "label": "string, entity type from: [{allowed_entity_labels_str}], or OTHER"}}
  ],
  "relation_hint": "string, relation described in USER_QUERY_TO_PROCESS, or empty string"
}}

RULES:
1. Max 2 entities in "entities" array. If none, "entities" is `[]`.
2. "label" MUST be from the provided list or "OTHER".
3. If no relation_hint, value is `""`.
4. If USER_QUERY_TO_PROCESS yields no entities or relation, output: `{{"entities": [], "relation_hint": ""}}`

YOUR_VALID_JSON_OUTPUT_ONLY:<|im_end|>""" # <--- 结尾引导更加直接

    messages = [
        {"role": "system", "content": system_prompt_for_entity_extraction}
    ]
    return messages


# 用于Dagster流水线中，从单个文本块抽取KG的提示词
KG_EXTRACTION_SINGLE_CHUNK_PROMPT_TEMPLATE_V1 = """
你是一个信息抽取助手。请从以下提供的文本中抽取出所有的人名(PERSON)、组织机构名(ORGANIZATION)和任务(TASK)实体。
同时，请抽取出以下两种关系：
1. WORKS_AT (当一个人在一个组织工作时，例如：PERSON WORKS_AT ORGANIZATION)
2. ASSIGNED_TO (当一个任务分配给一个人时，例如：TASK ASSIGNED_TO PERSON)

请严格按照以下JSON格式进行输出，不要包含任何额外的解释或Markdown标记：
{{
  "entities": [
    {{"text": "实体1原文", "label": "实体1类型"}},
    ...
  ],
  "relations": [
    {{"head_entity_text": "头实体文本", "head_entity_label": "头实体类型", "relation_type": "关系类型", "tail_entity_text": "尾实体文本", "tail_entity_label": "尾实体类型"}},
    ...
  ]
}}
如果文本中没有可抽取的实体或关系，请返回一个空的对应列表 (例如 {{"entities": [], "relations": []}})。

文本：
"{text_to_extract}"
""" # <--- 末尾引导词已删除

# 用于Dagster流水线中，从一批文本块抽取KG的提示词
KG_EXTRACTION_BATCH_PROMPT_TEMPLATE_V1 = """
你是一个信息抽取助手。你的任务是处理下面编号的【文本块列表】中的每一个文本块。
对于列表中的【每一个文本块】，请独立地抽取出所有的人名(PERSON)、组织机构名(ORGANIZATION)和任务(TASK)实体，以及它们之间可能存在的WORKS_AT和ASSIGNED_TO关系。

【输出格式要求】:
你的最终输出【必须】是一个JSON数组。
这个数组中的每个元素都对应输入【文本块列表】中相应顺序的文本块的抽取结果。
每个元素的结构【必须】严格符合以下JSON Schema：
{{
  "entities": [ 
    {{"text": "实体原文", "label": "实体类型"}}, 
    ... 
  ],
  "relations": [
    {{"head_entity_text": "头实体文本", "head_entity_label": "头实体类型", "relation_type": "关系类型", "tail_entity_text": "尾实体文本", "tail_entity_label": "尾实体类型"}},
    ...
  ]
}}
如果某个文本块中沒有可抽取的实体或关系，则其在JSON数组中对应的元素应为：{{"entities": [], "relations": []}}。
【绝对禁止】在最终的JSON数组之外包含任何其他文本、解释或Markdown标记。

【待处理的文本块列表】:
{formatted_text_block_list}
""" 

# GBNF for Knowledge Graph Extraction (proven stable with Qwen3-1.7B and create_completion)
KG_EXTRACTION_GBNF_STRING = r"""
root ::= "{" space "\"entities\"" ":" space entities "," space "\"relations\"" ":" space relations "}"

space ::= ([ \t\n\r])*
string ::= "\"" (char)* "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})

entities ::= "[" space (entities-item ("," space entities-item)*)? space "]"
entities-item ::= "{" space "\"text\"" ":" space string "," space "\"label\"" ":" space string "}"

relations ::= "[" space (relations-item ("," space relations-item)*)? space "]"
relations-item ::= "{" space "\"head_entity_text\"" ":" space string "," space "\"head_entity_label\"" ":" space string "," space "\"relation_type\"" ":" space string "," space "\"tail_entity_text\"" ":" space string "," space "\"tail_entity_label\"" ":" space string "}"
"""

# --- 新增：用于合并查询扩展和KG实体提取的Prompt和GBNF ---

# GBNF for the combined task
COMBINED_EXPANSION_KG_GBNF_STRING = r"""
root ::= "{" space "\"expanded_queries\"" ":" space string-array "," space "\"extracted_entities_for_kg\"" ":" space kg-extraction-object "}"

space ::= ([ \t\n\r])*
string ::= "\"" (char)* "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})

string-array ::= "[" space (string ("," space string)*)? space "]"

kg-extraction-object ::= "{" space "\"entities\"" ":" space entities "," space "\"relations\"" ":" space relations "}"

entities ::= "[" space (entities-item ("," space entities-item)*)? space "]"
entities-item ::= "{" space "\"text\"" ":" space string "," space "\"label\"" ":" space string "}"

relations ::= "[" space (relations-item ("," space relations-item)*)? space "]"
relations-item ::= "{" space "\"head_entity_text\"" ":" space string "," space "\"head_entity_label\"" ":" space string "," space "\"relation_type\"" ":" space string "," space "\"tail_entity_text\"" ":" space string "," space "\"tail_entity_label\"" ":" space string "}"
"""

# Prompt Template for the combined task
COMBINED_EXPANSION_KG_PROMPT_TEMPLATE = """<|im_start|>system
You are a highly efficient and structured data processing AI. Your task is to perform two actions based on the user's query and produce a single, valid JSON object as output.

**Actions to Perform:**
1.  **Query Expansion:** Generate 3 diverse, related sub-questions to explore different facets of the original query.
2.  **KG Entity/Relation Extraction:** Extract key entities (PERSON, ORGANIZATION, TASK, etc.) and their relationships (e.g., WORKS_AT, ASSIGNED_TO) from the original query for knowledge graph searching.

**Output Format (Strict JSON):**
You MUST output a single, valid JSON object that strictly adheres to the following structure. Do NOT include any explanations, markdown, or any text outside of the JSON object.

```json
{{
  "expanded_queries": [
    "string // Expanded question 1",
    "string // Expanded question 2",
    "string // Expanded question 3"
  ],
  "extracted_entities_for_kg": {{
    "entities": [
      {{"text": "string // Extracted entity text", "label": "string // Entity type"}}
    ],
    "relations": [
      {{
        "head_entity_text": "string",
        "head_entity_label": "string",
        "relation_type": "string",
        "tail_entity_text": "string",
        "tail_entity_label": "string"
      }}
    ]
  }}
}}
Example:
User Query: "Who is responsible for the Project Alpha documentation, and where do they work?"
Expected JSON Output:
{{
  "expanded_queries": [
    "Who is the primary contact for Project Alpha documentation?",
    "Which organization employs the person responsible for Project Alpha documentation?",
    "What are the recent updates or status of the Project Alpha documentation?"
  ],
  "extracted_entities_for_kg": {{
    "entities": [
      {{
        "text": "Project Alpha",
        "label": "PROJECT"
      }},
      {{
        "text": "documentation",
        "label": "TASK"
      }}
    ],
    "relations": []
  }}
}}
<|im_end|>
<|im_start|>user
User Query: "{user_query}"
Output JSON:
<|im_end|>
<|im_start|>assistant
"""

# --- START: 覆盖 GBNF 字符串 ---
# GBNF for the combined task with advanced filtering
COMBINED_PLANNING_GBNF_STRING = r"""
root ::= "{" space "\"expanded_queries\"" ":" space string-array "," space "\"extracted_entities_for_kg\"" ":" space kg-extraction-object ("," space "\"metadata_filter\"" ":" space (json-object | "null"))? space "}"

# --- Common Definitions ---
space ::= ([ \t\n\r])*
string ::= "\"" (char)* "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
boolean ::= "true" | "false"

# --- JSON structure for filter ---
json-value ::= string | number | boolean | json-object | json-array | "null"
json-object ::= "{" space (pair ("," space pair)*)? space "}"
pair ::= string ":" space json-value
json-array ::= "[" space (json-value ("," space json-value)*)? space "]"

# --- Specific parts for our main object ---
string-array ::= "[" space (string ("," space string)*)? space "]"

kg-extraction-object ::= "{" space "\"entities\"" ":" space entities "," space "\"relations\"" ":" space relations "}"
entities ::= "[" space (entities-item ("," space entities-item)*)? space "]"
entities-item ::= "{" space "\"text\"" ":" space string "," space "\"label\"" ":" space string "}"
relations ::= "[" space (relations-item ("," space relations-item)*)? space "]"
relations-item ::= "{" space "\"head_entity_text\"" ":" space string "," space "\"head_entity_label\"" ":" space string "," space "\"relation_type\"" ":" space string "," space "\"tail_entity_text\"" ":" space string "," space "\"tail_entity_label\"" ":" space string "}"
"""
# --- END: 覆盖 GBNF 字符串 ---


# V3: 更强大的规划器，支持层次化元数据过滤
# Prompt Template for the combined task - V3 with Advanced Metadata Filter
COMBINED_PLANNING_PROMPT_TEMPLATE = """<|im_start|>system
You are a highly efficient and structured query planner. Your task is to analyze the user's query and produce a single, valid JSON object to guide the retrieval process.

**Actions to Perform:**
1.  **Query Expansion:** Generate 2-3 diverse, related sub-questions.
2.  **KG Entity Extraction:** Extract key entities (PERSON, ORGANIZATION, TASK, etc.).
3.  **Advanced Metadata Filter Generation:** Analyze the query for specific contextual constraints like filenames, chapters, sections, or page numbers. Construct a filter object accordingly.

**Available Metadata Fields for Filtering:**
- `filename`: string (e.g., "report.pdf")
- `page`: integer (e.g., 5)
- `paragraph_type`: string (e.g., "table", "title", "narrative_text")
- `title_hierarchy_1`: string (e.g., "第一章 介绍")
- `title_hierarchy_2`: string (e.g., "财务分析")
- `title_hierarchy_...`: string (for deeper levels)

**Output Format (Strict JSON):**
You MUST output a single, valid JSON object. Do NOT include any explanations, markdown, or any text outside of the JSON object.

```json
{{
  "expanded_queries": [
    "string"
  ],
  "extracted_entities_for_kg": {{
    "entities": [
      {{"text": "string", "label": "string"}}
    ],
    "relations": []
  }},
  "metadata_filter": {{ ... }} or null
}}
Examples:
Example 1: Simple Filename Filter
User Query: "In the annual_report_2023.pdf file, what were the main conclusions?"
Expected JSON Output:
{{
  "expanded_queries": [
    "What are the key findings in the 2023 annual report?",
    "Summarize the executive summary of annual_report_2023.pdf."
  ],
  "extracted_entities_for_kg": {{
    "entities": [{{"text": "annual_report_2023.pdf", "label": "DOCUMENT"}}],
    "relations": []
  }},
  "metadata_filter": {{"filename": "annual_report_2023.pdf"}}
}}
Example 2: Advanced Chapter and Content Type Filter
User Query: "Show me the tables in the second chapter of the financial report."
Expected JSON Output:
{{
  "expanded_queries": [
    "What data is presented in the tables of chapter 2 of the financial report?",
    "List all tables from the second chapter of the financial analysis document."
  ],
  "extracted_entities_for_kg": {{
    "entities": [{{"text": "financial report", "label": "DOCUMENT"}}],
    "relations": []
  }},
  "metadata_filter": {{
    "$and": [
      {{"title_hierarchy_2": "财务分析"}},
      {{"paragraph_type": "table"}}
    ]
  }}
}}
Example 3: No Filter
User Query: "Who is the project manager for Project Alpha?"
Expected JSON Output:
{{
  "expanded_queries": [
    "Who leads Project Alpha?",
    "What are the responsibilities of the project manager for Project Alpha?"
  ],
  "extracted_entities_for_kg": {{
    "entities": [
      {{"text": "Project Alpha", "label": "PROJECT"}},
      {{"text": "project manager", "label": "TASK"}}
    ],
    "relations": []
  }},
  "metadata_filter": null
}}
<|im_end|>
<|im_start|>user
User Query: "{user_query}"
Output JSON:
<|im_end|>
<|im_start|>assistant
"""



def get_table_qa_messages(user_query: str, context_str: str) -> List[Dict[str, str]]:
    """
    构建一个专门用于处理表格问答（Table-QA）的LLM输入messages。
    这个Prompt指导LLM像数据分析师一样，精确地从Markdown表格中提取信息。
    """
    system_prompt_for_table_qa = f"""
你是一个精通Markdown表格的数据分析AI。你的【唯一任务】是根据用户提供的【用户问题】，在【上下文信息】的表格中查找并给出精确的答案。

**核心指令与行为准则：**

1.  **【定位关键信息】**:
    *   首先，在【用户问题】中识别出要查询的**关键实体** (例如, "产品B", "张三")。
    *   然后，在【上下文信息】的Markdown表格中，找到包含该**关键实体**的**那一行**。

2.  **【提取目标值】**:
    *   在定位到正确的行之后，根据【用户问题】的意图（例如，想查询“价格”、“年龄”、“城市”），找到对应的**列**。
    *   从该行和该列交叉的位置，提取出**精确的单元格数值**作为答案。

3.  **【答案格式】**:
    *   **如果找到答案**: 请直接、简洁地回答。模板："[关键实体]的[查询属性]是[提取的值]。"
        *   *示例*: "产品B的价格是150。"
    *   **如果找不到**: 如果在表格中找不到对应的行或列，导致无法回答，请使用这个固定的句子：“{NO_ANSWER_PHRASE_ANSWER_CLEAN}”

4.  **【绝对禁止】**:
    *   严禁对表格内容进行任何形式的计算、总结或推断（除非用户明确要求）。
    *   严禁使用表格之外的任何上下文信息。
    *   严禁输出任何与答案无关的解释或对话。

/no_think

请严格按照以上指令，像一个数据分析师一样精确地完成任务。
"""
    messages = [
        {"role": "system", "content": system_prompt_for_table_qa},
        {"role": "user", "content": f"用户问题: {user_query}\n\n上下文信息:\n{context_str}"}
    ]
    return messages
