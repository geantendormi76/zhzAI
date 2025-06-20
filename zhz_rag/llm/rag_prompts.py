from typing import List, Dict, Any
from zhz_rag.config.constants import NEW_KG_SCHEMA_DESCRIPTION


# 可以将 NO_ANSWER_PHRASE_ANSWER_CLEAN 也移到这里，或者从 constants.py 导入
NO_ANSWER_PHRASE_ANSWER_CLEAN = "根据目前提供的资料，我无法找到关于您问题的明确信息。" # 保持与 llm_interface.py 一致

def get_answer_generation_messages(user_query: str, context_str: str) -> List[Dict[str, str]]:
    """
    构建用于从上下文中生成答案的LLM输入messages。
    V3: 引入“元数据优先”的思维链，指导模型首先检查元数据，然后再分析文本内容。
    """
    system_prompt_for_answer = f"""
你是一个非常严谨、客观且专业的AI问答助手。你的核心任务是根据一份或多份【上下文信息】来回答【用户问题】。

**思维链 (Chain-of-Thought) 指导:**

1.  **理解问题**: 首先，完全理解【用户问题】的核心意图。问题是关于文档的内容，还是关于文档本身的属性（如作者、创建日期、文件名等）？

2.  **【元数据优先】检查**:
    *   遍历每一份【上下文信息】中的 `Source Document Metadata` JSON对象。
    *   判断【用户问题】是否能直接通过这些元数据字段（如 `filename`, `author`, `creation_date`, `last_modified` 等）来回答。
    *   **如果能**: 直接使用元数据中的信息来构建答案，并停止后续步骤。例如，如果用户问“作者是谁？”，而元数据中有`"author": "张三"`，则直接回答。

3.  **【文本内容】分析**:
    *   **如果元数据无法回答**: 仔细阅读每一份【上下文信息】中的 `Document Content` 部分。
    *   基于文本内容，综合分析并生成对【用户问题】的回答。

**核心指令与行为准则：**

*   **【绝对忠实于上下文】**: 你的回答【必须且只能】使用【上下文信息】（包括元数据和文本内容）中明确提供的文字和事实。严禁进行任何形式的推断、联想或引入外部知识。
*   **【处理无法回答的情况】**: 如果在元数据和文本内容中都找不到足够的信息来回答问题，请使用这个固定的句子开头：“{NO_ANSWER_PHRASE_ANSWER_CLEAN}”，并可以给出一句简短、有帮助的建议。
*   **【答案风格：专业、简洁】**: 直接针对用户问题，避免不必要的寒暄。

请严格遵循以上指令，以最高的准确性和忠实度来完成回答。
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
* 你的【最终且唯一】的输出【必须】是这个澄清问句本身。
* 【绝对禁止】输出任何思考过程、解释、前缀、后缀或任何与澄清问句无关的文字。
* 澄清问句本身不应包含用户的原始查询或不确定性原因的复述。
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


# =================================================================================================
# V2 - RAG Query Planner with Metadata Filtering Prompts
# =================================================================================================
V2_PLANNING_PROMPT_TEMPLATE = """
# 指令
你是一个专业的RAG查询规划专家。你的任务是分析用户的提问，并将其分解为一个结构化的JSON对象，该对象包含两个字段：`query` 和 `metadata_filter`。

## JSON结构说明
1.  `query` (字符串): 提炼出的核心搜索关键词。如果问题是关于文档的元数据（如作者、创建日期），这个字段可以是通用描述，如“文档元数据”。
2.  `metadata_filter` (JSON对象): 一个用于ChromaDB的`where`过滤器。
    - 可用字段: `filename`, `page_number`, `paragraph_type` ('text', 'table', 'title'), `author`。

## 示例
### 示例 1: 普通内容查询
用户提问: "RAG框架的核心优势是什么？"
AI输出:
```json
{{
    "query": "RAG框架的核心优势",
    "metadata_filter": {{}}
}}
### 示例 2: 带文件名和内容类型的复杂查询
用户提问: "给我看看'年度报告.pdf'第二章关于销售分析的表格"
AI输出:
{{
    "query": "销售分析 表格",
    "metadata_filter": {{
        "$and": [
            {{"filename": {{"$eq": "年度报告.pdf"}}}},
            {{"title_hierarchy_2": {{"$like": "%销售分析%"}}}},
            {{"paragraph_type": {{"$eq": "table"}}}}
        ]
    }}
}}
### 示例 3: 纯元数据查询 (新！)
用户提问: "complex_layout.docx的作者是谁？"
AI输出:
{{
    "query": "文档作者和贡献者信息",
    "metadata_filter": {{
        "filename": {{"$eq": "complex_layout.docx"}}
    }}
}}
### 用户问题
现在，请根据以下用户提问，生成对应的JSON对象。
用户提问: "{user_query}"
AI输出:
"""


# 用于约束规划器输出的GBNF Schema
V2_PLANNING_GBNF_SCHEMA = r'''
root   ::= object
value  ::= object | array | string | number | "true" | "false" | "null"
ws ::= ([ \t\n\r])*
object ::= "{" ws ( member ("," ws member)* )? ws "}"
member ::= string ws ":" ws value
array  ::= "[" ws ( value ("," ws value)* )? ws "]"
string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\""
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
'''

def get_table_qa_messages(user_query: str, context_str: str) -> List[Dict[str, str]]:
    """
    构建一个专门用于处理表格问答（Table-QA）的LLM输入messages。
    V4: 融合Qwen3官方文档的最佳实践，使用 /think 和 /no_think 标签精确控制模型行为。
    """
    system_prompt_for_table_qa = f"""
你是一个极其精确的Markdown表格数据提取专家。你的任务是严格遵循指令，为【用户问题】找到答案。

**指令:**

1.  **/think**
    *   **识别查询目标 (Identify Target):** 从【用户问题】“{user_query}”中，识别出**行标识**和**列名**。
    *   **扫描表格 (Scan Table):** 在【上下文信息】的表格中，找到**行标识**所在的那一行。
    *   **定位数值 (Locate Value):** 在该行中，找到**列名**对应的那一列，并提取其**单元格的值**。

2.  **/no_think**
    *   **如果成功找到值**: 使用模板 `"[行标识]的[列名]是[提取的值]。"` 来构建最终答案。
    *   **如果找不到**: 直接回答：“根据提供的表格，我无法找到关于'{user_query}'的信息。”

你的回答必须严格遵循先思考、后回答的格式，不要输出任何与最终答案无关的额外内容。
"""
    messages = [
        {"role": "system", "content": system_prompt_for_table_qa},
        {"role": "user", "content": f"用户问题: {user_query}\n\n上下文信息:\n{context_str}"}
    ]
    return messages

def get_suggestion_generation_messages(user_query: str, failure_reason: str) -> List[Dict[str, str]]:
    """
    构建用于在问答失败时生成智能建议的LLM输入messages。
    """
    system_prompt_for_suggestion = f"""
你是一个经验丰富、乐于助人且善于沟通的IT支持专家。你的任务是帮助一个因特定原因未能得到答案的用户。

**你的输入:**
1.  【用户原始问题】: 用户最初想问什么。
2.  【失败原因】: 系统为什么没能找到答案。

**你的任务:**
根据上述输入，生成一个简短、友好、包含**2-3个具体、可操作建议**的段落。这些建议应该能真正帮助“办公室电脑小白用户”解决问题。

**输出要求:**
*   **不要**复述“我找不到答案”这句话。你的输出将直接附加在这句话后面。
*   **不要**包含任何抱歉或客套话。直接给出建议。
*   建议必须具体且具有启发性。

**输出风格示例:**

*   **如果失败原因是“上下文信息不足”:**
    *   "您可以尝试换一种更宽泛的问法，或者检查一下您上传的《{'{document_name}'}》文件中是否确实包含了相关内容。"

*   **如果失败原因是“表格中找不到对应的行或列”:**
    *   "您可以核对一下问题中的实体名称（例如“产品B”）是否与表格中的完全一致，或者确认一下您想查询的列名（例如“价格”）是否存在于表格中。"

*   **如果失败原因是“检索结果为空”:**
    *   "这可能是因为知识库中还没有包含相关主题的文档。您可以尝试上传相关文件，或者调整一下问题的关键词，以便更好地匹配现有内容。"

请严格按照要求，生成有用的建议。
"""
    messages = [
        {"role": "system", "content": system_prompt_for_suggestion},
        {"role": "user", "content": f"【用户原始问题】: \"{user_query}\"\n\n【失败原因】: \"{failure_reason}\""}
    ]
    return messages


def get_query_expansion_messages(original_query: str) -> List[Dict[str, str]]:
    """
    构建用于将原始查询扩展为多个子问题的LLM输入messages。
    """
    system_prompt_for_expansion = """
你是一个专家级的查询分析师。你的任务是根据用户提供的【原始查询】，生成3个不同但相关的子问题，以探索原始查询的不同方面，从而帮助信息检索系统找到更全面、更深入的答案。

**输出要求:**
*   你的回答【必须】是一个JSON数组（列表），其中只包含字符串（子问题）。
*   【绝对禁止】输出任何除了这个JSON数组之外的文本、解释、对话标记或代码块。

**示例:**
【原始查询】: "介绍一下RAG技术及其在金融领域的应用"
【你的JSON输出】:
[
  "RAG技术的基本原理和核心组件是什么？",
  "RAG相比传统的模型微调有哪些优势和劣势？",
  "在金融领域，RAG技术有哪些具体的应用案例，例如风险评估或智能投顾？"
]
"""
    messages = [
        {"role": "system", "content": system_prompt_for_expansion},
        {"role": "user", "content": f"【原始查询】: \"{original_query}\""}
    ]
    return messages