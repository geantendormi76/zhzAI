# from zhz_rag.config.constants import NEW_KG_SCHEMA_DESCRIPTION
from datetime import datetime, timedelta
from typing import List, Dict, Any

# from zhz_rag.config.constants import NEW_KG_SCHEMA_DESCRIPTION
# ... (文件的其余部分保持不变)

# 可以将 NO_ANSWER_PHRASE_ANSWER_CLEAN 也移到这里，或者从 constants.py 导入
NO_ANSWER_PHRASE_ANSWER_CLEAN = "根据目前提供的资料，我无法找到关于您问题的明确信息。" # 保持与 llm_interface.py 一致


def get_answer_generation_messages(user_query: str, context_str: str) -> List[Dict[str, str]]:
    """
    构建用于从上下文中生成答案的LLM输入messages。
    V4: 引入“引用优先”的思维链，强制模型优先引用原文，以解决幻觉和过度泛化问题。
    """
    system_prompt_for_answer = f"""
你是一个极其严谨、客观且专业的AI问答助手。你的核心任务是根据一份或多份【上下文信息】来回答【用户问题】。

**思维链 (Chain-of-Thought) 指导 - 【引用优先原则】:**

1.  **理解问题**: 首先，完全理解【用户问题】的核心意图。

2.  **扫描上下文寻找直接引文**:
    *   仔细阅读每一份【上下文信息】，寻找能够**直接、逐字回答**【用户问题】的句子或段落。
    *   **如果找到**: 优先使用这些原文来构建你的答案。你可以对原文进行少量删减或连接，但不能改变其核心意思和措辞。

3.  **基于多处信息进行综合**:
    *   **如果找不到单一的直接引文**: 尝试从上下文的不同部分提取相关事实和数据点。
    *   将这些事实**像拼图一样组合起来**，形成一个连贯的答案。在综合时，**必须**使用原文中的短语和术语。

4.  **元数据检查**:
    *   如果问题是关于文档的属性（如作者、文件名），请直接从 `Source Document Metadata` 中提取信息回答。

**核心指令与行为准则：**

*   **【绝对忠实于原文】**: 你的回答【必须且只能】是【上下文信息】中明确文字的直接引用或忠实转述。**严禁进行任何形式的总结、归纳、推断或引入外部知识。** 你是一个信息的搬运工，不是思想的创造者。
*   **【处理无法回答的情况】**: 如果在元数据和文本内容中都找不到足够的信息来回答问题，请只回答这一句话，不要添加任何其他内容：“{NO_ANSWER_PHRASE_ANSWER_CLEAN}”
*   **【答案风格：专业、客观】**: 直接针对用户问题进行回答，避免使用“根据我分析...”或“我认为...”等主观性词汇。

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
# SIMPLIFIED_CYPHER_TEMPLATES = [
#     {
#         "id": "template_find_entity_attributes_by_text_label",
#         "description": "根据提供的实体文本和实体标签，查找该实体的所有基本属性。",
#         "template": "MATCH (n:ExtractedEntity {{text: $entity_text, label: $entity_label}}) RETURN n.text AS text, n.label AS label, n.id_prop AS id_prop LIMIT 1",
#         "params_needed": ["entity_text", "entity_label"]
#     }
# ]

# def get_cypher_generation_messages_with_templates(user_question: str) -> List[Dict[str, str]]: # 函数名保持一致
#     """
#     构建用于（基于【单个指定模板】）生成Cypher查询的LLM输入messages。
#     这个版本用于测试模型对单个模板的参数提取能力。
#     """
    
#     # 在这个测试版本中，我们假设总是使用第一个（也是唯一一个）模板
#     selected_template = SIMPLIFIED_CYPHER_TEMPLATES[0]
    
#     template_description_for_prompt = f"""你将使用以下Cypher查询模板：
# Template ID: {selected_template['id']}
# Description: {selected_template['description']}
# Cypher Structure: {selected_template['template']}
# Parameters Needed: {', '.join(selected_template['params_needed'])}
# """

#     system_prompt_for_cypher = f"""你是一个精确的参数提取助手。你的任务是根据用户问题，为下面提供的【唯一Cypher查询模板】提取参数，并构建一个Cypher查询。

# **【图谱Schema核心部分参考】**
# (你主要关注 `:ExtractedEntity` 节点及其属性: `text`, `label`, `id_prop`。其中 `label` 的常见值是 "PERSON", "ORGANIZATION", "TASK"。)
# {NEW_KG_SCHEMA_DESCRIPTION} 
# # ^^^ Schema描述已包含输出JSON格式 {{"status": "success/unable_to_generate", "query": "..."}} 的指导，请严格遵循该JSON输出格式。

# **【你的任务与输出要求】**
# 1.  仔细分析【用户问题】，理解其核心查询意图。
# 2.  判断该意图是否与提供的【当前需要填充的Cypher查询模板】描述相符。
# 3.  如果相符：
#     a.  从【用户问题】中提取填充该模板所需的所有【Parameters Needed】。确保参数值与Schema中的实体文本和标签格式相符（例如，标签应为大写 "PERSON", "ORGANIZATION", "TASK"）。
#     b.  将提取的参数值替换到模板的Cypher语句中（例如，`$entity_text` 替换为提取到的实体名）。
#     c.  最终输出一个JSON对象，格式为：`{{"status": "success", "query": "填充好参数的Cypher语句"}}`。
# 4.  如果不相符（例如，用户问题意图与模板描述不符，或无法从问题中提取到模板所需的所有关键参数）：
#     a.  最终输出一个JSON对象，格式为：`{{"status": "unable_to_generate", "query": "无法生成Cypher查询."}}`。
# 5.  【绝对禁止】输出任何除了上述指定JSON对象之外的文本、解释或思考过程。


# **【处理示例】**
# <example>
#   <user_question>我想知道张三的详细信息。</user_question>
#   <assistant_output_json>{{
#     "status": "success",
#     "query": "MATCH (n:ExtractedEntity {{text: '张三', label: 'PERSON'}}) RETURN n.text AS text, n.label AS label, n.id_prop AS id_prop LIMIT 1"
#   }}</assistant_output_json>
# </example>
# <example>
#   <user_question>项目Alpha的文档编写任务是什么？</user_question>
#   <assistant_output_json>{{
#     "status": "success",
#     "query": "MATCH (n:ExtractedEntity {{text: '项目alpha的文档编写任务', label: 'TASK'}}) RETURN n.text AS text, n.label AS label, n.id_prop AS id_prop LIMIT 1"
#   }}</assistant_output_json>
# </example>
# <example>
#   <user_question>法国的首都是哪里？</user_question>
#   <assistant_output_json>{{
#     "status": "unable_to_generate",
#     "query": "无法生成Cypher查询."
#   }}</assistant_output_json>
# </example>
# """
#     user_content = f"用户问题: {user_question}"

#     messages = [
#         {"role": "system", "content": system_prompt_for_cypher},
#         {"role": "user", "content": user_content}
#     ]
#     return messages

# --- 新增：实体与关系意图提取的提示词生成函数 ---
# def get_entity_relation_extraction_messages(user_question: str) -> List[Dict[str, str]]:
#     """
#     构建用于从用户查询中提取核心实体和关系意图的LLM输入messages。
#     目标是输出一个符合 ExtractedEntitiesAndRelationIntent Pydantic 模型结构的纯净JSON对象。
#     这个版本的Prompt极度强调JSON输出格式。
#     """
#     import re
#     match = re.search(r'label\s*:\s*STRING\s*\(实体类型。\s*允许的值\s*:\s*("([^"]+)"(?:,\s*"([^"]+)")*)\)', NEW_KG_SCHEMA_DESCRIPTION)
#     allowed_entity_labels_str = "PERSON, ORGANIZATION, TASK, DOCUMENT, PROJECT, REGION, PRODUCT, OTHER"
#     if match:
#         labels_group = match.group(1)
#         extracted_labels = re.findall(r'"([^"]+)"', labels_group)
#         if extracted_labels:
#             allowed_entity_labels_str = ", ".join(extracted_labels)
#             if "OTHER" not in extracted_labels:
#                 allowed_entity_labels_str += ", OTHER"

#     # --- V3 "最最严格" Prompt ---
#     system_prompt_for_entity_extraction = f"""<|im_start|>system
# USER_QUERY_TO_PROCESS:
# {user_question}

# TASK: Analyze USER_QUERY_TO_PROCESS. Output ONLY a valid JSON object.
# NO EXPLANATIONS. NO EXTRA TEXT. NO MARKDOWN. JUST JSON.

# JSON_OUTPUT_SCHEMA:
# {{
#   "entities": [
#     {{"text": "string, extracted entity text from USER_QUERY_TO_PROCESS", "label": "string, entity type from: [{allowed_entity_labels_str}], or OTHER"}}
#   ],
#   "relation_hint": "string, relation described in USER_QUERY_TO_PROCESS, or empty string"
# }}

# RULES:
# 1. Max 2 entities in "entities" array. If none, "entities" is `[]`.
# 2. "label" MUST be from the provided list or "OTHER".
# 3. If no relation_hint, value is `""`.
# 4. If USER_QUERY_TO_PROCESS yields no entities or relation, output: `{{"entities": [], "relation_hint": ""}}`

# YOUR_VALID_JSON_OUTPUT_ONLY:<|im_end|>""" # <--- 结尾引导更加直接

#     messages = [
#         {"role": "system", "content": system_prompt_for_entity_extraction}
#     ]
#     return messages



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

## 【【【核心规则 - 必须严格遵守】】】
1.  **文件名提取规则**: 只有当用户在问题中**明确**提到了一个带引号的文件名（例如 “年度报告.docx”）或带有文件扩展名（.pdf, .xlsx等）的文件时，你才可以在`metadata_filter`中设置`filename`字段。
2.  **严禁猜测**: 对于所有其他情况，**严禁**猜测或编造任何文件名。如果用户没有明确提及文件名，`metadata_filter`中就**不能**包含`filename`字段，或者`filename`字段的值必须为空。

## 示例
### 示例 1: 普通内容查询
用户提问: "RAG框架的核心优势是什么？"
AI输出:
```json
{{
    "query": "RAG框架的核心优势",
    "metadata_filter": {{}}
}}
```
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
### 示例 3: 纯元数据查询
用户提问: "complex_layout.docx的作者是谁？"
AI输出:
{{
    "query": "文档作者和贡献者信息",
    "metadata_filter": {{
        "filename": {{"$eq": "complex_layout.docx"}}
    }}
}}
### 示例 4: 暗示性查询 (无明确文件名)
用户提问: "关于笔记本电脑的库存情况"
AI输出:
```json
{{
    "query": "笔记本电脑 库存",
    "metadata_filter": {{}}
}}
```
### 用户问题
现在，请根据以下用户提问和上述所有规则，生成对应的JSON对象。
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
    V2: 构建用于将原始查询扩展为多个子问题的LLM输入messages。
    此版本经过优化，旨在生成更多样化的查询，特别是简洁的关键词组合，以提升BM25检索效果。
    """
    system_prompt_for_expansion = """
你是一个专家级的查询分析师。你的任务是根据用户提供的【原始查询】，生成3个不同但相关的子问题或关键词组合，以探索原始查询的不同方面，从而帮助信息检索系统找到更全面、更深入的答案。

**生成要求:**
1.  **多样性**: 生成的查询应涵盖不同角度，包括但不限于：
    *   对原始问题的直接改写。
    *   从原始问题中提取出的核心【实体】和【属性】的简洁关键词组合（例如：“笔记本电脑 库存”、“项目Alpha 状态”）。这对于关键词检索至关重要。
    *   基于用户意图推断出的相关探索性问题。
2.  **输出格式**: 你的回答【必须】是一个JSON对象，该对象只包含一个键`"queries"`，其值是一个包含字符串（子问题或关键词组合）的JSON数组。
3.  **严格约束**: 【绝对禁止】输出任何除了这个JSON对象之外的文本、解释、对话标记或代码块。

**示例:**
【原始查询】: "介绍一下RAG技术及其在金融领域的应用"
【你的JSON输出】:
```json
{
  "queries": [
    "RAG技术基本原理",
    "RAG金融领域应用案例",
    "检索增强生成与传统微调的对比"
  ]
}
【原始查询】: "笔记本电脑的库存还有多少？"
【你的JSON输出】:
{
  "queries": [
    "笔记本电脑 库存 数量",
    "所有电脑的库存列表",
    "如何查询电子产品库存"
  ]
}
"""
    messages = [
        {"role": "system", "content": system_prompt_for_expansion},
        {"role": "user", "content": f"【原始查询】: \"{original_query}\""}
    ]
    return messages



def get_fusion_messages(original_query: str, fusion_context: str) -> List[Dict[str, str]]:
    """
    构建用于将多个子答案融合成一个最终报告的LLM输入messages。
    """
    system_prompt_for_fusion = """
你是一个顶级的【信息整合与报告撰写专家】。
你的任务是将一系列针对【原始问题】的【子问题与子答案】进行分析、整合、去重，并最终撰写成一份逻辑清晰、内容全面、专业且连贯的【最终报告】。

**核心指令:**

1.  **目标导向**: 你的【最终报告】必须直接、完整地回答【原始问题】。
2.  **信息来源**: 你【只能】使用【子问题与子答案】中提供的信息。严禁引入任何外部知识或进行不合理的推断。
3.  **整合与去重**: 将不同子答案中的相关信息进行逻辑上的连接和整合。如果多个子答案提到相同的事实，请在最终报告中只提及一次，避免重复。
4.  **结构化输出**: 如果内容复杂，请使用标题、列表（如 1., 2., ... 或 -）等方式来组织你的【最终报告】，使其易于阅读。
5.  **专业风格**: 保持客观、中立的语气。直接开始撰写报告内容，不要添加如“好的，这是您的报告”等多余的开场白。
6.  **处理矛盾/不足**: 如果提供的子答案信息不足以形成一份有意义的报告，或者信息之间存在明显矛盾，请直接回答“根据现有信息，无法就您的问题给出一个全面统一的答案。”

请现在基于以下信息，开始你的报告撰写工作。
"""
    
    user_content = f"""
【原始问题】:
{original_query}

【子问题与子答案】:
{fusion_context}

---
【你的最终报告】:
"""

    messages = [
        {"role": "system", "content": system_prompt_for_fusion},
        {"role": "user", "content": user_content}
    ]
    return messages


def get_document_summary_messages(user_query: str, document_content: str) -> List[Dict[str, str]]:
    """
    构建用于“为单个文档，针对用户问题，生成一句核心摘要”的LLM输入messages。
    """
    system_prompt_for_summary = f"""
你是一个高度专注的【信息摘要AI】。你的唯一任务是阅读一份【文档内容】，并根据【用户原始问题】，用一句话总结出该文档中与问题最相关的核心信息。

**核心指令:**

1.  **绝对相关**: 你的摘要【必须】直接回应【用户原始问题】。
2.  **绝对简洁**: 你的回答【只能】是一句话，不能超过50个字。
3.  **基于原文**: 你的摘要【必须】完全基于【文档内容】。
4.  **无相关信息处理**: 如果文档内容与用户问题完全不相关，请【直接且仅】输出字符串："irrelevant"

**示例 1:**
【用户原始问题】: "RAG的优势是什么？"
【文档内容】: "...RAG技术通过结合检索器和生成器，显著提升了答案的准确性和时效性，这是它相较于传统微调方法的核心优势..."
【你的输出】:
RAG技术的核心优势在于通过结合检索与生成，提升了答案的准确性和时效性。

**示例 2:**
【用户原始问题】: "介绍一下ACME公司的组织架构。"
【文档内容】: "...RAG技术通过结合检索器和生成器，显著提升了答案的准确性和时效性..."
【你的输出】:
irrelevant
"""
    
    user_content = f"""
【用户原始问题】:
{user_query}

【文档内容】:
{document_content}

---
【你的输出】:
"""

    messages = [
        {"role": "system", "content": system_prompt_for_summary},
        {"role": "user", "content": user_content}
    ]
    return messages


def get_task_extraction_messages(user_query: str, llm_answer: str) -> List[Dict[str, str]]:
    """
    V4.4: Final refinement for task extraction prompt.
    Adds explicit rules to prevent misinterpreting simple queries as tasks.
    """
    today_str = datetime.now().strftime('%Y-%m-%d')

    system_prompt = f"""
你是超精准的任务信息提取AI。你的任务是分析【用户问题】和【AI的回答】，并参考【当前日期】，判断其中是否包含明确的、未来需要执行的待办事项。

**当前日期**: {today_str}

**核心规则:**
1.  **识别行动意图**: 仅当对话明确指向一个**未来**的、**具体**的行动时（如“提醒我...”、“...需要提交”、“记得...做某事”），才视为任务。
2.  **【排除规则 - 非常重要】**: 如果用户问题只是一个**单纯的查询或提问**（例如“...是什么？”、“...有多少？”、“...在哪里？”），即使它包含了时间状语（如“明天”），也**不应被视为待办任务**。查询的本质是获取信息，而非安排行动。
3.  **提取任务细节**: 如果确定是任务，则提取`title`, `due_date`, `reminder_offset_minutes`。

**输出格式:**
你的回答**必须**是一个JSON对象。
- **如果识别出任务**: `{{"task_found": true, "title": "任务标题", "due_date": "YYYY-MM-DD HH:MM:SS", "reminder_offset_minutes": <integer_minutes | null>}}`
- **如果没有任务**: `{{"task_found": false, "title": null, "due_date": null, "reminder_offset_minutes": null}}`

**示例:**

<example>
  <user_query>提醒我下午4点去参加项目Alpha的周会</user_query>
  <llm_answer></llm_answer>
  <assistant_json_output>
  {{"task_found": true, "title": "参加项目Alpha的周会", "due_date": "{today_str} 16:00:00", "reminder_offset_minutes": null}}
  </assistant_json_output>
</example>

<example>
  <user_query>明天需要向领导汇报一下笔记本电脑的库存情况，你能帮我查一下吗？</user_query>
  <llm_answer>笔记本电脑的库存是50台。</llm_answer>
  <assistant_json_output>
  {{"task_found": true, "title": "向领导汇报笔记本电脑的库存情况", "due_date": "{datetime.now().date() + timedelta(days=1)} 13:00:00", "reminder_offset_minutes": null}}
  </assistant_json_output>
</example>

<example>
  <user_query>办公椅C的库存是多少？</user_query>
  <llm_answer>办公椅C的库存是65。</llm_answer>
  <assistant_json_output>
  {{"task_found": false, "title": null, "due_date": null, "reminder_offset_minutes": null}}
  </assistant_json_output>
</example>
"""
    
    user_content = f"""
【用户问题】: "{user_query}"
【AI的回答】: "{llm_answer}"

请根据以上对话和规则，提取任务信息并输出JSON。
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    return messages


INTENT_CLASSIFICATION_GBNF_SCHEMA = r'''
root   ::= object
object ::= "{" ws "\"intent\"" ws ":" ws intent-enum ws "," ws "\"reasoning\"" ws ":" ws string ws "}"
intent-enum ::= "\"rag_query\"" | "\"task_creation\"" | "\"mixed_intent\""
string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\""
ws ::= ([ \t\n\r])*
'''


def get_intent_classification_messages(user_query: str) -> List[Dict[str, str]]:
    """
    V4.2: Builds the LLM input messages for classifying the user's primary intent.
    """
    system_prompt = """
You are a highly intelligent and efficient user intent classifier. Your sole purpose is to analyze the user's query and classify it into one of three categories: `rag_query`, `task_creation`, or `mixed_intent`.

**Category Definitions:**
- **`rag_query`**: The user is primarily asking for information, seeking knowledge, or requesting a summary from existing documents. These are classic "question answering" tasks.
- **`task_creation`**: The user is primarily giving a command to remember something, set a reminder, or schedule a future action. This intent does not require information retrieval from the knowledge base.
- **`mixed_intent`**: The user's query contains BOTH an informational question AND a task/reminder command.

**Your Output MUST be a valid JSON object with the following structure:**
```json
{
  "intent": "string, one of [rag_query, task_creation, mixed_intent]",
  "reasoning": "string, a brief explanation for your classification choice."
}
```
**No other text, explanations, or markdown formatting outside of this JSON object.**

**Examples:**

<example>
  <user_query>What are the key financial highlights from the NVIDIA 2024 annual report?</user_query>
  <assistant_json_output>
  {"intent": "rag_query", "reasoning": "The user is asking for specific information (financial highlights) from a document (NVIDIA 2024 report)."}
  </assistant_json_output>
</example>

<example>
  <user_query>Remind me to call the finance department at 4 PM today.</user_query>
  <assistant_json_output>
  {"intent": "task_creation", "reasoning": "The user is giving a direct command to set a reminder for a future action, without asking for information."}
  </assistant_json_output>
</example>

<example>
  <user_query>Can you find the project summary for 'Project Alpha' and also remind me to review it tomorrow morning?</user_query>
  <assistant_json_output>
  {"intent": "mixed_intent", "reasoning": "The user is both asking for information (find project summary) and requesting a future action (remind me to review)."}
  </assistant_json_output>
</example>
"""
    user_content = f"""
Analyze the following user query and provide your classification in the required JSON format.

User Query: "{user_query}"
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    return messages


# --- 新增：专门用于查询扩展的GBNF Schema ---
QUERY_EXPANSION_GBNF_SCHEMA = r'''
root   ::= object
object ::= "{" ws "\"queries\"" ws ":" ws array ws "}"
array  ::= "[" ws ( string ("," ws string)* )? ws "]"
string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\""
ws ::= ([ \t\n\r])*
'''