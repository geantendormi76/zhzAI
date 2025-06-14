# /home/zhz/zhz_rag/llm/rag_prompts.py
from typing import List, Dict, Any
from zhz_rag.config.constants import NEW_KG_SCHEMA_DESCRIPTION


# 可以将 NO_ANSWER_PHRASE_ANSWER_CLEAN 也移到这里，或者从 constants.py 导入
NO_ANSWER_PHRASE_ANSWER_CLEAN = "根据目前提供的资料，我无法找到关于您问题的明确信息。" # 保持与 llm_interface.py 一致

def get_answer_generation_messages(user_query: str, context_str: str) -> List[Dict[str, str]]:
    """
    构建用于从上下文中生成答案的LLM输入messages。
    """
    system_prompt_for_answer = f"""
你是一个非常严谨、客观且专业的AI问答助手。你的核心任务是根据【上下文信息】回答【用户问题】。

**核心指令与行为准则：**

1.  **【绝对忠实于上下文】**: 你的回答【必须且只能】使用【上下文信息】中明确提供的文字和事实。严禁进行任何形式的推断、联想、猜测或引入外部知识。如果上下文信息不足或不相关，请明确指出。
2.  **【逐点核实与直接证据】**: 对于用户问题中的每一个具体信息点或子问题，你都必须在【上下文信息】中找到清晰、直接的证据来支持你的回答。如果没有直接证据，则视为无法回答该点。
3.  **【引用来源 (如果可能且适用)】**: 如果你的答案直接引用或高度依赖【上下文信息】中的特定片段，请尽可能简要地指出信息来源。例如，如果上下文片段被标记了来源（如“来源文档A第3段”），你可以说“根据文档A第3段，...”。如果上下文没有明确的来源标记，则无需强行编造。**此项为次要优先级，准确回答问题是首要的。**
4.  **【处理无法回答的情况】**:
 *   **完全无法回答**: 如果【上下文信息】完全不包含与【用户问题】相关的任何信息，或者无法找到任何直接证据来回答问题的任何部分，请【只回答】：“{NO_ANSWER_PHRASE_ANSWER_CLEAN}”
 *   **部分无法回答**: 如果【用户问题】包含多个子问题或方面，而【上下文信息】只能回答其中的一部分：
     *   请只回答你能找到直接证据支持的部分。
     *   对于【上下文信息】中没有直接证据支持的其他子问题或方面，请明确指出，例如：“关于您提到的[某子问题/方面]，提供的上下文中未包含相关信息。”
     *   **禁止**对未提供信息的部分进行任何形式的猜测或尝试回答。
5.  **【答案风格：专业、简洁、直接】**:
 *   回答应直接针对用户问题，避免不必要的寒暄或冗余信息。
 *   语言表达应专业、客观、清晰易懂。
 *   如果答案包含多个要点，可以使用简洁的列表格式。
6.  **【避免重复与冗余】**: 如果多个上下文片段提供了相同的信息，请综合它们并给出不重复的答案。
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
    目标是输出一个符合 ExtractedEntitiesAndRelationIntent Pydantic 模型结构的JSON。
    """
    # 从 NEW_KG_SCHEMA_DESCRIPTION 中提取允许的实体标签，以便在提示中告知LLM
    # 这是一个简化的提取，实际应用中可能需要更精确地从Schema中获取
    # 假设 NEW_KG_SCHEMA_DESCRIPTION 中有类似 "label: STRING (实体类型。允许的值: "PERSON", "ORGANIZATION", "TASK")" 的描述
    import re
    match = re.search(r'label\s*:\s*STRING\s*\(实体类型。\s*允许的值\s*:\s*("([^"]+)"(?:,\s*"([^"]+)")*)\)', NEW_KG_SCHEMA_DESCRIPTION)
    allowed_entity_labels_str = "PERSON, ORGANIZATION, TASK" # 默认值
    if match:
        # 提取所有带引号的标签
        labels_group = match.group(1)
        extracted_labels = re.findall(r'"([^"]+)"', labels_group)
        if extracted_labels:
            allowed_entity_labels_str = ", ".join(extracted_labels)
    
    system_prompt_for_entity_extraction = f"""你的任务是仔细分析用户提供的【用户问题】，并识别出其中与知识图谱查询相关的核心信息。

**你需要识别以下内容：**
1.  **核心实体**：问题中提到的1到2个最关键的实体（人名、组织名、任务名等）。
2.  **实体类型**：为每个识别出的实体，从以下参考类型中推断其最可能的类型：{allowed_entity_labels_str}。如果无法确定，可以不指定类型。
3.  **关系意图**：如果用户问题暗示了实体间的特定关系，请用简洁的文本描述这个关系意图（例如 “查询工作地点”, “查找负责人”, “获取销售额”）。如果只是查询单个实体的属性，则关系意图不明确。

请在你的回答中清晰地包含这些分析结果。最终的结构化输出将由系统根据你的分析自动完成。
你只需要专注于准确地理解和提取这些信息。
"""
    # 移除了所有关于JSON输出格式的指令和示例，因为将由response_format处理
    # 也不再需要 /no_think，因为我们期望约束生成能处理好输出

    user_content = f"用户问题: {user_question}\n\n请分析此问题并提取相关实体和关系意图："

    messages = [
        {"role": "system", "content": system_prompt_for_entity_extraction},
        {"role": "user", "content": user_content}
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
""" # <--- 末尾引导词已删除