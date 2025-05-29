# zhz_agent/constants.py

NEW_KG_SCHEMA_DESCRIPTION = """
你的任务是根据用户的问题，严格利用以下【知识图谱Schema信息】生成一个或多个Cypher查询。

**【知识图谱Schema信息】**

1.  **节点 (Nodes):**
    *   **绝对核心规则：在生成的Cypher查询中，所有节点匹配时必须且只能使用 `:ExtractedEntity` 这个统一标签。严禁在MATCH模式中使用例如 :Person, :Organization, :Task 等更具体的标签名。节点的具体类型通过其 `label` 属性进行区分和筛选（例如，`(n:ExtractedEntity {label: 'PERSON'})`）。**
    *   每个 `:ExtractedEntity` 节点有且仅有以下两个核心属性:
        *   `text`: 字符串 (String)，表示实体的原始文本内容。
        *   `label`: 字符串 (String)，表示实体的类型。目前已知的实体类型包括: "PERSON", "ORGANIZATION", "TASK"。 (注意：虽然理论上可以有 "LOCATION" 等其他类型，但当前已定义的关系主要涉及这三者。)

2.  **关系 (Relationships):**
    *   目前仅支持以下两种关系类型，它们严格连接特定标签的 `:ExtractedEntity` 节点：
        *   关系名称: `:WORKS_AT`
            *   方向和类型: `(:ExtractedEntity {label:"PERSON"}) -[:WORKS_AT]-> (:ExtractedEntity {label:"ORGANIZATION"})`
            *   描述: 表示一个 PERSON 在一个 ORGANIZATION 工作。**此关系严格用于表示工作单位，目标节点必须是 `label:"ORGANIZATION"` 的 `:ExtractedEntity`。如果问题中提及“地点”但明显指代公司或机构，请查询 `ORGANIZATION` 类型的实体。**
            *   示例: `(person:ExtractedEntity {label:"PERSON", text:"张三"})-[:WORKS_AT]->(org:ExtractedEntity {label:"ORGANIZATION", text:"谷歌"})`
        *   关系名称: `:ASSIGNED_TO`
            *   方向和类型: `(:ExtractedEntity {label:"TASK"}) -[:ASSIGNED_TO]-> (:ExtractedEntity {label:"PERSON"})`
            *   描述: 表示一个任务分配给了一个人。
            *   示例: `(task:ExtractedEntity {label:"TASK", text:"项目Alpha的文档编写任务"})-[:ASSIGNED_TO]->(person:ExtractedEntity {label:"PERSON", text:"张三"})`
    *   **重要约束**：生成Cypher查询时，**必须且只能**使用上述明确定义的关系类型 (`:WORKS_AT`, `:ASSIGNED_TO`) 和节点属性 (`text`, `label`)。严禁使用任何未在此处定义的其他关系类型或节点属性。

**【Cypher查询生成规则】**

1.  **严格遵循Schema**:
    *   你的查询**必须完全基于**上面提供的【知识图谱Schema信息】。
    *   **节点标签必须固定为 `:ExtractedEntity`。例如，匹配一个“张三”这个人时，应写为 `(p:ExtractedEntity {label: 'PERSON', text: '张三'})`，绝对不能写成 `(p:Person {text: '张三'})`。**
    *   节点属性只能使用 `text` 和 `label`。
    *   关系类型只能使用 `:WORKS_AT` 和 `:ASSIGNED_TO`，并严格遵守其定义的方向和连接的实体类型。

2.  **匹配逻辑**:
    *   当用户问题中提及具体实体名称时，优先使用该实体的 `text` 属性进行精确匹配。
    *   同时，根据问题上下文或实体类型提示，使用 `label` 属性进行辅助筛选。

3.  **输出格式**:
    *   如果能生成有效查询，你的回答**必须只包含纯粹的Cypher查询语句本身**。
    *   如果根据问题和Schema无法生成有效的Cypher查询（例如，问题超出了Schema的表达能力，问题本身逻辑不通，或涉及未定义的关系/属性），**或者问题的核心查询意图（例如询问某个实体的一个特定但Schema中未定义的属性，或寻找一个Schema中未定义的关系类型来连接实体）无法通过已定义的节点属性或关系类型来精确满足，则必须只输出固定的短语：“无法生成Cypher查询。”不要试图通过返回实体本身的其他已知属性或已知的相关实体来“部分回答”该核心意图。如果一个问题询问某个任务的“具体内容”或“要求”，而Schema中没有为TASK实体定义这些属性或相关关系，那么就应该返回“无法生成Cypher查询。”**
    *   **绝对禁止**在有效的Cypher语句前后添加任何前缀、后缀、解释、注释或markdown标记。

**【查询示例 - 严格基于上述Schema和规则】**:

*   用户问题: "张三在哪里工作？"
    Cypher查询: MATCH (p:ExtractedEntity {text: '张三', label: 'PERSON'})-[:WORKS_AT]->(org:ExtractedEntity {label: 'ORGANIZATION'}) RETURN org.text AS organizationName

*   用户问题: "项目Alpha的文档编写任务分配给了谁？"
    Cypher查询: MATCH (task:ExtractedEntity {text: '项目Alpha的文档编写任务', label: 'TASK'})-[:ASSIGNED_TO]->(person:ExtractedEntity {label: 'PERSON'}) RETURN person.text AS personName

*   用户问题: "列出所有在谷歌工作的人。"
    Cypher查询: MATCH (p:ExtractedEntity {label: 'PERSON'})-[:WORKS_AT]->(org:ExtractedEntity {text: '谷歌', label: 'ORGANIZATION'}) RETURN p.text AS employeeName

*   用户问题: "张三负责哪些任务？"
    Cypher查询: MATCH (task:ExtractedEntity {label: 'TASK'})-[:ASSIGNED_TO]->(p:ExtractedEntity {text: '张三', label: 'PERSON'}) RETURN task.text AS taskName

*   用户问题: "谷歌公司有哪些员工？"
    Cypher查询: MATCH (p:ExtractedEntity {label: 'PERSON'})-[:WORKS_AT]->(org:ExtractedEntity {text: '谷歌', label: 'ORGANIZATION'}) RETURN p.text AS employeeName

*   用户问题: "查询所有任务及其负责人。"
    Cypher查询: MATCH (task:ExtractedEntity {label: 'TASK'})-[:ASSIGNED_TO]->(person:ExtractedEntity {label: 'PERSON'}) RETURN task.text AS taskName, person.text AS assignedPerson

*   用户问题: "百度的CEO是谁？" (此问题超出现有Schema表达能力)
    Cypher查询: 无法生成Cypher查询。

*   用户问题: "项目Alpha文档编写任务的具体内容是什么？" (核心意图是查询“具体内容”，但Schema中没有为TASK实体定义这些属性或相关关系，所以无法生成查询)
    Cypher查询: 无法生成Cypher查询。

*   用户问题: "张三目前的工作地点是哪个城市？" (Schema中 :WORKS_AT 指向 ORGANIZATION，没有直接的城市地点关系，且ORGANIZATION节点也没有城市属性)
    Cypher查询: 无法生成Cypher查询。

*   用户问题: "张三最近一次的工作变动是什么时候？" (此问题涉及Schema未定义的属性如日期)
    Cypher查询: 无法生成Cypher查询。

现在，请根据以下用户问题和上述Schema及规则生成Cypher查询。
"""