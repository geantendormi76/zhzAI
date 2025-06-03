# zhz_agent/zhz_rag/config/constants.py

NEW_KG_SCHEMA_DESCRIPTION = """
# 知识图谱结构 (KuzuDB) 与 Cypher 查询生成规则

## 1. 节点定义:
- 节点标签: `:ExtractedEntity` (你必须且只能使用此节点标签)
- 节点属性:
    - `id_prop`: STRING (主键，实体的唯一标识)
    - `text`: STRING (实体的名称或文本内容)
    - `label`: STRING (实体类型。允许的值: "PERSON", "ORGANIZATION", "TASK")

## 2. 关系定义:
- 关系类型: `:WorksAt`
    - 结构: `(:ExtractedEntity {label:"PERSON"}) -[:WorksAt]-> (:ExtractedEntity {label:"ORGANIZATION"})`
    - 含义: 一个人 (PERSON) 在一个组织 (ORGANIZATION) 工作。
- 关系类型: `:AssignedTo`
    - 结构: `(:ExtractedEntity {label:"TASK"}) -[:AssignedTo]-> (:ExtractedEntity {label:"PERSON"})`
    - 含义: 一个任务 (TASK) 被分配给一个人 (PERSON)。

## 3. Cypher 查询生成 - 输出为 JSON 对象:

    你的【完整且唯一】的回答，必须是一个包含 "status" 和 "query" 字段的JSON对象。
    - 如果你能根据用户问题和Schema生成一个有效的Cypher查询：
        - "status" 字段应为 "success"。
        - "query" 字段应为该Cypher查询字符串。
    - 如果你无法生成有效的Cypher查询：
        - "status" 字段应为 "unable_to_generate"。
        - "query" 字段应为 "无法生成Cypher查询."。
    【不要在JSON之外或query字段内（当status为success时）包含任何解释或额外文本。】

## 4. JSON 输出格式示例:

### 示例 1 (能够生成查询):
用户问题: "任务'FixBug123'分配给了谁？"
你的【完整且唯一】的 JSON 回答:
```json
{
  "status": "success",
  "query": "MATCH (t:ExtractedEntity {text: 'FixBug123', label: 'TASK'})-[:AssignedTo]->(p:ExtractedEntity {label: 'PERSON'}) RETURN p.text AS Assignee"
}
示例 2 (无法根据Schema回答):
用户问题: "法国的首都是哪里？"
你的【完整且唯一】的 JSON 回答:
{
  "status": "unable_to_generate",
  "query": "无法生成Cypher查询."
}
"""