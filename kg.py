# zhz_agent/kg.py
import json
import os
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, basic_auth, Result, Record # 移除了 EagerResult，因为我们主要处理 Result 和 Record
import asyncio
import logging

# --- 日志配置 (保持您之前的优秀配置) ---
_kg_py_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(_kg_py_dir, 'kg_retriever.log')
kg_logger = logging.getLogger(__name__) 
kg_logger.setLevel(logging.DEBUG) 
kg_logger.propagate = False
if kg_logger.hasHandlers():
    kg_logger.handlers.clear()
try:
    file_handler = logging.FileHandler(log_file_path, mode='w') 
    file_handler.setLevel(logging.DEBUG) 
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    kg_logger.addHandler(file_handler)
    kg_logger.info("--- KG logging reconfigured to write to kg_retriever.log (dedicated handler) ---")
except Exception as e:
    print(f"CRITICAL: Failed to configure file handler for kg_logger: {e}")

from zhz_agent.llm import generate_cypher_query
from zhz_agent.pydantic_models import RetrievedDocument
from dotenv import load_dotenv
load_dotenv()

NEW_KG_SCHEMA_DESCRIPTION = """
你的任务是根据用户的问题，利用以下知识图谱Schema信息生成一个或多个Cypher查询。
图谱中的主要节点标签是 :ExtractedEntity。
每个 :ExtractedEntity 节点有以下核心属性：
- text: 字符串，表示实体的原始文本。
- label: 字符串，表示实体的类型，例如 "PERSON", "ORGANIZATION", "TASK", "LOCATION"。

目前支持以下关系类型，它们连接 :ExtractedEntity 节点：
1. :WORKS_AT (从 label="PERSON" 的节点指向 label="ORGANIZATION" 的节点)
   - 示例: (person:ExtractedEntity {label:"PERSON"})-[:WORKS_AT]->(org:ExtractedEntity {label:"ORGANIZATION"})
2. :ASSIGNED_TO (从 label="TASK" 的节点指向 label="PERSON" 的节点)
   - 示例: (task:ExtractedEntity {label:"TASK"})-[:ASSIGNED_TO]->(person:ExtractedEntity {label:"PERSON"}) 

查询时，请优先使用节点的 text 属性进行匹配。
如果用户问题中提到了实体类型，请在Cypher查询中使用节点的 label 属性进行筛选。
例如，如果用户问“张三在哪里工作？”，你可能需要找到一个 text="张三" 且 label="PERSON" 的节点，然后查询其 WORKS_AT 关系。
如果用户问“项目Alpha的任务分配给了谁？”，你可能需要找到一个 text="项目Alpha的文档编写任务" 且 label="TASK" 的节点，然后查询其 ASSIGNED_TO 关系。
请只返回Cypher查询语句，不要包含任何解释或其他文本。如果无法根据问题和Schema生成有意义的查询，请返回 "无法生成Cypher查询。"

**【查询示例 - 严格基于上述Schema】**:

*   用户问题: "张三在哪里工作？"
    Cypher查询: MATCH (p:ExtractedEntity {text: '张三', label: 'PERSON'})-[:WORKS_AT]->(org:ExtractedEntity {label: 'ORGANIZATION'}) RETURN org.text AS organizationName

*   用户问题: "项目Alpha的文档编写任务分配给了谁？"
    Cypher查询: MATCH (task:ExtractedEntity {text: '项目Alpha的文档编写任务', label: 'TASK'})-[:ASSIGNED_TO]->(person:ExtractedEntity {label: 'PERSON'}) RETURN person.text AS personName

*   用户问题: "列出所有在谷歌工作的人。"
    Cypher查询: MATCH (p:ExtractedEntity {label: 'PERSON'})-[:WORKS_AT]->(org:ExtractedEntity {text: '谷歌', label: 'ORGANIZATION'}) RETURN p.text AS employeeName

*   用户问题: "张三负责哪些任务？"
    Cypher查询: MATCH (task:ExtractedEntity {label: 'TASK'})-[:ASSIGNED_TO]->(p:ExtractedEntity {text: '张三', label: 'PERSON'}) RETURN task.text AS taskName

*   用户问题: "谷歌公司有哪些员工？" (与 "列出所有在谷歌工作的人" 类似，测试LLM的理解)
    Cypher查询: MATCH (p:ExtractedEntity {label: 'PERSON'})-[:WORKS_AT]->(org:ExtractedEntity {text: '谷歌', label: 'ORGANIZATION'}) RETURN p.text AS employeeName

*   用户问题: "查询所有任务及其负责人。"
    Cypher查询: MATCH (task:ExtractedEntity {label: 'TASK'})-[:ASSIGNED_TO]->(person:ExtractedEntity {label: 'PERSON'}) RETURN task.text AS taskName, person.text AS assignedPerson

*   用户问题: "百度的CEO是谁？" (假设Schema中没有CEO的具体关系和属性)
    Cypher查询: 无法生成Cypher查询。

现在，请根据以下用户问题和上述Schema及规则生成Cypher查询。
"""

class KGRetriever:
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "zhz199276")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

    def __init__(self, llm_cypher_generator_func: callable = generate_cypher_query):
        self.llm_cypher_generator_func = llm_cypher_generator_func
        self._driver: Optional[GraphDatabase.driver] = None
        self._connect_to_neo4j()
        kg_logger.info(f"KGRetriever initialized. Connected to Neo4j: {self._driver is not None}")

    def _connect_to_neo4j(self):
        if self._driver is not None:
            try:
                self._driver.verify_connectivity()
                kg_logger.info("Neo4j connection already active and verified.")
                return
            except Exception:
                kg_logger.warning("Existing Neo4j driver failed connectivity test, attempting to reconnect.")
                try:
                    self._driver.close()
                except: pass
                self._driver = None
        try:
            self._driver = GraphDatabase.driver(self.NEO4J_URI, auth=basic_auth(self.NEO4J_USER, self.NEO4J_PASSWORD))
            with self._driver.session(database=self.NEO4J_DATABASE) as session:
                session.run("RETURN 1").consume()
            kg_logger.info(f"Successfully connected to Neo4j at {self.NEO4J_URI} on database '{self.NEO4J_DATABASE}'.")
        except Exception as e:
            kg_logger.error(f"Failed to connect to Neo4j: {e}", exc_info=True)
            self._driver = None
    
    def close(self):
        if self._driver:
            self._driver.close()
            kg_logger.info("Closed Neo4j connection.")
            self._driver = None

    def _convert_neo4j_value_to_json_serializable(self, value: Any) -> Any:
        """
        递归地将Neo4j返回的各种值转换为JSON可序列化的Python原生类型。
        """
        if hasattr(value, 'labels') and hasattr(value, 'properties') and hasattr(value, 'id'): # Node
            return {"_id": value.id, "_labels": list(value.labels), **dict(value.properties)}
        elif hasattr(value, 'type') and hasattr(value, 'properties') and hasattr(value, 'id') and hasattr(value, 'start_node') and hasattr(value, 'end_node'): # Relationship
            return {
                "_id": value.id, 
                "_type": value.type,
                "_start_node_id": value.start_node.id, 
                "_end_node_id": value.end_node.id,
                **dict(value.properties)
            }
        elif hasattr(value, 'nodes') and hasattr(value, 'relationships'): # Path
            return {
                "nodes": [self._convert_neo4j_value_to_json_serializable(n) for n in value.nodes],
                "relationships": [self._convert_neo4j_value_to_json_serializable(r) for r in value.relationships]
            }
        elif isinstance(value, list) or isinstance(value, tuple) or isinstance(value, set):
            return [self._convert_neo4j_value_to_json_serializable(item) for item in value]
        elif isinstance(value, dict): # Python dict, not Neo4j Node/Relationship as dict
            return {k: self._convert_neo4j_value_to_json_serializable(v) for k, v in value.items()}
        elif isinstance(value, (str, int, float, bool)) or value is None:
            return value
        else:
            kg_logger.warning(f"Encountered an unhandled Neo4j type ({type(value)}), converting to string: {str(value)}")
            return str(value)

    def execute_cypher_query_sync(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self._driver:
            kg_logger.error("Neo4j driver not initialized in execute_cypher_query_sync.")
            return []
        
        kg_logger.debug(f"Executing SYNC Cypher: {query} with params: {parameters}")
        results_list: List[Dict[str, Any]] = []
        try:
            with self._driver.session(database=self.NEO4J_DATABASE) as session:
                result_obj: Result = session.run(query, parameters)
                for record_instance in result_obj: # record_instance is a Record object
                    record_as_dict = {}
                    for key, value in record_instance.items():
                        record_as_dict[key] = self._convert_neo4j_value_to_json_serializable(value)
                    results_list.append(record_as_dict)
            kg_logger.debug(f"SYNC Cypher executed. Records: {len(results_list)}")
        except Exception as e:
            kg_logger.error(f"Failed to execute SYNC Cypher query: {query}. Error: {e}", exc_info=True)
        return results_list

    def _format_neo4j_record_for_retrieval(self, record_data: Dict[str, Any]) -> str:
        """
        将单条已转换为纯Python字典的Neo4j记录格式化为一段描述性文本。
        """
        parts = []
        for key, value in record_data.items():
            if isinstance(value, dict):
                # 尝试提取节点的text和label属性，或关系的type
                if '_labels' in value and 'text' in value: # 假设是转换后的Node
                    parts.append(f"{key}({value['text']}:{'/'.join(value['_labels'])})")
                elif '_type' in value: # 假设是转换后的Relationship
                    rel_props_str = ", ".join([f"{k_prop}: {v_prop}" for k_prop, v_prop in value.items() if not k_prop.startswith('_')])
                    parts.append(f"{key}(TYPE={value['_type']}{', PROPS=[' + rel_props_str + ']' if rel_props_str else ''})")
                else: # 其他字典
                    # 为了避免过长的输出，可以限制value的打印长度或深度
                    value_str = json.dumps(value, ensure_ascii=False, indent=None)
                    if len(value_str) > 100: value_str = value_str[:100] + "..."
                    parts.append(f"{key}: {value_str}")
            elif value is not None:
                parts.append(f"{key}: {str(value)}")
        
        return " | ".join(parts) if parts else "No specific details found in this record."

    async def retrieve_with_llm_cypher(self, query: str, top_k: int = 3) -> List[RetrievedDocument]:
        kg_logger.info(f"Starting KG retrieval with LLM-generated Cypher for query: '{query}', top_k: {top_k}")
        if not self._driver:
            kg_logger.warning("Neo4j driver not initialized. Cannot perform KG query.")
            return []

        kg_logger.info(f"Calling LLM to generate Cypher query using new schema...")
        cypher_query = await self.llm_cypher_generator_func(
            user_question=query,
            kg_schema_description=NEW_KG_SCHEMA_DESCRIPTION 
        )
        kg_logger.info(f"LLM generated Cypher query:\n---\n{cypher_query}\n---")

        if not cypher_query or cypher_query == "无法生成Cypher查询。":
            kg_logger.warning("LLM could not generate a valid Cypher query.")
            return []

        results = self.execute_cypher_query_sync(cypher_query)
        
        retrieved_docs = []
        for record_dict in results[:top_k]:
            content = self._format_neo4j_record_for_retrieval(record_dict)
            retrieved_docs.append(
                RetrievedDocument(
                    source_type="knowledge_graph",
                    content=content,
                    score=1.0, 
                    metadata={"cypher_query": cypher_query, "original_query": query}
                )
            )
        kg_logger.info(f"Retrieved {len(retrieved_docs)} documents from KG using LLM-generated Cypher.")
        return retrieved_docs

    def get_entity_details_manual(self, entity_text: str, entity_type_attr: Optional[str] = None) -> List[Dict[str, Any]]:
        if entity_type_attr:
            cypher = "MATCH (e:ExtractedEntity {text: $text, label: $label_attr}) RETURN e"
            params = {"text": entity_text, "label_attr": entity_type_attr.upper()}
        else:
            cypher = "MATCH (e:ExtractedEntity {text: $text}) RETURN e"
            params = {"text": entity_text}
        return self.execute_cypher_query_sync(cypher, params)

    def get_relations_manual(self, entity_text: str, entity_type_attr: str, relation_type: Optional[str] = None, direction: str = "BOTH") -> List[Dict[str, Any]]:
        rel_type_cypher = f":{relation_type.upper()}" if relation_type else "r" # 关系类型转大写
        
        if direction.upper() == "OUT":
            rel_clause = f"-[{rel_type_cypher}]->"
        elif direction.upper() == "IN":
            rel_clause = f"<-[{rel_type_cypher}]-"
        else: # BOTH
            rel_clause = f"-[{rel_type_cypher}]-"
            
        cypher = (
            f"MATCH (e:ExtractedEntity {{text: $text, label: $label_attr}}){rel_clause}(neighbor:ExtractedEntity) "
            f"RETURN e as entity, {rel_type_cypher if not relation_type else 'r'} as relationship, neighbor as related_entity"
        )
        # 如果 relation_type 为 None, Cypher 中 r 会匹配任何关系类型，但我们返回时仍用 'r' 作为键
        # 如果 relation_type 指定了，Cypher 中会用具体的类型，返回时也用 'r' 作为键
        # 为了统一，我们让 RETURN 语句中的关系变量总是 'r'
        if relation_type: # 如果指定了关系类型，确保返回的变量是 r
             cypher = (
                f"MATCH (e:ExtractedEntity {{text: $text, label: $label_attr}})-[r:{relation_type.upper()}]"
                f"{'->' if direction.upper() == 'OUT' else '<-' if direction.upper() == 'IN' else '-'}"
                f"(neighbor:ExtractedEntity) "
                f"RETURN e as entity, r as relationship, neighbor as related_entity"
            )
        else: # 如果没有指定关系类型，匹配任何关系
            cypher = (
                f"MATCH (e:ExtractedEntity {{text: $text, label: $label_attr}})-[r]-"
                f"{'(neighbor:ExtractedEntity)' if direction.upper() != 'BOTH' else '(neighbor:ExtractedEntity)'} " # 确保neighbor被定义
                f"RETURN e as entity, r as relationship, neighbor as related_entity"
            )


        params = {"text": entity_text, "label_attr": entity_type_attr.upper()}
        return self.execute_cypher_query_sync(cypher, params)

async def main_test():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    kg_logger.info("--- KGRetriever New Test ---")
    try:
        kg_retriever = KGRetriever() 
        if kg_retriever._driver is None:
            print("Failed to connect to Neo4j. Exiting test.")
            return 

        print("\n--- Test 1: Get entity details for '张三' (PERSON) ---")
        zhang_san_details = kg_retriever.get_entity_details_manual("张三", "PERSON")
        if zhang_san_details:
            print(json.dumps(zhang_san_details, indent=2, ensure_ascii=False))
        else:
            print("'张三' (PERSON) not found.")

        print("\n--- Test 2: Get 'WORKS_AT' relations for '张三' (PERSON) ---")
        zhang_san_relations = kg_retriever.get_relations_manual("张三", "PERSON", relation_type="WORKS_AT", direction="OUT")
        if zhang_san_relations:
            print(f"Found {len(zhang_san_relations)} WORKS_AT relations for '张三':")
            for item in zhang_san_relations:
                print(json.dumps(item, indent=2, ensure_ascii=False))
        else:
            print("No WORKS_AT relations found for '张三'.")

        print("\n--- Test 3: Retrieve with LLM-generated Cypher for '张三在哪里工作？' ---")
        llm_results_zs = await kg_retriever.retrieve_with_llm_cypher("张三在哪里工作？")
        if llm_results_zs:
            print(f"LLM query results for '张三在哪里工作？':")
            for doc in llm_results_zs:
                print(f"  Content: {doc.content}")
                print(f"  Metadata: {doc.metadata}")
        else:
            print("LLM query for '张三在哪里工作？' returned no results or failed.")

        print("\n--- Test 4: Retrieve with LLM-generated Cypher for '项目Alpha的文档编写任务分配给了谁？' ---")
        llm_results_task = await kg_retriever.retrieve_with_llm_cypher("项目Alpha的文档编写任务分配给了谁？")
        if llm_results_task:
            print(f"LLM query results for '项目Alpha的文档编写任务分配给了谁？':")
            for doc in llm_results_task:
                print(f"  Content: {doc.content}")
                print(f"  Metadata: {doc.metadata}")
        else:
            print("LLM query for '项目Alpha的文档编写任务分配给了谁？' returned no results or failed.")

        kg_retriever.close()
    except Exception as e:
        print(f"An error occurred during the KGRetriever test: {e}")

if __name__ == '__main__':
    asyncio.run(main_test())