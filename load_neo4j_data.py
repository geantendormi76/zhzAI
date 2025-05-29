# zhz_agent/load_neo4j_data.py
import json
import os
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv
import traceback

load_dotenv() # 确保加载 .env 文件中的NEO4J凭证

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") # 您需要确保这个密码是正确的

DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
SAMPLE_KG_PATH = os.path.join(DATA_PATH, "sample_kg.json")

def clear_database(driver):
    """清除数据库中的所有节点和关系"""
    with driver.session(database="neo4j") as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("Cleared all nodes and relationships from the database.")

def create_constraints(driver):
    """创建一些基本约束，确保节点属性的唯一性（如果适用）"""
    with driver.session(database="neo4j") as session:
        try:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (pr:Project) REQUIRE pr.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Region) REQUIRE r.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (prod:Product) REQUIRE prod.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
            # SalesAmount 通常不需要唯一约束，因为它可能重复（例如不同区域同一时期的销售）
            print("Ensured constraints are created (or already exist).")
        except Exception as e:
            print(f"Error creating constraints: {e}")


def load_data(driver, kg_data):
    """根据kg_data中的facts加载数据到Neo4j"""
    facts = kg_data.get("facts", [])
    
    with driver.session(database="neo4j") as session:
        entities_to_create = set()
        node_types_from_schema = { # 定义了主要实体的标签和它们的主要标识属性
            "Person": "name", "Project": "name", "Region": "name", 
            "Product": "name", "Document": "id", "Idea": "name" # 新增Idea类型
        }

        for fact in facts:
            subject_name = fact.get("subject")
            object_name = fact.get("object")
            fact_type = fact.get("type", "")

            subject_label = None
            # 基于fact_type或其他逻辑推断subject_label
            if "person_" in fact_type: subject_label = "Person"
            elif "region_" in fact_type: subject_label = "Region"
            elif "product_" in fact_type: subject_label = "Product"
            # ... 其他类型的映射 ...
            
            if subject_label and subject_name:
                prop_name = node_types_from_schema.get(subject_label, "name")
                entities_to_create.add((subject_label, prop_name, subject_name))

            object_label = None
            if not fact_type.endswith("_amount"): # 不是直接的销售额事实
                if "_project" in fact_type: object_label = "Project"
                elif "_product" in fact_type: object_label = "Product"
                elif "_document" in fact_type: object_label = "Document"
                elif "_idea" in fact_type: object_label = "Idea" # 新增对Idea类型的处理
                # ... 其他类型的映射 ...

                if object_label and object_name:
                    prop_name = node_types_from_schema.get(object_label, "name") # Document会用id, Idea会用name
                    entities_to_create.add((object_label, prop_name, object_name))
        
        for label, prop, value in entities_to_create:
            if value is not None:
                query = f"MERGE (n:{label} {{{prop}: $value}})"
                session.run(query, value=value)
                print(f"Merged node: ({label} {{{prop}: '{value}'}})")

        for fact in facts:
            s_name = fact.get("subject")
            rel = fact.get("relation")
            o_name = fact.get("object")
            fact_type = fact.get("type", "")
            period = fact.get("period")

            if fact_type == "region_sales_amount" and period:
                session.run("MERGE (r:Region {name: $s_name})", s_name=s_name)
                try:
                    # ... (销售额解析逻辑不变) ...
                    if isinstance(o_name, str) and '万元' in o_name:
                        numeric_val_str = o_name.replace('万元', '').strip()
                        numeric_val = float(numeric_val_str)
                        unit_val = '万元'
                    # ... (其他单位解析) ...
                    else:
                        numeric_val = float(o_name) # 尝试直接转换
                        unit_val = None 
                    
                    query = """
                    MATCH (r:Region {name: $s_name})
                    CREATE (sa:SalesAmount {numeric_amount: $num_val, period: $period, unit: $unit_val})
                    CREATE (r)-[:HAS_SALES_AMOUNT]->(sa)
                    """
                    session.run(query, s_name=s_name, num_val=numeric_val, period=period, unit_val=unit_val)
                    print(f"Created SalesAmount for {s_name}, {period}: {numeric_val} {unit_val or ''}")
                except ValueError:
                    print(f"Could not parse sales amount: {o_name} for {s_name}, {period}. Skipping this SalesAmount fact.")
                
            elif s_name and rel and o_name: 
                s_label, o_label = None, None
                s_prop, o_prop = "name", "name" 

                # --- 更精确的标签和属性推断 ---
                if fact_type == "person_project" and rel == "WORKS_ON":
                    s_label, o_label = "Person", "Project"
                elif fact_type == "person_idea" and rel == "PROPOSED_IDEA": # 新增
                    s_label, o_label = "Person", "Idea"
                elif fact_type == "region_product" and rel == "HAS_SALES_PRODUCT": # 假设type是 region_product
                    s_label, o_label = "Region", "Product"
                elif fact_type == "product_document" and rel == "RELATED_TO":
                    s_label, o_label = "Product", "Document"
                    o_prop = "id" # Document用id匹配
                # 您可以根据您的fact_type添加更多精确的映射规则

                if s_label and o_label:
                    query = f"""
                    MATCH (s:{s_label} {{{s_prop}: $s_name}})
                    MATCH (o:{o_label} {{{o_prop}: $o_name}})
                    MERGE (s)-[:{rel}]->(o)
                    """
                    session.run(query, s_name=s_name, o_name=o_name)
                    print(f"Merged relationship: ({s_label} {{{s_prop}:'{s_name}'}})-[:{rel}]->({o_label} {{{o_prop}:'{o_name}'}})")
                else:
                    print(f"Could not determine labels for fact: {fact} (s_label: {s_label}, o_label: {o_label}). Relationship not created.")
            else:
                print(f"Skipping incomplete fact: {fact}")


if __name__ == "__main__":
    driver = None
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Successfully connected to Neo4j.")
        
        clear_database(driver) # 清空数据库
        create_constraints(driver) # 创建约束

        with open(SAMPLE_KG_PATH, 'r', encoding='utf-8') as f:
            kg_data_to_load = json.load(f)
        
        load_data(driver, kg_data_to_load)
        
        print("\nData loading process completed.")
        print("You can now verify the data in Neo4j Browser (http://localhost:7474).")
        print("Example query to check SalesAmount:")
        print("MATCH (r:Region)-[:HAS_SALES_AMOUNT]->(sa:SalesAmount) RETURN r.name, sa.numeric_amount, sa.unit, sa.period")
        print("Example query to check Person-Project:")
        print("MATCH (p:Person)-[:WORKS_ON]->(proj:Project) RETURN p.name, proj.name")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        if driver:
            driver.close()
            print("Neo4j connection closed.")