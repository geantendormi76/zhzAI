# test_kuzu.py
import kuzu
import os
import pandas as pd
import re # <--- 添加导入
import unicodedata # <--- 添加导入

def normalize_text_for_id(text: str) -> str:
    if not isinstance(text, str):
        return str(text)
    try:
        normalized_text = unicodedata.normalize('NFKD', text)
        normalized_text = normalized_text.lower()
        normalized_text = normalized_text.strip()
        normalized_text = re.sub(r'\s+', ' ', normalized_text)
        return normalized_text
    except Exception as e:
        return text

DB_PATH = "/home/zhz/zhz_agent/zhz_rag/stored_data/kuzu_default_db"

def run_queries(conn: kuzu.Connection):
    print("\n--- Querying Data ---")

    # 规范化查询条件中的文本
    task_text_orig = "项目Alpha的文档编写任务"
    person_text_orig = "张三"
    org_text_orig = "谷歌"

    task_text_norm = normalize_text_for_id(task_text_orig)
    person_text_norm = normalize_text_for_id(person_text_orig)
    org_text_norm = normalize_text_for_id(org_text_orig)

    print(f"Normalized for query: '{task_text_orig}' -> '{task_text_norm}'")
    print(f"Normalized for query: '{person_text_orig}' -> '{person_text_norm}'")
    print(f"Normalized for query: '{org_text_orig}' -> '{org_text_norm}'")


    queries_to_run = {
        "Total ExtractedEntity Nodes": "MATCH (n:ExtractedEntity) RETURN count(n) AS total_entities;",
        "All ExtractedEntity Nodes (Limit 5)": "MATCH (n:ExtractedEntity) RETURN n.id_prop, n.text, n.label LIMIT 5;",
        # 使用规范化后的文本进行查询
        "Specific Entity (张三 - normalized)": f"MATCH (n:ExtractedEntity {{text: '{person_text_norm}', label: 'PERSON'}}) RETURN n.id_prop, n.text, n.label;",
        "Specific Entity (项目Alpha... - normalized)": f"MATCH (n:ExtractedEntity {{text: '{task_text_norm}', label: 'TASK'}}) RETURN n.id_prop, n.text, n.label;",
        "Specific Entity (谷歌 - normalized)": f"MATCH (n:ExtractedEntity {{text: '{org_text_norm}', label: 'ORGANIZATION'}}) RETURN n.id_prop, n.text, n.label;",
        
        "Total WorksAt Relationships": "MATCH ()-[r:WorksAt]->() RETURN count(r) AS total_works_at_rels;",
        "Total AssignedTo Relationships": "MATCH ()-[r:AssignedTo]->() RETURN count(r) AS total_assigned_to_rels;",
        
        # 关系查询也使用规范化文本（如果条件中包含文本）
        "Who works at 谷歌? (normalized)": f"MATCH (p:ExtractedEntity {{label: 'PERSON'}})-[r:WorksAt]->(o:ExtractedEntity {{text: '{org_text_norm}', label: 'ORGANIZATION'}}) RETURN p.text AS person_name;",
        "Where does 张三 work? (normalized)": f"MATCH (p:ExtractedEntity {{text: '{person_text_norm}', label: 'PERSON'}})-[r:WorksAt]->(o:ExtractedEntity {{label: 'ORGANIZATION'}}) RETURN o.text AS organization_name;",
        "What task is assigned to 张三? (normalized)": f"MATCH (t:ExtractedEntity {{label: 'TASK'}})-[r:AssignedTo]->(p:ExtractedEntity {{text: '{person_text_norm}', label: 'PERSON'}}) RETURN t.text AS task_name;",
        "Who is the task '项目Alpha...' assigned to? (normalized)": f"MATCH (t:ExtractedEntity {{text: '{task_text_norm}', label: 'TASK'}})-[r:AssignedTo]->(p:ExtractedEntity {{label: 'PERSON'}}) RETURN p.text AS person_name;",
        
        "All WorksAt Relationships (Source and Target Text)": "MATCH (src:ExtractedEntity)-[r:WorksAt]->(tgt:ExtractedEntity) RETURN src.text AS source_text, src.label AS source_label, tgt.text AS target_text, tgt.label AS target_label;",
        "All AssignedTo Relationships (Source and Target Text)": "MATCH (src:ExtractedEntity)-[r:AssignedTo]->(tgt:ExtractedEntity) RETURN src.text AS source_text, src.label AS source_label, tgt.text AS target_text, tgt.label AS target_label;",
    }

    for description, query_str in queries_to_run.items():
        print(f"\nExecuting Query: {description}")
        print(f"Cypher: {query_str}")
        try:
            results = conn.execute(query_str)
            # 使用 pandas 显示结果更美观
            df = pd.DataFrame(results.get_as_df())
            if not df.empty:
                print(df.to_string())
            else:
                print("  Query returned no results.")
            results.close() # 记得关闭结果集
        except Exception as e:
            print(f"  Error executing query: {e}")

def main():
    print(f"Kuzu Python client version: {kuzu.__version__}")
    print(f"Attempting to connect to KuzuDB at: {DB_PATH}")

    if not os.path.exists(DB_PATH):
        print(f"ERROR: Database directory not found at {DB_PATH}")
        print("Please ensure the Dagster KuzuDB pipeline has run successfully and created the database.")
        return

    db = None
    conn = None
    try:
        # --- 连接到已存在的数据库 (只读模式足够用于查询) ---
        # 如果需要执行写操作（例如，在测试中临时修改），可以使用 read_only=False
        db = kuzu.Database(DB_PATH, read_only=True)
        conn = kuzu.Connection(db)
        print(f"Successfully connected to KuzuDB. Database path: {os.path.abspath(DB_PATH)}")

        # --- 列出所有表 (使用我们从研究报告中知道的方法) ---
        print("\n--- Listing Tables ---")
        try:
            node_tables = conn._get_node_table_names()
            print(f"Node Tables: {node_tables}")
            rel_tables_info = conn._get_rel_table_names()
            rel_tables = [info['name'] for info in rel_tables_info]
            print(f"Rel Tables: {rel_tables}")
            all_tables = node_tables + rel_tables
            print(f"All Tables: {all_tables}")

            # 验证核心表是否存在
            expected_tables = ["ExtractedEntity", "WorksAt", "AssignedTo"]
            missing = [t for t in expected_tables if t not in all_tables]
            if not missing:
                print("Core tables (ExtractedEntity, WorksAt, AssignedTo) are present.")
            else:
                print(f"WARNING: Missing core tables: {missing}")

        except Exception as e_list_tables:
            print(f"Error listing tables: {e_list_tables}")
            print("This might indicate an issue with the KuzuDB connection or version compatibility for these internal methods.")

        # --- 执行查询 ---
        if conn: # 确保连接有效
            run_queries(conn)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nKuzuDB test script finished.")
        # Kuzu Connection 没有显式的 close() 方法
        # Kuzu Database 对象在其 __del__ 方法中处理关闭
        if db is not None:
            del db # 确保数据库对象被垃圾回收，从而关闭

if __name__ == "__main__":
    main()