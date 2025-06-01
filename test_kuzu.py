import kuzu
import os
import pandas as pd # 导入 pandas 以便更好地显示结果

# 1. 定义数据库文件的存储路径 (指向 Dagster 创建的数据库)
DB_PATH = "./zhz_rag/stored_data/kuzu_default_db" # 确保这是正确的相对路径或绝对路径

def print_query_results(conn, query_string, description="Query"):
    print(f"\n--- {description} ---")
    print(f"Executing Query: {query_string.strip()}")
    try:
        results = conn.execute(query_string)
        # 使用 pandas DataFrame 来显示结果，更美观
        df = results.get_as_df()
        if df.empty:
            print("  Query returned no results.")
        else:
            print(df.to_string()) # to_string() 会打印整个 DataFrame
        results.close() # 关闭结果集
    except Exception as e:
        print(f"  Error executing query: {e}")

def main():
    if not os.path.exists(DB_PATH) or not os.path.isdir(DB_PATH):
        print(f"KuzuDB directory not found at: {DB_PATH}")
        print("Please ensure the Dagster pipeline has run successfully and created the database.")
        return

    db = None
    conn = None
    try:
        # 1. 初始化数据库和连接 (以只读模式打开，因为我们只是验证)
        # 如果需要执行写操作进行测试，可以将 read_only 改为 False
        print(f"Connecting to KuzuDB at: {os.path.abspath(DB_PATH)}")
        db = kuzu.Database(DB_PATH, read_only=True) 
        conn = kuzu.Connection(db)
        print("Successfully connected to KuzuDB.")

        # 2. 验证 Schema 是否存在 (可选，因为 Dagster 资产已验证)
        #    但我们可以用 conn._get... 方法再次确认
        try:
            node_tables = conn._get_node_table_names()
            rel_tables_info = conn._get_rel_table_names()
            rel_table_names = [info['name'] for info in rel_tables_info]
            print(f"\nDetected Node Tables: {node_tables}")
            print(f"Detected Rel Tables: {rel_table_names}")
            
            expected_tables = ["ExtractedEntity", "WorksAt", "AssignedTo"]
            all_expected_found = True
            for tbl in expected_tables:
                if tbl not in (node_tables + rel_table_names):
                    print(f"Warning: Expected table '{tbl}' not found in detected schema.")
                    all_expected_found = False
            if all_expected_found:
                print("All expected tables are present in the schema.")
                
        except Exception as e_schema:
            print(f"Error retrieving schema information: {e_schema}")


        # 3. 查询数据
        
        # 查询所有 ExtractedEntity 节点 (限制数量以防过多)
        print_query_results(conn, 
                            "MATCH (e:ExtractedEntity) RETURN e.id_prop, e.text, e.label LIMIT 10", 
                            description="All ExtractedEntity Nodes (Sample)")

        # 根据我们 doc1.txt 的内容 "项目Alpha的文档编写任务分配给了张三。张三在谷歌工作。"
        # 我们期望的实体：
        # - {text: "项目Alpha的文档编写任务", label: "TASK"}
        # - {text: "张三", label: "PERSON"}
        # - {text: "谷歌", label: "ORGANIZATION"}
        # 我们期望的关系：
        # - ("项目Alpha的文档编写任务")-[:ASSIGNED_TO]->("张三")
        # - ("张三")-[:WORKS_AT]->("谷歌")

        # 查询特定实体 "张三"
        print_query_results(conn, 
                            "MATCH (p:ExtractedEntity {text: '张三', label: 'PERSON'}) RETURN p.id_prop, p.text, p.label", 
                            description="Details for Entity '张三'")

        # 查询 "张三" 在哪里工作
        print_query_results(conn, 
                            """
                            MATCH (p:ExtractedEntity {text: '张三', label: 'PERSON'})
                                  -[:WorksAt]->
                                  (o:ExtractedEntity {label: 'ORGANIZATION'})
                            RETURN p.text AS person, o.text AS organization
                            """, 
                            description="'张三' WorksAt Which Organization?")

        # 查询 "项目Alpha的文档编写任务" 分配给了谁
        print_query_results(conn, 
                            """
                            MATCH (t:ExtractedEntity {text: '项目Alpha的文档编写任务', label: 'TASK'})
                                  -[:AssignedTo]->
                                  (p:ExtractedEntity {label: 'PERSON'})
                            RETURN t.text AS task, p.text AS assignee
                            """, 
                            description="'项目Alpha的文档编写任务' AssignedTo Whom?")
                            
        # 查询所有关系 (限制数量)
        # 注意：KuzuDB 的 Cypher 中，匿名关系可能需要更具体的写法，或者通过节点匹配路径
        # 尝试获取所有 WorksAt 关系
        print_query_results(conn, 
                            """
                            MATCH (e1:ExtractedEntity)-[r:WorksAt]->(e2:ExtractedEntity)
                            RETURN e1.text AS from_entity, type(r) AS rel_type, e2.text AS to_entity
                            LIMIT 10
                            """, 
                            description="All WorksAt Relationships (Sample)")

        # 尝试获取所有 AssignedTo 关系
        print_query_results(conn, 
                            """
                            MATCH (e1:ExtractedEntity)-[r:AssignedTo]->(e2:ExtractedEntity)
                            RETURN e1.text AS from_entity, type(r) AS rel_type, e2.text AS to_entity
                            LIMIT 10
                            """, 
                            description="All AssignedTo Relationships (Sample)")
                            
        # 计算总实体数和总关系数
        try:
            result_node_count = conn.execute("MATCH (n:ExtractedEntity) RETURN count(n) AS total_nodes")
            node_count_df = result_node_count.get_as_df()
            if not node_count_df.empty:
                print(f"\nTotal ExtractedEntity nodes: {node_count_df['total_nodes'].iloc[0]}")
            result_node_count.close()

            # 计算关系总数可能需要分别计算每种关系类型的数量然后相加
            # 或者如果 Kuzu 支持 MATCH ()-[r]-() RETURN count(r)
            result_rel_count_worksat = conn.execute("MATCH ()-[r:WorksAt]->() RETURN count(r) AS total_worksat")
            worksat_count_df = result_rel_count_worksat.get_as_df()
            if not worksat_count_df.empty:
                print(f"Total WorksAt relationships: {worksat_count_df['total_worksat'].iloc[0]}")
            result_rel_count_worksat.close()
            
            result_rel_count_assignedto = conn.execute("MATCH ()-[r:AssignedTo]->() RETURN count(r) AS total_assignedto")
            assignedto_count_df = result_rel_count_assignedto.get_as_df()
            if not assignedto_count_df.empty:
                print(f"Total AssignedTo relationships: {assignedto_count_df['total_assignedto'].iloc[0]}")
            result_rel_count_assignedto.close()

        except Exception as e_count:
            print(f"Error getting counts: {e_count}")


    except Exception as e:
        print(f"An error occurred with KuzuDB: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nKuzuDB data verification script finished.")
        # conn 和 db 对象会在 try 块结束或发生异常时，由于其作用域结束而被 Python 的垃圾回收器处理
        # KuzuDB 的 Python 对象设计为在 __del__ 中释放资源
        if conn:
            # Kuzu Connection 对象没有显式的 close() 方法
            pass
        if db:
            # Kuzu Database 对象也没有显式的 close() 方法，依赖 __del__
            del db # 可以显式 del 来尝试触发 __del__
        print("KuzuDB resources (if any were held by script) should be released.")

if __name__ == '__main__':
    main()