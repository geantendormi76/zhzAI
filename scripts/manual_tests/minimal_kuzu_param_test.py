# /home/zhz/zhz_agent/scripts/manual_tests/minimal_kuzu_param_test.py
import kuzu
import os
import shutil

DB_PATH = "./test_kuzu_param_db" # 使用一个全新的临时数据库

def main():
    print(f"KuzuDB Python version: {kuzu.__version__}")

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print(f"Removed existing test DB at {DB_PATH}")

    db = None
    conn = None
    try:
        print(f"Creating new test DB at {DB_PATH}...")
        db = kuzu.Database(DB_PATH)
        conn = kuzu.Connection(db)
        print("Test DB and connection created.")

        print("\n--- Creating Node Table ---")
        conn.execute("CREATE NODE TABLE User(name STRING, age INT64, PRIMARY KEY (name))")
        print("Node table 'User' created.")

        print("\n--- Testing Parameterized Insert via PreparedStatement ---")
        insert_query_cypher = "CREATE (u:User {name: $input_name, age: $input_age})"
        params_to_insert = {"input_name": "Alice", "input_age": 30}
        
        try:
            print(f"Preparing statement: {insert_query_cypher}")
            prepared_insert = conn.prepare(insert_query_cypher)
            print(f"Executing prepared statement with params: {params_to_insert}")
            # 直接传递参数字典
            conn.execute(prepared_insert, params_to_insert) 
            print("Parameterized insert successful.")
        except Exception as e_insert:
            print(f"ERROR during parameterized insert: {e_insert}")
            import traceback
            traceback.print_exc()
            return # 如果插入失败，后续查询无意义

        print("\n--- Verifying Inserted Data ---")
        select_query_cypher = "MATCH (u:User {name: $find_name}) RETURN u.name, u.age"
        params_to_select = {"find_name": "Alice"}
        try:
            print(f"Preparing statement: {select_query_cypher}")
            prepared_select = conn.prepare(select_query_cypher)
            print(f"Executing prepared statement with params: {params_to_select}")
            query_result = conn.execute(prepared_select, params_to_select)
            
            if query_result.has_next():
                row = query_result.get_next()
                print(f"Query Result: Name={row}, Age={row}") # Kuzu返回的是元组
                if row == "Alice" and row == 30:
                    print("Data verification successful!")
                else:
                    print(f"Data verification FAILED. Expected ('Alice', 30), got {row}")
            else:
                print("Data verification FAILED: No data returned.")
            query_result.close()
        except Exception as e_select:
            print(f"ERROR during data verification: {e_select}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"An overall error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- Test Finished ---")
        if os.path.exists(DB_PATH):
            print(f"Cleaning up test DB at {DB_PATH}")
            # 在finally块中确保db和conn对象存在才尝试del
            if conn is not None:
                del conn # KuzuDB的Connection没有close()
            if db is not None:
                del db   # 依赖析构函数关闭数据库
            # shutil.rmtree(DB_PATH) # 暂时注释掉，方便查看数据库文件

if __name__ == "__main__":
    main()
