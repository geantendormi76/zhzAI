# debug_kuzu_visibility.py
import kuzu
import os
import shutil
import time
import pandas as pd

DB_PATH = "./zhz_rag/stored_data/kuzu_test_visibility_db" # 使用一个新的测试路径

def setup_and_verify_schema():
    print(f"--- Phase 1: Setup and Initial Verification ---")
    if os.path.exists(DB_PATH):
        print(f"Removing existing database at {DB_PATH}")
        shutil.rmtree(DB_PATH)
    
    db_setup = None
    conn_setup = None
    try:
        print(f"Creating new database at {DB_PATH}")
        db_setup = kuzu.Database(DB_PATH, read_only=False)
        conn_setup = kuzu.Connection(db_setup)
        print("Database and connection created for setup.")

        ddl_queries = [
            "CREATE NODE TABLE IF NOT EXISTS ExtractedEntity (id_prop STRING, text STRING, label STRING, PRIMARY KEY (id_prop))",
            "CREATE REL TABLE IF NOT EXISTS WorksAt (FROM ExtractedEntity TO ExtractedEntity)",
            "CREATE REL TABLE IF NOT EXISTS AssignedTo (FROM ExtractedEntity TO ExtractedEntity)"
        ]
        for query in ddl_queries:
            print(f"Executing DDL: {query}")
            conn_setup.execute(query)
        print("DDL execution complete.")

        print("Executing CHECKPOINT...")
        conn_setup.execute("CHECKPOINT;")
        print("CHECKPOINT complete.")

        # Phase 1 Verification
        print("Verifying tables immediately after creation (same session)...")
        try:
            # 尝试 SHOW TABLES;
            print("Attempting 'SHOW TABLES;' in Phase 1...")
            result = conn_setup.execute("SHOW TABLES;") # <--- 修改这里
            df = pd.DataFrame(result.get_as_df())
            print(f"Tables found using 'SHOW TABLES;' (Phase 1):\n{df}")
            # 检查 'name' 列是否存在，并且 'ExtractedEntity' 是否在其中
            if not df.empty and 'name' in df.columns and "ExtractedEntity" in df["name"].tolist():
                print("Phase 1 Verification (SHOW TABLES;): ExtractedEntity table FOUND.")
            elif not df.empty and 'name' not in df.columns:
                print("Phase 1 Verification (SHOW TABLES;): 'name' column not found in SHOW TABLES result.")
                print(f"Columns available in SHOW TABLES result: {df.columns.tolist()}")
            else: # df is empty or ExtractedEntity not in name column
                print("Phase 1 Verification (SHOW TABLES;): ExtractedEntity table NOT FOUND or result was empty.")
        except Exception as e:
            print(f"Error during Phase 1 verification with 'SHOW TABLES;': {e}")
            print("Falling back to direct query for ExtractedEntity in Phase 1...")
            try:
                result_direct = conn_setup.execute("MATCH (e:ExtractedEntity) RETURN count(e) AS entity_count;")
                df_direct = pd.DataFrame(result_direct.get_as_df())
                print(f"Direct query result (Phase 1):\n{df_direct}")
                count_val = df_direct['entity_count'].iloc[0] if not df_direct.empty else -1
                print(f"Phase 1 Verification (Direct Query): ExtractedEntity table SEEMS TO EXIST (count: {count_val}).")
            except Exception as e_direct_phase1:
                print(f"Error during Phase 1 direct query verification: {e_direct_phase1}")

    except Exception as e:
        print(f"Error during Phase 1 setup: {e}")
    finally:
        if conn_setup:
            print("Closing setup connection.")
            # conn_setup.close() # Kuzu Connection 没有显式 close
        if db_setup:
            print("Deleting setup database instance reference (will trigger close).")
            del db_setup # 依赖 __del__
        print("--- Phase 1 Complete ---")

def verify_in_new_session():
    print(f"\n--- Phase 2: Verification in a New Session ---")
    if not os.path.exists(DB_PATH):
        print(f"Database at {DB_PATH} does not exist. Cannot perform Phase 2.")
        return

    db_verify = None
    conn_verify = None
    try:
        print(f"Opening existing database at {DB_PATH} for verification.")
        db_verify = kuzu.Database(DB_PATH, read_only=False) # 打开同一个数据库
        conn_verify = kuzu.Connection(db_verify)
        print("Database and connection created for verification.")

        # Phase 2 Verification
        print("Verifying tables in new session...")
        try:
            # 尝试 SHOW TABLES;
            print("Attempting 'SHOW TABLES;' in Phase 2...")
            result = conn_verify.execute("SHOW TABLES;") # <--- 修改这里
            df = pd.DataFrame(result.get_as_df())
            print(f"Tables found using 'SHOW TABLES;' (Phase 2):\n{df}")
            if not df.empty and 'name' in df.columns and "ExtractedEntity" in df["name"].tolist():
                print("Phase 2 Verification (SHOW TABLES;): ExtractedEntity table FOUND.")
            elif not df.empty and 'name' not in df.columns:
                print("Phase 2 Verification (SHOW TABLES;): 'name' column not found in SHOW TABLES result.")
                print(f"Columns available in SHOW TABLES result: {df.columns.tolist()}")
            else: # df is empty or ExtractedEntity not in name column
                print("Phase 2 Verification (SHOW TABLES;): ExtractedEntity table NOT FOUND or result was empty.")
        except Exception as e:
            print(f"Error during Phase 2 verification with 'SHOW TABLES;': {e}")
            print("Falling back to direct query for ExtractedEntity in Phase 2...")
            try:
                result_direct = conn_verify.execute("MATCH (e:ExtractedEntity) RETURN count(e) AS entity_count;")
                df_direct = pd.DataFrame(result_direct.get_as_df())
                print(f"Direct query result (Phase 2):\n{df_direct}")
                count_val = df_direct['entity_count'].iloc[0] if not df_direct.empty else -1
                print(f"Phase 2 Verification (Direct Query): ExtractedEntity table SEEMS TO EXIST (count: {count_val}).")
            except Exception as e_direct_phase2:
                print(f"Error during Phase 2 direct query verification: {e_direct_phase2}")

    except Exception as e:
        print(f"Error during Phase 2 setup: {e}")
    finally:
        if conn_verify:
            print("Closing verification connection.")
        if db_verify:
            print("Deleting verification database instance reference.")
            del db_verify
        print("--- Phase 2 Complete ---")

if __name__ == "__main__":
    print(f"Kuzu Python client version: {kuzu.__version__}")
    setup_and_verify_schema()
    print("\nWaiting a moment before trying to open in a new session (simulating process switch)...")
    time.sleep(2) # 短暂等待，模拟进程切换的间隙
    verify_in_new_session()