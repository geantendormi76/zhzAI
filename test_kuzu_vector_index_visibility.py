import kuzu
import pandas as pd
import os
import shutil
import time
import numpy as np

# --- 配置 ---
DB_DIR_NAME = "test_kuzu_visibility_db_0.10.0"
DB_PATH = os.path.join(os.getcwd(), DB_DIR_NAME)
EMBEDDING_DIM = 1024

# --- 新增：指定你手动下载的扩展文件路径 ---
# 请将此路径修改为您实际存放 libvector.kuzu_extension 的路径
MANUAL_EXTENSION_FILE_PATH = "/home/zhz/.kuzu/extension/0.10.0/linux_amd64/vector/libvector.kuzu_extension" # 这是根据您的截图推断的路径

# --- 清理和准备目录 ---
if os.path.exists(DB_PATH):
    print(f"--- Cleaning up existing test database at: {DB_PATH} ---")
    shutil.rmtree(DB_PATH)
os.makedirs(DB_PATH, exist_ok=True)
print(f"--- Test database will be created at: {DB_PATH} ---")

# --- 新增：创建 extensions 子目录并复制扩展文件 ---
extensions_dir_in_db = os.path.join(DB_PATH, "extensions")
os.makedirs(extensions_dir_in_db, exist_ok=True)
print(f"--- Created extensions directory: {extensions_dir_in_db} ---")

if os.path.exists(MANUAL_EXTENSION_FILE_PATH):
    target_path_libvector = os.path.join(extensions_dir_in_db, "libvector.kuzu_extension")
    target_path_vector = os.path.join(extensions_dir_in_db, "vector.kuzu_extension")
    try:
        shutil.copy2(MANUAL_EXTENSION_FILE_PATH, target_path_libvector)
        print(f"Copied '{MANUAL_EXTENSION_FILE_PATH}' to '{target_path_libvector}'")
        shutil.copy2(MANUAL_EXTENSION_FILE_PATH, target_path_vector) # 也复制一份名为 vector.kuzu_extension
        print(f"Copied '{MANUAL_EXTENSION_FILE_PATH}' to '{target_path_vector}'")
    except Exception as e_copy:
        print(f"ERROR: Could not copy extension file: {e_copy}")
        # 如果复制失败，后续 LOAD 可能会失败，但我们仍然尝试运行
else:
    print(f"ERROR: Manual extension file not found at '{MANUAL_EXTENSION_FILE_PATH}'. LOAD VECTOR will likely fail if Kuzu doesn't find it elsewhere.")


def create_schema_and_data(db_path: str):
    print("\n--- Session 1: Creating schema, loading data, and creating vector index ---")
    db_writer = None
    conn_writer = None
    try:
        print(f"[Writer] Opening database in READ_WRITE mode at {db_path}...")
        db_writer = kuzu.Database(db_path)
        conn_writer = kuzu.Connection(db_writer)
        print("[Writer] Database and connection opened.")

        # print("[Writer] Attempting to INSTALL VECTOR extension...") # <--- 注释掉 INSTALL
        # try:
        #     conn_writer.execute("INSTALL VECTOR;") 
        #     print("[Writer] INSTALL VECTOR command finished.") 
        # except Exception as e_install:
        #     print(f"[Writer] INSTALL VECTOR failed: {e_install}")
        
        print("[Writer] Attempting to LOAD VECTOR extension (expecting to load from local file)...")
        conn_writer.execute("LOAD VECTOR;")
        print("[Writer] VECTOR extension loaded.")

        print("[Writer] Creating NODE TABLE ExtractedEntity...")
        conn_writer.execute(f"""
            CREATE NODE TABLE ExtractedEntity (
                id_prop STRING PRIMARY KEY,
                text STRING,
                label STRING,
                embedding FLOAT[{EMBEDDING_DIM}])
        """)
        print("[Writer] ExtractedEntity table created.")

        print("[Writer] Loading data into ExtractedEntity...")
        entities_to_load = [
            {"id_prop": "node1", "text": "This is node one", "label": "DOC", "embedding": np.random.rand(EMBEDDING_DIM).astype(np.float32).tolist()},
            {"id_prop": "node2", "text": "Another node here", "label": "DOC", "embedding": np.random.rand(EMBEDDING_DIM).astype(np.float32).tolist()}
        ]
        for entity in entities_to_load:
            conn_writer.execute(
                "CREATE (e:ExtractedEntity {id_prop: $id, text: $text, label: $label, embedding: $embedding})",
                {"id": entity["id_prop"], "text": entity["text"], "label": entity["label"], "embedding": entity["embedding"]}
            )
        print(f"[Writer] Loaded {len(entities_to_load)} entities.")

        print("[Writer] Creating vector index 'entity_embedding_idx'...")
        conn_writer.execute(f"""
            CALL CREATE_VECTOR_INDEX(
                'ExtractedEntity',
                'entity_embedding_idx',
                'embedding',
                metric := 'cosine'
            )
        """)
        print("[Writer] Vector index creation command executed.")

        print("[Writer] Verifying index creation in current session (Writer)...")
        result_writer_show_idx = conn_writer.execute("CALL SHOW_INDEXES() RETURN *;")
        df_writer_show_idx = pd.DataFrame(result_writer_show_idx.get_as_df())
        print("[Writer] Indexes visible to WRITER session:")
        print(df_writer_show_idx.to_string())
        if not df_writer_show_idx.empty and 'entity_embedding_idx' in df_writer_show_idx['index name'].tolist():
            print("[Writer] SUCCESS: 'entity_embedding_idx' IS VISIBLE in WRITER session.")
        else:
            print("[Writer] FAILURE: 'entity_embedding_idx' IS NOT VISIBLE in WRITER session.")

        print("[Writer] Executing CHECKPOINT...")
        conn_writer.execute("CHECKPOINT;")
        print("[Writer] CHECKPOINT executed.")

    except Exception as e:
        print(f"[Writer] ERROR in Session 1: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn_writer:
            print("[Writer] Closing connection...")
            del conn_writer
        if db_writer:
            print("[Writer] Closing database (via del)...")
            del db_writer
        print("--- Session 1 Finished ---")

# ... (verify_index_in_new_session 函数保持不变) ...
def verify_index_in_new_session(db_path: str, read_only_mode: bool):
    mode_str = "READ_ONLY" if read_only_mode else "READ_WRITE"
    print(f"\n--- Session 2: Verifying index in a NEW session ({mode_str} mode) ---")
    db_reader = None
    conn_reader = None
    try:
        print(f"[Reader] Opening database in {mode_str} mode at {db_path}...")
        db_reader = kuzu.Database(db_path, read_only=read_only_mode)
        conn_reader = kuzu.Connection(db_reader)
        print("[Reader] Database and connection opened.")

        print("[Reader] Loading VECTOR extension...")
        conn_reader.execute("LOAD VECTOR;")
        print("[Reader] VECTOR extension loaded.")

        print("[Reader] Verifying index visibility in NEW session (Reader)...")
        result_reader_show_idx = conn_reader.execute("CALL SHOW_INDEXES() RETURN *;")
        df_reader_show_idx = pd.DataFrame(result_reader_show_idx.get_as_df())
        print(f"[Reader] Indexes visible to READER session ({mode_str}):")
        print(df_reader_show_idx.to_string())
        
        index_found_in_reader = False
        if not df_reader_show_idx.empty and 'index name' in df_reader_show_idx.columns and \
           'entity_embedding_idx' in df_reader_show_idx['index name'].tolist():
            print(f"[Reader] SUCCESS: 'entity_embedding_idx' IS VISIBLE in READER session ({mode_str}).")
            index_found_in_reader = True
        else:
            print(f"[Reader] FAILURE: 'entity_embedding_idx' IS NOT VISIBLE in READER session ({mode_str}).")
            if 'index name' not in df_reader_show_idx.columns and not df_reader_show_idx.empty:
                 print(f"[Reader] Note: 'index name' column not found in SHOW_INDEXES output. Columns: {df_reader_show_idx.columns.tolist()}")


        if index_found_in_reader:
            print("[Reader] Attempting to query vector index...")
            query_vector = np.random.rand(EMBEDDING_DIM).astype(np.float32).tolist()
            query_result = conn_reader.execute(
                """
                CALL QUERY_VECTOR_INDEX('ExtractedEntity', 'entity_embedding_idx', $query_vec, 2)
                YIELD node, distance
                RETURN node.id_prop, distance
                ORDER BY distance ASC
                """,
                {"query_vec": query_vector}
            )
            print("[Reader] Vector query executed. Results (if any):")
            results_list = []
            while query_result.has_next():
                results_list.append(query_result.get_next())
            if results_list:
                for row in results_list:
                    print(f"  {row}")
            else:
                print("  No results from vector query.")
        else:
            print("[Reader] Skipping vector query as index was not visible.")


    except Exception as e:
        print(f"[Reader] ERROR in Session 2 ({mode_str}): {e}")
        import traceback
        traceback.print_exc()
    finally:
        if conn_reader:
            print(f"[Reader] Closing connection ({mode_str})...")
            del conn_reader
        if db_reader:
            print(f"[Reader] Closing database ({mode_str}) (via del)...")
            del db_reader
        print(f"--- Session 2 ({mode_str}) Finished ---")

if __name__ == "__main__":
    print(f"KuzuDB Python Client Version: {kuzu.__version__}")
    
    create_schema_and_data(DB_PATH)
    
    print("\nWaiting for 5 seconds before starting Session 2 (to simulate delay and allow OS to flush)...")
    time.sleep(5)
    
    verify_index_in_new_session(DB_PATH, read_only_mode=True)
    
    print("\nWaiting for 2 seconds before starting another Session 2 (Read-Write)...")
    time.sleep(2)

    verify_index_in_new_session(DB_PATH, read_only_mode=False)

    print("\n--- Test Script Finished ---")