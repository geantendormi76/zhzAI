import duckdb
import os
import traceback

db_path = os.getenv(
    "DUCKDB_KG_FILE_PATH", 
    "/home/zhz/zhz_agent/zhz_rag/stored_data/duckdb_knowledge_graph.db"
)
wal_path = db_path + ".wal"

print(f"--- Consolidating DuckDB WAL File (Attempt 2) ---")
print(f"Target database file: {db_path}")

if not os.path.exists(db_path):
    print(f"ERROR: Database file '{db_path}' not found.")
    exit()

con = None
try:
    # 1. 以读写模式连接到数据库文件
    print(f"Opening {db_path} in read-write mode...")
    # 在这里，我们希望 connect() 内部的 WAL 重放能够成功
    # 如果它在此时就需要 vss, 我们可能需要更底层的配置方式
    # 或者，如果 connect() 成功了，但 WAL 还没完全处理完，后续的 INSTALL/LOAD 可能帮助它完成
    
    config_options = {
        "allow_unsigned_extensions": "true", # 如果vss扩展是社区或本地构建且未签名
        # "autoinstall_known_extensions": "true", # 尝试让DuckDB更积极地自动安装已知扩展
        # "autoload_known_extensions": "true"   # 尝试让DuckDB更积极地自动加载已知扩展
        # 注意：以上两个 autoload/autoinstall 选项在较新版本中可能默认开启或行为有所不同
    }
    print(f"Connecting with config: {config_options}")
    con = duckdb.connect(database=db_path, read_only=False, config=config_options)
    print("Connection object obtained.")

    # 2. 立即尝试 INSTALL 和 LOAD VSS 扩展
    # 使用 FORCE INSTALL 来确保它尝试获取最新兼容版本并覆盖
    print("Attempting to FORCE INSTALL vss extension...")
    con.execute("FORCE INSTALL vss;") 
    print("FORCE INSTALL vss executed.")
    
    print("Attempting to LOAD vss extension...")
    con.execute("LOAD vss;")
    print("LOAD vss executed.")
    
    # 3. 开启 HNSW 持久化
    try:
        con.execute("SET hnsw_enable_experimental_persistence=true;")
        print("HNSW experimental persistence enabled (or re-confirmed).")
    except Exception as e_set_persist:
        print(f"Notice: Could not set hnsw_enable_experimental_persistence: {e_set_persist}")

    # 4. 执行一个简单的查询来鼓励WAL的写入/合并
    print("Executing a simple query to potentially trigger WAL processing...")
    con.execute("SELECT count(*) FROM duckdb_tables();") # 查询任何元数据表都可以
    print("Simple query executed.")

    # 5. 关闭连接，这将自动触发checkpoint，合并WAL文件
    print("Closing connection to trigger WAL consolidation...")
    con.close()
    con = None 
    print(f"Connection to {db_path} closed. WAL consolidation should have occurred.")

    if os.path.exists(wal_path):
        print(f"WARNING: WAL file '{wal_path}' still exists after consolidation attempt.")
    else:
        print(f"SUCCESS: WAL file '{wal_path}' no longer exists, consolidation likely successful.")

except Exception as e:
    print(f"\n--- AN UNEXPECTED ERROR OCCURRED during WAL consolidation ---")
    print(f"DETAILS: {type(e).__name__}: {e}")
    traceback.print_exc()
finally:
    if con:
        print("Ensuring connection is closed in finally block...")
        con.close()
    print("\n--- WAL Consolidation Script Finished ---")