import duckdb
import os

db_path = os.path.join("/home/zhz/zhz_agent", "zhz_rag", "stored_data", "test_manual.db")
db_dir = os.path.dirname(db_path)
os.makedirs(db_dir, exist_ok=True) # 确保目录存在

try:
    print(f"Attempting to connect to DuckDB at: {db_path}")
    con = duckdb.connect(database=db_path, read_only=False)
    print("Successfully connected to DuckDB and created/opened the file.")
    con.execute("CREATE TABLE IF NOT EXISTS test_table (id INTEGER, value VARCHAR);")
    print("Successfully executed a CREATE TABLE command.")
    con.close()
    print("Connection closed.")
    # 清理测试文件
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Cleaned up test file: {db_path}")
    if os.path.exists(db_path + ".wal"): # DuckDB 可能会创建 .wal 文件
        os.remove(db_path + ".wal")
        print(f"Cleaned up WAL file: {db_path}.wal")

except Exception as e:
    print(f"Error connecting to or operating on DuckDB: {e}")
    import traceback
    traceback.print_exc()