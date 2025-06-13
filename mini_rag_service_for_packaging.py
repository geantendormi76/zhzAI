# /home/zhz/zhz_agent/mini_rag_service_for_packaging.py
import os
import sys
import logging
from fastapi import FastAPI
import uvicorn
import duckdb
import time

# --- sentence-transformers Reranker ---
from sentence_transformers.cross_encoder import CrossEncoder

# --- llama-cpp-python Embedding ---
from llama_cpp import Llama, LlamaGrammar # LlamaGrammar 可能不需要，但 Llama 是必须的
# ---

# --- 加载 .env 文件 ---
from dotenv import load_dotenv
project_root = os.path.abspath(os.path.dirname(__file__))
dotenv_path = os.path.join(project_root, '.env')
if os.path.exists(dotenv_path):
    logging.info(f"Loading environment variables from: {dotenv_path}")
    load_dotenv(dotenv_path=dotenv_path, override=True)
else:
    logging.warning(f".env file not found at {dotenv_path}. Relying on system environment variables.")
# ---

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 解决 PyInstaller 打包时的路径问题 ---
def get_resource_path(relative_path: str) -> str:
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    resource_full_path = os.path.join(base_path, relative_path)
    logger.info(f"Resource path requested for '{relative_path}', resolved to: '{resource_full_path}'")
    if not os.path.exists(resource_full_path):
        logger.warning(f"Resource path does NOT exist: {resource_full_path}")
        if os.path.exists(base_path):
            logger.info(f"Contents of base_path ('{base_path}'): {os.listdir(base_path)[:10]}")
        else:
            logger.warning(f"Base path ('{base_path}') also does not exist.")
    return resource_full_path

# --- FastAPI 应用定义 ---
app = FastAPI(title="Mini RAG Service for Packaging Test")

# --- 全局变量 ---
duckdb_conn = None
reranker_model = None
embedding_model = None # <--- 新增：Embedding 模型对象
embedding_model_dimensions = 0 # <--- 新增：Embedding 模型的维度

# --- 模型和数据路径配置 ---
DUCKDB_TEMP_DIR = "packaging_test_data"
DUCKDB_FILE_NAME = "mini_test.db"

# Embedding 模型路径和参数 (从 .env 获取)
EMBEDDING_MODEL_PATH_ENV = os.getenv("EMBEDDING_MODEL_PATH")
EMBEDDING_N_CTX_ENV = os.getenv("EMBEDDING_N_CTX", "2048") # 提供默认值
EMBEDDING_N_GPU_LAYERS_ENV = os.getenv("EMBEDDING_N_GPU_LAYERS", "0") # 提供默认值


@app.on_event("startup")
async def startup_event():
    global duckdb_conn, reranker_model, embedding_model, embedding_model_dimensions
    logger.info("FastAPI application startup...")
    
    # --- DuckDB 初始化 (保持不变) ---
    db_dir_path = get_resource_path(DUCKDB_TEMP_DIR)
    db_full_path: str
    if not os.path.exists(db_dir_path):
        logger.info(f"Directory {db_dir_path} does not exist. Attempting to create.")
        try:
            os.makedirs(db_dir_path)
            logger.info(f"Successfully created directory for DuckDB: {db_dir_path}")
            db_full_path = os.path.join(db_dir_path, DUCKDB_FILE_NAME)
        except Exception as e:
            logger.error(f"Failed to create directory {db_dir_path}: {e}", exc_info=True)
            db_full_path = ":memory:"
            logger.warning("Falling back to in-memory DuckDB due to directory creation failure.")
    else:
        logger.info(f"Directory {db_dir_path} already exists.")
        db_full_path = os.path.join(db_dir_path, DUCKDB_FILE_NAME)
    logger.info(f"Attempting to connect to DuckDB at: {db_full_path}")
    try:
        duckdb_conn = duckdb.connect(database=db_full_path, read_only=False)
        logger.info(f"Successfully connected to DuckDB: {db_full_path}")
        duckdb_conn.execute("CREATE SCHEMA IF NOT EXISTS test_schema;")
        duckdb_conn.execute("CREATE TABLE IF NOT EXISTS test_schema.items (id INTEGER, name VARCHAR);")
        table_check = duckdb_conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'test_schema' AND table_name = 'items';").fetchone()
        if table_check:
            logger.info("DuckDB table 'test_schema.items' confirmed or created.")
            duckdb_conn.execute("DELETE FROM test_schema.items;")
            duckdb_conn.execute("INSERT INTO test_schema.items VALUES (1, 'Test Item Package 1'), (2, 'Test Item Package 2');")
            logger.info("Inserted test data into DuckDB.")
        else:
            logger.error("Failed to confirm or create DuckDB table 'test_schema.items'.")
    except Exception as e:
        logger.error(f"Failed to initialize or connect to DuckDB: {e}", exc_info=True)
        duckdb_conn = None

    # --- 加载 Reranker 模型 (保持不变) ---
    logger.info("Attempting to load Reranker model...")
    reranker_model_packaged_path = get_resource_path(os.path.join("mini_service_payload", "Qwen3-Reranker-0.6B-seq-cls"))
    actual_reranker_path_to_load = None

    if os.path.isdir(reranker_model_packaged_path):
        logger.info(f"Found potential packaged Reranker model at: {reranker_model_packaged_path}")
        actual_reranker_path_to_load = reranker_model_packaged_path
    else:
        logger.warning(f"Packaged Reranker model not found at '{reranker_model_packaged_path}'. Falling back to environment variable RERANKER_MODEL_PATH.")
        actual_reranker_path_to_load = os.getenv("RERANKER_MODEL_PATH")

    if actual_reranker_path_to_load and os.path.isdir(actual_reranker_path_to_load):
        try:
            logger.info(f"Loading Reranker model from: {actual_reranker_path_to_load}")
            reranker_model = CrossEncoder(actual_reranker_path_to_load, max_length=512)
            logger.info("Reranker model loaded successfully.")
            logger.info("Performing a test inference with Reranker model...")
            test_sentence_pairs = [['Query: What is FastAPI?', 'FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.']]
            scores = reranker_model.predict(test_sentence_pairs)
            logger.info(f"Reranker test inference scores: {scores}")
        except Exception as e:
            logger.error(f"Failed to load or test Reranker model from '{actual_reranker_path_to_load}': {e}", exc_info=True)
            reranker_model = None
    else:
        logger.error(f"Reranker model path ('{actual_reranker_path_to_load}') is not a valid directory. Cannot load Reranker.")
        reranker_model = None

    # --- 加载 Embedding 模型 ---
    logger.info("Attempting to load Embedding model...")
    embedding_model_packaged_path = get_resource_path(os.path.join("mini_service_payload", "Qwen3-Embedding-0.6B-Q8_0.gguf"))
    actual_embedding_path_to_load = None

    if os.path.isfile(embedding_model_packaged_path):
        logger.info(f"Found potential packaged Embedding model at: {embedding_model_packaged_path}")
        actual_embedding_path_to_load = embedding_model_packaged_path
    else:
        logger.warning(f"Packaged Embedding model not found at '{embedding_model_packaged_path}'. Falling back to environment variable EMBEDDING_MODEL_PATH.")
        actual_embedding_path_to_load = EMBEDDING_MODEL_PATH_ENV # 使用之前从env获取的

    if actual_embedding_path_to_load and os.path.isfile(actual_embedding_path_to_load):
        try:
            logger.info(f"Loading Embedding model from: {actual_embedding_path_to_load}")
            logger.info(f"Embedding params: n_ctx={EMBEDDING_N_CTX_ENV}, n_gpu_layers={EMBEDDING_N_GPU_LAYERS_ENV}")
            embedding_model = Llama(
                model_path=actual_embedding_path_to_load,
                n_ctx=int(EMBEDDING_N_CTX_ENV),
                n_gpu_layers=int(EMBEDDING_N_GPU_LAYERS_ENV),
                embedding=True,
                verbose=False 
            )
            embedding_model_dimensions = embedding_model.n_embd()
            logger.info(f"Embedding model loaded successfully. Dimensions: {embedding_model_dimensions}")
            logger.info("Performing a test embedding with Embedding model...")
            test_text_to_embed = "This is a test sentence for embedding."
            embedding_vector = embedding_model.embed(test_text_to_embed)
            logger.info(f"Embedding test successful. Vector length: {len(embedding_vector)}. First 5 dims: {embedding_vector[:5]}")
        except Exception as e:
            logger.error(f"Failed to load or test Embedding model from '{actual_embedding_path_to_load}': {e}", exc_info=True)
            embedding_model = None
            embedding_model_dimensions = 0
    else:
        logger.error(f"Embedding model path ('{actual_embedding_path_to_load}') is not a valid file. Cannot load Embedding model.")
        embedding_model = None
        embedding_model_dimensions = 0


@app.on_event("shutdown")
async def shutdown_event():
    global duckdb_conn, reranker_model, embedding_model # <--- 添加 embedding_model
    logger.info("FastAPI application shutdown...")
    if duckdb_conn:
        try:
            duckdb_conn.close()
            logger.info("DuckDB connection closed.")
        except Exception as e:
            logger.error(f"Error closing DuckDB connection: {e}")
    if reranker_model:
        logger.info("Clearing Reranker model.")
        del reranker_model
    if embedding_model: # <--- 清理 embedding_model
        logger.info("Clearing Embedding model (llama.cpp object).")
        # llama-cpp-python 对象在 del 时会自动释放资源，无需显式 close
        del embedding_model


@app.get("/")
async def read_root():
    logger.info("Root endpoint '/' accessed.")
    return {"message": "Mini RAG Service is running!"}

@app.get("/duckdb_test")
async def test_duckdb():
    # ... (保持不变)
    global duckdb_conn
    logger.info("/duckdb_test endpoint accessed.")
    if not duckdb_conn:
        logger.error("DuckDB connection is not available.")
        return {"error": "DuckDB not initialized"}
    try:
        result = duckdb_conn.execute("SELECT * FROM test_schema.items;").fetchall()
        logger.info(f"DuckDB query result: {result}")
        return {"data": result}
    except Exception as e:
        logger.error(f"Error querying DuckDB: {e}", exc_info=True)
        return {"error": str(e)}


@app.get("/reranker_test")
async def test_reranker():
    # ... (保持不变)
    global reranker_model
    logger.info("/reranker_test endpoint accessed.")
    if not reranker_model:
        logger.error("Reranker model is not available.")
        return {"error": "Reranker model not loaded"}
    try:
        query = "What is the capital of France?"
        docs = [
            "Paris is the capital and most populous city of France.",
            "Berlin is the capital of Germany.",
            "France is a country in Western Europe."
        ]
        sentence_pairs = [[query, doc] for doc in docs]
        scores = reranker_model.predict(sentence_pairs)
        logger.info(f"Reranker test with query '{query}', scores: {scores}")
        return {"query": query, "ranked_documents": [{"doc": d, "score": float(s)} for d, s in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)]}
    except Exception as e:
        logger.error(f"Error during Reranker test: {e}", exc_info=True)
        return {"error": str(e)}


@app.get("/embedding_test") # <--- 新增测试接口
async def test_embedding():
    global embedding_model, embedding_model_dimensions
    logger.info("/embedding_test endpoint accessed.")
    if not embedding_model:
        logger.error("Embedding model is not available.")
        return {"error": "Embedding model not loaded"}
    try:
        text_to_embed = "Hello, world! This is an embedding test."
        embedding = embedding_model.embed(text_to_embed)
        logger.info(f"Embedding test for '{text_to_embed}'. Vector length: {len(embedding)}. First 5 dims: {embedding[:5]}")
        return {
            "text": text_to_embed, 
            "embedding_preview": embedding[:10], # 只返回前10个维度作为预览
            "embedding_length": len(embedding),
            "expected_dimensions": embedding_model_dimensions
        }
    except Exception as e:
        logger.error(f"Error during Embedding test: {e}", exc_info=True)
        return {"error": str(e)}

if __name__ == "__main__":
    logger.info("Starting Uvicorn server directly for mini_rag_service_for_packaging.py")
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="info")