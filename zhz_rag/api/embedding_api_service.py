# zhz_rag/api/embedding_api_service.py

import os
import sys
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv # <--- 确保导入

# --- START: 覆盖这部分代码 ---
# 无论从哪里运行，都确保能找到项目根目录的 .env 文件
# __file__ -> /home/zhz/zhz_agent/zhz_rag/api/embedding_api_service.py
# .parents[0] -> .../api
# .parents[1] -> .../zhz_rag
# .parents[2] -> /home/zhz/zhz_agent
try:
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    dotenv_path = project_root / '.env'
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)
        print(f"EmbeddingApiService: Successfully loaded .env from absolute path: {dotenv_path}")
    else:
        print(f"EmbeddingApiService: .env file not found at {dotenv_path}. Relying on system environment variables.")
except Exception as e:
    print(f"EmbeddingApiService: Error loading .env file: {e}")
# --- END: 覆盖结束 ---
# --- 配置路径以导入 llama_cpp ---
# 确保 llama_cpp 可被找到
try:
    from llama_cpp import Llama
except ImportError:
    print("FATAL: llama-cpp-python is not installed. Please run 'pip install llama-cpp-python'")
    sys.exit(1)

# --- 日志配置 ---
api_logger = logging.getLogger("EmbeddingApiServiceLogger")
api_logger.setLevel(logging.INFO)
if not api_logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    api_logger.addHandler(handler)
    api_logger.propagate = False

# --- Pydantic 模型定义 ---
class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="A list of texts to be embedded.")

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(description="A list of embedding vectors.")
    dimensions: int = Field(description="The dimension of the embedding vectors.")

# --- 全局变量用于缓存模型 ---
embedding_model: Optional[Llama] = None
model_dimension: Optional[int] = None

# --- FastAPI 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model, model_dimension
    api_logger.info("--- Embedding API Service: Initializing... ---")
    
    model_path = os.getenv("EMBEDDING_MODEL_PATH")
    n_ctx = int(os.getenv("EMBEDDING_N_CTX", "2048"))
    n_gpu_layers = int(os.getenv("EMBEDDING_N_GPU_LAYERS", "0"))

    if not model_path or not os.path.exists(model_path):
        api_logger.error(f"FATAL: Embedding model path not found or not set in .env: {model_path}")
        raise FileNotFoundError(f"Embedding model not found at {model_path}")

    try:
        api_logger.info(f"Loading embedding model from: {model_path}")
        embedding_model = Llama(
            model_path=model_path,
            embedding=True,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )
        model_dimension = embedding_model.n_embd()
        api_logger.info(f"Embedding model loaded successfully. Dimension: {model_dimension}")
    except Exception as e:
        api_logger.error(f"Failed to load embedding model: {e}", exc_info=True)
        embedding_model = None
        model_dimension = None

    yield

    api_logger.info("--- Embedding API Service: Shutting down. ---")
    # 清理模型（如果需要）
    if embedding_model:
        del embedding_model
        embedding_model = None

# --- FastAPI 应用实例 ---
app = FastAPI(
    title="Standalone Embedding API Service",
    description="A dedicated API service for generating text embeddings using a GGUF model.",
    version="1.0.0",
    lifespan=lifespan
)

# --- API 端点 ---
@app.post("/embed", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    if not embedding_model or not model_dimension:
        raise HTTPException(status_code=503, detail="Embedding model is not available.")
    
    if not request.texts:
        return EmbeddingResponse(embeddings=[], dimensions=model_dimension)
    
    try:
        api_logger.info(f"Received request to embed {len(request.texts)} texts.")
        response = embedding_model.create_embedding(request.texts)
        
        embeddings_raw = response.get('data', [])
        final_embeddings: List[List[float]] = []
        for item in embeddings_raw:
            embedding_data = item.get('embedding', [])
            
            # --- 核心修复：处理可能的嵌套列表 ---
            # 检查embedding_data是否是列表，并且其第一个元素也是列表
            if isinstance(embedding_data, list) and embedding_data and isinstance(embedding_data[0], list):
                # 如果是嵌套列表，我们只取第一个内部列表
                embedding_vector_raw = embedding_data[0]
                api_logger.debug("Detected a nested list in embedding output, taking the first element.")
            else:
                # 否则，我们假设它已经是我们期望的扁平列表
                embedding_vector_raw = embedding_data
            
            # 确保向量中的每个元素都是float
            embedding_vector = [float(x) for x in embedding_vector_raw]
            final_embeddings.append(embedding_vector)
            # --- 修复结束 ---

        api_logger.info(f"Successfully generated {len(final_embeddings)} embeddings.")
        return EmbeddingResponse(embeddings=final_embeddings, dimensions=model_dimension)
    except Exception as e: 
        api_logger.error(f"Error during embedding creation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings: {e}")

@app.get("/health")
async def health_check():
    if embedding_model and model_dimension:
        return {"status": "ok", "model_loaded": True, "dimension": model_dimension}
    else:
        return {"status": "error", "model_loaded": False, "message": "Embedding model failed to load."}

if __name__ == "__main__":
    # 默认运行在 8089 端口，避免与现有服务冲突
    port = int(os.getenv("EMBEDDING_API_PORT", "8089"))
    api_logger.info(f"Starting Embedding API Service on port {port}...")
    uvicorn.run("zhz_rag.api.embedding_api_service:app", host="0.0.0.0", port=port, reload=False)