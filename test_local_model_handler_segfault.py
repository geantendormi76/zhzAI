# test_local_model_handler_segfault.py
import os
import logging
import sys
from dotenv import load_dotenv

# --- 添加项目根目录到 sys.path ---
# 获取当前脚本文件所在的目录 (假设此脚本放在 zhz_agent 根目录下)
_current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 将 zhz_agent 目录添加到 sys.path
sys.path.insert(0, _current_script_dir)
print(f"Added to sys.path: {_current_script_dir}")
print(f"Current sys.path: {sys.path}")


# --- 日志配置 (简化版，确保能看到 LocalModelHandler 的日志) ---
# 配置根日志记录器以捕获所有日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger("SegfaultTest")


# --- 动态加载 .env ---
# .env 文件通常在项目根目录
_dotenv_path = os.path.join(_current_script_dir, ".env") # 假设脚本在项目根目录

if os.path.exists(_dotenv_path):
    load_dotenv(dotenv_path=_dotenv_path)
    logger.info(f"Successfully loaded .env file from: {_dotenv_path}")
else:
    logger.warning(f".env file not found at {_dotenv_path}. Relying on system environment variables or defaults.")
    load_dotenv()


# --- 确保 LocalModelHandler 可以被导入 ---
try:
    from zhz_rag.llm.local_model_handler import LocalModelHandler
except ImportError as e:
    logger.error(f"Failed to import LocalModelHandler. Ensure PYTHONPATH is set correctly or script is in the right location. Error: {e}")
    sys.exit(1)


def run_test():
    embedding_gguf_model_path = os.getenv("EMBEDDING_MODEL_PATH")
    n_ctx_embed = int(os.getenv("EMBEDDING_N_CTX", 2048))
    n_gpu_layers_embed = int(os.getenv("EMBED_N_GPU_LAYERS", 0)) # 注意 .env 中是 EMBEDDING_N_GPU_LAYERS

    if not embedding_gguf_model_path:
        logger.error("EMBEDDING_MODEL_PATH not found in environment variables. Cannot run test.")
        return

    logger.info("--- Initializing LocalModelHandler for test ---")
    try:
        handler = LocalModelHandler(
            embedding_model_path=embedding_gguf_model_path,
            n_ctx_embed=n_ctx_embed,
            n_gpu_layers_embed=n_gpu_layers_embed,
            pooling_type_embed=2 # CLS
        )
        if not handler.embedding_model:
            logger.error("Failed to initialize embedding model in LocalModelHandler.")
            return
        logger.info("LocalModelHandler initialized successfully for embedding.")
    except Exception as e:
        logger.error(f"Error initializing LocalModelHandler: {e}", exc_info=True)
        return

    test_query = "项目的主要目标是什么？"
    processed_test_query = test_query + "<|endoftext|>" if not test_query.endswith("<|endoftext|>") else test_query

    # Test 1: _blocking_embed_query_internal (the one causing segfault in API)
    logger.info(f"\n--- Testing _blocking_embed_query_internal with: '{test_query}' ---")
    try:
        query_embedding = handler._blocking_embed_query_internal(processed_test_query)
        if query_embedding:
            logger.info(f"SUCCESS: _blocking_embed_query_internal returned embedding of dim {len(query_embedding)}")
            # logger.info(f"Embedding (first 5): {query_embedding[:5]}")
        else:
            logger.error("FAILURE: _blocking_embed_query_internal returned empty or None.")
    except Exception as e:
        logger.error(f"ERROR during _blocking_embed_query_internal: {e}", exc_info=True)
        # 如果这里发生段错误，Python的try-except可能捕获不到，进程会直接退出

    logger.info("-" * 50)

    # Test 2: _blocking_embed_documents_internal (for comparison, expects a list)
    logger.info(f"\n--- Testing _blocking_embed_documents_internal with: ['{test_query}'] ---")
    try:
        doc_embeddings = handler._blocking_embed_documents_internal([processed_test_query])
        if doc_embeddings and doc_embeddings[0]:
            logger.info(f"SUCCESS: _blocking_embed_documents_internal returned embedding of dim {len(doc_embeddings[0])}")
            # logger.info(f"Embedding (first 5 of first doc): {doc_embeddings[0][:5]}")
        else:
            logger.error("FAILURE: _blocking_embed_documents_internal returned empty or None for the list.")
    except Exception as e:
        logger.error(f"ERROR during _blocking_embed_documents_internal: {e}", exc_info=True)

    logger.info("\n--- Test script finished ---")

if __name__ == "__main__":
    run_test()
