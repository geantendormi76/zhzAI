# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/zhz_rag_pipeline/resources.py
import dagster as dg
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Iterator
import httpx
import json
import litellm
import os
import kuzu # 确保导入 kuzu
import shutil
from contextlib import contextmanager
from pydantic import Field as PydanticField
from pydantic import PrivateAttr
import asyncio

try:
    from zhz_rag.llm.local_model_handler import LocalModelHandler
except ImportError as e:
    print(f"FATAL: Could not import LocalModelHandler from zhz_rag.llm.local_model_handler. "
          f"Please ensure 'zhz_rag' package is installed and accessible in the Dagster environment's PYTHONPATH. Error: {e}")
    raise


# --- 新的 GGUFEmbeddingResource 定义 ---
class GGUFEmbeddingResourceConfig(dg.Config):
    embedding_model_path: str = PydanticField(
        description="Path to the GGUF embedding model file."
    )
    n_ctx: int = PydanticField(
        default=2048,
        description="Context size for the embedding model."
    )
    n_gpu_layers: int = PydanticField(
        default=0,
        description="Number of GPU layers to offload for the embedding model."
    )

class GGUFEmbeddingResource(dg.ConfigurableResource):
    embedding_model_path: str
    n_ctx: int
    n_gpu_layers: int

    _model_handler: LocalModelHandler = PrivateAttr(default=None)
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)

    # --- 添加辅助方法 ---
    def _run_async_in_new_loop(self, coro):
        """辅助函数，在当前线程中创建一个新事件循环来运行协程。"""
        try:
            # 尝试获取当前线程已有的循环，如果没有则创建一个新的
            # 这对于 Dagster 这种可能在不同线程中执行资源初始化的场景更稳健
            loop = asyncio.get_event_loop_policy().new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
            loop.close()
            return result
        except RuntimeError as e:
            if "cannot run event loop while another loop is running" in str(e):
                # 如果当前线程已经有一个正在运行的循环 (理论上 Dagster 资产执行时不会)
                # 这种情况比较复杂，直接在同步代码中调用异步代码的最佳实践是使用 asyncio.to_thread
                # 但这里资源方法是被同步资产调用的，我们期望它返回实际结果。
                # 对于 Dagster 资源，更常见的是资源方法本身是同步的，如果它们需要调用异步操作，
                # 它们需要自己管理事件循环的运行。
                self._logger.error(f"GGUFEmbeddingResource: Nested event loop issue in _run_async_in_new_loop: {e}")
                # 另一种选择是，如果 LocalModelHandler 的方法是纯CPU密集型，
                # 也许 LocalModelHandler 的方法应该设计成同步的，然后在需要异步的地方用 to_thread。
                # 但我们之前为了 LlamaCppEmbeddingFunction 的 asyncio.run 改成了 async.
                # 这里我们先尝试创建一个新的循环。
            self._logger.error(f"GGUFEmbeddingResource: Error running async task in new loop: {e}", exc_info=True)
            raise # 将原始错误重新抛出
    # --- 结束添加 ---

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger = context.log
        self._logger.info(f"Initializing GGUFEmbeddingResource...")
        
        if not os.path.exists(self.embedding_model_path):
            error_msg = f"GGUF embedding model file not found at: {self.embedding_model_path}"
            self._logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            # 只初始化嵌入功能
            self._model_handler = LocalModelHandler(
                embedding_model_path=self.embedding_model_path,
                n_ctx_embed=self.n_ctx,
                n_gpu_layers_embed=self.n_gpu_layers
            )
            if not self._model_handler.embedding_model:
                raise RuntimeError("Failed to load GGUF embedding model inside LocalModelHandler.")
            
            self._logger.info(f"GGUFEmbeddingResource initialized successfully. Model: {self.embedding_model_path}")
        except Exception as e:
            self._logger.error(f"Failed to initialize GGUFEmbeddingResource: {e}", exc_info=True)
            raise

    def encode(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """
        提供与 SentenceTransformer.encode 类似的接口。
        忽略 kwargs 以保持兼容性。
        """
        if not hasattr(self, '_model_handler'):
            raise RuntimeError("GGUFEmbeddingResource is not initialized. Call setup_for_execution first.")
        
        logger_instance = self._logger if self._logger else dg.get_dagster_logger()
        logger_instance.debug(f"GGUFEmbeddingResource: Encoding {len(texts)} texts using LocalModelHandler.")

        # --- 修改调用方式 ---
        # LocalModelHandler.embed_documents 是 async def
        # GGUFEmbeddingResource.encode 是同步方法，在 Dagster 资产中被同步调用
        # 我们需要在这里同步地执行异步的 embed_documents 方法
        try:
            # 使用辅助方法来运行异步的 embed_documents
            embeddings = self._run_async_in_new_loop(
                self._model_handler.embed_documents(texts)
            )
            return embeddings
        except Exception as e:
             logger_instance.error(f"GGUFEmbeddingResource: Error during encode by calling LocalModelHandler: {e}", exc_info=True)
             # 根据上游资产的期望，这里可能需要返回一个特定格式的错误，或者让异常传播
             # 为了简单，我们先让异常传播
             raise
        # --- 结束修改 ---


    def get_embedding_dimension(self) -> int:
        """返回嵌入模型的维度大小。"""
        if not hasattr(self, '_model_handler'):
            raise RuntimeError("GGUFEmbeddingResource is not initialized.")
        
        dimension = self._model_handler.get_embedding_dimension()
        if dimension is None:
            raise ValueError("Could not determine embedding dimension from the loaded GGUF model.")
        return dimension


# --- ChromaDBResource ---
class ChromaDBResourceConfig(dg.Config):
    collection_name: str = "rag_documents"
    persist_directory: str = "/home/zhz/zhz_agent/zhz_rag/stored_data/chromadb_index/"

class ChromaDBResource(dg.ConfigurableResource):
    collection_name: str
    persist_directory: str

    _client: chromadb.Client = PrivateAttr(default=None)
    _collection: chromadb.Collection = PrivateAttr(default=None)
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger = context.log
        self._logger.info(f"Initializing ChromaDB client and collection '{self.collection_name}'...")
        self._logger.info(f"ChromaDB data will be persisted to: {self.persist_directory}")
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self._logger.info(f"ChromaDB collection '{self.collection_name}' initialized/loaded. Count: {self._collection.count()}")
        except Exception as e:
            self._logger.error(f"Failed to initialize ChromaDB: {e}", exc_info=True)
            raise

    def add_embeddings(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]] = None):
        logger_instance = self._logger if self._logger else dg.get_dagster_logger()
        if self._collection is None:
            logger_instance.error("ChromaDB collection is not initialized. Cannot add embeddings.")
            raise RuntimeError("ChromaDB collection is not initialized.")
        
        if not (len(ids) == len(embeddings) and (metadatas is None or len(ids) == len(metadatas))):
            logger_instance.error("Length mismatch for ids, embeddings, or metadatas.")
            raise ValueError("Length of ids, embeddings, and metadatas (if provided) must be the same.")

        if not ids:
            logger_instance.info("No ids provided to add_embeddings, skipping.")
            return

        logger_instance.info(f"Adding/updating {len(ids)} embeddings to ChromaDB collection '{self.collection_name}'...")
        self._collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
        logger_instance.info(f"Embeddings added/updated. Collection count now: {self._collection.count()}")

    def query_embeddings(self, query_embeddings: List[List[float]], n_results: int = 5) -> chromadb.QueryResult:
        logger_instance = self._logger if self._logger else dg.get_dagster_logger()
        if self._collection is None:
            logger_instance.error("ChromaDB collection is not initialized. Cannot query embeddings.")
            raise RuntimeError("ChromaDB collection is not initialized.")
        logger_instance.debug(f"Querying ChromaDB collection '{self.collection_name}' with {len(query_embeddings)} vectors, n_results={n_results}.")
        return self._collection.query(query_embeddings=query_embeddings, n_results=n_results)

# --- LocalLLMAPIResource ---
class LocalLLMAPIResourceConfig(dg.Config):
    api_url: str = "http://127.0.0.1:8088/v1/chat/completions" # <--- 修改
    default_temperature: float = 0.1
    default_max_new_tokens: int = 2048

class LocalLLMAPIResource(dg.ConfigurableResource):
    # 我们将 SGLangAPIResource 重命名为 LocalLLMAPIResource
    api_url: str
    default_temperature: float
    default_max_new_tokens: int
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger = context.log
        self._logger.info(f"LocalLLMAPIResource configured with API URL: {self.api_url}") # <--- 修改日志信息

    # generate_structured_output 方法的逻辑需要调整以适配 OpenAI 兼容的 API
    async def generate_structured_output(
        self, prompt: str, json_schema: Dict[str, Any],
        temperature: Optional[float] = None, max_new_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        logger_instance = self._logger if self._logger else dg.get_dagster_logger()
        temp_to_use = temperature if temperature is not None else self.default_temperature
        tokens_to_use = max_new_tokens if max_new_tokens is not None else self.default_max_new_tokens

        messages = [{"role": "user", "content": prompt}] # 简化处理

        payload = {
            "model": "local_kg_extraction_model", # 模型名对于本地服务不重要，但需要有
            "messages": messages,
            "temperature": temp_to_use,
            "max_tokens": tokens_to_use,
            "response_format": { # 我们的 local_llm_service.py 支持这个
                "type": "json_object",
                "schema": json_schema
            }
        }
        logger_instance.debug(f"Sending request to Local LLM Service. Prompt (start): {prompt[:100]}...")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(self.api_url, json=payload) # 使用 self.api_url
                response.raise_for_status()
                response_json = response.json()
                
                # OpenAI 兼容 API 的响应格式是 {"choices": [{"message": {"content": "..."}}]}
                if response_json.get("choices") and response_json["choices"][0].get("message"):
                    generated_text = response_json["choices"][0]["message"].get("content", "")
                    logger_instance.debug(f"Local LLM raw response text: {generated_text}")
                    try:
                        parsed_output = json.loads(generated_text)
                        return parsed_output
                    except json.JSONDecodeError as e:
                        logger_instance.error(f"Failed to decode JSON from Local LLM output: {generated_text}. Error: {e}", exc_info=True)
                        raise ValueError(f"Local LLM output was not valid JSON: {generated_text}") from e
                else:
                    raise ValueError(f"Local LLM response format is incorrect: {response_json}")
        except httpx.HTTPStatusError as e:
            logger_instance.error(f"Local LLM API HTTP error: {e.response.status_code} - {e.response.text}", exc_info=True)
            raise
        except httpx.RequestError as e:
            logger_instance.error(f"Local LLM API request error: {e}", exc_info=True)
            raise
        except Exception as e:
            logger_instance.error(f"Unexpected error during Local LLM call: {e}", exc_info=True)
            raise

# --- GeminiAPIResource ---
class GeminiAPIResourceConfig(dg.Config):
    model_name: str = PydanticField(default="gemini/gemini-1.5-flash-latest", description="Name of the Gemini model.")
    proxy_url: Optional[str] = PydanticField(default_factory=lambda: os.getenv("LITELLM_PROXY_URL"), description="Optional proxy URL for LiteLLM.")
    default_temperature: float = 0.1
    default_max_tokens: int = 2048
    
class GeminiAPIResource(dg.ConfigurableResource):
    model_name: str
    proxy_url: Optional[str]
    default_temperature: float
    default_max_tokens: int
    _api_key: Optional[str] = PrivateAttr(default=None)
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger = context.log
        self._api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if self.model_name and not self.model_name.startswith("gemini/"):
            if "gemini" in self.model_name.lower():
                self._logger.info(f"Model name '{self.model_name}' auto-prefixed to 'gemini/'.")
                self.model_name = f"gemini/{self.model_name.split('/')[-1]}"
            else:
                self._logger.warning(f"Model name '{self.model_name}' does not start with 'gemini/'.")
        if not self._api_key:
            self._logger.warning("Gemini API key not found. API calls will likely fail.")
        else:
            self._logger.info(f"GeminiAPIResource initialized. Model: {self.model_name}, Proxy: {self.proxy_url or 'Not set'}")

    async def call_completion(
        self, messages: List[Dict[str, str]],
        temperature: Optional[float] = None, max_tokens: Optional[int] = None,
    ) -> Optional[str]:
        logger_instance = self._logger if self._logger else dg.get_dagster_logger()
        if not self._api_key:
            logger_instance.error("Gemini API key is not configured.")
            return None
        temp_to_use = temperature if temperature is not None else self.default_temperature
        tokens_to_use = max_tokens if max_tokens is not None else self.default_max_tokens
        litellm_params = {
            "model": self.model_name, "messages": messages, "api_key": self._api_key,
            "temperature": temp_to_use, "max_tokens": tokens_to_use,
        }
        if self.proxy_url:
            litellm_params["proxy"] = {"http": self.proxy_url, "https": self.proxy_url} # type: ignore
        logger_instance.debug(f"Calling LiteLLM (Gemini) with params (excluding messages): { {k:v for k,v in litellm_params.items() if k != 'messages'} }")
        raw_output_text: Optional[str] = None
        try:
            response = await litellm.acompletion(**litellm_params) # type: ignore
            if response and response.choices and response.choices[0].message and response.choices[0].message.content:
                raw_output_text = response.choices[0].message.content
                logger_instance.debug(f"LiteLLM (Gemini) raw response (first 300 chars): {raw_output_text[:300]}...")
            else:
                logger_instance.warning(f"LiteLLM (Gemini) returned empty/malformed response: {response}")
        except Exception as e_generic:
            logger_instance.error(f"Error calling Gemini via LiteLLM: {e_generic}", exc_info=True)
        return raw_output_text


class KuzuDBReadWriteResource(dg.ConfigurableResource):
    db_path_str: str = PydanticField(
        default=os.path.join("zhz_rag", "stored_data", "kuzu_default_db"), # 这是期望相对于 ZHZ_AGENT_PROJECT_ROOT 的路径
        description="Path to the KuzuDB database directory."
    )

    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)
    _resolved_db_path: str = PrivateAttr()
    _db: Optional[kuzu.Database] = PrivateAttr(default=None)

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger = context.log

        project_root_from_env = os.getenv("ZHZ_AGENT_PROJECT_ROOT")
        python_path_env = os.getenv("PYTHONPATH")
        
        determined_project_root = None

        if project_root_from_env and os.path.isdir(os.path.join(project_root_from_env, "zhz_rag")):
            determined_project_root = project_root_from_env
            self._logger.info(f"KuzuDBReadWriteResource: Using ZHZ_AGENT_PROJECT_ROOT: {determined_project_root}")
        elif python_path_env:
            # PYTHONPATH 可能包含多个路径，用 os.pathsep 分隔 (Linux是':', Windows是';')
            first_python_path = python_path_env.split(os.pathsep)[0]
            if os.path.isdir(os.path.join(first_python_path, "zhz_rag")):
                determined_project_root = first_python_path
                self._logger.info(f"KuzuDBReadWriteResource: Using first path from PYTHONPATH: {determined_project_root}")
            else: # Fallback if PYTHONPATH doesn't seem right
                determined_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
                self._logger.warning(f"KuzuDBReadWriteResource: PYTHONPATH ('{python_path_env}') does not seem to point to project root. Falling back to relative path guess: {determined_project_root}")
        else: # Fallback if no relevant env var
            determined_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            self._logger.warning(f"KuzuDBReadWriteResource: Neither ZHZ_AGENT_PROJECT_ROOT nor PYTHONPATH found for project root. Falling back to relative path guess: {determined_project_root}")

        # 确保我们猜的根目录至少包含 zhz_rag 子目录，这是一个基本检查
        if not os.path.isdir(os.path.join(determined_project_root, "zhz_rag")):
            self._logger.error(f"KuzuDBReadWriteResource: CRITICAL - Could not reliably determine project root. Path '{determined_project_root}' does not contain 'zhz_rag'. Please set ZHZ_AGENT_PROJECT_ROOT environment variable to '/home/zhz/zhz_agent'.")
            # 在这种情况下，后续操作很可能会失败，但我们还是构造一个路径以暴露问题
        
        project_root_for_dagster = determined_project_root
        # --- 修改结束 ---
        
        self._logger.info(f"KuzuDBReadWriteResource: Determined project_root_for_dagster: {project_root_for_dagster}")

        if os.path.isabs(self.db_path_str):
            self._resolved_db_path = self.db_path_str
        else:
            self._resolved_db_path = os.path.join(project_root_for_dagster, self.db_path_str)
        
        self._logger.info(
            f"KuzuDBReadWriteResource: FINAL Resolved DB path is '{os.path.abspath(self._resolved_db_path)}'."
        )

        # --- 确保每次都清理并重建数据库目录 ---
        if os.path.exists(self._resolved_db_path):
            shutil.rmtree(self._resolved_db_path)
            self._logger.info(f"Cleaned up existing database directory: {self._resolved_db_path}")
        os.makedirs(self._resolved_db_path, exist_ok=True)
        self._logger.info(f"KuzuDBReadWriteResource: Recreated database directory: {self._resolved_db_path}")
        # --- 清理逻辑结束 ---

        # --- 添加开始：准备本地扩展 ---
        manual_extension_source_path = os.getenv("KUZU_VECTOR_EXTENSION_PATH", "/home/zhz/.kuzu/extension/0.10.0/linux_amd64/vector/libvector.kuzu_extension") # 从环境变量读取或使用默认
        
        extensions_dir_in_db = os.path.join(self._resolved_db_path, "extensions")
        os.makedirs(extensions_dir_in_db, exist_ok=True)
        self._logger.info(f"KuzuDBReadWriteResource: Ensured extensions directory exists: {extensions_dir_in_db}")

        if os.path.exists(manual_extension_source_path):
            target_path_vector = os.path.join(extensions_dir_in_db, "vector.kuzu_extension")
            try:
                shutil.copy2(manual_extension_source_path, target_path_vector)
                self._logger.info(f"KuzuDBReadWriteResource: Copied vector extension to '{target_path_vector}'")
            except Exception as e_copy:
                self._logger.error(f"KuzuDBReadWriteResource: ERROR - Could not copy vector extension: {e_copy}")
        else:
            self._logger.error(f"KuzuDBReadWriteResource: ERROR - Manual vector extension file not found at '{manual_extension_source_path}'. LOAD VECTOR will likely fail if Kuzu doesn't find it elsewhere.")
        # --- 添加结束 ---

        try:
            self._db = kuzu.Database(self._resolved_db_path)
            self._logger.info(f"Shared kuzu.Database object CREATED successfully for the entire run.")
        except Exception as e:
            self._logger.error(f"Failed to initialize shared kuzu.Database object: {e}", exc_info=True)
            raise

    def teardown_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger.info(f"Tearing down KuzuDBReadWriteResource for run...")
        if self._db is not None:
            self._db = None # KuzuDB 对象依赖析构函数来关闭数据库和释放锁
            self._logger.info("Shared kuzu.Database object dereferenced, relying on destructor to close and release lock.")
        else:
            self._logger.info("Shared kuzu.Database object was already None during teardown.")
        self._logger.info("KuzuDBReadWriteResource teardown complete.")

    @contextmanager
    def get_connection(self) -> Iterator[kuzu.Connection]:
        if self._db is None:
            raise ConnectionError("Shared KuzuDB Database object is not initialized. This should not happen if setup_for_execution was successful.")
        
        conn = None
        try:
            conn = kuzu.Connection(self._db)
            self._logger.debug("New kuzu.Connection obtained from shared Database object.")
            yield conn
        finally:
            if conn is not None:
                self._logger.debug("kuzu.Connection from shared DB object is going out of scope.")
                
class KuzuDBReadOnlyResource(dg.ConfigurableResource):
    db_path_str: str = PydanticField(
        default=os.path.join("zhz_rag", "stored_data", "kuzu_default_db"),
        description=(
            "Path to the KuzuDB database directory for read-only access. "
            "Can be relative to the project root (if not starting with '/') or absolute."
        )
    )
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)
    _resolved_db_path: str = PrivateAttr()

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger = context.log
        if os.path.isabs(self.db_path_str):
            self._resolved_db_path = self.db_path_str
        else:
            self._resolved_db_path = os.path.abspath(self.db_path_str)

        self._logger.info(f"KuzuDBReadOnlyResource setup: resolved_path='{self._resolved_db_path}'")
        if not os.path.exists(self._resolved_db_path):
            self._logger.error(f"KuzuDB path {self._resolved_db_path} does not exist for ReadOnly access. Operations will likely fail.")

    def teardown_for_execution(self, context: dg.InitResourceContext) -> None:
        logger_instance = self._logger if self._logger else dg.get_dagster_logger()
        logger_instance.info("KuzuDBReadOnlyResource teardown complete.")

    @contextmanager
    def get_readonly_connection(self) -> Iterator[kuzu.Connection]:
        logger_instance = self._logger if self._logger else dg.get_dagster_logger()
        db_instance: Optional[kuzu.Database] = None
        logger_instance.info(f"Attempting to open KuzuDB(RO) at {self._resolved_db_path} for readonly session.")
        
        if not os.path.exists(self._resolved_db_path):
            logger_instance.error(f"KuzuDB directory {self._resolved_db_path} not found for read-only access.")
            raise FileNotFoundError(f"KuzuDB directory {self._resolved_db_path} not found for read-only access.")

        try:
            db_instance = kuzu.Database(self._resolved_db_path, read_only=True)
            logger_instance.info(f"KuzuDB(RO) session opened at {self._resolved_db_path}")
            conn = kuzu.Connection(db_instance)
            yield conn
        except Exception as e:
            logger_instance.error(f"Error during KuzuDB(RO) session: {e}", exc_info=True)
            raise
        finally:
            if db_instance:
                del db_instance
                logger_instance.info(f"KuzuDB(RO) Database object for session at {self._resolved_db_path} dereferenced (closed).")