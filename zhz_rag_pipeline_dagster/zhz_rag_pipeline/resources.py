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
import shutil
from contextlib import contextmanager
from pydantic import Field as PydanticField
from pydantic import PrivateAttr
import asyncio
import time	
import gc
import duckdb

# --- START: 添加 HardwareManager 导入 ---
try:
    # 假设 hardware_manager.py 位于项目的 utils 目录下，
    # 并且 zhz_agent 是项目的根目录且在 PYTHONPATH 中
    from utils.hardware_manager import HardwareManager, HardwareInfo
except ImportError as e_hal_import:
    # 如果 utils 目录不在 zhz_agent 下，或者 PYTHONPATH 问题，可能需要调整
    # 例如，如果 hardware_manager.py 与 resources.py 在同一目录的父目录的utils下：
    # from ..utils.hardware_manager import HardwareManager, HardwareInfo
    print(f"ERROR: Failed to import HardwareManager/HardwareInfo from utils.hardware_manager: {e_hal_import}. "
          "Ensure it's in the correct path relative to this file or PYTHONPATH is set. "
          "Proceeding without HAL capabilities.")
    HardwareManager = None
    HardwareInfo = None # type: ignore
# --- END: 添加 HardwareManager 导入 ---

try:
    from zhz_rag.llm.local_model_handler import LocalModelHandler
except ImportError as e:
    print(f"FATAL: Could not import LocalModelHandler from zhz_rag.llm.local_model_handler. Error: {e}")
    raise

class GGUFEmbeddingResourceConfig(dg.Config):
    # --- 采纳外部AI建议的修改 ---
    embedding_model_path: str  # 必需字段，直接类型注解
    n_ctx: int = 2048          # 可选字段，直接赋予Python默认值
    n_gpu_layers: int = 0      # 可选字段，直接赋予Python默认值
    # --- 修改结束 ---

class GGUFEmbeddingResource(dg.ConfigurableResource):
    embedding_model_path: str
    n_ctx: int
    n_gpu_layers: int

    _model_handler: Optional[LocalModelHandler] = PrivateAttr(default=None)
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)
    _dimension: Optional[int] = PrivateAttr(default=None)
    # _dimension_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock) # 在纯同步场景下暂时不需要

    def _run_async_in_new_loop(self, coro):
        # 这个辅助函数在同步方法中运行异步协程是正确的
        # 对于 Dagster 这种多进程/多线程环境，确保每次都用新循环可能更安全
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(coro)
        finally:
            loop.close()
        return result

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger = context.log
        self._logger.info(f"Initializing GGUFEmbeddingResource with model: {self.embedding_model_path}")
        
        if not os.path.exists(self.embedding_model_path):
            error_msg = f"GGUF embedding model file not found at: {self.embedding_model_path}"
            self._logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            self._model_handler = LocalModelHandler(
                embedding_model_path=self.embedding_model_path,
                n_ctx_embed=self.n_ctx,
                n_gpu_layers_embed=self.n_gpu_layers,
                # embedding_pool_size 可以考虑从Config传入或使用默认
            )
            self._logger.info(f"GGUFEmbeddingResource initialized. LocalModelHandler created. Dimension will be fetched on first use.")
        except Exception as e:
            self._logger.error(f"Failed to initialize GGUFEmbeddingResource (LocalModelHandler creation failed): {e}", exc_info=True)
            raise

    def _ensure_dimension_is_known(self) -> int:
        if self._dimension is None:
            if self._model_handler is None: # 确保 _model_handler 已初始化
                self._logger.error("GGUFEmbeddingResource: LocalModelHandler not initialized in _ensure_dimension_is_known.")
                raise RuntimeError("LocalModelHandler not initialized, cannot get dimension.")
            
            self._logger.info("GGUFEmbeddingResource: Dimension not yet known, attempting to fetch from LocalModelHandler...")
            try:
                # _get_embedding_dimension_from_worker_once() 是 LocalModelHandler 的异步方法
                dim = self._run_async_in_new_loop(self._model_handler._get_embedding_dimension_from_worker_once())
                if dim is None or dim <= 0:
                    raise ValueError(f"LocalModelHandler returned invalid dimension: {dim}")
                self._dimension = dim
                self._logger.info(f"GGUFEmbeddingResource: Dimension fetched and cached: {self._dimension}")
            except Exception as e:
                self._logger.error(f"GGUFEmbeddingResource: Failed to fetch embedding dimension: {e}", exc_info=True)
                raise RuntimeError(f"Could not determine embedding dimension: {e}") from e
        return self._dimension

    def encode(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        if self._model_handler is None:
            self._logger.error("GGUFEmbeddingResource: LocalModelHandler not initialized in encode.")
            raise RuntimeError("GGUFEmbeddingResource is not initialized (model_handler is None).")
        
        self._ensure_dimension_is_known() 

        logger_instance = self._logger if self._logger else dg.get_dagster_logger()
        logger_instance.debug(f"GGUFEmbeddingResource: Encoding {len(texts)} texts using LocalModelHandler (async call).")
        
        try:
            embeddings = self._run_async_in_new_loop(
                self._model_handler.embed_documents(texts)
            )
            return embeddings
        except Exception as e:
             logger_instance.error(f"GGUFEmbeddingResource: Error during encode: {e}", exc_info=True)
             raise


    def get_embedding_dimension(self) -> int:
        """返回嵌入模型的维度大小。如果尚未获取，则会尝试获取。"""
        if not hasattr(self, '_model_handler') or self._model_handler is None: # 增加对 _model_handler 是否为 None 的检查
            self._logger.error("GGUFEmbeddingResource: LocalModelHandler not initialized in get_embedding_dimension.")
            raise RuntimeError("GGUFEmbeddingResource is not initialized (model_handler is None).")
        
        # 直接调用内部方法来确保维度是已知的
        # _ensure_dimension_is_known 会处理维度为 None 的情况，并主动去获取
        return self._ensure_dimension_is_known()


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

    def add_embeddings(
        self, 
        ids: List[str], 
        embeddings: List[List[float]], 
        documents: Optional[List[str]] = None, # <--- 添加 documents 参数，并设为可选
        metadatas: Optional[List[Dict[str, Any]]] = None # <--- 将 metadatas 也设为可选，与ChromaDB客户端一致
    ):
        logger_instance = self._logger if self._logger else dg.get_dagster_logger()
        if self._collection is None:
            logger_instance.error("ChromaDB collection is not initialized. Cannot add embeddings.")
            raise RuntimeError("ChromaDB collection is not initialized.")
        
        # 参数长度校验
        num_ids = len(ids)
        if not (num_ids == len(embeddings) and \
                (documents is None or num_ids == len(documents)) and \
                (metadatas is None or num_ids == len(metadatas))):
            logger_instance.error(
                f"Length mismatch: ids({num_ids}), embeddings({len(embeddings)}), "
                f"documents({len(documents) if documents else 'None'}), metadatas({len(metadatas) if metadatas else 'None'})."
            )
            raise ValueError("Length of ids, embeddings, and documents/metadatas (if provided) must be the same.")

        if not ids:
            logger_instance.info("No ids provided to add_embeddings, skipping.")
            return

        logger_instance.info(f"Adding/updating {len(ids)} items to ChromaDB collection '{self.collection_name}'...")
        try:
            self._collection.add(
                ids=ids, 
                embeddings=embeddings, 
                documents=documents, # <--- 将 documents 参数传递给 collection.add
                metadatas=metadatas
            )
            logger_instance.info(f"Items added/updated. Collection count now: {self._collection.count()}")
        except Exception as e_add:
            logger_instance.error(f"Error during self._collection.add: {e_add}", exc_info=True)
            raise
        
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
            timeout_config = httpx.Timeout(
                connect=30.0,    # 连接超时30秒
                read=300.0,      # 读取超时300秒 (5分钟)
                write=300.0,     # 写入超时300秒 (5分钟)
                pool=30.0        # 从连接池获取连接的超时30秒
            )
            async with httpx.AsyncClient(timeout=timeout_config) as client:
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

class DuckDBResource(dg.ConfigurableResource):
    db_file_path: str = PydanticField(
        default=os.path.join(os.getenv("ZHZ_AGENT_PROJECT_ROOT", "/home/zhz/zhz_agent"), "zhz_rag", "stored_data", "duckdb_knowledge_graph.db"),
        description="Path to the DuckDB database file for the knowledge graph."
    )
    # 可以在这里添加其他配置，例如 read_only, vss_persistence 等

    _conn: Optional[duckdb.DuckDBPyConnection] = PrivateAttr(default=None)
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)

    # --- 【用这个版本覆盖原来的 setup_for_execution 方法】 ---
    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger = context.log
        db_path_to_use = self.db_file_path
        
        db_dir = os.path.dirname(db_path_to_use)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            self._logger.info(f"Created directory for DuckDB database: {db_dir}")

        self._logger.info(f"Attempting to connect to DuckDB at: {db_path_to_use}")
        
        try:
            # 连接数据库
            self._conn = duckdb.connect(database=db_path_to_use, read_only=False)
            self._logger.info(f"Successfully connected to DuckDB: {db_path_to_use}")
            
            # 强制加载VSS扩展
            # 即使之前已加载，重复LOAD通常是安全的，或者会快速返回
            # 关键在于确保当前连接的上下文中VSS是可用的
            try:
                self._logger.info("DuckDBResource: Forcefully attempting to INSTALL and LOAD vss extension for current connection.")
                self._conn.execute("INSTALL vss;")      # 尝试安装（如果未安装）
                self._conn.execute("LOAD vss;")         # 尝试加载
                self._logger.info("DuckDBResource: VSS extension INSTALL and LOAD sequence completed for current connection.")
                
                # 启用HNSW持久化（如果需要）
                try:
                    self._conn.execute("SET hnsw_enable_experimental_persistence=true;")
                    self._logger.info("DuckDBResource: HNSW experimental persistence enabled for current connection.")
                except Exception as e_persistence:
                    self._logger.warning(f"DuckDBResource: Failed to enable HNSW experimental persistence: {e_persistence}. This might affect index persistence if new HNSW indexes are created by this resource/asset.")
            
            except duckdb.CatalogException as e_vss_cat:
                if "already loaded" in str(e_vss_cat).lower() or "already installed" in str(e_vss_cat).lower():
                    self._logger.info(f"DuckDBResource: VSS extension reported as already installed/loaded: {e_vss_cat}")
                    # 即使已加载，也尝试再次LOAD以确保当前会话可用
                    try:
                        self._conn.execute("LOAD vss;")
                        self._logger.info("DuckDBResource: VSS extension re-LOADED successfully.")
                    except Exception as e_reload:
                        self._logger.error(f"DuckDBResource: Failed to re-LOAD VSS extension even if reported as installed/loaded: {e_reload}", exc_info=True)
                        raise RuntimeError(f"Critical: Failed to ensure VSS is loaded for DuckDB connection: {e_reload}") from e_reload
                else:
                    self._logger.error(f"DuckDBResource: CatalogException during VSS setup (not 'already loaded/installed'): {e_vss_cat}", exc_info=True)
                    raise RuntimeError(f"Failed to setup VSS extension for DuckDB: {e_vss_cat}") from e_vss_cat
            except Exception as e_vss_other:
                self._logger.error(f"DuckDBResource: Other exception during VSS setup: {e_vss_other}", exc_info=True)
                raise RuntimeError(f"Failed to setup VSS extension for DuckDB: {e_vss_other}") from e_vss_other

        except duckdb.duckdb.Error as e_connect_wal: # 捕获包括WAL重放错误在内的DuckDB连接错误
            self._logger.error(f"Failed to connect to DuckDB at {db_path_to_use} or error during WAL replay: {e_connect_wal}", exc_info=True)
            # 检查错误信息是否与VSS有关
            if "unknown index type 'HNSW'" in str(e_connect_wal):
                self._logger.error("DuckDBResource: WAL replay failed due to HNSW index. This strongly indicates VSS was not loaded prior to the operation that created/modified the index or during this connection attempt before WAL replay.")
            raise # 将原始的DuckDB连接错误重新抛出
        except Exception as e_connect_generic: # 捕获其他可能的连接错误
            self._logger.error(f"Generic error connecting to DuckDB at {db_path_to_use}: {e_connect_generic}", exc_info=True)
            raise

    def teardown_for_execution(self, context: dg.InitResourceContext) -> None:
        if self._conn:
            self._logger.info(f"Closing DuckDB connection to: {self.db_file_path}")
            self._conn.close()
            self._conn = None
        self._logger.info("DuckDBResource teardown complete.")

    @contextmanager
    def get_connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        if not self._conn:
            # 这种情况下，我们应该在 setup_for_execution 中确保连接已建立
            # 或者在这里尝试重新连接，但这会使逻辑复杂化。
            # Dagster 资源通常期望 setup_for_execution 已经准备好了资源。
            self._logger.error("DuckDB connection not established during setup. Cannot yield connection.")
            raise ConnectionError("DuckDB connection was not established by setup_for_execution.")
        
        # 简单的版本：直接 yield 已经建立的连接
        # 注意：这种共享连接的方式对于并发的 Dagster ops (如果使用非 in_process_executor) 需要小心
        # 但对于 in_process_executor 和我们的流水线是安全的。
        yield self._conn


# --- START: 定义 SystemResource ---
class SystemResource(dg.ConfigurableResource):
    """
    Dagster resource to provide system hardware information and recommendations
    based on Hardware Abstraction Layer (HAL).
    """
    _hw_manager: Optional[Any] = PrivateAttr(default=None)
    _hw_info: Optional[Any] = PrivateAttr(default=None)
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)
    
    # 可以添加一些配置项来微调HAL的行为，如果需要的话
    # 例如：safety_vram_buffer_gb_override: Optional[float] = None

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger = context.log
        self._logger.info("Initializing SystemResource...")
        if HardwareManager:
            try:
                self._hw_manager = HardwareManager()
                self._hw_info = self._hw_manager.get_hardware_info()
                if self._hw_info:
                    self._logger.info(f"SystemResource: Hardware detection successful: {self._hw_info}")
                else:
                    self._logger.warning("SystemResource: HardwareManager did not return hardware info.")
            except Exception as e:
                self._logger.error(f"SystemResource: Error initializing HardwareManager: {e}", exc_info=True)
        else:
            self._logger.warning("SystemResource: HardwareManager class not available. HAL features will be disabled.")
        self._logger.info("SystemResource initialized.")

    def get_hardware_info(self) -> Optional[Any]:
        """Returns the detected HardwareInfo object, or None if detection failed."""
        if not self._hw_info:
            self._logger.warning("Accessing hardware info, but it was not successfully detected during initialization.")
        return self._hw_info

    def get_recommended_llm_gpu_layers(
        self, 
        model_total_layers: int, 
        model_size_on_disk_gb: float, 
        context_length_tokens: int,
        # 可以暴露HAL方法中的其他参数作为此方法的参数，或使用默认/固定值
        kv_cache_gb_per_1k_ctx: float = 0.25, 
        safety_buffer_vram_gb: float = 1.5
    ) -> int:
        """
        Delegates to HardwareManager to recommend n_gpu_layers.
        Returns 0 if HAL is not available or GPU is not suitable.
        """
        if self._hw_manager:
            try:
                return self._hw_manager.recommend_llm_gpu_layers(
                    model_total_layers=model_total_layers,
                    model_size_on_disk_gb=model_size_on_disk_gb,
                    kv_cache_gb_per_1k_ctx=kv_cache_gb_per_1k_ctx,
                    context_length_tokens=context_length_tokens,
                    safety_buffer_vram_gb=safety_buffer_vram_gb
                )
            except Exception as e:
                self._logger.error(f"Error getting GPU layer recommendation from HAL: {e}", exc_info=True)
                return 0 # Fallback to CPU on error
        self._logger.warning("HardwareManager not available, defaulting to 0 GPU layers.")
        return 0 # Fallback to CPU

    def get_recommended_concurrent_tasks(self, task_type: str = "cpu_bound_llm") -> int:
        """
        Delegates to HardwareManager to recommend concurrent task number.
        Returns 1 if HAL is not available.
        """
        if self._hw_manager:
            try:
                return self._hw_manager.recommend_concurrent_tasks(task_type=task_type)
            except Exception as e:
                self._logger.error(f"Error getting concurrent task recommendation from HAL: {e}", exc_info=True)
                return 1 # Fallback to 1 on error
        self._logger.warning("HardwareManager not available, defaulting to 1 concurrent task.")
        return 1 # Fallback
# --- END: 定义 SystemResource ---