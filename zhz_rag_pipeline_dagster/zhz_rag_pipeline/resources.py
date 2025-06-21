# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/zhz_rag_pipeline/resources.py

import logging
import dagster as dg
import chromadb
from typing import List, Dict, Any, Optional, Iterator
import httpx
import json
import os
from contextlib import asynccontextmanager, contextmanager # <--- 修正: 导入 contextmanager
from pydantic import Field as PydanticField, PrivateAttr
import asyncio
import time 
import duckdb
import sys
from queue import Empty
from pathlib import Path


# --- 日志和硬件管理器导入 ---
try:
    from zhz_rag.utils.interaction_logger import get_logger
except ImportError:
    import logging
    def get_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

try:
    from zhz_rag.utils.hardware_manager import HardwareManager, HardwareInfo
except ImportError as e_hal_import:
    print(f"ERROR: Failed to import HardwareManager/HardwareInfo: {e_hal_import}. HAL features will be disabled.")
    HardwareManager = None
    HardwareInfo = None # type: ignore


# --- GGUFEmbeddingResource: API客户端版本 ---

class GGUFEmbeddingResourceConfig(dg.Config):
    api_url: str = PydanticField( # <--- 修正: 使用 PydanticField 别名
        default="http://127.0.0.1:8089",
        description="URL of the standalone embedding API service."
    )

class GGUFEmbeddingResource(dg.ConfigurableResource):
    api_url: str

    _client: httpx.AsyncClient = PrivateAttr()
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)
    _dimension: Optional[int] = PrivateAttr(default=None)
    _batch_size: int = PrivateAttr(default=128) # 提供一个保守的默认值

    def setup_for_execution(self, context: Optional[dg.InitResourceContext] = None) -> None:
        """
        初始化资源。现在可以接受一个可选的Dagster上下文。
        如果context为None（在FastAPI等非Dagster环境中使用），则使用默认配置。
        """
        self._logger = context.log if context else logging.getLogger("GGUFEmbeddingResource")
        self._client = httpx.AsyncClient(base_url=self.api_url, timeout=600.0)
        self._logger.info(f"GGUFEmbeddingResource configured to use API at: {self.api_url}")

        # 动态计算批处理大小
        try:
            # 检查是否在Dagster上下文中，并尝试获取system_resource
            if context and hasattr(context, 'resources_by_key') and "system_resource" in context.resources_by_key:
                system_resource = context.resources_by_key["system_resource"]
                physical_cores = system_resource._hw_info.cpu_physical_cores if system_resource._hw_info else 4
                self._batch_size = max(128, physical_cores * 64)
                self._logger.info(f"Dynamically set embedding batch size to {self._batch_size} based on {physical_cores} physical cores (from Dagster context).")
            else:
                # 在非Dagster环境中，使用默认值
                self._logger.warning("Running outside of full Dagster context or SystemResource not available. Using default batch size of 128.")
                self._batch_size = 128
        except Exception as e:
            self._logger.error(f"Failed to dynamically set batch size: {e}. Using default 128.", exc_info=True)
            self._batch_size = 128
        # 健康检查
        try:
            response = httpx.get(f"{self.api_url}/health")
            response.raise_for_status()
            health_data = response.json()
            if health_data.get("model_loaded"):
                self._dimension = health_data.get("dimension")
                self._logger.info(f"Embedding service is healthy. Dimension confirmed: {self._dimension}")
            else:
                raise RuntimeError(f"Embedding service at {self.api_url} is not healthy.")
        except Exception as e:
            self._logger.error(f"Failed to connect to embedding service: {e}")
            raise RuntimeError("Could not initialize GGUFEmbeddingResource.") from e

    def teardown_for_execution(self, context: dg.InitResourceContext) -> None:
        if hasattr(self, '_client') and not self._client.is_closed:
            async def _close():
                await self._client.aclose()
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_close())
            except RuntimeError:
                asyncio.run(_close())
            self._logger.info("GGUFEmbeddingResource client closed.")

    def get_embedding_dimension(self) -> int:
        if self._dimension is None:
            raise ValueError("Embedding dimension not available.")
        return self._dimension

    def encode(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        
        # --- 核心：批处理循环 ---
        for i in range(0, len(texts), self._batch_size):
            batch_texts = texts[i:i + self._batch_size]
            self._logger.info(f"Processing batch {i // self._batch_size + 1}/{(len(texts) + self._batch_size - 1) // self._batch_size} with {len(batch_texts)} texts.")
            
            async def _async_encode_batch():
                try:
                    response = await self._client.post("/embed", json={"texts": batch_texts})
                    response.raise_for_status()
                    data = response.json()
                    return data.get("embeddings", [])
                except httpx.RequestError as e:
                    self._logger.error(f"Request to embedding service failed for a batch: {e}")
                    return [[] for _ in batch_texts] # 返回空列表表示此批次失败
                except Exception as e:
                    self._logger.error(f"An unexpected error occurred during embedding a batch: {e}")
                    return [[] for _ in batch_texts]

            # 在循环内部执行异步调用
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(_async_encode_batch(), loop)
                    batch_embeddings = future.result(timeout=600)
                else:
                    batch_embeddings = asyncio.run(_async_encode_batch())
            except RuntimeError:
                batch_embeddings = asyncio.run(_async_encode_batch())

            all_embeddings.extend(batch_embeddings)
            time.sleep(0.1) # 在批次之间加入一个微小的延迟，避免瞬间打爆API

        return all_embeddings

class ChromaDBResourceConfig(dg.Config):
    collection_name: str = PydanticField(
        default="zhz_rag_collection",
        description="Name of the ChromaDB collection."
    )
    persist_directory: str = PydanticField(
        # 确保路径与你的项目结构和期望的存储位置一致
        default=os.path.join(os.getenv("ZHZ_AGENT_PROJECT_ROOT", "/home/zhz/zhz_agent"), "zhz_rag", "stored_data", "chromadb_index"),
        description="Directory to persist ChromaDB data."
    )
    # 可以添加更多ChromaDB客户端的配置，例如auth, headers等
    # client_settings: Optional[Dict[str, Any]] = None # 例如 chromadb.Settings

class ChromaDBResource(dg.ConfigurableResource):
    collection_name: str
    persist_directory: str

    _client: Optional[chromadb.PersistentClient] = PrivateAttr(default=None)
    _collection: Optional[Any] = PrivateAttr(default=None)
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)
    _batch_size: int = PrivateAttr(default=4096) # ChromaDB的推荐批处理大小通常在4k-5k之间

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger = context.log
        os.makedirs(self.persist_directory, exist_ok=True)
        self._logger.info(f"ChromaDB persist directory: {self.persist_directory}")
        
        try:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._logger.info(f"ChromaDB client initialized. Attempting to get or create collection: '{self.collection_name}'")
            self._collection = self._client.get_or_create_collection(name=self.collection_name)
            self._logger.info(f"Successfully got or created ChromaDB collection: '{self.collection_name}'. Collection ID: {self._collection.id}")
            self._logger.info(f"Current item count in collection '{self.collection_name}': {self._collection.count()}")
            # 我们可以根据ChromaDB的实际限制动态设置批处理大小，但为简单起见先用一个安全值
            # self._batch_size = chromadb.get_max_batch_size()
            self._logger.info(f"ChromaDB batch size set to: {self._batch_size}")
        except Exception as e:
            self._logger.error(f"Failed to initialize ChromaDB client or collection: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize ChromaDBResource due to: {e}") from e

    def teardown_for_execution(self, context: dg.InitResourceContext) -> None:
        if self._client:
            self._logger.info("ChromaDBResource teardown: Client was persistent, no explicit close needed.")
        self._client = None
        self._collection = None

    def add_embeddings(
        self, 
        ids: List[str], 
        embeddings: List[List[float]], 
        documents: Optional[List[str]] = None, 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        if self._collection is None:
            msg = "ChromaDB collection is not initialized. Cannot add embeddings."
            self._logger.error(msg)
            raise RuntimeError(msg)
        
        if not ids:
            self._logger.warning("add_embeddings called with empty IDs list. Nothing to add.")
            return

        total_items = len(ids)
        for i in range(0, total_items, self._batch_size):
            batch_end = min(i + self._batch_size, total_items)
            batch_ids = ids[i:batch_end]
            batch_embeddings = embeddings[i:batch_end]
            batch_documents = documents[i:batch_end] if documents else None
            batch_metadatas = metadatas[i:batch_end] if metadatas else None

            self._logger.info(
                f"Adding batch {i // self._batch_size + 1}/{(total_items + self._batch_size - 1) // self._batch_size} "
                f"with {len(batch_ids)} items to ChromaDB."
            )

            try:
                self._collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
            except Exception as e:
                self._logger.error(
                    f"Failed to add batch starting at index {i} to ChromaDB: {e}",
                    exc_info=True
                )
                # 根据策略，您可以选择在此处继续、重试或直接抛出异常中断整个过程
                # 为确保数据完整性，我们选择抛出异常
                raise
        
        self._logger.info(f"Successfully added/updated all {total_items} items. Collection count now: {self._collection.count()}")

    def query_embeddings(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        if self._collection is None:
            msg = "ChromaDB collection is not initialized. Cannot query embeddings."
            self._logger.error(msg)
            raise RuntimeError(msg)

        if include is None:
            include = ["metadatas", "documents", "distances"]

        try:
            self._logger.info(f"Querying collection '{self.collection_name}' with {len(query_embeddings)} vector(s), n_results={n_results}, filter={where_filter is not None}.")
            results = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where_filter,
                include=include
            )
            return results
        except Exception as e:
            self._logger.error(f"Failed to query embeddings from ChromaDB: {e}", exc_info=True)
            raise
        

class LocalLLMAPIResourceConfig(dg.Config):
    api_url: str = "http://127.0.0.1:8088/v1/chat/completions"
    default_temperature: float = 0.1
    default_max_new_tokens: int = 2048

class LocalLLMAPIResource(dg.ConfigurableResource):
    api_url: str
    default_temperature: float
    default_max_new_tokens: int
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)
    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger = context.log
        self._logger.info(f"LocalLLMAPIResource configured with API URL: {self.api_url}")
    async def generate_structured_output(self, prompt: str, json_schema: Dict[str, Any], temperature: Optional[float] = None, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        logger_instance = self._logger if self._logger else dg.get_dagster_logger()
        temp_to_use = temperature if temperature is not None else self.default_temperature
        tokens_to_use = max_new_tokens if max_new_tokens is not None else self.default_max_new_tokens
        messages = [{"role": "user", "content": prompt}]
        payload = {"model": "local_kg_extraction_model", "messages": messages, "temperature": temp_to_use, "max_tokens": tokens_to_use, "response_format": {"type": "json_object", "schema": json_schema}}
        logger_instance.debug(f"Sending request to Local LLM Service. Prompt (start): {prompt[:100]}...")
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                response = await client.post(self.api_url, json=payload)
                response.raise_for_status()
                response_json = response.json()
                if response_json.get("choices") and response_json["choices"][0].get("message"):
                    generated_text = response_json["choices"][0]["message"].get("content", "")
                    return json.loads(generated_text)
                raise ValueError(f"Local LLM response format is incorrect: {response_json}")
        except Exception as e:
            logger_instance.error(f"Error during Local LLM call: {e}", exc_info=True)
            raise

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
        if not self._api_key: self._logger.warning("Gemini API key not found.")
        else: self._logger.info(f"GeminiAPIResource initialized. Model: {self.model_name}, Proxy: {self.proxy_url or 'Not set'}")
    async def call_completion(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Optional[str]:
        import litellm
        logger_instance = self._logger if self._logger else dg.get_dagster_logger()
        if not self._api_key: return None
        litellm_params = {"model": self.model_name, "messages": messages, "api_key": self._api_key, "temperature": temperature or self.default_temperature, "max_tokens": max_tokens or self.default_max_tokens}
        if self.proxy_url: litellm_params["proxy"] = {"http": self.proxy_url, "https": self.proxy_url}
        try:
            response = await litellm.acompletion(**litellm_params)
            return response.choices[0].message.content if response and response.choices else None
        except Exception as e:
            logger_instance.error(f"Error calling Gemini via LiteLLM: {e}", exc_info=True)
            return None
        
class DuckDBResource(dg.ConfigurableResource):
    db_file_path: str = PydanticField(
        default=os.path.join(os.getenv("ZHZ_AGENT_PROJECT_ROOT", "/home/zhz/zhz_agent"), "zhz_rag", "stored_data", "duckdb_knowledge_graph.db"),
        description="Path to the DuckDB database file."
    )
    _conn: Optional[duckdb.DuckDBPyConnection] = PrivateAttr(default=None)
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger = context.log
        self._logger.info("<<<<< DuckDBResource SETUP_FOR_EXECUTION - START >>>>>")

        os.makedirs(os.path.dirname(self.db_file_path), exist_ok=True)
        
        try:
            self._logger.info(f"Connecting to DuckDB at: {self.db_file_path}")
            self._conn = duckdb.connect(database=self.db_file_path, read_only=False)
            self._logger.info(f"Successfully connected to DuckDB at: {self.db_file_path}")

            self._logger.info("Attempting to INSTALL and LOAD vss extension.")
            self._conn.execute("INSTALL vss;")
            self._conn.execute("LOAD vss;")
            self._conn.execute("SET hnsw_enable_experimental_persistence=true;")
            self._logger.info("DuckDB VSS extension loaded and persistence enabled successfully.")

        except Exception as e:
            self._logger.error(f"Error during DuckDB connection or VSS setup: {e}", exc_info=True)
            error_str = str(e).lower()
            if "already installed" in error_str or "already loaded" in error_str:
                self._logger.warning(f"VSS extension seems to be already installed/loaded, continuing...")
            else:
                raise RuntimeError(f"DuckDB connection/VSS setup failed: {e}") from e
        
        self._logger.info("<<<<< DuckDBResource SETUP_FOR_EXECUTION - END >>>>>")

    def teardown_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger.info(">>>>> DuckDBResource TEARDOWN_FOR_EXECUTION - START <<<<<")
        if self._conn:
            try:
                self._logger.info(f"Executing CHECKPOINT on DuckDB connection for: {self.db_file_path}")
                self._conn.execute("CHECKPOINT;")
                self._logger.info(f"CHECKPOINT executed successfully for: {self.db_file_path}")
            except Exception as e_checkpoint:
                self._logger.error(f"Error executing CHECKPOINT for DuckDB: {e_checkpoint}", exc_info=True)
            finally:
                self._logger.info(f"Closing DuckDB connection for: {self.db_file_path}")
                self._conn.close()
                self._conn = None 
        else:
            self.logger.info("No active DuckDB connection to teardown.")
        self._logger.info(">>>>> DuckDBResource TEARDOWN_FOR_EXECUTION - END <<<<<")

    @contextmanager
    def get_connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        if not self._conn:
            raise ConnectionError("DuckDB connection not established. Ensure setup_for_execution was successful.")
        yield self._conn

class SystemResource(dg.ConfigurableResource):
    _hw_manager: Optional[Any] = PrivateAttr(default=None)
    _hw_info: Optional[Any] = PrivateAttr(default=None)
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)
    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger = context.log
        if HardwareManager:
            self._hw_manager = HardwareManager()
            self._hw_info = self._hw_manager.get_hardware_info()
            self._logger.info(f"SystemResource hardware detection: {self._hw_info}")
        else:
            self._logger.warning("HardwareManager not available.")
    def get_recommended_concurrent_tasks(self, task_type: str = "cpu_bound_llm") -> int:
        if self._hw_manager: return self._hw_manager.recommend_concurrent_tasks(task_type=task_type)
        return 1