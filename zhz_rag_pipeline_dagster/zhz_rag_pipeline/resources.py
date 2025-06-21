# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/zhz_rag_pipeline/resources.py

import logging
import dagster as dg
import chromadb
from typing import List, Dict, Any, Optional, Iterator
import httpx
import json
import os
from contextlib import asynccontextmanager, contextmanager
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
    HardwareInfo = None


# --- GGUFEmbeddingResource: API客户端版本 ---

class GGUFEmbeddingResourceConfig(dg.Config):
    """
    GGUFEmbeddingResource 的配置类。
    """
    api_url: str = PydanticField(
        default="http://127.0.0.1:8089",
        description="URL of the standalone embedding API service."
    )

class GGUFEmbeddingResource(dg.ConfigurableResource):
    """
    用于与GGUF嵌入API服务交互的Dagster资源。
    负责初始化HTTP客户端、执行健康检查和文本编码。
    """
    api_url: str

    _client: httpx.AsyncClient = PrivateAttr()
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)
    _dimension: Optional[int] = PrivateAttr(default=None)
    _batch_size: int = PrivateAttr(default=128)

    def setup_for_execution(self, context: Optional[dg.InitResourceContext] = None) -> None:
        """
        初始化资源。现在可以接受一个可选的Dagster上下文。
        如果context为None（在FastAPI等非Dagster环境中使用），则使用默认配置。

        Args:
            context: Dagster的初始化资源上下文，可选。
        """
        self._logger = context.log if context else logging.getLogger("GGUFEmbeddingResource")
        self._client = httpx.AsyncClient(base_url=self.api_url, timeout=600.0)

        # 动态计算批处理大小
        try:
            if context and hasattr(context, 'resources_by_key') and "system_resource" in context.resources_by_key:
                system_resource = context.resources_by_key["system_resource"]
                physical_cores = system_resource._hw_info.cpu_physical_cores if system_resource._hw_info else 4
                self._batch_size = max(128, physical_cores * 64)
            else:
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
            else:
                raise RuntimeError(f"Embedding service at {self.api_url} is not healthy.")
        except Exception as e:
            self._logger.error(f"Failed to connect to embedding service: {e}")
            raise RuntimeError("Could not initialize GGUFEmbeddingResource.") from e

    def teardown_for_execution(self, context: dg.InitResourceContext) -> None:
        """
        在资源执行结束后关闭HTTP客户端。

        Args:
            context: Dagster的初始化资源上下文。
        """
        if hasattr(self, '_client') and not self._client.is_closed:
            async def _close():
                await self._client.aclose()
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_close())
            except RuntimeError:
                asyncio.run(_close())

    def get_embedding_dimension(self) -> int:
        """
        获取嵌入向量的维度。

        Returns:
            int: 嵌入向量的维度。

        Raises:
            ValueError: 如果嵌入维度不可用。
        """
        if self._dimension is None:
            raise ValueError("Embedding dimension not available.")
        return self._dimension

    def encode(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """
        将文本列表编码为嵌入向量。

        Args:
            texts: 待编码的文本列表。
            **kwargs: 额外的关键字参数。

        Returns:
            List[List[float]]: 文本对应的嵌入向量列表。
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        
        # 批处理循环
        for i in range(0, len(texts), self._batch_size):
            batch_texts = texts[i:i + self._batch_size]
            
            async def _async_encode_batch():
                try:
                    response = await self._client.post("/embed", json={"texts": batch_texts})
                    response.raise_for_status()
                    data = response.json()
                    return data.get("embeddings", [])
                except httpx.RequestError as e:
                    self._logger.error(f"Request to embedding service failed for a batch: {e}")
                    return [[] for _ in batch_texts]
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
    """
    ChromaDBResource 的配置类。
    """
    collection_name: str = PydanticField(
        default="zhz_rag_collection",
        description="Name of the ChromaDB collection."
    )
    persist_directory: str = PydanticField(
        default=os.path.join(os.getenv("ZHZ_AGENT_PROJECT_ROOT", "/home/zhz/zhz_agent"), "zhz_rag", "stored_data", "chromadb_index"),
        description="Directory to persist ChromaDB data."
    )

class ChromaDBResource(dg.ConfigurableResource):
    """
    用于管理ChromaDB向量数据库的Dagster资源。
    支持初始化客户端、添加嵌入和查询嵌入。
    """
    collection_name: str
    persist_directory: str

    _client: Optional[chromadb.PersistentClient] = PrivateAttr(default=None)
    _collection: Optional[Any] = PrivateAttr(default=None)
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)
    _batch_size: int = PrivateAttr(default=4096)

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        """
        初始化ChromaDB客户端和集合。

        Args:
            context: Dagster的初始化资源上下文。

        Raises:
            RuntimeError: 如果ChromaDB客户端或集合初始化失败。
        """
        self._logger = context.log
        os.makedirs(self.persist_directory, exist_ok=True)
        
        try:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = self._client.get_or_create_collection(name=self.collection_name)
        except Exception as e:
            self._logger.error(f"Failed to initialize ChromaDB client or collection: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize ChromaDBResource due to: {e}") from e

    def teardown_for_execution(self, context: dg.InitResourceContext) -> None:
        """
        清理ChromaDB资源。

        Args:
            context: Dagster的初始化资源上下文。
        """
        if self._client:
            pass # PersistentClient doesn't need explicit close
        self._client = None
        self._collection = None

    def add_embeddings(
        self, 
        ids: List[str], 
        embeddings: List[List[float]], 
        documents: Optional[List[str]] = None, 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        向ChromaDB集合添加嵌入向量。

        Args:
            ids: 文档ID列表。
            embeddings: 嵌入向量列表。
            documents: 原始文档文本列表，可选。
            metadatas: 文档元数据列表，可选。

        Raises:
            RuntimeError: 如果ChromaDB集合未初始化。
            Exception: 如果添加批次到ChromaDB失败。
        """
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
                raise
        
    def query_embeddings(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        从ChromaDB集合查询嵌入向量。

        Args:
            query_embeddings: 查询嵌入向量列表。
            n_results: 返回结果的数量，默认为5。
            where_filter: 用于过滤结果的条件字典，可选。
            include: 包含在结果中的字段，默认为["metadatas", "documents", "distances"]。

        Returns:
            Optional[Dict[str, Any]]: 查询结果字典，如果查询失败则为None。

        Raises:
            RuntimeError: 如果ChromaDB集合未初始化。
            Exception: 如果查询嵌入失败。
        """
        if self._collection is None:
            msg = "ChromaDB collection is not initialized. Cannot query embeddings."
            self._logger.error(msg)
            raise RuntimeError(msg)

        if include is None:
            include = ["metadatas", "documents", "distances"]

        try:
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
    """
    LocalLLMAPIResource 的配置类。
    """
    api_url: str = "http://127.0.0.1:8088/v1/chat/completions"
    default_temperature: float = 0.1
    default_max_new_tokens: int = 2048

class LocalLLMAPIResource(dg.ConfigurableResource):
    """
    用于与本地LLM API服务交互的Dagster资源。
    """
    api_url: str
    default_temperature: float
    default_max_new_tokens: int
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        """
        初始化本地LLM API资源。

        Args:
            context: Dagster的初始化资源上下文。
        """
        self._logger = context.log

    async def generate_structured_output(self, prompt: str, json_schema: Dict[str, Any], temperature: Optional[float] = None, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        从本地LLM生成结构化输出。

        Args:
            prompt: 发送给LLM的提示。
            json_schema: 期望的JSON输出结构。
            temperature: 生成温度，可选。
            max_new_tokens: 生成的最大新token数量，可选。

        Returns:
            Dict[str, Any]: LLM生成的结构化JSON输出。

        Raises:
            ValueError: 如果LLM响应格式不正确。
            Exception: 如果调用本地LLM失败。
        """
        logger_instance = self._logger if self._logger else dg.get_dagster_logger()
        temp_to_use = temperature if temperature is not None else self.default_temperature
        tokens_to_use = max_new_tokens if max_new_tokens is not None else self.default_max_new_tokens
        messages = [{"role": "user", "content": prompt}]
        payload = {"model": "local_kg_extraction_model", "messages": messages, "temperature": temp_to_use, "max_tokens": tokens_to_use, "response_format": {"type": "json_object", "schema": json_schema}}
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
    """
    GeminiAPIResource 的配置类。
    """
    model_name: str = PydanticField(default="gemini/gemini-1.5-flash-latest", description="Name of the Gemini model.")
    proxy_url: Optional[str] = PydanticField(default_factory=lambda: os.getenv("LITELLM_PROXY_URL"), description="Optional proxy URL for LiteLLM.")
    default_temperature: float = 0.1
    default_max_tokens: int = 2048
    
class GeminiAPIResource(dg.ConfigurableResource):
    """
    用于通过LiteLLM调用Gemini API的Dagster资源。
    """
    model_name: str
    proxy_url: Optional[str]
    default_temperature: float
    default_max_tokens: int
    _api_key: Optional[str] = PrivateAttr(default=None)
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        """
        初始化Gemini API资源，加载API密钥。

        Args:
            context: Dagster的初始化资源上下文。
        """
        self._logger = context.log
        self._api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self._api_key: self._logger.warning("Gemini API key not found.")

    async def call_completion(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Optional[str]:
        """
        调用Gemini API生成文本补全。

        Args:
            messages: 对话消息列表。
            temperature: 生成温度，可选。
            max_tokens: 生成的最大token数量，可选。

        Returns:
            Optional[str]: 生成的文本内容，如果调用失败则为None。
        """
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
    """
    用于管理DuckDB数据库连接的Dagster资源。
    支持连接数据库、加载VSS扩展和执行检查点。
    """
    db_file_path: str = PydanticField(
        default=os.path.join(os.getenv("ZHZ_AGENT_PROJECT_ROOT", "/home/zhz/zhz_agent"), "zhz_rag", "stored_data", "duckdb_knowledge_graph.db"),
        description="Path to the DuckDB database file."
    )
    _conn: Optional[duckdb.DuckDBPyConnection] = PrivateAttr(default=None)
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        """
        初始化DuckDB连接并加载VSS扩展。

        Args:
            context: Dagster的初始化资源上下文。

        Raises:
            RuntimeError: 如果DuckDB连接或VSS设置失败。
        """
        self._logger = context.log

        os.makedirs(os.path.dirname(self.db_file_path), exist_ok=True)
        
        try:
            self._conn = duckdb.connect(database=self.db_file_path, read_only=False)

            self._conn.execute("INSTALL vss;")
            self._conn.execute("LOAD vss;")
            self._conn.execute("SET hnsw_enable_experimental_persistence=true;")

        except Exception as e:
            self._logger.error(f"Error during DuckDB connection or VSS setup: {e}", exc_info=True)
            error_str = str(e).lower()
            if "already installed" in error_str or "already loaded" in error_str:
                self._logger.warning(f"VSS extension seems to be already installed/loaded, continuing...")
            else:
                raise RuntimeError(f"DuckDB connection/VSS setup failed: {e}") from e
        
    def teardown_for_execution(self, context: dg.InitResourceContext) -> None:
        """
        在资源执行结束后关闭DuckDB连接并执行检查点。

        Args:
            context: Dagster的初始化资源上下文。
        """
        if self._conn:
            try:
                self._conn.execute("CHECKPOINT;")
            except Exception as e_checkpoint:
                self._logger.error(f"Error executing CHECKPOINT for DuckDB: {e_checkpoint}", exc_info=True)
            finally:
                self._conn.close()
                self._conn = None 
        else:
            self._logger.info("No active DuckDB connection to teardown.")

    @contextmanager
    def get_connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """
        获取DuckDB数据库连接的上下文管理器。

        Yields:
            duckdb.DuckDBPyConnection: DuckDB数据库连接对象。

        Raises:
            ConnectionError: 如果DuckDB连接未建立。
        """
        if not self._conn:
            raise ConnectionError("DuckDB connection not established. Ensure setup_for_execution was successful.")
        yield self._conn

class SystemResource(dg.ConfigurableResource):
    """
    用于获取系统硬件信息和推荐并发任务数的Dagster资源。
    """
    _hw_manager: Optional[Any] = PrivateAttr(default=None)
    _hw_info: Optional[Any] = PrivateAttr(default=None)
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        """
        初始化系统资源，检测硬件信息。

        Args:
            context: Dagster的初始化资源上下文。
        """
        self._logger = context.log
        if HardwareManager:
            self._hw_manager = HardwareManager()
            self._hw_info = self._hw_manager.get_hardware_info()
        else:
            self._logger.warning("HardwareManager not available.")

    def get_recommended_concurrent_tasks(self, task_type: str = "cpu_bound_llm") -> int:
        """
        获取推荐的并发任务数。

        Args:
            task_type: 任务类型，默认为"cpu_bound_llm"。

        Returns:
            int: 推荐的并发任务数。
        """
        if self._hw_manager: return self._hw_manager.recommend_concurrent_tasks(task_type=task_type)
        return 1
