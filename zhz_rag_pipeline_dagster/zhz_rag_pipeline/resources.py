# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/zhz_rag_pipeline/resources.py
import dagster as dg
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Union, Optional, ContextManager, Iterator,Tuple
import logging
import httpx
import asyncio
import json
# from neo4j import GraphDatabase, Driver, Result # Neo4j不再直接用于此资源
import litellm
import os
import kuzu # 确保导入 kuzu
import shutil
from pydantic import PrivateAttr, Field as PydanticField
from contextlib import contextmanager
import portalocker # <--- 重新导入 portalocker
import time # <--- 导入 time，可能用于短暂等待

# --- SentenceTransformerResource ---
class SentenceTransformerResourceConfig(dg.Config):
    model_name_or_path: str = "/home/zhz/models/bge-small-zh-v1.5"

class SentenceTransformerResource(dg.ConfigurableResource):
    model_name_or_path: str
    _model: SentenceTransformer = PrivateAttr(default=None)
    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger = context.log
        self._logger.info(f"Initializing SentenceTransformer model from: {self.model_name_or_path}")
        try:
            self._model = SentenceTransformer(self.model_name_or_path)
            self._logger.info("SentenceTransformer model initialized successfully.")
        except Exception as e:
            self._logger.error(f"Failed to initialize SentenceTransformer model: {e}", exc_info=True)
            raise

    def encode(self, texts: List[str], batch_size: int = 32, normalize_embeddings: bool = True) -> List[List[float]]:
        if self._model is None:
            if self._logger:
                self._logger.error("SentenceTransformer model is not initialized in encode method.")
            raise RuntimeError("SentenceTransformer model is not initialized.")
        
        logger_instance = self._logger if self._logger else dg.get_dagster_logger()
        logger_instance.debug(f"Encoding {len(texts)} texts. Normalize embeddings: {normalize_embeddings}")
        
        embeddings_np = self._model.encode(
            texts, 
            batch_size=batch_size, 
            convert_to_tensor=False, 
            normalize_embeddings=normalize_embeddings
        )
        return [emb.tolist() for emb in embeddings_np]
    
    # --- [添加开始] ---
    def get_embedding_dimension(self) -> int:
        """返回嵌入模型的维度大小。"""
        if self._model and hasattr(self._model, 'get_sentence_embedding_dimension'):
            return self._model.get_sentence_embedding_dimension()
        return 512

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
        default=os.path.join("zhz_rag", "stored_data", "kuzu_default_db"),
        description="Path to the KuzuDB database directory."
    )

    _logger: Optional[dg.DagsterLogManager] = PrivateAttr(default=None)
    _resolved_db_path: str = PrivateAttr()
    _db: Optional[kuzu.Database] = PrivateAttr(default=None) # 这个 db 对象将在整个作业运行期间保持

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        """
        在作业开始时被调用一次。创建并持有 kuzu.Database 对象。
        """
        self._logger = context.log
        # 使用 context.instance.storage_directory() 可能更稳健，但我们先用之前的逻辑
        project_root = os.environ.get("DAGSTER_PROJECT_ROOT", os.getcwd())
        self._resolved_db_path = os.path.join(project_root, self.db_path_str)
        
        self._logger.info(
            f"KuzuDBReadWriteResource setup for run: DB path is '{self._resolved_db_path}'."
        )
        
        # 在每次作业运行开始时，清理旧数据库，确保从干净状态开始
        if os.path.exists(self._resolved_db_path):
            shutil.rmtree(self._resolved_db_path)
            self._logger.info(f"Cleaned up existing database directory: {self._resolved_db_path}")

        try:
            # 确保父目录存在
            os.makedirs(os.path.dirname(self._resolved_db_path), exist_ok=True)
            # 创建数据库实例，并将其保存在 self._db 中，供整个作业生命周期使用
            self._db = kuzu.Database(self._resolved_db_path)
            self._logger.info(f"Shared kuzu.Database object CREATED successfully for the entire run.")
        except Exception as e:
            self._logger.error(f"Failed to initialize shared kuzu.Database object: {e}", exc_info=True)
            raise

    def teardown_for_execution(self, context: dg.InitResourceContext) -> None:
        """
        在作业结束时被调用一次。销毁 kuzu.Database 对象以释放所有资源。
        """
        self._logger.info(f"Tearing down KuzuDBReadWriteResource for run...")
        if self._db is not None:
            # del self._db 是依赖析构函数，为了更明确，我们直接设为 None
            self._db = None
            self._logger.info("Shared kuzu.Database object destroyed, releasing file lock.")
        self._logger.info("KuzuDBReadWriteResource teardown complete.")

    @contextmanager
    def get_connection(self) -> Iterator[kuzu.Connection]:
        """
        一个上下文管理器，从【共享的】DB 对象获取一个新连接。
        这确保了所有资产操作都在同一个数据库实例上。
        """
        if self._db is None:
            raise ConnectionError("Shared KuzuDB Database object is not initialized. This should not happen if setup_for_execution was successful.")
        
        conn = None
        try:
            # 从保存在 self._db 中的共享数据库实例创建连接
            conn = kuzu.Connection(self._db)
            self._logger.debug("New kuzu.Connection obtained from shared Database object.")
            yield conn
        finally:
            if conn is not None:
                # 连接对象在使用后会被垃圾回收，无需显式关闭
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