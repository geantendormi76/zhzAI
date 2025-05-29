# zhz_rag_pipeline/resources.py
import dagster as dg
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings # 用于更细致的配置
from typing import List, Dict, Any, Union, Optional
import logging
import httpx # 需要导入 httpx
import asyncio # 如果SGLang调用是异步的
import json
from neo4j import GraphDatabase, Driver, Result # 导入neo4j驱动相关类


# 定义Resource的配置模型 (如果需要的话，例如模型路径)
class SentenceTransformerResourceConfig(dg.Config):
    model_name_or_path: str = "/home/zhz/models/bge-small-zh-v1.5" # 默认模型路径
    # device: str = "cpu" # 可以添加设备配置，如 "cuda"

# 定义Resource
class SentenceTransformerResource(dg.ConfigurableResource):
    model_name_or_path: str
    _model: SentenceTransformer = None
    _logger: logging.Logger = None # <--- 新增：声明 _logger 实例变量

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger = dg.get_dagster_logger() # <--- 新增：初始化 _logger
        self._logger.info(f"Initializing SentenceTransformer model from: {self.model_name_or_path}") # <--- 修改：使用 self._logger
        try:
            self._model = SentenceTransformer(
                self.model_name_or_path, 
            )
            self._logger.info("SentenceTransformer model initialized successfully.")
        except Exception as e:
            self._logger.error(f"Failed to initialize SentenceTransformer model: {e}")
            raise

    def encode(self, texts: List[str], batch_size: int = 32, normalize_embeddings: bool = True) -> List[List[float]]:
        if self._model is None:
            # 可以在这里也用 self._logger 记录错误
            if self._logger: # 检查 _logger 是否已初始化 (虽然理论上 setup 后应该有了)
                self._logger.error("SentenceTransformer model is not initialized in encode method.")
            raise RuntimeError("SentenceTransformer model is not initialized.")
        
        self._logger.debug(f"Encoding {len(texts)} texts. Normalize embeddings: {normalize_embeddings}") 
        
        embeddings_np = self._model.encode(
            texts, 
            batch_size=batch_size, 
            convert_to_tensor=False, 
            normalize_embeddings=normalize_embeddings
        )
        return [emb.tolist() for emb in embeddings_np]

# 定义ChromaDB Resource的配置模型
class ChromaDBResourceConfig(dg.Config):
    # path: Optional[str] = "/home/zhz/dagster_home/chroma_db" # 持久化存储路径
    # host: Optional[str] = None # 如果连接远程ChromaDB服务器
    # port: Optional[int] = None
    collection_name: str = "rag_documents" # 默认集合名称
    # settings: Optional[Dict[str, Any]] = None # 高级Chroma设置
    # 使用更具体的持久化路径配置
    persist_directory: str = "/home/zhz/dagster_home/chroma_data" # 推荐的持久化目录

# 定义ChromaDB Resource
class ChromaDBResource(dg.ConfigurableResource):
    collection_name: str
    persist_directory: str

    _client: chromadb.Client = None
    _collection: chromadb.Collection = None
    _logger: logging.Logger = None # <--- 新增：用于存储logger实例

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger = dg.get_dagster_logger() 

        self._logger.info(f"Initializing ChromaDB client and collection '{self.collection_name}'...")
        self._logger.info(f"ChromaDB data will be persisted to: {self.persist_directory}")
        try:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
            
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"} # <--- 新增/修改：指定距离度量为余弦
            )
            self._logger.info(f"ChromaDB collection '{self.collection_name}' initialized/loaded with cosine distance. Item count: {self._collection.count()}")
        except Exception as e:
            self._logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_embeddings(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]] = None):
        if self._collection is None:
            # 可以在这里也用 self._logger 记录错误，或者直接抛出异常
            self._logger.error("ChromaDB collection is not initialized. Cannot add embeddings.")
            raise RuntimeError("ChromaDB collection is not initialized.")
        
        if not (len(ids) == len(embeddings) and (metadatas is None or len(ids) == len(metadatas))):
            self._logger.error("Length mismatch for ids, embeddings, or metadatas.")
            raise ValueError("Length of ids, embeddings, and metadatas (if provided) must be the same.")

        if not ids:
            self._logger.info("No ids provided to add_embeddings, skipping.")
            return

        self._logger.info(f"Adding/updating {len(ids)} embeddings to ChromaDB collection '{self.collection_name}'...") # <--- 修改：使用 self._logger
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
        self._logger.info(f"Embeddings added/updated. Collection count now: {self._collection.count()}") # <--- 修改：使用 self._logger

    def query_embeddings(self, query_embeddings: List[List[float]], n_results: int = 5) -> chromadb.QueryResult:
        if self._collection is None:
            self._logger.error("ChromaDB collection is not initialized. Cannot query embeddings.")
            raise RuntimeError("ChromaDB collection is not initialized.")
        # query方法本身可能没有太多需要日志记录的，除非你想记录查询参数或结果数量
        self._logger.debug(f"Querying ChromaDB collection '{self.collection_name}' with {len(query_embeddings)} vectors, n_results={n_results}.")
        return self._collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results
        )
    
# 定义SGLang Resource的配置模型
class SGLangAPIResourceConfig(dg.Config):
    api_url: str = "http://127.0.0.1:30000/generate"
    default_temperature: float = 0.1
    default_max_new_tokens: int = 512 # 根据KG提取的典型输出长度调整
    # default_stop_tokens: List[str] = ["<|im_end|>"] # 可以设默认停止标记

# 定义SGLang Resource
class SGLangAPIResource(dg.ConfigurableResource):
    api_url: str
    default_temperature: float
    default_max_new_tokens: int
    # default_stop_tokens: List[str]

    _logger: logging.Logger = None

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger = dg.get_dagster_logger()
        self._logger.info(f"SGLangAPIResource configured with API URL: {self.api_url}")
        # 这里不需要实际的连接或初始化，因为是无状态的HTTP API调用

    async def generate_structured_output(
        self, 
        prompt: str, 
        json_schema: Dict[str, Any],
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        # stop_tokens: Optional[List[str]] = None # 可选
    ) -> Dict[str, Any]: # 期望返回解析后的JSON字典
        
        temp_to_use = temperature if temperature is not None else self.default_temperature
        tokens_to_use = max_new_tokens if max_new_tokens is not None else self.default_max_new_tokens
        # stop_to_use = stop_tokens if stop_tokens is not None else self.default_stop_tokens

        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": temp_to_use,
                "max_new_tokens": tokens_to_use,
                # "stop": stop_to_use, # 根据模型调整
                "stop": ["<|im_end|>"], # 假设Qwen
                "json_schema": json.dumps(json_schema) # 确保传递JSON字符串
            }
        }
        self._logger.debug(f"Sending request to SGLang. Prompt (start): {prompt[:100]}... Schema: {json.dumps(json_schema)}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client: # 增加超时时间
                response = await client.post(self.api_url, json=payload)
                response.raise_for_status()
                
                response_json = response.json()
                generated_text = response_json.get("text", "").strip()
                
                self._logger.debug(f"SGLang raw response text: {generated_text}")
                # 尝试解析
                try:
                    parsed_output = json.loads(generated_text)
                    return parsed_output
                except json.JSONDecodeError as e:
                    self._logger.error(f"Failed to decode SGLang JSON output: {generated_text}. Error: {e}")
                    raise ValueError(f"SGLang output was not valid JSON: {generated_text}") from e

        except httpx.HTTPStatusError as e:
            self._logger.error(f"SGLang API HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            self._logger.error(f"SGLang API request error: {e}")
            raise
        except Exception as e:
            self._logger.error(f"Unexpected error during SGLang call: {e}")
            raise

# 定义Neo4j Resource的配置模型
class Neo4jResourceConfig(dg.Config):
    uri: str = "bolt://localhost:7687" # 从您的 .env 文件获取
    user: str = "neo4j"                 # 从您的 .env 文件获取
    password: str = "zhz199276"          # 从您的 .env 文件获取
    database: str = "neo4j"             # 默认数据库，可以配置

# 定义Neo4j Resource
class Neo4jResource(dg.ConfigurableResource):
    uri: str
    user: str
    password: str
    database: str

    _driver: Driver = None # Neo4j驱动实例
    _logger: logging.Logger = None

    def setup_for_execution(self, context: dg.InitResourceContext) -> None:
        self._logger = dg.get_dagster_logger()
        self._logger.info(f"Initializing Neo4j Driver for URI: {self.uri}, Database: {self.database}")
        try:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # 验证连接 (可选但推荐)
            with self._driver.session(database=self.database) as session:
                session.run("RETURN 1").consume() # 一个简单的查询来测试连接
            self._logger.info("Neo4j Driver initialized and connection verified successfully.")
        except Exception as e:
            self._logger.error(f"Failed to initialize Neo4j Driver or verify connection: {e}")
            raise

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Result:
        """
        执行一个Cypher查询并返回结果。
        """
        if self._driver is None:
            self._logger.error("Neo4j Driver is not initialized.")
            raise RuntimeError("Neo4j Driver is not initialized.")
        
        self._logger.debug(f"Executing Neo4j query: {query} with parameters: {parameters}")
        with self._driver.session(database=self.database) as session:
            result = session.run(query, parameters)
            return result # 返回 Result 对象，调用方可以处理

    def execute_write_queries(self, queries_with_params: List[tuple[str, Dict[str, Any]]]):
        """
        在单个事务中执行多个写操作查询。
        queries_with_params: 一个元组列表，每个元组是 (cypher_query_string, parameters_dict)
        """
        if self._driver is None:
            self._logger.error("Neo4j Driver is not initialized.")
            raise RuntimeError("Neo4j Driver is not initialized.")

        with self._driver.session(database=self.database) as session:
            with session.begin_transaction() as tx:
                for query, params in queries_with_params:
                    self._logger.debug(f"Executing in transaction: {query} with params: {params}")
                    tx.run(query, params)
                tx.commit()
        self._logger.info(f"Executed {len(queries_with_params)} write queries in a transaction.")


    def teardown_for_execution(self, context: dg.InitResourceContext) -> None:
        """
        在Dagster进程结束时关闭Neo4j驱动程序。
        """
        if self._driver is not None:
            self._logger.info("Closing Neo4j Driver.")
            self._driver.close()
            self._driver = None