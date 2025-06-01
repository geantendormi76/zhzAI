# /home/zhz/zhz_agent/zhz_rag/setup.py
from setuptools import find_packages, setup

setup(
    name="zhz_rag_core",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=1.10.18,<1.11.0",     # 强制 Pydantic V1.x
        "protobuf>=4.25.0,<5.0",       # 强制 Protobuf V4.x
        "packaging>=23.2,<24.0",
        "rich>=13.7.0,<14.0.0",
        
        "fastapi>=0.95.0,<0.100.0",
        "starlette>=0.20.4,<0.28.0",

        "langchain-core>=0.1.0,<0.2.0",
        "langchain-text-splitters>=0.0.1,<0.1.0",

        "httpx",
        "python-dotenv",
        "neo4j",
        "sentence-transformers",
        "transformers==4.38.2", 
        "torch",
        "numpy<2.0",                 
        "chromadb==0.4.24", 
        "bm25s",
        "jieba",
        "uvicorn", 
        "litellm>=1.15.0,<1.16.0", # <-- 修改：指定与 Pydantic V1 兼容的 litellm 版本
        "pandas",
        "sqlalchemy",
        "databases",
        "apscheduler",
        "pytz",
    ],
)