# /home/zhz/zhz_agent/zhz_rag/setup.py
from setuptools import find_packages, setup

setup(
    name="zhz_rag_core",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Pydantic 版本由主 requirements.txt 控制
        # LiteLLM 版本由主 requirements.txt 控制
        # ChromaDB 版本由主 requirements.txt 控制

        "protobuf>=4.25.0,<5.30.0", # 放宽 protobuf 上限，因为 pydantic 2.11.5 可能需要较新的
        "packaging>=23.2,<25.0",
        "rich>=13.7.0,<14.0.0",
        
        "fastapi>=0.110.0,<0.116.0", # 保持较新
        "starlette>=0.35.0,<0.47.0", # 保持较新

        "langchain-core>=0.1.50,<0.4.0", # 较新 langchain 可能更好兼容
        "langchain-text-splitters>=0.0.1,<0.3.0",

        "httpx>=0.27.0", # 使用较新 httpx
        "python-dotenv>=1.0.0",
        "neo4j>=5.0.0", # neo4j 驱动
        "sentence-transformers>=2.2.0", # sentence-transformers
        "transformers>=4.38.0,<4.39.0", # 固定您之前的版本或小幅更新
        "torch>=2.0.0",
        "numpy<2.0", # 保持 Numpy < 2.0
        "bm25s",
        "jieba",
        "uvicorn[standard]", # 添加 standard extras
        "pandas>=2.0.0",
        "sqlalchemy>=2.0.0",
        "databases[aiosqlite]>=0.9.0", # for async sqlite
        "apscheduler>=3.10.0",
        "pytz",
    ],
)