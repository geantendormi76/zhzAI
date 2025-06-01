# /home/zhz/zhz_agent/zhz_rag_pipeline_dagster/setup.py
from setuptools import find_packages, setup

setup(
    name="zhz_rag_pipeline_dagster_project", # 给一个包名
    version="0.0.1",
    packages=find_packages(), # 会找到 zhz_rag_pipeline 这个包
    install_requires=[
        "dagster",
        "dagster-webserver",
        # 添加其他 zhz_rag_pipeline_dagster 项目直接依赖的库
        # 例如 kuzu, dagster-pydantic (如果之后还要用) 等
        # 但核心的 zhz_rag 包的依赖不在这里列出，它应该是独立安装的
    ],
)