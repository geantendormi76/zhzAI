# zhz_agent/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
import logging
from typing import Type, List, Dict, Any, Optional, ClassVar

# --- [修改] 导入 -> 改为绝对导入 ---
from zhz_rag.utils.db_utils import database, sqlalchemy_engine, Base, get_scheduler
from zhz_rag.api.task_manager_api import router as tasks_router
from zhz_rag.task_management import db_models # 确保 SQLAlchemy 模型被导入

@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    print("--- Main FastAPI 应用启动 (已集成任务管理API 和 APScheduler) ---") # [修改] 更新了描述

    # --- 数据库初始化 ---
    await database.connect()
    print("数据库已连接。")
    try:
        Base.metadata.create_all(bind=sqlalchemy_engine)
        print("数据库表已执行 create_all。")
        from sqlalchemy import inspect
        inspector = inspect(sqlalchemy_engine)
        if inspector.has_table("tasks"):
            print("'tasks' 表已成功创建/存在于数据库中。")
        else:
            print("警告: 'tasks' 表在 create_all 之后仍未找到！这通常意味着模型没有在 create_all 之前被正确导入。")
            print(f"   已知的表: {inspector.get_table_names()}") # 打印所有实际创建的表
            print(f"   Base.metadata.tables: {Base.metadata.tables.keys()}") # 打印 SQLAlchemy 元数据中已注册的表
    except Exception as e:
        print(f"创建或检查数据库表时出错: {e}")
        import traceback
        traceback.print_exc() # 打印详细的异常堆栈


    # --- [修改] APScheduler 初始化 (使用 get_scheduler) ---
    current_scheduler = get_scheduler() # <--- 获取调度器实例
    try:
        logging.getLogger('apscheduler').setLevel(logging.DEBUG) # 设置为 DEBUG 级别

        if not current_scheduler.running: # <--- 只有在未运行时才启动
            current_scheduler.start()
            print("APScheduler 已启动并使用数据库作业存储。")
        else:
            print("APScheduler 已在运行。")
    except Exception as e:
        print(f"APScheduler 启动失败: {e}")

    print("RAG 组件的初始化和管理在 zhz_agent_mcp_server.py。")
    print("任务管理API已在 /tasks 路径下可用。")

    yield # FastAPI 应用在此运行

    print("--- Main FastAPI 应用关闭 ---")
    current_scheduler_on_shutdown = get_scheduler() # <--- 再次获取以确保是同一个实例
    if current_scheduler_on_shutdown and current_scheduler_on_shutdown.running:
        current_scheduler_on_shutdown.shutdown()
        print("APScheduler 已关闭。")
    await database.disconnect()
    print("数据库已断开连接。")
    print("RAG 组件的清理在 zhz_agent_mcp_server.py。")

# --- App 定义 (保持不变) ---
app = FastAPI(
    title="Hybrid RAG Backend with Task Management",
    description="主 FastAPI 应用，负责接收请求、编排 Agent，并提供任务管理API。",
    version="0.2.1",
    lifespan=lifespan
)

app.include_router(tasks_router)

@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the Hybrid RAG Backend Main App.",
        "available_services": {
            "task_management": "/tasks/docs",
            "rag_via_mcpo": "mcpo proxy at port 8006 (see mcpo_servers.json)"
        }
    }

if __name__ == "__main__":
    print("--- 启动 Main FastAPI 服务器 (包含任务管理API) ---")
    uvicorn.run("zhz_agent.main:app", host="0.0.0.0", port=8000, reload=True) # Ensure correct run command