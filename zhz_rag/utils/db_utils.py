# ZHZ_AGENT/database.py
import os
from databases import Database
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from typing import Optional

# --- APScheduler 相关导入 ---
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
# --- [修改] 明确导入并使用 pytz ---
import pytz #

# --- 数据库配置 ---
ZHZ_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = "ZHZ_AGENT_tasks.db"
DATABASE_FILE_PATH = os.path.join(ZHZ_AGENT_DIR, DB_NAME)
DATABASE_URL = f"sqlite+aiosqlite:///{DATABASE_FILE_PATH}"

database = Database(DATABASE_URL)
sqlalchemy_engine = create_engine(DATABASE_URL.replace("+aiosqlite", ""))
Base = declarative_base() #

# --- 全局调度器实例定义 ---
scheduler: Optional[AsyncIOScheduler] = None

def get_scheduler() -> AsyncIOScheduler:
    """获取或创建调度器实例，并配置作业存储和 UTC 时区。"""
    global scheduler
    if scheduler is None:
        jobstore_url = f"sqlite:///{DATABASE_FILE_PATH}"
        jobstores = {
            'default': SQLAlchemyJobStore(url=jobstore_url, tablename='apscheduler_jobs_v2') #
        }
        # --- [修复] 明确使用 pytz.utc 设置时区 ---
        scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            timezone=pytz.utc # <--- 强制使用 pytz.utc #
        )
        import logging
        logging.getLogger('apscheduler').setLevel(logging.DEBUG)
        print(f"APScheduler initialized with timezone: {pytz.utc}") # 确认使用 pytz.utc #
    return scheduler #