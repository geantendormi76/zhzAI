# zhz_agent/database_models.py
from sqlalchemy import Column, String, DateTime, Integer, Text, Enum as SQLAlchemyEnum, ForeignKey, Boolean, JSON
from sqlalchemy.sql import func
import uuid

# --- [修改] 从 pydantic_models 导入枚举 -> 改为绝对导入 ---
from zhz_agent.pydantic_models import TaskStatus, ReminderMethod

# --- [修改] 从新的 database.py 导入 Base -> 改为绝对导入 ---
from zhz_agent.database import Base # <--- 确保只从这里导入 Base #

class TaskDB(Base): # 命名为 TaskDB 以区分 Pydantic 的 TaskModel
    __tablename__ = "tasks"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, index=True, nullable=False)
    description = Column(Text, nullable=True) #
    status = Column(SQLAlchemyEnum(TaskStatus), default=TaskStatus.PENDING, nullable=False) #
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False) #
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False) #
    due_date = Column(DateTime(timezone=True), nullable=True) #
    reminder_time = Column(DateTime(timezone=True), nullable=True) #
    reminder_offset_minutes = Column(Integer, nullable=True) #
    reminder_methods = Column(JSON, default=[ReminderMethod.NOTIFICATION.value], nullable=False) #
    priority = Column(Integer, default=0, nullable=False) #
    tags = Column(JSON, default=[], nullable=False) #
    action_type = Column(String, nullable=True) #
    action_payload = Column(JSON, default={}, nullable=True) #
    execution_result = Column(Text, nullable=True) #
    last_executed_at = Column(DateTime(timezone=True), nullable=True) #

    def __repr__(self):
        return f"<TaskDB(id={self.id}, title='{self.title}', status='{self.status.value}')>"