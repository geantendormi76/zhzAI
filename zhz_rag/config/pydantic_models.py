# zhz_agent/pydantic_models.py
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import uuid

# --- RAG Models ---
class QueryRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    query: str = Field(description="用户提出的原始查询文本。", json_schema_extra=lambda schema: schema.pop('default', None) if schema.get('default') is None else None)
    top_k_vector: int = Field(description="期望检索的向量搜索结果数量。", json_schema_extra=lambda schema: schema.pop('default', None) if schema.get('default') is None else None)
    top_k_kg: int = Field(description="期望检索的知识图谱结果数量。", json_schema_extra=lambda schema: schema.pop('default', None) if schema.get('default') is None else None)
    top_k_bm25: int = Field(description="期望检索的 BM25 关键词搜索结果数量。", json_schema_extra=lambda schema: schema.pop('default', None) if schema.get('default') is None else None)

    @model_validator(mode='before')
    @classmethod
    def remove_internal_params(cls, data: Any) -> Any:
        if isinstance(data, dict):
            print(f"Pydantic DEBUG (QueryRequest before validation): Received data for validation: {str(data)[:500]}")
            removed_security_context = data.pop('security_context', None)
            if removed_security_context:
                print(f"Pydantic INFO (QueryRequest before validation): Removed 'security_context': {str(removed_security_context)[:100]}")
            removed_agent_fingerprint = data.pop('agent_fingerprint', None)
            if removed_agent_fingerprint:
                print(f"Pydantic INFO (QueryRequest before validation): Removed 'agent_fingerprint': {str(removed_agent_fingerprint)[:100]}")
        return data

class RetrievedDocument(BaseModel):
    source_type: str
    content: str
    score: Optional[float] = Field(default=None, json_schema_extra=lambda schema: schema.pop('default', None) if schema.get('default') is None else None)
    metadata: Optional[Dict[str, Any]] = Field(default=None, json_schema_extra=lambda schema: schema.pop('default', None) if schema.get('default') is None else None)

class HybridRAGResponse(BaseModel):
    original_query: str
    answer: str
    retrieved_sources: List[RetrievedDocument]
    debug_info: Optional[Dict[str, Any]] = Field(default=None, json_schema_extra=lambda schema: schema.pop('default', None) if schema.get('default') is None else None)


# --- Task Management Models ---
class TaskStatus(str, Enum):
    PENDING = "pending"      # 待处理 (新创建，尚未到执行时间)
    ACTIVE = "active"        # 活动 (已到执行时间，等待执行或正在执行)
    COMPLETED = "completed"  # 已完成
    CANCELLED = "cancelled"  # 已取消
    FAILED = "failed"        # 执行失败
    REMINDING = "reminding"    # 提醒中 (可选状态)

class ReminderMethod(str, Enum):
    NOTIFICATION = "notification" # 桌面通知

class TaskModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="任务的唯一ID (自动生成)")
    title: str = Field(description="任务标题")
    description: Optional[str] = Field(None, description="任务的详细描述")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="任务当前状态")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="任务创建时间 (UTC)")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="任务最后更新时间 (UTC)")
    due_date: Optional[datetime] = Field(None, description="任务截止日期或计划执行时间 (UTC)")
    reminder_time: Optional[datetime] = Field(None, description="任务提醒时间 (UTC)")
    reminder_offset_minutes: Optional[int] = Field(None, description="提醒时间相对于due_date的提前分钟数 (例如10分钟前)")
    reminder_methods: List[ReminderMethod] = Field(default=[ReminderMethod.NOTIFICATION], description="提醒方式列表")
    priority: int = Field(default=0, description="任务优先级 (例如 0:普通, 1:重要, 2:紧急)")
    tags: List[str] = Field(default_factory=list, description="任务标签") # 确保默认为空列表
    action_type: Optional[str] = Field(None, description="任务到期时需要执行的动作类型 (例如 'navigate', 'send_message', 'run_report')")
    action_payload: Dict[str, Any] = Field(default_factory=dict, description="执行动作时需要的参数 (例如导航的目的地)") # 确保默认为空字典
    execution_result: Optional[str] = Field(None, description="任务执行后的结果或错误信息")
    last_executed_at: Optional[datetime] = Field(None, description="上次执行时间 (UTC)")

    model_config = ConfigDict(
        use_enum_values=True,
        from_attributes=True,
    )

class CreateTaskRequest(BaseModel):
    title: str
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    reminder_offset_minutes: Optional[int] = None # 例如 "10" 代表提前10分钟
    reminder_methods: Optional[List[ReminderMethod]] = [ReminderMethod.NOTIFICATION]
    priority: Optional[int] = 0
    tags: Optional[List[str]] = None
    action_type: Optional[str] = None
    action_payload: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(extra='forbid')

class UpdateTaskRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[TaskStatus] = None
    due_date: Optional[datetime] = None
    reminder_offset_minutes: Optional[int] = None
    reminder_methods: Optional[List[ReminderMethod]] = None
    priority: Optional[int] = None
    tags: Optional[List[str]] = None
    action_type: Optional[str] = None
    action_payload: Optional[Dict[str, Any]] = None
    model_config = ConfigDict(extra='forbid')