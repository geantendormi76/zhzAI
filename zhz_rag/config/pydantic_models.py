# /home/zhz/zhz_agent/zhz_rag/config/pydantic_models.py
from pydantic import BaseModel, Field, root_validator
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import uuid

# --- RAG Models ---
class QueryRequest(BaseModel):
    query: str = Field(description="用户提出的原始查询文本。", json_schema_extra=lambda schema: schema.pop('default', None) if schema.get('default') is None else None)
    top_k_vector: int = Field(description="期望检索的向量搜索结果数量。", json_schema_extra=lambda schema: schema.pop('default', None) if schema.get('default') is None else None)
    top_k_kg: int = Field(description="期望检索的知识图谱结果数量。", json_schema_extra=lambda schema: schema.pop('default', None) if schema.get('default') is None else None)
    top_k_bm25: int = Field(description="期望检索的 BM25 关键词搜索结果数量。", json_schema_extra=lambda schema: schema.pop('default', None) if schema.get('default') is None else None)
    top_k_final: int = Field(default=3, description="最终融合后返回的文档数量。") 

    @root_validator(pre=True)
    @classmethod
    def remove_internal_params(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(values, dict):
            values.pop('security_context', None)
            values.pop('agent_fingerprint', None)
        return values

    class Config:
        extra = 'forbid'

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
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    REMINDING = "reminding"

class ReminderMethod(str, Enum):
    NOTIFICATION = "notification"

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
    tags: List[str] = Field(default_factory=list, description="任务标签")
    action_type: Optional[str] = Field(None, description="任务到期时需要执行的动作类型 (例如 'navigate', 'send_message', 'run_report')")
    action_payload: Dict[str, Any] = Field(default_factory=dict, description="执行动作时需要的参数 (例如导航的目的地)")
    execution_result: Optional[str] = Field(None, description="任务执行后的结果或错误信息")
    last_executed_at: Optional[datetime] = Field(None, description="上次执行时间 (UTC)")

    class Config:
        use_enum_values = True
        # Pydantic V2 uses model_config, or just rely on default behavior if orm_mode was for SQLAlchemy V1 style
        # For Pydantic V2, if you need ORM mode (e.g. for SQLAlchemy integration):
        # model_config = {"from_attributes": True} 
        # However, if you are not directly using it with an ORM that way, orm_mode=True might be a V1 relic.
        # For now, I will keep it as orm_mode = True as per your original file,
        # but be aware it might need adjustment if you are on Pydantic V2 and using its ORM features.
        orm_mode = True


class CreateTaskRequest(BaseModel):
    title: str
    description: Optional[str] = None
    due_date: Optional[datetime] = None
    reminder_offset_minutes: Optional[int] = None
    reminder_methods: Optional[List[ReminderMethod]] = [ReminderMethod.NOTIFICATION]
    priority: Optional[int] = 0
    tags: Optional[List[str]] = None
    action_type: Optional[str] = None
    action_payload: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = 'forbid'

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

    class Config:
        extra = 'forbid'


class IdentifiedEntity(BaseModel):
    text: str = Field(description="识别出的实体文本。")
    label: Optional[str] = Field(None, description="推断的实体类型 (例如 PERSON, ORGANIZATION, TASK)。")

class ExtractedEntitiesAndRelationIntent(BaseModel):
    entities: List[IdentifiedEntity] = Field(default_factory=list, description="从用户查询中识别出的核心实体列表（通常1-2个）。")
    relation_hint: Optional[str] = Field(None, description="如果用户查询暗示了实体间的特定关系，这里是关系的文本描述或关键词（例如 “工作于”, “负责”, “销售额”）。")