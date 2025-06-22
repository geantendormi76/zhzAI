# /home/zhz/zhz_agent/zhz_rag/config/pydantic_models.py
from pydantic import BaseModel, Field, root_validator
from typing import List, Dict, Any, Optional, Union, Literal
from enum import Enum
from datetime import datetime
import uuid

# --- RAG Models ---
class QueryRequest(BaseModel):
    query: str
    # --- START: 新增 filters 字段 ---
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata filter to apply during retrieval.")
    # --- END: 新增 filters 字段 ---
    # --- 修改：为所有 top_k 参数提供默认值，使其变为可选 ---
    top_k_vector: int = Field(default=3, description="Number of results to retrieve from vector search.")
    top_k_bm25: int = Field(default=3, description="Number of results to retrieve from BM25 search.")
    top_k_kg: int = Field(default=2, description="Number of results to retrieve from Knowledge Graph search.")
    top_k_final: int = Field(default=3, description="Number of final results after fusion and reranking.")
    # --- 结束修改 ---

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the main objectives of the project?",
                "filters": {"must": [{"key": "filename", "match": {"value": "report.docx"}}]}, # <-- 更新示例
                "top_k_vector": 3,
                "top_k_bm25": 3,
                "top_k_kg": 2,
                "top_k_final": 3
            }
        }
        # 移除了 root_validator 和 extra='forbid' 以简化并遵循新的实践
        # 如果您仍需要严格的字段检查，可以将 extra='forbid' 放回

class RetrievedDocument(BaseModel):
    source_type: str
    content: str
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

# --- V4.1: Actionable Suggestion Models ---
class PendingTaskSuggestion(BaseModel):
    """
    当用户未指定提醒时间时，返回此结构，由前端引导用户确认。
    """
    suggestion_type: Literal["pending_confirmation"] = "pending_confirmation"
    title: str = Field(description="The suggested title for the task.")
    due_date: str = Field(description="The suggested due date for the task in 'YYYY-MM-DD HH:MM:SS' format.")
    default_reminder_options: List[int] = Field(default=[10, 30, 60], description="Default reminder offsets in minutes for the UI to display.")

class AutoScheduledTaskConfirmation(BaseModel):
    """
    当用户已明确指定提醒时间时，返回此结构，告知用户任务已自动创建。
    """
    suggestion_type: Literal["auto_scheduled"] = "auto_scheduled"
    title: str = Field(description="The title of the task that was automatically created.")
    due_date: str = Field(description="The due date of the task in 'YYYY-MM-DD HH:MM:SS' format.")
    reminder_offset_minutes: int = Field(description="The specific reminder offset in minutes that was set based on user's request.")
    confirmation_message: str = Field(description="A user-friendly message confirming the action.")


# --- V4.2: User Intent Classification Models ---
class IntentType(str, Enum):
    """Enumeration for the classified user intent."""
    RAG_QUERY = "rag_query"
    TASK_CREATION = "task_creation"
    MIXED_INTENT = "mixed_intent"

class UserIntent(BaseModel):
    """
    Represents the classified intent of the user's query.
    """
    intent: IntentType = Field(description="The primary intent classified by the LLM.")
    reasoning: str = Field(description="A brief explanation from the LLM on why it chose this intent.")



class HybridRAGResponse(BaseModel):
    original_query: str
    answer: str
    retrieved_sources: List[RetrievedDocument]
    actionable_suggestion: Optional[Union[PendingTaskSuggestion, AutoScheduledTaskConfirmation]] = Field(
        default=None, 
        description="An actionable task extracted from the dialogue, if any. The structure depends on whether user input is needed."
    )
    debug_info: Optional[Dict[str, Any]] = None


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
        from_attributes = True


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

class ExtractedRelationItem(BaseModel): # 新建一个类名以避免与可能的其他同名类冲突
    head_entity_text: str
    head_entity_label: str
    relation_type: str
    tail_entity_text: str
    tail_entity_label: str

class ExtractedEntitiesAndRelationIntent(BaseModel):
    entities: List[IdentifiedEntity] = Field(default_factory=list, description="从用户查询中识别出的核心实体列表。")
    # --- 新增 "relations" 字段 ---
    relations: List[ExtractedRelationItem] = Field(default_factory=list, description="从用户查询中识别出的关系列表。")
    # --- "relation_hint" 字段可以保留，或者如果您觉得 "relations" 列表更全面，可以考虑移除或标记为废弃 ---
    relation_hint: Optional[str] = Field(None, description="[可选的旧字段] 如果用户查询暗示了实体间的特定关系，这里是关系的文本描述或关键词。新的 'relations' 列表更推荐。")


class QueryExpansionAndKGExtractionOutput(BaseModel):
    expanded_queries: List[str] = Field(
        default_factory=list,
        description="A list of expanded or related questions generated from the original query."
    )
    extracted_entities_for_kg: ExtractedEntitiesAndRelationIntent = Field(
        default_factory=ExtractedEntitiesAndRelationIntent,
        description="Structured entities and relations extracted for Knowledge Graph querying."
    )
    metadata_filter: Optional[Dict[str, Any]] = Field( # <--- 新增字段
        default=None,
        description="An optional metadata filter (e.g., {'filename': 'report.docx'}) to apply during retrieval, if a specific source is mentioned."
    )
    
class RagQueryPlan(BaseModel):
    """
    Represents the output of the V2 RAG query planner.
    It contains the core query string and a metadata filter for precise retrieval.
    """
    query: str = Field(description="The refined, core query string for semantic vector search.")
    metadata_filter: Dict[str, Any] = Field(
        default_factory=dict,
        description="A ChromaDB-compatible 'where' filter for metadata-based pre-filtering of document chunks."
    )

