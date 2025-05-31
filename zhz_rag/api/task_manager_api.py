# zhz_agent/task_manager_service.py
from fastapi import APIRouter, HTTPException, Depends, Body, Query, Path, status
from typing import List, Optional, Any, cast
from datetime import datetime, timedelta
import uuid
import traceback # 导入 traceback
import pytz

# --- [修改] 从 pydantic_models 导入我们定义的模型 -> 改为绝对导入 ---
from zhz_rag.config.pydantic_models import TaskModel, CreateTaskRequest, UpdateTaskRequest, TaskStatus, ReminderMethod

# --- [修改] 从 database_models 导入 SQLAlchemy 表模型 -> 改为绝对导入 ---
from zhz_rag.task_management.db_models import TaskDB

# --- [修改] 从新的 database.py 导入 database 对象 和 get_scheduler -> 改为绝对导入 ---
from zhz_rag.utils.db_utils import database, get_scheduler # 将 db_utils 修改为 database

# --- [修改] 从 .task_jobs 导入作业函数 -> 改为绝对导入 ---
from zhz_rag.task_management.jobs import send_task_reminder, execute_task_action
from apscheduler.triggers.date import DateTrigger # 用于指定精确的运行时间
from apscheduler.jobstores.base import JobLookupError # <--- [修改] 导入 JobLookupError 的正确路径

# APIRouter 实例
router = APIRouter(
    prefix="/tasks",
    tags=["Task Management"],
    responses={404: {"description": "Not found"}},
)

def _ensure_utc(dt: datetime) -> datetime:
    """确保 datetime 对象是 UTC 时区感知的。"""
    if dt.tzinfo is None:
        return pytz.utc.localize(dt) # 如果是朴素时间，假定它是UTC并设为UTC
    return dt.astimezone(pytz.utc) # 如果是其他时区，转换为UTC

def _schedule_task_jobs(task: TaskModel):
    current_scheduler = get_scheduler() # 获取 scheduler 实例
    print(f"DEBUG SCHEDULER: _schedule_task_jobs called. Scheduler instance: {current_scheduler}, Is running: {current_scheduler.running if current_scheduler else 'N/A'}")
    if not current_scheduler or not current_scheduler.running:
        print("SCHEDULER_ERROR: APScheduler 未运行，无法调度作业。")
        return

    # 提醒作业
    if task.reminder_time and task.status == TaskStatus.PENDING:
        reminder_job_id = f"reminder_{task.id}"
        try:
            reminder_methods_list = task.reminder_methods
            reminder_utc = _ensure_utc(task.reminder_time)
            print(f"SCHEDULER DEBUG: Passing reminder_methods to job: {reminder_methods_list}") # 添加日志

            current_scheduler.add_job(
                send_task_reminder,
                trigger=DateTrigger(run_date=reminder_utc),
                args=[task.id, task.title, reminder_methods_list], # <--- [修复] 直接传递列表
                id=reminder_job_id,
                name=f"Reminder for task {task.title[:20]}",
                replace_existing=True
            )
            print(f"SCHEDULER: 已为任务 '{task.id}' 添加/更新提醒作业，运行于 {task.reminder_time}")
        except Exception as e:
            print(f"SCHEDULER_ERROR: 添加提醒作业失败 for task '{task.id}': {e}")
            traceback.print_exc() # 打印详细错误堆栈

    # 执行作业
    if task.due_date and task.status == TaskStatus.PENDING:
        execution_job_id = f"execution_{task.id}"
        try:
            due_utc = _ensure_utc(task.due_date) # <--- [新增] 确保时间是 UTC 感知的
            print(f"SCHEDULER DEBUG: Adding execution job at {due_utc} ({due_utc.tzinfo})") # <--- [新增] 添加时区日志
            current_scheduler.add_job(
                execute_task_action,
                trigger=DateTrigger(run_date=due_utc),
                args=[task.id, task.action_type, task.action_payload],
                id=execution_job_id,
                name=f"Execution for task {task.title[:20]}",
                replace_existing=True
            )
            print(f"SCHEDULER: 已为任务 '{task.id}' 添加/更新执行作业，运行于 {task.due_date}")
        except Exception as e:
            print(f"SCHEDULER_ERROR: 添加执行作业失败 for task '{task.id}': {e}")

def _cancel_task_jobs(task_id: str):
    """从 APScheduler 取消作业"""
    current_scheduler = get_scheduler()
    if not current_scheduler or not current_scheduler.running:
        print("SCHEDULER_ERROR: APScheduler 未运行，无法取消作业。")
        return

    reminder_job_id = f"reminder_{task_id}"
    execution_job_id = f"execution_{task_id}"

    try:
        current_scheduler.remove_job(reminder_job_id)
        print(f"SCHEDULER: 已移除提醒作业 for task '{task_id}'")
    except JobLookupError:
        print(f"SCHEDULER_INFO: 提醒作业 '{reminder_job_id}' 未找到，无需移除。")
    except Exception as e:
        print(f"SCHEDULER_ERROR: 移除提醒作业失败 for task '{task_id}': {e}")

    try:
        current_scheduler.remove_job(execution_job_id)
        print(f"SCHEDULER: 已移除执行作业 for task '{task_id}'")
    except JobLookupError:
        print(f"SCHEDULER_INFO: 执行作业 '{execution_job_id}' 未找到，无需移除。")
    except Exception as e:
        print(f"SCHEDULER_ERROR: 移除执行作业失败 for task '{task_id}': {e}")

@router.post("/", response_model=TaskModel, status_code=status.HTTP_201_CREATED)
async def create_task(task_request: CreateTaskRequest = Body(...)):
    """
    创建一个新任务。
    """
    now = datetime.utcnow()
    task_id = str(uuid.uuid4())

    reminder_time_val = None
    if task_request.due_date and task_request.reminder_offset_minutes is not None:
        reminder_time_val = task_request.due_date - timedelta(minutes=task_request.reminder_offset_minutes)

    reminder_methods_values = [
        method.value if hasattr(method, 'value') else str(method)
        for method in (task_request.reminder_methods or [ReminderMethod.NOTIFICATION])
    ]

    insert_query = TaskDB.__table__.insert().values(
        id=task_id,
        title=task_request.title,
        description=task_request.description,
        status=TaskStatus.PENDING,
        created_at=now,
        updated_at=now,
        due_date=task_request.due_date,
        reminder_time=reminder_time_val,
        reminder_offset_minutes=task_request.reminder_offset_minutes,
        reminder_methods=reminder_methods_values, # <--- 确保存入的是字符串列表
        priority=task_request.priority or 0,
        tags=task_request.tags or [],
        action_type=task_request.action_type,
        action_payload=task_request.action_payload or {}
    )

    try:
        await database.execute(insert_query)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to create task in database: {e}")

    created_task_db = await database.fetch_one(TaskDB.__table__.select().where(TaskDB.id == task_id))
    if not created_task_db:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve task after creation")

    response_task = TaskModel.model_validate(dict(created_task_db))

    response_task.reminder_methods = [
        m.value if hasattr(m, 'value') else str(m)
        for m in response_task.reminder_methods
    ]

    _schedule_task_jobs(response_task)
    print(f"TASK_MANAGER: Created task '{response_task.id}' with title '{response_task.title}' in DB")
    return response_task

@router.get("/", response_model=List[TaskModel])
async def list_tasks(
    status_filter: Optional[TaskStatus] = Query(None, alias="status"),
    priority_filter: Optional[int] = Query(None, alias="priority"),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100)
):
    """
    获取任务列表，支持过滤和分页。
    """
    query = TaskDB.__table__.select()
    if status_filter:
        query = query.where(TaskDB.status == status_filter)
    if priority_filter is not None:
        query = query.where(TaskDB.priority == priority_filter)

    query = query.order_by(TaskDB.created_at.desc()).offset(skip).limit(limit)

    db_tasks = await database.fetch_all(query)
    return [TaskModel.model_validate(dict(task)) for task in db_tasks]

@router.get("/{task_id}", response_model=TaskModel)
async def get_task(task_id: str = Path(..., description="要获取的任务ID")):
    """
    根据ID获取单个任务的详细信息。
    """
    query = TaskDB.__table__.select().where(TaskDB.id == task_id)
    db_task = await database.fetch_one(query)
    if not db_task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    return TaskModel.model_validate(dict(db_task))

@router.put("/{task_id}", response_model=TaskModel)
async def update_task(
    task_id: str = Path(..., description="要更新的任务ID"),
    task_update: UpdateTaskRequest = Body(...)
):
    """
    更新现有任务。
    """
    existing_task_query = TaskDB.__table__.select().where(TaskDB.id == task_id)
    db_task = await database.fetch_one(existing_task_query)
    if not db_task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    update_data = task_update.model_dump(exclude_unset=True)
    update_data["updated_at"] = datetime.utcnow()

    if "reminder_methods" in update_data and update_data["reminder_methods"] is not None:
        update_data["reminder_methods"] = [
            method.value if hasattr(method, 'value') else str(method)
            for method in update_data["reminder_methods"]
        ]

    current_due_date = update_data.get("due_date", cast(Optional[datetime], db_task.due_date))
    current_offset = update_data.get("reminder_offset_minutes", cast(Optional[int], db_task.reminder_offset_minutes))

    if current_due_date and current_offset is not None:
        update_data["reminder_time"] = current_due_date - timedelta(minutes=current_offset)
    elif "due_date" in update_data and current_offset is None:
         update_data["reminder_time"] = None


    update_query = TaskDB.__table__.update().where(TaskDB.id == task_id).values(**update_data)
    await database.execute(update_query)

    updated_db_task = await database.fetch_one(existing_task_query)
    if not updated_db_task:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve task after update")

    response_task = TaskModel.model_validate(dict(updated_db_task))

    response_task.reminder_methods = [
        m.value if hasattr(m, 'value') else str(m)
        for m in response_task.reminder_methods
    ]

    _cancel_task_jobs(task_id)
    _schedule_task_jobs(response_task)
    print(f"TASK_MANAGER: Updated task '{response_task.id}' in DB")
    return response_task

@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(task_id: str = Path(..., description="要删除的任务ID")):
    """
    删除一个任务。
    """
    existing_task_query = TaskDB.__table__.select().where(TaskDB.id == task_id)
    db_task = await database.fetch_one(existing_task_query)
    if not db_task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    delete_query = TaskDB.__table__.delete().where(TaskDB.id == task_id)
    await database.execute(delete_query)

    _cancel_task_jobs(task_id)
    print(f"TASK_MANAGER: Deleted task '{task_id}' from DB")
    return None

@router.post("/{task_id}/complete", response_model=TaskModel)
async def mark_task_as_complete(task_id: str = Path(..., description="要标记为完成的任务ID")):
    """
    将任务标记为已完成。
    """
    existing_task_query = TaskDB.__table__.select().where(TaskDB.id == task_id)
    db_task_row = await database.fetch_one(existing_task_query)
    if not db_task_row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")

    db_task = TaskModel.model_validate(dict(db_task_row))
    if db_task.status == TaskStatus.COMPLETED:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Task is already completed")

    update_data = {
        "status": TaskStatus.COMPLETED,
        "updated_at": datetime.utcnow()
    }
    update_query = TaskDB.__table__.update().where(TaskDB.id == task_id).values(**update_data)
    await database.execute(update_query)

    completed_db_task = await database.fetch_one(existing_task_query)
    if not completed_db_task:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve task after marking complete")

    response_task = TaskModel.model_validate(dict(completed_db_task))
    _cancel_task_jobs(task_id)
    print(f"TASK_MANAGER: Marked task '{response_task.id}' as completed in DB")
    return response_task