# zhz_agent/task_jobs.py
from datetime import datetime
from typing import Dict, Any
import os
import traceback
import httpx # <--- 确保 httpx 已导入
import json # <--- 确保 json 已导入

# 从 .database 导入 database 对象以便查询任务详情
# 从 .pydantic_models 导入 TaskModel 以便类型转换
# 从 .main 导入 scheduler 以便在需要时重新调度（虽然通常作业函数不直接操作调度器）
# 更好的做法是通过参数传递必要的信息，而不是依赖全局导入
WINDOWS_HOST_IP = os.getenv("WINDOWS_HOST_IP_FOR_WSL", "192.168.3.11") # <--- 请务必替换为您真实的Windows IP
LOCAL_AGENT_PORT = os.getenv("LOCAL_AGENT_PORT", "8003") # 与 local_agent_app.py 中的端口一致

# 如果 WINDOWS_HOST_IP 仍然是占位符，给出提示
if WINDOWS_HOST_IP == "在此处填写您上一步找到的Windows主机IP":
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("REMINDER_JOB WARNING: WINDOWS_HOST_IP 未正确设置在 task_jobs.py 中!")
    print("请编辑 task_jobs.py 文件，将 '在此处填写您上一步找到的Windows主机IP' 替换为实际IP地址。")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

LOCAL_AGENT_NOTIFY_URL = f"http://{WINDOWS_HOST_IP}:{LOCAL_AGENT_PORT}/notify"

async def send_task_reminder(task_id: str, task_title: str, reminder_methods: list):
    """
    实际发送任务提醒的函数。
    """
    print(f"REMINDER_JOB: [{datetime.utcnow()}] 正在为任务 '{task_id}' - '{task_title}' 发送提醒。")
    for method in reminder_methods:
        if method == "notification": # 假设 ReminderMethod.NOTIFICATION.value 是 "notification"
            print(f"  REMINDER_JOB: 尝试通过 Local Agent 发送桌面通知: '{task_title}'")
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        LOCAL_AGENT_NOTIFY_URL,
                        json={"title": f"任务提醒: {task_title}", "message": f"任务 '{task_title}' 即将到期或需要关注。"}
                    )
                    response.raise_for_status() # Raise an exception for bad status codes
                    print(f"  REMINDER_JOB: 本地代理通知请求发送成功. 状态: {response.status_code}")
            except httpx.RequestError as e:
                print(f"  REMINDER_JOB: 通过本地代理发送通知失败 (RequestError): {e}")
                traceback.print_exc()
            except Exception as e:
                print(f"  REMINDER_JOB: 通过本地代理发送通知失败 (General Error): {e}")
                traceback.print_exc()
        # elif method == "email": #
        #     print(f"  REMINDER_JOB: 模拟发送邮件提醒...")

async def execute_task_action(task_id: str, action_type: str, action_payload: Dict[str, Any]):
    """
    实际执行任务动作的函数。
    """
    print(f"EXECUTION_JOB: [{datetime.utcnow()}] 正在为任务 '{task_id}' 执行动作 '{action_type}'。")
    print(f"  EXECUTION_JOB: 动作参数: {action_payload}")

    final_result = f"动作 '{action_type}' 已模拟执行。"
    success = True

    if action_type == "navigate":
        destination = action_payload.get("destination")
        if destination:
            print(f"  EXECUTION_JOB: 模拟导航到 '{destination}'...")
            final_result = f"已模拟为导航到 '{destination}' 准备好路线。"
        else:
            final_result = "导航动作失败：缺少目的地。"
            success = False
    elif action_type == "log_event":
        event_details = action_payload.get("event_details", "无详情")
        print(f"  EXECUTION_JOB: 记录事件: '{event_details}'")
        final_result = f"事件 '{event_details}' 已记录。"
    else:
        final_result = f"未知的动作类型: {action_type}"
        success = False

    # 更新数据库中的任务状态和结果 (需要访问数据库)
    # 这部分逻辑最好通过API调用或服务层来完成，以避免循环导入和分散DB操作
    # 这里我们只打印信息，实际应用中需要实现DB更新
    print(f"  EXECUTION_JOB: 任务 '{task_id}' 执行完毕。结果: {final_result}, 状态: {'COMPLETED' if success else 'FAILED'}")
