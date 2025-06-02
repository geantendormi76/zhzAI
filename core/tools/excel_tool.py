# /home/zhz/zhz_agent/core/tools/excel_tool.py

import httpx # 用于直接 HTTP 调用本地代理
import json
from typing import Type, Dict, Any, Union, List, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import  os

# --- 日志记录 ---
import logging
logger = logging.getLogger(__name__)

# 本地 Excel 代理服务的地址
# LOCAL_AGENT_BASE_URL = "http://localhost:8003" # 这个端口需要与 local_agent_app.py 中的一致
# 为了使其更灵活，从环境变量读取，并提供默认值
WINDOWS_HOST_IP = os.getenv("WINDOWS_HOST_IP_FOR_WSL", "192.168.3.11") # 确保这个IP正确
LOCAL_AGENT_PORT = os.getenv("LOCAL_AGENT_PORT", "8003")
LOCAL_AGENT_EXCEL_SQO_ENDPOINT = f"http://{WINDOWS_HOST_IP}:{LOCAL_AGENT_PORT}/excel_sqo_mcp/execute_operation"


class ExcelOperationToolInput(BaseModel):
    # Worker Agent 会迭代 Manager 生成的 SQO 列表，
    # 每次调用这个工具时，传递一个 SQO 字典，以及 file_path 和 sheet_name
    sqo_dict: Dict[str, Any] = Field(description="单个结构化查询对象 (SQO) 的JSON字典。")
    file_path: str = Field(description="目标Excel文件的绝对路径。")
    sheet_name: Union[str, int] = Field(default=0, description="目标工作表的名称或索引 (0-based)。")

class ExcelOperationTool(BaseTool):
    name: str = "excel_operation_tool"
    description: str = ("【Excel操作工具】通过结构化查询对象(SQO)对Excel文件执行复杂的数据查询、筛选、聚合等操作。"
                        "此工具接收单个SQO以及文件路径和工作表名，并调用本地代理执行。")
    args_schema: Type[BaseModel] = ExcelOperationToolInput

    def _run(
        self,
        sqo_dict: Dict[str, Any],
        file_path: str,
        sheet_name: Union[str, int] = 0,
        **kwargs: Any
    ) -> str:
        """
        同步执行方法，调用本地代理执行单个Excel SQO。
        """
        logger.info(f"ExcelOperationTool._run called with file_path: '{file_path}', sheet_name: '{sheet_name}'")
        logger.debug(f"SQO to execute: {sqo_dict}")

        if not isinstance(sqo_dict, dict):
            return "错误：传递给 ExcelOperationTool 的 sqo_dict 参数必须是一个字典。"
        if not sqo_dict.get("operation_type"):
            return "错误：SQO字典中缺少 'operation_type' 字段。"

        # 构造发送给 local_agent_app.py 的请求体
        # local_agent_app.py 的 /excel_sqo_mcp/execute_operation 端点期望的请求体是 ExecuteSQORequest 模型，
        # 它有一个名为 'sqo' 的字段，其值才是我们这里的 sqo_dict (已经包含了 operation_type, file_path, sheet_name)
        
        # 我们需要确保传递给本地代理的 SQO 中包含 file_path 和 sheet_name，
        # 因为 Manager Agent 生成的原始 SQO 列表中的字典不包含它们。
        # Worker Agent 在调用此工具前，应该已经将它们补充完整。
        # 但为了工具的健壮性，我们在这里也检查一下，并可以覆盖或添加。
        
        sqo_for_local_agent = sqo_dict.copy() # 创建副本以修改
        sqo_for_local_agent["file_path"] = file_path # 确保或覆盖 file_path
        sqo_for_local_agent["sheet_name"] = sheet_name # 确保或覆盖 sheet_name

        request_payload_to_local_agent = {"sqo": sqo_for_local_agent}

        logger.info(f"Sending request to Local Excel Agent: {LOCAL_AGENT_EXCEL_SQO_ENDPOINT}")
        logger.debug(f"Payload to Local Excel Agent: {request_payload_to_local_agent}")

        try:
            # 使用 httpx 进行同步调用
            with httpx.Client(timeout=60.0) as client:
                response = client.post(LOCAL_AGENT_EXCEL_SQO_ENDPOINT, json=request_payload_to_local_agent)
            
            logger.info(f"Local Excel Agent response status: {response.status_code}")
            response_json = response.json() # local_agent_app.py 返回的是 SQOResponse 模型

            if response.status_code == 200 and response_json.get("success"):
                result = response_json.get("result")
                logger.info(f"Excel operation successful. Result type: {type(result)}")
                # 将结果转换为字符串以便 Agent 处理
                if isinstance(result, (list, dict)):
                    try:
                        return json.dumps(result, ensure_ascii=False, indent=2)
                    except TypeError: # 处理无法JSON序列化的类型，例如某些Pandas特殊类型
                        return str(result)
                return str(result)
            else:
                error_message = response_json.get("error", "未知错误")
                error_details = response_json.get("error_details")
                full_error = f"本地Excel代理错误: {error_message}"
                if error_details:
                    full_error += f" | 详情: {str(error_details)[:200]}" # 限制详情长度
                logger.error(full_error)
                return full_error

        except httpx.RequestError as e:
            error_msg = f"请求本地Excel代理时出错: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg
        except json.JSONDecodeError as e:
            error_msg = f"解析本地Excel代理响应JSON时出错: {e}. 响应文本: {response.text[:200]}"
            logger.error(error_msg, exc_info=True)
            return error_msg
        except Exception as e:
            error_msg = f"执行Excel操作时发生未知错误: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg

# 确保 core/tools 目录也有一个 __init__.py 文件
# touch /home/zhz/zhz_agent/core/tools/__init__.py (如果不存在)