# /home/zhz/zhz_agent/core/tools/excel_tool.py

import httpx
import json
from typing import Type, Dict, Any, Union, List, Optional # 确保导入了 List
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import os
import logging
import traceback # 确保导入 traceback

logger = logging.getLogger(__name__)

# 从环境变量读取 Windows Host IP 和 Local Agent 端口
WINDOWS_HOST_IP = os.getenv("WINDOWS_HOST_IP_FOR_WSL", "192.168.3.11") 
LOCAL_AGENT_PORT = os.getenv("LOCAL_AGENT_PORT", "8003")
LOCAL_AGENT_EXCEL_SQO_ENDPOINT = f"http://{WINDOWS_HOST_IP}:{LOCAL_AGENT_PORT}/excel_sqo_mcp/execute_operation"


class ExcelOperationToolInput(BaseModel):
    # Worker Agent 会迭代 Manager 生成的 SQO 列表，
    # 每次调用这个工具时，传递一个 SQO 字典，以及 file_path 和 sheet_name。
    # sqo_dict 已经是 Manager 生成的 SQO 操作定义，不包含 file_path 和 sheet_name。
    # file_path 和 sheet_name 由 Worker Agent 从用户原始请求中提取并传入。
    
    sqo_operation_definition: Dict[str, Any] = Field(description="单个SQO操作定义字典，不包含file_path和sheet_name。")
    target_excel_file_path: str = Field(description="目标Excel文件的绝对路径。")
    target_excel_sheet_name: Union[str, int] = Field(default=0, description="目标工作表的名称或索引 (0-based)。")

class ExcelOperationTool(BaseTool):
    name: str = "excel_operation_tool"
    description: str = ("【Excel操作工具】通过结构化查询对象(SQO)对Excel文件执行复杂的数据查询、筛选、聚合等操作。"
                        "此工具接收单个SQO操作定义、文件路径和工作表名，并调用在Windows上运行的本地代理来执行实际操作。")
    args_schema: Type[BaseModel] = ExcelOperationToolInput

    def _run(
        self,
        sqo_operation_definition: Dict[str, Any],
        target_excel_file_path: str,
        target_excel_sheet_name: Union[str, int] = 0,
        **kwargs: Any
    ) -> str:
        logger.info(f"ExcelOperationTool._run called. File: '{target_excel_file_path}', Sheet: '{target_excel_sheet_name}'")
        logger.debug(f"SQO Operation Definition received: {sqo_operation_definition}")

        if not isinstance(sqo_operation_definition, dict):
            return "错误：传递给 ExcelOperationTool 的 sqo_operation_definition 参数必须是一个字典。"
        if not sqo_operation_definition.get("operation_type"):
            return "错误：SQO操作定义字典中缺少 'operation_type' 字段。"

        # 构造发送给 local_agent_app.py 的完整SQO，补充 file_path 和 sheet_name
        full_sqo_for_local_agent = sqo_operation_definition.copy()
        full_sqo_for_local_agent["file_path"] = target_excel_file_path
        full_sqo_for_local_agent["sheet_name"] = target_excel_sheet_name
        
        # local_agent_app.py 的 /excel_sqo_mcp/execute_operation 端点期望的请求体是 ExecuteSQORequest 模型，
        # 它有一个名为 'sqo' 的字段，其值才是我们这里的 full_sqo_for_local_agent
        request_payload_to_local_agent = {"sqo": full_sqo_for_local_agent}

        logger.info(f"Sending request to Local Excel Agent: {LOCAL_AGENT_EXCEL_SQO_ENDPOINT}")
        logger.debug(f"Payload to Local Excel Agent: {json.dumps(request_payload_to_local_agent, ensure_ascii=False)}")

        try:
            with httpx.Client(timeout=120.0) as client: # 增加超时
                response = client.post(LOCAL_AGENT_EXCEL_SQO_ENDPOINT, json=request_payload_to_local_agent)
            
            logger.info(f"Local Excel Agent response status: {response.status_code}")
            response_json = response.json() # local_agent_app.py 返回的是 SQOResponse 模型

            if response.status_code == 200 and response_json.get("success"):
                result = response_json.get("result")
                logger.info(f"Excel operation successful. Result type: {type(result)}")
                if isinstance(result, (list, dict)):
                    try:
                        # 对于列表或字典，以JSON字符串形式返回给Agent通常更易于处理
                        return json.dumps(result, ensure_ascii=False, indent=2)
                    except TypeError: # 处理无法JSON序列化的类型
                        return str(result)
                return str(result) # 其他类型直接转字符串
            else:
                error_message = response_json.get("error", "未知错误")
                error_details = response_json.get("error_details")
                full_error = f"本地Excel代理错误: {error_message}"
                if error_details:
                    full_error += f" | 详情: {str(error_details)[:300]}" # 限制详情长度
                logger.error(full_error + f" | Full SQO sent: {json.dumps(full_sqo_for_local_agent, ensure_ascii=False)}")
                return full_error

        except httpx.TimeoutException as e:
            error_msg = f"调用本地Excel代理超时: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg
        except httpx.ConnectError as e:
            error_msg = f"无法连接到本地Excel代理 ({LOCAL_AGENT_EXCEL_SQO_ENDPOINT}): {e}. 请确保Windows端的local_agent_app.py正在运行，并且WSL可以访问到Windows的IP和端口。"
            logger.error(error_msg, exc_info=True)
            return error_msg
        except httpx.RequestError as e:
            error_msg = f"请求本地Excel代理时发生网络错误: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg
        except json.JSONDecodeError as e:
            error_msg = f"解析本地Excel代理响应JSON时出错: {e}. 响应文本: {response.text[:300]}"
            logger.error(error_msg, exc_info=True)
            return error_msg
        except Exception as e:
            error_msg = f"执行Excel操作时发生未知错误: {e}"
            logger.error(error_msg, exc_info=True)
            return f"{error_msg} | Traceback: {traceback.format_exc()[:500]}"