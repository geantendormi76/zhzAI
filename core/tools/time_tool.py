# /home/zhz/zhz_agent/core/tools/time_tool.py
from datetime import datetime
import pytz # 用于处理时区
from typing import Type, Optional, Any
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import logging

logger = logging.getLogger(__name__)

class GetCurrentTimeToolInput(BaseModel):
    timezone_str: Optional[str] = Field(
        default="Asia/Shanghai", 
        description="可选参数。IANA时区名称，例如 'Asia/Shanghai', 'America/New_York', 'UTC'. 默认为 'Asia/Shanghai'."
    )

class GetCurrentTimeTool(BaseTool):
    name: str = "get_current_time_tool"
    description: str = (
        "【时间工具】获取并返回当前的日期和时间。可以指定时区（例如 'Asia/Shanghai', 'America/New_York', 'UTC'），"
        "如果未指定，则默认为 'Asia/Shanghai' (中国标准时间)。"
        "当你需要知道“现在几点了”、“今天是什么日期”或在执行与时间相关的操作（如设置提醒）前获取基准时间时使用。"
    )
    args_schema: Type[BaseModel] = GetCurrentTimeToolInput

    def _run(self, timezone_str: Optional[str] = "Asia/Shanghai", **kwargs: Any) -> str:
        logger.info(f"GetCurrentTimeTool._run called with timezone_str: '{timezone_str}'")
        try:
            if not timezone_str or not timezone_str.strip():
                effective_timezone_str = "Asia/Shanghai"
                logger.info(f"Timezone was empty, defaulting to {effective_timezone_str}")
            else:
                effective_timezone_str = timezone_str

            target_tz = pytz.timezone(effective_timezone_str)
            now_in_tz = datetime.now(target_tz)
            formatted_time = now_in_tz.strftime("%Y-%m-%d %H:%M:%S %Z%z")
            logger.info(f"Current time in {effective_timezone_str}: {formatted_time}")
            return f"当前时间 ({effective_timezone_str}): {formatted_time}"
        except pytz.exceptions.UnknownTimeZoneError:
            logger.error(f"Unknown timezone provided: '{timezone_str}'")
            return f"错误：提供的时区 '{timezone_str}' 无效。请使用标准的IANA时区名称。"
        except Exception as e:
            logger.error(f"Error in GetCurrentTimeTool: {e}", exc_info=True)
            return f"获取当前时间时发生错误: {str(e)}"