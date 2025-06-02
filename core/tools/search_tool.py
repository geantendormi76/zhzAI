# /home/zhz/zhz_agent/core/tools/search_tool.py

import asyncio
from typing import Type, Optional, Dict, Any, List
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import json
from utils.common_utils import call_mcpo_tool

# --- 日志记录 ---
import logging
logger = logging.getLogger(__name__)


class WebSearchToolInput(BaseModel):
    query: str = Field(description="要进行网络搜索的关键词或问题。")
    max_results: Optional[int] = Field(default=5, description="希望返回的最大搜索结果数量。")

class WebSearchTool(BaseTool):
    name: str = "web_search_tool"
    description: str = ("【网络搜索工具】使用DuckDuckGo搜索引擎在互联网上查找与用户查询相关的信息。"
                        "返回搜索结果列表，每个结果包含标题、链接和摘要。")
    args_schema: Type[BaseModel] = WebSearchToolInput
    mcp_service_name: str = "ddgsearch" # 与 mcpo_servers.json 中定义的服务名一致 (如果使用)
    mcp_tool_path: str = "search"      # DuckDuckGo 搜索服务提供的端点

    def _run(
        self,
        query: str,
        max_results: Optional[int] = 5,
        **kwargs: Any
    ) -> str:
        """
        同步执行方法，调用 MCP 的 ddgsearch 服务。
        """
        logger.info(f"WebSearchTool._run called with query: '{query}', max_results: {max_results}")
        payload = {
            "query": query,
            "max_results": max_results,
        }
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, self._acall_mcp(payload))
                    result = future.result(timeout=60) 
            else:
                result = asyncio.run(self._acall_mcp(payload))
        except Exception as e:
            logger.error(f"Error running WebSearchTool for query '{query}': {e}", exc_info=True)
            return f"Error executing Web Search tool: {str(e)}"
        
        return self._handle_mcp_result(result)

    async def _acall_mcp(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        实际调用 MCP 服务的异步方法。
        """
        tool_path_on_mcp = f"{self.mcp_service_name}/{self.mcp_tool_path}"
        logger.info(f"WebSearchTool: Calling MCP endpoint '{tool_path_on_mcp}' with payload: {payload}")
        mcp_response = await call_mcpo_tool(tool_path_on_mcp, payload)
        return mcp_response

    def _handle_mcp_result(self, mcp_response: Dict[str, Any]) -> str:
        """
        处理来自 MCP ddgsearch 服务的响应。
        ddgsearch 服务成功时应该返回一个包含搜索结果列表的 JSON。
        """
        if mcp_response.get("error"):
            error_msg = f"Web Search tool failed: {mcp_response['error']}"
            logger.error(error_msg)
            return error_msg

        # ddgsearch MCP 服务返回的原始结果可能是一个字典，其中 "results" 键包含列表
        # 或者直接是一个列表（取决于MCP服务的封装）
        # 我们需要查看 ddgsearch 服务的实际输出格式来确定如何解析
        # 假设它返回一个包含 "results" 键的字典，其值为列表，每个列表项是一个包含 "title", "href", "body" 的字典
        
        search_results = mcp_response.get("results") # 假设 MCP 服务返回的 JSON 中有一个 'results' 键
        
        if isinstance(search_results, list):
            if not search_results:
                return "网络搜索没有找到相关结果。"
            
            formatted_results = ["网络搜索结果："]
            for i, res in enumerate(search_results):
                if isinstance(res, dict):
                    title = res.get("title", "无标题")
                    link = res.get("href", "#")
                    snippet = res.get("body", "无摘要")
                    formatted_results.append(f"{i+1}. {title}\n   链接: {link}\n   摘要: {snippet[:150]}...\n")
                else: # 如果列表项不是字典，直接转字符串
                    formatted_results.append(f"{i+1}. {str(res)[:200]}...")

            return "\n".join(formatted_results)
        elif search_results is not None: # 如果 results 存在但不是列表
             return f"网络搜索返回了意外格式的数据: {str(search_results)[:300]}"
        else: # 如果 mcp_response 中没有 "results" 键，或者 "results" 为 None
            logger.warning(f"Web Search tool: 'results' key not found or is None in MCP response. Raw response: {mcp_response}")
            return f"网络搜索未能获取结果。原始响应: {json.dumps(mcp_response, ensure_ascii=False, indent=2)}"

# 确保 core/tools 目录也有一个 __init__.py 文件
# touch /home/zhz/zhz_agent/core/tools/__init__.py (如果不存在)