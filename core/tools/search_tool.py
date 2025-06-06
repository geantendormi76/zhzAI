# /home/zhz/zhz_agent/core/tools/search_tool.py

import asyncio
from typing import Type, Optional, Dict, Any, List
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
import json
from utils.common_utils import call_mcpo_tool # 确保从正确的相对路径导入
import traceback

# --- 日志记录 ---
import logging
logger = logging.getLogger(__name__)


class WebSearchToolInput(BaseModel):
    query: str = Field(description="要进行网络搜索的关键词或问题。")
    max_results: Optional[int] = Field(default=5, description="希望返回的最大搜索结果数量。")

class WebSearchTool(BaseTool):
    name: str = "web_search_tool" # Agent将使用这个名字
    description: str = ("【网络搜索工具】使用DuckDuckGo搜索引擎在互联网上查找与用户查询相关的信息。"
                        "返回搜索结果列表，每个结果包含标题、链接和摘要。")
    args_schema: Type[BaseModel] = WebSearchToolInput
    mcp_service_name: str = "ddgsearch" # 与 mcpo_servers.json 中定义的服务名一致
    mcp_tool_path: str = "search"      # DuckDuckGo 搜索服务提供的端点

    async def _acall_mcp(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        实际调用 MCP 服务的异步方法。
        """
        tool_path_on_mcp = f"{self.mcp_service_name}/{self.mcp_tool_path}"
        logger.info(f"WebSearchTool: Calling MCP endpoint '{tool_path_on_mcp}' with payload: {payload}")
        # call_mcpo_tool 期望返回一个包含 success 和 data/error 的字典
        mcp_wrapper_response = await call_mcpo_tool(tool_path_on_mcp, payload)
        
        # 直接返回 call_mcpo_tool 的结果，让 _handle_mcp_result 处理
        return mcp_wrapper_response

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
        
        mcp_response_data: Dict[str, Any] = {
            "success": False, # 默认失败
            "error": "Initialization error in _run before async call for WebSearchTool",
            "error_type": "RUN_INIT_ERROR_WEBSEARCH"
        }

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, self._acall_mcp(payload))
                    mcp_response_data = future.result(timeout=60) 
            else:
                mcp_response_data = asyncio.run(self._acall_mcp(payload))
        except Exception as e:
            logger.error(f"Error running WebSearchTool for query '{query}': {e}", exc_info=True)
            mcp_response_data = {
                "success": False,
                "error": f"Exception in WebSearchTool _run: {str(e)}",
                "error_type": "RUN_EXCEPTION_WEBSEARCH",
                "traceback": traceback.format_exc()
            }
        
        return self._handle_mcp_result(mcp_response_data)


    def _handle_mcp_result(self, mcp_response: Dict[str, Any]) -> str:
        logger.info(f"WebSearchTool._handle_mcp_result received raw MCP wrapper response: {str(mcp_response)[:1000]}...")

        if not isinstance(mcp_response, dict):
            return f"TOOL_ERROR: {self.name} received invalid response format from MCP call wrapper (expected dict, got {type(mcp_response)})."

        if mcp_response.get("success") is False:
            error_msg = mcp_response.get("error", "Unknown error from ddgsearch MCP call via wrapper.")
            error_type = mcp_response.get("error_type", "MCP_CALL_FAILED")
            logger.error(f"WebSearchTool failed via MCP wrapper. Type: {error_type}, Error: {error_msg}")
            return f"TOOL_ERROR: {self.name} failed: {error_msg}"

        ddg_service_data = mcp_response.get("data")
        
        if isinstance(ddg_service_data, dict) and "content" in ddg_service_data:
            content_list = ddg_service_data.get("content")
            if isinstance(content_list, list) and content_list:
                first_content_item = content_list[0]
                if isinstance(first_content_item, dict) and first_content_item.get("type") == "text":
                    search_results_text = first_content_item.get("text")
                    if search_results_text and search_results_text.strip():
                        logger.info("WebSearchTool: Successfully extracted search results text from MCP 'content' field.")
                        # --- 修改：直接返回提取到的文本 ---
                        return search_results_text.strip() 
                        # --- 修改结束 ---
                    else:
                        logger.warning("WebSearchTool: Extracted search results text is empty.")
                        return "网络搜索没有找到相关结果（返回内容为空）。" # 如果文本为空，也返回这个
        
        # 如果上面的分支没有成功返回（例如 ddg_service_data 结构不对），则尝试旧的 "results" 逻辑
        # if isinstance(ddg_service_data, dict) and "results" in ddg_service_data:
        #     search_results = ddg_service_data.get("results")
        #     if isinstance(search_results, list):
        #         if not search_results:
        #             return "网络搜索没有找到相关结果。"
        #         formatted_results = ["网络搜索结果："]
        #         for i, res in enumerate(search_results[:5]): 
        #             if isinstance(res, dict):
        #                 title = res.get("title", "无标题")
        #                 link = res.get("href", "#")
        #                 snippet = res.get("body", "无摘要")
        #                 formatted_results.append(f"{i+1}. {title}\n   链接: {link}\n   摘要: {snippet[:150]}...\n")
        #             else:
        #                 formatted_results.append(f"{i+1}. {str(res)[:200]}...")
        #         return "\n".join(formatted_results)

        # 如果所有尝试都失败了
        logger.warning(f"WebSearchTool: Could not extract 'results' list or 'content' text as expected from ddgsearch service data. Raw service data: {str(ddg_service_data)[:500]}")
        return f"网络搜索未能获取到预期的结果格式。服务原始响应 (data part): {json.dumps(ddg_service_data, ensure_ascii=False, indent=2) if isinstance(ddg_service_data, dict) else str(ddg_service_data)[:300]}"