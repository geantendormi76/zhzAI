# /home/zhz/zhz_agent/core/llms/custom_litellm_wrapper.py

import os
import json
import traceback
from typing import List, Dict, Any, Optional, Union, Sequence

import litellm
from crewai.llms.base_llm import BaseLLM as CrewAIBaseLLM
from crewai.tools import BaseTool # 用于类型提示

# --- 日志记录 ---
import logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # 可以根据需要调整日志级别
# if not logger.hasHandlers():
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)

class CustomLiteLLMWrapper(CrewAIBaseLLM):
    """
    一个通用的 CrewAI LLM 包装器，通过 LiteLLM 调用各种 LLM 服务。
    能够处理本地 OpenAI 兼容的端点和通过 LiteLLM 支持的云端模型。
    """
    model_name: str # LiteLLM 使用的模型名称，例如 "gemini/gemini-1.5-flash-latest" 或 "local/qwen3-1.7b-gguf"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    custom_llm_provider: Optional[str] = None # 例如 "openai" 用于本地 OpenAI 兼容服务
    
    # LiteLLM 支持的额外参数，例如 temperature, max_tokens, top_p 等
    # 这些参数可以在实例化时传入，或者在调用 call 方法时覆盖
    litellm_params: Dict[str, Any] = {}

    # CrewAI 工具相关的参数
    tool_config: Optional[Dict[str, Any]] = None # 用于控制工具调用的模式，例如 Gemini 的 function_calling_config
    _cached_tools_for_litellm: Optional[List[Dict[str, Any]]] = None # 缓存转换后的工具定义

    def __init__(
        self,
        model: str, # CrewAI BaseLLM 需要 model 参数
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_llm_provider: Optional[str] = None,
        temperature: float = 0.7, # 默认温度
        max_tokens: Optional[int] = 2048, # 默认最大 token 数
        top_p: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
        tool_config: Optional[Dict[str, Any]] = None, # 例如 {"function_calling_config": {"mode": "AUTO"}}
        agent_tools: Optional[List[BaseTool]] = None, # CrewAI Agent 的工具列表，用于转换为 LiteLLM 格式
        **kwargs: Any # 其他传递给 LiteLLM 的参数
    ):
        super().__init__(model=model) # 调用父类的构造函数

        # --- 新增日志 ---
        logger.info(f"CustomLiteLLMWrapper __init__ for '{model}': Received agent_tools type: {type(agent_tools)}")
        if agent_tools is not None:
            logger.info(f"CustomLiteLLMWrapper __init__ for '{model}': agent_tools content (names): {[tool.name for tool in agent_tools if hasattr(tool, 'name')]}")
        else:
            logger.info(f"CustomLiteLLMWrapper __init__ for '{model}': agent_tools is None.")
        # --- 结束新增日志 ---
        
        self.model_name = model
        self.api_base = api_base
        self.api_key = api_key
        self.custom_llm_provider = custom_llm_provider
        
        self.litellm_params = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stop": stop_sequences,
            **kwargs # 合并其他未知参数
        }
        # 清理 litellm_params 中的 None 值
        self.litellm_params = {k: v for k, v in self.litellm_params.items() if v is not None}

        self.tool_config = tool_config
        if agent_tools:
            self._cached_tools_for_litellm = self._convert_crewai_tools_to_litellm_format(agent_tools)
            logger.info(f"CustomLiteLLMWrapper for '{self.model_name}': Cached {len(self._cached_tools_for_litellm)} tools.")
        else:
            logger.info(f"CustomLiteLLMWrapper for '{self.model_name}': No agent_tools provided for caching.")

    def _remove_unwanted_fields_from_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        递归移除 Pydantic schema 中可能导致某些 LLM (如 Gemini) 报错的字段，
        例如顶层的 'title' 和属性定义中的 'default'。
        也确保 'object' 类型的 schema 包含 'properties' 键。
        """
        if not isinstance(schema, dict):
            return schema

        schema.pop('title', None) # 移除顶层 title

        if "properties" in schema and isinstance(schema["properties"], dict):
            if "type" not in schema: # 确保 object 类型有 type 字段
                schema["type"] = "object"
            for prop_name, prop_def in list(schema["properties"].items()): # 使用 list 进行迭代以允许修改
                if isinstance(prop_def, dict):
                    prop_def.pop('default', None) # 移除属性的 default
                    prop_def.pop('title', None)   # 移除属性的 title
                    self._remove_unwanted_fields_from_schema(prop_def) # 递归处理嵌套 schema
        elif schema.get("type") == "object" and "properties" not in schema:
            # 如果是 object 类型但没有 properties，某些 LLM (如 Gemini) 会报错
            schema["properties"] = {}

        # 移除顶层的 default (如果存在且不应该存在于顶层)
        # 通常 default 应该在属性级别，但以防万一
        # schema.pop('default', None) # 这个可能过于激进，先注释掉

        # 递归处理其他嵌套字典
        for key, value in schema.items():
            if isinstance(value, dict):
                self._remove_unwanted_fields_from_schema(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        value[i] = self._remove_unwanted_fields_from_schema(item)
        return schema
        
    def _convert_crewai_tools_to_litellm_format(self, tools: Optional[List[BaseTool]]) -> Optional[List[Dict[str, Any]]]:
        """
        将 CrewAI 的 BaseTool 列表转换为 LiteLLM期望的 "tools" 格式。
        LiteLLM 的格式通常与 OpenAI 的 function calling 格式类似。
        """
        if not tools:
            return None
        
        litellm_tool_definitions = []
        for tool_instance in tools:
            tool_name = tool_instance.name
            tool_description = tool_instance.description
            
            parameters_schema: Dict[str, Any]
            if not hasattr(tool_instance, 'args_schema') or not tool_instance.args_schema:
                # 如果工具没有定义参数 schema，则默认为一个没有参数的 object 类型
                parameters_schema = {"type": "object", "properties": {}}
            else:
                try:
                    # Pydantic V2 使用 model_json_schema(), V1 使用 schema()
                    if hasattr(tool_instance.args_schema, 'model_json_schema'):
                        pydantic_schema = tool_instance.args_schema.model_json_schema()
                    else:
                        pydantic_schema = tool_instance.args_schema.schema() # type: ignore
                    
                    # 清理 schema，移除 'title' 和 'default' 等字段
                    cleaned_schema = self._remove_unwanted_fields_from_schema(pydantic_schema.copy())
                    parameters_schema = cleaned_schema
                except Exception as e:
                    logger.error(f"Error processing schema for tool {tool_name}: {e}. Defaulting to empty params.")
                    parameters_schema = {"type": "object", "properties": {}}
            
            litellm_tool_definitions.append({
                "type": "function", # LiteLLM/OpenAI 的标准类型
                "function": {
                    "name": tool_name,
                    "description": tool_description,
                    "parameters": parameters_schema
                }
            })
        logger.debug(f"Converted CrewAI tools to LiteLLM format: {json.dumps(litellm_tool_definitions, indent=2)}")
        return litellm_tool_definitions

    def call(self, messages: List[Dict[str, str]], tools: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> str:
        """
        使用 LiteLLM 调用 LLM。
        CrewAI 的 BaseLLM.call 方法期望返回一个字符串。
        """
        logger.info(f"CustomLiteLLMWrapper.call for '{self.model_name}' invoked.")
        logger.debug(f"  Messages: {json.dumps(messages, indent=2, ensure_ascii=False)}")
        logger.debug(f"  Tools provided to call: {'Yes' if tools else 'No'}")
        logger.debug(f"  kwargs: {kwargs}")

        # 合并参数，调用时传入的 kwargs 优先级更高
        current_litellm_params = {**self.litellm_params, **kwargs}

        litellm_call_args: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "custom_llm_provider": self.custom_llm_provider,
            **current_litellm_params # temperature, max_tokens, etc.
        }

        # 处理工具（函数调用）
        # CrewAI 在调用 Agent 的 LLM 时，会根据 Agent 的 tools 属性自动准备 tools 参数
        # 我们需要确保这里的 tools 参数与 LiteLLM 兼容
        final_tools_for_litellm = tools # 直接使用 CrewAI 传递过来的 tools
        if not final_tools_for_litellm and self._cached_tools_for_litellm:
            logger.info("  No tools passed to call, using cached tools for LiteLLM.")
            final_tools_for_litellm = self._cached_tools_for_litellm
        
        if final_tools_for_litellm:
            litellm_call_args["tools"] = final_tools_for_litellm
            # 根据 self.tool_config 设置 tool_choice
            if self.tool_config and "function_calling_config" in self.tool_config:
                fc_config = self.tool_config["function_calling_config"]
                mode = fc_config.get("mode", "AUTO").upper() # "AUTO", "ANY", "NONE"
                
                if mode == "ANY" and fc_config.get("allowed_function_names"):
                    # 对于 Gemini，如果 mode 是 ANY 且指定了函数名，tool_choice 应该是一个特定结构
                    # 对于 OpenAI，tool_choice 可以是 {"type": "function", "function": {"name": "my_function"}}
                    # LiteLLM 会尝试适配，我们先按 OpenAI 的方式设置
                    litellm_call_args["tool_choice"] = {
                        "type": "function", 
                        "function": {"name": fc_config["allowed_function_names"][0]} # 假设只取第一个
                    }
                elif mode in ["AUTO", "ANY", "NONE"]: # ANY 在没有指定函数名时，行为类似 AUTO
                    litellm_call_args["tool_choice"] = mode.lower()
                else: # 默认为 auto
                    litellm_call_args["tool_choice"] = "auto"
                logger.debug(f"  Setting tool_choice to: {litellm_call_args['tool_choice']}")
            else: # 如果没有 tool_config，默认 tool_choice 为 auto
                 litellm_call_args["tool_choice"] = "auto"
                 logger.debug(f"  No tool_config, defaulting tool_choice to 'auto'.")


        # 移除值为 None 的参数，因为 litellm.completion 不喜欢 None 的 api_key 等
        litellm_call_args_cleaned = {k: v for k, v in litellm_call_args.items() if v is not None}

        # --- 新增：如果 api_base 为 None (表示直接调用云端模型)，则尝试使用 LITELLM_PROXY_URL ---
        if self.api_base is None:
            local_proxy_url = os.getenv("LITELLM_PROXY_URL")
            if local_proxy_url:
                litellm_call_args_cleaned["proxy"] = { # LiteLLM 的 proxy 参数期望一个字典
                    "http": local_proxy_url,
                    "https": local_proxy_url,
                }
                logger.info(f"  Using local proxy for direct cloud call: {local_proxy_url}")
            else:
                logger.info("  api_base is None, but LITELLM_PROXY_URL is not set. Proceeding without proxy.")
        # --- 结束新增代理逻辑 ---
        
        logger.info(f"  Attempting to call litellm.completion for model '{self.model_name}'...")
        # 在打印参数前确保 proxy 参数也被包含（如果设置了）
        debug_params_to_print = {k: v for k, v in litellm_call_args_cleaned.items() if k != 'messages'}
        if "proxy" in litellm_call_args_cleaned: # 确保打印时能看到 proxy
            debug_params_to_print["proxy_used"] = litellm_call_args_cleaned["proxy"]
            
        logger.debug(f"  LiteLLM Call Args (cleaned, messages excluded, proxy shown if used): {debug_params_to_print}")
        
        # --- 添加以下详细日志 ---
        logger.info("--------------------------------------------------------------------")
        logger.info(f"DEBUGGING TOOLS PASSED TO LITELLM for model {self.model_name} (Manager Agent call):") # 区分是Manager还是Worker的调用

        effective_tools_to_log = litellm_call_args_cleaned.get("tools")

        if effective_tools_to_log:
            try:
                tools_json_str = json.dumps(effective_tools_to_log, indent=2, ensure_ascii=False)
                logger.info(f"  Tools (Content):\n{tools_json_str}")
                # 专门检查 web_search_tool 是否存在
                web_search_tool_found_in_definition = any(
                    tool.get("function", {}).get("name") == "web_search_tool" 
                    for tool in effective_tools_to_log if isinstance(tool, dict)
                )
                if web_search_tool_found_in_definition:
                    logger.info("  >>>> web_search_tool IS PRESENT in the tools definition passed to LiteLLM. <<<<")
                else:
                    logger.warning("  >>>> web_search_tool IS MISSING from the tools definition passed to LiteLLM! <<<<")

            except Exception as e_log_json:
                logger.error(f"  Error serializing tools for logging: {e_log_json}")
                logger.info(f"  Tools (Raw Object, could not serialize): {effective_tools_to_log}")
        else:
            logger.info("  Tools: Not present in litellm_call_args_cleaned (effective_tools_to_log is None or empty)")
            
        if "tool_choice" in litellm_call_args_cleaned:
            logger.info(f"  Tool Choice: {litellm_call_args_cleaned['tool_choice']}")
        else:
            logger.info("  Tool Choice: Not present in litellm_call_args_cleaned")
        logger.info("--------------------------------------------------------------------")
        # --- 详细日志结束 ---

        response = None 
        try:
            response = litellm.completion(**litellm_call_args_cleaned) 
            logger.info(f"  litellm.completion call for '{self.model_name}' succeeded.") # <--- 添加日志
            logger.debug(f"  LiteLLM Raw Response object type: {type(response)}")
            if hasattr(response, 'model_dump_json'):
                logger.debug(f"  LiteLLM Raw Response (JSON): {response.model_dump_json(indent=2)}")
            else:
                logger.debug(f"  LiteLLM Raw Response (str): {str(response)[:500]}")

            # --- 新增日志，检查原始的 usage ---
            if hasattr(response, 'usage') and response.usage:
                logger.info(f"  DEBUG USAGE (from LiteLLM response object): {response.usage}")
                # 如果 response.usage 是 Pydantic 模型，可以尝试打印其字典形式
                if hasattr(response.usage, 'model_dump'):
                    logger.info(f"  DEBUG USAGE (dict): {response.usage.model_dump()}")
                else:
                    logger.info(f"  DEBUG USAGE (raw object): {response.usage}")
            else:
                logger.warning("  DEBUG USAGE: LiteLLM response object does not have .usage or it's empty.")
            # --- 结束新增 ---

        except Exception as e:
            logger.error(f"LiteLLM completion call FAILED for model '{self.model_name}': {e}", exc_info=True) # <--- 修改日志
            return f"LLM_CALL_ERROR: 调用模型 '{self.model_name}' 失败: {str(e)}"

        # 从 LiteLLM 响应中提取内容或工具调用
        # LiteLLM 的 ModelResponse 结构与 OpenAI 的 ChatCompletion 类似
        llm_message_response = response.choices[0].message
        
        if hasattr(llm_message_response, 'tool_calls') and llm_message_response.tool_calls:
            logger.info(f"  LLM returned structured tool_calls: {llm_message_response.tool_calls}")
            # 构造 ReAct 格式的字符串
            tool_call = llm_message_response.tool_calls[0] # 假设只有一个工具调用
            action = tool_call.function.name
            action_input = tool_call.function.arguments # 这是 JSON 字符串
            
            # Gemini 可能也会在 content 中生成 Thought，如果它遵循 ReAct
            thought_prefix = ""
            if llm_message_response.content and "Thought:" in llm_message_response.content:
                thought_prefix = llm_message_response.content.split("Action:")[0] # 取 Action:之前的部分作为 Thought

            react_string = f"{thought_prefix.strip()}\nAction: {action}\nAction Input: {action_input}"
            logger.info(f"  Constructed ReAct string from tool_calls: {react_string}")
            return react_string.strip() # 返回 ReAct 字符串
        
        elif llm_message_response.content:
            content_str = llm_message_response.content
            logger.info(f"  LLM returned content (first 200 chars): {content_str[:200]}")
            # 如果 content 本身就是 ReAct 格式，也直接返回
            return content_str.strip() # 返回字符串
        
        else:
            logger.warning("  LLM response did not contain structured tool_calls or text content.")
            return ""

    def get_token_ids(self, text: str) -> List[int]:
        """
        获取文本的 token ID 列表。
        LiteLLM 提供了 litellm.encode 和 litellm.decode 方法。
        """
        try:
            # 注意：litellm.encode 可能需要 model 参数来确定使用哪个 tokenizer
            return litellm.encode(model=self.model_name, text=text)
        except Exception as e:
            logger.warning(f"get_token_ids failed for model '{self.model_name}': {e}. Returning empty list.")
            # CrewAI 在某些情况下即使这里返回空列表也能继续，但最好能正确实现
            return []

    # --- 添加这个方法 ---
    @property
    def supports_function_calling(self) -> bool:
        logger.debug(f"CustomLiteLLMWrapper.supports_function_calling() called for model {self.model_name}, returning True.")
        return True
    # --- 添加结束 ---

    # CrewAI 可能还会用到的一些属性
    @property
    def _llm_type(self) -> str:
        return f"custom_litellm_{self.model_name.replace('/', '_')}"

    @property
    def identifying_params(self) -> Dict[str, Any]:
        """返回用于标识此LLM实例的参数字典。"""
        return {
            "model_name": self.model_name,
            "api_base": self.api_base,
            "custom_llm_provider": self.custom_llm_provider,
            **self.litellm_params
        }

# --- 示例用法 (可选，用于测试此文件) ---
async def main_test_wrapper():
    logger.info("--- Testing CustomLiteLLMWrapper ---")

    # 测试本地 Qwen (假设服务在 http://localhost:8088/v1)
    try:
        logger.info("\n--- Testing Local Qwen ---")
        local_qwen_llm = CustomLiteLLMWrapper(
            model="local/qwen3-1.7b-gguf", # 这个名称需要与 LiteLLM 调用时匹配
            api_base="http://localhost:8088/v1",
            api_key="nokey",
            custom_llm_provider="openai",
            temperature=0.1
        )
        messages_qwen = [{"role": "user", "content": "你好，请用中文介绍一下你自己。不要超过50个字。"}]
        # CrewAI 通常是同步调用 call 方法，但我们的 call 内部是同步执行 litellm.completion
        # 如果要测试异步行为，需要 litellm.acompletion 和异步的 call
        response_qwen = local_qwen_llm.call(messages=messages_qwen)
        logger.info(f"Local Qwen Response: {response_qwen}")
        
        # 测试 token_ids
        # token_ids_qwen = local_qwen_llm.get_token_ids("你好，世界")
        # logger.info(f"Token IDs for '你好，世界' from Qwen (via LiteLLM encode): {token_ids_qwen}")

    except Exception as e:
        logger.error(f"Error testing local Qwen: {e}", exc_info=True)

    # 测试云端 Gemini (假设通过配置好的 LiteLLM 网关)
    # 需要设置 CLOUD_LITELLM_GW_API_BASE 环境变量
    # 例如: export CLOUD_LITELLM_GW_API_BASE="http://your-litellm-proxy.com/v1"
    #       export GEMINI_API_KEY="your_actual_gemini_key_if_proxy_doesnt_handle_it_or_proxy_key"
    # CLOUD_LITELLM_GW_API_BASE_TEST = os.getenv("CLOUD_LITELLM_GW_API_BASE_TEST")
    # GEMINI_API_KEY_TEST = os.getenv("GEMINI_API_KEY_TEST") # 或者网关的key

    # if CLOUD_LITELLM_GW_API_BASE_TEST and GEMINI_API_KEY_TEST:
    #     try:
    #         print("\n--- Testing Cloud Gemini via Gateway ---")
    #         gemini_llm = CustomLiteLLMWrapper(
    #             model="gemini/gemini-1.5-flash-latest",
    #             api_base=CLOUD_LITELLM_GW_API_BASE_TEST,
    #             api_key=GEMINI_API_KEY_TEST, # Key for the gateway or Gemini if gateway passes it
    #             temperature=0.5
    #         )
    #         messages_gemini = [{"role": "user", "content": "What is the capital of France?"}]
    #         response_gemini = gemini_llm.call(messages=messages_gemini)
    #         print(f"Cloud Gemini Response: {response_gemini}")
    #     except Exception as e:
    #         print(f"Error testing cloud Gemini: {e}")
    # else:
    #     print("\nSkipping Cloud Gemini test as CLOUD_LITELLM_GW_API_BASE_TEST or GEMINI_API_KEY_TEST is not set.")

if __name__ == "__main__":
    # import asyncio
    # asyncio.run(main_test_wrapper())
    pass