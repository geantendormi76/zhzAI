#/home/zhz/zhz_agent/custom_llm.py
import os
import json
import httpx
import asyncio
import traceback
from typing import List, Dict, Any, Optional, Union, Sequence, Type 
# --- CrewAI & LiteLLM Imports ---
from crewai.tools import BaseTool
from crewai.llms.base_llm import BaseLLM as CrewAIBaseLLM
import litellm

# --- [修改] Local Imports -> 改为绝对导入 ---
from zhz_rag.llm.llm_interface import call_sglang_llm # For SGLang LLM
from dotenv import load_dotenv

load_dotenv()

# --- SGLang Config ---
SGLANG_API_URL_FOR_LLM = os.getenv("SGLANG_API_URL", "http://localhost:30000/generate")

# --- CustomGeminiLLM (from ceshi/run_agent.py with fixes) ---
class CustomGeminiLLM(CrewAIBaseLLM):
    model_name: str
    api_key: str
    max_tokens: Optional[int] = 2048
    tool_config: Optional[Dict[str, Any]] = None
    stop: Optional[List[str]] = None
    _gemini_tools_cache: Optional[List[Dict[str, Any]]] = None

    def __init__(self, model: str, api_key: str, temperature: float = 0.1, max_tokens: Optional[int] = 2048, tool_config: Optional[Dict[str, Any]] = None, stop: Optional[List[str]] = None, agent_tools: Optional[List[BaseTool]] = None, **kwargs):
        super().__init__(model=model, temperature=temperature)
        self.model_name = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.tool_config = tool_config or {"function_calling_config": {"mode": "AUTO"}}
        self.stop = stop
        if agent_tools:
            self._gemini_tools_cache = self._convert_crewai_tools_to_gemini_format(agent_tools)
            print(f"CustomGeminiLLM __init__: Cached {len(self._gemini_tools_cache)} tools.")
        else:
            print("CustomGeminiLLM __init__: No agent_tools provided for caching.")

    def _remove_unwanted_fields(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(schema, dict):
            return schema

        schema.pop('title', None)

        if "properties" in schema:
            if "type" not in schema:
                schema["type"] = "object"
            for prop_name, prop_def in list(schema["properties"].items()):
                if isinstance(prop_def, dict):
                    prop_def.pop('default', None)
                    prop_def.pop('title', None)
                    self._remove_unwanted_fields(prop_def)
        elif schema.get("type") == "object" and "properties" not in schema:
            schema["properties"] = {}

        keys_to_delete = [k for k, v in schema.items() if k == 'default']
        for k in keys_to_delete:
            del schema[k]

        for k, v in schema.items():
            if isinstance(v, dict):
                self._remove_unwanted_fields(v)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        v[i] = self._remove_unwanted_fields(item)
        return schema

    def _convert_crewai_tools_to_gemini_format(self, tools: Optional[List[BaseTool]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        gemini_tool_declarations = []
        for tool_instance in tools:
            tool_name = tool_instance.name
            tool_description = tool_instance.description
            if not hasattr(tool_instance, 'args_schema') or not tool_instance.args_schema:
                parameters_schema = {"type": "object", "properties": {}}
            else:
                try:
                    if hasattr(tool_instance.args_schema, 'model_json_schema'):
                        pydantic_schema = tool_instance.args_schema.model_json_schema()
                    else:
                        pydantic_schema = tool_instance.args_schema.schema()
                    cleaned_schema = self._remove_unwanted_fields(pydantic_schema.copy())
                    parameters_schema = cleaned_schema
                except Exception as e:
                    print(f"Error processing schema for tool {tool_name}: {e}")
                    parameters_schema = {"type": "object", "properties": {}}
            gemini_tool_declarations.append({
                "name": tool_name,
                "description": tool_description,
                "parameters": parameters_schema
            })
        final_tools_for_litellm = []
        for declaration in gemini_tool_declarations:
            final_tools_for_litellm.append({
                "type": "function",
                "function": declaration
            })
        return final_tools_for_litellm

    def call(self, messages: Union[str, List[Dict[str, str]]], tools: Optional[List[dict]] = None, callbacks: Optional[List[Any]] = None, **kwargs: Any) -> Union[str, Any]:
        print(f"CustomGeminiLLM CALL method invoked.")
        print(f"  CALL - Tools received by CustomLLM.call: {'Yes' if tools else 'No'}")
        print(f"  CALL - Callbacks received by CustomLLM.call: {'Yes' if callbacks else 'No'}")

        if isinstance(messages, str):
            processed_messages = [{"role": "user", "content": messages}]
        else:
            processed_messages = messages

        litellm_params = {
            "model": self.model_name,
            "messages": processed_messages,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": self.stop
        }

        # --- Proxy Addition ---
        proxy_url = os.getenv("LITELLM_PROXY_URL")
        if proxy_url:
            litellm_params["proxy"] = {
                "http": proxy_url,
                "https": proxy_url,
            }
            print(f"CustomGeminiLLM.call - Using proxy: {proxy_url}")
        else:
            print("CustomGeminiLLM.call - No proxy configured (LITELLM_PROXY_URL not set).")

        # --- Tool Handling (tools: null fix) ---
        final_tools_for_litellm = None
        received_tools_to_process = tools
        if not received_tools_to_process and self._gemini_tools_cache:
            print("  CALL - INFO: Tools argument was None, using cached tools.")
            received_tools_to_process = self._gemini_tools_cache

        if received_tools_to_process:
            cleaned_tools_for_litellm = []
            for tool_dict in received_tools_to_process:
                current_tool_def = tool_dict.copy()
                if current_tool_def.get("type") == "function" and "function" in current_tool_def:
                    func_def = current_tool_def["function"].copy()
                    if "parameters" in func_def:
                        func_def["parameters"] = self._remove_unwanted_fields(func_def["parameters"].copy())
                    current_tool_def["function"] = func_def
                    cleaned_tools_for_litellm.append(current_tool_def)
                else:
                    cleaned_tools_for_litellm.append(tool_dict)
            final_tools_for_litellm = cleaned_tools_for_litellm

        if final_tools_for_litellm:
            litellm_params["tools"] = final_tools_for_litellm
            fc_config = self.tool_config.get("function_calling_config", {})
            mode = fc_config.get("mode", "AUTO").upper()
            allowed_names = fc_config.get("allowed_function_names")

            if mode == "ANY" and allowed_names:
                litellm_params["tool_choice"] = {
                    "type": "function",
                    "function": {"name": allowed_names[0]}
                }
            elif mode in ["AUTO", "ANY", "NONE"]:
                litellm_params["tool_choice"] = mode.lower()
            else:
                litellm_params["tool_choice"] = "auto"
            print(f"CustomGeminiLLM DEBUG: Setting tool_choice to: {litellm_params['tool_choice']}")

        if callbacks:
            litellm_params["callbacks"] = callbacks

        try:
            print(f"CustomGeminiLLM.call - LiteLLM PARAMS (Preview): model={litellm_params['model']}, msgs_count={len(litellm_params['messages'])}, tools={'Yes' if 'tools' in litellm_params else 'No'}, tool_choice={litellm_params.get('tool_choice')}, proxy={'Yes' if 'proxy' in litellm_params else 'No'}")
            response = litellm.completion(**litellm_params)
        except Exception as e:
            print(f"CRITICAL ERROR: LiteLLM completion call failed: {e}")
            if callbacks:
                for handler in callbacks:
                    if hasattr(handler, 'on_llm_error'):
                        try:
                            handler.on_llm_error(error=e, llm=self, **kwargs)
                        except Exception as cb_err:
                            print(f"Error in callback on_llm_error: {cb_err}")
            raise

        llm_message_response = response.choices[0].message
        if hasattr(llm_message_response, 'tool_calls') and llm_message_response.tool_calls:
            print(f"CustomGeminiLLM.call - Detected tool_calls: {llm_message_response.tool_calls}")
            # --- ReAct Format Workaround (AttributeError fix) ---
            tool_call = llm_message_response.tool_calls[0]
            action = tool_call.function.name
            action_input = tool_call.function.arguments
            react_string = f"Action: {action}\nAction Input: {action_input}"
            print(f"CustomGeminiLLM.call - Returning ReAct string: {react_string}")
            return react_string
        else:
            print(f"CustomGeminiLLM.call - Returning text content.")
            return llm_message_response.content or ""

    def get_token_counter_instance(self):
        class GeminiTokenCounter:
            def __init__(self, model_name):
                self.model_name = model_name

            def count_tokens(self, text: Union[str, List[Dict[str,str]]]) -> int:
                try:
                    if isinstance(text, list):
                        return litellm.token_counter(model=self.model_name, messages=text)
                    return litellm.token_counter(model=self.model_name, text=str(text))
                except Exception as e:
                    print(f"Warning: Token counting failed ({e}), falling back to rough estimate.")
                    if isinstance(text, list):
                        return sum(len(str(m.get("content","")).split()) for m in text)
                    return len(str(text).split())
        return GeminiTokenCounter(model_name=self.model_name)


# --- CustomSGLangLLM (from hybrid_rag/custom_llm.py) ---
class CustomSGLangLLM(CrewAIBaseLLM):
    endpoint_url: str = SGLANG_API_URL_FOR_LLM
    model_name: str = "qwen2-3b-instruct"
    temperature: float = 0.1
    max_new_tokens_val: int = 1024

    def __init__(self, endpoint: Optional[str] = None, model: Optional[str] = None, temperature: Optional[float] = None, max_new_tokens: Optional[int] = None, **kwargs: Any):
        super().__init__(**kwargs)
        if endpoint: self.endpoint_url = endpoint
        if model: self.model_name = model
        if temperature is not None: self.temperature = temperature
        if max_new_tokens is not None: self.max_new_tokens_val = max_new_tokens
        print(f"CustomSGLangLLM initialized. Endpoint: {self.endpoint_url}, Model: {self.model_name}, Temp: {self.temperature}, MaxTokens: {self.max_new_tokens_val}")

    def _prepare_sglang_prompt(self, messages: Sequence[Dict[str, str]]) -> str:
        prompt_str = ""
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role and content:
                prompt_str += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        prompt_str += "<|im_start|>assistant\n"
        return prompt_str

    def call(self, messages: Sequence[Dict[str, str]], **kwargs: Any) -> str:
        print(f"CustomSGLangLLM.call received messages: {messages}")
        sglang_prompt = self._prepare_sglang_prompt(messages)
        print(f"CustomSGLangLLM.call prepared sglang_prompt (first 200 chars): {sglang_prompt[:200]}...")
        stop_sequences_for_sglang = kwargs.get("stop", ["<|im_end|>", "<|endoftext|>"])

        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            async def async_runner():
                return await call_sglang_llm(
                    prompt=sglang_prompt,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens_val,
                    stop_sequences=stop_sequences_for_sglang
                )

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, async_runner())
                    response_text = future.result(timeout=120)
            else:
                response_text = asyncio.run(async_runner())

        except Exception as e:
            print(f"CustomSGLangLLM.call: Error during SGLang call: {type(e).__name__} - {e}")
            traceback.print_exc()
            return f"LLM_CALL_ERROR: 调用SGLang服务失败 - {str(e)}"

        if response_text is None:
            print("CustomSGLangLLM.call: SGLang returned None.")
            return "LLM_CALL_ERROR: SGLang服务未返回任何文本。"

        print(f"CustomSGLangLLM.call: SGLang returned text (first 200 chars): {response_text[:200]}...")
        return response_text

    def get_token_ids(self, text: str) -> List[int]:
        print("CustomSGLangLLM.get_token_ids: Not implemented, returning empty list.")
        return []

    @property
    def support_function_calling(self) -> bool:
        return False

    @property
    def support_stop_words(self) -> bool:
        return True

    @property
    def available_models(self) -> List[str]:
        return [self.model_name]

    @property
    def context_window(self) -> int:
        return 32768

    @property
    def identifying_params(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "endpoint_url": self.endpoint_url,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens_val,
        }