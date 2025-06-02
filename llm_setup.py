# /home/zhz/zhz_agent/llm_setup.py

import os
from typing import Any, Optional, List, Dict

# LiteLLM 本身通常是直接调用其 API，例如 litellm.completion
# 如果要将其适配给 CrewAI 的 Agent，我们需要将其包装成一个符合 CrewAI BaseLLM 接口的类。
# CrewAI 自身也提供了通过 LiteLLM 使用各种模型的集成，我们也可以研究直接使用 CrewAI 的方式。
# 为简化起见，我们先创建一个函数，返回配置好的 LiteLLM 调用参数，或者一个简单的包装器。

# --- 从 agent_orchestrator_service.py 中获取的配置常量 ---
# 本地 LLM 服务地址 (Qwen3)
LOCAL_QWEN_API_BASE = os.getenv("LOCAL_LLM_API_BASE", "http://localhost:8088/v1") # 确保与 local_llm_service.py 的端口一致
LOCAL_QWEN_MODEL_NAME_FOR_LITELLM = os.getenv("LOCAL_LLM_MODEL_NAME", "local/qwen3-1.7b-gguf") # 这个是给 litellm 的 model 参数，需要与 LiteLLM 配置或调用方式对应

# 云端 LiteLLM 网关地址 (用于 Gemini 等)
CLOUD_LITELLM_GW_API_BASE = os.getenv("CLOUD_LITELLM_GW_API_BASE", "YOUR_CLOUD_LITELLM_GATEWAY_URL_HERE/v1") # 假设您的网关也提供 /v1 路径
GEMINI_MODEL_NAME_FOR_LITELLM = os.getenv("CLOUD_LLM_MODEL_NAME", "gemini/gemini-1.5-flash-latest") # 或您希望通过网关调用的模型

# API Keys (通常 LiteLLM 会在其自己的配置或环境变量中处理，但这里可以作为参考)
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # 如果直接调用 Gemini 而非通过网关
# 注意：本地模型的 API Key 通常是 "nokey" 或在服务中不强制

# --- CrewAI LLM 包装器 (如果需要直接传递给 CrewAI Agent) ---
# CrewAI 期望一个 LLM 对象，我们可以创建一个简单的包装器，或者使用 CrewAI 提供的 LiteLLM 集成
# from crewai.llms.base_llm import BaseLLM as CrewAIBaseLLM
# import litellm
#
# class LiteLLMWrapperForCrewAI(CrewAIBaseLLM):
#     model: str
#     api_base: Optional[str] = None
#     api_key: Optional[str] = None
#     custom_llm_provider: Optional[str] = None # 例如 "openai" for OpenAI-compatible endpoints
#     litellm_kwargs: Dict[str, Any] = {}
#
#     def __init__(self, model: str, api_base: Optional[str] = None, api_key: Optional[str] = None, custom_llm_provider: Optional[str] = None, **kwargs):
#         super().__init__(model=model) # CrewAI BaseLLM 需要 model 参数
#         self.model = model
#         self.api_base = api_base
#         self.api_key = api_key
#         self.custom_llm_provider = custom_llm_provider
#         self.litellm_kwargs = kwargs
#
#     def call(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
#         # CrewAI 的 BaseLLM.call 方法通常期望返回一个字符串
#         # LiteLLM 的 completion 返回一个 ModelResponse 对象
#         params = {
#             "model": self.model,
#             "messages": messages,
#             "api_base": self.api_base,
#             "api_key": self.api_key,
#             "custom_llm_provider": self.custom_llm_provider,
#             **self.litellm_kwargs, # 包含 temperature, max_tokens 等
#             **kwargs # 运行时可能传递的额外参数
#         }
#         # 移除值为 None 的参数，因为 litellm.completion 不喜欢 None 的 api_key 等
#         params_cleaned = {k: v for k, v in params.items() if v is not None}
#
#         response = litellm.completion(**params_cleaned)
#
#         # 从 LiteLLM 响应中提取内容
#         content = ""
#         if response.choices and response.choices[0].message and response.choices[0].message.content:
#             content = response.choices[0].message.content
#         return content
#
#     def get_token_ids(self, text: str) -> List[int]:
#         # LiteLLM 通常不直接暴露 tokenizer，但可以尝试通过 litellm.token_counter (如果它返回ids)
#         # 或者如果使用特定模型，可以尝试加载其 tokenizer
#         # 为简单起见，我们先返回空列表或引发 NotImplementedError
#         # return litellm.encode(model=self.model, text=text) # 如果 litellm.encode 可用
#         raise NotImplementedError("get_token_ids is not implemented for this LiteLLM wrapper.")


def get_local_qwen_llm_instance(
    temperature: float = 0.7,
    max_tokens: int = 2048,
    # crewai_wrapper: bool = False # 是否返回 CrewAI 兼容的包装器
) -> Any: # 返回 Any 以便后续决定是直接返回配置字典还是包装器实例
    """
    获取配置好的本地 Qwen3 LLM 实例（通过 LiteLLM 调用）。
    """
    print(f"LLM Setup: Configuring Local Qwen LLM via LiteLLM.")
    print(f"  Model: {LOCAL_QWEN_MODEL_NAME_FOR_LITELLM}")
    print(f"  API Base: {LOCAL_QWEN_API_BASE}")
    # if crewai_wrapper:
    #     return LiteLLMWrapperForCrewAI(
    #         model=LOCAL_QWEN_MODEL_NAME_FOR_LITELLM,
    #         api_base=LOCAL_QWEN_API_BASE,
    #         api_key="nokey", # 本地服务通常不需要 key
    #         custom_llm_provider="openai", # 因为我们的本地服务是 OpenAI 兼容的
    #         temperature=temperature,
    #         max_tokens=max_tokens
    #     )
    # else:
    # 返回一个配置字典，调用方可以使用 litellm.completion(**config, messages=...)
    return {
        "model": LOCAL_QWEN_MODEL_NAME_FOR_LITELLM,
        "api_base": LOCAL_QWEN_API_BASE,
        "api_key": "nokey",
        "custom_llm_provider": "openai",
        "temperature": temperature,
        "max_tokens": max_tokens
    }


def get_cloud_gemini_llm_instance(
    temperature: float = 0.5,
    max_tokens: int = 4096,
    # crewai_wrapper: bool = False
) -> Any:
    """
    获取配置好的云端 Gemini LLM 实例（通过云端 LiteLLM 网关调用）。
    """
    print(f"LLM Setup: Configuring Cloud Gemini LLM via LiteLLM Gateway.")
    print(f"  Model: {GEMINI_MODEL_NAME_FOR_LITELLM}")
    print(f"  API Base (Gateway): {CLOUD_LITELLM_GW_API_BASE}")

    if CLOUD_LITELLM_GW_API_BASE == "YOUR_CLOUD_LITELLM_GATEWAY_URL_HERE/v1":
        print("LLM Setup WARNING: CLOUD_LITELLM_GW_API_BASE is not configured. Cloud LLM calls will likely fail.")
        # 可以选择返回 None 或者一个无效的配置，让调用方处理
        return None

    # if crewai_wrapper:
    #     return LiteLLMWrapperForCrewAI(
    #         model=GEMINI_MODEL_NAME_FOR_LITELLM,
    #         api_base=CLOUD_LITELLM_GW_API_BASE,
    #         # API Key 通常由云端 LiteLLM 网关管理，客户端调用网关时可能不需要直接提供
    #         # 或者网关本身可能需要某种形式的认证 key
    #         api_key=os.getenv("CLOUD_LITELLM_GATEWAY_API_KEY"), # 假设网关可能需要一个 key
    #         # custom_llm_provider 可能不需要，因为网关会处理到具体云服务的转换
    #         temperature=temperature,
    #         max_tokens=max_tokens
    #     )
    # else:
    return {
        "model": GEMINI_MODEL_NAME_FOR_LITELLM,
        "api_base": CLOUD_LITELLM_GW_API_BASE,
        "api_key": os.getenv("CLOUD_LITELLM_GATEWAY_API_KEY"), # 网关本身的key
        "temperature": temperature,
        "max_tokens": max_tokens
    }

# --- 示例用法 (可选，用于测试此文件) ---
async def main_test_llm_setup():
    print("--- Testing LLM Setup ---")

    print("\n--- Getting Local Qwen Config ---")
    local_qwen_config = get_local_qwen_llm_instance()
    if local_qwen_config:
        print(f"Local Qwen Config: {local_qwen_config}")
        # 模拟调用
        try:
            messages = [{"role": "user", "content": "你好！"}]
            print(f"Simulating LiteLLM call with local Qwen config for: {messages}")
            # response = await litellm.acompletion(**local_qwen_config, messages=messages)
            # print(f"Simulated local Qwen response (first choice): {response.choices[0].message.content if response.choices else 'No response'}")
            print("Actual LiteLLM call commented out for setup test.")
        except Exception as e:
            print(f"Error simulating local Qwen call: {e}")
    else:
        print("Failed to get local Qwen config.")

    print("\n--- Getting Cloud Gemini Config ---")
    cloud_gemini_config = get_cloud_gemini_llm_instance()
    if cloud_gemini_config:
        print(f"Cloud Gemini Config: {cloud_gemini_config}")
        if cloud_gemini_config.get("api_base") == "YOUR_CLOUD_LITELLM_GATEWAY_URL_HERE/v1":
            print("Skipping simulated cloud Gemini call as gateway URL is a placeholder.")
        else:
            try:
                messages = [{"role": "user", "content": "Hello!"}]
                print(f"Simulating LiteLLM call with cloud Gemini config for: {messages}")
                # response = await litellm.acompletion(**cloud_gemini_config, messages=messages)
                # print(f"Simulated cloud Gemini response (first choice): {response.choices[0].message.content if response.choices else 'No response'}")
                print("Actual LiteLLM call commented out for setup test.")
            except Exception as e:
                print(f"Error simulating cloud Gemini call: {e}")
    else:
        print("Failed to get cloud Gemini config (likely due to placeholder URL).")

if __name__ == "__main__":
    # 为了运行异步的 main_test_llm_setup
    # import asyncio
    # asyncio.run(main_test_llm_setup())
    pass