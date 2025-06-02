# /home/zhz/zhz_agent/core/llm_manager.py

import os
from typing import Any, Optional

# 从相对路径导入包装器
from .llms.custom_litellm_wrapper import CustomLiteLLMWrapper

# --- 配置常量 (可以从项目级配置文件或环境变量加载) ---
# 本地 LLM 服务地址 (Qwen3)
LOCAL_QWEN_API_BASE = os.getenv("LOCAL_LLM_API_BASE", "http://localhost:8088/v1")
LOCAL_QWEN_MODEL_NAME_FOR_LITELLM = os.getenv("LOCAL_LLM_MODEL_NAME_FOR_LITELLM", "local/qwen3-1.7b-gguf") # 与 LiteLLM 调用匹配的名称

# 云端 LiteLLM 网关地址 (用于 Gemini 等)
CLOUD_LITELLM_GW_API_BASE = os.getenv("CLOUD_LITELLM_GW_API_BASE", "YOUR_CLOUD_LITELLM_GATEWAY_URL_HERE/v1")
GEMINI_MODEL_NAME_FOR_LITELLM = os.getenv("CLOUD_LLM_MODEL_NAME_FOR_LITELLM", "gemini/gemini-1.5-flash-latest")
CLOUD_LITELLM_GATEWAY_API_KEY = os.getenv("CLOUD_LITELLM_GATEWAY_API_KEY") # 网关本身可能需要的 API Key

# 默认 LLM 参数
DEFAULT_TEMPERATURE_LOCAL = 0.7
DEFAULT_MAX_TOKENS_LOCAL = 2048

DEFAULT_TEMPERATURE_CLOUD = 0.5
DEFAULT_MAX_TOKENS_CLOUD = 4096


def get_local_qwen_llm(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    agent_tools: Optional[list] = None, # CrewAI Agent 的工具列表
    tool_config: Optional[dict] = None  # Gemini 风格的 tool_config
) -> CustomLiteLLMWrapper:
    """
    获取配置好的本地 Qwen3 LLM 实例 (CrewAI 兼容)。
    """
    temp = temperature if temperature is not None else DEFAULT_TEMPERATURE_LOCAL
    mt = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS_LOCAL

    print(f"LLM Manager: Creating Local Qwen LLM instance.")
    print(f"  Model: {LOCAL_QWEN_MODEL_NAME_FOR_LITELLM}, API Base: {LOCAL_QWEN_API_BASE}")
    print(f"  Temp: {temp}, Max Tokens: {mt}")
    
    return CustomLiteLLMWrapper(
        model=LOCAL_QWEN_MODEL_NAME_FOR_LITELLM,
        api_base=LOCAL_QWEN_API_BASE,
        api_key="nokey", # 本地服务通常不需要 key
        custom_llm_provider="openai", # 因为我们的本地服务是 OpenAI 兼容的
        temperature=temp,
        max_tokens=mt,
        agent_tools=agent_tools,
        tool_config=tool_config
    )

def get_cloud_gemini_llm(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    agent_tools: Optional[list] = None, # CrewAI Agent 的工具列表
    tool_config: Optional[dict] = None  # Gemini 风格的 tool_config
) -> Optional[CustomLiteLLMWrapper]:
    """
    获取配置好的云端 Gemini LLM 实例 (CrewAI 兼容)，通过 LiteLLM 网关调用。
    如果网关未配置，则返回 None。
    """
    temp = temperature if temperature is not None else DEFAULT_TEMPERATURE_CLOUD
    mt = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS_CLOUD

    print(f"LLM Manager: Creating Cloud Gemini LLM instance via Gateway.")
    print(f"  Model: {GEMINI_MODEL_NAME_FOR_LITELLM}, Gateway API Base: {CLOUD_LITELLM_GW_API_BASE}")
    print(f"  Temp: {temp}, Max Tokens: {mt}")

    if CLOUD_LITELLM_GW_API_BASE == "YOUR_CLOUD_LITELLM_GATEWAY_URL_HERE/v1":
        print("LLM Manager WARNING: CLOUD_LITELLM_GW_API_BASE is not configured. Cannot create cloud LLM instance.")
        return None
    
    return CustomLiteLLMWrapper(
        model=GEMINI_MODEL_NAME_FOR_LITELLM,
        api_base=CLOUD_LITELLM_GW_API_BASE,
        api_key=CLOUD_LITELLM_GATEWAY_API_KEY, # 网关本身可能需要的 key
        # custom_llm_provider 在通过网关调用时通常不需要，除非网关本身是 OpenAI 兼容的代理
        temperature=temp,
        max_tokens=mt,
        agent_tools=agent_tools,
        tool_config=tool_config
    )

# --- (可选) 一个选择 LLM 的辅助函数 ---
def get_llm_instance(
    llm_type: str = "local_qwen", # "local_qwen" 或 "cloud_gemini"
    **kwargs # 其他传递给具体 LLM 创建函数的参数
) -> Optional[CustomLiteLLMWrapper]:
    if llm_type == "local_qwen":
        return get_local_qwen_llm(**kwargs)
    elif llm_type == "cloud_gemini":
        return get_cloud_gemini_llm(**kwargs)
    else:
        print(f"LLM Manager ERROR: Unknown LLM type '{llm_type}'. Returning None.")
        return None

if __name__ == "__main__":
    # 测试
    print("--- Testing LLM Manager ---")
    
    print("\n--- Getting Local Qwen LLM ---")
    local_llm = get_llm_instance("local_qwen", temperature=0.2)
    if local_llm:
        print(f"Local LLM instance: {local_llm.model_name}, Temp: {local_llm.litellm_params.get('temperature')}")
        # 可以在这里尝试调用，但需要运行 local_llm_service
        # try:
        #     response = local_llm.call(messages=[{"role":"user", "content":"你好"}])
        #     print(f"Test call to local_llm: {response}")
        # except Exception as e:
        #     print(f"Error calling local_llm: {e}")
    
    print("\n--- Getting Cloud Gemini LLM ---")
    # 需要设置环境变量 CLOUD_LITELLM_GW_API_BASE 和 CLOUD_LITELLM_GATEWAY_API_KEY
    # os.environ["CLOUD_LITELLM_GW_API_BASE"] = "http://your-gateway-url/v1" 
    # os.environ["CLOUD_LITELLM_GATEWAY_API_KEY"] = "your_gateway_key"
    cloud_llm = get_llm_instance("cloud_gemini", max_tokens=100)
    if cloud_llm:
        print(f"Cloud LLM instance: {cloud_llm.model_name}, Max Tokens: {cloud_llm.litellm_params.get('max_tokens')}")
    else:
        print("Cloud LLM instance could not be created (check gateway config).")