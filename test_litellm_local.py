# test_litellm_local.py
import litellm
import os
import traceback
import json

# 不再依赖 LITELLM_CONFIG_PATH 来测试这个本地模型
# if "LITELLM_CONFIG_PATH" in os.environ:
#     del os.environ["LITELLM_CONFIG_PATH"] # 确保它不干扰

litellm.set_verbose = True 

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "中国的首都是哪里？请用中文回答。并简要介绍一下它。"}
]

custom_llm_model_name_in_litellm = "local/qwen3-1.7b-gguf" 
custom_llm_api_base = "http://localhost:8088/v1"        
custom_llm_api_key = "nokey"                            

print("--- Test Case: Calling local LLM via LiteLLM with direct parameters ---")
try:
    print(f"Attempting to call '{custom_llm_model_name_in_litellm}' via LiteLLM...")
    print(f"  Target API Base: {custom_llm_api_base}")
    print(f"  Using API Key: {custom_llm_api_key}")
    print(f"  Messages: {json.dumps(messages, ensure_ascii=False, indent=2)}")
    
    response = litellm.completion(
        model=custom_llm_model_name_in_litellm,
        messages=messages,
        max_tokens=1024, 
        temperature=0.7,
        custom_llm_provider="openai", 
        api_base=custom_llm_api_base,
        api_key=custom_llm_api_key 
    )
    
    print(f"\n--- Response from '{custom_llm_model_name_in_litellm}' ---")
    if response.choices and response.choices[0].message and response.choices[0].message.content is not None:
        print("\nProcessed Content (from LiteLLM response):")
        print(response.choices[0].message.content)
    else:
        print("\nNo content found in LiteLLM response choices.")
        print("Full Response (if any):")
        print(response)

    print("\nUsage Information (from LiteLLM response):")
    if response.usage:
        print(f"  Prompt Tokens: {response.usage.prompt_tokens}")
        print(f"  Completion Tokens: {response.usage.completion_tokens}")
        print(f"  Total Tokens: {response.usage.total_tokens}")
    else:
        print("  Usage information not available.")

except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"Error calling '{custom_llm_model_name_in_litellm}' via LiteLLM: {e}")
    traceback.print_exc()

print("\n--- test_litellm_local.py finished ---")