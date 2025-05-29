import httpx
import json
import asyncio

# SGLang 服务器的地址
SGLANG_SERVER_URL = "http://127.0.0.1:30000/generate"

# 我们要提取人名和组织名的示例文本
sample_text = "张三是阿里巴巴的一名工程师，李四在腾讯工作，王五加入了百度。"

# 定义期望输出的JSON Schema
# (确保required字段中的名称与properties中的键名一致)
json_schema_definition = {
    "type": "object",
    "properties": {
        "persons": {
            "type": "array",
            "items": {"type": "string"},
            "description": "从文本中提取出的人名列表。"
        },
        "organizations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "从文本中提取出的组织名列表。"
        }
    },
    "required": ["persons", "organizations"]
}

# 构建Prompt，明确指示LLM按照JSON Schema输出
# 注意：这里我们直接将JSON Schema的描述作为提示的一部分，有时这能帮助LLM更好地理解期望格式。
# 另一种方式是在SGLang的API参数中直接传递json_schema对象，并简化prompt。
# 我们先尝试这种明确在prompt中描述schema的方式，如果效果不好，再调整。
prompt_template = f"""从以下文本中提取所有的人名和组织名。
请严格按照以下JSON格式进行输出，不要包含任何额外的解释或Markdown标记：
{{
  "persons": ["人名1", "人名2", ...],
  "organizations": ["组织名1", "组织名2", ...]
}}

其中：
- "persons" 字段是一个包含所有识别人名的字符串数组。
- "organizations" 字段是一个包含所有识别组织名的字符串数组。

文本：
"{sample_text}"

JSON输出：
"""

# SGLang API 请求体
payload = {
    "text": prompt_template,
    "sampling_params": {
        "temperature": 0.1,       # 较低的温度，鼓励更确定的输出
        "max_new_tokens": 256,    # 足够容纳提取结果
        "stop": ["<|im_end|>"], # 根据您的Qwen模型调整停止标记
        # "stop_token_ids": [151645], # Qwen2.5-3B-Instruct的 <|im_end|> token id，可选
        "json_schema": json.dumps(json_schema_definition) # <--- 关键：传递JSON Schema
    },
    # "stream": False # 我们这里不需要流式输出
}

async def call_sglang_with_schema():
    print(f"--- 发送请求到 SGLang ({SGLANG_SERVER_URL}) ---")
    print(f"Prompt (部分):\n{prompt_template[:300]}...\n")
    print(f"JSON Schema:\n{json.dumps(json_schema_definition, indent=2, ensure_ascii=False)}\n")
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(SGLANG_SERVER_URL, json=payload)
            response.raise_for_status() # 如果HTTP状态码是4xx或5xx，则抛出异常

            print(f"--- SGLang 响应状态码: {response.status_code} ---")
            
            response_json = response.json()
            print(f"--- SGLang 完整响应 (JSON): ---")
            print(json.dumps(response_json, indent=2, ensure_ascii=False))
            
            generated_text = response_json.get("text", "").strip()
            
            print("\n--- LLM 生成的原始文本: ---")
            print(generated_text)
            
            # 尝试解析生成的文本为JSON
            try:
                parsed_output = json.loads(generated_text)
                print("\n--- 解析后的JSON对象: ---")
                print(json.dumps(parsed_output, indent=2, ensure_ascii=False))
                
                # 可以在这里添加对parsed_output是否符合schema的进一步验证（如果需要）
                # 例如，检查persons和organizations字段是否存在且为列表
                if isinstance(parsed_output, dict) and \
                   "persons" in parsed_output and isinstance(parsed_output["persons"], list) and \
                   "organizations" in parsed_output and isinstance(parsed_output["organizations"], list):
                    print("\n--- 验证通过：输出的JSON结构基本符合预期！ ---")
                else:
                    print("\n--- 验证失败：输出的JSON结构不完全符合预期。 ---")
                    
            except json.JSONDecodeError as e:
                print(f"\n--- 错误：LLM生成的文本不是有效的JSON！ ---")
                print(f"错误信息: {e}")
            except Exception as e:
                print(f"\n--- 解析或验证JSON时发生未知错误: {e} ---")

    except httpx.HTTPStatusError as e:
        print(f"\n--- HTTP错误: SGLang服务返回状态码 {e.response.status_code} ---")
        print(f"响应内容: {e.response.text}")
    except httpx.RequestError as e:
        print(f"\n--- 请求错误: 无法连接到SGLang服务 ---")
        print(f"错误信息: {e}")
    except Exception as e:
        print(f"\n--- 发生未知错误: {e} ---")

if __name__ == "__main__":
    asyncio.run(call_sglang_with_schema())