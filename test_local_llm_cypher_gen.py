# test_local_llm_cypher_gen.py
import httpx
import json
import os

# 从 zhz_rag.config.constants 导入 NEW_KG_SCHEMA_DESCRIPTION
# 这需要确保 PYTHONPATH 正确设置，或者脚本与 zhz_agent 在同一父目录下然后使用相对导入
# 为了简单，我们直接复制 Schema 描述到这里进行测试，或者确保能导入
try:
    from zhz_rag.config.constants import NEW_KG_SCHEMA_DESCRIPTION
except ImportError:
    print("WARNING: Could not import NEW_KG_SCHEMA_DESCRIPTION. Using a placeholder.")
    NEW_KG_SCHEMA_DESCRIPTION = "Node: :ExtractedEntity(text, label), Rel: :WorksAt, :AssignedTo"


LOCAL_LLM_URL = "http://localhost:8088/v1/chat/completions"

def construct_cypher_gen_messages(user_question: str, schema: str) -> list:
    # 使用您在 sglang_wrapper.py 中为 Cypher 生成构造的 messages 格式
    # 或者更直接地，使用 Qwen3 的标准聊天模板
    # 例如，一个简化的版本：
    # 注意：这里的 system prompt 应该与您 NEW_KG_SCHEMA_DESCRIPTION 的意图一致
    # 并且包含我们最新优化的指令
    system_prompt = NEW_KG_SCHEMA_DESCRIPTION # 使用我们优化后的完整 Schema 和指令

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"用户问题: {user_question}"}
    ]

async def call_llm_for_cypher(question: str):
    print(f"\n--- Testing Cypher Generation for: {question} ---")
    messages = construct_cypher_gen_messages(question, NEW_KG_SCHEMA_DESCRIPTION)
    
    payload = {
        "model": "qwen3-1.7b-gguf", # 与 local_llm_service.py 中 ChatCompletionRequest.model 一致
        "messages": messages,
        "temperature": 0.0, # 对于代码生成，低temperature
        "max_tokens": 1024,   # 限制输出长度，避免因长度导致的其他问题
        "stop": ["<|im_end|>", "无法生成Cypher查询."] 
        # 注意：如果GBNF生效，stop可能不是主要控制因素了
    }
    
    # 打印将要发送的 messages 的 token 估算（如果可以的话）
    # from llama_cpp import Llama
    # temp_tokenizer_model = Llama(model_path="/path/to/your/qwen3-1.7b.gguf", verbose=False) # 需要实际模型路径
    # prompt_str_for_token_count = ""
    # for msg in messages:
    #     prompt_str_for_token_count += f"{msg['role']}\n{msg['content']}\n"
    # token_count = len(temp_tokenizer_model.tokenize(prompt_str_for_token_count.encode('utf-8')))
    # print(f"Estimated prompt tokens (approx): {token_count}")
    # del temp_tokenizer_model

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(LOCAL_LLM_URL, json=payload)
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                response_data = response.json()
                print("LLM Response JSON:")
                print(json.dumps(response_data, indent=2, ensure_ascii=False))
                if response_data.get("choices") and response_data["choices"][0].get("message"):
                    content = response_data["choices"][0]["message"].get("content")
                    print(f"Extracted Content:\n---\n{content}\n---")
            else:
                print(f"Error Response Text: {response.text}")
        except Exception as e:
            print(f"An error occurred: {e}")

async def main():
    # 确保 local_llm_service.py 正在运行
    test_questions = [
        "张三在哪里工作？",
        "项目Alpha的文档编写任务分配给了谁？",
        "法国的首都是哪里？" # 测试 "无法生成"
    ]
    for q in test_questions:
        await call_llm_for_cypher(q)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())