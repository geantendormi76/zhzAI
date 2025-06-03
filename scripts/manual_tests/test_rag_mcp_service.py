# /home/zhz/zhz_agent/scripts/manual_tests/test_rag_mcp_service.py
import httpx
import json
import asyncio
import os # <--- 添加 os 模块导入
import sys # <--- 添加 sys 模块导入

# --- 配置项目根目录到 sys.path ---
# 这使得我们可以像在项目根目录运行一样导入模块
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- 结束 sys.path 配置 ---

MCPO_BASE_URL = os.getenv("MCPO_BASE_URL", "http://localhost:8006") # 从环境变量或默认值获取
RAG_SERVICE_PATH = "zhz_rag_mcp_service/query_rag_v2"
TARGET_URL = f"{MCPO_BASE_URL}/{RAG_SERVICE_PATH}"

async def call_rag_service(query: str, top_k_vector: int = 3, top_k_kg: int = 2, top_k_bm25: int = 3, top_k_final: int = 3):
    payload = {
        "query": query,
        "top_k_vector": top_k_vector,
        "top_k_kg": top_k_kg,
        "top_k_bm25": top_k_bm25,
        "top_k_final": top_k_final
    }
    print(f"\n--- Sending request to RAG MCP Service for query: '{query}' ---")
    print(f"URL: {TARGET_URL}")
    print(f"Payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client: 
            response = await client.post(TARGET_URL, json=payload)
        
        print(f"\nResponse Status Code: {response.status_code}")
        try:
            response_data = response.json()
            print("Response JSON:")
            print(json.dumps(response_data, ensure_ascii=False, indent=2))
            
            if "final_answer" in response_data:
                print(f"\nFinal Answer: {response_data['final_answer']}")
            if "retrieved_context_docs" in response_data and isinstance(response_data["retrieved_context_docs"], list):
                print(f"Number of retrieved context docs for answer: {len(response_data['retrieved_context_docs'])}")

        except json.JSONDecodeError:
            print("Error: Response is not valid JSON.")
            print(f"Raw Response Text (first 500 chars):\n{response.text[:500]}")
        except Exception as e_resp:
            print(f"Error processing response content: {e_resp}")
            print(f"Raw Response Text (first 500 chars):\n{response.text[:500]}")

    except httpx.RequestError as e_req:
        print(f"HTTP Request Error: {e_req}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    print("--- End of RAG Service request ---")

async def main():
    print("--- Starting RAG MCP Service Test Script ---")
    
    # 确保 mcpo 和依赖服务 (local_llm_service.py) 已启动
    # 确保 .env 文件被加载 (如果脚本依赖环境变量，例如 MCPO_BASE_URL)
    from dotenv import load_dotenv
    dotenv_path_script = os.path.join(PROJECT_ROOT, '.env') # 假设 .env 在项目根目录
    if os.path.exists(dotenv_path_script):
        load_dotenv(dotenv_path=dotenv_path_script)
        print(f"Loaded .env file from: {dotenv_path_script}")
    else:
        print(f".env file not found at {dotenv_path_script}, using defaults or existing env vars.")


    test_queries = [
        "张三在哪里工作？",
        "项目Alpha的文档编写任务分配给了谁？",
        "法国的首都是哪里？"
    ]

    for query in test_queries:
        await call_rag_service(query)
        await asyncio.sleep(1) 

    print("\n--- RAG MCP Service Test Script Finished ---")

if __name__ == "__main__":
    asyncio.run(main())