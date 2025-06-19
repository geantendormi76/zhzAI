import asyncio
from zhz_rag.llm.llm_interface import generate_expansion_and_entities

async def main():
    # 测试用例1：复杂的层次化查询
    query1 = "在《2024年度报告.pdf》的第二章里，关于新产品线的财务数据表格怎么样？"
    print(f"--- Testing Query 1: {query1} ---")
    result1 = await generate_expansion_and_entities(query1)
    if result1:
        print(result1.model_dump_json(indent=2))
    else:
        print("Failed to get planning output for Query 1.")

    print("\n" + "="*50 + "\n")

    # 测试用例2：简单的无过滤查询
    query2 = "张三在哪个项目工作？"
    print(f"--- Testing Query 2: {query2} ---")
    result2 = await generate_expansion_and_entities(query2)
    if result2:
        print(result2.model_dump_json(indent=2))
    else:
        print("Failed to get planning output for Query 2.")

if __name__ == "__main__":
    asyncio.run(main())