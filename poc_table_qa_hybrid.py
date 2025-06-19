import os
import json
import pandas as pd
import io
from llama_cpp import Llama, LlamaGrammar
from dotenv import load_dotenv
import logging

# --- 1. Configure logging and environment variables ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger("POC_TableQA_Hybrid")
load_dotenv()

# --- 2. Load model path ---
MODEL_PATH = os.getenv("LOCAL_LLM_GGUF_MODEL_PATH")

if not MODEL_PATH or not os.path.exists(MODEL_PATH) or "Qwen3-1.7B" not in MODEL_PATH:
    logger.error("="*50)
    logger.error("错误：请确保您的 .env 文件中的 LOCAL_LLM_GGUF_MODEL_PATH")
    logger.error(f"正确指向了 Qwen3-1.7B 模型。当前路径: {MODEL_PATH}")
    logger.error("="*50)
    exit(1)

# --- 3. Define the Prompt for "Instruction Generation" ---
# This Prompt is very direct, telling the model its only task is to "fill in the blanks".
TABLE_QA_INSTRUCTION_PROMPT_TEMPLATE = """
# 指令
你是一个专门从用户问题中提取表格查询指令的AI。你的唯一任务是分析【用户问题】和【表格列名】，然后输出一个包含`row_identifier`和`column_identifier`的JSON对象。

## 表格信息
【表格列名】: {column_names}

## 用户问题
【用户问题】: "{user_query}"

## 输出要求
请严格按照以下格式输出一个JSON对象，不要包含任何其他文字或解释。
```json
{{
  "row_identifier": "string, 用户问题中提到的具体行名",
  "column_identifier": "string, 用户问题中提到的具体列名"
}}
```
你的JSON输出:
"""

# --- 4. 采用主项目中最终验证通过的、最健壮的通用JSON GBNF Schema ---
INSTRUCTION_JSON_GBNF_SCHEMA = r'''
root   ::= object
value  ::= object | array | string | number | "true" | "false" | "null"
ws ::= ([ \t\n\r])*
object ::= "{" ws ( member ("," ws member)* )? ws "}"
member ::= string ws ":" ws value
array  ::= "[" ws ( value ("," ws value)* )? ws "]"
string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\""
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
'''

def run_table_qa_poc():
    """Execute the "Instruction Generation + Code Execution" PoC"""
    logger.info("--- 开始执行“指令生成+代码执行”PoC ---")

    # --- 5. Simulate input data ---
    user_query = "产品B的价格是多少？"
    table_markdown_context = """
产品|价格|库存
---|---|---
产品A|3500|120
产品B|4800|80
产品C|5200|95"""

    logger.info(f"用户问题: {user_query}")
    logger.info(f"表格上下文:\n{table_markdown_context}")

    # --- 6. Initialize LLM and GBNF ---
    try:
        grammar = LlamaGrammar.from_string(INSTRUCTION_JSON_GBNF_SCHEMA)
        llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=2048, verbose=False)
        logger.info("Llama模型和GBNF语法已成功加载。")
    except Exception as e:
        logger.error(f"初始化Llama或GBNF时出错: {e}", exc_info=True)
        return

    # --- 7. Convert Markdown table to Pandas DataFrame ---
    try:
        # Use io.StringIO to simulate a file from the string, allowing pandas to read it.
        # skipinitialspace=True is used to handle spaces between '|' symbols and column names.
        df = pd.read_csv(io.StringIO(table_markdown_context), sep='|', skipinitialspace=True).dropna(axis=1, how='all').iloc[1:]
        # Clean up spaces in column names.
        df.columns = [col.strip() for col in df.columns]
        # Set the first column as the index for easy lookup by row name.
        df = df.set_index(df.columns[0].strip())
        logger.info("成功将Markdown表格转换为Pandas DataFrame:\n" + str(df))
    except Exception as e:
        logger.error(f"转换Markdown到DataFrame时出错: {e}", exc_info=True)
        return

    # --- 8. Execute LLM call to generate instructions ---
    column_names_for_prompt = ", ".join(df.columns.tolist())
    prompt = TABLE_QA_INSTRUCTION_PROMPT_TEMPLATE.format(
        column_names=column_names_for_prompt,
        user_query=user_query
    )

    logger.info("--- 正在调用LLM生成查询指令... ---")
    try:
        response = llm.create_completion(
            prompt=prompt,
            grammar=grammar,
            max_tokens=256,
            temperature=0.0 # For instruction extraction, no randomness is needed.
        )
        instruction_str = response['choices'][0]['text']
        logger.info(f"LLM成功返回指令字符串: {instruction_str}")
    except Exception as e:
        logger.error(f"调用LLM生成指令时出错: {e}", exc_info=True)
        return

    # --- 9. Execute code lookup ---
    logger.info("--- 正在执行代码进行表格查找... ---")
    try:
        instruction_json = json.loads(instruction_str)
        row_id = instruction_json.get("row_identifier")
        col_id = instruction_json.get("column_identifier")

        if not row_id or not col_id:
            logger.error("LLM生成的指令中缺少'row_identifier'或'column_identifier'。")
            return

        # Lookup in DataFrame
        # .at[] is the fastest way to access a single value.
        answer = df.at[row_id, col_id]

        logger.info("="*50)
        logger.info(f"🎉 PoC成功！")
        logger.info(f"   - LLM生成的指令: {instruction_json}")
        logger.info(f"   - Python代码查找到的最终答案: {answer}")
        logger.info("="*50)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.error("="*50)
        logger.error(f"PoC失败：处理LLM指令或查找DataFrame时出错: {e}")
        logger.error(f"LLM原始输出: {instruction_str}")
        logger.error("="*50)

if __name__ == "__main__":
    run_table_qa_poc()
