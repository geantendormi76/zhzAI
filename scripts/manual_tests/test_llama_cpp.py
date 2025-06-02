from llama_cpp import Llama
import os

MODEL_DIR = "/home/zhz/models/Qwen3-1.7B-GGUF" # 您存放模型的目录
# 列出该目录下所有的 .gguf 文件，让用户选择或自动选择一个
gguf_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".gguf")]

if not gguf_files:
    print(f"No GGUF models found in {MODEL_DIR}")
    exit()

print("Available GGUF models:")
for i, fname in enumerate(gguf_files):
    print(f"{i+1}. {fname}")

# 为了测试，我们直接使用您下载的 Qwen 模型文件名
# 您可以根据实际文件名修改，或者实现一个选择逻辑
# 假设您的模型文件名中包含 "Qwen1.7B-GGUF" 或类似标识
qwen_model_filename = None
for fname in gguf_files:
    if "Qwen" in fname and "1.7B" in fname and fname.endswith(".gguf"): # 简单匹配
        qwen_model_filename = fname
        break

if not qwen_model_filename:
    print("Could not automatically find a Qwen 1.7B GGUF model. Please specify one.")
    # 或者让用户输入选择：
    # choice = int(input(f"Select a model by number (1-{len(gguf_files)}): ")) - 1
    # qwen_model_filename = gguf_files[choice]
    exit()

MODEL_PATH = os.path.join(MODEL_DIR, qwen_model_filename)
print(f"\nUsing model: {MODEL_PATH}")

# --- LLM 初始化参数 ---
# n_gpu_layers: 设置为 > 0 可以将部分层卸载到 GPU (如果兼容且已安装 GPU 支持的 llama.cpp 版本)
# 设置为 0 则完全使用 CPU。对于 RTX 3060 (12GB)，可以尝试设置一个较大的值，例如 20-30，具体取决于模型大小和VRAM。
# 对于1.7B模型，如果VRAM足够，可以尝试更高的值，甚至 -1 (全部卸载)。
# 我们先从 CPU 开始测试，n_gpu_layers=0
N_GPU_LAYERS = 0
N_CTX = 2048 #模型的上下文窗口大小，Qwen1.7B 可能支持更长的，具体查阅模型卡片
N_BATCH = 512 # 提示处理的批处理大小

try:
    print(f"Initializing Llama model from: {MODEL_PATH}...")
    print(f"Parameters: n_gpu_layers={N_GPU_LAYERS}, n_ctx={N_CTX}, n_batch={N_BATCH}")
    
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        n_batch=N_BATCH,
        verbose=True # 打印更多加载信息
    )
    print("Llama model initialized successfully.")

    # --- 测试 1: 简单文本生成 ---
    prompt_simple = "中国的首都是哪里？请用中文回答。"
    print(f"\n--- Test 1: Simple Text Generation ---")
    print(f"Prompt: {prompt_simple}")
    output_simple = llm(
        prompt_simple,
        max_tokens=50, # 生成的最大 token 数
        stop=["\n", " Human:", " Assistant:"], # 停止词
        echo=False # 不回显 prompt
    )
    print(f"LLM Output: {output_simple['choices'][0]['text'].strip()}")

    # --- 测试 2: Text-to-Cypher (初步) ---
    # 使用简化的 Schema 和问题进行测试
    # 您可以将 NEW_KG_SCHEMA_DESCRIPTION 从 constants.py 复制过来或导入
    # 这里为了独立性，我们先用一个极简的描述
    kg_schema_simple = """
    节点标签: ExtractedEntity (属性: text, label)
    关系类型: WORKS_AT (从 PERSON 到 ORGANIZATION), ASSIGNED_TO (从 TASK 到 PERSON)
    实体标签值: PERSON, ORGANIZATION, TASK
    """
    user_question_cypher = "张三在哪里工作？"
    prompt_cypher = f"""
    你是一个将自然语言问题转换为Cypher查询的助手。
    严格根据以下Schema生成Cypher查询：
    {kg_schema_simple}
    用户问题: {user_question_cypher}
    Cypher查询: MATCH (p:ExtractedEntity {{text: '张三', label: 'PERSON'}})-[:WORKS_AT]->(o:ExtractedEntity {{label: 'ORGANIZATION'}}) RETURN o.text
    用户问题: 项目Alpha分配给了谁？
    Cypher查询: MATCH (t:ExtractedEntity {{text: '项目Alpha', label: 'TASK'}})-[:ASSIGNED_TO]->(p:ExtractedEntity {{label: 'PERSON'}}) RETURN p.text
    用户问题: {user_question_cypher}
    Cypher查询:""" # Few-shot prompting

    print(f"\n--- Test 2: Text-to-Cypher (Simple) ---")
    print(f"User question for Cypher: {user_question_cypher}")
    # print(f"Cypher Prompt (simplified): {prompt_cypher}") # Prompt 可能很长
    
    output_cypher = llm(
        prompt_cypher,
        max_tokens=150,
        stop=["\n", "用户问题:"],
        temperature=0.1, # 对于代码生成，低temperature通常更好
        echo=False
    )
    generated_cypher = output_cypher['choices'][0]['text'].strip()
    print(f"Generated Cypher: {generated_cypher}")

    # --- 测试 3: 上下文问答 (初步) ---
    context_qa = "根据KuzuDB的文档，它是一个嵌入式的图数据库，支持Cypher查询语言。"
    user_question_qa = "KuzuDB支持什么查询语言？"
    prompt_qa = f"""
    根据以下上下文回答问题。
    上下文: {context_qa}
    问题: {user_question_qa}
    答案:"""

    print(f"\n--- Test 3: Contextual QA (Simple) ---")
    print(f"Context: {context_qa}")
    print(f"User question for QA: {user_question_qa}")
    output_qa = llm(
        prompt_qa,
        max_tokens=100,
        stop=["\n", "问题:"],
        temperature=0.2,
        echo=False
    )
    generated_answer_qa = output_qa['choices'][0]['text'].strip()
    print(f"Generated Answer: {generated_answer_qa}")

except Exception as e:
    print(f"An error occurred with Llama.cpp: {e}")
    import traceback
    traceback.print_exc()
finally:
    print("\nLlama.cpp test script finished.")
    if 'llm' in locals() and hasattr(llm, 'close'): # llama_cpp Llama对象没有close方法
         pass # llm对象在Python中通常通过垃圾回收来释放资源