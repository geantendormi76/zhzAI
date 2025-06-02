# /home/zhz/zhz_agent/local_llm_service.py

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from llama_cpp import Llama, LlamaGrammar
import re # 导入 re 模块

# --- 配置 ---
MODEL_DIR = os.getenv("LOCAL_LLM_MODEL_DIR", "/home/zhz/models/Qwen3-1.7B-GGUF")
MODEL_FILENAME = os.getenv("LOCAL_LLM_MODEL_FILENAME") 
N_GPU_LAYERS = int(os.getenv("LOCAL_LLM_N_GPU_LAYERS", 0))
N_CTX = int(os.getenv("LOCAL_LLM_N_CTX", 2048))
N_BATCH = int(os.getenv("LOCAL_LLM_N_BATCH", 512))
SERVICE_PORT = int(os.getenv("LOCAL_LLM_SERVICE_PORT", 8088))
SERVICE_HOST = "0.0.0.0"

# --- 全局变量持有 Llama 模型实例 ---
llama_model: Optional[Llama] = None
model_path_global: Optional[str] = None

# --- Pydantic 模型定义 (兼容 OpenAI Chat Completion API) ---
# (Pydantic 模型定义与上一版本完全相同，此处省略以减少篇幅，请保留您已有的定义)
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: Optional[int] = 512
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None

class ChatCompletionChoiceDelta(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = None

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: ChatCompletionChoiceDelta
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]

class ChatCompletionMessage(BaseModel):
    role: str
    content: Optional[str] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str] = "stop"

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo
    system_fingerprint: Optional[str] = None

# --- FastAPI 应用和生命周期管理 ---
# (lifespan 函数与上一版本完全相同，此处省略，请保留您已有的定义)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llama_model, model_path_global
    print("--- Local LLM Service: Lifespan startup ---")
    
    model_file_to_load = MODEL_FILENAME
    if not model_file_to_load:
        try:
            gguf_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".gguf")]
            if not gguf_files:
                raise RuntimeError(f"No GGUF models found in {MODEL_DIR}")
            if len(gguf_files) > 1:
                print(f"Warning: Multiple GGUF models found in {MODEL_DIR}. Using the first one: {gguf_files[0]}")
            model_file_to_load = gguf_files[0]
        except FileNotFoundError:
            raise RuntimeError(f"Model directory not found: {MODEL_DIR}")


    model_path_global = os.path.join(MODEL_DIR, model_file_to_load)
    print(f"Attempting to load Llama model from: {model_path_global}")
    print(f"Parameters: n_gpu_layers={N_GPU_LAYERS}, n_ctx={N_CTX}, n_batch={N_BATCH}")

    try:
        llama_model = Llama(
            model_path=model_path_global,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
            n_batch=N_BATCH,
            verbose=True
        )
        print("Llama model loaded successfully.")
    except Exception as e:
        print(f"FATAL: Failed to load Llama model: {e}")
        raise RuntimeError(f"Failed to load Llama model: {e}") from e
    
    yield
    
    print("--- Local LLM Service: Lifespan shutdown ---")

app = FastAPI(
    title="Local LLM Service (OpenAI Compatible)",
    description=f"Provides an OpenAI-compatible API for a locally hosted GGUF model via llama-cpp-python.",
    version="0.1.0",
    lifespan=lifespan
)

# --- 辅助函数 ---
# (convert_messages_to_prompt 函数与上一版本完全相同，此处省略，请保留您已有的定义)
def convert_messages_to_prompt(messages: List[ChatMessage], model_chat_format: Optional[str]) -> str:
    if llama_model and hasattr(llama_model, 'chat_format') and llama_model.chat_format:
        if hasattr(llama_model, 'apply_chat_template'):
            try:
                dict_messages = [msg.dict() for msg in messages] # Pydantic V1
                return llama_model.apply_chat_template(dict_messages, add_generation_prompt=True, tokenize=False)
            except Exception as e:
                print(f"Warning: llama_model.apply_chat_template failed: {e}. Falling back to manual formatting.")
        else:
             print(f"Warning: llama_model.apply_chat_template not available. Falling back to manual formatting.")
    
    prompt = ""
    for msg in messages:
        prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def post_process_llm_output(content: Optional[str], finish_reason: Optional[str]) -> Optional[str]:
    if content is None:
        return None
    
    print(f"DEBUG_POST_PROCESS: Input to post_process_llm_output:\n---\n{content}\n---")
    print(f"DEBUG_POST_PROCESS: Finish reason: {finish_reason}")

    processed_content = content
    # 优先尝试移除完整的 <think>...</think> 块
    # 正则表达式：匹配 <think...> (可能带有属性或空格) 开始，到 </think> 结束的所有内容，包括换行
    # re.DOTALL 使 . 匹配换行符
    # re.IGNORECASE 使匹配不区分大小写
    think_block_pattern = re.compile(r"<think[^>]*>.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE)
    processed_content = think_block_pattern.sub("", processed_content)
    
    print(f"DEBUG_POST_PROCESS: After think_block_pattern.sub:\n---\n{processed_content}\n---")

    # 如果因为长度截断，并且内容仍然以 <think 开头（没有闭合），说明整个输出都是不完整的思考
    # 这种情况下，我们可能不希望显示它，或者返回一个特定消息
    if finish_reason == "length" and \
       re.match(r"<think[^>]*>", processed_content.strip(), flags=re.IGNORECASE) and \
       not re.search(r"</think\s*>", processed_content, flags=re.IGNORECASE):
        print("DEBUG_POST_PROCESS: Incomplete think block due to length, returning empty or placeholder.")
        # 可以选择返回空字符串，或者一个提示信息
        # return "" 
        # 或者让后续的strip()处理，如果只剩下一个开头的<think>标签，也会被移除
        # 但更稳妥的是在这里就处理掉

    # 再次尝试移除可能单独存在的 <think> 或 </think> 标签（以防万一）
    stray_think_tag_pattern = re.compile(r"</?\s*think[^>]*?>\s*", flags=re.IGNORECASE)
    processed_content = stray_think_tag_pattern.sub("", processed_content)

    print(f"DEBUG_POST_PROCESS: After stray_think_tag_pattern.sub:\n---\n{processed_content}\n---")
    
    final_stripped_content = processed_content.strip()
    print(f"DEBUG_POST_PROCESS: After final strip:\n---\n{final_stripped_content}\n---")

    return final_stripped_content

# --- API 端点 ---

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global llama_model, model_path_global
    if llama_model is None:
        raise HTTPException(status_code=503, detail="Llama model is not loaded or failed to load.")

    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming responses are not yet supported by this service.")

    response_content_raw: Optional[str] = None
    response_content_final: Optional[str] = None
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    finish_reason = "stop" # 默认值
    
    request_model_name = request.model

    if hasattr(llama_model, "create_chat_completion"):
        try:
            dict_messages = [msg.dict() for msg in request.messages]
            
            print(f"DEBUG_FastAPI: Calling llama_model.create_chat_completion with messages: {dict_messages}")
            completion = llama_model.create_chat_completion(
                messages=dict_messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop,
            )
            response_content_raw = completion['choices'][0]['message']['content']
            print(f"DEBUG_FastAPI: Raw content from llama_model: '{response_content_raw}'")

            prompt_tokens = completion['usage']['prompt_tokens']
            completion_tokens = completion['usage']['completion_tokens']
            total_tokens = completion['usage']['total_tokens']
            finish_reason = completion['choices'][0].get('finish_reason', 'stop') # <--- 获取 finish_reason
        except Exception as e:
            print(f"Error during llama_model.create_chat_completion: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating completion: {str(e)}")
    else:
        print("Warning: llama_model.create_chat_completion not found. Using direct call with formatted prompt.")
        prompt_str = convert_messages_to_prompt(request.messages, getattr(llama_model, 'chat_format', None))
        try:
            output = llama_model(
                prompt_str,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop,
                echo=False
            )
            response_content_raw = output['choices'][0]['text'].strip()
            print(f"DEBUG_FastAPI: Raw content from llama_model (direct call): '{response_content_raw}'")

            prompt_tokens = output['usage']['prompt_tokens']
            completion_tokens = output['usage']['completion_tokens']
            total_tokens = output['usage']['total_tokens']
            finish_reason = output['choices'][0].get('finish_reason', 'stop') # <--- 获取 finish_reason
        except Exception as e:
            print(f"Error during llama_model direct call: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating completion: {str(e)}")

    # 在调用 post_process_llm_output 时传递 finish_reason
    response_content_processed = post_process_llm_output(response_content_raw, finish_reason) # <--- 修改点
    
    if response_content_processed == "" and response_content_raw and response_content_raw.strip() != "":
        print(f"Warning: Post-processing resulted in empty content. Original content was: '{response_content_raw[:200]}...' Using raw content instead.")
        response_content_final = response_content_raw 
    else:
        response_content_final = response_content_processed
    
    print(f"DEBUG_FastAPI: Final content to be returned: '{response_content_final}'")

    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())
    
    effective_model_name = request_model_name
    if not effective_model_name and model_path_global:
        effective_model_name = model_path_global.split('/')[-1]
    elif not effective_model_name:
        effective_model_name = "local-llm-unknown"

    return ChatCompletionResponse(
        id=response_id,
        object="chat.completion",
        created=created_time,
        model=effective_model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=response_content_final),
                finish_reason=finish_reason
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
    )

# (/v1/models 和 if __name__ == "__main__": 部分与上一版本完全相同，此处省略，请保留您已有的定义)
@app.get("/v1/models", response_model=Dict[str, Any])
async def list_models():
    global model_path_global
    # 这个 ID 应该与 config.yaml 中 litellm_params.model 的 "openai/" 后面的部分匹配
    model_id_for_clients = "qwen3local" # <--- 修改这里，与 config.yaml 一致

    return {
        "object": "list",
        "data": [
            {
                "id": model_id_for_clients, 
                "object": "model",
                "created": int(os.path.getctime(model_path_global)) if model_path_global and os.path.exists(model_path_global) else int(time.time()),
                "owned_by": "user",
            }
        ]
    }

if __name__ == "__main__":
    print(f"--- Starting Local LLM FastAPI Service on {SERVICE_HOST}:{SERVICE_PORT} ---")
    print(f"--- Model will be loaded from DIR: {MODEL_DIR}, FILE: {MODEL_FILENAME or 'Auto-detected GGUF'} ---")
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT)