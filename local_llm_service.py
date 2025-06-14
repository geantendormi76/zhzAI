# /home/zhz/zhz_agent/local_llm_service.py
import os
import time
import uuid
import traceback
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Union
import json

# --- START: 添加 HardwareManager 导入 ---
try:
    # 假设 hardware_manager.py 在 zhz_agent/utils/ 目录下
    from utils.hardware_manager import HardwareManager
except ImportError:
    # 如果 utils 不在 zhz_agent 的直接子目录，或者PYTHONPATH设置问题，可能需要调整导入路径
    # 例如，如果 zhz_agent 是项目根目录，且 utils 是根目录下的文件夹：
    # from utils.hardware_manager import HardwareManager
    # 如果 hardware_manager.py 与 local_llm_service.py 在同一目录（不推荐）：
    # from hardware_manager import HardwareManager
    print("ERROR: Failed to import HardwareManager. Ensure it's in the correct path and PYTHONPATH is set.")
    # 定义一个占位符，以便服务至少能尝试启动（尽管功能会受限）
    HardwareManager = None
# --- END: 添加 HardwareManager 导入 ---

from zhz_rag.config.pydantic_models import ExtractedEntitiesAndRelationIntent

import uvicorn
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from llama_cpp import Llama, LlamaGrammar
import re

# --- 配置 ---
MODEL_DIR = os.getenv("LOCAL_LLM_MODEL_DIR", "/home/zhz/models/Qwen3-1.7B-GGUF")
MODEL_FILENAME = os.getenv("LOCAL_LLM_MODEL_FILENAME")
# N_GPU_LAYERS 将在 lifespan 中被 HAL 动态设置，这里的初始值可以作为无法检测时的回退
INITIAL_N_GPU_LAYERS = int(os.getenv("LOCAL_LLM_N_GPU_LAYERS", 0))
N_CTX = int(os.getenv("LOCAL_LLM_N_CTX", 4096))
N_BATCH = int(os.getenv("LOCAL_LLM_N_BATCH", 512))
SERVICE_PORT = int(os.getenv("LOCAL_LLM_SERVICE_PORT", 8088))
SERVICE_HOST = "0.0.0.0"

GBNF_FILE_PATH = os.path.join(os.path.dirname(__file__), "core", "grammars", "cypher_or_unable_output.gbnf")

# --- 全局变量 ---
llama_model: Optional[Llama] = None
model_path_global: Optional[str] = None
# logit_bias 相关的全局变量 (在 lifespan 中初始化)
failure_phrase_token_ids: List[int] = []
logit_bias_for_failure_phrase: Optional[Dict[int, float]] = None
# 初始化为环境变量或默认值，稍后会被HAL覆盖
N_GPU_LAYERS: int = INITIAL_N_GPU_LAYERS


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llama_model, model_path_global, failure_phrase_token_ids, logit_bias_for_failure_phrase
    # 声明我们要修改全局变量 N_GPU_LAYERS
    global N_GPU_LAYERS

    print("--- Local LLM Service: Lifespan startup (with GBNF and logit_bias prep) ---")

    # --- START: HardwareManager 集成 ---
    final_n_gpu_layers_to_use = INITIAL_N_GPU_LAYERS  # 默认使用初始值
    if HardwareManager:  # 确保导入成功
        try:
            print("Initializing HardwareManager for dynamic configuration...")
            hw_manager = HardwareManager()
            hw_info = hw_manager.get_hardware_info()

            if hw_info:
                print(f"Detected Hardware: {hw_info}")
                # Qwen3-1.7B 模型参数
                model_total_layers = 28  # 根据你的模型实际层数调整
                model_size_on_disk_gb = 1.8  # Qwen3-1.7B-Q8_0.gguf 约 1.7-1.8 GB
                context_length_tokens = N_CTX  # 使用当前配置的上下文长度

                # 从 HardwareManager 获取推荐的GPU层数
                recommended_layers = hw_manager.recommend_llm_gpu_layers(
                    model_total_layers=model_total_layers,
                    model_size_on_disk_gb=model_size_on_disk_gb,
                    context_length_tokens=context_length_tokens
                    # 可以按需传递 kv_cache_gb_per_1k_ctx 和 safety_buffer_vram_gb
                )
                print(f"HardwareManager recommended n_gpu_layers: {recommended_layers}")

                # 应用推荐值（优雅降级已在 recommend_llm_gpu_layers 内部处理）
                final_n_gpu_layers_to_use = recommended_layers

                if final_n_gpu_layers_to_use == 0 and hw_info.gpu_available:
                    print("INFO: GPU is available, but HAL recommended 0 GPU layers (CPU inference). This might be due to low VRAM or other heuristics.")
                elif final_n_gpu_layers_to_use > 0:
                    print(f"INFO: GPU available and HAL recommended {final_n_gpu_layers_to_use} layers for GPU offload.")
                elif not hw_info.gpu_available:
                    print("INFO: No compatible GPU detected by HAL. Using CPU inference (0 GPU layers).")

            else:
                print("WARNING: HardwareManager failed to get hardware info. Using initial n_gpu_layers value.")
        except Exception as e_hal:
            print(f"ERROR during HardwareManager initialization or recommendation: {e_hal}. Using initial n_gpu_layers value.")
            traceback.print_exc()
    else:
        print("WARNING: HardwareManager class not available. Using initial n_gpu_layers value.")

    N_GPU_LAYERS = final_n_gpu_layers_to_use  # 更新全局变量
    print(f"Final n_gpu_layers to be used for Llama model: {N_GPU_LAYERS}")
    # --- END: HardwareManager 集成 ---

    model_file_to_load = MODEL_FILENAME
    if not model_file_to_load:
        print(f"MODEL_FILENAME environment variable not set. Attempting to auto-detect GGUF file in {MODEL_DIR}...")
        try:
            gguf_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".gguf")]
            if not gguf_files:
                error_msg = f"No GGUF models found in directory: {MODEL_DIR}"
                print(f"ERROR: {error_msg}")
                app.state.cypher_path_grammar = None  # Ensure state variable exists even on error
                raise RuntimeError(error_msg)
            if len(gguf_files) > 1:
                print(f"Warning: Multiple GGUF models found in {MODEL_DIR}. Using the first one: {gguf_files[0]}")
            model_file_to_load = gguf_files[0]
            print(f"Auto-detected GGUF file: {model_file_to_load}")
        except FileNotFoundError:
            error_msg = f"Model directory not found: {MODEL_DIR}"
            print(f"ERROR: {error_msg}")
            app.state.cypher_path_grammar = None
            raise RuntimeError(error_msg)
        except Exception as e_find_model:
            error_msg = f"Error auto-detecting GGUF file: {e_find_model}"
            print(f"ERROR: {error_msg}")
            app.state.cypher_path_grammar = None
            raise RuntimeError(error_msg)

    model_path_global = os.path.join(MODEL_DIR, model_file_to_load)
    print(f"Attempting to load Llama model from: {model_path_global}")
    # 这里确保使用更新后的 N_GPU_LAYERS
    print(f"Parameters: n_gpu_layers={N_GPU_LAYERS}, n_ctx={N_CTX}, n_batch={N_BATCH}")

    try:
        llama_model = Llama(
            model_path=model_path_global,
            n_gpu_layers=N_GPU_LAYERS,  # <--- 确保这里使用的是更新后的 N_GPU_LAYERS
            n_ctx=N_CTX,
            n_batch=N_BATCH,
            verbose=True
        )
        print("Llama model loaded successfully.")

        # --- 初始化 logit_bias 相关 ---
        failure_phrase_str = "无法生成Cypher查询."
        if llama_model:
            try:
                # add_bos=False, special=False (通常用于非起始的、纯文本的词元化)
                failure_phrase_token_ids = llama_model.tokenize(failure_phrase_str.encode("utf-8"), add_bos=False, special=False)
                # 为这些 token ID 设置正向偏置，例如 10.0 (可以调整)
                # 避免偏置 EOS token (如果它意外地出现在短语的词元化结果中)
                eos_token_id = llama_model.token_eos()
                logit_bias_for_failure_phrase = {
                    token_id: 10.0 for token_id in failure_phrase_token_ids if token_id != eos_token_id
                }
                print(f"Successfully tokenized failure phrase '{failure_phrase_str}' to IDs: {failure_phrase_token_ids}")
                print(f"Logit bias for failure phrase: {logit_bias_for_failure_phrase}")
            except Exception as e_tokenize:
                print(f"ERROR: Failed to tokenize failure phrase for logit_bias: {e_tokenize}")
                failure_phrase_token_ids = []
                logit_bias_for_failure_phrase = None
        # --- 结束 logit_bias 初始化 ---

        gbnf_grammar_instance: Optional[LlamaGrammar] = None
        print(f"Attempting to load GBNF grammar from: {GBNF_FILE_PATH}")
        if os.path.exists(GBNF_FILE_PATH):
            try:
                gbnf_grammar_instance = LlamaGrammar.from_file(GBNF_FILE_PATH)
                print("GBNF grammar (success/failure paths) loaded successfully into lifespan.")
            except Exception as e_gbnf:
                print(f"ERROR: Failed to load or parse GBNF grammar file '{GBNF_FILE_PATH}': {e_gbnf}")
                traceback.print_exc()
        else:
            print(f"ERROR: GBNF grammar file not found at '{GBNF_FILE_PATH}'.")

        app.state.cypher_path_grammar = gbnf_grammar_instance

    except Exception as e:
        print(f"FATAL: Failed to load Llama model or prepare GBNF/logit_bias: {e}")
        app.state.cypher_path_grammar = None
        logit_bias_for_failure_phrase = None  # Ensure this is also cleared

    yield
    print("--- Local LLM Service: Lifespan shutdown ---")

app = FastAPI(
    title="Local LLM Service (OpenAI Compatible)",
    description="Uses GBNF with logit_bias for conditional JSON output.",
    version="0.1.7",
    lifespan=lifespan
)


class ChatMessage(BaseModel):
    role: str
    content: str
    
class ResponseFormat(BaseModel): # 新增这个辅助模型
    type: str = "json_object" # 默认为 json_object
    schema_definition: Optional[Dict[str, Any]] = Field(default=None, alias="schema") # 使用 alias 兼容 OpenAI 的 "schema" 字段

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: Optional[int] = 1024
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    response_format: Optional[ResponseFormat] = None # <--- 新增此行


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


def post_process_llm_output(content: Optional[str], finish_reason: Optional[str]) -> Optional[str]:
    if content is None:
        return None
    processed_content = content
    # Remove <think>...</think> blocks
    think_block_pattern = re.compile(r"<think[^>]*>.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE)
    processed_content = think_block_pattern.sub("", processed_content)

    # Handle potentially incomplete <think> tags if generation was cut short
    if finish_reason == "length" and \
       re.match(r"<think[^>]*>", processed_content.strip(), flags=re.IGNORECASE) and \
       not re.search(r"</think\s*>", processed_content, flags=re.IGNORECASE):
        print("DEBUG_POST_PROCESS: Incomplete think block due to length, attempting to remove partial tag.")
        # More aggressive removal of any leading <think...> tag if it's incomplete
        processed_content = re.sub(r"^<think[^>]*>", "", processed_content.strip(), flags=re.IGNORECASE).strip()

    # Remove any remaining stray <think> or </think> tags
    stray_think_tag_pattern = re.compile(r"</?\s*think[^>]*?>\s*", flags=re.IGNORECASE)
    processed_content = stray_think_tag_pattern.sub("", processed_content)
    return processed_content.strip()


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion_endpoint(fastapi_req: FastAPIRequest, request: ChatCompletionRequest):
    global llama_model, model_path_global, logit_bias_for_failure_phrase

    # --- 添加调试日志 ---
    print("\n--- local_llm_service: Received /v1/chat/completions request ---")
    try:
        print(f"Request Body (raw): {await fastapi_req.body()}")  # 打印原始请求体
        print(f"Request Model: {request.model}")
        print(f"Request Messages (count): {len(request.messages) if request.messages else 0}")
        print(f"Request Temperature: {request.temperature}")
        print(f"Request Max Tokens: {request.max_tokens}")
        print(f"Request Stop: {request.stop}")
    except Exception as e_req_log:
        print(f"Error logging request details: {e_req_log}")
    # --- 调试日志结束 ---

    loaded_cypher_path_grammar: Optional[LlamaGrammar] = getattr(fastapi_req.app.state, 'cypher_path_grammar', None)

    if llama_model is None:
        raise HTTPException(status_code=503, detail="Llama model is not loaded or failed to load.")
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming responses are not yet supported by this service.")

    response_content_raw: Optional[str] = None
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    finish_reason = "stop"
    request_model_name = request.model
    final_json_output_str_for_client: Optional[str] = None

    if not hasattr(llama_model, "create_chat_completion"):
        print("CRITICAL_ERROR: llama_model.create_chat_completion method not found.")
        # Construct a proper JSON string for the error case directly
        error_response_obj = {"status": "unable_to_generate", "query": "LLM service misconfiguration."}
        final_json_output_str_for_client = json.dumps(error_response_obj, ensure_ascii=False)
        # Early return with the error structure if model is misconfigured
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            object="chat.completion",
            created=int(time.time()),
            model=request_model_name or (os.path.basename(model_path_global) if model_path_global else "local-llm-misconfigured"),
            choices=[ChatCompletionChoice(index=0, message=ChatCompletionMessage(role="assistant", content=final_json_output_str_for_client), finish_reason="error")],
            usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        )

    try:
        dict_messages = [msg.model_dump() for msg in request.messages]
        completion_params: Dict[str, Any] = {
            "messages": dict_messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or 1024,
            "stop": request.stop,
        }

        # --- 修改：使用请求中传递的 response_format (如果存在且有效) ---
        if request.response_format and \
        request.response_format.type == "json_object" and \
        request.response_format.schema_definition:
            try:
                # 直接使用客户端提供的 schema
                completion_params["response_format"] = {
                    "type": "json_object",
                    "schema": request.response_format.schema_definition 
                }
                print(f"DEBUG_FastAPI: Enabled JSON mode using schema provided in request for model {request.model}.")
                # 可选打印详细schema:
                # try:
                #     print(f"DEBUG_FastAPI: JSON Schema from request: {json.dumps(request.response_format.schema_definition, indent=2, ensure_ascii=False)}")
                # except Exception as e_dumps:
                #     print(f"DEBUG_FastAPI: Could not dump schema for logging: {e_dumps}")
            except Exception as e_schema_assign:
                print(f"ERROR_FastAPI: Failed to assign response_format from request: {e_schema_assign}")
        elif "kg_entity_extraction" in request.model.lower():
            # 回退逻辑：如果模型名暗示KG抽取，但请求中没有提供有效的response_format，
            # 我们可以选择报错，或者使用一个默认的单对象抽取schema（但这不适用于批处理）。
            # 为了批处理的正确性，我们应该依赖客户端提供正确的schema。
            # 因此，如果进行批处理，客户端必须提供 BATCH_KG_EXTRACTION_SCHEMA。
            # 如果是单对象抽取，客户端也应该提供 DEFAULT_KG_EXTRACTION_SCHEMA。
            # 这里可以加一个警告，如果需要JSON输出但schema未提供。
            print(f"WARNING_FastAPI: Model name {request.model} suggests JSON output, but no valid response_format.schema provided in the request. LLM might not produce structured JSON.")
        # --- 结束修改 ---

        is_cypher_gen_task = False
        if dict_messages and dict_messages[0]["role"] == "system":
            system_content_for_check = dict_messages[0]["content"]
            keyword_to_check = "知识图谱结构 (KuzuDB) 与 Cypher 查询生成规则"  # Ensure this matches constants.py
            processed_system_content = system_content_for_check.lower()
            processed_keyword = keyword_to_check.lower()
            print(f"DEBUG_FastAPI_CypherTaskCheck: System prompt content (LOWERCASED, first 300 chars): '{processed_system_content[:300]}...'")
            print(f"DEBUG_FastAPI_CypherTaskCheck: Keyword to check (LOWERCASED): '{processed_keyword}'")
            if processed_keyword in processed_system_content:
                is_cypher_gen_task = True
                print("DEBUG_FastAPI_CypherTaskCheck: Cypher generation task DETECTED (after lowercasing and keyword adjustment).")
            else:
                print("DEBUG_FastAPI_CypherTaskCheck: Cypher generation task NOT DETECTED (keyword missing after lowercasing).")
        else:
            print("DEBUG_FastAPI_CypherTaskCheck: No system message found or messages empty, not a Cypher task.")

        if is_cypher_gen_task:
            print("DEBUG_FastAPI: Cypher generation task DETECTED.")
            if loaded_cypher_path_grammar is not None:
                print("DEBUG_FastAPI: Applying GBNF grammar (success/failure paths) FROM APP.STATE.")
                completion_params["grammar"] = loaded_cypher_path_grammar

                if logit_bias_for_failure_phrase:
                    print(f"DEBUG_FastAPI: Applying logit_bias for failure phrase: {logit_bias_for_failure_phrase}")
                    completion_params["logit_bias"] = logit_bias_for_failure_phrase
            else:
                print("DEBUG_FastAPI: GBNF grammar FROM APP.STATE IS NONE. Proceeding without grammar for Cypher task.")
        else:
            print("DEBUG_FastAPI: Not a Cypher task. No grammar or specific logit_bias applied.")

        print(f"DEBUG_FastAPI: Calling llama_model.create_chat_completion with params (preview): "
              f"model={request.model}, temp={completion_params['temperature']}, "
              f"max_tokens={completion_params['max_tokens']}, stop={completion_params['stop']}, "
              f"json_mode_enabled={'response_format' in completion_params}")

        completion = llama_model.create_chat_completion(**completion_params)

        # --- 添加调试日志 ---
        print(f"DEBUG_FastAPI: Raw completion object from llama_model.create_chat_completion: {completion}")
        # --- 调试日志结束 ---

        response_content_raw = completion['choices'][0]['message']['content']
        prompt_tokens = completion['usage']['prompt_tokens']
        completion_tokens = completion['usage']['completion_tokens']
        total_tokens = completion['usage']['total_tokens']
        finish_reason = completion['choices'][0].get('finish_reason', 'stop')
        print(f"DEBUG_FastAPI: Raw content from llama_model: '{response_content_raw}'")

    except Exception as e:
        print(f"Error during llama_model.create_chat_completion: {e}")
        traceback.print_exc()
        # Ensure this is a JSON string
        final_json_output_str_for_client = json.dumps({"status": "unable_to_generate", "query": "LLM call failed during generation."})
        finish_reason = "error"  # Indicate an error finish

    if final_json_output_str_for_client is None:  # Only process if no error above set this
        processed_llm_text = post_process_llm_output(response_content_raw, finish_reason)

        if is_cypher_gen_task:
            standard_success_template = {"status": "success", "query": ""}
            standard_unable_json_obj = {"status": "unable_to_generate", "query": "无法生成Cypher查询."}
            final_json_to_return_obj = standard_unable_json_obj  # Default to unable

            if processed_llm_text:
                cleaned_text_for_json_parse = processed_llm_text.strip()
                # Remove Markdown code block fences if present
                if cleaned_text_for_json_parse.startswith("```json"):
                    cleaned_text_for_json_parse = cleaned_text_for_json_parse[len("```json"):].strip()
                if cleaned_text_for_json_parse.endswith("```"):
                    cleaned_text_for_json_parse = cleaned_text_for_json_parse[:-len("```")].strip()

                try:
                    data = json.loads(cleaned_text_for_json_parse)
                    if isinstance(data, dict) and "status" in data and "query" in data:
                        if data.get("status") == "success" and isinstance(data.get("query"), str) and data.get("query").strip():
                            final_json_to_return_obj = data
                            print(f"DEBUG_FastAPI: LLM output is a valid 'success' JSON (GBNF success path likely): {json.dumps(final_json_to_return_obj, ensure_ascii=False)}")
                        elif data.get("status") == "unable_to_generate" and data.get("query") == "无法生成Cypher查询.":
                            final_json_to_return_obj = data  # Already standard
                            print(f"DEBUG_FastAPI: LLM output is a valid 'unable_to_generate' JSON (GBNF failure path likely): {json.dumps(final_json_to_return_obj, ensure_ascii=False)}")
                        else:  # JSON has status/query but not matching expected values
                            print(f"DEBUG_FastAPI: LLM JSON has unexpected status/query content. Status: '{data.get('status')}', Query: '{str(data.get('query'))[:100]}'. Defaulting to standard 'unable_to_generate'.")
                            # final_json_to_return_obj remains standard_unable_json_obj
                    else:
                        print(f"DEBUG_FastAPI: LLM output parsed as JSON, but not the expected dict with status/query: '{cleaned_text_for_json_parse}'. Defaulting to standard 'unable_to_generate'.")
                except json.JSONDecodeError:
                    print(f"DEBUG_FastAPI: LLM output was not valid JSON. Raw (after post_process): '{processed_llm_text}'. Defaulting to standard 'unable_to_generate'.")
                except Exception as e_parse:
                    print(f"DEBUG_FastAPI: Unexpected error parsing LLM output: {e_parse}. Raw: '{processed_llm_text}'. Defaulting to 'unable_to_generate'.")
            else:
                print("DEBUG_FastAPI: LLM output was empty after post_processing. Defaulting to standard 'unable_to_generate' JSON.")

            final_json_output_str_for_client = json.dumps(final_json_to_return_obj, ensure_ascii=False)
        else:
            # For non-Cypher tasks, return the processed text directly
            final_json_output_str_for_client = processed_llm_text if processed_llm_text is not None else ""

    print(f"DEBUG_FastAPI: Final content string to be returned to client: '{final_json_output_str_for_client}'")

    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created_time = int(time.time())
    effective_model_name = request_model_name
    if not effective_model_name and model_path_global:
        effective_model_name = os.path.basename(model_path_global)
    elif not effective_model_name:
        effective_model_name = "local-llm-unknown"

    # Ensure final_json_output_str_for_client is a string, even if empty (for non-Cypher tasks)
    if final_json_output_str_for_client is None:
        final_json_output_str_for_client = ""  # Or some other default string

    return ChatCompletionResponse(
        id=response_id,
        object="chat.completion",
        created=created_time,
        model=effective_model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=final_json_output_str_for_client),
                finish_reason=finish_reason
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
    )


@app.get("/v1/models", response_model=Dict[str, Any])
async def list_models():
    global model_path_global
    model_id_for_clients = "qwen3local_gguf_gbnf_logit_bias"
    model_name_to_display = "Qwen3-1.7B-GGUF (GBNF+LogitBias)"

    if model_path_global and os.path.exists(model_path_global):
        model_name_to_display = os.path.basename(model_path_global)
        created_timestamp = int(os.path.getctime(model_path_global))
    else:
        # Fallback if model_path_global is not set or file doesn't exist
        model_name_to_display = "Qwen3-1.7B-GGUF (Model path not resolved)"
        created_timestamp = int(time.time())

    return {
        "object": "list",
        "data": [
            {
                "id": model_id_for_clients,
                "object": "model",
                "created": created_timestamp,
                "owned_by": "user",
                "description": f"Locally hosted GGUF model: {model_name_to_display}. Uses GBNF and logit_bias for Cypher tasks."
            }
        ]
    }


if __name__ == "__main__":
    print(f"--- Starting Local LLM FastAPI Service on {SERVICE_HOST}:{SERVICE_PORT} ---")
    print(f"--- Model will be loaded from DIR: {MODEL_DIR}, FILE: {MODEL_FILENAME or 'Auto-detected GGUF'} ---")
    print(f"--- GBNF Grammar for Cypher/Unable output will be loaded from: {GBNF_FILE_PATH} ---")
    uvicorn.run("local_llm_service:app", host=SERVICE_HOST, port=SERVICE_PORT, reload=False)