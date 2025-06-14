from llama_cpp import Llama

model_path = "/home/zhz/models/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf"
n_gpu_layers_to_test = 30 # 或者其他值

print(f"Attempting to load model with n_gpu_layers={n_gpu_layers_to_test}")
try:
    llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers_to_test, verbose=True, n_ctx=512)
    print("Model loaded. Performing a dummy inference...")
    output = llm("This is a test.", max_tokens=10)
    print(f"Inference output: {output}")
    print("Test complete. Check nvidia-smi now.")
    # 在这里暂停，或者让脚本持续运行一段时间，以便观察nvidia-smi
    import time
    time.sleep(60) 
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
