# zhz_agent/utils/hardware_manager.py
import psutil
import torch # 我们会用 torch.cuda 来检测 NVIDIA GPU
import logging
from typing import Optional, Dict, Any

# 配置日志记录器
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO) # 默认设置为 INFO，可以按需调整


class HardwareInfo:
    """用于存储检测到的硬件信息的简单数据类"""
    def __init__(self,
                 cpu_logical_cores: int,
                 cpu_physical_cores: int,
                 total_system_ram_gb: float,
                 gpu_available: bool = False,
                 gpu_name: Optional[str] = None,
                 gpu_vram_total_gb: float = 0.0):
        self.cpu_logical_cores = cpu_logical_cores
        self.cpu_physical_cores = cpu_physical_cores
        self.total_system_ram_gb = total_system_ram_gb
        self.gpu_available = gpu_available
        self.gpu_name = gpu_name
        self.gpu_vram_total_gb = gpu_vram_total_gb

    def __str__(self):
        gpu_str = (f"GPU: {self.gpu_name} with {self.gpu_vram_total_gb:.2f} GB VRAM"
                   if self.gpu_available else "GPU: Not available or not detected")
        return (f"HardwareInfo(CPU: {self.cpu_physical_cores} Physical/{self.cpu_logical_cores} Logical Cores, "
                f"RAM: {self.total_system_ram_gb:.2f} GB, {gpu_str})")


class HardwareManager:
    """
    硬件抽象层 (HAL)，用于检测系统当前的硬件特性，
    并为上层应用提供一个统一的接口来查询和利用这些资源。
    """
    def __init__(self):
        logger.info("Initializing HardwareManager...")
        self.hw_info: Optional[HardwareInfo] = None
        self._detect_hardware()
        if self.hw_info:
            logger.info(f"Hardware detection complete: {self.hw_info}")
        else:
            logger.error("Hardware detection failed to produce valid HardwareInfo.")

    def _detect_cpu_info(self) -> Dict[str, Any]:
        """检测CPU信息"""
        try:
            logical_cores = psutil.cpu_count(logical=True)
            physical_cores = psutil.cpu_count(logical=False)
            # psutil.cpu_count(logical=False) 在某些系统上可能返回None，需要处理
            if physical_cores is None:
                logger.warning("Could not determine physical core count, using logical core count as fallback for physical.")
                physical_cores = logical_cores
            logger.info(f"CPU Info: Logical cores={logical_cores}, Physical cores={physical_cores}")
            return {
                "cpu_logical_cores": logical_cores,
                "cpu_physical_cores": physical_cores
            }
        except Exception as e:
            logger.error(f"Error detecting CPU info: {e}", exc_info=True)
            return {
                "cpu_logical_cores": os.cpu_count() or 1, # Fallback
                "cpu_physical_cores": os.cpu_count() or 1 # Fallback
            }

    def _detect_memory_info(self) -> Dict[str, Any]:
        """检测内存信息"""
        try:
            virtual_mem = psutil.virtual_memory()
            total_ram_gb = virtual_mem.total / (1024 ** 3)
            logger.info(f"Memory Info: Total RAM={total_ram_gb:.2f} GB")
            return {"total_system_ram_gb": total_ram_gb}
        except Exception as e:
            logger.error(f"Error detecting memory info: {e}", exc_info=True)
            return {"total_system_ram_gb": 0.0} # Fallback

    def _detect_gpu_info(self) -> Dict[str, Any]:
        """检测GPU信息 (目前仅NVIDIA GPU通过torch.cuda)"""
        gpu_details = {
            "gpu_available": False,
            "gpu_name": None,
            "gpu_vram_total_gb": 0.0
        }
        try:
            if torch.cuda.is_available():
                gpu_details["gpu_available"] = True
                # 默认检测第一个GPU的信息
                device_id = 0 # torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(device_id)
                props = torch.cuda.get_device_properties(device_id)
                vram_total_gb = props.total_memory / (1024 ** 3)
                
                gpu_details["gpu_name"] = gpu_name
                gpu_details["gpu_vram_total_gb"] = vram_total_gb
                logger.info(f"NVIDIA GPU Detected: Name='{gpu_name}', VRAM={vram_total_gb:.2f} GB")
            else:
                logger.info("torch.cuda.is_available() returned False. No compatible NVIDIA GPU detected by PyTorch.")
        except ImportError:
            logger.warning("PyTorch (torch) not installed or CUDA part not available. Cannot detect NVIDIA GPU via torch.cuda.")
        except Exception as e:
            logger.error(f"Error detecting NVIDIA GPU info via torch.cuda: {e}", exc_info=True)
        
        # 未来可以在这里添加pynvml的检测逻辑作为补充或主要方式，
        # 以及针对AMD/Intel GPU的检测尝试。
        # 例如，使用pynvml：
        # try:
        #     import pynvml
        #     pynvml.nvmlInit()
        #     handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 假设第一个GPU
        #     gpu_name_pynvml = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
        #     mem_info_pynvml = pynvml.nvmlDeviceGetMemoryInfo(handle)
        #     vram_total_gb_pynvml = mem_info_pynvml.total / (1024**3)
        #     if not gpu_details["gpu_available"]: # 如果torch没检测到，用pynvml的结果
        #         gpu_details["gpu_available"] = True
        #         gpu_details["gpu_name"] = gpu_name_pynvml
        #         gpu_details["gpu_vram_total_gb"] = vram_total_gb_pynvml
        #         logger.info(f"NVIDIA GPU Detected (via pynvml): Name='{gpu_name_pynvml}', VRAM={vram_total_gb_pynvml:.2f} GB")
        #     pynvml.nvmlShutdown()
        # except ImportError:
        #     logger.info("pynvml not installed. Skipping pynvml GPU check.")
        # except pynvml.NVMLError as e_pynvml:
        #     logger.warning(f"pynvml error during GPU detection: {e_pynvml}. This might happen if NVIDIA drivers are not set up correctly.")
        # except Exception as e_pynvml_other:
        #     logger.error(f"Unexpected error during pynvml GPU detection: {e_pynvml_other}", exc_info=True)

        return gpu_details

    def _detect_hardware(self):
        """执行所有硬件检测并填充 HardwareInfo"""
        cpu_info = self._detect_cpu_info()
        mem_info = self._detect_memory_info()
        gpu_info = self._detect_gpu_info()
        
        self.hw_info = HardwareInfo(
            cpu_logical_cores=cpu_info["cpu_logical_cores"],
            cpu_physical_cores=cpu_info["cpu_physical_cores"],
            total_system_ram_gb=mem_info["total_system_ram_gb"],
            gpu_available=gpu_info["gpu_available"],
            gpu_name=gpu_info["gpu_name"],
            gpu_vram_total_gb=gpu_info["gpu_vram_total_gb"]
        )

    def get_hardware_info(self) -> Optional[HardwareInfo]:
        """返回检测到的硬件信息"""
        return self.hw_info

    def recommend_llm_gpu_layers(self, model_total_layers: int, model_size_on_disk_gb: float, kv_cache_gb_per_1k_ctx: float = 0.25, context_length_tokens: int = 4096, safety_buffer_vram_gb: float = 1.5) -> int:
        """
        根据可用VRAM推荐LLM应卸载到GPU的层数。
        这是一个启发式方法，具体效果取决于模型架构和llama.cpp的实现。
        """
        if not self.hw_info or not self.hw_info.gpu_available or self.hw_info.gpu_vram_total_gb == 0:
            logger.info("GPU not available or VRAM info missing, recommending 0 GPU layers (CPU only).")
            return 0

        # 估算KV缓存在VRAM中的大小
        estimated_kv_cache_vram_gb = (context_length_tokens / 1000) * kv_cache_gb_per_1k_ctx
        
        # 可用于模型权重的VRAM = 总VRAM - KV缓存估算 - 安全余量
        available_vram_for_model_weights_gb = self.hw_info.gpu_vram_total_gb - estimated_kv_cache_vram_gb - safety_buffer_vram_gb
        logger.info(f"VRAM Details: Total={self.hw_info.gpu_vram_total_gb:.2f}GB, "
                    f"Est. KV Cache={estimated_kv_cache_vram_gb:.2f}GB (for {context_length_tokens} ctx), "
                    f"Safety Buffer={safety_buffer_vram_gb:.2f}GB. "
                    f"Available for weights={available_vram_for_model_weights_gb:.2f}GB.")

        if available_vram_for_model_weights_gb <= 0:
            logger.info("Not enough VRAM available for model weights after accounting for KV cache and safety buffer. Recommending 0 GPU layers.")
            return 0
        
        # 估算模型完全在GPU上时，权重部分大致占用的VRAM (与磁盘大小近似，但通常略大)
        # 这是一个粗略估计，实际占用可能因模型结构和llama.cpp加载方式而异
        # 假设模型在VRAM中的大小约等于其在磁盘上的大小
        
        # 计算可以卸载的层数比例
        # 如果可用VRAM大于模型大小，则可以卸载所有层
        if available_vram_for_model_weights_gb >= model_size_on_disk_gb:
            logger.info(f"Sufficient VRAM to offload all layers. Recommending all {model_total_layers} layers to GPU (-1 for llama.cpp).")
            return -1 # -1 通常表示卸载所有可能的层
        else:
            # 按比例计算可卸载的层数
            # proportion_of_model_can_fit = available_vram_for_model_weights_gb / model_size_on_disk_gb
            # recommended_layers = int(model_total_layers * proportion_of_model_can_fit)
            
            # 更保守的方式：如果不能完全放下，就优先CPU，或者只放很少一部分层
            # 这里我们采用一个简单的线性插值，但实际中可能需要更复杂的模型
            # 考虑到llama.cpp中层是不均匀的，且卸载部分层时CPU与GPU数据传输也有开销
            # 一个非常简化的策略：如果能放下超过一半，就尝试放下一半；如果连一半都放不下，放更少或不放
            if available_vram_for_model_weights_gb >= model_size_on_disk_gb / 2:
                recommended_layers = max(1, int(model_total_layers * 0.5)) # 尝试卸载一半
                logger.info(f"VRAM can fit about half the model. Recommending ~{recommended_layers} GPU layers.")
            elif available_vram_for_model_weights_gb >= model_size_on_disk_gb / 4:
                recommended_layers = max(1, int(model_total_layers * 0.25)) # 尝试卸载四分之一
                logger.info(f"VRAM can fit about a quarter of the model. Recommending ~{recommended_layers} GPU layers.")
            else:
                recommended_layers = 0 # VRAM非常紧张，建议CPU
                logger.info("VRAM very limited. Recommending 0 GPU layers.")

            # 确保不为0，除非真的完全放不下 (上面已经处理了available_vram_for_model_weights_gb <= 0的情况)
            # 并且不超过总层数
            return min(max(0, recommended_layers), model_total_layers)

    def recommend_concurrent_tasks(self, task_type: str = "cpu_bound_llm") -> int:
        """根据CPU核心数和任务类型推荐并发任务数"""
        if not self.hw_info:
            logger.warning("Hardware info not available, defaulting to 1 concurrent task.")
            return 1
        
        if task_type == "cpu_bound_llm":
            # 对于LLM CPU推理，即使是多核，单个推理任务也可能占满一个或多个核心
            # 过多并发可能导致严重的上下文切换和缓存竞争，反而降低效率
            # 保守起见，推荐1，或者物理核心数的一半，取较小者
            # 如果未来LLM服务本身能很好地利用多核进行单请求加速，则另当别论
            # 但目前我们假设一个请求主要由一个worker的单线程（或少量线程）处理
            # return 1 
            # 或者稍微激进一点，如果物理核心多：
            base_cores = self.hw_info.cpu_physical_cores
            if base_cores >= 8:
                return max(1, base_cores // 4)
            elif base_cores >= 4:
                return max(1, base_cores // 2)
            else:
                return 1
        elif task_type == "io_bound":
            # 对于IO密集型任务，可以使用更多的并发
            return self.hw_info.cpu_logical_cores * 2 # 示例值
        else: # 一般CPU密集型
            return self.hw_info.cpu_physical_cores

# --- main 用于测试 HardwareManager ---
if __name__ == "__main__":
    import os # 需要导入os才能在_detect_cpu_info的fallback中使用os.cpu_count()
    print("--- Testing HardwareManager ---")
    hw_manager = HardwareManager()
    hw_info = hw_manager.get_hardware_info()
    
    if hw_info:
        print("\n--- Detected Hardware ---")
        print(f"  CPU Logical Cores: {hw_info.cpu_logical_cores}")
        print(f"  CPU Physical Cores: {hw_info.cpu_physical_cores}")
        print(f"  Total System RAM: {hw_info.total_system_ram_gb:.2f} GB")
        if hw_info.gpu_available:
            print(f"  GPU Name: {hw_info.gpu_name}")
            print(f"  GPU VRAM Total: {hw_info.gpu_vram_total_gb:.2f} GB")
        else:
            print("  GPU: Not available or not detected.")
        
        print("\n--- Recommendations ---")
        # 假设一个1.7B Q8模型 (约1.8GB磁盘大小)，模型总共32层 (Qwen1.7B是28层，这里用32示意)
        # 上下文长度4096，每1k上下文KV缓存占用0.25GB VRAM (fp16时约0.23GB/1k，这里取个近似值)
        # 安全余量1.5GB VRAM
        model_layers = 28 # Qwen3-1.7B
        model_disk_size = 1.8 # GB
        ctx_len = 4096
        kv_per_1k_ctx = 0.25 
        vram_buffer = 1.5

        recommended_gpu_layers = hw_manager.recommend_llm_gpu_layers(
            model_total_layers=model_layers,
            model_size_on_disk_gb=model_disk_size,
            kv_cache_gb_per_1k_ctx=kv_per_1k_ctx,
            context_length_tokens=ctx_len,
            safety_buffer_vram_gb=vram_buffer
        )
        print(f"  Recommended LLM GPU Layers (for {model_disk_size}GB model, {model_layers} layers, {ctx_len} ctx): {recommended_gpu_layers}")
        
        recommended_llm_workers = hw_manager.recommend_concurrent_tasks(task_type="cpu_bound_llm")
        print(f"  Recommended Concurrent LLM Tasks (cpu_bound_llm): {recommended_llm_workers}")
        
        recommended_io_workers = hw_manager.recommend_concurrent_tasks(task_type="io_bound")
        print(f"  Recommended Concurrent I/O Tasks: {recommended_io_workers}")
    else:
        print("Failed to get hardware information.")