# zhz_agent/utils/hardware_manager.py
import os
import psutil
import torch
import logging
from typing import Optional, Dict, Any

# 配置日志记录器
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)

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
            if physical_cores is None:
                logger.warning("Could not determine physical core count, using logical core count as fallback.")
                physical_cores = logical_cores
            logger.info(f"CPU Info: Logical cores={logical_cores}, Physical cores={physical_cores}")
            return {"cpu_logical_cores": logical_cores, "cpu_physical_cores": physical_cores}
        except Exception as e:
            logger.error(f"Error detecting CPU info: {e}", exc_info=True)
            return {"cpu_logical_cores": 1, "cpu_physical_cores": 1}

    def _detect_memory_info(self) -> Dict[str, Any]:
        """检测内存信息"""
        try:
            virtual_mem = psutil.virtual_memory()
            total_ram_gb = virtual_mem.total / (1024 ** 3)
            logger.info(f"Memory Info: Total RAM={total_ram_gb:.2f} GB")
            return {"total_system_ram_gb": total_ram_gb}
        except Exception as e:
            logger.error(f"Error detecting memory info: {e}", exc_info=True)
            return {"total_system_ram_gb": 0.0}

    def _detect_gpu_info(self) -> Dict[str, Any]:
        """检测GPU信息 (目前仅NVIDIA GPU通过torch.cuda)"""
        gpu_details = {"gpu_available": False, "gpu_name": None, "gpu_vram_total_gb": 0.0}
        try:
            if torch.cuda.is_available():
                gpu_details["gpu_available"] = True
                device_id = 0
                gpu_name = torch.cuda.get_device_name(device_id)
                props = torch.cuda.get_device_properties(device_id)
                vram_total_gb = props.total_memory / (1024 ** 3)
                gpu_details["gpu_name"] = gpu_name
                gpu_details["gpu_vram_total_gb"] = vram_total_gb
                logger.info(f"NVIDIA GPU Detected: Name='{gpu_name}', VRAM={vram_total_gb:.2f} GB")
            else:
                logger.info("torch.cuda.is_available() returned False. No compatible NVIDIA GPU detected.")
        except Exception as e:
            logger.error(f"Error detecting NVIDIA GPU info: {e}", exc_info=True)
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
        """根据可用VRAM推荐LLM应卸载到GPU的层数。"""
        if not self.hw_info or not self.hw_info.gpu_available or self.hw_info.gpu_vram_total_gb == 0:
            logger.info("GPU not available, recommending 0 GPU layers (CPU only).")
            return 0

        estimated_kv_cache_vram_gb = (context_length_tokens / 1000) * kv_cache_gb_per_1k_ctx
        available_vram_for_model_weights_gb = self.hw_info.gpu_vram_total_gb - estimated_kv_cache_vram_gb - safety_buffer_vram_gb
        logger.info(f"VRAM Details: Total={self.hw_info.gpu_vram_total_gb:.2f}GB, Est. KV Cache={estimated_kv_cache_vram_gb:.2f}GB, Safety Buffer={safety_buffer_vram_gb:.2f}GB. Available for weights={available_vram_for_model_weights_gb:.2f}GB.")

        if available_vram_for_model_weights_gb <= 0:
            return 0
        
        if available_vram_for_model_weights_gb >= model_size_on_disk_gb:
            logger.info(f"Sufficient VRAM to offload all {model_total_layers} layers.")
            return -1
        else:
            proportion_of_model_can_fit = available_vram_for_model_weights_gb / model_size_on_disk_gb
            recommended_layers = int(model_total_layers * proportion_of_model_can_fit)
            logger.info(f"VRAM can fit ~{proportion_of_model_can_fit:.0%}. Recommending {recommended_layers} GPU layers.")
            return min(max(0, recommended_layers), model_total_layers)

    def recommend_concurrent_tasks(self, task_type: str = "cpu_bound_llm") -> int:
        """根据CPU核心数和任务类型推荐并发任务数"""
        if not self.hw_info:
            return 1
        if task_type == "cpu_bound_llm":
            return max(1, self.hw_info.cpu_physical_cores // 2)
        elif task_type == "io_bound":
            return self.hw_info.cpu_logical_cores * 2
        else:
            return self.hw_info.cpu_physical_cores

# --- main 用于测试 HardwareManager ---
if __name__ == "__main__":
    import os # 需要导入os才能在_detect_cpu_info的fallback中使用os.cpu_count()
    print("--- Testing HardwareManager ---")
    hw_manager = HardwareManager()
    hw_info = hw_manager.get_hardware_info()
    
    if hw_info:
        print("\n--- Detected Hardware ---")
        print(f"   CPU Logical Cores: {hw_info.cpu_logical_cores}")
        print(f"   CPU Physical Cores: {hw_info.cpu_physical_cores}")
        print(f"   Total System RAM: {hw_info.total_system_ram_gb:.2f} GB")
        if hw_info.gpu_available:
            print(f"   GPU Name: {hw_info.gpu_name}")
            print(f"   GPU VRAM Total: {hw_info.gpu_vram_total_gb:.2f} GB")
        else:
            print("   GPU: Not available or not detected.")
        
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
        print(f"   Recommended LLM GPU Layers (for {model_disk_size}GB model, {model_layers} layers, {ctx_len} ctx): {recommended_gpu_layers}")
        
        recommended_llm_workers = hw_manager.recommend_concurrent_tasks(task_type="cpu_bound_llm")
        print(f"   Recommended Concurrent LLM Tasks (cpu_bound_llm): {recommended_llm_workers}")
        
        recommended_io_workers = hw_manager.recommend_concurrent_tasks(task_type="io_bound")
        print(f"   Recommended Concurrent I/O Tasks: {recommended_io_workers}")

    else:
        print("Failed to get hardware information.")
