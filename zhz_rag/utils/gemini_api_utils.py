# 文件: zhz_rag/utils/gemini_api_utils.py

import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging
from typing import Optional

# --- 加载 .env ---
# 这样即使在非Dagster环境中也能获取环境变量
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
dotenv_path = os.path.join(project_root, '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)

# --- 配置日志 ---
gemini_util_logger = logging.getLogger(__name__)

class GeminiAPIClient:
    """
    一个独立的、不依赖于Dagster的Gemini API客户端。
    """
    def __init__(self, api_key: str, proxy_url: Optional[str] = None):
        self._api_key = api_key
        self._proxy_url = proxy_url
        self._configure_client()
        gemini_util_logger.info("GeminiAPIClient initialized.")

    def _configure_client(self):
        """配置google-generativeai客户端，包括代理。"""
        try:
            if self._proxy_url:
                os.environ['https_proxy'] = self._proxy_url
                os.environ['http_proxy'] = self._proxy_url
                gemini_util_logger.info(f"Using proxy for Gemini API: {self._proxy_url}")
            
            genai.configure(api_key=self._api_key)
        except Exception as e:
            gemini_util_logger.error(f"Failed to configure Gemini client: {e}", exc_info=True)
            raise

    @classmethod
    def from_env(cls):
        """通过环境变量方便地创建实例。"""
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables.")
        
        proxy_url = os.getenv("LITELLM_PROXY_URL") # 复用这个代理配置
        return cls(api_key=api_key, proxy_url=proxy_url)

    def get_model(self, model_name: str = "gemini-1.5-flash-latest"):
        """获取一个配置好的GenerativeModel实例。"""
        try:
            return genai.GenerativeModel(model_name)
        except Exception as e:
            gemini_util_logger.error(f"Failed to get Gemini model '{model_name}': {e}", exc_info=True)
            return None