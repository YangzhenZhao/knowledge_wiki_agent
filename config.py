import os
from dotenv import load_dotenv

load_dotenv(override=True)

# 模型提供者配置: "ollama" 或 "openai"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Embedding 提供者配置: "ollama" 或 "openai"
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "ollama")

# 是否启用 Function Calling（某些 API 如 GLM 可能不支持）
ENABLE_FUNCTION_CALLING = os.getenv("ENABLE_FUNCTION_CALLING", "false").lower() == "true"

# Ollama 配置
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
# Embedding 模型（推荐使用 nomic-embed-text 或 mxbai-embed-large）
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# OpenAI API 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# OpenAI Embedding 模型（使用 GLM 时可设置为 embedding-3）
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# ChromaDB 配置
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# 服务配置
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Admin 认证配置
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "your-secret-admin-key-change-me")

# 应用配置
APP_TITLE = os.getenv("APP_TITLE", "知识库问答 Agent")


def get_llm_model():
    """获取当前配置的 LLM 模型名称"""
    if LLM_PROVIDER == "openai":
        return OPENAI_MODEL
    return OLLAMA_MODEL