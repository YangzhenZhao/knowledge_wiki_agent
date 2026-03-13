import os
from dotenv import load_dotenv

load_dotenv()

# Ollama 配置
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
# Embedding 模型（推荐使用 nomic-embed-text 或 mxbai-embed-large）
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ChromaDB 配置
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# 服务配置
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Admin 认证配置
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "your-secret-admin-key-change-me")

# 应用配置
APP_TITLE = os.getenv("APP_TITLE", "知识库问答 Agent")