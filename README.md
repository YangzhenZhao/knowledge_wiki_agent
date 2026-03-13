# 知识库问答 Agent

基于 **Ollama + ChromaDB** 的本地知识库问答系统。

## 功能特点

- 上传文本文档或文件（.txt, .md）
- 从网页 URL 抓取内容并存储
- 添加自定义问答对
- 基于向量检索的智能问答
- 流式输出回答
- 简洁的 Web 界面

## 技术栈

- **后端**: FastAPI
- **向量数据库**: ChromaDB（轻量、本地运行）
- **LLM**: Ollama（本地模型）
- **前端**: 原生 HTML/CSS/JavaScript

## 快速开始

### 1. 安装 Ollama 并下载模型

```bash
# 安装 Ollama（macOS）
brew install ollama

# 启动 Ollama 服务
ollama serve

# 下载对话模型（推荐 qwen2.5:7b 或其他中文模型）
ollama pull qwen2.5:7b

# 下载 Embedding 模型（用于向量检索）
ollama pull nomic-embed-text
```

### 2. 安装 Python 依赖

```bash
cd knowledge_wiki_agent
pip install -r requirements.txt
```

### 3. 配置环境变量（可选）

```bash
cp .env.example .env
# 编辑 .env 文件，修改模型名称、Admin 密钥等配置
```

### 4. 启动服务

```bash
python main.py
```

服务启动后访问: http://localhost:8000

## 权限说明

系统区分 **Admin** 和 **普通用户** 权限：

| 功能 | Admin | 普通用户 |
|------|-------|----------|
| 上传文档/文件/URL | ✓ | ✗ |
| 添加/删除问答对 | ✓ | ✗ |
| 删除文档 | ✓ | ✗ |
| 查询知识库 | ✓ | ✓ |
| 查看文档/问答列表 | ✓ | ✓ |

### Admin 访问方式

在 `.env` 中设置 `ADMIN_API_KEY`，然后通过带 token 的 URL 访问：

```
http://localhost:8000?token=your-admin-key
```

或在 API 请求中添加 Header：

```bash
curl -X POST http://localhost:8000/api/documents \
  -H "X-API-Key: your-admin-key" \
  -H "Content-Type: application/json" \
  -d '{"title": "测试", "content": "内容"}'
```

## 使用说明

### 上传知识

1. **文档管理** - 上传文本文档或直接粘贴内容
2. **问答管理** - 添加问题和答案对，可添加标签

### 问答

在问答页面输入问题，系统会：
1. 从知识库中检索相关内容
2. 将相关内容作为上下文发送给 Ollama
3. 流式输出回答

## API 接口

| 方法 | 路径 | 说明 | 权限 |
|------|------|------|------|
| POST | `/api/documents` | 上传文档（JSON） | Admin |
| POST | `/api/documents/file` | 上传文档（文件） | Admin |
| POST | `/api/documents/url` | 从网页URL抓取并存储 | Admin |
| GET | `/api/documents` | 获取文档列表 | 公开 |
| DELETE | `/api/documents/{id}` | 删除文档 | Admin |
| POST | `/api/qa` | 添加问答对 | Admin |
| GET | `/api/qa` | 获取问答列表 | 公开 |
| DELETE | `/api/qa/{id}` | 删除问答 | Admin |
| POST | `/api/query` | 问答查询 | 公开 |
| POST | `/api/query/stream` | 流式问答查询 | 公开 |
| GET | `/api/status` | 系统状态 | 公开 |

## 项目结构

```
knowledge_wiki_agent/
├── main.py           # FastAPI 主程序
├── config.py         # 配置文件
├── models.py         # 数据模型
├── vector_store.py   # 向量数据库管理
├── rag.py            # RAG 引擎
├── requirements.txt  # 依赖
├── .env.example      # 环境变量示例
└── static/
    └── index.html    # 前端页面
```