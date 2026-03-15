"""
知识库问答 Agent - FastAPI 主程序
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import json
import traceback
import re
import asyncio
from datetime import datetime

from models import DocumentUpload, QAUpload, QueryRequest, QueryResponse, WebUrlUpload
from vector_store import vector_store
from rag import rag_engine
from skill_loader import skill_loader
from config import (
    HOST, PORT, LLM_PROVIDER,
    OLLAMA_MODEL, OLLAMA_EMBED_MODEL,
    OPENAI_MODEL,
    ADMIN_API_KEY, APP_TITLE,
    get_llm_model
)

# API 超时时间（秒）
API_TIMEOUT = 90

app = FastAPI(
    title="知识库问答 Agent",
    description="基于 Ollama + ChromaDB 的知识库问答系统",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")


# ==================== Admin 认证 ====================

async def verify_admin(api_key: str = Header(None, alias="X-API-Key")):
    """验证 Admin 权限"""
    if api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return True


def extract_error_message(error: Exception) -> str:
    """从 API 错误中提取友好的错误消息"""
    error_str = str(error)
    # 尝试从 GLM API 错误中提取 message
    # 格式: Error code: 400 - {'error': {'code': '1301', 'message': '...'}}
    try:
        # 尝试匹配 JSON 部分（支持单引号的 Python dict 格式）
        json_match = re.search(r"\{.*\}", error_str)
        if json_match:
            # 尝试作为 JSON 解析（双引号）
            try:
                error_data = json.loads(json_match.group())
            except json.JSONDecodeError:
                # 如果失败，尝试用 ast 解析 Python 字典格式（单引号）
                import ast
                error_data = ast.literal_eval(json_match.group())

            if 'error' in error_data and isinstance(error_data['error'], dict):
                return error_data['error'].get('message', error_str)
    except (json.JSONDecodeError, AttributeError, ValueError, SyntaxError):
        pass
    return error_str


# ==================== 页面路由 ====================

@app.get("/", response_class=HTMLResponse)
async def index():
    """主页"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


# ==================== 文档 API ====================

@app.post("/api/documents")
async def upload_document(document: DocumentUpload, _: bool = Depends(verify_admin)):
    """上传文档"""
    try:
        doc_id = vector_store.add_document(
            title=document.title,
            content=document.content
        )
        return {"success": True, "id": doc_id, "message": "文档上传成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/file")
async def upload_document_file(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    _: bool = Depends(verify_admin)
):
    """上传文档文件"""
    try:
        content = await file.read()
        text = content.decode('utf-8')
        doc_title = title or file.filename

        doc_id = vector_store.add_document(
            title=doc_title,
            content=text
        )
        return {"success": True, "id": doc_id, "message": "文档上传成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/url")
async def upload_document_url(web_url: WebUrlUpload, _: bool = Depends(verify_admin)):
    """通过URL抓取网页内容并存储"""
    import httpx
    from bs4 import BeautifulSoup

    try:
        # 抓取网页
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                web_url.url,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; KnowledgeWikiBot/1.0)"}
            )
            response.raise_for_status()

        # 解析网页内容
        soup = BeautifulSoup(response.text, 'html.parser')

        # 移除脚本和样式
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # 提取文本
        text = soup.get_text(separator='\n', strip=True)

        # 清理多余的空行
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        content = '\n'.join(lines)

        if not content:
            raise HTTPException(status_code=400, detail="无法从网页提取有效内容")

        # 使用提供的标题或从网页获取
        title = web_url.title
        if not title:
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else web_url.url

        # 存储到向量数据库
        doc_id = vector_store.add_document(
            title=title,
            content=content
        )

        return {
            "success": True,
            "id": doc_id,
            "message": "网页内容存储成功",
            "title": title,
            "content_length": len(content)
        }
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"网页请求失败: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def list_documents():
    """列出所有文档"""
    try:
        docs = vector_store.list_documents()
        return {"success": True, "documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str, _: bool = Depends(verify_admin)):
    """删除文档"""
    try:
        success = vector_store.delete_document(doc_id)
        if success:
            return {"success": True, "message": "文档删除成功"}
        else:
            raise HTTPException(status_code=404, detail="文档不存在")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 问答 API ====================

@app.post("/api/qa")
async def upload_qa(qa: QAUpload, _: bool = Depends(verify_admin)):
    """上传问答对"""
    try:
        qa_id = vector_store.add_qa_pair(
            question=qa.question,
            answer=qa.answer,
            tags=qa.tags
        )
        return {"success": True, "id": qa_id, "message": "问答对上传成功"}
    except Exception as e:
        print(f"Error in upload_qa: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/qa")
async def list_qa():
    """列出所有问答对"""
    try:
        qa_pairs = vector_store.list_qa_pairs()
        return {"success": True, "qa_pairs": qa_pairs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/qa/{qa_id}")
async def delete_qa(qa_id: str, _: bool = Depends(verify_admin)):
    """删除问答对"""
    try:
        success = vector_store.delete_qa_pair(qa_id)
        if success:
            return {"success": True, "message": "问答对删除成功"}
        else:
            raise HTTPException(status_code=404, detail="问答对不存在")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 查询 API ====================

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest, api_key: str = Header(None, alias="X-API-Key")):
    """查询知识库"""
    try:
        # 使用 asyncio.wait_for 添加超时控制
        result = await asyncio.wait_for(
            asyncio.to_thread(
                rag_engine.query,
                question=request.question,
                top_k=request.top_k
            ),
            timeout=API_TIMEOUT
        )

        # 只有 admin 才返回来源
        sources = []
        if api_key == ADMIN_API_KEY:
            for doc in result['sources']['documents']:
                sources.append({
                    "type": "document",
                    "title": doc['title'],
                    "content": doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'],
                    "score": round(doc['score'], 3)
                })
            for qa in result['sources']['qa_pairs']:
                sources.append({
                    "type": "qa",
                    "question": qa['question'],
                    "answer": qa['answer'],
                    "score": round(qa['score'], 3)
                })

        return QueryResponse(
            answer=result['answer'],
            sources=sources
        )
    except asyncio.TimeoutError:
        print(f"Timeout in /api/query after {API_TIMEOUT}s")
        raise HTTPException(status_code=504, detail=f"请求超时，请稍后重试（超时时间: {API_TIMEOUT}秒）")
    except Exception as e:
        print(f"Error in /api/query: {e}")
        print(traceback.format_exc())
        error_msg = extract_error_message(e)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    """流式查询知识库"""

    def generate():
        import time
        start_time = time.time()

        try:
            print(f"[Stream] 开始处理问题: {request.question[:50]}...")
            for chunk in rag_engine.query_stream(
                question=request.question,
                top_k=request.top_k
            ):
                # 检查是否超时
                elapsed = time.time() - start_time
                if elapsed > API_TIMEOUT:
                    print(f"[Stream] 超时，已运行 {elapsed:.1f}秒")
                    yield f"data: {json.dumps({'error': f'请求超时，请稍后重试（超时时间: {API_TIMEOUT}秒）', 'done': True})}\n\n"
                    return

                yield f"data: {json.dumps({'content': chunk})}\n\n"

            print("[Stream] 成功完成")
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            print(f"Error in /api/query/stream: {e}")
            print(traceback.format_exc())
            error_msg = extract_error_message(e)
            print(f"[Stream] 发送错误: {error_msg}")
            yield f"data: {json.dumps({'error': error_msg, 'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


# ==================== 系统 API ====================

@app.get("/api/status")
async def get_status():
    """获取系统状态"""
    return {
        "success": True,
        "provider": LLM_PROVIDER,
        "model": get_llm_model(),
        "embed_model": OLLAMA_EMBED_MODEL,
        "document_count": len(vector_store.list_documents()),
        "qa_count": len(vector_store.list_qa_pairs()),
        "app_title": APP_TITLE,
        "skills": [skill.name for skill in skill_loader.list_all()]
    }


@app.get("/api/skills")
async def list_skills():
    """列出所有可用技能"""
    skills = []
    for skill in skill_loader.list_all():
        skills.append({
            "name": skill.name,
            "description": skill.description,
            "trigger_words": skill.trigger_words,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "enum": p.enum if p.enum else None
                }
                for p in skill.parameters
            ],
            "examples": skill.examples,
            "file": skill.file_path
        })
    return {"success": True, "skills": skills}


@app.post("/api/skills/{skill_name}/execute")
async def execute_skill(skill_name: str, arguments: dict):
    """直接执行指定技能"""
    result = skill_loader.execute(skill_name, arguments)
    return {"success": True, "result": result}


@app.post("/api/skills/reload")
async def reload_skills(_: bool = Depends(verify_admin)):
    """重新加载所有技能（热重载）"""
    skill_loader.reload()
    return {"success": True, "message": "技能已重新加载", "count": len(skill_loader.list_all())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)