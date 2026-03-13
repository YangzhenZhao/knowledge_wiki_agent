"""
知识库问答 Agent - FastAPI 主程序
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import json
import traceback
from datetime import datetime

from models import DocumentUpload, QAUpload, QueryRequest, QueryResponse, WebUrlUpload
from vector_store import vector_store
from rag import rag_engine
from config import HOST, PORT, OLLAMA_MODEL, OLLAMA_EMBED_MODEL

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


# ==================== 页面路由 ====================

@app.get("/", response_class=HTMLResponse)
async def index():
    """主页"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


# ==================== 文档 API ====================

@app.post("/api/documents")
async def upload_document(document: DocumentUpload):
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
    title: Optional[str] = Form(None)
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
async def upload_document_url(web_url: WebUrlUpload):
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
async def delete_document(doc_id: str):
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
async def upload_qa(qa: QAUpload):
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
async def delete_qa(qa_id: str):
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
async def query(request: QueryRequest):
    """查询知识库"""
    try:
        result = rag_engine.query(
            question=request.question,
            top_k=request.top_k
        )

        # 格式化来源
        sources = []
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    """流式查询知识库"""

    def generate():
        try:
            for chunk in rag_engine.query_stream(
                question=request.question,
                top_k=request.top_k
            ):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

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
        "model": OLLAMA_MODEL,
        "embed_model": OLLAMA_EMBED_MODEL,
        "document_count": len(vector_store.list_documents()),
        "qa_count": len(vector_store.list_qa_pairs())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)