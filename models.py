from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class DocumentUpload(BaseModel):
    """文档上传模型"""
    title: str
    content: str


class QAUpload(BaseModel):
    """问答上传模型"""
    question: str
    answer: str
    tags: Optional[List[str]] = []


class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str
    top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    """查询响应模型"""
    answer: str
    sources: List[dict]
    timestamp: datetime = datetime.now()


class WebUrlUpload(BaseModel):
    """网页URL上传模型"""
    url: str
    title: Optional[str] = None


class DocumentResponse(BaseModel):
    """文档响应模型"""
    id: str
    title: str
    content: str
    created_at: datetime


class QAResponse(BaseModel):
    """问答响应模型"""
    id: str
    question: str
    answer: str
    tags: List[str]
    created_at: datetime