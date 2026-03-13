"""
向量数据库管理模块
使用 ChromaDB 存储和检索文档
Embedding 使用 Ollama
"""
# 修复旧版 Linux 系统的 sqlite3 版本问题
# ChromaDB 需要 sqlite3 >= 3.35.0
import sys
if sys.platform != 'darwin':  # 非 macOS 系统才需要替换
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.config import Settings
from typing import List, Optional
import uuid
from datetime import datetime
import traceback
import ollama
from openai import OpenAI

from config import (
    CHROMA_PERSIST_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    EMBED_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_EMBED_MODEL
)


class VectorStore:
    """向量存储管理器"""

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        # Ollama 客户端，指定远程 host
        self.ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
        # OpenAI 客户端
        self.openai_client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL
        )
        # 文档集合
        self.documents = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "知识文档库"}
        )
        # 问答集合
        self.qa_pairs = self.client.get_or_create_collection(
            name="qa_pairs",
            metadata={"description": "问答对库"}
        )

    def _get_embedding(self, text: str) -> List[float]:
        """生成文本嵌入向量，支持 Ollama 和 OpenAI 两种提供者"""
        try:
            if EMBED_PROVIDER == "openai":
                return self._get_openai_embedding(text)
            else:
                return self._get_ollama_embedding(text)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            print(traceback.format_exc())
            provider = "OpenAI" if EMBED_PROVIDER == "openai" else "Ollama"
            embed_model = OPENAI_EMBED_MODEL if EMBED_PROVIDER == "openai" else OLLAMA_EMBED_MODEL
            raise Exception(
                f"生成嵌入向量失败: {str(e)}\n"
                f"当前使用: {provider} ({embed_model})\n"
                f"请检查配置是否正确。"
            )

    def _get_ollama_embedding(self, text: str) -> List[float]:
        """使用 Ollama 生成文本嵌入向量"""
        response = self.ollama_client.embeddings(
            model=OLLAMA_EMBED_MODEL,
            prompt=text
        )
        return response['embedding']

    def _get_openai_embedding(self, text: str) -> List[float]:
        """使用 OpenAI API 生成文本嵌入向量（支持 GLM 等兼容 API）"""
        response = self.openai_client.embeddings.create(
            model=OPENAI_EMBED_MODEL,
            input=text
        )
        return response.data[0].embedding

    def add_document(self, title: str, content: str) -> str:
        """添加文档到向量库"""
        doc_id = str(uuid.uuid4())

        # 分块处理长文档
        chunks = self._split_text(content, chunk_size=500, overlap=50)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            embedding = self._get_embedding(chunk)

            self.documents.add(
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[{
                    "doc_id": doc_id,
                    "title": title,
                    "chunk_index": i,
                    "content": chunk,
                    "created_at": datetime.now().isoformat()
                }],
                documents=[chunk]
            )

        return doc_id

    def add_qa_pair(self, question: str, answer: str, tags: List[str] = None) -> str:
        """添加问答对到向量库"""
        qa_id = str(uuid.uuid4())
        embedding = self._get_embedding(question)

        self.qa_pairs.add(
            ids=[qa_id],
            embeddings=[embedding],
            metadatas=[{
                "question": question,
                "answer": answer,
                "tags": ",".join(tags or []),
                "created_at": datetime.now().isoformat()
            }],
            documents=[question]
        )

        return qa_id

    def search(self, query: str, top_k: int = 3) -> dict:
        """搜索相关内容"""
        query_embedding = self._get_embedding(query)

        # 搜索文档
        doc_results = self.documents.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # 搜索问答
        qa_results = self.qa_pairs.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return {
            "documents": self._format_doc_results(doc_results),
            "qa_pairs": self._format_qa_results(qa_results)
        }

    def _split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """将长文本分割成小块"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap

        return chunks

    def _format_doc_results(self, results: dict) -> List[dict]:
        """格式化文档搜索结果"""
        if not results['metadatas'] or not results['metadatas'][0]:
            return []

        return [
            {
                "id": meta.get("doc_id"),
                "title": meta.get("title"),
                "content": meta.get("content"),
                "score": 1 - (results['distances'][0][i] if results.get('distances') else 0)
            }
            for i, meta in enumerate(results['metadatas'][0])
        ]

    def _format_qa_results(self, results: dict) -> List[dict]:
        """格式化问答搜索结果"""
        if not results['metadatas'] or not results['metadatas'][0]:
            return []

        return [
            {
                "question": meta.get("question"),
                "answer": meta.get("answer"),
                "tags": meta.get("tags", "").split(",") if meta.get("tags") else [],
                "score": 1 - (results['distances'][0][i] if results.get('distances') else 0)
            }
            for i, meta in enumerate(results['metadatas'][0])
        ]

    def list_documents(self) -> List[dict]:
        """列出所有文档"""
        results = self.documents.get()
        if not results['metadatas']:
            return []

        # 按 doc_id 去重
        seen = set()
        docs = []
        for meta in results['metadatas']:
            doc_id = meta.get('doc_id')
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                docs.append({
                    "id": doc_id,
                    "title": meta.get('title'),
                    "created_at": meta.get('created_at')
                })
        return docs

    def list_qa_pairs(self) -> List[dict]:
        """列出所有问答对"""
        results = self.qa_pairs.get()
        if not results['metadatas']:
            return []

        return [
            {
                "id": results['ids'][i],
                "question": meta.get('question'),
                "answer": meta.get('answer'),
                "tags": meta.get('tags', "").split(",") if meta.get('tags') else [],
                "created_at": meta.get('created_at')
            }
            for i, meta in enumerate(results['metadatas'])
        ]

    def delete_document(self, doc_id: str) -> bool:
        """删除文档"""
        # 获取所有相关 chunk
        results = self.documents.get()
        chunk_ids = [
            results['ids'][i]
            for i, meta in enumerate(results['metadatas'])
            if meta.get('doc_id') == doc_id
        ]
        if chunk_ids:
            self.documents.delete(ids=chunk_ids)
            return True
        return False

    def delete_qa_pair(self, qa_id: str) -> bool:
        """删除问答对"""
        try:
            self.qa_pairs.delete(ids=[qa_id])
            return True
        except:
            return False


# 全局向量存储实例
vector_store = VectorStore()