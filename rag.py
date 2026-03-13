"""
RAG 模块 - 检索增强生成
结合向量检索和 Ollama 生成回答
"""
import ollama
from typing import List
from config import OLLAMA_MODEL, OLLAMA_BASE_URL
from vector_store import vector_store


class RAGEngine:
    """RAG 引擎"""

    def __init__(self):
        self.model = OLLAMA_MODEL
        # 配置 Ollama 客户端
        self.client = ollama.Client(host=OLLAMA_BASE_URL)

    def query(self, question: str, top_k: int = 3) -> dict:
        """基于知识库回答问题"""
        # 1. 检索相关内容
        search_results = vector_store.search(question, top_k=top_k)

        # 2. 构建上下文
        context_parts = []

        # 添加相关文档
        for doc in search_results['documents']:
            context_parts.append(f"【文档: {doc['title']}】\n{doc['content']}")

        # 添加相关问答
        for qa in search_results['qa_pairs']:
            context_parts.append(f"【已知问答】\n问: {qa['question']}\n答: {qa['answer']}")

        context = "\n\n".join(context_parts)

        # 3. 构建 prompt
        if context:
            prompt = f"""你是一个知识库助手。请根据以下知识库内容回答用户问题。
如果知识库中没有相关信息，请根据你的知识回答，但要说明这不是来自知识库。

=== 知识库内容 ===
{context}

=== 用户问题 ===
{question}

请给出准确、有帮助的回答："""
        else:
            prompt = f"""请回答以下问题：
{question}"""

        # 4. 调用 Ollama 生成回答
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "answer": response['message']['content'],
            "sources": {
                "documents": search_results['documents'],
                "qa_pairs": search_results['qa_pairs']
            }
        }

    def query_stream(self, question: str, top_k: int = 3):
        """流式输出回答"""
        search_results = vector_store.search(question, top_k=top_k)

        context_parts = []
        for doc in search_results['documents']:
            context_parts.append(f"【文档: {doc['title']}】\n{doc['content']}")
        for qa in search_results['qa_pairs']:
            context_parts.append(f"【已知问答】\n问: {qa['question']}\n答: {qa['answer']}")

        context = "\n\n".join(context_parts)

        if context:
            prompt = f"""你是一个知识库助手。请根据以下知识库内容回答用户问题。
如果知识库中没有相关信息，请根据你的知识回答，但要说明这不是来自知识库。

=== 知识库内容 ===
{context}

=== 用户问题 ===
{question}

请给出准确、有帮助的回答："""
        else:
            prompt = f"请回答以下问题：\n{question}"

        # 流式输出
        for chunk in self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        ):
            yield chunk['message']['content']


# 全局 RAG 引擎实例
rag_engine = RAGEngine()