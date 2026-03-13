"""
RAG 模块 - 检索增强生成
结合向量检索和 LLM 生成回答（支持 Ollama 和 OpenAI）
"""
from typing import List
from config import (
    LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_BASE_URL,
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
)
from vector_store import vector_store

if LLM_PROVIDER == "openai":
    from openai import OpenAI


class RAGEngine:
    """RAG 引擎"""

    def __init__(self):
        self.provider = LLM_PROVIDER
        if self.provider == "openai":
            self.model = OPENAI_MODEL
            self.client = OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL
            )
        else:
            import ollama
            self.model = OLLAMA_MODEL
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

        # 4. 调用 LLM 生成回答
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content
        else:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response['message']['content']

        return {
            "answer": answer,
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
        if self.provider == "openai":
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            for chunk in self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            ):
                yield chunk['message']['content']


# 全局 RAG 引擎实例
rag_engine = RAGEngine()