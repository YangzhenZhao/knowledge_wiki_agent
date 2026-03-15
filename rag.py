"""
RAG 模块 - 检索增强生成
结合向量检索和 LLM 生成回答（支持 Ollama 和 OpenAI）
支持 Function Calling 调用 Markdown Skills
"""
from typing import List
from config import (
    LLM_PROVIDER, OLLAMA_MODEL, OLLAMA_BASE_URL,
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
    ENABLE_FUNCTION_CALLING
)
from vector_store import vector_store
from skill_loader import skill_loader

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
        """基于知识库回答问题，支持 Function Calling"""
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
            system_prompt = """你是一个知识库助手。请根据以下知识库内容回答用户问题。
如果知识库中没有相关信息，请根据你的知识回答，但要说明这不是来自知识库。
如果用户的问题需要计算，请使用 calculator 工具。"""
            user_prompt = f"""=== 知识库内容 ===
{context}

=== 用户问题 ===
{question}"""
        else:
            system_prompt = "你是一个智能助手，请回答用户问题。如果需要计算，请使用 calculator 工具。"
            user_prompt = question

        # 4. 调用 LLM（支持 Function Calling）
        if self.provider == "openai":
            return self._query_openai_with_tools(system_prompt, user_prompt, question, search_results)
        else:
            # Ollama 暂不支持 function calling，走普通流程
            return self._query_ollama_simple(system_prompt, user_prompt, question, search_results)

    def _query_openai_with_tools(self, system_prompt: str, user_prompt: str, question: str, search_results: dict) -> dict:
        """OpenAI 带 Function Calling 的查询"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 检查是否启用 Function Calling
        if ENABLE_FUNCTION_CALLING:
            # 获取所有工具 schema
            tools = skill_loader.get_all_schemas()

            # 检测是否需要强制调用计算器
            tool_choice = "auto"
            needs_calc = self._needs_calculator(question)
            if needs_calc and skill_loader.get("calculator"):
                tool_choice = {"type": "function", "function": {"name": "calculator"}}

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice=tool_choice
                )
            except Exception as e:
                # 如果 API 不支持 tools 参数，回退到普通调用
                print(f"API 不支持 function calling，回退到普通模式: {e}")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
        else:
            # 不启用 Function Calling，直接调用
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )

        message = response.choices[0].message

        # 检查是否需要调用工具
        if ENABLE_FUNCTION_CALLING and message.tool_calls:
            # 执行工具调用
            tool_results = []
            for tool_call in message.tool_calls:
                skill_name = tool_call.function.name
                import json
                arguments = json.loads(tool_call.function.arguments)
                result = skill_loader.execute(skill_name, arguments)
                print(f"调用工具 {skill_name} 结果: {result}")
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": skill_name,
                    "content": result
                })

            # 将工具结果发回 LLM 生成最终回答
            messages.append(message)
            for tr in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_call_id"],
                    "content": tr["content"]
                })

            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            answer = final_response.choices[0].message.content
        else:
            answer = message.content

        # 处理 answer 为 None 的情况
        if answer is None:
            answer = "抱歉，无法生成回答。"

        return {
            "answer": answer,
            "sources": {
                "documents": search_results['documents'],
                "qa_pairs": search_results['qa_pairs']
            }
        }

    def _query_ollama_simple(self, system_prompt: str, user_prompt: str, question: str, search_results: dict) -> dict:
        """Ollama 简单查询（不支持 function calling）"""
        # 尝试检测是否需要计算器
        if self._needs_calculator(question):
            import re
            # 尝试提取数学表达式
            expr_match = re.search(r'[\d\+\-\*\/\(\)\.\^\s]+', question)
            if expr_match:
                expr = expr_match.group().strip()
                calc_result = skill_loader.execute("calculator", {"expression": expr})
                user_prompt = f"{user_prompt}\n\n（计算器结果：{calc_result}）"

        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        answer = response['message']['content']

        return {
            "answer": answer,
            "sources": {
                "documents": search_results['documents'],
                "qa_pairs": search_results['qa_pairs']
            }
        }

    def _needs_calculator(self, question: str) -> bool:
        """检测问题是否需要计算器"""
        keywords = ["计算", "等于", "多少", "+", "-", "*", "/", "乘", "除", "加", "减"]
        return any(kw in question for kw in keywords)

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
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                )
                has_content = False
                for chunk in stream:
                    # 检查是否有内容
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta and hasattr(delta, 'content') and delta.content:
                            has_content = True
                            yield delta.content
                        # 检查 finish_reason 是否为 content_filter
                        if chunk.choices[0].finish_reason == "content_filter":
                            raise Exception("内容被安全过滤，请修改问题后重试")
                    # 如果没有任何内容，可能是被过滤了
                if not has_content:
                    raise Exception("API 未返回任何内容，可能触发了内容过滤")
            except Exception as api_error:
                print(f"OpenAI API error in query_stream: {api_error}")
                raise api_error
        else:
            for chunk in self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            ):
                yield chunk['message']['content']


# 全局 RAG 引擎实例
rag_engine = RAGEngine()