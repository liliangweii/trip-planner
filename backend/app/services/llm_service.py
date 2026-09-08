"""LLM 服务：OpenAI 兼容接口（DeepSeek / Qwen / GPT 任一）。

对应设计文档 §6.1。
"""
from langchain_openai import ChatOpenAI

from app.config import settings


def get_llm(streaming: bool = False) -> ChatOpenAI:
    """获取 ChatOpenAI 实例（OpenAI 兼容接口）。

    Args:
        streaming: 是否开启流式输出（/api/chat SSE 时为 True）。
    """
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0.3,
        streaming=streaming,
    )
