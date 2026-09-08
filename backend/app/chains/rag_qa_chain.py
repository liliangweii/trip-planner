"""RAG 问答链（P3 实现）：混合检索 → 上下文拼装 → 历史记忆 → LLM 流式 → 引用事件。

对应设计文档 §6.2：
- 带编号上下文 format_docs（[n] (标题) 正文），Prompt 硬约束"依据 [n] 回答并标注引用"
- 会话记忆：SQLChatMessageHistory(SQLite)，按 session_id 隔离，取最近 MAX_ROUNDS 轮
- 流式：SSE 逐 token 推送，结束后附带 citations 事件（chunk_id/source/snippet 溯源）

流式事件协议（与前端 SSE 解析约定）：
    ("token", str)      → 普通文本 token
    ("citations", list) → 引用列表 [{chunk_id, source, title, snippet}]
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import settings
from app.memory.history import get_session_history, MAX_ROUNDS
from app.rag.prompts import TRAVEL_QA_SYSTEM_PROMPT
from app.rag.retriever import get_retriever
from app.services.llm_service import get_llm

# 历史窗口：最近 10 轮（每轮 = 1 Human + 1 AI = 20 条消息），超限的旧消息不进 Prompt
_HISTORY_MESSAGE_CAP = MAX_ROUNDS * 2


def format_docs(docs: list[Document]) -> str:
    """带编号拼装检索上下文：[n] (标题) 正文。"""
    return "\n\n".join(
        f"[{i + 1}] ({d.metadata.get('title', '')}) {d.page_content}"
        for i, d in enumerate(docs)
    )


def _recent_history(session_id: str) -> list:
    """取最近对话轮次（保留尾巴），转成 LangChain 消息列表。"""
    history = get_session_history(session_id)
    messages = history.messages
    if len(messages) > _HISTORY_MESSAGE_CAP:
        messages = messages[-_HISTORY_MESSAGE_CAP:]
    return list(messages)


def _persist_turn(session_id: str, question: str, answer: str) -> None:
    """把本轮问答写入 SQLite 会话历史。"""
    history = get_session_history(session_id)
    history.add_message(HumanMessage(content=question))
    history.add_message(AIMessage(content=answer))


def _build_messages(session_id: str, question: str, context: str, docs: list[Document]) -> list:
    """组装 LLM 消息：系统约束 + 最近历史 + 编号上下文 + 当前问题。"""
    history = _recent_history(session_id)

    human = HumanMessage(
        content=(
            "检索到的攻略上下文：\n"
            f"{context}\n\n"
            f"用户问题：{question}"
        )
    )
    return [SystemMessage(content=TRAVEL_QA_SYSTEM_PROMPT), *history, human]


def _to_citations(docs: list[Document]) -> list[dict]:
    """把命中文档转成引用事件负载（chunk_id = Milvus pk，用于前端溯源）。"""
    out = []
    for d in docs:
        meta = d.metadata
        out.append(
            {
                "chunk_id": str(meta.get("pk", "")),
                "source": meta.get("source", ""),
                "title": meta.get("title", ""),
                "score": round(float(meta.get("score", 0.0)), 4),
                "snippet": d.page_content[:120],
            }
        )
    return out


async def stream_answer(session_id: str, question: str, city: str | None = None) -> AsyncIterator[tuple]:
    """流式回答：yield ("token", str) 与最后的 ("citations", list)。

    Args:
        session_id: 会话 ID（多轮记忆隔离键，/api/trip/plan 共用实现连续体验）。
        question: 用户问题。
        city: 可选城市过滤（检索时限定分区）。
    """
    docs: list[Document] = []
    try:
        docs = get_retriever().search(question, city=city)
    except Exception as exc:  # noqa: BLE001
        # 检索失败不阻断对话：空上下文进 Prompt，LLM 会如实说明"暂无相关知识"
        import loguru

        loguru.logger.warning("检索失败，降级为空上下文: {}", exc)

    context = format_docs(docs)
    messages = _build_messages(session_id, question, context, docs)
    llm = get_llm(streaming=True)

    answer_parts: list[str] = []
    try:
        async for chunk in llm.astream(messages):
            token = getattr(chunk, "content", "") or ""
            if token:
                answer_parts.append(token)
                yield ("token", token)
    finally:
        # 无论流是否中断都落库，避免丢轮次
        answer = "".join(answer_parts)
        if answer:
            _persist_turn(session_id, question, answer)
        # 引用事件在流尾随答案一起返回（前端据此渲染"参考攻略"）
        if docs:
            yield ("citations", _to_citations(docs[: settings.rerank_top_k]))
