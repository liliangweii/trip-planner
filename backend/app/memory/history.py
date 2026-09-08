"""会话记忆：SQLChatMessageHistory（SQLite 默认）+ 按会话隔离。

对应设计文档 §6.4。记忆窗口裁剪：保留最近 10 轮，超限做摘要压缩。
"""
from langchain_community.chat_message_histories import SQLChatMessageHistory

from app.config import settings

MAX_ROUNDS = 10  # 保留最近 10 轮


def get_session_history(session_id: str) -> SQLChatMessageHistory:
    """按 session_id 获取/创建会话历史。

    /api/trip/plan 与 /api/chat 共用 session_id，
    实现"先聊需求 → 一键生成计划"的连续体验。
    """
    return SQLChatMessageHistory(
        session_id=session_id,
        # langchain-community 新版参数名为 connection（接受 SQLAlchemy URL 字符串）
        connection=settings.sql_history_url,
    )
