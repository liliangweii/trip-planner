"""POST /api/chat —— RAG 行程问答（SSE 流式）。

对应设计文档 §6.2、§7。流式事件协议：
    data: <token 文本>
    data: {"type": "citations", "items": [...]}     ← 答案结束后的引用溯源
    data: [DONE]

请求体：{"session_id": "...", "question": "..."}；city 可选过滤。
"""
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

router = APIRouter()


class ChatRequest(BaseModel):
    """问答请求体。"""

    session_id: str = Field(..., description="会话 ID，用于隔离多轮记忆")
    question: str = Field(..., description="用户问题")
    city: str | None = Field(None, description="可选：限定检索城市")


@router.post("")
async def chat(req: ChatRequest) -> StreamingResponse:
    """RAG 多轮问答，SSE 流式返回 token + 引用溯源事件。"""
    from app.chains.rag_qa_chain import stream_answer

    async def event_generator():
        try:
            async for kind, payload in stream_answer(req.session_id, req.question, city=req.city):
                if kind == "token":
                    # 逐 token 推送（裸文本行，前端累加渲染）
                    yield f"data: {payload}\n\n"
                elif kind == "citations":
                    event = json.dumps(
                        {"type": "citations", "items": payload},
                        ensure_ascii=False,
                    )
                    yield f"data: {event}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("问答流式异常: {}", exc)
            msg = str(exc)
            # 402 = 余额不足：给用户可操作提示，而非裸抛供应商错误
            if "402" in msg or "Insufficient Balance" in msg:
                msg = "LLM 账户余额不足（402），请到供应商平台充值后重试。"
            yield f"data: [ERROR] {msg}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
