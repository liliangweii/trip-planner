"""FastAPI 应用入口：CORS + 路由注册 + 统一异常处理。

对应设计文档 §2 网关层职责、§7 API 设计。
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import settings
from app.api.routes import chat, kb, search, trip


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化资源，关闭时释放。"""
    logger.info("TripRAG 后端启动中... collection={}", settings.milvus_collection)
    # TODO: P1 阶段在此预检 Milvus 连通性、初始化集合/索引
    yield
    logger.info("TripRAG 后端关闭。")


app = FastAPI(
    title="TripRAG 智游助手",
    description="基于 LangChain + Milvus 的 RAG 智能旅行规划平台",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS：允许前端 Vite 开发服务器跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由注册
app.include_router(trip.router, prefix="/api/trip", tags=["行程规划"])
app.include_router(chat.router, prefix="/api/chat", tags=["RAG 问答"])
app.include_router(kb.router, prefix="/api/kb", tags=["知识库管理"])
app.include_router(search.router, prefix="/api/search", tags=["检索调试"])


@app.get("/health", tags=["健康检查"])
async def health() -> JSONResponse:
    """健康检查：含 Milvus 连通性。

    Milvus 可达 → 200 {"status": "ok", "milvus": "connected"}
    Milvus 不可达 → 503 {"status": "degraded", "milvus": "unreachable"}
    """
    try:
        from app.rag.milvus_store import get_client

        get_client().list_collections()
        return JSONResponse(status_code=200, content={"status": "ok", "milvus": "connected"})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Milvus 健康检查失败: {}", exc)
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "milvus": "unreachable"},
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):  # noqa: ANN001
    """统一异常处理：网关层不泄露内部栈，仅返回结构化错误。"""
    logger.exception("未处理异常: {} {}", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "internal_error", "detail": str(exc)},
    )
