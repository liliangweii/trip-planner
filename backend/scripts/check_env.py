"""配置层自检：验证 .env 加载与字段映射。

运行方式（必须从 backend/ 目录运行）：
    cd backend
    python scripts/check_env.py

思考题答案（写在注释里防止遗忘）：
pydantic-settings 的 env_file=".env" 是相对【当前工作目录】解析的，
不是相对文件位置。从项目根目录运行时读不到 backend/.env，
所以必须先 cd 到 backend/ 再运行本脚本。
"""
from __future__ import annotations

import sys
from pathlib import Path

# 让脚本可直接运行：把 backend 目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings


def mask(key: str) -> str:
    """敏感 Key 打码：只露前 6 位。"""
    if not key:
        return "(未配置)"
    return f"{key[:6]}***" if len(key) > 8 else "***"


def main() -> None:
    rows: list[tuple[str, str]] = [
        ("LLM_API_KEY", mask(settings.llm_api_key)),
        ("LLM_BASE_URL", settings.llm_base_url),
        ("LLM_MODEL", settings.llm_model),
        ("MILVUS_URI", settings.milvus_uri),
        ("MILVUS_COLLECTION", settings.milvus_collection),
        ("EMBEDDING_PROVIDER", settings.embedding_provider),
        ("EMBEDDING_API_BASE", settings.embedding_api_base),
        ("EMBEDDING_API_KEY", mask(settings.embedding_api_key)),
        ("EMBEDDING_MODEL", settings.embedding_model),
        ("AMAP_API_KEY", mask(settings.amap_api_key)),
        ("SQL_HISTORY_URL", settings.sql_history_url),
        ("RETRIEVAL_TOP_K / RERANK_TOP_K", f"{settings.retrieval_top_k} / {settings.rerank_top_k}"),
        ("CORS_ORIGINS", str(settings.cors_origins)),
    ]

    width = max(len(k) for k, _ in rows)
    print("=== TripRAG 配置自检 ===")
    for k, v in rows:
        print(f"{k:<{width}} = {v}")
    print("========================")
    print("工作目录:", Path.cwd())


if __name__ == "__main__":
    main()
