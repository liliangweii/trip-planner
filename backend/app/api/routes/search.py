"""GET /api/search —— 裸检索调试端点（检索透镜）。

对应设计文档 §7。
"""
from fastapi import APIRouter, Query

router = APIRouter()


@router.get("")
async def search(
    query: str = Query(..., description="检索查询"),
    city: str | None = Query(None),
    category: str | None = Query(None),
    top_k: int = Query(5, ge=1, le=50),
) -> dict:
    """返回命中的 chunks 及分数，供前端调试页展示。

    响应示例：
    {
      "success": true,
      "data": [{"text": "...", "score": 0.87, "metadata": {...}}]
    }
    """
    from app.rag.retriever import debug_search

    chunks = await debug_search(query=query, city=city, category=category, top_k=top_k)
    return {"success": True, "data": chunks}
