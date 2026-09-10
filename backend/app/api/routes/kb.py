"""/api/kb/* —— 知识库管理。

对应设计文档 §7：
- POST   /api/kb/ingest           上传文件或重扫城市目录入库
- GET    /api/kb/documents        文档列表（按 city/category 过滤）
- DELETE /api/kb/documents/{id}   删除某文档全部 chunks（id = source 路径）

与 scripts/ingest.py 共用 app/rag/ingest.py 管道，逻辑单一来源。
"""
from pathlib import Path

from fastapi import APIRouter, File, Query, UploadFile
from pydantic import BaseModel

from app.rag.ingest import delete_by_source, ingest_directory, ingest_file, list_documents
from app.services.cache import clear_plan_cache

router = APIRouter()


class IngestResponse(BaseModel):
    success: bool
    inserted_chunks: int
    message: str = ""


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    city: str = Query(..., description="目标城市"),
    category: str = Query(..., description="attraction/food/hotel/route/tips"),
    file: UploadFile | None = File(None, description="可选：单文件上传（否则扫 {city}/{category} 目录）"),
) -> IngestResponse:
    """入库：multipart 文件上传或城市目录重扫。

    同步 def 而非 async：重 IO 操作交给 FastAPI 线程池，避免阻塞事件循环。
    """
    try:
        if file is not None:
            # 上传文件落盘到临时路径再走统一管道
            tmp = Path("uploads") / f"{city}_{category}_{file.filename}"
            tmp.parent.mkdir(exist_ok=True)
            tmp.write_bytes(file.file.read())
            n = ingest_file(tmp, city=city, category=category)
            tmp.unlink(missing_ok=True)
        else:
            n = ingest_directory(city=city, category=category)
        # 语料变了 → 旧行程缓存立即失效，否则新内容要等 TTL 才生效
        cleared = clear_plan_cache()
        msg = "入库完成" if not cleared else f"入库完成，已清理 {cleared} 条行程缓存"
        return IngestResponse(success=True, inserted_chunks=n, message=msg)
    except Exception as exc:  # noqa: BLE001
        return IngestResponse(success=False, inserted_chunks=0, message=str(exc))


@router.get("/documents")
def documents(
    city: str | None = Query(None),
    category: str | None = Query(None),
) -> dict:
    """知识库文档列表（按 source 去重，含 chunk 数）。"""
    return {"success": True, "data": list_documents(city=city, category=category)}


@router.delete("/documents/{doc_id:path}")
def delete_document(doc_id: str) -> dict:
    """删除某文档的全部 chunks。doc_id 为 source 路径（URL 编码）。"""
    from urllib.parse import unquote

    source = unquote(doc_id)
    delete_by_source(source)
    # 语料变更 → 行程缓存失效
    cleared = clear_plan_cache()
    return {"success": True, "deleted_source": source, "cleared_plan_cache": cleared}
