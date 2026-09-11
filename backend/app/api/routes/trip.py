"""POST /api/trip/plan —— 行程规划 Agent。

对应设计文档 §6.3、§7。
"""
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from loguru import logger

from app.models.schemas import TripPlan, TripRequest
from app.services.pdf_export import trip_plan_to_pdf

router = APIRouter()


@router.post("/plan", response_model=TripPlan)
def plan_trip(req: TripRequest) -> TripPlan:
    """生成旅行计划（Agent 流程，同步 JSON 返回）。

    sync def：Agent 是多轮工具调用（秒级~分钟级重 IO），交给 FastAPI 线程池执行，
    避免阻塞事件循环；内部已含三级降级，理论上总能返回合法 TripPlan。

    缓存：相同请求参数直接命中 Redis 返回（毫秒级）；缓存不可用则自动走原流程。
    """
    from app.chains.planner_agent import run_planner
    from app.services.cache import get_cached_plan, set_cached_plan

    # 1) 先查缓存（Redis 不可用/未命中均返回 None，自动走原流程）
    cached = get_cached_plan(req)
    if cached is not None:
        logger.info("行程规划缓存命中: city={} days={}", req.city, req.travel_days)
        return cached

    # 缓存未命中 → 必然调用大模型（run_planner 内部：fast=L3 确定性管道+单次LLM；
    # agent=L1 Agent→L2 RAG直出→L3 兜底）。下面这条日志用于确认"cache miss 已触发 LLM 规划"。
    logger.info("缓存未命中，调用大模型生成行程: city={} days={}", req.city, req.travel_days)
    try:
        plan = run_planner(req)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"规划失败：{exc}") from exc

    # 降级产物（大模型未成功参与）明确告警，避免被误认为"正常空计划"
    if getattr(plan, "degraded", False):
        logger.warning(
            "行程规划为降级产物（未走大模型真实生成）: city={} reason={}",
            req.city, plan.degraded_reason,
        )

    # 2) 完整产物写缓存（降级模板在缓存层内部会跳过）
    set_cached_plan(req, plan)
    return plan


@router.post("/export/pdf")
def export_plan_pdf(plan: TripPlan) -> Response:
    """导出行程计划为 PDF（后端 reportlab 渲染，中文 CID 字体，无需字体文件）。

    前端把已生成的 TripPlan 原样回传即可；返回 application/pdf 触发浏览器下载。
    """
    try:
        pdf_bytes = trip_plan_to_pdf(plan)
    except Exception as exc:  # noqa: BLE001
        logger.exception("PDF 导出失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"PDF 导出失败：{exc}") from exc

    # RFC 5987 / 2231 文件名编码：兼容中文文件名（多数浏览器取 filename*=）
    safe_name = f"{plan.city or 'trip'}_行程计划.pdf"
    disposition = f"attachment; filename={quote(safe_name)}; filename*=UTF-8''{quote(safe_name)}"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": disposition},
    )
