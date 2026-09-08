"""POST /api/trip/plan —— 行程规划 Agent。

对应设计文档 §6.3、§7。
"""
from fastapi import APIRouter, HTTPException

from app.models.schemas import TripPlan, TripRequest

router = APIRouter()


@router.post("/plan", response_model=TripPlan)
def plan_trip(req: TripRequest) -> TripPlan:
    """生成旅行计划（Agent 流程，同步 JSON 返回）。

    sync def：Agent 是多轮工具调用（秒级~分钟级重 IO），交给 FastAPI 线程池执行，
    避免阻塞事件循环；内部已含三级降级，理论上总能返回合法 TripPlan。
    """
    from app.chains.planner_agent import run_planner

    try:
        return run_planner(req)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"规划失败：{exc}") from exc
