"""行程规划 Agent（P4 实现）：四工具 Tool-Calling Agent + 结构化输出 + 三级降级。

对应设计文档 §6.3、§5.5：
- L1（完整）：LangChain Classic Tool-Calling Agent——
    先 search_travel_knowledge 拿"该城市玩法与贴士"（带 pk 引用）→
    amap POI/天气/路线补实时数据 → 最终输出 TripPlan JSON
- L2（降级）：Agent 失败/输出非法 → 跳过工具调用，纯知识库上下文 + LLM 直出
    （坐标用 0.0 降级，仍保留引用）
- L3（兜底）：LLM 也不可用 → 从检索结果构建最小合法 TripPlan（纯模板，零依赖）

防幻觉引用（§5.5 硬约束）：结构化解析后，剥离所有 chunk_id 不在
"本次召回集合"中的引用。
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from loguru import logger

from app.config import settings
from app.models.schemas import DayPlan, TripPlan, TripRequest
from app.rag.prompts import PLANNER_AGENT_PROMPT
from app.rag.retriever import get_retriever
from app.services.llm_service import get_llm
from app.tools.amap import amap_poi_search_tool, amap_route_plan_tool, amap_weather_tool
from app.tools.knowledge_search import search_travel_knowledge_tool

# ---------- Schema 提示（写进 LLM 指令，保证可被 json.loads + model_validate） ----------

SCHEMA_HINT = """最终只输出一个 TripPlan JSON 对象（不要 markdown 代码围栏、不要解释），结构如下：
{"city": str, "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD",
 "days": [{"date": "YYYY-MM-DD", "day_index": int 从1递增, "description": str,
   "transportation": str, "accommodation": str,
   "hotel": {"name": str, "address": str, "location": {"longitude": float, "latitude": float},
             "price_per_night": float, "description": str} 或 null,
   "attractions": [{"name": str, "address": str,
     "location": {"longitude": float, "latitude": float},
     "visit_duration": int 分钟, "description": str 须在句末标注[n],
     "category": str, "ticket_price": float,
     "citations": [{"chunk_id": str, "source": str, "snippet": str}]}],
   "meals": [{"type": "breakfast|lunch|dinner", "name": str, "description": str,
     "location": 同 location 或 null}]}],
 "weather_info": [{"date": str, "day_weather": str, "night_weather": str, "temperature": str}],
 "overall_suggestions": str 须引用[n],
 "budget": {"currency": "CNY", "total": float,
   "breakdown": {"transport": float, "accommodation": float, "food": float, "tickets": float}},
 "references": [{"chunk_id": str, "source": str, "snippet": str}]}

硬约束：attractions/overall_suggestions 的事实只能来自 search_travel_knowledge 返回的 [n] 片段并标注；
citations.chunk_id 必须使用工具返回的 pk；无坐标数据时 location 用 0.0。"""

_PK_RE = re.compile(r"pk=(\d+)")
_CTX_MAX = 280  # 每条上下文截断长度（控制 token）


# ---------- 检索 ----------

def _retrieve(req: TripRequest) -> list[Document]:
    """为规划检索知识库：主攻略 + （偏好含美食时）追加 food 专项。"""
    retriever = get_retriever()
    queries = [f"{req.city} 旅行玩法攻略 {''.join(req.preferences)}"]
    if "美食" in req.preferences:
        queries.append(f"{req.city} 美食推荐")
    docs: list[Document] = []
    seen: set[str] = set()
    for q in queries:
        for d in retriever.search(q, city=req.city, top_k=6):
            pk = str(d.metadata.get("pk", ""))
            if pk and pk not in seen:
                seen.add(pk)
                docs.append(d)
    return docs[:10]


def _build_context(docs: list[Document]) -> str:
    """带编号知识库上下文（含 pk 供溯源）。"""
    lines = []
    for i, d in enumerate(docs):
        meta = d.metadata
        snippet = d.page_content.replace("\n", " ")[:_CTX_MAX]
        lines.append(
            f"[{i + 1}] pk={meta.get('pk', '')} | {meta.get('category', '')} | "
            f"{meta.get('title', '')}\n{snippet}"
        )
    return "\n\n".join(lines)


def _collect_pks(docs: list[Document], observations: list[str]) -> set[str]:
    """本次召回集合 = 检索文档 pk ∪ Agent 工具观测文本中出现的 pk。"""
    pks = {str(d.metadata.get("pk", "")) for d in docs if d.metadata.get("pk")}
    for obs in observations:
        pks.update(_PK_RE.findall(str(obs)))
    return pks


def _filter_citations(plan: TripPlan, allowed: set[str]) -> TripPlan:
    """剥离 chunk_id 不在召回集合中的引用（防幻觉引用）。"""
    plan.references = [c for c in plan.references if c.chunk_id in allowed]
    for day in plan.days:
        for attr in day.attractions:
            attr.citations = [c for c in attr.citations if c.chunk_id in allowed]
    return plan


# ---------- JSON 提取 / 校验 ----------

def _extract_plan_json(text: str) -> dict | None:
    """从 Agent 输出中稳健提取 JSON（容忍代码围栏与前后废话）。"""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(t[start : end + 1])
    except json.JSONDecodeError:
        return None


def _validate(data: dict) -> TripPlan | None:
    try:
        return TripPlan.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("TripPlan 结构化校验失败: {}", str(exc)[:200])
        return None


# ---------- 三级降级 ----------

def _build_agent_executor():
    """L1 Agent：Tool-Calling（from langchain_classic，LangChain 1.x 迁移点）。"""
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

    tools = [
        search_travel_knowledge_tool,
        amap_poi_search_tool,
        amap_weather_tool,
        amap_route_plan_tool,
    ]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLANNER_AGENT_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm=get_llm(), tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=10,
        handle_parsing_errors=True,
        return_intermediate_steps=True,  # 构造参数：把工具观测带回，用于收集召回 pk
    )


def _level1(req: TripRequest) -> tuple[TripPlan | None, list[str]]:
    """完整 Agent 流程。返回 (plan 或 None, 工具观测文本列表)。"""
    instruction = (
        f"请为以下需求生成 {req.city} 旅行计划。\n"
        f"需求：{req.model_dump_json()}\n\n"
        f"执行步骤：\n"
        f"1) 先调用 search_travel_knowledge 检索「{req.city}」的玩法/景点/美食/贴士（注意引用返回的 pk）；\n"
        f"2) 对计划中的主要景点调用 amap_poi_search 获取地址与经纬度；\n"
        f"3) 调用 amap_weather 查询行程日期的天气；\n"
        f"4) 如需要可调用 amap_route_plan 规划景点间交通（transit 需提供 city 参数）。\n\n"
        f"{SCHEMA_HINT}"
    )
    executor = _build_agent_executor()
    try:
        result = executor.invoke({"input": instruction})
    except Exception as exc:  # noqa: BLE001
        logger.warning("L1 Agent 执行异常，降级 L2: {}", exc)
        return None, []

    observations = [str(obs) for _, obs in result.get("intermediate_steps", [])]
    data = _extract_plan_json(result.get("output", ""))
    plan = _validate(data) if data else None
    if plan is None:
        logger.warning("L1 输出无法解析为 TripPlan，降级 L2")
    return plan, observations


def _level2(req: TripRequest, docs: list[Document]) -> TripPlan | None:
    """RAG 直出：无 Agent/高德，知识库上下文 + LLM 结构化输出。"""
    context = _build_context(docs)
    messages = [
        (
            "system",
            PLANNER_AGENT_PROMPT
            + "\n\n下方知识库片段是本轮唯一事实来源，引用其 [编号] 并保留 pk。\n\n"
            + context,
        ),
        (
            "human",
            f"需求：{req.model_dump_json()}\n\n{SCHEMA_HINT}"
            + "\n（降级模式：无实时 POI/天气，location 一律 0.0，weather_info 可为空数组）",
        ),
    ]
    try:
        text = get_llm().invoke(messages).content
        data = _extract_plan_json(str(text))
        return _validate(data) if data else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("L2 RAG 直出失败，降级 L3: {}", exc)
        return None


def _level3(req: TripRequest, docs: list[Document]) -> TripPlan:
    """纯模板兜底：零 LLM/API 依赖，从检索结果构建最小合法 TripPlan。"""
    try:
        start = datetime.strptime(req.start_date, "%Y-%m-%d")
    except ValueError:
        start = None

    days = []
    for i in range(req.travel_days):
        date = (start + timedelta(days=i)).strftime("%Y-%m-%d") if start else ""
        days.append(
            DayPlan(
                date=date,
                day_index=i + 1,
                description=f"第{i + 1}天：{req.city} 自由探索（模板降级产物，建议人工核对）",
                transportation=req.transportation,
                accommodation=req.accommodation,
            )
        )

    references = []
    for i, d in enumerate(docs[:8]):
        meta = d.metadata
        references.append(
            {
                "chunk_id": str(meta.get("pk", "")),
                "source": meta.get("title") or meta.get("source", ""),
                "snippet": d.page_content[:120],
            }
        )
    titles = "、".join({d.metadata.get("title", "") for d in docs[:8] if d.metadata.get("title")})
    return TripPlan(
        city=req.city,
        start_date=req.start_date,
        end_date=req.end_date,
        days=days,
        overall_suggestions=f"知识库检索到以下参考攻略：{titles}。此为降级模板，建议调用完整规划。",
        references=references,  # type: ignore[arg-type]
    )


# ---------- 对外入口 ----------

def run_planner(req: TripRequest) -> TripPlan:
    """执行规划（三级降级，保证任何情况下都返回合法 TripPlan）。"""
    docs = _retrieve(req)
    allowed = {str(d.metadata.get("pk", "")) for d in docs if d.metadata.get("pk")}

    # L1 完整 Agent
    if settings.llm_api_key:
        plan, observations = _level1(req)
        if plan is not None:
            allowed |= _collect_pks(docs, observations)
            return _filter_citations(plan, allowed)
        # L2 RAG 直出
        plan = _level2(req, docs)
        if plan is not None:
            return _filter_citations(plan, allowed)

    logger.warning("进入 L3 模板兜底（city={}）", req.city)
    return _filter_citations(_level3(req, docs), allowed)
