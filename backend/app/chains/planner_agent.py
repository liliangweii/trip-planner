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
             "price_per_night": float, "description": str}（须为行程推荐 1 家，多日可重复同一家；仅在完全无住宿材料时才为 null）,
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
citations.chunk_id 必须使用工具返回的 pk；无坐标数据时 location 用 0.0；
weather_info 只能填天气材料（高德仅未来约 4 天）中真实出现的日期，行程超出窗口则必须为 []，严禁编造。"""

_PK_RE = re.compile(r"pk=(\d+)")
_CTX_MAX = 280  # 每条上下文截断长度（控制 token）


# ---------- 检索 ----------

def _retrieve(req: TripRequest) -> list[Document]:
    """为规划检索知识库：主攻略 + （偏好含美食时）追加 food 专项 + 始终追加 hotel 专项。"""
    retriever = get_retriever()
    queries = [f"{req.city} 旅行玩法攻略 {''.join(req.preferences)}"]
    if "美食" in req.preferences:
        queries.append(f"{req.city} 美食推荐")
    # 始终检索住宿/酒店，保证规划有可引用的酒店素材（否则 hard 约束下 hotel 只能为 null）
    queries.append(f"{req.city} 酒店住宿推荐 {req.accommodation}")
    docs: list[Document] = []
    seen: set[str] = set()
    for q in queries:
        for d in retriever.search(q, city=req.city, top_k=6):
            pk = str(d.metadata.get("pk", ""))
            if pk and pk not in seen:
                seen.add(pk)
                docs.append(d)
    return docs[:12] # 截取前 12 条文档


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
        if day.hotel and day.hotel.citations:
            day.hotel.citations = [c for c in day.hotel.citations if c.chunk_id in allowed]
    return plan


def _gen_rules(req: TripRequest) -> str:
    """生成期硬性要求：必须推荐酒店 + 必须按预算约束分配。"""
    if req.budget_total:
        budget_clause = (
            f"预算约束：本次总预算为 {req.budget_total} 元（人民币）。"
            "budget.total 与 breakdown（transport/accommodation/food/tickets）各项之和"
            f"不得超过 {req.budget_total} 元；据此反推住宿档次、餐饮与门票取舍，"
            "优先保证核心景点，超预算时削减非必要消费，不得出现总额超预算的规划。"
        )
    else:
        budget_clause = (
            "预算约束：用户未提供预算，请给出合理的总预估与 breakdown"
            "（transport/accommodation/food/tickets）。"
        )
    return (
        "住宿要求：必须依据知识库中的住宿/酒店信息，为本次行程推荐 1 家酒店，"
        "将其填入每一天的 hotel 字段（多日行程推荐同一家即可），并为其标注 citations；"
        "只有在完全没有任何住宿材料时才可将 hotel 设为 null。\n"
        + budget_clause
    )


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


def _fill_missing_coords(plan: TripPlan, req: TripRequest) -> TripPlan:
    """降级计划坐标补全：对所有坐标为 0 的景点，按名称确定性直调高德 POI 补真实坐标。

    L2/L3 的坐标是 0.0（降级设计）；这一步把"取坐标"从 Agent 的随机决策
    变成确定性代码（P4 复盘优化项），保证降级计划也能在地图上打点。
    任何失败都静默跳过（保持降级可用性优先）。
    """
    if not settings.amap_api_key:
        return plan
    try:
        from app.services.amap_service import poi_location

        for day in plan.days:
            for attr in day.attractions:
                loc = attr.location
                if loc is not None and (abs(loc.longitude) > 0.001 or abs(loc.latitude) > 0.001):
                    continue  # 已有真实坐标（L1 产物）不重复请求
                hit = poi_location(attr.name, req.city)
                if hit is None:
                    continue
                attr.location.longitude = hit["lng"]
                attr.location.latitude = hit["lat"]
                if not attr.address:
                    attr.address = hit["address"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("坐标补全失败（忽略，维持降级计划）: {}", exc)
    return plan


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
        max_iterations=8,  # 瘦身：减少潜在的长尾工具往返
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
        f"调用纪律（为提速强制）：知识库检索 ≤2 次、amap_weather 恰 1 次、amap_poi_search ≤2 次；"
        f"路线规划仅在确需给出交通换乘时才调用。description 每条 ≤80 字。\n\n"
        f"{SCHEMA_HINT}\n\n{_gen_rules(req)}"
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
        # 输出采样入日志：便于定位是"无 JSON"还是"JSON 校验失败"
        snippet = str(result.get("output", ""))[:300]
        logger.warning("L1 输出无法解析为 TripPlan，降级 L2 | 输出前300字: {}", snippet)
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
            + "\n（降级模式：无实时 POI/天气，location 一律 0.0，weather_info 可为空数组）"
            + f"\n\n{_gen_rules(req)}",
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


# ---------- 快速管道（fast 模式，默认） ----------

def _run_fast(req: TripRequest) -> TripPlan:
    """确定性管道 + 单次 LLM 结构化生成（约 20-50s）。

    提速原理：把 Agent 的 N 轮串行 LLM 往返压缩为 1 轮——
    检索/天气用代码直调，坐标由 _fill_missing_coords 事后确定性补全，
    LLM 只负责"读材料 → 一次性编排撰写"。
    """
    docs = _retrieve(req) # 检索
    allowed = {str(d.metadata.get("pk", "")) for d in docs if d.metadata.get("pk")}

    weather_text = ""
    if settings.amap_api_key:
        try:
            from datetime import timedelta

            from app.services.amap_service import weather_casts

            start = datetime.strptime(req.start_date, "%Y-%m-%d")
            trip_dates = {
                (start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(req.travel_days)
            }
            matched = [c for c in weather_casts(req.city) if c["date"] in trip_dates]
            if matched:
                rows = [
                    "{date}: 白天{day_weather} {temperature}".format(
                        date=c["date"], day_weather=c["day_weather"], temperature=c["temperature"]
                    )
                    for c in matched
                ]
                weather_text = "[行程日期天气预报（高德，仅含与行程重叠的日期）]\n" + "\n".join(rows)
            # 无匹配 = 行程日期超出高德约 4 天可预报窗口 → weather_text 留空，诚实不编造
        except Exception as exc:  # noqa: BLE001
            logger.debug("天气获取失败（忽略）: {}", exc)

    facts = [_build_context(docs)]
    if weather_text:
        facts.append(weather_text)

    system = (
        PLANNER_AGENT_PROMPT
        + "\n\n下方是唯一事实来源（知识库片段 + 实时天气），引用其 [编号] 并保留 pk；"
        "天气仅供行程参考，不得作为知识库 citation 的来源。"
    )
    human = (
        f"需求：{req.model_dump_json()}\n\n事实材料：\n"
        + "\n\n".join(facts)
        + "\n\n"
        + SCHEMA_HINT
        + "\n\n生成要求：description/overall_suggestions 简洁（每条≤80字）且必须标注 [编号]；"
        "坐标未知就写 0（后端会自动补全为真实 POI 坐标），严禁编造经纬度；"
        "citations.chunk_id 只能使用材料中的 pk。"
        "weather_info 只允许填材料「天气预报」中真实出现的日期；"
        "若没有天气材料（行程超出高德约 4 天可预报窗口），weather_info 必须为 []，严禁编造天气。"
        f"\n\n{_gen_rules(req)}"
    )
    try:
        text = get_llm().invoke([("system", system), ("human", human)]).content
        data = _extract_plan_json(str(text))
        plan = _validate(data) if data else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("fast 模式生成失败，降级模板: {}", exc)
        plan = None

    if plan is None:
        logger.warning("进入模板兜底（fast 解析失败, city={}）", req.city)
        plan = _level3(req, docs)
    return _filter_citations(_fill_missing_coords(plan, req), allowed)


# ---------- 对外入口 ----------

def run_planner(req: TripRequest) -> TripPlan:
    """规划入口：按 PLANNER_MODE 分发 fast（默认，快）或 agent（完整编排）。"""
    if settings.planner_mode == "agent":
        return _run_agent_mode(req)
    return _run_fast(req)


def _run_agent_mode(req: TripRequest) -> TripPlan:
    """Agent 完整模式：L1 完整 Agent → L2 RAG 直出 → L3 模板兜底。"""
    docs = _retrieve(req)
    allowed = {str(d.metadata.get("pk", "")) for d in docs if d.metadata.get("pk")}

    # L1 完整 Agent
    if settings.llm_api_key:
        plan, observations = _level1(req)
        if plan is not None:
            allowed |= _collect_pks(docs, observations)
            return _filter_citations(_fill_missing_coords(plan, req), allowed)
        # L2 RAG 直出
        plan = _level2(req, docs)
        if plan is not None:
            return _filter_citations(_fill_missing_coords(plan, req), allowed)

    logger.warning("进入 L3 模板兜底（city={}）", req.city)
    return _filter_citations(_level3(req, docs), allowed)
