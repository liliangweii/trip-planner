"""高德地图 Web 服务 API 客户端（httpx 直调，返回给 LLM 的精简文本）。

对应设计文档 §6.3。设计取舍：
- 不返回原始 JSON（大而杂乱，浪费 LLM token），解析后输出紧凑要点文本
- 任何 API 异常都返回可读错误串（而非抛异常）——工具层约定：Agent 看到错误文本
  可以决定重试或放弃，而不是中断整个规划
- Key 未配置时给出明确指引
"""
from __future__ import annotations

import httpx
from loguru import logger

from app.config import settings

AMAP_BASE = "https://restapi.amap.com"


def _key() -> str:
    if not settings.amap_api_key:
        raise RuntimeError(
            "未配置 AMAP_API_KEY：请在 backend/.env 设置（lbs.amap.com 控制台申请 Web 服务 Key）"
        )
    return settings.amap_api_key


def _get(path: str, params: dict, timeout: int = 10) -> dict:
    """GET 高德 REST，统一解析 status/错误。"""
    params = {"key": _key(), **params}
    resp = httpx.get(f"{AMAP_BASE}{path}", params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "1":
        raise RuntimeError(f"高德 API 返回错误: {data.get('info', 'unknown')}")
    return data


# ---------- POI 搜索 ----------

def poi_search(query: str, city: str, citylimit: bool = True, page: int = 1) -> str:
    """POI 文本搜索（/v3/place/text），返回 top POI 要点。"""
    data = _get(
        "/v3/place/text",
        {
            "keywords": query,
            "city": city,
            "citylimit": "true" if citylimit else "false",
            "offset": 5,
            "page": page,
            "extensions": "base",
        },
    )
    pois = data.get("pois", [])
    if not pois:
        return f"高德未找到 {city} 的「{query}」相关 POI。"

    lines = []
    for p in pois[:5]:
        loc = p.get("location", "0,0")  # "lng,lat"
        lng, lat = (loc.split(",") + ["0", "0"])[:2]
        lines.append(
            "POI: {name} | 类别: {type} | 地址: {address} | 坐标: {lng},{lat}".format(
                name=p.get("name", ""),
                type=(p.get("type") or "").split(";")[0],
                address=p.get("address", ""),
                lng=lng,
                lat=lat,
            )
        )
    return "\n".join(lines)


# ---------- POI 坐标抽取（供降级计划补点用） ----------

def poi_location(query: str, city: str) -> dict | None:
    """返回第一个 POI 的结构化坐标 {name,address,lng,lat}；无结果返回 None。

    与 poi_search（给 LLM 的文本）互补：这是给确定性代码用的结构化入口。
    """
    try:
        data = _get(
            "/v3/place/text",
            {
                "keywords": query,
                "city": city,
                "citylimit": "true",
                "offset": 3,
                "page": 1,
                "extensions": "base",
            },
        )
    except RuntimeError:
        return None
    pois = data.get("pois") or []
    if not pois:
        return None
    p = pois[0]
    loc = (p.get("location") or "0,0").split(",")
    try:
        lng, lat = float(loc[0]), float(loc[1])
    except (ValueError, IndexError):
        return None
    if abs(lng) < 0.001 and abs(lat) < 0.001:
        return None
    return {"name": p.get("name", ""), "address": p.get("address", ""), "lng": lng, "lat": lat}


def poi_photos(query: str, city: str) -> str:
    """返回 POI 第一张图片 URL（高德 photos），无图或异常时返回空串。

    用于给行程中的景点/酒店补充图片。统一转 https 以避免 http 页面 mixed-content 拦截。
    """
    try:
        data = _get(
            "/v3/place/text",
            {
                "keywords": query,
                "city": city,
                "citylimit": "true",
                "offset": 1,
                "page": 1,
                "extensions": "all",
            },
        )
    except RuntimeError:
        return ""
    pois = data.get("pois") or []
    if not pois:
        return ""
    photos = pois[0].get("photos") or []
    if not photos:
        return ""
    url = (photos[0].get("url") or "").strip()
    if not url:
        return ""
    # 统一 https，避免 http 页面 mixed-content 拦截
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    return url


# ---------- 天气 ----------

def weather(city: str) -> str:
    """天气查询（/v3/weather/weatherInfo, extensions=all 预报）。"""
    data = _get(
        "/v3/weather/weatherInfo",
        {"city": city, "extensions": "all"},
    )
    forecasts = data.get("forecasts", [])
    if not forecasts:
        return f"高德未返回 {city} 天气。"
    casts = forecasts[0].get("casts", [])[:5]
    lines = [f"{city}天气预报："]
    for c in casts:
        lines.append(
            "{date}: 白天{dayweather} {daytemp}℃ / 夜间{nightweather} {nighttemp}℃".format(
                date=c.get("date", ""),
                dayweather=c.get("dayweather", "未知"),
                daytemp=c.get("daytemp", "?"),
                nightweather=c.get("nightweather", "未知"),
                nighttemp=c.get("nighttemp", "?"),
            )
        )
    return "\n".join(lines)


def weather_casts(city: str) -> list[dict]:
    """返回结构化逐日预报列表（最多约 5 天，自今天起）。

    注意能力边界：高德仅提供未来约 4 天预报，行程日期超出该窗口时
    调用方必须把 weather_info 置空，绝不能把窗口内预报错贴到未来日期。
    """
    try:
        data = _get(
            "/v3/weather/weatherInfo",
            {"city": city, "extensions": "all"},
        )
    except RuntimeError:
        return []
    forecasts = data.get("forecasts") or []
    if not forecasts:
        return []
    casts = []
    for c in (forecasts[0].get("casts") or [])[:8]:
        casts.append(
            {
                "date": c.get("date", ""),
                "day_weather": c.get("dayweather", ""),
                "night_weather": c.get("nightweather", ""),
                "temperature": f"{c.get('daytemp', '?')}℃~{c.get('nighttemp', '?')}℃",
            }
        )
    return casts


# ---------- 路线规划 ----------

_ROUTE_PATH = {
    "driving": "/v3/direction/driving",
    "walking": "/v3/direction/walking",
    "riding": "/v4/direction/bicycling",
    "transit": "/v3/direction/transit/integrated",
}


def route_plan(origin: str, destination: str, mode: str = "driving", city: str | None = None) -> str:
    """路线规划，返回距离/耗时/方案概要。

    Args:
        origin/destination: "lng,lat"。
        mode: driving / walking / transit(公交，需 city) / riding。
    """
    path = _ROUTE_PATH.get(mode)
    if not path:
        return f"不支持的出行方式: {mode}（可选 driving/walking/transit/riding）"

    params: dict = {"origin": origin, "destination": destination}
    if mode == "transit":
        if not city:
            return "公交规划需要 city 参数（城市名/adcode）。"
        params["city"] = city

    try:
        data = _get(path, params, timeout=15)
    except RuntimeError as exc:
        return f"路线规划失败：{exc}"

    if mode == "riding":
        paths = (data.get("data") or {}).get("paths", [])
    else:
        paths = (data.get("route") or {}).get("paths", [])

    if not paths:
        return "高德未返回可行路线。"
    p = paths[0]
    dist = int(p.get("distance", 0))
    dur = int(p.get("duration", 0))
    summary = {
        "driving": f"驾车 {dist/1000:.1f} km，约 {dur/60:.0f} 分钟",
        "walking": f"步行 {dist/1000:.1f} km，约 {dur/60:.0f} 分钟",
        "riding": f"骑行 {dist/1000:.1f} km，约 {dur/60:.0f} 分钟",
        "transit": f"公交 {dist/1000:.1f} km，约 {dur/60:.0f} 分钟",
    }
    return summary.get(mode, f"路线 {dist} m / {dur} s")
