"""Tool②③④ 高德工具：POI 搜索 / 天气 / 路线规划。

对应设计文档 §6.3：@tool 封装 httpx 直调高德 Web 服务。
统一约定：工具返回人类可读文本（服务层已解析精简），错误也以文本返回，
让 Tool-Calling Agent 自主决定重试或降级，绝不抛异常中断 Agent。
"""
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.services.amap_service import poi_search, route_plan, weather


class PoiSearchInput(BaseModel):
    query: str = Field(..., description="POI 搜索关键词，如'故宫'")
    city: str = Field(..., description="城市名，如'北京'")
    citylimit: bool = Field(default=True, description="是否仅限当前城市")


@tool("amap_poi_search", args_schema=PoiSearchInput)
def amap_poi_search_tool(query: str, city: str, citylimit: bool = True) -> str:
    """高德 POI 搜索：返回景点/地点的地址与经纬度坐标（用于行程打点）。"""
    try:
        return poi_search(query=query, city=city, citylimit=citylimit)
    except Exception as exc:  # noqa: BLE001
        return f"[高德 POI 搜索失败] {exc}"


class WeatherInput(BaseModel):
    city: str = Field(..., description="城市名，如'北京'")


@tool("amap_weather", args_schema=WeatherInput)
def amap_weather_tool(city: str) -> str:
    """高德天气查询：返回逐日预报。注意：高德仅提供自今天起约未来 4 天预报；
    若行程日期不在返回的日期内，请在 TripPlan 中把 weather_info 置为 []，不要错贴或编造。"""
    try:
        return weather(city=city)
    except Exception as exc:  # noqa: BLE001
        return f"[高德天气查询失败] {exc}"


class RoutePlanInput(BaseModel):
    origin: str = Field(..., description="起点经纬度，格式 'lng,lat'（来自 POI 搜索）")
    destination: str = Field(..., description="终点经纬度，格式 'lng,lat'")
    mode: str = Field(default="driving", description="出行方式：driving/walking/transit/riding")
    city: str | None = Field(None, description="公交(transit)模式需要城市名，如'北京'")


@tool("amap_route_plan", args_schema=RoutePlanInput)
def amap_route_plan_tool(
    origin: str,
    destination: str,
    mode: str = "driving",
    city: str | None = None,
) -> str:
    """高德路线规划：返回两点间距离、耗时与方案概要。"""
    try:
        return route_plan(origin=origin, destination=destination, mode=mode, city=city)
    except Exception as exc:  # noqa: BLE001
        return f"[高德路线规划失败] {exc}"
