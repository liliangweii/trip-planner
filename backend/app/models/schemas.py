"""Pydantic 数据模型（核心 Schemas）。

对应设计文档 §4。
"""
from pydantic import BaseModel, Field


class Location(BaseModel):
    """经纬度。"""

    longitude: float
    latitude: float


class Citation(BaseModel):
    """RAG 溯源引用（设计文档 §4 ★ 新增）。"""

    chunk_id: str = Field(..., description="Milvus 主键")
    source: str = Field(..., description="语料标题/出处")
    url: str = ""
    snippet: str = Field(..., description="命中原文片段（截断展示）")


class Attraction(BaseModel):
    """景点。"""

    name: str
    address: str
    location: Location
    visit_duration: int = Field(..., description="分钟", ge=0)
    description: str
    category: str
    ticket_price: float = 0
    citations: list[Citation] = Field(default_factory=list, description="该景点信息来源")


class Meal(BaseModel):
    """餐食。"""

    type: str = Field(..., description="breakfast/lunch/dinner")
    name: str = ""
    description: str = ""
    location: Location | None = None


class Hotel(BaseModel):
    """住宿。"""

    name: str
    address: str
    location: Location
    price_per_night: float = 0
    description: str = ""
    citations: list[Citation] = Field(default_factory=list, description="该酒店信息来源")


class WeatherInfo(BaseModel):
    """天气信息。"""

    date: str
    day_weather: str = ""
    night_weather: str = ""
    temperature: str = ""


class Budget(BaseModel):
    """预算。"""

    currency: str = "CNY"
    total: float = 0
    breakdown: dict[str, float] = Field(
        default_factory=dict, description="transport/accommodation/food/tickets/other"
    )


class DayPlan(BaseModel):
    """单日行程。"""

    date: str
    day_index: int
    description: str
    transportation: str
    accommodation: str
    hotel: Hotel | None = None
    attractions: list[Attraction] = Field(default_factory=list)
    meals: list[Meal] = Field(default_factory=list, description="breakfast/lunch/dinner")


class TripRequest(BaseModel):
    """行程规划请求体（设计文档 §4）。"""

    city: str = Field(..., description="目的地城市")
    start_date: str = Field(..., description="YYYY-MM-DD")
    end_date: str = Field(..., description="YYYY-MM-DD")
    travel_days: int = Field(default=1, ge=1, le=15)
    transportation: str = Field(default="公共交通", description="公共交通/自驾/步行偏好")
    accommodation: str = Field(default="经济型", description="住宿偏好")
    preferences: list[str] = Field(default_factory=list, description="旅行风格标签")
    free_text_input: str = Field(default="", description="自由补充需求")
    budget_total: float | None = Field(
        default=None, description="总预算（人民币元），可选；提供后规划须在该范围内分配"
    )


class TripPlan(BaseModel):
    """行程规划结果（设计文档 §4 ★ references 全局引用列表）。"""

    city: str
    start_date: str
    end_date: str
    days: list[DayPlan] = Field(default_factory=list)
    weather_info: list[WeatherInfo] = Field(default_factory=list)
    overall_suggestions: str = ""
    budget: Budget = Field(default_factory=Budget)
    references: list[Citation] = Field(
        default_factory=list, description="全局引用列表，前端展示'参考攻略'"
    )
    # 降级标记：True 表示这是 L3 模板兜底产物（大模型未成功生成），并非真实规划。
    # 由 planner_agent 在 LLM 调用失败/未配置时置位，便于前端/调用方区分"空计划"与真实计划。
    degraded: bool = Field(
        default=False,
        description="是否降级产物（大模型未成功参与生成）。True 时整体_suggestions 含降级原因。",
    )
    degraded_reason: str = Field(
        default="",
        description="降级原因（如大模型 402 余额不足/未配置/输出无法解析）。空=正常生成。",
    )
