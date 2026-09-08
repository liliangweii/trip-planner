"""应用配置：基于 pydantic-settings，从环境变量/.env 加载。

对应设计文档 §9.3 配置项表。
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局配置，字段与 backend/.env.example 一一对应。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- LLM（OpenAI 兼容接口）----
    llm_api_key: str = Field(default="", description="OpenAI 兼容 LLM 的 API Key")
    llm_base_url: str = Field(
        default="https://api.deepseek.com/v1",
        description="LLM 服务基础地址",
    )
    llm_model: str = Field(default="deepseek-chat", description="LLM 模型名")

    # ---- Milvus ----
    milvus_uri: str = Field(default="http://localhost:19530", description="Milvus 连接地址")
    milvus_collection: str = Field(default="travel_knowledge", description="集合名")
    milvus_token: str = Field(default="", description="Milvus 鉴权 token（可选）")

    # ---- Embedding ----
    embedding_provider: Literal["local_bge", "api"] = Field(
        default="local_bge",
        description="local_bge=本地 BGE-M3；api=OpenAI 兼容 Embedding",
    )
    embedding_api_base: str = Field(default="", description="方案 B：Embedding API 地址")
    embedding_api_key: str = Field(default="", description="方案 B：Embedding API Key")
    embedding_model: str = Field(default="BAAI/bge-m3", description="Embedding 模型名")

    # ---- Reranker ----
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="CrossEncoder 重排序模型",
    )
    # ---- Rerank API（方案 B：重排也走 API，本机无 GPU 不跑本地模型）----
    rerank_enabled: bool = Field(default=True, description="是否启用 API 精排")
    rerank_api_base: str = Field(
        default="https://api.siliconflow.cn/v1",
        description="重排序 API 地址（OpenAI 兼容 /rerank）",
    )
    rerank_api_key: str = Field(default="", description="重排序 API Key")

    # ---- 高德地图 ----
    amap_api_key: str = Field(default="", description="高德 Web 服务 Key")

    # ---- 会话记忆 ----
    sql_history_url: str = Field(
        default="sqlite:///./chat_history.db",
        description="会话记忆存储（SQLite 默认）",
    )

    # ---- 检索参数 ----
    retrieval_top_k: int = Field(default=20, description="召回 top_k")
    rerank_top_k: int = Field(default=5, description="重排序后取 top_k 进 Prompt")
    rrf_k: int = Field(default=60, description="RRF 融合常数 k")

    # ---- 规划模式 ----
    planner_mode: Literal["agent", "fast"] = Field(
        default="fast",
        description=(
            "agent=Tool-Calling Agent 完整编排（自适应但慢，~200s）；"
            "fast=确定性管道(检索+天气+POI 补点) + 单次 LLM 结构化生成（快，~30s）"
        ),
    )

    # ---- 服务运行 ----
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"],
        description="允许的前端来源",
    )


@lru_cache
def get_settings() -> Settings:
    """单例化配置，避免重复读取 .env。"""
    return Settings()


settings = get_settings()
