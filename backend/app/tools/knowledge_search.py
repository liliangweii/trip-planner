"""Tool① 知识库检索工具：包装混合检索器，入参 {query, city, category}。

对应设计文档 §6.3 Tool 列表。返回带编号与元数据的攻略片段，
供规划 Agent 获取"该城市的玩法与贴士"；pk 即 Citation.chunk_id 溯源键。
"""
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.rag.retriever import get_retriever


class KnowledgeSearchInput(BaseModel):
    """知识库检索工具入参。"""

    query: str = Field(..., description="检索查询文本")
    city: str = Field(..., description="目标城市，用于分区路由与过滤")
    category: str | None = Field(
        None,
        description="可选类别：attraction/food/hotel/route/tips",
    )


@tool("search_travel_knowledge", args_schema=KnowledgeSearchInput)
def search_travel_knowledge_tool(query: str, city: str, category: str | None = None) -> str:
    """检索旅行知识库，返回带编号、来源与引用键的攻略片段。"""
    docs = get_retriever().search(query, city=city, category=category)

    if not docs:
        return f"未检索到 {city} 的相关攻略。"

    lines = []
    for i, d in enumerate(docs):
        meta = d.metadata
        lines.append(
            f"[{i + 1}] (pk={meta.get('pk', '')}) "
            f"({meta.get('title', '')}/{meta.get('category', '')}) {d.page_content}"
        )
    return "\n\n".join(lines)
