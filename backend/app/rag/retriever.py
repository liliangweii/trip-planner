"""混合检索器（P3 实现）：稠密 ANN + BM25 全文稀疏双路召回 → RRF 融合 → API 精排。

对应设计文档 §5.4 四级流水线（查询改写 MultiQuery 暂未启用，记入 P6 调参清单）：
① 混合召回：稠密路（BGE-M3 dense, COSINE）+ 稀疏路（Milvus BM25 全文检索，query 直接传文本）
② RRF 融合：对双路排名取倒数排名和，k=60（对参数不敏感，设计文档默认）
③ 标量过滤：city/category 表达式（partition key 自动路由到城市分区）
④ 精排：SiliconFlow /rerank（reranker.py），取 settings.rerank_top_k 进 Prompt

返回 LangChain Document（含 pk/source/title/section/category/city metadata），
既能被 LCEL 链消费，pk 又能作为 Citation.chunk_id 溯源。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document

from app.config import settings
from app.rag.embeddings import embed_texts
from app.rag.milvus_store import (
    CITY_FIELD,
    COLLECTION,
    DENSE_FIELD,
    SOURCE_FIELD,
    SPARSE_FIELD,
    TEXT_FIELD,
    TITLE_FIELD,
    SECTION_FIELD,
    CATEGORY_FIELD,
    get_client,
)
from app.rag.reranker import rerank as api_rerank

# 输出到 Prompt 的字段：正文 + 溯源必需元数据
_OUTPUT_FIELDS = [
    TEXT_FIELD,
    CITY_FIELD,
    CATEGORY_FIELD,
    TITLE_FIELD,
    SECTION_FIELD,
    SOURCE_FIELD,
]


def _build_filter(city: str | None = None, category: str | None = None) -> str | None:
    """构造标量过滤表达式（partition key 城市自动路由）。"""
    parts = []
    if city:
        parts.append(f"{CITY_FIELD} == '{city.replace(chr(39), chr(92) + chr(39))}'")
    if category:
        parts.append(f"{CATEGORY_FIELD} == '{category.replace(chr(39), chr(92) + chr(39))}'")
    return " and ".join(parts) if parts else None


def _hits_to_docs(hits: list[dict]) -> list[Document]:
    """把 Milvus search 命中的 {id, distance, entity} 转成 LangChain Document。"""
    docs: list[Document] = []
    for hit in hits:
        entity = hit.get("entity") or {}
        meta = {
            "pk": str(hit.get("id")),
            "source": entity.get(SOURCE_FIELD, ""),
            "title": entity.get(TITLE_FIELD, ""),
            "section": entity.get(SECTION_FIELD, ""),
            "category": entity.get(CATEGORY_FIELD, ""),
            "city": entity.get(CITY_FIELD, ""),
            "score": float(hit.get("distance", 0.0)),
        }
        docs.append(Document(page_content=entity.get(TEXT_FIELD, ""), metadata=meta))
    return docs


@dataclass
class HybridRetriever:
    """稠密+稀疏双路混合检索器。"""

    client: object = field(default_factory=get_client)
    collection: str = COLLECTION
    recall_k: int = settings.retrieval_top_k  # 召回 top_k = 20
    rrf_k: int = settings.rrf_k  # RRF 融合常数 = 60
    rerank_top_k: int = settings.rerank_top_k  # 精排后取 5

    def _search_one(self, query_vec_or_text, anns_field: str, metric: str, expr: str | None) -> list[dict]:
        """执行单路搜索。稠密路传向量，稀疏路（BM25 全文）直接传查询文本。"""
        search_params = {
            "metric_type": metric,
            "params": {"ef": 128} if metric == "COSINE" else {"drop_ratio_search": 0.0},
        }
        hits = self.client.search(
            collection_name=self.collection,
            data=[query_vec_or_text],
            anns_field=anns_field,
            filter=expr,
            limit=self.recall_k,
            output_fields=_OUTPUT_FIELDS,
            search_params=search_params,
        )
        return hits[0] if hits else []

    def _rrf_fuse(self, dense_hits: list[dict], sparse_hits: list[dict]) -> list[tuple[str, float, dict]]:
        """RRF 融合：score(pk) = Σ 1/(k + rank)，rank 从 1 起。"""
        scores: dict[str, float] = {}
        for hits in (dense_hits, sparse_hits):
            for rank, hit in enumerate(hits, start=1):
                pk = str(hit.get("id"))
                scores[pk] = scores.get(pk, 0.0) + 1.0 / (self.rrf_k + rank)
        # 取融合分最高的 recall_k 个，按分降序
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: self.recall_k]

        # 还原实体字段（稠密优先；稀疏未命中时从稠密补）
        by_pk: dict[str, dict] = {}
        for hits in (dense_hits, sparse_hits):
            for hit in hits:
                by_pk.setdefault(str(hit.get("id")), hit)
        out = []
        for pk, score in ranked:
            hit = by_pk[pk]
            hit["_rrf_score"] = score
            out.append((pk, score, hit))
        return out

    def search(
        self,
        query: str,
        city: str | None = None,
        category: str | None = None,
        top_k: int | None = None,
        enable_rerank: bool = True,
    ) -> list[Document]:
        """混合检索主入口：双路召回 → RRF 融合 → API 精排 → Document 列表。"""
        expr = _build_filter(city, category)

        # ① 稠密路：查询文本 → API Embedding → COSINE 近邻
        q_vec = embed_texts([query])[0]
        dense_hits = self._search_one(q_vec, DENSE_FIELD, "COSINE", expr)

        # ② 稀疏路：BM25 全文检索（函数集合上直接传文本，库内分词算 BM25 权重）
        sparse_hits = self._search_one(query, SPARSE_FIELD, "BM25", expr)

        # ③ RRF 融合
        fused = self._rrf_fuse(dense_hits, sparse_hits)
        fused_docs = [_hits_to_docs([hit])[0] for _, _, hit in fused]
        for doc, (_, score, _) in zip(fused_docs, fused):
            doc.metadata["score"] = score  # 融合分覆盖单路距离

        # ④ 精排（可选，失败自动降级）
        if enable_rerank:
            fused_docs = api_rerank(query, fused_docs, top_k=top_k)
        else:
            fused_docs = fused_docs[: top_k or self.rerank_top_k]
        return fused_docs

    def invoke(self, query: str, config: dict | None = None) -> list[Document]:
        """LCEL 兼容入口：链内以 Runnable 方式调用（config 携带过滤条件）。"""
        config = config or {}
        return self.search(
            query,
            city=config.get("city"),
            category=config.get("category"),
            top_k=config.get("top_k"),
            enable_rerank=config.get("enable_rerank", True),
        )


_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """进程级单例检索器（复用 Milvus 客户端连接）。"""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


async def debug_search(
    query: str,
    city: str | None = None,
    category: str | None = None,
    top_k: int = 5,
) -> list[dict]:
    """裸检索调试：/api/search 用，返回 text + score + metadata。"""
    docs = get_retriever().search(query, city=city, category=category, top_k=top_k)
    return [
        {"text": d.page_content, "score": d.metadata.get("score"), "metadata": d.metadata}
        for d in docs
    ]
