"""BGE-M3 Embedding 封装：稠密(1024) + 稀疏(BM25) 一体。

对应设计文档 §5.2 ④、§5.4 方案 A/B。

方案 A（默认）：本地 BGE-M3，稠密+稀疏一体，零 API 成本。
方案 B：API 型 Embedding，仅稠密，稀疏路用 Milvus 2.4+ 内置 BM25 补齐。
"""
from app.config import settings


def get_embedding_function():
    """按配置返回 Embedding 函数。

    - local_bge：返回 BGE-M3 EmbeddingFunction（pymilvus.model.hybrid）
    - api：返回 LangChain OpenAI 兼容 Embeddings（仅稠密）
    """
    if settings.embedding_provider == "local_bge":
        try:
            from pymilvus.model.hybrid import BGEM3EmbeddingFunction
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "本地 BGE-M3 需要 pymilvus[model] 与 sentence-transformers，"
                "请安装：pip install 'pymilvus[model]' sentence-transformers"
            ) from exc
        return BGEM3EmbeddingFunction(model_name=settings.embedding_model, use_fp16=False)

    # 方案 B：API Embedding（硅基流动等 OpenAI 兼容接口）
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_api_base,
        chunk_size=16,  # 单请求批量上限：防止批量入库触发 429/超长
        timeout=60,
    )


def get_dense_dim() -> int:
    """稠密向量维度（BGE-M3 = 1024）。"""
    return 1024


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量向量化文本 → 稠密向量列表。

    分批请求（chunk_size）与 429 退避重试由 OpenAIEmbeddings 内部处理，
    入库管道直接整批调用即可。稀疏向量由 Milvus BM25 函数自动生成，无需在此计算。
    """
    if settings.embedding_provider == "local_bge":
        raise RuntimeError(
            "集合 schema 使用 BM25 函数自动生成稀疏向量，当前入库仅支持方案 B"
            "（EMBEDDING_PROVIDER=api）。方案 A 的本地 BGE-M3 需要改写 schema 后接入。"
        )
    embedding = get_embedding_function()
    return embedding.embed_documents(list(texts))
