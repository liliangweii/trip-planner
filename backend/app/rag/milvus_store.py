"""Milvus 集合管理：创建/索引/upsert。

对应设计文档 §5.3 集合设计 + §5.4 方案 B（API Embedding）：
- 稠密向量：API Embedding（硅基流动 BAAI/bge-m3，1024 维）生成后随行写入
- 稀疏向量：Milvus 全文检索 BM25 函数基于 text 字段自动生成，
  要求 text 字段 enable_analyzer=True 且配置中文分词器（否则 BM25 按空格切分，中文失效）
- 因此 insert 时【禁止】手动携带 sparse_vector 字段

注意：BM25 函数为 Milvus 2.5+ 特性（docker-compose 已升级到 v2.5.x）。
"""
from functools import lru_cache

from app.config import settings
from app.rag.embeddings import get_dense_dim

COLLECTION = settings.milvus_collection
DENSE_FIELD = "dense_vector"
SPARSE_FIELD = "sparse_vector"
TEXT_FIELD = "text"
CITY_FIELD = "city"
CATEGORY_FIELD = "category"
TITLE_FIELD = "title"
SECTION_FIELD = "section"
SOURCE_FIELD = "source"


@lru_cache(maxsize=1)
def get_client():
    """获取 MilvusClient 单例（进程内复用连接）。"""
    from pymilvus import MilvusClient

    return MilvusClient(
        uri=settings.milvus_uri,
        token=settings.milvus_token or None,
    )


def ensure_collection() -> None:
    """创建集合 + 索引（幂等：已存在直接返回）。

    Collection: travel_knowledge
    - id            INT64 PK auto_id
    - text          VARCHAR(8192) + 中文 analyzer（BM25 函数输入）
    - dense_vector  FLOAT_VECTOR(1024)   HNSW COSINE
    - sparse_vector SPARSE_FLOAT_VECTOR  BM25 函数自动填充，SPARSE_INVERTED_INDEX
    - city          VARCHAR(64) partition_key（查询自动路由分区）
    - category/title/section/source  标量字段
    """
    from pymilvus import DataType, Function, FunctionType, MilvusClient

    client = get_client()
    if client.has_collection(COLLECTION):
        # 已存在也要确保已加载：delete/query/search 均需 collection 处于 loaded 状态
        client.load_collection(COLLECTION)
        return

    schema = MilvusClient.create_schema(
        description="旅行知识库（稠密+稀疏混合检索）",
    )
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    # 中文语料必须指定 chinese 分词器：默认 standard 按空格切词，中文会被切成整句
    schema.add_field(
        TEXT_FIELD,
        DataType.VARCHAR,
        max_length=8192,
        enable_analyzer=True,
        enable_match=True,
        analyzer_params={"type": "chinese"},
    )
    schema.add_field(DENSE_FIELD, DataType.FLOAT_VECTOR, dim=get_dense_dim())
    schema.add_field(SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(CITY_FIELD, DataType.VARCHAR, max_length=64, is_partition_key=True)
    schema.add_field(CATEGORY_FIELD, DataType.VARCHAR, max_length=32)
    schema.add_field(TITLE_FIELD, DataType.VARCHAR, max_length=256)
    schema.add_field(SECTION_FIELD, DataType.VARCHAR, max_length=256)
    schema.add_field(SOURCE_FIELD, DataType.VARCHAR, max_length=512)

    # 全文检索函数：insert 时对 text 分词计算 BM25 词权重 → sparse_vector（自动）
    bm25_fn = Function(
        name="text_bm25",
        function_type=FunctionType.BM25,
        input_field_names=[TEXT_FIELD],
        output_field_names=[SPARSE_FIELD],
    )
    schema.add_function(bm25_fn)

    client.create_collection(
        collection_name=COLLECTION,
        schema=schema,
        # 按城市分区，查询自动路由分区缩小扫描范围
        num_partitions=16,
    )

    # 索引：稠密 HNSW(COSINE) + 稀疏 SPARSE_INVERTED_INDEX(BM25，与函数配套)
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name=DENSE_FIELD,
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    index_params.add_index(
        field_name=SPARSE_FIELD,
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )
    client.create_index(collection_name=COLLECTION, index_params=index_params)
    # 建完集合+索引立即加载，保证后续 insert 后的 delete/query/search 可用
    client.load_collection(COLLECTION)


def upsert_documents(rows: list[dict]) -> int:
    """插入文档行到 Milvus。

    每行需含：text, dense_vector, city, category, title, section, source。
    注意：sparse_vector 由 BM25 函数自动生成，行内禁止携带该字段。
    """
    client = get_client()
    ensure_collection()
    client.insert(collection_name=COLLECTION, data=rows)
    return len(rows)


def drop_collection() -> None:
    """删除集合（开发期调 schema 用，重建后需重新入库）。"""
    client = get_client()
    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)
