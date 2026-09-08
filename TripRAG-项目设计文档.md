# TripRAG 智游助手 —— 基于 LangChain + Milvus 的 RAG 智能旅行规划平台

> 项目设计文档 | 版本 v1.0

---

## 1. 项目概述

### 1.1 一句话定位

一个 **RAG 深度驱动**的 AI 旅行规划平台：以 LangChain 编排 LLM 工作流，以 Milvus 为向量知识库检索真实旅行攻略语料，结合高德地图工具调用，生成**有据可查、可溯源**的个性化旅行计划与行程问答。

### 1.2 项目背景（为什么做 RAG）

传统 LLM 直接生成旅行计划存在三大痛点：

| 痛点 | 表现 | 本项目的解法 |
|---|---|---|
| **幻觉** | 编造不存在的景点、错误的门票价格与开放时间 | 行程内容必须来自知识库检索命中的真实攻略语料，带引用溯源 |
| **知识过时** | 模型训练截止后的景区政策、票价变化无从知晓 | 语料可随时增删，知识库与模型解耦，更新攻略无需重训模型 |
| **无个性化** | 通用回答，无法沉淀领域知识 | 城市维度的攻略语料库 + 元数据过滤（城市/类别/标签），检索即个性化 |

### 1.3 核心能力清单

- 📚 **旅行知识库**：攻略/游记语料的「加载 → 切分 → 向量化 → 入库」全管道，支持 Markdown/PDF/网页多源语料
- 🔍 **混合检索**：Milvus 稠密向量（BGE-M3）+ 稀疏向量（BM25）双路召回，RRF 融合 + CrossEncoder 重排序
- 🤖 **LangChain Agent**：工具调用 Agent 自动编排「知识库检索 → 高德 POI/天气/路线 → 综合规划」
- 💬 **多轮行程问答**：带会话记忆的流式 RAG 问答，答案附引用来源
- 🗺️ **地图可视化**：高德 JS API 展示行程打点（沿用成熟前端方案）
- 📐 **可评测**：RAGAS 框架量化评估（忠实度/答案相关性/上下文召回率），持续优化检索质量

### 1.4 与原项目（HelloAgents 版）的关系

本项目是原「HelloAgents 智能旅行助手」的**架构升级版**，保留业务形态与前端方案，重写 AI 编排层并新增 RAG 能力：

| 维度 | 原项目 | 本项目 |
|---|---|---|
| Agent 框架 | HelloAgents（SimpleAgent × 4） | **LangChain**（LCEL 链 + Tool-Calling Agent） |
| 知识来源 | 仅高德地图 MCP 实时数据 | **Milvus 向量知识库（新增核心）** + 高德地图工具 |
| 检索能力 | 无 | 混合检索 + 重排序 + 引用溯源 |
| 生成依据 | 纯 LLM 生成（易幻觉） | **RAG 增强生成**（先检索后生成，可溯源） |
| 会话能力 | 单轮请求-响应 | **多轮对话记忆**（RunnableWithMessageHistory） |
| 评测 | 无 | RAGAS 检索/生成质量评测 |
| 保留不变 | FastAPI 后端、Vue3+TS+Vite+AntD 前端、高德地图集成、Pydantic 数据模型 | 同左 |

---

## 2. 总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     前端 Vue 3 + TypeScript + Vite               │
│        规划表单页 / RAG 问答页 / 行程结果页（高德地图可视化）        │
└───────────────┬─────────────────────────────┬───────────────────┘
                │ REST (JSON)                  │ SSE (流式问答)
┌───────────────▼─────────────────────────────▼───────────────────┐
│                        FastAPI 网关层                             │
│   /api/trip/plan   /api/chat(SSE)   /api/kb/*   /api/search     │
└───────────────┬──────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────┐
│                     LangChain 编排层（核心）                       │
│                                                                  │
│  ┌───────────────┐   ┌──────────────────────────────────────┐   │
│  │ RAG 问答链     │   │ 行程规划 Agent (create_tool_calling_  │   │
│  │ LCEL + Memory │   │ agent + AgentExecutor)               │   │
│  │ 流式输出       │   │  ├─ Tool① search_travel_knowledge    │   │
│  └──────┬────────┘   │  │    （Milvus 混合检索，带引用）      │   │
│         │            │  ├─ Tool② amap_poi_search            │   │
│         │            │  ├─ Tool③ amap_weather               │   │
│         │            │  ├─ Tool④ amap_route_plan            │   │
│         │            │  └─ 结构化输出 → TripPlan JSON        │   │
│         │            └──────────────┬───────────────────────┘   │
└─────────┼───────────────────────────┼───────────────────────────┘
          │                           │
┌─────────▼─────────────┐   ┌─────────▼─────────────┐   ┌────────────────────┐
│   Milvus 向量数据库     │   │  高德地图开放平台       │   │  LLM 服务          │
│  travel_knowledge 集合 │   │  POI/天气/路线 REST    │   │  OpenAI 兼容接口    │
│  稠密+稀疏混合检索      │   └───────────────────────┘   │  (DeepSeek 等)     │
└─────────▲─────────────┘                               └────────────────────┘
          │ 离线入库管道
┌─────────┴──────────────────────────────────────────────────────────┐
│  语料加载(MD/PDF/网页) → 结构感知切分 → 元数据抽取 → BGE-M3 向量化   │
│  → 稠密+稀疏向量 → Milvus 入库（脚本 scripts/ingest.py）            │
└────────────────────────────────────────────────────────────────────┘
```

**分层职责**：

- **网关层（FastAPI）**：参数校验、CORS、SSE 流式透传、统一异常处理，不含业务逻辑
- **编排层（LangChain）**：所有 AI 逻辑所在层——检索链、Agent、记忆、提示词模板
- **存储层（Milvus）**：向量 + 元数据的持久化与高性能检索，与编排层仅通过 Retriever 接口耦合
- **离线管道**：与在线服务完全解耦，语料更新只重跑入库脚本

---

## 3. 技术选型及理由

| 类别 | 选型 | 版本 | 理由 |
|---|---|---|---|
| Agent/编排框架 | **LangChain**（langchain-core / langchain / langchain-community） | ≥0.3.x | LCEL 声明式编排、生态最全的 Retriever/Loader/Tool 体系，业界主流，便于简历对标 |
| 向量数据库 | **Milvus**（milvus-standalone via Docker） | ≥2.4 | 原生支持**稠密+稀疏混合检索**与标量过滤、Partition 分区、亿级扩展性，Milvus 2.4+ 支持 full-text/BM25 |
| Milvus SDK | pymilvus + **langchain-milvus** | latest | 官方 LangChain VectorStore 集成，混合检索 API 成熟 |
| Embedding | **BAAI/bge-m3** | - | 单模型同时输出稠密(1024维)+稀疏+colbert 三路向量，中文效果标杆；本地推理无 API 成本（可切换为 API 型 Embedding，见 §5.4） |
| 重排序 | **BAAI/bge-reranker-v2-m3** | - | CrossEncoder 精排，显著提升 top-k 命中质量 |
| LLM | OpenAI 兼容接口（DeepSeek / Qwen / GPT 任一） | - | ChatOpenAI 类直接兼容，`with_structured_output` 输出 JSON |
| Web 框架 | FastAPI + Uvicorn | 同原项目 | SSE 流式支持好、Pydantic v2 原生集成 |
| 会话记忆 | RunnableWithMessageHistory + SQLChatMessageHistory(SQLite) | - | 轻量起步，可平滑换 Redis/Postgres |
| 前端 | Vue 3.5 + TypeScript + Vite 6 + Ant Design Vue 4 + vue-router + Axios | 同原项目 | 沿用已验证方案，聚焦后端差异化 |
| 地图 | @amap/amap-jsapi-loader | 同原项目 | - |
| 评测 | RAGAS | ≥0.2 | RAG 评测事实标准：忠实度/相关性/上下文精确率/召回率 |
| 部署 | Docker Compose（Milvus + etcd + MinIO + Attu） | - | 官方 standalone 编排，一键起库 |

---

## 4. 数据模型设计（核心 Pydantic Schemas）

```python
# app/models/schemas.py（关键模型，完整版随代码实现）

class TripRequest(BaseModel):
    city: str                          # 目的地城市
    start_date: str                    # YYYY-MM-DD
    end_date: str
    travel_days: int = Field(ge=1, le=15)
    transportation: str = "公共交通"    # 公共交通/自驾/步行偏好
    accommodation: str = "经济型"       # 住宿偏好
    preferences: list[str] = []        # 旅行风格标签: 历史文化/美食/亲子/徒步...
    free_text_input: str = ""          # 自由补充需求

class Citation(BaseModel):             # ★ RAG 溯源（新增）
    chunk_id: str                      # Milvus 主键
    source: str                        # 语料标题/出处
    url: str = ""
    snippet: str                       # 命中原文片段（截断展示）

class Attraction(BaseModel):
    name: str
    address: str
    location: Location                 # {longitude, latitude}
    visit_duration: int                # 分钟
    description: str
    category: str
    ticket_price: float = 0
    citations: list[Citation] = []     # ★ 该景点信息来源

class DayPlan(BaseModel):
    date: str
    day_index: int
    description: str
    transportation: str
    accommodation: str
    hotel: Hotel | None
    attractions: list[Attraction]
    meals: list[Meal]                  # breakfast/lunch/dinner

class TripPlan(BaseModel):
    city: str
    start_date: str
    end_date: str
    days: list[DayPlan]
    weather_info: list[WeatherInfo]
    overall_suggestions: str
    budget: Budget
    references: list[Citation] = []    # ★ 全局引用列表（前端展示"参考攻略"）
```

---

## 5. RAG 核心设计（本项目灵魂）

### 5.1 知识库规划

| 维度 | 设计 |
|---|---|
| 语料来源 | ① 开源游记/攻略数据集（如马蜂窝游记公开数据集）；② 自整理 Markdown 攻略；③ WebBaseLoader 抓取公开旅行网站页面；④ 冷启动可用 LLM 批量生成种子语料再人工校对 |
| 语料类别 | `attraction`（景点）/ `food`（美食）/ `hotel`（住宿）/ `route`（路线玩法）/ `tips`（实用贴士：交通卡、防坑、最佳季节） |
| 组织方式 | 按**城市**分目录存放：`knowledge_base/{city}/{category}/*.md`，文件名即语料标题 |
| 规模目标 | 一期 5 个热门城市 × 每城 20~50 篇文档，切分后 2000~5000 chunks |

### 5.2 数据处理管道（离线）

```
原始语料 (md/pdf/html)
   │ ① Loader: UnstructuredMarkdownLoader / PyPDFLoader / WebBaseLoader
   ▼
原始 Document
   │ ② 结构感知切分：MarkdownHeaderTextSplitter(按 #/##/### 标题)
   │    → 子块再走 RecursiveCharacterTextSplitter(chunk_size=800, overlap=120)
   │    ★ 标题层级写入 metadata.title_path，保证语义块完整
   ▼
Chunks [{text, metadata}]
   │ ③ 元数据增强：从目录/文件名解析 city、category；正则/LLM 抽取
   │    景点名、评分、价位标签（可选）
   ▼
带元数据 Chunks
   │ ④ 向量化：BGEM3EmbeddingFunction → dense(1024) + sparse(BM25词权重)
   ▼
稠密 + 稀疏向量
   │ ⑤ upsert 入 Milvus（auto_id，按 city 做 Partition Key）
   ▼
travel_knowledge 集合
```

入库脚本骨架（`scripts/ingest.py`）：

```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient

headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
sub_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

def load_and_split(path: str, city: str, category: str) -> list[dict]:
    docs = UnstructuredMarkdownLoader(path).load()
    rows = []
    for doc in docs:
        for sec in MarkdownHeaderTextSplitter(headers).split_text(doc.page_content):
            for chunk in sub_splitter.split_text(sec.page_content):
                rows.append({
                    "text": chunk,
                    "city": city,
                    "category": category,
                    "title": sec.metadata.get("h1", ""),
                    "section": sec.metadata.get("h2", ""),
                    "source": path,
                })
    return rows

# embedding + upsert（sparse 用 BGE-M3 内置 BM25 权重）
```

### 5.3 Milvus 集合设计

**Collection：`travel_knowledge`**

| 字段 | 类型 | 说明 |
|---|---|---|
| id | INT64, PK, auto_id | 主键 |
| text | VARCHAR(8192) | chunk 原文 |
| dense_vector | FLOAT_VECTOR(1024) | BGE-M3 稠密向量 |
| sparse_vector | SPARSE_FLOAT_VECTOR | BGE-M3 稀疏向量（BM25 权重） |
| city | VARCHAR(64), **partition_key** | 城市 → 查询自动路由分区，大幅缩小扫描范围 |
| category | VARCHAR(32) | attraction/food/hotel/route/tips |
| title | VARCHAR(256) | 文档标题（h1） |
| section | VARCHAR(256) | 所属小节（h2） |
| source | VARCHAR(512) | 原文出处路径/URL |

**索引与检索参数**：

| 项 | 配置 |
|---|---|
| dense_vector 索引 | HNSW（M=16, efConstruction=200），metric=**COSINE** |
| sparse_vector 索引 | SPARSE_INVERTED_INDEX，metric=**BM25** |
| 融合策略 | **RRFRanker(k=60)**（比 WeightedRanker 对参数不敏感，作为默认） |
| 检索 top_k | 召回 20 → 重排序取 5 |

### 5.4 检索策略（在线 Retriever）

四级流水线，全部用 LangChain 组件拼装：

```python
# app/rag/retriever.py（骨架）
from langchain_milvus import MilvusVectorStore
from langchain.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# ① 查询改写：用户口语化提问 → 多角度检索查询（解决"查询-语料表述不对称"）
multi_query = MultiQueryRetriever.from_llm(retriever=dense_retriever, llm=llm)

# ② 混合召回：稠密 + 稀疏，RRF 融合
hybrid = MilvusVectorStore(..., vector_field=["dense_vector", "sparse_vector"])

# ③ 标量过滤：表达式 "city == '北京' and category in ['attraction','tips']"
#    （由 Agent/链路根据 TripRequest 自动生成 filter 表达式）

# ④ 精排：bge-reranker-v2-m3 对 top20 重排，取 top5 进 Prompt
```

**Embedding 方案二选一**（文档默认方案 A）：

- **方案 A（默认）**：本地 BGE-M3（sentence-transformers / pymilvus model），零 API 成本、稠密稀疏一体
- **方案 B**：API 型 Embedding（硅基流动/OpenAI 兼容 `/embeddings`），零 GPU 依赖，但只有稠密向量 → 检索退化为纯 ANN + 标量过滤，稀疏路用 Milvus 2.4+ 内置 BM25 函数补齐

### 5.5 生成与引用

- Prompt 结构：`系统角色 → 检索到的攻略上下文（编号 [1][2]...） → 高德实时数据 → 任务指令（严格 JSON Schema）`
- **硬约束**：`overall_suggestions` 与每个 `Attraction.description` 必须基于 `[n]` 引用内容撰写；模型通过 `with_structured_output(TripPlan)` 输出
- 解析后校验：引用的 chunk_id 必须存在于本次召回集合，否则剥除该引用（防幻觉引用）

---

## 6. LangChain 编排设计

### 6.1 模型层（app/services/llm_service.py）

```python
from langchain_openai import ChatOpenAI

def get_llm(streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,           # e.g. deepseek-chat
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0.3,
        streaming=streaming,
    )
```

### 6.2 RAG 问答链（/api/chat，流式）

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory

prompt = ChatPromptTemplate.from_messages([
    ("system", TRAVEL_QA_SYSTEM_PROMPT),          # 含“仅依据[编号]上下文回答，并标注引用”约束
    MessagesPlaceholder("history"),
    ("human", "上下文：\n{context}\n\n问题：{question}"),
])

def format_docs(docs):                              # 带编号拼装上下文
    return "\n\n".join(f"[{i+1}] ({d.metadata['title']}) {d.page_content}"
                       for i, d in enumerate(docs))

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

chat_service = RunnableWithMessageHistory(          # 多轮记忆，按 session_id 隔离
    rag_chain, get_session_history,
    input_messages_key="question", history_messages_key="history",
)
# FastAPI 侧用 chat_service.stream(...) 以 SSE 推送 token
```

### 6.3 行程规划 Agent（/api/trip/plan）

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent

tools = [
    search_travel_knowledge_tool,   # 包装 §5.4 Retriever，入参{query, city, category}
    amap_poi_search_tool,           # httpx 直调高德 Web 服务 API（POI 搜索）
    amap_weather_tool,              # 高德天气
    amap_route_plan_tool,           # 高德路线规划
]

agent = create_tool_calling_agent(llm=get_llm(), tools=tools,
                                  prompt=PLANNER_AGENT_PROMPT)   # ReAct 式思考+工具调用
executor = AgentExecutor(agent=agent, tools=tools, verbose=True,
                         max_iterations=8, handle_parsing_errors=True)

# 执行流：先 search_travel_knowledge 拿“该城市的玩法与贴士”（带引用）
#        → amap POI/天气/路线补实时数据 → 生成 TripPlan JSON
# 输出经 with_structured_output(TripPlan) 解析；失败走三级降级（同原项目策略）
```

**设计说明**：
- 知识库检索作为 **Agent 的第一个工具**而非前置固定步骤——让模型根据请求自主决定检索几个 query（如偏好含"美食"时追加 food 类别检索）
- 高德工具由 LangChain `@tool` 装饰器封装 httpx 直调（原项目的 MCP 方案可选保留，二者不冲突）

### 6.4 会话记忆

- `get_session_history(session_id)` → SQLite 表存储（`SQLChatMessageHistory`）
- 记忆窗口裁剪：保留最近 10 轮，超限做摘要压缩（`ConversationSummaryBufferMemory` 思路，LCEL 实现）
- `/api/trip/plan` 与 `/api/chat` 共用 session_id，实现"先聊需求 → 一键生成计划"的连续体验

---

## 7. API 设计

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/trip/plan` | 生成旅行计划（Agent 流程，同步 JSON 返回） |
| POST | `/api/chat` | RAG 行程问答，**SSE 流式**返回 |
| POST | `/api/kb/ingest` | 上传/指定语料入库（multipart 文件或城市目录重扫） |
| GET | `/api/kb/documents` | 知识库文档列表（按 city/category 过滤） |
| DELETE | `/api/kb/documents/{doc_id}` | 删除某文档的全部 chunks |
| GET | `/api/search` | 裸检索调试端点：`?query=&city=&category=&top_k=`，返回 chunks + 分数（前端可做"检索透镜"调试页） |
| GET | `/health` | 健康检查（含 Milvus 连通性） |

**请求/响应示例**：

```jsonc
// POST /api/trip/plan 请求体 = TripRequest
// 响应（截取）：
{
  "success": true,
  "data": {
    "city": "北京",
    "days": [
      {
        "date": "2025-05-01",
        "attractions": [{
          "name": "故宫博物院",
          "description": "明清两代皇宫……建议上午入场避开人流 [1]",
          "ticket_price": 60,
          "citations": [{ "chunk_id": "45398...", "source": "北京故宫游览全攻略", "snippet": "门票旺季60元……" }]
        }]
      }
    ],
    "references": [ /* 全局引用列表，前端渲染"参考攻略"卡片 */ ]
  }
}
```

---

## 8. 项目目录结构

```
trip-rag-planner/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI 入口 + CORS + 路由注册
│   │   ├── config.py                  # pydantic-settings 配置
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── trip.py            # POST /api/trip/plan
│   │   │       ├── chat.py            # POST /api/chat (SSE)
│   │   │       ├── kb.py              # /api/kb/* 知识库管理
│   │   │       └── search.py          # /api/search 调试检索
│   │   ├── chains/
│   │   │   ├── rag_qa_chain.py        # §6.2 RAG 问答链
│   │   │   └── planner_agent.py       # §6.3 规划 Agent
│   │   ├── rag/
│   │   │   ├── loaders.py             # 多源文档加载
│   │   │   ├── splitters.py           # 结构感知切分
│   │   │   ├── embeddings.py          # BGE-M3 封装
│   │   │   ├── milvus_store.py        # 集合创建/索引/upsert
│   │   │   ├── retriever.py           # 混合检索+重排序流水线
│   │   │   └── prompts.py             # 提示词集中管理
│   │   ├── tools/
│   │   │   ├── knowledge_search.py    # Tool① 知识库检索
│   │   │   └── amap.py                # Tool②③④ 高德工具(@tool 封装)
│   │   ├── memory/history.py          # 会话记忆
│   │   ├── models/schemas.py          # Pydantic 模型
│   │   └── services/{llm_service.py, amap_service.py}
│   ├── scripts/
│   │   ├── ingest.py                  # 入库主脚本
│   │   ├── seed_corpus.py             # 冷启动语料生成
│   │   └── eval_ragas.py              # RAGAS 评测
│   ├── knowledge_base/                # 语料：{city}/{category}/*.md
│   ├── requirements.txt
│   └── .env.example
├── frontend/                          # 同原项目结构 (Vue3+TS+Vite+AntD+AMap)
├── deploy/
│   └── docker-compose.yml             # milvus-standalone + etcd + minio + attu
└── README.md
```

---

## 9. 环境与部署

### 9.1 基础环境

Python 3.12+ / Node.js 18+ / Docker & Docker Compose / （方案 A 需）CUDA GPU 或较强 CPU

### 9.2 Milvus 启动（deploy/docker-compose.yml）

采用 Milvus 官方 standalone 编排：`milvus-standalone`（端口 19530）+ `etcd` + `minio`，可选 `attu`（图形化管理界面，端口 8000）。`docker compose up -d` 一键启动。

### 9.3 配置项（backend/.env.example）

| 变量 | 说明 | 示例 |
|---|---|---|
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | OpenAI 兼容 LLM | sk-xxx / https://api.deepseek.com/v1 / deepseek-chat |
| `MILVUS_URI` | Milvus 连接地址 | http://localhost:19530 |
| `MILVUS_COLLECTION` | 集合名 | travel_knowledge |
| `EMBEDDING_PROVIDER` | local_bge / api | local_bge |
| `EMBEDDING_API_BASE` / `EMBEDDING_API_KEY` / `EMBEDDING_MODEL` | 方案 B 时使用 | - |
| `RERANKER_MODEL` | 重排序模型 | BAAI/bge-reranker-v2-m3 |
| `AMAP_API_KEY` | 高德 Web 服务 Key | - |
| `SQL_HISTORY_URL` | 会话记忆存储 | sqlite:///./chat_history.db |

### 9.4 启动顺序

```bash
# 1. 起 Milvus
cd deploy && docker compose up -d
# 2. 后端
cd backend && pip install -r requirements.txt && cp .env.example .env
python scripts/ingest.py --city 北京            # 首次建集合+入库
uvicorn app.main:app --reload --port 8000
# 3. 前端
cd frontend && npm install && npm run dev        # http://localhost:5173
```

---

## 10. 开发里程碑（建议 6 个阶段）

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| P1 基础设施 | Milvus 部署、集合/索引创建、配置层、LLM 连通 | `/health` 返回 Milvus 连通 OK |
| P2 入库管道 | Loader/Splitter/Metadata/向量化/入库脚本 | 某城市 20+ 文档入库，Attu 中可查 |
| P3 RAG 问答链 | 混合检索 + 重排 + LCEL 链 + SSE | `/api/chat` 多轮流式回答且带引用 |
| P4 Agent 规划 | 4 个工具封装 + Tool-Calling Agent + 结构化输出 + 降级 | `/api/trip/plan` 产出带 citations 的 TripPlan |
| P5 前端 | 迁移原前端 + 新增问答页/引用卡片/检索调试页 | 全链路演示可跑 |
| P6 评测优化 | RAGAS 基线评测 → 调参（chunk_size/top_k/融合权重） | 忠实度 & 上下文召回率相对基线提升并记录 |

---

## 11. 评测方案（RAGAS）

- **构建金标集**：50~100 条（问题 → 期望要点 → 应引用的语料）
- **核心指标**：
  - `faithfulness` 忠实度（答案是否忠于检索上下文——直接衡量幻觉）
  - `answer_relevancy` 答案相关性
  - `context_precision` / `context_recall` 上下文精确率与召回率（衡量检索质量）
- **调参实验记录表**：chunk_size(500/800/1200) × top_k(3/5/8) × RRF/加权融合 → 各指标对比，选最优配置写入 README

---

## 12. 风险与优化路线

| 风险 | 应对 |
|---|---|
| 语料版权 | 仅使用开源数据集/自整理/LLM 生成语料，入库记录 source 溯源 |
| 检索质量不足（跨文档推理弱） | 二期引入 **Parent-Child 分块**（小块检索、大块上下文）与 **RAG-Fusion** |
| 本地 Embedding 算力不足 | 切换方案 B（API Embedding）+ Milvus 内置 BM25 |
| Agent 迭代不稳定 | max_iterations + 解析错误自愈 + 失败降级模板计划 |
| 多城市扩展 | city 作为 Partition Key 天然水平扩展；语料管线参数化 |

**演进方向（二期+）**：LangGraph 状态机重构多 Agent 编排 → GraphRAG（景点-美食-路线知识图谱）→ 用户行程反馈回流语料库（自进化知识库）→ Redis 语义缓存降低重复检索成本。

---

## 13. 附录：后端核心依赖（requirements.txt 草案）

```
# LangChain 生态
langchain-core>=0.3
langchain>=0.3
langchain-community>=0.3
langchain-openai>=0.2
langchain-milvus>=0.1
langchain-text-splitters>=0.3

# 向量库与模型
pymilvus>=2.4
sentence-transformers>=3.0    # BGE-M3 / reranker 本地推理
FlagEmbedding>=1.3            # 可选：BGE 系官方推理库

# Web 与工程
fastapi>=0.115
uvicorn[standard]>=0.32
pydantic>=2.7
pydantic-settings>=2.3
httpx>=0.27
python-dotenv>=1.0
loguru>=0.7
SQLAlchemy>=2.0               # 会话记忆存储
ragas>=0.2                    # 评测
```
