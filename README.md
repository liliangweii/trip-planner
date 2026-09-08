# TripRAG 智游助手

> 基于 LangChain + Milvus 的 RAG 智能旅行规划平台

一个 **RAG 深度驱动**的 AI 旅行规划平台：以 LangChain 编排 LLM 工作流，以 Milvus 为向量知识库检索真实旅行攻略语料，结合高德地图工具调用，生成**有据可查、可溯源**的个性化旅行计划与行程问答。

## ✨ 核心能力

- 📚 **旅行知识库**：攻略/游记语料的「加载 → 切分 → 向量化 → 入库」全管道，支持 Markdown/PDF/网页多源语料
- 🔍 **混合检索**：Milvus 稠密向量（BGE-M3）+ 稀疏向量（BM25）双路召回，RRF 融合 + CrossEncoder 重排序
- 🤖 **LangChain Agent**：工具调用 Agent 自动编排「知识库检索 → 高德 POI/天气/路线 → 综合规划」
- 💬 **多轮行程问答**：带会话记忆的流式 RAG 问答，答案附引用来源
- 🗺️ **地图可视化**：高德 JS API 展示行程打点
- 📐 **可评测**：RAGAS 框架量化评估（忠实度/答案相关性/上下文召回率）

## 📁 项目结构

```
trip-rag-planner/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI 入口 + CORS + 路由注册
│   │   ├── config.py                  # pydantic-settings 配置
│   │   ├── api/routes/                # trip / chat(SSE) / kb / search
│   │   ├── chains/                    # rag_qa_chain / planner_agent
│   │   ├── rag/                       # loaders / splitters / embeddings
│   │   │                              #   / milvus_store / retriever / prompts
│   │   ├── tools/                     # knowledge_search / amap
│   │   ├── memory/history.py          # 会话记忆
│   │   ├── models/schemas.py          # Pydantic 模型
│   │   └── services/                  # llm_service / amap_service
│   ├── scripts/                       # ingest / seed_corpus / eval_ragas
│   ├── knowledge_base/                # 语料：{city}/{category}/*.md
│   ├── requirements.txt
│   └── .env.example
├── frontend/                          # Vue3 + TS + Vite + AntD + AMap
├── deploy/
│   └── docker-compose.yml             # milvus-standalone + etcd + minio + attu
└── README.md
```

## 🚀 快速开始

### 1. 启动 Milvus

```bash
cd deploy
docker compose up -d
# Attu 管理界面：http://localhost:8000
```

### 2. 启动后端

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env        # 填入 LLM_API_KEY / AMAP_API_KEY 等

# 首次建集合 + 入库语料
python scripts/ingest.py --city 北京

# 启动服务
uvicorn app.main:app --reload --port 8000
```

### 3. 启动前端

```bash
cd frontend
npm install
cp .env.example .env        # 填入 VITE_AMAP_KEY
npm run dev                 # http://localhost:5173
```

## 🔧 配置说明

详见 [`backend/.env.example`](backend/.env.example)，关键配置项：

| 变量 | 说明 |
|---|---|
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | OpenAI 兼容 LLM |
| `MILVUS_URI` | Milvus 连接地址 |
| `EMBEDDING_PROVIDER` | `local_bge`（本地 BGE-M3）或 `api`（API 型 Embedding） |
| `AMAP_API_KEY` | 高德 Web 服务 Key |
| `SQL_HISTORY_URL` | 会话记忆存储 |

## 📐 开发里程碑

| 阶段 | 内容 |
|---|---|
| P1 基础设施 | Milvus 部署、集合/索引创建、配置层、LLM 连通 |
| P2 入库管道 | Loader/Splitter/Metadata/向量化/入库脚本 |
| P3 RAG 问答链 | 混合检索 + 重排 + LCEL 链 + SSE |
| P4 Agent 规划 | 4 个工具封装 + Tool-Calling Agent + 结构化输出 |
| P5 前端 | 迁移前端 + 新增问答页/引用卡片/检索调试页 |
| P6 评测优化 | RAGAS 基线评测 → 调参优化 |

## 📚 技术栈

| 类别 | 选型 |
|---|---|
| Agent/编排 | LangChain ≥0.3 |
| 向量数据库 | Milvus ≥2.4（Docker standalone） |
| Embedding | BAAI/bge-m3（稠密+稀疏一体） |
| 重排序 | BAAI/bge-reranker-v2-m3 |
| LLM | OpenAI 兼容接口（DeepSeek / Qwen / GPT） |
| 后端 | FastAPI + Uvicorn + Pydantic v2 |
| 前端 | Vue 3.5 + TypeScript + Vite 6 + Ant Design Vue 4 |
| 地图 | @amap/amap-jsapi-loader |
| 评测 | RAGAS ≥0.2 |

## 📖 文档

完整设计文档见 [`TripRAG-项目设计文档.md`](TripRAG-项目设计文档.md)。
