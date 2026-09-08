# 旅行知识库目录

按 **城市 / 类别** 组织语料：`{city}/{category}/*.md`

- `attraction` 景点
- `food` 美食
- `hotel` 住宿
- `route` 路线玩法
- `tips` 实用贴士（交通卡、防坑、最佳季节等）

示例：

```
knowledge_base/
├── 北京/
│   ├── attraction/
│   │   └── 故宫博物院游览全攻略.md
│   ├── food/
│   │   └── 北京烤鸭探店指南.md
│   ├── hotel/
│   ├── route/
│   └── tips/
└── 上海/
    └── ...
```

入库命令（见 `scripts/ingest.py`）：

```bash
python scripts/ingest.py --city 北京
```

冷启动可用 LLM 生成种子语料（见 `scripts/seed_corpus.py`）：

```bash
python scripts/seed_corpus.py --city 北京 --category attraction --count 5
```
