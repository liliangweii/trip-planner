"""冷启动语料生成：用 LLM 按「城市 × 类别 × 主题」批量生成种子攻略语料。

对应设计文档 §5.1 语料来源④。产物先落 knowledge_base/{city}/{category}/*.md，
入库前需人工校对（LLM 可能编造票价/开放时间，见 prompt 中的硬约束）。

用法：
    python scripts/seed_corpus.py --city 北京                # 按预设主题表全量生成
    python scripts/seed_corpus.py --city 北京 --category food
    python scripts/seed_corpus.py --city 北京 --category attraction --topic 颐和园
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from app.services.llm_service import get_llm

KB_ROOT = Path(__file__).resolve().parent.parent / "knowledge_base"

# 类别说明（写进 prompt 帮助模型对齐结构）
CATEGORY_DESC = {
    "attraction": "景点类：单景点或几个相近景点的游览攻略（地址/交通/开放时间/门票/看点/建议时长/注意事项）",
    "food": "美食类：某类美食或某片区的觅食攻略（代表店铺/招牌菜/人均/排队情况/点单建议）",
    "hotel": "住宿类：某商圈或风格的住宿攻略（价位区间/交通便利度/适合人群/预订建议）",
    "route": "路线玩法类：一条具体行程路线（按天或按半日组织，串起景点+美食+休息点）",
    "tips": "实用贴士类：交通卡/最佳季节/防坑/特殊人群出行等通用经验",
}

# 北京预设主题表（避开已手写的故宫博物院攻略，防止主题重复）
BEIJING_TOPICS = {
    "attraction": ["颐和园游览全攻略", "八达岭长城游览全攻略", "天坛公园游览全攻略", "雍和宫与周边游览全攻略"],
    "food": ["北京烤鸭去哪吃", "老北京涮羊肉指南", "炸酱面与老字号面馆", "豆汁焦圈卤煮等京味小吃", "簋街夜宵觅食地图"],
    "hotel": ["王府井商圈住宿攻略", "前门大栅栏住宿攻略", "什刹海四合院民宿攻略", "国贸CBD商务住宿攻略"],
    "route": ["北京三日经典路线", "北京亲子游四日路线", "北京胡同Citywalk路线", "北京历史文化深度游路线"],
    "tips": ["北京交通出行攻略", "北京最佳旅行季节", "北京旅游防坑指南", "带老人小孩游北京贴士"],
}

SEED_PROMPT = """请为城市「{city}」写一篇「{topic}」的旅行攻略，类别：{category_desc}。

格式要求（重要）：
1. 以「# {topic}」开头，用 ## 分小节，Markdown 格式，正文 900-1400 字。
2. 结构建议：概述 → 实用信息（地址/交通/门票/开放时间）→ 亮点/推荐 → 注意事项。
3. 只输出 Markdown 正文，不要任何解释或前后缀。

事实准确约束（硬性）：
- 门票价格、开放时间等易变信息若不确定，写「以官方最新公告为准」，严禁编造具体数字。
- 不要杜撰具体人名、电话号码或精确营业时间。
- 内容是给真实游客看的，宁可保守不可虚构。
"""


def generate_one(city: str, category: str, topic: str) -> str:
    """用 LLM 生成一篇种子语料。"""
    llm = get_llm()
    prompt = SEED_PROMPT.format(
        city=city,
        topic=topic,
        category_desc=CATEGORY_DESC.get(category, category),
    )
    for attempt in range(2):
        resp = llm.invoke(prompt)
        content = _clean(resp.content if hasattr(resp, "content") else str(resp))
        if len(content) >= 100:
            return content
        logger.warning("{} 生成过短（{} 字），第 {} 次重试", topic, len(content), attempt + 2)
    return content


def _clean(content: str) -> str:
    """去除代码围栏与首尾空白（部分模型会多包一层 ```markdown ```）。"""
    content = content.strip()
    if content.startswith("```"):
        # 去掉开头的围栏行（```markdown 或 ```）
        content = content.split("\n", 1)[-1].strip()
    if content.endswith("```"):
        content = content[: -3].rstrip()
    return content.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="TripRAG 冷启动语料生成")
    parser.add_argument("--city", default="北京")
    parser.add_argument("--category", choices=list(CATEGORY_DESC), help="仅生成该类别")
    parser.add_argument("--topic", help="单主题生成")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的同名文件")
    args = parser.parse_args()

    topics = BEIJING_TOPICS if args.city == "北京" else {}
    if args.category:
        topics = {args.category: topics.get(args.category, [])}
    if args.topic:
        topics = {args.category or "attraction": [args.topic]}

    total = 0
    for category, topic_list in topics.items():
        if not topic_list:
            logger.warning("{} 无预设主题，跳过。可用 --topic 指定。", category)
            continue
        for topic in topic_list:
            fp = KB_ROOT / args.city / category / f"{topic}.md"
            if fp.exists() and not args.overwrite:
                logger.info("已存在跳过：{}（加 --overwrite 覆盖）", fp.name)
                continue
            logger.info("生成中 [{}] {}", category, topic)
            md = generate_one(args.city, category, topic)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(md, encoding="utf-8")
            total += 1
            logger.info("✓ {}（{} 字）", fp, len(md))

    logger.info("共生成 {} 篇，位于 {}", total, KB_ROOT / args.city)


if __name__ == "__main__":
    main()
