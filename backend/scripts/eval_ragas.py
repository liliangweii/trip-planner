"""RAGAS 评测脚本：忠实度 / 答案相关性 / 上下文精确率与召回率。

对应设计文档 §11 评测方案。

用法：
    python scripts/eval_ragas.py --goldset data/goldset.json
    python scripts/eval_ragas.py --goldset data/goldset.json --chunk-size 800 --top-k 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger


def load_goldset(path: str) -> list[dict]:
    """加载金标集：[{question, ground_truth, ground_contexts}]。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_eval(goldset: list[dict], chunk_size: int, top_k: int) -> dict:
    """运行 RAGAS 评测，返回各指标。"""
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "RAGAS 评测需要 ragas 与 datasets，请安装：pip install ragas datasets"
        ) from exc

    # TODO: P6 阶段：对每条金标运行 RAG 链，收集 question/answer/contexts/ground_truth
    rows = []
    for item in goldset:
        rows.append(
            {
                "question": item["question"],
                "answer": "",  # TODO: 填入 RAG 答案
                "contexts": [],  # TODO: 填入召回的上下文
                "ground_truth": item["ground_truth"],
            }
        )

    ds = Dataset.from_list(rows)
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return {k: float(v) for k, v in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS 评测")
    parser.add_argument("--goldset", required=True, help="金标集 JSON 路径")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    goldset = load_goldset(args.goldset)
    logger.info("加载金标 {} 条 | chunk_size={} top_k={}", len(goldset), args.chunk_size, args.top_k)

    metrics = run_eval(goldset, chunk_size=args.chunk_size, top_k=args.top_k)
    logger.info("评测结果：")
    for k, v in metrics.items():
        logger.info("  {} = {:.4f}", k, v)


if __name__ == "__main__":
    main()
