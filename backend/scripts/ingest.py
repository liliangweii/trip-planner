"""语料入库主脚本（CLI 壳）：业务逻辑在 app/rag/ingest.py。

用法：
    python scripts/ingest.py --city 北京
    python scripts/ingest.py --city 北京 --category attraction
    python scripts/ingest.py --file path/to/doc.md --city 北京 --category route
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本可直接运行：把 backend 目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from app.rag.ingest import ingest_directory, ingest_file


def main() -> None:
    parser = argparse.ArgumentParser(description="TripRAG 语料入库")
    parser.add_argument("--city", help="目标城市")
    parser.add_argument("--category", help="类别：attraction/food/hotel/route/tips")
    parser.add_argument("--file", help="单文件路径（需配合 --city --category）")
    args = parser.parse_args()

    if args.file:
        if not (args.city and args.category):
            parser.error("使用 --file 时必须提供 --city 和 --category")
        n = ingest_file(args.file, city=args.city, category=args.category)
    elif args.city:
        n = ingest_directory(city=args.city, category=args.category)
    else:
        parser.error("至少提供 --city 或 --file")
        return

    logger.info("入库完成，共 {} chunks", n)


if __name__ == "__main__":
    main()
