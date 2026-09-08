"""LLM 连通性自检：DeepSeek / OpenAI 兼容接口。

运行方式（从 backend/ 目录运行）：
    cd backend
    python scripts/check_llm.py

复用 app/services/llm_service.py 的 get_llm()，
P6 评测脚本也会复用同一入口，保证配置一致性。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.llm_service import get_llm


def main() -> None:
    llm = get_llm()
    print(f"模型: {settings.llm_model} | base_url: {settings.llm_base_url}")
    print("发送测试消息中...")

    resp = llm.invoke("只回复四个字：链路畅通")
    print("LLM 回复:", resp.content)
    print("LLM 连通 OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"LLM 连通失败: {exc}")
        print("排查提示：")
        print("1. 402 Insufficient Balance → DeepSeek 账户余额不足，去 platform.deepseek.com 充值")
        print("2. 401 → API Key 错误，检查 .env 的 LLM_API_KEY")
        print("3. 404/400 model not found → 模型名错误，检查 LLM_MODEL")
        print("4. ConnectTimeout → 检查网络/代理")
        sys.exit(1)
