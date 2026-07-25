"""
WORM 日志 (V5.1 铁律 #3: 一切留痕, 检查清单 #15 配套)
==============================================================
信号 / 下单 / 人工动作全记录 — 无法归因 = 无法止损.
JSONL 追加式 (Write Once Read Many), 严禁覆盖旧文件.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

EVENT_TYPES = ("signal", "order", "manual", "risk", "system")


class WormLogger:
    """日内事件 WORM 记录器 (每日一个文件, 追加不覆盖)."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def _path(self, trade_date: str) -> str:
        return os.path.join(self.log_dir, f"intraday_{trade_date}.jsonl")

    def log(self, trade_date: str, event_type: str, payload: dict) -> None:
        """记录一条事件. event_type ∈ signal/order/manual/risk/system."""
        assert event_type in EVENT_TYPES, f"event_type 须为 {EVENT_TYPES}"
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "type": event_type,
            **payload,
        }
        with open(self._path(str(trade_date)), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def read_day(self, trade_date: str) -> list[dict]:
        """读取某日全部事件 (复盘/对账用)."""
        path = self._path(str(trade_date))
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
