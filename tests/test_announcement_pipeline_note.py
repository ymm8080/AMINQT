# -*- coding: utf-8 -*-
"""run_announcement_pipeline 结果登记的冒烟测试 (2026-09-15).

背景: 2026-09-15 事故 —— 日更尾行只测"有值"却答"数据完整"; 同日公告管线的
`fina_indicator: 0 rows` 被误读为"源故障/缓存停更", 实为财报季外的正确终态
(Q2 收官 08-31 后公告窗天然为空). 故在该源返回 0 行时于 msg 附加季节性说明.

同时回归一个真实踩过的坑: 给 msg 加分支时曾整块删掉 `results[src] = {...}`,
导致结果字典为空 —— 报告只剩表头, 无异常无告警. 这里锁住"必须登记".
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_announcement_pipeline as ap  # noqa: E402

_FINA_MSG = "财务指标 PIT (含 ann_date)"


class _FakeSupply:
    """只实现被测分支用到的源; 其余返回空, 不触网."""

    def __init__(self, fina_df: pd.DataFrame):
        self._fina_df = fina_df

    def fetch_fina_indicator(self, start_date=None, end_date=None, refresh=False):
        return self._fina_df

    def fetch_holdertrade(self, start_date=None, end_date=None, refresh=False):
        return pd.DataFrame()

    def fetch_holdernumber(self, start_date=None, end_date=None, refresh=False):
        return pd.DataFrame()


def _run(fina_df: pd.DataFrame):
    with (
        patch.object(ap, "DataSupplyChain", lambda: _FakeSupply(fina_df)),
        patch.object(ap, "_fetch_anns_d", lambda *a, **k: pd.DataFrame()),
        patch.object(ap, "_fetch_forecast_increment", lambda *a, **k: pd.DataFrame()),
    ):
        return ap.fetch_announcement_data("20260915", refresh=False)


def test_fina_empty_row_is_registered_and_annotated():
    """0 行时必须登记结果, 且 msg 说明这是财报季外预期, 非源故障."""
    results, _ = _run(pd.DataFrame())
    assert "fina_indicator" in results, "结果未登记 — results[src] 被删则报告静默变空"
    r = results["fina_indicator"]
    assert r["rows"] == 0
    assert r["status"] == "empty"
    assert "非源故障" in r["msg"]
    assert "报告期" in r["msg"]


def test_fina_nonempty_has_no_seasonal_note():
    """有数据时不得附加季节性说明 (否则是真故障时的误导)."""
    df = pd.DataFrame({"symbol": ["000001"], "ann_date": [pd.Timestamp("2026-08-31")]})
    results, _ = _run(df)
    r = results["fina_indicator"]
    assert r["rows"] == 1
    assert r["status"] == "ok"
    assert "非源故障" not in r["msg"]


def test_all_sources_registered():
    """每个被遍历的源都要登记 —— 漏登记同样造成报告静默缺行."""
    results, _ = _run(pd.DataFrame())
    for src in ap.ANNOUNCEMENT_SOURCES:
        assert src in results, f"{src} 未登记"
        assert set(results[src]) >= {"rows", "status", "msg"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
