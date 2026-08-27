"""parallel 交付链解禁硬过滤 (2026-08-26 用户定案 "解禁可以做硬过滤").

规则与 legacy FINAL STOCK SCAN 同源 (risk_overlays.share_float_upcoming_scan):
ref=交付日, PIT ann_date<=ref, 未来 30 自然日内解禁累计 >5% 总股本 → 剔除.
parallel 接线差异: 在 TOP-10 排名前从候选池剔除 (第 11 名递补, 清单保持满 10),
而 legacy 是 TOP-N 截断后剔除 (清单可短) — 用户需要 Top10 满额档 (feedback-need-top10).
"""

from __future__ import annotations

import pandas as pd

from scripts import _shortlist_t5_t10 as sl


def _pool(n: int = 12, board: str = "main") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [f"{i + 1:06d}" for i in range(n)],
            "board": board,
            "pred_mag_10d": [0.10 - i * 0.001 for i in range(n)],
            "pred_prob": [0.6 - i * 0.01 for i in range(n)],
        }
    )


def test_unlock_filter_removes_excluded_before_top10(monkeypatch):
    """被咬票不进 TOP-10, 由池内下一名递补 → 清单仍满 10 只."""
    seen: dict = {}

    def _scan(symbols, ref_date, **kwargs):
        seen["symbols"] = list(symbols)
        seen["ref"] = ref_date
        return {"000001"}

    monkeypatch.setattr(sl, "share_float_upcoming_scan", _scan)
    res = sl.unlock_hard_filter(_pool(), pd.Timestamp("2026-08-26"))
    out = sl.rank_and_truncate(res)

    assert "000001" not in set(out["symbol"])
    assert len(out) == 10  # 递补后满额
    assert set(out["symbol"]) == {f"{i + 2:06d}" for i in range(10)}
    assert seen["ref"] == pd.Timestamp("2026-08-26")


def test_unlock_filter_failopen_on_scan_error(monkeypatch):
    """扫描抛异常 → fail-open 放行全部候选 (安全网不因抖动清空名单, legacy 同款)."""

    def _scan(symbols, ref_date, **kwargs):
        raise RuntimeError("cache boom")

    monkeypatch.setattr(sl, "share_float_upcoming_scan", _scan)
    pool = _pool()
    res = sl.unlock_hard_filter(pool, pd.Timestamp("2026-08-26"))
    assert list(res["symbol"]) == list(pool["symbol"])


def test_unlock_filter_no_exclusion_passthrough(monkeypatch):
    monkeypatch.setattr(sl, "share_float_upcoming_scan", lambda s, r, **k: set())
    pool = _pool()
    res = sl.unlock_hard_filter(pool, pd.Timestamp("2026-08-26"))
    assert len(res) == len(pool)


def test_unlock_filter_empty_pool(monkeypatch):
    called = []

    def _scan(symbols, ref_date, **kwargs):
        called.append(symbols)
        return set()

    monkeypatch.setattr(sl, "share_float_upcoming_scan", _scan)
    res = sl.unlock_hard_filter(pd.DataFrame(), pd.Timestamp("2026-08-26"))
    assert res.empty
    assert called == []  # 空池不触发扫描


def test_unlock_filter_zfills_symbols_before_scan(monkeypatch):
    """非补零 symbol 先 zfill(6) 再比对解禁缓存 (缓存侧恒 6 位)."""
    seen = {}

    def _scan(symbols, ref_date, **kwargs):
        seen["symbols"] = list(symbols)
        return {"000001"}

    monkeypatch.setattr(sl, "share_float_upcoming_scan", _scan)
    pool = _pool()
    pool["symbol"] = pool["symbol"].str.lstrip("0")  # 制造非补零 ("000001"→"1")
    res = sl.unlock_hard_filter(pool, pd.Timestamp("2026-08-26"))
    assert all(len(s) == 6 for s in seen["symbols"])
    assert "000001" not in set(res["symbol"].astype(str).str.zfill(6))
