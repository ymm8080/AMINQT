"""P21.1: 40笔样本统计 (D.5/D.10, 日内侧路径) + 双档 GT-Score 月度对比 (D.6)."""

from __future__ import annotations

import pytest

import app.intraday.v51.sample_stats as ss
from app.intraday.v51.sample_stats import TradeJournal, TradeRecord
from app.pipeline1 import trade_discipline as td
from app.pipeline1.gt_score import (
    dual_profile_verdict,
    gt_score,
    monthly_gt_scores,
)


# ============================================================
# 40 笔样本统计 (日内侧路径, 逻辑单点同源)
# ============================================================
class TestSampleStatsSingleSource:
    def test_same_source_as_pipeline1(self):
        """单点同源: v51.sample_stats 只是转出口, 禁止第二份统计逻辑."""
        assert ss.TradeJournal is td.TradeJournal
        assert ss.TradeRecord is td.TradeRecord
        assert ss.UNLOCK_MIN_TRADES == td.UNLOCK_MIN_TRADES == 40
        assert ss.UNLOCK_MIN_EXPECTANCY == pytest.approx(0.005)
        assert ss.UNLOCK_MAX_CONSEC_LOSS == 5


def _fill(j: TradeJournal, pnls: list[float]) -> None:
    for i, pnl in enumerate(pnls):
        j.record(
            TradeRecord(
                symbol=f"S{i}",
                signal_grade="A",
                prob_up=0.7,
                rank_score=1.0,
                entry_date="2026-07-01",
                entry_price=10.0,
            )
        )
        j.close_trade(f"S{i}", "2026-07-03", 10.0 * (1 + pnl), 2, pnl)


class TestUnlockViaIntradayPath:
    def test_sample_insufficient_blocks_unlock(self):
        """D.5: 样本不足 40 笔 → 不得解锁也不得调参."""
        j = TradeJournal()
        _fill(j, [0.02] * 39)
        r = j.unlock_check()
        assert not r["unlock"] and "39/40" in r["reason"]

    def test_unlock_double_gate(self):
        """D.10 双闸门: 期望>+0.5%/笔 且 最大连亏≤5, 缺一不可."""
        good = TradeJournal()
        _fill(good, [0.01] * 40)  # 期望+1%, 无连亏
        assert good.unlock_check()["unlock"]
        low_exp = TradeJournal()
        _fill(low_exp, [0.003] * 40)  # 期望+0.3% < 0.5%
        assert not low_exp.unlock_check()["unlock"]
        streak = TradeJournal()
        _fill(streak, [-0.04] * 6 + [0.02] * 34)  # 连亏6 > 5
        assert not streak.unlock_check()["unlock"]


# ============================================================
# 双档 GT-Score 对比 (D.6 档位裁决, 月度)
# ============================================================
class TestMonthlyGtScores:
    def test_monthly_bucket_matches_scalar(self):
        """按月分桶: 每月分数 = 该月子序列直接调用 gt_score."""
        dates = [f"2026-06-{d:02d}" for d in range(1, 11)] + [
            f"2026-07-{d:02d}" for d in range(1, 11)
        ]
        ics = [0.01] * 10 + [0.02] * 10
        tos = [0.3] * 20
        r = monthly_gt_scores(dates, ics, tos)
        assert set(r) == {"2026-06", "2026-07"}
        assert r["2026-06"] == pytest.approx(gt_score([0.01] * 10, [0.3] * 10))
        assert r["2026-07"] == pytest.approx(gt_score([0.02] * 10, [0.3] * 10))


class TestDualProfileVerdict:
    def test_three_consecutive_below_forces_switch(self):
        """D.6: 攻击档连续 3 个月 GT-Score 低于稳定档 → 强制切回 stable."""
        stable = {"2026-05": 1.0, "2026-06": 1.0, "2026-07": 1.0}
        agg = {"2026-05": 0.9, "2026-06": 0.8, "2026-07": 0.7}
        r = dual_profile_verdict(stable, agg)
        assert r["force_switch_to_stable"] and r["trailing_below"] == 3

    def test_non_consecutive_does_not_switch(self):
        """中间一个月反超 → 连续段清零, 不裁决."""
        stable = {"2026-05": 1.0, "2026-06": 1.0, "2026-07": 1.0, "2026-08": 1.0}
        agg = {"2026-05": 0.9, "2026-06": 1.1, "2026-07": 0.8, "2026-08": 0.7}
        r = dual_profile_verdict(stable, agg)
        assert not r["force_switch_to_stable"] and r["trailing_below"] == 2

    def test_insufficient_overlap_no_verdict(self):
        """重叠月份不足 3 个月 → 样本不足, 不得裁决 (安全网#15)."""
        stable = {"2026-06": 1.0, "2026-07": 1.0}
        agg = {"2026-06": 0.5, "2026-07": 0.5}
        r = dual_profile_verdict(stable, agg)
        assert not r["force_switch_to_stable"] and r["overlap_months"] == 2

    def test_only_overlap_months_counted(self):
        """只有两档都有数据的月份参与对比."""
        stable = {"2026-04": 5.0, "2026-05": 1.0, "2026-06": 1.0, "2026-07": 1.0}
        agg = {"2026-05": 0.9, "2026-06": 0.8, "2026-07": 0.7}
        r = dual_profile_verdict(stable, agg)
        assert r["overlap_months"] == 3 and r["force_switch_to_stable"]
