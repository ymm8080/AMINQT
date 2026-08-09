"""PIPELINE1 V3.8 阶段二部件测试: 公告因子 / E10 模拟盘 / #84 死叉假阳性率."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from app.pipeline1.announcement import AnnouncementFactor
from app.pipeline1.bear_protocol import death_cross_false_positive_report
from app.pipeline1.paper_trading import (
    FillRateTracker,
    ShadowListTracker,
    backtest_vs_paper_deviation,
    bad_trade_review,
)


# ============================================================
# 公告因子 (安全网 #17, 附录B)
# ============================================================
class TestAnnouncementFactor:
    def test_keyword_sentiment(self):
        assert AnnouncementFactor.analyze_sentiment("中报预增50%") == 0.5
        assert AnnouncementFactor.analyze_sentiment("大股东减持, 证监会立案") == -1.0
        assert AnnouncementFactor.analyze_sentiment("") == 0.0
        assert AnnouncementFactor.analyze_sentiment("日常经营公告") == 0.0

    def test_manual_entry_and_score_decay(self):
        af = AnnouncementFactor()
        af.add_manual_entry("600519", "2026-07-25", "财报", title="中报预增50%")
        assert af.compute_announce_score("600519", "2026-07-25") == pytest.approx(0.5)
        # 时间衰减: 2 天后 0.5×0.8² = 0.32
        assert af.compute_announce_score("600519", "2026-07-27") == pytest.approx(
            0.5 * 0.64
        )
        # 超过 5 日 → 0
        assert af.compute_announce_score("600519", "2026-08-05") == 0.0

    def test_event_window_b4(self):
        af = AnnouncementFactor()
        af.add_manual_entry("600519", "2026-07-25", "财报", score=0.5)
        assert af.is_event_window("600519", "2026-07-23")  # 前2日 (窗口前3日)
        assert af.is_event_window("600519", "2026-07-26")  # 后1日
        assert not af.is_event_window("600519", "2026-07-27")  # 后2日出窗
        af.add_manual_entry("600000", "2026-07-25", "解禁", score=-0.5)
        assert af.is_event_window("600000", "2026-07-20")  # 前5日
        assert af.is_event_window("600000", "2026-07-28")  # 后3日
        assert not af.is_event_window("600000", "2026-07-29")  # 后4日出窗
        # "其他" 类型不进事件窗口
        af.add_manual_entry("600001", "2026-07-25", "其他", score=0.5)
        assert not af.is_event_window("600001", "2026-07-25")

    def test_sector_freeze(self):
        af = AnnouncementFactor()
        for _i, sym in enumerate(["600001", "600002", "600003"]):
            af.add_manual_entry(
                sym, "2026-07-25", "风险提示", score=-0.5, industry="白酒"
            )
        af.add_manual_entry(
            "600004", "2026-07-25", "风险提示", score=-0.5, industry="电池"
        )
        assert af.get_sector_freeze("2026-07-25") == {"白酒"}  # ≥3 只利空才冻结
        assert af.get_sector_freeze("2026-07-26") == set()

    def test_attach_scores_and_event_blacklist(self):
        af = AnnouncementFactor()
        af.add_manual_entry("600519", "2026-07-25", "财报", title="预增")
        cands = pd.DataFrame({"symbol": ["600519", "600000"]})
        out = af.attach_scores(cands, "2026-07-25")
        # 财报事件窗口内 → announce_score 置 -1.0 (禁买标记)
        assert out.loc[out["symbol"] == "600519", "announce_score"].iloc[0] == -1.0
        assert out.loc[out["symbol"] == "600519", "event_window"].iloc[0]
        assert out.loc[out["symbol"] == "600000", "announce_score"].iloc[0] == 0.0

    def test_persistence_roundtrip(self, tmp_path):
        af = AnnouncementFactor(str(tmp_path))
        af.add_manual_entry("600519", "2026-07-25", "重组", score=0.8)
        af2 = AnnouncementFactor(str(tmp_path))
        df = af2.load_announcements("600519", "2026-07-01", "2026-07-31")
        assert len(df) == 1 and df["announce_score"].iloc[0] == 0.8
        # B.5 schema 列齐全
        assert {
            "symbol",
            "announce_date",
            "announce_type",
            "announce_score",
            "event_window_flag",
        } <= set(df.columns)


# ============================================================
# E10 成交率门禁
# ============================================================
class TestFillRate:
    def test_rolling_rate_and_gate(self):
        fr = FillRateTracker()
        fr.record("2026-07-24", ["A", "B", "C", "D"], ["A", "B", "C", "D"])
        fr.record("2026-07-25", ["A", "B", "C", "D"], ["A", "B", "C"])  # 75%
        # 加权: 7/8 = 87.5% ≥ 80% → 过
        assert fr.rolling_rate(10) == pytest.approx(0.875)
        assert fr.gate_pass(10)["pass"]

    def test_gate_rejects_below_80pct(self):
        fr = FillRateTracker()
        fr.record("2026-07-25", ["A", "B", "C", "D", "E"], ["A", "B", "C"])  # 60%
        r = fr.gate_pass(10)
        assert not r["pass"] and r["fill_rate"] == pytest.approx(0.6)


# ============================================================
# D.8 影子清单
# ============================================================
class TestShadowList:
    def test_nav_lifecycle(self):
        st = ShadowListTracker(profile="stable", initial_capital=100.0)
        lst = pd.DataFrame({"symbol": ["A", "B"], "weight": [0.5, 0.25]})
        st.record_list("2026-07-24", lst)
        # T+1: 建仓 A(100×0.5=50@10) B(剩余50×0.25=12.5@20), 现金 37.5
        nav1 = st.mark_to_market("2026-07-25", pd.Series({"A": 10.0, "B": 20.0}))
        assert nav1 == pytest.approx(100.0)
        # T+2: A 涨 10% → NAV = 37.5 + 55 + 12.5
        nav2 = st.mark_to_market("2026-07-26", pd.Series({"A": 11.0, "B": 20.0}))
        assert nav2 == pytest.approx(37.5 + 55 + 12.5)
        # T+3: 持仓满 3 日卖出, 全回现金 (B 涨 10%: 12.5→13.75)
        nav3 = st.mark_to_market("2026-07-27", pd.Series({"A": 11.0, "B": 22.0}))
        assert nav3 == pytest.approx(37.5 + 55 + 13.75)
        assert len(st._positions) == 0
        curve = st.nav_curve()
        assert len(curve) == 3 and (curve["profile"] == "stable").all()

    def test_profile_guard(self):
        with pytest.raises(AssertionError):
            ShadowListTracker(profile="unknown")

    def test_zero_cost_price_guard(self):
        # 零价票: 不建仓, NAV 保持现金, 不产生 inf (M3 回归)
        st = ShadowListTracker(profile="stable", initial_capital=100.0)
        lst = pd.DataFrame({"symbol": ["A", "B"], "weight": [0.5, 0.25]})
        st.record_list("2026-07-24", lst)
        nav1 = st.mark_to_market("2026-07-25", pd.Series({"A": 0.0, "B": 20.0}))
        assert nav1 == pytest.approx(100.0)  # A(价0) 不建仓, B 建仓后现金仍在
        assert st._positions.get("A") is None
        assert math.isfinite(nav1)


# ============================================================
# E10 偏差门禁 + 坏单复盘
# ============================================================
class TestDeviationAndBadTrades:
    def test_deviation_pass_and_fail(self):
        bt = pd.Series([1.0, 1.10])  # +10%
        ok = backtest_vs_paper_deviation(bt, pd.Series([1.0, 1.09]))  # 偏差 10%
        assert ok["pass"]
        bad = backtest_vs_paper_deviation(bt, pd.Series([1.0, 1.04]))  # 偏差 60%
        assert not bad["pass"]

    def test_bad_trade_review(self):
        preds = pd.DataFrame(
            {
                "symbol": ["A", "B", "C"],
                "pred_ret_1d": [0.03, 0.03, 0.001],  # C 未达预测大涨门槛
            }
        )
        actual = pd.Series({"A": -0.08, "B": 0.05, "C": -0.09})
        bad = bad_trade_review(preds, actual)
        assert list(bad["symbol"]) == ["A"]  # 只有 A: 预测大涨+实际大跌


# ============================================================
# #84 死叉假阳性率回测
# ============================================================
class TestDeathCrossFalsePositive:
    def _series(self):
        dates = pd.bdate_range("2025-01-01", periods=60)
        close = pd.Series([100.0] * 30 + [94.0] * 5 + [101.0] * 25, index=dates)
        hist = pd.Series(0.0, index=dates)
        hist.iloc[29] = 0.5
        hist.iloc[30] = -0.5  # 第30日死叉 (同时跌破20日线)
        return close, hist

    def test_false_positive_detected(self):
        close, hist = self._series()
        r = death_cross_false_positive_report(close, hist)
        assert r["n_signals"] == 1
        # 信号后 10 日内收复 20 日线 (94→101 快速拉回) → 假阳性
        assert r["n_false_positive"] == 1
        assert r["false_positive_rate"] == 1.0
        assert r["use_two_day_confirm"]  # >50% → 须改双日确认

    def test_no_recovery_not_false_positive(self):
        close, hist = self._series()
        close.iloc[35:] = 90.0  # 死叉后持续阴跌, 不收复 → 真信号
        r = death_cross_false_positive_report(close, hist)
        assert r["n_false_positive"] == 0
        assert not r["use_two_day_confirm"]
