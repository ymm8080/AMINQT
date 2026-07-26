"""PIPELINE1 V3.8 附录D/E 测试: profiles / D.3纪律状态机 / D.5日志 / 动态引擎."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config.profiles import PROFILES, get_profile, is_aggressive
from app.pipeline1.dynamic_engine import DynamicEngine, DynamicKnobs
from app.pipeline1.trade_discipline import (
    TradeJournal,
    TradeRecord,
    TradingDiscipline,
)


# ============================================================
# 附录D.1 双档位
# ============================================================
class TestProfiles:
    def test_three_profiles_exist(self):
        assert set(PROFILES) == {"stable", "aggressive", "aggressive_b"}
        assert PROFILES["aggressive"]["single_cap"] == 1.00
        assert PROFILES["aggressive"]["stop_loss"] == -0.04
        assert PROFILES["stable"]["stop_loss"] is None

    def test_grade_A_entry_gates(self):
        g = PROFILES["aggressive"]["grade_A_entry"]
        # 入场门槛全部登记 (缺一不可, D.1)
        assert g["prob_up_calibrated"] == 0.68
        assert g["rank_score_top"] == 2
        assert g["pain_prob_max"] == 0.15
        assert g["main_board_only"] and g["event_window_blacklist"]

    def test_get_profile_and_switch(self):
        assert get_profile("stable")["max_positions"] == 15
        assert get_profile()["max_positions"] == 1  # ACTIVE=aggressive_b (B档生产)
        assert is_aggressive() and not is_aggressive("stable")
        with pytest.raises(KeyError):
            get_profile("nonexistent")

    def test_c_profile_locked_until_d10(self):
        """P21.0: 生产档=B档(75%); C档(100%)锁定待D.10+40笔双闸门."""
        from app.config.profiles import ACTIVE_PROFILE, C_PROFILE_LOCKED

        assert C_PROFILE_LOCKED  # 解锁前不得翻 False (D.10 裁决对象)
        assert ACTIVE_PROFILE == "aggressive_b"  # V3.8 定稿生产档
        assert get_profile()["single_cap"] == 0.75  # B档单票75%
        assert get_profile()["prob_entry"] == 0.58  # B档准入线 (Table 4)
        assert get_profile()["daily_loss_limit"] == 0.03  # 75%×4%
        # C档参数仍可读 (影子清单/D.10回测需要), 仅不可启用为生产档
        assert get_profile("aggressive")["single_cap"] == 1.00


# ============================================================
# D.3 三条硬规则
# ============================================================
class TestTradingDiscipline:
    def test_hard_stop_and_time_stop(self):
        d = TradingDiscipline()
        assert d.check_hard_stop(-0.041)  # -4.1% → 无条件砍
        assert not d.check_hard_stop(-0.039)
        assert d.check_time_stop(2, 0.005)  # 2日涨<1% → 撤
        assert not d.check_time_stop(2, 0.015)
        assert not d.check_time_stop(1, 0.005)

    def test_daily_fuse_locks_then_recovers(self):
        d = TradingDiscipline()
        r = d.on_daily_pnl(day=1, daily_pnl_pct=-0.045)  # 触及 4% 日限
        assert r["action"] == "LOCK_TODAY"
        assert not d.can_buy(1)  # 当日锁仓
        d.on_daily_pnl(day=2, daily_pnl_pct=0.01)  # 次日恢复
        assert d.can_buy(2)
        assert d.worm_log()[0]["event"] == "DAILY_FUSE"

    def test_halt_requires_attribution_and_week(self):
        d = TradingDiscipline()
        d.on_nav(day=1, nav=100.0)  # 峰值
        r = d.on_nav(day=2, nav=84.0)  # 回撤 -16% > 15%
        assert r["action"] == "HALT"
        assert not d.can_buy(3)
        # 未满 7 天 → 拒绝
        assert not d.resume_after_halt(day=5, attribution="x")["resumed"]
        # 满 7 天但无归因 → 拒绝 (未归因前不得重启)
        assert not d.resume_after_halt(day=9, attribution="  ")["resumed"]
        # 满 7 天 + 书面归因 → 重启
        assert d.resume_after_halt(day=9, attribution="模式失效, 复核特征")["resumed"]
        assert d.can_buy(9)


# ============================================================
# D.5 交易日志 + 样本统计 + D.10 解锁
# ============================================================
def _fill_journal(j: TradeJournal, pnls: list[float]) -> None:
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


class TestTradeJournal:
    def test_sample_stats(self):
        j = TradeJournal()
        _fill_journal(j, [0.04, 0.06, -0.04, -0.04, 0.05])
        s = j.sample_stats()
        assert s["n_trades"] == 5
        assert s["win_rate"] == pytest.approx(0.6)
        assert s["pl_ratio"] == pytest.approx(0.05 / 0.04)
        assert s["max_consec_loss"] == 2

    def test_unlock_gates(self):
        j = TradeJournal()
        _fill_journal(j, [0.01] * 10)  # 样本不足 40
        assert not j.unlock_check()["unlock"]
        j2 = TradeJournal()
        _fill_journal(j2, [0.01] * 40)  # 期望 +1% > 0.5%, 无连亏
        assert j2.unlock_check()["unlock"]
        j3 = TradeJournal()
        _fill_journal(j3, [-0.02] * 6 + [0.01] * 34)  # 连亏 6 > 5 → 否决
        assert not j3.unlock_check()["unlock"]


# ============================================================
# 附录E 动态参数引擎 (影子模式)
# ============================================================
class TestDynamicEngine:
    def test_per_stock_calc_full_position_natural(self):
        eng = DynamicEngine()
        # stop=4% → position = 4%/4% = 100% (满仓是自然解, 非设定)
        out = eng.per_stock_calc(p=0.70, pred_q50=0.08, atr_pct=0.03)
        assert out["entry"]
        assert out["stop"] == pytest.approx(max(1.2 * 0.03, 0.08 / 2))
        assert out["position"] == pytest.approx(1.0)
        assert out["daily_fuse"] == pytest.approx(out["position"] * out["stop"])

    def test_high_vol_shrinks_position(self):
        eng = DynamicEngine()
        # ATR 大 → stop 宽 → 仓位自动缩 (治"满仓全是高波动股")
        out = eng.per_stock_calc(p=0.75, pred_q50=0.20, atr_pct=0.06)
        assert out["entry"]  # RR=2.0, kelly=0.625 过闸
        assert out["stop"] == pytest.approx(0.10)
        assert out["position"] == pytest.approx(0.4)  # 4%/10% → 自动缩仓

    def test_entry_gates(self):
        eng = DynamicEngine()
        assert not eng.per_stock_calc(0.60, 0.08, 0.03)["entry"]  # p<0.65
        assert not eng.per_stock_calc(0.70, 0.08, 0.03, pain_prob=0.20)["entry"]
        # kelly 否决: p 刚过线但 RR 低 → kelly<0.25
        assert not eng.per_stock_calc(0.65, 0.024, 0.03)["entry"]

    def test_damped_score(self):
        s = pd.Series([0.10, 0.08])
        uw = pd.Series([0.12, 0.04])
        adj = DynamicEngine.damped_score(s, uw)
        # 0.08/1.04 ≈ 0.077 > 0.10/1.12 ≈ 0.089? 否 → 验证阻尼方向
        assert adj.iloc[0] < s.iloc[0]
        assert adj.iloc[1] == pytest.approx(0.08 / 1.04)

    def test_bucket_ic(self):
        rng = np.random.default_rng(1)
        n = 500
        atr = rng.uniform(0.01, 0.10, n)
        # score 与 label 在高 ATR 区间相关, 低 ATR 区间噪声
        score = np.where(atr > 0.08, atr + rng.normal(0, 0.01, n), rng.normal(0, 1, n))
        label = np.where(atr > 0.08, atr + rng.normal(0, 0.02, n), rng.normal(0, 1, n))
        df = pd.DataFrame({"ATR_pct": atr, "score": score, "label": label})
        r = DynamicEngine.bucket_ic(df, "score", "label")
        assert set(r["buckets"]) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
        assert r["high_vol_ic"] > 0.02 and r["action"] == "ok"

    def test_bucket_ic_low_detects_dampen(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame(
            {
                "ATR_pct": rng.uniform(0.01, 0.10, 500),
                "score": rng.normal(0, 1, 500),
                "label": rng.normal(0, 1, 500),  # 纯噪声 → 各桶 IC≈0
            }
        )
        r = DynamicEngine.bucket_ic(df, "score", "label")
        assert not r["high_vol_ok"] and r["action"] == "dampen"

    def test_knobs_frozen(self):
        with pytest.raises(Exception):
            DynamicKnobs().min_win_prob = 0.7  # frozen: 旋钮只能季度窗口改
