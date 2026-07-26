"""P21.3 实盘解锁双闸门: D.10 裁决选C 且 40笔双闸门达标 → 方可切C."""

from __future__ import annotations

from app.config.profiles import resolve_live_profile
from app.pipeline1.trade_discipline import TradeJournal, TradeRecord


def _journal(pnls: list[float]) -> TradeJournal:
    j = TradeJournal()
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
    return j


class TestResolveLiveProfile:
    def test_d10_not_approved_always_b(self):
        """闸门1: D.10 回测未裁决选C → 永远 B (即便样本达标)."""
        good = _journal([0.01] * 40)
        assert resolve_live_profile(good, d10_c_approved=False) == "aggressive_b"

    def test_insufficient_sample_stays_b(self):
        """闸门2: 裁决选C 但样本不足 40 笔 → 仍按 B 执行 (前40笔按B)."""
        j = _journal([0.02] * 39)
        assert resolve_live_profile(j, d10_c_approved=True) == "aggressive_b"

    def test_double_gate_pass_unlocks_c(self):
        """双闸门全过: 裁决选C + 40笔 期望>0.5% 且 连亏≤5 → 切 C."""
        good = _journal([0.01] * 40)  # 期望+1%, 无连亏
        assert resolve_live_profile(good, d10_c_approved=True) == "aggressive"

    def test_expectancy_shortfall_stays_b(self):
        """期望 ≤0.5%/笔 → 否决切C."""
        j = _journal([0.003] * 40)
        assert resolve_live_profile(j, d10_c_approved=True) == "aggressive_b"

    def test_consec_loss_streak_stays_b(self):
        """最大连亏 >5 → 否决切C."""
        j = _journal([-0.04] * 6 + [0.02] * 34)
        assert resolve_live_profile(j, d10_c_approved=True) == "aggressive_b"

    def test_no_journal_defaults_conservative(self):
        """无实盘日志 → 默认保守 B (失败要大声, 不默许升档)."""
        assert resolve_live_profile(None, d10_c_approved=True) == "aggressive_b"
