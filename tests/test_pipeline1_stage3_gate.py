"""P19.2 阶段三: 20笔样本裁决门禁 (期望>+0.3%/笔 且 最大连亏≤4)."""

from __future__ import annotations

from app.pipeline1.trade_discipline import (
    STAGE3_MAX_CONSEC_LOSS,
    STAGE3_MIN_EXPECTANCY,
    STAGE3_MIN_TRADES,
    TradeJournal,
    TradeRecord,
)


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


class TestStage3Gate:
    def test_thresholds_distinct_from_d10(self):
        """阶段三口径独立于 D.10 (20/0.3%/≤4 vs 40/0.5%/≤5), 不得混用."""
        assert STAGE3_MIN_TRADES == 20
        assert STAGE3_MIN_EXPECTANCY == 0.003
        assert STAGE3_MAX_CONSEC_LOSS == 4

    def test_insufficient_sample(self):
        j = _journal([0.01] * 19)
        r = j.stage3_gate()
        assert not r["pass"] and "19/20" in r["reason"]

    def test_pass_when_both_gates_met(self):
        j = _journal([0.008] * 20)  # 期望+0.8%, 无连亏
        assert j.stage3_gate()["pass"]

    def test_fail_on_low_expectancy(self):
        j = _journal([0.002] * 20)  # 期望+0.2% < 0.3%
        assert not j.stage3_gate()["pass"]

    def test_fail_on_consec_loss_over_4(self):
        j = _journal([-0.04] * 5 + [0.02] * 15)  # 连亏5 > 4
        assert not j.stage3_gate()["pass"]

    def test_boundary_consec_loss_4_passes(self):
        j = _journal([-0.04] * 4 + [0.02] * 16)  # 连亏恰4 → 过
        assert j.stage3_gate()["pass"]
