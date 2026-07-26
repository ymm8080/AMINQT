"""P21.2 E-5 月度旋钮体检: 空仓比例/实际胜率/实际盈亏比 (只告警不调参)."""

from __future__ import annotations

from app.pipeline1.dynamic_engine import knob_health_check
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


class TestKnobHealthCheck:
    def test_healthy_no_flags(self):
        """胜率/盈亏比达标 + 空仓比例正常 → 无告警."""
        j = _journal([0.08] * 7 + [-0.04] * 3)  # 胜率70%≥65%, 盈亏比2.0≥1.8
        r = knob_health_check(j, idle_ratio=0.75)
        assert r["flags"] == []
        assert r["n_trades"] == 10

    def test_low_win_rate_flagged(self):
        """实际胜率 < 旋钮 65% → 告警 (校准/闸门季度复检)."""
        j = _journal([0.08] * 5 + [-0.04] * 5)  # 胜率 50%
        r = knob_health_check(j, idle_ratio=0.75)
        assert any("胜率" in f for f in r["flags"])

    def test_low_pl_ratio_flagged(self):
        """实际盈亏比 < 旋钮 1.8 → 告警 (RR 闸门/移动止盈复检)."""
        j = _journal([0.03] * 8 + [-0.04] * 2)  # 胜率80%但盈亏比 0.75
        r = knob_health_check(j, idle_ratio=0.75)
        assert any("盈亏比" in f for f in r["flags"])

    def test_abnormal_idle_ratio_flagged(self):
        """空仓比例远低于预期 75% → 门槛被击穿告警."""
        j = _journal([0.08] * 7 + [-0.04] * 3)
        r = knob_health_check(j, idle_ratio=0.40)
        assert any("空仓比例" in f for f in r["flags"])

    def test_empty_journal_flagged_not_crash(self):
        """无成交样本 → 提示人工确认, 不抛异常."""
        r = knob_health_check(TradeJournal(), idle_ratio=1.0)
        assert r["n_trades"] == 0 and any("无成交样本" in f for f in r["flags"])

    def test_no_param_suggestion_in_output(self):
        """安全网#15: 体检表永不输出建议调参值 (只告警, 不自动调参)."""
        j = _journal([0.01] * 5 + [-0.04] * 5)
        r = knob_health_check(j, idle_ratio=0.40)
        assert not any(k.startswith("suggest") or k.startswith("new_") for k in r)
