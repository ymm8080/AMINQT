"""P19.1 阶段二: 2倍滑点敏感性门禁 (净超额≥5%才过门禁)."""

from __future__ import annotations

import pytest

from app.pipeline1.sensitivity import MIN_EXCESS_2X, slippage_sensitivity_verdict


class TestSlippageSensitivityVerdict:
    def test_pass_when_2x_excess_above_5pct(self):
        r = slippage_sensitivity_verdict(excess_1x=0.12, excess_2x=0.07)
        assert r["pass"] and r["excess_2x"] == pytest.approx(0.07)
        assert r["erosion"] == pytest.approx(0.05)  # 成本侵蚀 5 个点

    def test_boundary_exactly_5pct_passes(self):
        assert slippage_sensitivity_verdict(0.10, MIN_EXCESS_2X)["pass"]

    def test_fail_when_2x_below_5pct(self):
        r = slippage_sensitivity_verdict(excess_1x=0.10, excess_2x=0.03)
        assert not r["pass"] and "一票否决" in r["reason"]

    def test_fail_when_2x_negative(self):
        """2倍滑点下超额转负 → 超额是成本模型幻觉, 否决."""
        r = slippage_sensitivity_verdict(excess_1x=0.04, excess_2x=-0.01)
        assert not r["pass"]
        assert r["erosion"] == pytest.approx(0.05)
