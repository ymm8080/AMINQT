"""V3.8 第七批: D.10 B/C 裁决协议."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.pipeline1.backtest_adjudication import (
    CRISIS_SEGMENTS,
    BacktestJournal,
    adjudicate_b_vs_c,
    inject_tail_shocks,
    resolve_range,
    segment_robustness,
    worst_trades_audit,
)


class TestResolveRange:
    def test_preset_not_exploratory(self):
        r = resolve_range(preset="bear_2018")
        assert r["start"] == "2018-01-01" and not r["exploratory"]

    def test_custom_is_exploratory(self):
        r = resolve_range(custom=("2023-05-01", "2023-08-01"))
        assert r["exploratory"] and r["preset"] is None  # 不得作裁决依据

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError):
            resolve_range(preset="golden_2019")


class TestTailShockInjection:
    def test_c_loses_more_than_b(self):
        nav = pd.Series(np.cumprod(np.full(300, 1.001)))
        c = inject_tail_shocks(nav, position_cap=1.00, seed=1)
        b = inject_tail_shocks(nav, position_cap=0.75, seed=1)
        # C 满仓吃满跳空 → 注入后净值显著低于 B
        assert c.iloc[-1] < b.iloc[-1]
        assert c.iloc[-1] < nav.iloc[-1]  # 注入有效

    def test_reproducible_seed(self):
        nav = pd.Series(np.cumprod(np.full(200, 1.001)))
        a = inject_tail_shocks(nav, 1.0, seed=7)
        b = inject_tail_shocks(nav, 1.0, seed=7)
        assert a.equals(b)  # 随机种子固定 (量化铁律)


class TestWorstTradesAudit:
    def test_flags_unexecutable_stops(self):
        trades = pd.DataFrame({
            "pnl": [-0.045, -0.08, -0.02, 0.03, -0.041],
        })
        r = worst_trades_audit(trades, n=20)
        assert r["n_unexecutable"] == 1  # -0.08 超出 -4.5% 可执行范围
        assert r["executable_distortion"]
        assert r["worst"].iloc[0]["pnl"] == -0.08

    def test_clean_trades_no_distortion(self):
        trades = pd.DataFrame({"pnl": [-0.04, -0.038, 0.02]})
        assert not worst_trades_audit(trades)["executable_distortion"]


class TestSegmentRobustness:
    def test_any_segment_breach_rejects(self):
        dd = {"bear_2018": -0.10, "covid_2020": -0.16,
              "bear_2022": -0.08, "liquidity_2024": -0.05}
        r = segment_robustness(dd)
        assert not r["pass"] and "covid_2020" in r["failed_segments"]

    def test_all_pass(self):
        dd = {s: -0.10 for s in CRISIS_SEGMENTS}
        assert segment_robustness(dd)["pass"]

    def test_missing_segment_rejects(self):
        assert not segment_robustness({"bear_2018": -0.05})["pass"]


class TestAdjudication:
    def _audit(self, distortion=False):
        return {"executable_distortion": distortion}

    def test_choose_c_only_when_all_pass(self):
        dd = {s: -0.08 for s in CRISIS_SEGMENTS}
        r = adjudicate_b_vs_c(0.03, 0.04, -0.10, dd, self._audit())
        assert r["choose"] == "C"

    def test_any_failure_chooses_b(self):
        dd = {s: -0.08 for s in CRISIS_SEGMENTS}
        # GT 落后
        assert adjudicate_b_vs_c(0.05, 0.04, -0.10, dd, self._audit())["choose"] == "B"
        # 注入回撤破线
        assert adjudicate_b_vs_c(0.03, 0.04, -0.16, dd, self._audit())["choose"] == "B"
        # 可执行性失真
        assert adjudicate_b_vs_c(0.03, 0.04, -0.10, dd,
                                 self._audit(distortion=True))["choose"] == "B"


class TestBacktestJournal:
    def test_worm_append(self, tmp_path):
        j = BacktestJournal(str(tmp_path))
        j.log("run1", {"range": "bear_2018"}, {"gt": 0.03})
        j.log("run2", {"range": "covid_2020"}, {"gt": -0.01})  # 失败区间一并保留
        with open(tmp_path / "backtest_runs.jsonl", encoding="utf-8") as fh:
            lines = fh.readlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["metrics"]["gt"] == -0.01
