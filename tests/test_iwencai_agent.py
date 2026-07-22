# -*- coding: utf-8 -*-
"""Tests for app/models/iwencai_agent.py — 全部 mock, 不触网."""

import numpy as np
import pandas as pd
import pytest

from app.models.iwencai_agent import IwencaiAgent, RANK_CONDITIONS


def make_df(closes, vols, ctrl=None):
    """构造日线 DataFrame (open/high/low 取 close 近似)."""
    df = pd.DataFrame(
        {
            "open": list(closes),
            "high": list(closes),
            "low": list(closes),
            "close": list(closes),
            "volume": list(vols),
        }
    )
    if ctrl is not None:
        df["tech_ths_ctrl_ratio"] = list(ctrl)
    return df


def flat_df(n=60, close=10.0, vol=1000.0, **kw):
    return make_df([close] * n, [vol] * n, **kw)


@pytest.fixture
def agent():
    return IwencaiAgent()


# ── ① 排名交集 ────────────────────────────────────────────────────


class TestRankIntersection:
    def test_intersection_across_five_conditions(self, agent, monkeypatch):
        results = {
            RANK_CONDITIONS[0]: ["A", "B", "C", "D"],
            RANK_CONDITIONS[1]: ["B", "C", "D"],
            RANK_CONDITIONS[2]: ["C", "D"],
            RANK_CONDITIONS[3]: ["C", "D", "E"],
            RANK_CONDITIONS[4]: ["C", "E"],
        }
        monkeypatch.setattr(
            agent,
            "query",
            lambda cond, top_n=50: [
                {"symbol": s, "name": s, "iwencai_score": None, "match_reasons": [cond]}
                for s in results[cond]
            ],
        )
        assert agent.query_rank_intersection() == ["C"]

    def test_empty_intersection(self, agent, monkeypatch):
        monkeypatch.setattr(
            agent,
            "query",
            lambda cond, top_n=50: [
                {
                    "symbol": cond[:1],
                    "name": "",
                    "iwencai_score": None,
                    "match_reasons": [],
                }
            ],
        )
        assert agent.query_rank_intersection() == []


class TestTemplates:
    def test_unknown_template_raises(self, agent):
        with pytest.raises(ValueError, match="未知问财模板"):
            agent.query_by_template("no_such_template")

    def test_template_renders_condition(self, agent, monkeypatch):
        seen = []
        monkeypatch.setattr(
            agent,
            "query",
            lambda cond, top_n=50: seen.append((cond, top_n)) or [],
        )
        agent.query_by_template("super_strong", 板块="半导体")
        assert seen[0][0].startswith("超强主力")
        assert "板块半导体" in seen[0][0]


# ── ② 形态 (正/负) ───────────────────────────────────────────────


class TestPatterns:
    def _prep(self, df):
        out = IwencaiAgent._prep(df)
        assert out is not None
        return out

    def test_volume_surge_pullback_positive(self):
        closes = [10.0] * 55 + [10.5, 10.4, 10.35, 10.3, 10.3]
        vols = [1000.0] * 55 + [3000.0, 800.0, 700.0, 600.0, 500.0]
        df = self._prep(make_df(closes, vols))
        assert IwencaiAgent._pat_volume_surge_pullback(df) is True

    def test_volume_surge_pullback_negative_no_surge(self):
        df = self._prep(flat_df())
        assert IwencaiAgent._pat_volume_surge_pullback(df) is False

    def test_volume_surge_pullback_negative_no_pullback(self):
        closes = [10.0] * 58 + [10.5, 11.0]
        vols = [1000.0] * 58 + [3000.0, 900.0]
        df = self._prep(make_df(closes, vols))
        assert IwencaiAgent._pat_volume_surge_pullback(df) is False

    def test_low_volume_rise_positive(self):
        closes = [10.0] * 56 + [10.0, 10.1, 10.2, 10.3]
        vols = [1000.0] * 56 + [1000.0, 1000.0, 1000.0, 500.0]
        df = self._prep(make_df(closes, vols))
        assert IwencaiAgent._pat_low_volume_rise(df) is True

    def test_low_volume_rise_negative_high_vol(self):
        closes = [10.0] * 56 + [10.0, 10.1, 10.2, 10.3]
        vols = [1000.0] * 56 + [1000.0, 1000.0, 1000.0, 2000.0]
        df = self._prep(make_df(closes, vols))
        assert IwencaiAgent._pat_low_volume_rise(df) is False

    def test_ctrl_rising_positive(self):
        ctrl = np.linspace(0.30, 0.50, 60)
        df = self._prep(flat_df(ctrl=ctrl))
        assert IwencaiAgent._pat_ctrl_rising(df) is True

    def test_ctrl_rising_negative_flat(self):
        df = self._prep(flat_df(ctrl=[0.4] * 60))
        assert IwencaiAgent._pat_ctrl_rising(df) is False

    def test_ctrl_rising_missing_column(self):
        df = self._prep(flat_df())
        assert IwencaiAgent._pat_ctrl_rising(df) is False

    def test_main_force_snatch_positive(self):
        closes = [10.0] * 55 + [10.0, 10.0, 10.0, 10.5, 10.5]
        vols = [1000.0] * 55 + [1000.0, 1000.0, 1000.0, 4000.0, 900.0]
        df = self._prep(make_df(closes, vols))
        assert IwencaiAgent._pat_main_force_snatch(df) is True

    def test_main_force_snatch_negative(self):
        df = self._prep(flat_df())
        assert IwencaiAgent._pat_main_force_snatch(df) is False


# ── ③ 剔除 (正/负) ───────────────────────────────────────────────


class TestExclusions:
    def _prep(self, df):
        return IwencaiAgent._prep(df)

    def test_volume_drop_positive(self):
        closes = [10.0] * 57 + [10.0, 9.5, 10.0]
        vols = [1000.0] * 57 + [1000.0, 2500.0, 900.0]
        df = self._prep(make_df(closes, vols))
        assert IwencaiAgent._exc_volume_drop(df) is True

    def test_volume_drop_negative(self):
        df = self._prep(flat_df())
        assert IwencaiAgent._exc_volume_drop(df) is False

    def test_high_volume_top_positive(self):
        closes = [10.0] * 59 + [12.0]
        vols = [400.0] * 59 + [3000.0]
        df = self._prep(make_df(closes, vols))
        assert IwencaiAgent._exc_high_volume_top(df) is True

    def test_high_volume_top_negative_no_huge_vol(self):
        closes = [10.0] * 59 + [12.0]
        vols = [1000.0] * 60
        df = self._prep(make_df(closes, vols))
        assert IwencaiAgent._exc_high_volume_top(df) is False

    def test_off_ma20_positive(self):
        closes = [10.0] * 59 + [12.0]
        df = self._prep(make_df(closes, [1000.0] * 60))
        assert IwencaiAgent._exc_off_ma20(df) is True

    def test_off_ma20_negative(self):
        closes = [10.0] * 59 + [10.5]
        df = self._prep(make_df(closes, [1000.0] * 60))
        assert IwencaiAgent._exc_off_ma20(df) is False


# ── build_candidate_pool 集成 ────────────────────────────────────


class TestBuildCandidatePool:
    def test_full_flow(self, agent, monkeypatch):
        # A: 缩量上涨, 无剔除 → 入池
        df_a = make_df(
            [10.0] * 56 + [10.0, 10.1, 10.2, 10.3],
            [1000.0] * 56 + [1000.0, 1000.0, 1000.0, 500.0],
        )
        # B: 缩量上涨 但近 5 日有放量下跌 → 剔除
        df_b = make_df(
            [10.0] * 55 + [10.0, 9.5, 10.1, 10.2, 10.3],
            [1000.0] * 55 + [1000.0, 2500.0, 1000.0, 1000.0, 500.0],
        )
        # C: 无任何形态 → 过滤
        df_c = flat_df()
        # E: 数据不足 → 跳过
        df_e = flat_df(n=10)

        monkeypatch.setattr(
            agent, "query_rank_intersection", lambda: ["A", "B", "C", "E"]
        )
        base_pool = ["A", "B", "C", "D", "E"]  # D 不在排名交集
        daily = {"A": df_a, "B": df_b, "C": df_c, "D": df_a, "E": df_e}
        assert agent.build_candidate_pool(base_pool, daily) == ["A"]

    def test_empty_base_pool(self, agent, monkeypatch):
        monkeypatch.setattr(agent, "query_rank_intersection", lambda: ["A"])
        assert agent.build_candidate_pool([], {"A": flat_df()}) == []
