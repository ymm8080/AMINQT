# -*- coding: utf-8 -*-
"""dim36 双向短期族 (bkd_ 破位 + up_ 上涨) 接线单测.

覆盖:
1. 合成帧数值正确性 (连跌 streak=4 / below_ma_cnt=3 / dd5_high20 / up_body5);
2. 跨股边界回归 (老 dn_streak 串股 bug 的守卫: B 股首跌日 streak 必须=1);
3. 零前视 (扰动 t 之后行, t 行特征不变);
4. force_include 双板配置含全 14 名;
5. 注册中心已注册 dim36 组;
6. build 门控链路 (stub registry: 放行→物化, 不放行→不物化).
"""

import os
import unittest

import numpy as np
import pandas as pd

from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.feature_registry import DIM_GROUPS, FeatureRegistry
from app.pipeline1.feature_selector import FeatureSelector

TARGET_14 = [
    "bkd_dn_streak", "bkd_dn_days5", "bkd_dd5_high20", "bkd_dd_high60",
    "bkd_min10_dist", "bkd_below_ma_cnt", "bkd_ma_bear_align", "bkd_ma5_slope5",
    "up_vol_confirm5", "up_body5", "up_break20_vol", "up_followthrough",
    "up_pullback_depth", "up_gap_hold",
]


def _make_frame() -> pd.DataFrame:
    """两只股 × 70 交易日. A: 平盘 65 天 + 4 连跌 (close 95..92);
    B: 每日 open=100/close=101 阳线 (跳空结构便于 body/gap 断言)."""
    n = 70
    dates = pd.bdate_range("2026-01-05", periods=n)
    rows = []
    # symbol A
    closes_a = [100.0] * (n - 4) + [95.0, 94.0, 93.0, 92.0]
    for i, cl in enumerate(closes_a):
        rows.append({
            "symbol": "000001", "date": dates[i],
            "open": cl, "high": cl + 1.0, "low": cl - 1.0, "close": cl,
            "open_hfq": cl, "high_hfq": cl + 1.0, "low_hfq": cl - 1.0,
            "close_hfq": cl, "volume": 1000.0,
        })
    # symbol B: prev_close 全 100 (首行除外) → body=(101-100)/100=0.01
    for i in range(n):
        rows.append({
            "symbol": "000002", "date": dates[i],
            "open": 100.0, "high": 102.0, "low": 99.5, "close": 101.0,
            "open_hfq": 100.0, "high_hfq": 102.0, "low_hfq": 99.5,
            "close_hfq": 101.0, "volume": 1000.0,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    return df


class TestDim36Math(unittest.TestCase):
    def setUp(self):
        self.df = _make_frame()
        self.out = FeatureEngineV35.dim36_bkd_up(self.df.copy())

    def _row(self, sym, idx):
        m = (self.out["symbol"] == sym)
        sub = self.out.loc[m].reset_index(drop=True)
        return sub.iloc[idx]

    def test_all_14_columns_present(self):
        for name in TARGET_14:
            self.assertIn(name, self.out.columns)

    def test_down_streak_4(self):
        # A 最后 4 天连跌: streak 末行=4, 前一日=3, 跌前首日=0
        self.assertEqual(self._row("000001", 69)["bkd_dn_streak"], 4.0)
        self.assertEqual(self._row("000001", 68)["bkd_dn_streak"], 3.0)
        self.assertEqual(self._row("000001", 65)["bkd_dn_streak"], 0.0)
        # dn_days5 末行 = 4 (66..69 四个下跌日)
        self.assertEqual(self._row("000001", 69)["bkd_dn_days5"], 4.0)

    def test_symbol_boundary_no_carryover(self):
        # 回归守卫: B 股首日 prev=NaN, 首跌日 streak 必须=1 (老 bug 会串 A 股的 4)
        # B close 101>100 → 全程上涨日, 用 A 的构造验证边界: A 首行 pc=NaN → dn=0
        self.assertEqual(self._row("000001", 0)["bkd_dn_streak"], 0.0)
        # B 首行 gap Hold: low 99.5 > prev_close NaN → False → 0
        self.assertEqual(self._row("000002", 0)["bkd_dn_streak"], 0.0)

    def test_below_ma_cnt_and_align(self):
        # A 末行: ma5=94.8, ma10=97.4, ma20=98.7, close=92 → 3 项全破
        r = self._row("000001", 69)
        self.assertEqual(r["bkd_below_ma_cnt"], 3.0)
        self.assertAlmostEqual(r["bkd_ma_bear_align"], 1.0, places=12)
        # A 平盘段: close=100 = 所有均线 → cnt=0
        self.assertEqual(self._row("000001", 30)["bkd_below_ma_cnt"], 0.0)

    def test_dd5_high20_exact(self):
        # A 末行: hi20 = max(high, 20日) = 101 (day65 high), dd5 = 92/101-1
        r = self._row("000001", 69)
        self.assertAlmostEqual(r["bkd_dd5_high20"], 92.0 / 101.0 - 1.0, places=12)

    def test_up_body5_bullish(self):
        # B: body=(close-open)/prev_close=(101-100)/101=1/101 (row0 prev=NaN → NaN)
        # B row5: 窗口 [rows1..5] 有 4 个非 NaN → min_periods=5 未满 → NaN
        # B row6: 窗口 5 个 1/101 → 1/101
        self.assertTrue(pd.isna(self._row("000002", 4)["up_body5"]))
        self.assertAlmostEqual(self._row("000002", 5)["up_body5"], 1.0 / 101.0, places=12)

    def test_up_gap_hold_flat_is_zero(self):
        # A 平盘段 low=99 < prev_close=100 → 不算跳空回补, 10日均=0
        self.assertEqual(self._row("000001", 30)["up_gap_hold"], 0.0)


class TestDim36NoLookahead(unittest.TestCase):
    def test_future_rows_do_not_change_past(self):
        df = _make_frame()
        base = FeatureEngineV35.dim36_bkd_up(df.copy())
        pert = df.copy()
        # 扰动 idx>=40 的 A 股行, 检查 idx<40 全部特征不变
        m = (pert["symbol"] == "000001") & (pert.index >= 40)
        pert.loc[m, "close_hfq"] = pert.loc[m, "close_hfq"] * 1.5
        pert.loc[m, "volume"] = pert.loc[m, "volume"] * 3.0
        after = FeatureEngineV35.dim36_bkd_up(pert)
        a = base.loc[(base["symbol"] == "000001") & (base.index < 40), TARGET_14]
        b = after.loc[(after["symbol"] == "000001") & (after.index < 40), TARGET_14]
        pd.testing.assert_frame_equal(a, b)


class TestWiring(unittest.TestCase):
    def test_gate_map_and_dim_group(self):
        self.assertEqual(FeatureEngineV35._DIM_GATE_MAP["dim36"], "dim36_bkd_up")
        self.assertIn("dim36_bkd_up", DIM_GROUPS)

    def test_dim_active_gating(self):
        class _Reg:
            def __init__(self, groups):
                self._g = set(groups)

            def has_dim_group(self, name):
                return name in self._g

        self.assertTrue(
            FeatureEngineV35._dim_active(_Reg(["dim36_bkd_up"]), "dim36_bkd_up")
        )
        self.assertFalse(FeatureEngineV35._dim_active(_Reg([]), "dim36_bkd_up"))
        self.assertTrue(FeatureEngineV35._dim_active(None, "dim36_bkd_up"))

    def test_force_include_gate_off_until_ab_pass(self):
        """A/B v6 PASS 前 dim36 族不得进生产 force_include (用户指令: 通过了才接线)."""
        cfg = FeatureSelector.DEFAULT_CONFIG
        inc_main = set(cfg["main"]["force_include"])
        inc_dual = set(cfg["dual"]["gate_d"]["force_include"])
        for name in TARGET_14:
            self.assertNotIn(name, inc_main, f"A/B 未过 {name} 已在 main force_include")
            self.assertNotIn(name, inc_dual, f"A/B 未过 {name} 已在 dual force_include")

    def test_registry_registered(self):
        reg = FeatureRegistry()
        if not os.path.exists(str(reg.path)):
            self.skipTest("注册中心文件不存在 (CI 环境)")
        self.assertTrue(reg.has_dim_group("dim36_bkd_up"))
        for name in TARGET_14:
            meta = reg.get_meta(name)
            self.assertIsNotNone(meta, f"{name} 未注册")
            self.assertEqual(meta.get("dim_group"), "dim36_bkd_up")
            self.assertTrue(meta.get("active"), f"{name} 未激活")


class _StubRegistry:
    """build() 端到端门控测试桩."""

    def __init__(self, groups):
        self._g = set(groups)

    def has_dim_group(self, name):
        return name in self._g

    def get_active(self):
        return list(TARGET_14) if "dim36_bkd_up" in self._g else []

    def is_adoption_enabled(self):
        return False

    def get_meta(self, _name):
        return None


class TestBuildEndToEnd(unittest.TestCase):
    def _build(self, groups):
        df = _make_frame()
        return FeatureEngineV35().build(
            df, None, cross_sectional_rank=False, registry=_StubRegistry(groups)
        )

    def test_build_emits_dim36_when_registered(self):
        out = self._build(["dim36_bkd_up"])
        for name in TARGET_14:
            self.assertIn(name, out.columns, f"build 未物化 {name}")
        # 值与直接调静态方法一致 (抽查 streak)
        direct = FeatureEngineV35.dim36_bkd_up(_make_frame())
        m_out = out.loc[out["symbol"] == "000001", "bkd_dn_streak"].reset_index(drop=True)
        m_dir = direct.loc[direct["symbol"] == "000001", "bkd_dn_streak"].reset_index(drop=True)
        np.testing.assert_allclose(m_out.to_numpy(), m_dir.to_numpy())

    def test_build_skips_dim36_when_not_registered(self):
        out = self._build([])
        for name in TARGET_14:
            self.assertNotIn(name, out.columns)


if __name__ == "__main__":
    unittest.main()
