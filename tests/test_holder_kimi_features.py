"""KIMI 股东增减持比例特征 (dim29) 单元测试.

覆盖:
  - sh_ratio_30d: 30 交易日净增减持比例滚动累计 (PIT shift(1) 排除当日)
  - 逐股隔离 (_apply_per_stock 组内独立)
  - 缺失 sh_net_ratio 列 → NaN 回退
  - 既有 GLM 输出 (sh_ann_decay 等) 无回归
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.pipeline1.feature_engine_v35 import FeatureEngineV35


def _base_panel() -> pd.DataFrame:
    """单股 40 交易日合成面板, 事件日 idx 5/15/25 有净增减持比例."""
    dates = pd.bdate_range("2025-01-01", periods=40)
    n = len(dates)
    evt = {5: 0.02, 15: -0.03, 25: 0.04}  # idx -> sh_net_ratio
    net = np.full(n, np.nan)
    g = np.full(n, np.nan)
    c = np.full(n, np.nan)
    for idx, v in evt.items():
        net[idx] = v
        g[idx] = v / 2
        c[idx] = v / 4
    sign = [np.nan if np.isnan(x) else np.sign(x) for x in net]
    amt = [np.nan if np.isnan(x) else abs(x) * 1e7 for x in net]
    df = pd.DataFrame(
        {
            "symbol": ["600519"] * n,
            "date": dates,
            "sh_net_ratio": net,
            "sh_g_ratio": g,
            "sh_c_ratio": c,
            "sh_net_change_sign": sign,
            "sh_change_amt_total": amt,
            "sh_evt_start_date": pd.to_datetime([None] * n),
            "sh_evt_end_date": pd.to_datetime([None] * n),
        }
    )
    # 事件窗口: 公告 A=idx5, 变动窗口 S=idx5 ~ E=idx20
    df.loc[5, "sh_evt_start_date"] = dates[5]
    df.loc[5, "sh_evt_end_date"] = dates[20]
    return df


class TestHolderKimiFeatures:
    def test_sh_ratio_30d_pit_and_decay(self):
        """事件日 T 的累计不含当日公告 (PIT); 非事件日正确衰减."""
        out = FeatureEngineV35.dim29_holdertrade(_base_panel())
        sub = out[out["symbol"] == "600519"].sort_values("date").reset_index(drop=True)
        r = sub["sh_ratio_30d"]

        # PIT: 事件日 T 的累计 = 前 30 交易日的和, 不含 T 当日净增减持
        assert r[5] == pytest.approx(0.0)  # 事件日 idx5, 不含 +0.02
        assert r[15] == pytest.approx(0.02)  # 事件日 idx15, 不含 -0.03
        assert r[25] == pytest.approx(-0.01)  # 事件日 idx25, 不含 +0.04

        # 逐日手算期望: idx5=0.02, idx15=-0.03, idx25=0.04, fillna(0)+rolling(30)+shift(1)
        expected = (
            [np.nan]
            + [0.0] * 5  # idx1-5  事件前无累计
            + [0.02] * 10  # idx6-15 计入 idx5
            + [-0.01] * 10  # idx16-25 计入 idx5+idx15
            + [0.03] * 10  # idx26-35 计入 idx5+idx15+idx25
            + [0.01] * 4  # idx36-39 idx5 滑出 30 日窗 (只余 idx15+idx25)
        )
        assert len(r) == 40 and len(expected) == 40
        for i, (got, exp) in enumerate(zip(r, expected, strict=False)):
            if np.isnan(exp):
                assert np.isnan(got), f"idx {i}: expect NaN, got {got}"
            else:
                assert got == pytest.approx(exp), f"idx {i}: expect {exp}, got {got}"

    def test_sh_ratio_30d_per_symbol_isolation(self):
        """逐股计算, 一股的事件不影响另一股."""
        dates = pd.bdate_range("2025-01-01", periods=40)
        n = len(dates)

        def build(sym: str, evt_idx: int, val: float) -> pd.DataFrame:
            net = np.full(n, np.nan)
            net[evt_idx] = val
            return pd.DataFrame(
                {
                    "symbol": [sym] * n,
                    "date": dates,
                    "sh_net_ratio": net,
                    "sh_net_change_sign": [
                        np.nan if np.isnan(x) else np.sign(x) for x in net
                    ],
                    "sh_change_amt_total": [
                        np.nan if np.isnan(x) else abs(x) * 1e7 for x in net
                    ],
                }
            )

        df = pd.concat(
            [build("600519", 5, 0.02), build("000001", 30, 0.10)], ignore_index=True
        )
        out = FeatureEngineV35.dim29_holdertrade(df)
        a = (
            out[out["symbol"] == "600519"]
            .sort_values("date")
            .reset_index(drop=True)["sh_ratio_30d"]
        )
        b = (
            out[out["symbol"] == "000001"]
            .sort_values("date")
            .reset_index(drop=True)["sh_ratio_30d"]
        )
        assert a[6] == pytest.approx(0.02)  # A 的 idx5 事件次日计入
        assert a[30] == pytest.approx(0.02)  # B 的 idx30 事件不影响 A
        assert b[30] == pytest.approx(0.0)  # B 事件日 T 不含当日 +0.10
        assert b[31] == pytest.approx(0.10)  # B 次日才计入

    def test_sh_ratio_30d_missing_net_ratio_nan(self):
        """有 GLM 列但无 sh_net_ratio 列: sh_ratio_30d 全 NaN (per_stock 回退)."""
        dates = pd.bdate_range("2025-01-06", periods=4)
        df = pd.DataFrame(
            {
                "symbol": ["600519"] * 4,
                "date": dates,
                "sh_net_change_sign": [np.nan, 1.0, np.nan, np.nan],
                "sh_change_amt_total": [np.nan, 1000.0, np.nan, np.nan],
            }
        )
        out = FeatureEngineV35.dim29_holdertrade(df)
        assert "sh_ratio_30d" in out.columns
        assert out["sh_ratio_30d"].isna().all()

    def test_sh_ratio_30d_no_holdertrade_cols_nan(self):
        """无任何增减持列 (has_ht=False): 全部特征 (含 sh_ratio_30d) 全 NaN, 不崩."""
        dates = pd.bdate_range("2025-01-06", periods=3)
        df = pd.DataFrame({"symbol": ["600519"] * 3, "date": dates})
        out = FeatureEngineV35.dim29_holdertrade(df)
        assert out["sh_ratio_30d"].isna().all()

    def test_glm_event_features_still_produced(self):
        """既有 GLM 输出无回归: sh_ann_decay/sh_end_decay/sh_is_executing 等仍产出."""
        out = FeatureEngineV35.dim29_holdertrade(_base_panel())
        sub = out[out["symbol"] == "600519"].sort_values("date").reset_index(drop=True)
        for c in (
            "sh_ann_decay",
            "sh_end_decay",
            "sh_is_executing",
            "sh_net_sign_20d",
            "sh_net_sign_60d",
            "sh_change_amt_20d",
            "sh_change_amt_60d",
            "sh_insider_signal",
            "sh_change_frequency",
            "sh_amt_vs_amount",
        ):
            assert c in out.columns
        # 公告日 idx5: 恐慌峰值 1.0, 进入执行期 → exec=1
        assert sub["sh_ann_decay"][5] == pytest.approx(1.0)
        assert sub["sh_is_executing"][5] == pytest.approx(1.0)
        # 结束日 idx20 (排他): 反弹峰值 1.0, 视为已结束 → exec=0
        assert sub["sh_end_decay"][20] == pytest.approx(1.0)
        assert sub["sh_is_executing"][20] == pytest.approx(0.0)


class TestHoldertradeStalenessFix:
    """[2026-08-31 冻结修复] ffill 续写值不参与特征, 事件出窗即归零.

    面板 GLM 4 列为 _daily_fetch ffill 续写值 (非公告日逐日复制最近公告行).
    旧实现把 ffill 常数 rolling sum → 最后一次事件永不衰减 (002881 案例:
    2023-08 减持 3910万 冻结成 7.8亿/20d 看空信号 2 年). 新实现以台阶检测
    重建事件日, 事件日之外一律 0.
    """

    @staticmethod
    def _panel(n: int, sign: list[float], amt: list[float]) -> pd.DataFrame:
        dates = pd.bdate_range("2025-01-01", periods=n)
        return pd.DataFrame(
            {
                "symbol": ["002881"] * n,
                "date": dates,
                "sh_net_change_sign": sign,
                "sh_change_amt_total": amt,
                "amount": [1e8] * n,
            }
        )

    def test_event_true_window_sum_not_inflated(self):
        """单次事件 100 万: amt_20d = 100万 (事件日一次), 非 ffill×20 = 2000万."""
        n = 60
        sign = [np.nan] * n
        amt = [np.nan] * n
        sign[10] = -1.0
        amt[10] = 1_000_000.0
        out = FeatureEngineV35.dim29_holdertrade(self._panel(n, sign, amt))
        r = out.sort_values("date").reset_index(drop=True)
        # 事件日 (idx10) 与窗口内: 合计 = 一次事件额
        assert r["sh_change_amt_20d"][10] == pytest.approx(1_000_000.0)
        assert r["sh_change_amt_20d"][29] == pytest.approx(1_000_000.0)
        # 出窗 (idx20 起滑出 20 日窗): 归零
        assert r["sh_change_amt_20d"][30] == pytest.approx(0.0)
        # sign 同口径: 20 日内 -1, 出窗后 insider_signal = 0
        assert r["sh_insider_signal"][19] == pytest.approx(-1.0)
        assert r["sh_insider_signal"][30] == pytest.approx(0.0)

    def test_frozen_ffill_values_do_not_participate(self):
        """ffill 冻结 (sign 恒 -1, amt 恒 0, 无台阶): 旧实现 insider_signal 永远 -1;
        新实现仅首行幽灵事件计一次, 20 日后归零."""
        n = 60
        sign = [-1.0] * n
        amt = [0.0] * n
        out = FeatureEngineV35.dim29_holdertrade(self._panel(n, sign, amt))
        r = out.sort_values("date").reset_index(drop=True)
        # 首行幽灵事件 → 前 20 日 -1, 之后 (事件出窗) 0 — 不再永续
        assert r["sh_insider_signal"][0] == pytest.approx(-1.0)
        assert r["sh_insider_signal"][20] == pytest.approx(0.0)
        assert (r["sh_insider_signal"][20:] == 0.0).all()
        # amt 恒 0: 20d 合计恒 0
        assert (r["sh_change_amt_20d"] == 0.0).all()

    def test_new_event_breaks_frozen_state(self):
        """冻结后被新公告行覆盖 (台阶): 新事件日正确计入."""
        n = 60
        # 0..29 冻结 -1/0 (旧事件残值), 30 起新事件 +1/500万
        sign = [-1.0] * 30 + [1.0] * 30
        amt = [0.0] * 30 + [5_000_000.0] * 30
        out = FeatureEngineV35.dim29_holdertrade(self._panel(n, sign, amt))
        r = out.sort_values("date").reset_index(drop=True)
        # idx30 = 新事件日: +1 方向, amt_20d = 500万 (一次)
        assert r["sh_insider_signal"][30] == pytest.approx(1.0)
        assert r["sh_change_amt_20d"][30] == pytest.approx(5_000_000.0)
        # idx49 仍在窗内; idx50 起新事件滑出 → 但首行幽灵事件也早已出窗 → 0
        assert r["sh_change_amt_20d"][49] == pytest.approx(5_000_000.0)
        assert r["sh_change_amt_20d"][50] == pytest.approx(0.0)

    def test_amt0_event_row_counts_in_frequency(self):
        """amt=0 的公告行 (台阶) 也计入 sh_change_frequency (旧实现按 amt!=0 漏计)."""
        n = 60
        sign = [-1.0] * 10 + [-1.0] * 50  # 需 amt/vol 台阶才可见
        amt = [0.0] * 10 + [0.0] * 50
        vol = [1_000.0] * 10 + [2_000.0] * 50  # idx10 vol 台阶 = 新公告行 (amt=0)
        dates = pd.bdate_range("2025-01-01", periods=n)
        df = pd.DataFrame(
            {
                "symbol": ["002881"] * n,
                "date": dates,
                "sh_net_change_sign": sign,
                "sh_change_amt_total": amt,
                "sh_change_vol": vol,
                "amount": [1e8] * n,
            }
        )
        out = FeatureEngineV35.dim29_holdertrade(df)
        r = out.sort_values("date").reset_index(drop=True)
        # idx0 幽灵 + idx10 新公告 = 窗内 2 次
        assert r["sh_change_frequency"][10] == pytest.approx(2.0)
        # idx50 (40 交易日后): idx0 出 60 窗前? 60 日窗 > 50 → 仍 2
        assert r["sh_change_frequency"][50] == pytest.approx(2.0)

    def test_amt_vs_amount_decays_with_window(self):
        """sh_amt_vs_amount 随 20 日窗衰减 (旧实现 ffilled 永续)."""
        n = 60
        sign = [np.nan] * n
        amt = [np.nan] * n
        sign[10] = 1.0
        amt[10] = 2_000_000.0
        out = FeatureEngineV35.dim29_holdertrade(self._panel(n, sign, amt))
        r = out.sort_values("date").reset_index(drop=True)
        assert r["sh_amt_vs_amount"][10] == pytest.approx(2_000_000.0 / 1e8)
        assert r["sh_amt_vs_amount"][30] == pytest.approx(0.0)

    def test_no_event_stock_stays_zero(self):
        """全 NaN 股票 (从未有公告): 特征全 0, 不再 NaN/幽灵."""
        n = 30
        sign = [np.nan] * n
        amt = [np.nan] * n
        out = FeatureEngineV35.dim29_holdertrade(self._panel(n, sign, amt))
        r = out.sort_values("date").reset_index(drop=True)
        assert (r["sh_insider_signal"] == 0.0).all()
        assert (r["sh_change_amt_20d"] == 0.0).all()
        assert (r["sh_change_frequency"] == 0.0).all()

