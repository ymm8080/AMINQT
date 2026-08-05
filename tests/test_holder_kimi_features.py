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
        for i, (got, exp) in enumerate(zip(r, expected)):
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
