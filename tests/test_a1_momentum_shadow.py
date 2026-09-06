"""差值加速度影子单纯函数单测 (2026-09-04, scripts/_a1_momentum_shadow.py).

口径锁死 (250d 回放, 2026-09-04 用户拍板, 勿静默改):
  名单 = 全市场 (r5 - r5_prev) 差值 top10 纯排, 无累计动量臂、无三桶过滤。
  沿革: a1union 纯并集 → a1tri 三桶 → a1diff 差值 (本版)。
  严格比值 r5/r5p 判死 ≈ 基线; 三桶过滤杀差值臂 79% 赢家, 均勿再加。
"""

import numpy as np
import pandas as pd

from scripts._a1_momentum_shadow import R_WIN, diff_picks


def _close(n: int = 15) -> pd.DataFrame:
    """可人工判读收盘矩阵: A 后5日+20% / B 前5日+40%后平 / C 加速上行 / D 平后小跌."""
    idx = pd.date_range("2026-01-01", periods=n, freq="B")
    a = np.concatenate([np.full(n - 5, 10.0), np.full(5, 12.0)])          # diff=+0.20
    b = np.concatenate([np.full(n - 10, 10.0), np.full(10, 14.0)])[:n]    # diff=-0.40
    c = np.concatenate([np.full(n - 10, 10.0), np.full(5, 10.5),
                        np.full(5, 11.55)])[:n]                           # diff=+0.05
    d = np.concatenate([np.full(n - 5, 10.0), np.full(5, 9.5)])           # diff=-0.05
    return pd.DataFrame({"600001": a, "600002": b, "600003": c, "600004": d}, index=idx)


def test_diff_picks_orders_by_acceleration():
    out = diff_picks(_close(), pd.bdate_range("2026-01-01", periods=15)[-1])
    assert list(out["symbol"]) == ["600001", "600003", "600004", "600002"]
    assert out["diff"].is_monotonic_decreasing
    assert list(out["rank"]) == [1, 2, 3, 4]
    a = out.iloc[0]
    assert abs(a["r5"] - 0.2) < 1e-9 and abs(a["r5p"]) < 1e-9


def test_diff_picks_top_n_cap():
    close = pd.concat([_close()] * 3, axis=1)
    close.columns = [f"{600000 + i:06d}" for i in range(close.shape[1])]
    out = diff_picks(close, close.index[-1], arm_top_n=10)
    assert len(out) == 10  # 12 只候选 cap 到 10
    assert len(diff_picks(close, close.index[-1], arm_top_n=2)) == 2


def test_diff_picks_short_history_empty():
    out = diff_picks(_close(10), _close(10).index[-1])  # 需 2*5+1=11 交易日
    assert out.empty
    assert list(out.columns) == ["rank", "symbol", "diff", "r5", "r5p"]


def test_diff_picks_uses_only_history_up_to_day_ts():
    close = _close(15)
    out = diff_picks(close, close.index[9])  # 只剩 10 日历史 → 空
    assert out.empty


def test_diff_picks_excludes_insufficient_history_symbols():
    close = _close()
    close["600009"] = np.nan  # 全缺 → 剔除
    close.loc[close.index[:-8], "600008"] = np.nan  # 不足 11 日 → 剔除
    out = diff_picks(close, close.index[-1])
    assert "600009" not in list(out["symbol"])
    assert "600008" not in list(out["symbol"])


def test_diff_picks_excludes_bse():
    close = _close()
    close["920999.BJ"] = close["600001"] * 1.1  # 更高差值也会被剔
    out = diff_picks(close, close.index[-1])
    assert "920999.BJ" not in list(out["symbol"])
    assert len(out) == 4  # 剔后 top10 补满剩余 A 股


def test_diff_picks_deterministic():
    ts = _close().index[-1]
    pd.testing.assert_frame_equal(diff_picks(_close(), ts), diff_picks(_close(), ts))


def test_window_constant_is_five():
    """差值两腿各 5 交易日 — 回放口径, 勿静默改."""
    assert R_WIN == 5
