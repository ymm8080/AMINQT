"""隔板口袋影子单纯函数单测 (2026-09-04, scripts/_gap_pocket_shadow.py).

口径锁死 (250d 回放, 2026-09-04 拍板 + 09-05 dry 0.7→0.8 拍板, 勿静默改):
  池 = 首板后 d3~7 无再板 & 守板价(收>=首板收) & 缩量(近3日均量/首板量<0.8)
       & 安全闸(10日涨幅<=30% 且 换手<=15%, nan→0 视为过闸); 键 = bias60 最热 top15。
  窗口 d3~7 与 TOP15 已拍板, 勿再扫; dry 阈值 0.8 为终版, 池条件线已关闭。
"""

import numpy as np
import pandas as pd

from scripts._gap_pocket_shadow import (
    BIAS_WIN,
    D_HI,
    D_LO,
    DRY_MAX,
    GAP_TOP_N,
    R10_MAX,
    TOV_MAX,
    gap_pocket_picks,
)

N = 80  # ≥ BIAS_WIN+1, 保证 bias 有值


def _frames():
    """4 只票: A 合格(d=4) / B 破板价 / C 事件内再板 / D 不缩量."""
    idx = pd.bdate_range("2026-01-01", periods=N)
    i0 = N - 5  # 首板在 4 夜前 → d=4

    def close(path):
        v = np.full(N, 10.0)
        for k, val in path.items():
            v[k] = val
        return v

    a = close({i0: 11.0, i0 + 1: 11.1, i0 + 2: 11.2, i0 + 3: 11.3, i0 + 4: 11.4})   # 合格
    b = close({i0: 11.0, i0 + 1: 10.9, i0 + 2: 10.8, i0 + 3: 10.7, i0 + 4: 10.6})   # 守板 fail
    c = close({i0: 11.0, i0 + 1: 11.1, i0 + 2: 12.2, i0 + 3: 12.3, i0 + 4: 12.4})   # 再板 fail
    d = close({i0: 11.0, i0 + 1: 11.05, i0 + 2: 11.1, i0 + 3: 11.15, i0 + 4: 11.2})  # 缩量 fail

    def pct_of(board_days):
        v = np.full(N, 0.5)
        for k in board_days:
            v[k] = 10.0
        return v

    pct = pd.DataFrame({"600001": pct_of([i0]), "600002": pct_of([i0]),
                        "600003": pct_of([i0, i0 + 2]), "600004": pct_of([i0])}, index=idx)
    close_df = pd.DataFrame({"600001": a, "600002": b, "600003": c, "600004": d}, index=idx)
    # A/B/C 近3日均量 0.4×首板量 (缩量过); D 全程 3e8 → dry=1.0 fail
    amt = pd.DataFrame({
        col: np.r_[np.full(i0, 1e8), 3e8, np.full(N - i0 - 1, 1.2e8)]
        for col in ("600001", "600002", "600003")}, index=idx)
    amt["600004"] = 3e8
    tov = pd.DataFrame(np.full((N, 4), 5.0), index=idx,
                       columns=["600001", "600002", "600003", "600004"])
    return close_df, pct, amt, tov


def _single(d):
    """一只干净合格票 (首板在 d 夜前, 守板+缩量+过闸). 返回四张透视表."""
    idx = pd.bdate_range("2026-01-01", periods=N)
    i0 = N - 1 - d
    v = np.full(N, 10.0)
    v[i0:] = np.linspace(11.0, 11.0 + 0.1 * d, d + 1)
    close_df = pd.DataFrame({"600100": v}, index=idx)
    pct = pd.DataFrame({"600100": np.where(np.arange(N) == i0, 10.0, 0.5)}, index=idx)
    amt = pd.DataFrame({"600100": np.r_[np.full(i0, 1e8), 3e8,
                                        np.full(N - i0 - 1, 1.2e8)]}, index=idx)
    tov = pd.DataFrame({"600100": np.full(N, 5.0)}, index=idx)
    return close_df, pct, amt, tov


def _call(frames, day_ts=None, **kw):
    close_df, pct, amt, tov = frames
    return gap_pocket_picks(close_df, pct, amt, tov,
                            day_ts or close_df.index[-1], **kw)


def test_picks_only_qualified_pool():
    out = _call(_frames())
    assert list(out["symbol"]) == ["600001"]  # B 破板价 / C 再板 / D 不缩量 全剔
    assert out.iloc[0]["d"] == 4
    assert abs(out.iloc[0]["dry"] - 0.4) < 1e-9
    assert abs(out.iloc[0]["pull"] - 11.4 / 11.0 + 1) < 1e-9


def test_rank_is_bias_descending():
    f = _frames()
    close_df, pct, amt, tov = f
    # 加一只 bias 更热的合格票 (底价相同、板上翻更高) → 应排第 1
    hot = np.full(N, 10.0)
    hot[N - 4:] = 12.5  # r10 = 0.25 过闸; bias ≈ 0.229 > 600001 的 0.129
    close_df["600005"] = hot
    pct["600005"] = np.where(np.arange(N) == N - 4, 10.0, 0.5)
    amt["600005"] = np.r_[np.full(N - 4, 1e8), 3e8, np.full(3, 1.2e8)]
    tov["600005"] = 5.0
    out = gap_pocket_picks(close_df, pct, amt, tov, close_df.index[-1])
    assert list(out["symbol"]) == ["600005", "600001"]
    assert out["bias"].is_monotonic_decreasing


def test_gate_filters_hot_and_high_turnover():
    f = _frames()
    close_df, pct, amt, tov = f
    i0 = N - 5
    hot = close_df.copy()
    hot.iloc[: i0 - 5, 0] = 8.0  # C[t-10]=8 → r10≈0.42 > 30% → 闸剔
    assert _call((hot, pct, amt, tov)).empty
    high_tov = tov.copy()
    high_tov["600001"] = 20.0  # 换手 > 15 → 闸剔
    assert _call((close_df, pct, amt, high_tov)).empty
    edge_tov = tov.copy()
    edge_tov["600001"] = TOV_MAX  # 恰好 15 → 过闸
    out = _call((close_df, pct, amt, edge_tov))
    assert list(out["symbol"]) == ["600001"]


def test_gate_nan_treated_as_pass():
    f = _frames()
    close_df, pct, amt, tov = f
    tov_nan = tov.copy()
    tov_nan["600001"] = np.nan
    out = _call((close_df, pct, amt, tov_nan))
    assert list(out["symbol"]) == ["600001"]
    assert out.iloc[0]["tov"] == 0.0


def test_window_bounds_d3_to_d7():
    for d, expect in ((2, False), (3, True), (7, True), (8, False)):
        out = _call(_single(d))
        assert ("600100" in list(out["symbol"])) == expect, f"d={d}"
        if expect:
            assert out.iloc[0]["d"] == d


def test_top_n_cap():
    f = _frames()
    close_df, pct, amt, tov = f
    for k in range(20):
        col = f"{601000 + k:06d}"
        i0 = N - 1 - 3 - (k % 5)  # d ∈ 3..7 分散
        v = np.full(N, 10.0)
        v[i0] = 11.0
        v[i0 + 1:] = 11.011
        close_df[col] = v
        pct[col] = np.where(np.arange(N) == i0, 10.0, 0.5)
        amt[col] = np.r_[np.full(i0, 1e8), 3e8, np.full(N - i0 - 1, 1.2e8)]
        tov[col] = 5.0
    out = _call(f, top_n=10)
    assert len(out) == 10
    assert out["bias"].is_monotonic_decreasing
    assert list(out["rank"]) == list(range(1, 11))
    assert len(_call(f, top_n=5)) == 5


def test_short_history_empty():
    close_df, pct, amt, tov = _frames()
    out = gap_pocket_picks(close_df.iloc[:30], pct.iloc[:30], amt.iloc[:30],
                           tov.iloc[:30], close_df.index[29])
    assert out.empty
    assert list(out.columns) == ["rank", "symbol", "d", "dry", "pull", "r10", "tov"]


def test_uses_only_history_up_to_day_ts():
    close_df, pct, amt, tov = _frames()
    ts = close_df.index[-1]
    out = gap_pocket_picks(close_df.iloc[:-1], pct.iloc[:-1], amt.iloc[:-1],
                           tov.iloc[:-1], ts)  # 截断窗内 d=3 仍可判
    assert set(out["symbol"]).issubset({"600001", "600002", "600003", "600004"})


def test_excludes_bse():
    f = _frames()
    close_df, pct, amt, tov = f
    close_df["920999.BJ"] = close_df["600001"] * 1.2
    pct["920999.BJ"] = pct["600001"]
    amt["920999.BJ"] = amt["600001"]
    tov["920999.BJ"] = 5.0
    out = _call(f)
    assert "920999.BJ" not in list(out["symbol"])
    assert len(out) == 1


def test_deterministic():
    f = _frames()
    pd.testing.assert_frame_equal(_call(f), _call(f))


def test_constants_locked():
    """回放拍板口径, 勿静默改."""
    assert GAP_TOP_N == 15
    assert (D_LO, D_HI) == (3, 7)
    assert DRY_MAX == 0.8
    assert R10_MAX == 0.30
    assert TOV_MAX == 15.0
    assert BIAS_WIN == 60
