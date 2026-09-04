"""A1 动量影子单纯函数单测 (2026-09-04, scripts/_a1_momentum_shadow.py).

口径锁死 (125d 回放判决 2, 勿静默改):
  A1 = bias_20/60/120/250 截面百分位秩 nanmean;
  剔当日强涨 pctChg>=9.5% (双创 19%); A1 降序 top10。
"""

import numpy as np
import pandas as pd

from scripts._a1_momentum_shadow import BIAS_COLS, a1_top10


def _day(n: int = 12) -> pd.DataFrame:
    """构造可人工判读的单日截面: symbol i 的 bias 列值单调, i 越大动量越高."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "symbol": f"{600000 + i}",
                "bias_20": float(i),
                "bias_60": float(i),
                "bias_120": float(i),
                "bias_250": float(i),
                "pctChg": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_a1_picks_highest_momentum_in_order():
    df = _day()
    out = a1_top10(df)
    assert list(out["rank"]) == list(range(1, 11))
    # i 越大 bias 越高 → 截面秩越高 → top10 是 i=11..2 降序
    assert list(out["symbol"]) == [f"{600000 + i}" for i in range(11, 1, -1)]
    # a1 = 四列截面秩的均值; 满截面时最强者 = 1.0
    assert out["a1"].iloc[0] == 1.0
    assert out["a1"].is_monotonic_decreasing


def test_partial_nan_bias_uses_nanmean():
    """部分 bias 列 NaN → 用可用列均值 (nanmean 语义), 不整行丢弃."""
    df = _day()
    df.loc[df.index[-1], "bias_250"] = np.nan  # 最强股缺一列
    df.loc[df.index[0], BIAS_COLS] = np.nan  # 最弱股全缺
    out = a1_top10(df)
    syms = list(out["symbol"])
    assert f"{600000 + 11}" == syms[0]  # 缺一列仍以 3 列满秩登顶
    assert f"{600000 + 0}" not in syms  # 全缺被丢


def test_hot_exclusion_main_board():
    df = _day()
    df.loc[df.index[-1], "pctChg"] = 9.9  # 主板最强股强涨
    out = a1_top10(df)
    assert f"{600000 + 11}" not in list(out["symbol"])
    assert list(out["symbol"])[0] == f"{600000 + 10}"


def test_hot_exclusion_creates_board_threshold_19():
    df = _day(14)
    c30 = df.index[df["symbol"] == "600012"][0]
    c68 = df.index[df["symbol"] == "600013"][0]
    df.loc[c30, "symbol"] = "300012"
    df.loc[c68, "symbol"] = "688013"
    df.loc[c30, "pctChg"] = 9.7  # 双创 <19% 不剔
    df.loc[c68, "pctChg"] = 19.5  # 双创 >=19% 剔
    out = a1_top10(df)
    syms = list(out["symbol"])
    assert "300012" in syms
    assert "688013" not in syms


def test_symbol_zero_padded():
    df = _day(3)
    df["symbol"] = ["1", "600001", "600002"]
    out = a1_top10(df)
    assert "000001" in list(out["symbol"])


def test_fewer_than_10_candidates():
    df = _day(6)
    out = a1_top10(df)
    assert len(out) == 6
    assert list(out["rank"]) == [1, 2, 3, 4, 5, 6]


def test_deterministic():
    a = a1_top10(_day())
    b = a1_top10(_day())
    pd.testing.assert_frame_equal(a, b)
