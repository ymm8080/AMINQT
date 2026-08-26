"""跨模块影子排名单测 — build_shadow 纯函数 (2026-08-26).

验证意图: 百分位归一口径 (第1名 1.0/末名 0.0/单票 1.0), 板组内互不污染,
缺席模块 pct=0, 双源票 0.5/0.5 混排应压过单源票, TOP-N 截断, 输出协议列齐.
"""

import pandas as pd

from scripts._shadow_xmodule_blend import (
    LEGACY_BOARD,
    SHADOW_COLS,
    build_shadow,
)

W = {"legacy": 0.5, "parallel": 0.5}


def _legacy(rows):
    # rows = (symbol, board, prob_up)
    return pd.DataFrame(
        rows, columns=["symbol", "board_group", "key"]
    )


def test_pct_normalization_top1_and_last():
    """3 票: 第1名 pct=1.0, 第2名 0.5, 第3名 0.0."""
    legacy = _legacy([("600001", "main", 0.9), ("600002", "main", 0.6), ("600003", "main", 0.3)])
    res = build_shadow(legacy, None, W, 10)
    by = res.set_index("symbol")["legacy_pct"]
    assert by["600001"] == 1.0 and by["600002"] == 0.5 and by["600003"] == 0.0
    assert (res["in_parallel"] == False).all()  # noqa: E712
    assert (res["source"] == "legacy").all()


def test_pct_single_row_is_one():
    """组内单票 n=1 → pct=1.0 而非除零."""
    legacy = _legacy([("600001", "main", 0.9), ("300001", "dual", 0.5)])
    res = build_shadow(legacy, None, W, 10)
    dual = res[res["board"] == "dual"].iloc[0]
    assert dual["legacy_pct"] == 1.0


def test_pct_is_within_board_not_global():
    """各板内独立归一: main 末名 0.0 与 dual 头名 1.0 同帧并存."""
    legacy = _legacy(
        [("600001", "main", 0.9), ("600002", "main", 0.1), ("300001", "dual", 0.5)]
    )
    res = build_shadow(legacy, None, W, 10).set_index("symbol")
    assert res.loc["600002", "legacy_pct"] == 0.0
    assert res.loc["300001", "legacy_pct"] == 1.0


def test_missing_module_pct_zero_both_sides_outrank_single():
    """双源票 blend=1.0 > 单源中位票; 缺席侧 pct=0 且 source 如实标模块."""
    legacy = _legacy([("600001", "main", 0.9), ("600002", "main", 0.8), ("600004", "main", 0.7)])
    parallel = _legacy([("600001", "main", 0.95), ("600003", "main", 0.7)])
    res = build_shadow(legacy, parallel, W, 10).set_index("symbol")
    assert res.loc["600001", "blend"] == 1.0
    assert res.loc["600001", "source"] == "both"
    assert res.loc["600001", "parallel_pct"] == 1.0
    # 600002 是 legacy 3 票第 2 名 pct=0.5, parallel 缺 → blend=0.25
    assert res.loc["600002", "parallel_pct"] == 0.0
    assert res.loc["600002", "blend"] == 0.25
    # 600003 是 parallel 2 票末名 pct=0.0, 但仍是 parallel 成员 (与"缺席"区分)
    assert res.loc["600003", "legacy_pct"] == 0.0
    assert res.loc["600003", "source"] == "parallel"
    assert res.loc["600003", "in_parallel"] == True  # noqa: E712
    # 排名: both 票第一, 单源 0.25 票其次, 0.0 末位
    assert res.loc["600001", "shadow_rank"] == 1
    assert res.loc["600002", "shadow_rank"] == 2


def test_top_n_cut_per_board():
    """TOP-N 按板截断: main 池 5 票取 3, dual 不受影响."""
    legacy = _legacy(
        [(f"60000{i}", "main", 0.9 - i * 0.1) for i in range(5)]
        + [("300001", "dual", 0.5)]
    )
    res = build_shadow(legacy, None, W, 3)
    assert (res[res["board"] == "main"].shape[0] == 3)
    assert (res[res["board"] == "dual"].shape[0] == 1)
    # 截掉的必是低分票
    assert "600004" not in set(res["symbol"])


def test_union_pool_symbol_appears_once():
    """并池去重: 同 symbol 双源各一行 → 输出一行 (source=both)."""
    legacy = _legacy([("600001", "main", 0.9)])
    parallel = _legacy([("600001", "main", 0.8)])
    res = build_shadow(legacy, parallel, W, 10)
    assert len(res) == 1
    assert res.iloc[0]["source"] == "both"


def test_both_empty_returns_empty_with_protocol_cols():
    res = build_shadow(None, None, W, 10)
    assert res.empty
    assert list(res.columns) == SHADOW_COLS[1:]


def test_legacy_board_mapping_values():
    """GEM/STAR → dual, main → main (映射表内容锁定)."""
    assert LEGACY_BOARD == {"main": "main", "GEM": "dual", "STAR": "dual"}


def test_shadow_rank_contiguous_per_board():
    """shadow_rank 板内从 1 连续编号."""
    legacy = _legacy(
        [("600001", "main", 0.9), ("600002", "main", 0.8), ("300001", "dual", 0.5)]
    )
    res = build_shadow(legacy, None, W, 10)
    for _, g in res.groupby("board"):
        assert list(g["shadow_rank"]) == list(range(1, len(g) + 1))
