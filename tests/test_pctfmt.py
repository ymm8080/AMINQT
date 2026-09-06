"""交付 CSV 百分比显示层单测: fmt_pct_columns 格式化 + parse_pct 两代容忍读.

背景 (2026-09-05 用户: 预测值用百分比): 写出层格式化副本, 上游保持数值;
WORM 旧文件 (数值) 与新文件 ("xx.xx%") 盘上并存, 链上读者 parse_pct 两代通吃.
"""

import numpy as np
import pandas as pd
import pytest

from scripts._pctfmt import (
    PCT_COLS_LEGACY,
    PCT_COLS_PARALLEL,
    fmt_pct_columns,
    parse_pct,
)


def test_fmt_multiplies_ratio_and_appends_pct():
    df = pd.DataFrame({"pred_ret_10d": [0.0321, 0.5, -0.0244]})
    out = fmt_pct_columns(df, ("pred_ret_10d",))
    assert list(out["pred_ret_10d"]) == ["3.21%", "50.00%", "-2.44%"]
    # 纯显示层: 入参不改
    assert df["pred_ret_10d"].iloc[0] == pytest.approx(0.0321)


def test_fmt_already_pct_cols_skip_multiply():
    df = pd.DataFrame({"pctChg": [-3.105590, 1.285347], "pred_prob_10d": [0.78, 0.6]})
    out = fmt_pct_columns(df, ("pctChg", "pred_prob_10d"), already_pct_cols=("pctChg",))
    assert list(out["pctChg"]) == ["-3.11%", "1.29%"]  # 只加 % 不 ×100
    assert list(out["pred_prob_10d"]) == ["78.00%", "60.00%"]


def test_fmt_nan_to_empty_and_missing_cols_ignored():
    df = pd.DataFrame({"prob_up": [0.9, np.nan], "score": [1.234, 5.678]})
    out = fmt_pct_columns(df, ("prob_up", "pred_ret_3d"))  # pred_ret_3d 不在
    assert list(out["prob_up"]) == ["90.00%", ""]
    # 未列出列 (复合分) 原样数值
    assert out["score"].tolist() == pytest.approx([1.234, 5.678])
    assert "pred_ret_3d" not in out.columns


def test_parse_pct_roundtrip_both_generations():
    # 新一代: "xx.xx%" 文本
    assert parse_pct("78.16%") == pytest.approx(0.7816)
    assert parse_pct("-2.44%") == pytest.approx(-0.0244)
    # 旧一代: 数值/数值串透传
    assert parse_pct(0.7816) == pytest.approx(0.7816)
    assert parse_pct("0.7816") == pytest.approx(0.7816)
    # NaN/空
    assert np.isnan(parse_pct(np.nan))
    assert np.isnan(parse_pct(None))
    assert np.isnan(parse_pct("nan"))
    assert np.isnan(parse_pct(""))


def test_parse_pct_garbage_raises():
    with pytest.raises(ValueError):
        parse_pct("abc")


def test_roundtrip_fmt_then_parse():
    df = pd.DataFrame({"pred_prob_10d": [0.7816, 0.5], "ret_10d": [0.123, -0.02]})
    out = fmt_pct_columns(df, ("pred_prob_10d", "ret_10d"))
    for c in ("pred_prob_10d", "ret_10d"):
        assert out[c].map(parse_pct).tolist() == pytest.approx(df[c].tolist())


def test_mixed_generation_files_concat(tmp_path):
    """过渡期实况: glob 拼接新旧两代文件, parse_pct 后统一数值."""
    old = pd.DataFrame({"symbol": ["600001"], "pred_prob_10d": [0.6]})
    new = pd.DataFrame({"symbol": ["600002"], "pred_prob_10d": ["55.00%"]})
    old.to_csv(tmp_path / "parallel_shortlist_20260901__A.csv", index=False)
    new.to_csv(tmp_path / "parallel_shortlist_20260902__B.csv", index=False)
    parts = [
        pd.read_csv(
            tmp_path / "parallel_shortlist_20260901__A.csv", dtype={"symbol": str}
        ),
        pd.read_csv(
            tmp_path / "parallel_shortlist_20260902__B.csv", dtype={"symbol": str}
        ),
    ]
    df = pd.concat(parts, ignore_index=True)
    df["pred_prob_10d"] = df["pred_prob_10d"].map(parse_pct)
    assert df["pred_prob_10d"].tolist() == pytest.approx([0.6, 0.55])


def test_pct_col_lists_cover_delivered_prediction_columns():
    """两份列清单必须覆盖交付文件的核心预测列 (机器读者的解析依据)."""
    for c in ("pred_mag_10d", "pred_prob_10d", "pred_ret_3d"):
        assert c in PCT_COLS_PARALLEL
    for c in ("day_change", "prob_up_10d", "pred_ret_10d", "weight"):
        assert c in PCT_COLS_LEGACY
    # 复合分/秩绝不可进清单 (列位语义不是比例, 格式化即毁)
    for banned in ("score", "rank_blend", "norm_score"):
        assert banned not in PCT_COLS_PARALLEL
        assert banned not in PCT_COLS_LEGACY
