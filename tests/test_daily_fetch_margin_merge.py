"""_daily_fetch margin 合并回归测试 (2026-08-17 事故).

面板 07-31 V3 重建后 margin 4 列已重命名 (rzye→margin_balance 等), 而 merges 通用
路径按"源列名 in panel_cols"守卫 → 永远跳过, margin 全靠 ffill 冻结; 08-14 加的
T+1 回退拉到真实值也被丢弃. merge_margin_renamed 直接映射修复该路径.
"""

from __future__ import annotations

import pandas as pd

from app.pipeline1.data_supply import merge_margin_renamed

PANEL_COLS = [
    "symbol",
    "date",
    "open",
    "close",
    "margin_balance",
    "short_balance",
    "margin_buy_amt",
    "short_sell_vol",
]


def _today_frame(symbols, with_margin=False):
    df = pd.DataFrame({"symbol": symbols, "date": pd.Timestamp("2026-08-17")})
    if with_margin:
        for c in [
            "margin_balance",
            "short_balance",
            "margin_buy_amt",
            "short_sell_vol",
        ]:
            df[c] = 999.0
    return df


def _margin_frame(symbols, values=None):
    raw = pd.DataFrame(
        {
            "symbol": [s + ".SZ" for s in symbols],
            "rzye": values or [100.0 + i for i in range(len(symbols))],
            "rqye": [200.0 + i for i in range(len(symbols))],
            "rzmre": [300.0 + i for i in range(len(symbols))],
            "rqyl": [400.0 + i for i in range(len(symbols))],
        }
    )
    raw["symbol"] = raw["symbol"].str.replace(".SZ", "")
    return raw


class TestMergeMarginRenamed:
    def test_panel_cols_without_raw_names_still_merges(self):
        """回归: 面板只有重命名列 (margin_balance 等) 时, 真实值必须写入今日行."""
        df = _today_frame(["000001", "000002", "000006"])
        margin = _margin_frame(["000001", "000002", "000006"], [111.0, 222.0, 333.0])
        merge_margin_renamed(df, margin, PANEL_COLS)
        assert df.loc[0, "margin_balance"] == 111.0
        assert df.loc[1, "margin_balance"] == 222.0
        assert df.loc[2, "margin_balance"] == 333.0
        assert df.loc[0, "short_balance"] == 200.0
        assert df.loc[0, "margin_buy_amt"] == 300.0
        assert df.loc[0, "short_sell_vol"] == 400.0

    def test_does_not_overwrite_existing_values(self):
        """今日行已有实时值 → 不覆盖 (与 fillna 语义一致)."""
        df = _today_frame(["000001"], with_margin=True)
        margin = _margin_frame(["000001"], [111.0])
        merge_margin_renamed(df, margin, PANEL_COLS)
        assert df.loc[0, "margin_balance"] == 999.0

    def test_missing_symbols_stay_nan(self):
        """margin 未覆盖的 symbol → NaN (留给 ffill)."""
        df = _today_frame(["000001", "000002"])
        margin = _margin_frame(["000001"])
        merge_margin_renamed(df, margin, PANEL_COLS)
        assert df.loc[0, "margin_balance"] == 100.0
        assert pd.isna(df.loc[1, "margin_balance"])

    def test_empty_margin_is_noop(self):
        df = _today_frame(["000001"])
        merge_margin_renamed(df, pd.DataFrame(), PANEL_COLS)
        assert "margin_balance" not in df.columns

    def test_missing_panel_target_column_is_skipped(self):
        """面板缺目标列 → 跳过该列, 不报错."""
        df = _today_frame(["000001"])
        cols = [c for c in PANEL_COLS if c != "short_balance"]
        merge_margin_renamed(df, _margin_frame(["000001"]), cols)
        assert df.loc[0, "margin_balance"] == 100.0
        assert "short_balance" not in df.columns
