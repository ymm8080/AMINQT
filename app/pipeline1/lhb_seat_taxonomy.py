# -*- coding: utf-8 -*-
"""LHB 席位分类表构建 (生产模块, 2026-09-23 从 tmp_t/_0923_w13_seat_taxonomy.py 第 1~5 节
逐字迁移). 「首板点名页」的 6 个纯标注列 (scripts/_firstboard_pages.py::_lhb_annotations)
消费本模块产出的 labeled 表; 口径必须与研究脚本完全一致, 不得"改进".

分类: 机构专用 / 沪深股通 / 拉萨散户团 / 高频游资(统计) / 普通营业部
输入: top_inst 席位明细 raw (10 列, 见 SEAT_RAW_COLS)
输出: (labeled_df, master_df)
  labeled_df — 原始 10 列 + dt/exalter_norm/category (逐行标签, 供页面 join)
  master_df  — exalter 级主表 (席位数/覆盖/买卖额)
全向量化 (groupby agg), 无逐行循环。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CAT_ORDER = ["机构专用", "沪深股通", "拉萨散户团", "高频游资", "普通营业部"]

# 原始席位明细 10 列 (top_inst 投影后的既有 raw schema; 顺序即落盘顺序)
SEAT_RAW_COLS = [
    "trade_date",
    "ts_code",
    "exalter",
    "buy",
    "buy_rate",
    "sell",
    "sell_rate",
    "net_buy",
    "side",
    "reason",
]
# labeled 表 schema = raw 10 列 + 派生 3 列 (顺序与 lhb_seat_detail_labeled.parquet 一致)
SEAT_LABELED_COLS = SEAT_RAW_COLS + ["dt", "exalter_norm", "category"]
# master 表 schema (rule_cat 已并入 category, 不落盘)
SEAT_MASTER_COLS = [
    "exalter_norm",
    "exalter",
    "n_rows",
    "n_stocks",
    "n_days",
    "first_date",
    "last_date",
    "total_buy",
    "total_sell",
    "net_total",
    "buy_sell_ratio",
    "category",
]


def _empty_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(columns=SEAT_LABELED_COLS),
        pd.DataFrame(columns=SEAT_MASTER_COLS),
    )


def label_seat_frame(seat_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """席位明细 raw → (逐行 labeled, 席位 master).

    空输入返回两个空 DataFrame (不抛异常) — 日更当日 top_inst 无数据时页面走空标注,
    绝不因无 LHB 事件而中断链路."""
    if seat_df is None or len(seat_df) == 0:
        return _empty_frames()

    df = seat_df[SEAT_RAW_COLS].copy()
    df["dt"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")

    # exalter 清洗: 去首尾空白 + 归一化键 (去全部空白 + 全角括号→半角, 合并同名异写)
    df["exalter"] = df["exalter"].str.strip()
    df["exalter_norm"] = (
        df["exalter"]
        .str.replace(r"\s+", "", regex=True)
        .str.replace("（", "(")
        .str.replace("）", ")")
    )

    # ============ 规则类别 (后覆盖前: 营业部 → 拉萨 → 沪深股通 → 机构专用) ============
    is_inst = df["exalter"].str.contains("机构专用", na=False)
    is_connect = df["exalter"].str.contains("沪股通|深股通", na=False, regex=True)
    is_lasa = df["exalter"].str.contains("拉萨", na=False)

    rule_cat = pd.Series("普通营业部", index=df.index, dtype=object)
    rule_cat[is_lasa] = "拉萨散户团"
    rule_cat[is_connect] = "沪深股通"
    rule_cat[is_inst] = "机构专用"
    df["_seat_category"] = rule_cat

    # ============ 席位统计 (exalter 级, 向量化 agg) ============
    df["_buy_f"] = df["buy"].fillna(0.0)
    df["_sell_f"] = df["sell"].fillna(0.0)
    g = df.groupby("exalter_norm", sort=False)
    master = g.agg(
        exalter=("exalter", "first"),  # 原始写法代表
        n_rows=("ts_code", "size"),
        n_stocks=("ts_code", "nunique"),
        n_days=("trade_date", "nunique"),
        first_date=("dt", "min"),
        last_date=("dt", "max"),
        total_buy=("_buy_f", "sum"),
        total_sell=("_sell_f", "sum"),
        rule_cat=("_seat_category", "first"),
    ).reset_index()
    # 规则类在所有行应一致 (归一化键不会把两个规则类合并), 不一致即口径 bug → 大声失败
    chk = df.groupby("exalter_norm")["_seat_category"].nunique()
    assert (chk == 1).all(), "同一 exalter_norm 规则类别不一致!"

    master["net_total"] = master["total_buy"] - master["total_sell"]
    master["buy_sell_ratio"] = np.where(
        master["total_sell"] > 0, master["total_buy"] / master["total_sell"], np.nan
    )

    # ============ 高频游资阈值 (统计, 在规则外「普通营业部」席位内取分位) ============
    rest = master[master["rule_cat"] == "普通营业部"]
    thresh_stk = rest["n_stocks"].quantile(0.90)
    thresh_rows = max(100, rest["n_rows"].quantile(0.90))
    is_hf = (master["rule_cat"] == "普通营业部") & (
        (master["n_stocks"] >= thresh_stk) & (master["n_rows"] >= thresh_rows)
    )
    master["category"] = master["rule_cat"]
    master.loc[is_hf, "category"] = "高频游资"
    master["category"] = pd.Categorical(
        master["category"], categories=CAT_ORDER, ordered=True
    )

    # ============ 明细打标签 (回填全行, 按归一化键 merge) ============
    labeled = df.drop(columns=["_seat_category", "_buy_f", "_sell_f"]).merge(
        master[["exalter_norm", "category"]], on="exalter_norm", how="left"
    )
    assert labeled["category"].notna().all()
    labeled = labeled[SEAT_LABELED_COLS].reset_index(drop=True)
    master = master.drop(columns=["rule_cat"])[SEAT_MASTER_COLS].reset_index(drop=True)
    return labeled, master
