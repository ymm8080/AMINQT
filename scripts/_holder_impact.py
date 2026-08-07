"""量化增减持日频数据的整体经济影响 (ht 股票 + ±1月窗口)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import _holder_scheme_ic as H  # noqa: E402  (复用加载/特征/标签)


def main() -> None:
    df = pd.read_parquet(H.PANEL, columns=["symbol", "date", "close_hfq", "board"])
    df = df[df["board"] == "main"].copy().dropna(subset=["close_hfq"])
    df["date"] = pd.to_datetime(df["date"])
    raw = pd.read_parquet(H.RAW)
    raw["date"] = pd.to_datetime(raw["date"])
    for t in ("G", "P", "C"):
        raw["sr_" + t.lower()] = np.where(
            raw["holder_type"] == t, raw["signed_ratio"], 0.0
        )
    agg = raw.groupby(["symbol", "date"], as_index=False).agg(
        net_ratio=("signed_ratio", "sum"),
        g_ratio=("sr_g", "sum"),
        p_ratio=("sr_p", "sum"),
        c_ratio=("sr_c", "sum"),
        evt_start_date=("evt_start_date", "min"),
        evt_end_date=("evt_end_date", "max"),
    )
    df = df.merge(agg, on=["symbol", "date"], how="left")
    df = H.per_symbol_features(df)
    df = H.add_labels(df)

    ht = set(raw["symbol"].unique())
    sub = df[df["symbol"].isin(ht) & (df["_active"] == 1.0)].copy()

    base1 = df["label_pm_1d"].mean()
    base5 = df["label_pm_5d"].mean()

    print("=" * 88)
    print("1. 事件频率 (2023-01 ~ 2026-07, 3.5年)")
    print("=" * 88)
    ev = raw.drop_duplicates(["symbol", "date"])
    print(f"  公告事件日: {len(ev)} 涉及股票: {ev['symbol'].nunique()}")
    print(
        f"  主板有事件股票: {len(ht & set(df['symbol']))} / 全主板 {df['symbol'].nunique()}"
    )
    per = ev.groupby("symbol").size()
    print(
        f"  每只股票事件次数: 均值 {per.mean():.1f}, 中位 {per.median():.0f}, 最多 {per.max()}"
    )
    print(f"  平均每股每年事件: {per.mean() / 3.5:.2f} 次")
    print(f"  全主板行属 ht±1月窗口: {len(sub) / len(df) * 100:.1f}%")
    inwin_evt = sub["net_ratio"].notna().sum()
    print(f"  窗口内当日有公告的行: {inwin_evt} ({inwin_evt / len(sub) * 100:.1f}%)")

    print()
    print("=" * 88)
    print("2. 基线前向收益 (label_pm, 全主板)")
    print("=" * 88)
    print(f"  base 1d={base1 * 100:.3f}%  5d={base5 * 100:.3f}%")

    print()
    print("=" * 88)
    print("3. kimi_ratio_30d 五分位 → 前向收益 (ht±1月窗口)")
    print("=" * 88)
    q = sub.dropna(subset=["kimi_ratio_30d", "label_pm_1d", "label_pm_5d"])
    q["bin"] = pd.qcut(q["kimi_ratio_30d"], 5, labels=False, duplicates="drop")
    g = q.groupby("bin").agg(
        n=("label_pm_1d", "size"),
        f1=("label_pm_1d", "mean"),
        f5=("label_pm_5d", "mean"),
        x=("kimi_ratio_30d", "mean"),
    )
    g["f1_%"] = g["f1"] * 100
    g["f5_%"] = g["f5"] * 100
    print(g[["n", "x", "f1_%", "f5_%"]].round(4).to_string())
    ls1 = g.iloc[-1]["f1"] - g.iloc[0]["f1"]
    ls5 = g.iloc[-1]["f5"] - g.iloc[0]["f5"]
    print(f"  Q5-Q1 spread: 1d={ls1 * 100:+.3f}%  5d={ls5 * 100:+.3f}%")

    print()
    print("=" * 88)
    print("4. 增持 vs 减持公告 次日收益 (事件日, 窗口内)")
    print("=" * 88)
    evd = sub[sub["net_ratio"].notna() & sub["label_pm_1d"].notna()]
    up = evd[evd["net_ratio"] > 0]
    dn = evd[evd["net_ratio"] < 0]
    print(f"  增持公告(n={len(up)}): 1d={up['label_pm_1d'].mean() * 100:+.3f}%")
    print(f"  减持公告(n={len(dn)}): 1d={dn['label_pm_1d'].mean() * 100:+.3f}%")
    print(
        f"  spread = {(up['label_pm_1d'].mean() - dn['label_pm_1d'].mean()) * 100:+.3f}%"
    )
    print(
        f"  全事件样本 1d 均值 = {evd['label_pm_1d'].mean() * 100:+.3f}%  vs 基线 {base1 * 100:.3f}%"
    )

    print()
    print("=" * 88)
    print("5. 有近期事件 vs 无事件 (窗口内, 同股票集合)")
    print("=" * 88)
    # 有公告的当日 vs 窗口内非公告日
    evt_row = sub[sub["net_ratio"].notna()]["label_pm_1d"]
    noevt_row = sub[sub["net_ratio"].isna()]["label_pm_1d"]
    print(f"  公告当日(n={len(evt_row)}): 1d={evt_row.mean() * 100:+.3f}%")
    print(f"  窗口内非公告日(n={len(noevt_row)}): 1d={noevt_row.mean() * 100:+.3f}%")
    print(f"  差异 = {(evt_row.mean() - noevt_row.mean()) * 100:+.3f}%")

    # 有事件股票 vs 无事件股票的全体基线
    ht_sym_all = df[df["symbol"].isin(ht)]["label_pm_1d"]
    nonht_sym = df[~df["symbol"].isin(ht)]["label_pm_1d"]
    print()
    print(
        f"  有事件股票全体(任意日): 1d={ht_sym_all.mean() * 100:+.3f}%  (n={len(ht_sym_all)})"
    )
    print(
        f"  无事件股票全体(任意日): 1d={nonht_sym.mean() * 100:+.3f}%  (n={len(nonht_sym)})"
    )


if __name__ == "__main__":
    main()
