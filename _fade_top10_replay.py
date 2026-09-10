# -*- coding: utf-8 -*-
"""历史 TOP10 交付清单回放: 冲高回落标记对选票质量的影响.

两类标记 (清单日晚 t 收盘后均可知, 无 look-ahead):
  A. 状态型 fade_score: vol20/turn20/r20/pos250 的日截面秩复合, 高=爱冲高回落体质
  B. 事件型 fade_today: t 当日该票本身冲高回落 (g>=3% & 吐回>=70% & 未触板)
执行日 D = t 的下一交易日, 结果:
  o2c=close(D)/open(D)-1 (开盘买入), c2c=ret(D), low_o=low(D)/open(D)-1 (日内最深浮亏),
  ret5=close(D+4)/close(D)-1, fade_D=D 日是否冲高回落
"""
import os
import re

import numpy as np
import pandas as pd

LISTDIR = r"D:/AMINQT/DAILY OPERATION/STOCK LIST"
PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
IDX = r"data/processed/sh_index_daily_20260909.parquet"


def load_lists() -> pd.DataFrame:
    rows = []
    for f in sorted(os.listdir(LISTDIR)):
        m = re.match(r"^legacy_stocklist_(\d{8})__.*\.csv$", f)
        if not m:
            continue
        rows.append((m.group(1), f))
    # 每个清单日取排序最后一个文件 (修订版优先)
    best = {}
    for d, f in rows:
        best[d] = f
    out = []
    for d in sorted(best):
        df = pd.read_csv(os.path.join(LISTDIR, best[d]), dtype={"symbol": str})
        df["list_date"] = pd.to_datetime(d)
        out.append(df[["symbol", "board", "list_date"]])
    L = pd.concat(out, ignore_index=True)
    L = L.drop_duplicates(subset=["list_date", "symbol"])
    print(f"[lists] {L['list_date'].nunique()} 个清单日, {len(L)} 只票 (去重后)")
    return L


def main():
    picks = load_lists()
    df = pd.read_parquet(PANEL, columns=[
        "symbol", "date", "open", "high", "low", "close", "pre_close",
        "amount", "turnover_rate"])
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    c = df.pivot(index="date", columns="symbol", values="close").sort_index()
    o = df.pivot(index="date", columns="symbol", values="open").reindex_like(c)
    h = df.pivot(index="date", columns="symbol", values="high").reindex_like(c)
    l = df.pivot(index="date", columns="symbol", values="low").reindex_like(c)
    pc = df.pivot(index="date", columns="symbol", values="pre_close").reindex_like(c)
    amt = df.pivot(index="date", columns="symbol", values="amount").reindex_like(c)
    turn = df.pivot(index="date", columns="symbol", values="turnover_rate").reindex_like(c)
    syms, dates = c.columns, c.index

    ret = c / pc - 1
    g = h / pc - 1
    giveback = (h - c) / (h - pc)
    r20 = c.pct_change(20, fill_method=None)
    vol20 = ret.rolling(20).std()
    turn20 = turn.rolling(20).mean()
    pos250 = c / h.rolling(250, min_periods=60).max() - 1
    ratio = pd.Series([0.2 if s[:3] in ("300", "301", "688", "689") else 0.1 for s in syms], index=syms)
    touch = g.sub(ratio, axis=1) >= -0.004

    # 状态型复合分: 四特征日截面 pct-rank 均值 (全市场口径)
    comp = (vol20.rank(axis=1, pct=True) + turn20.rank(axis=1, pct=True)
            + r20.rank(axis=1, pct=True) + pos250.rank(axis=1, pct=True)) / 4
    fade_today = (g >= 0.03) & (giveback >= 0.7) & (~touch)

    # 执行日结果
    nxt = dates.to_series().shift(-1).reindex(dates)          # D = 下一交易日
    o2c = (c.shift(-1) / o.shift(-1) - 1)                      # D 开盘买到收盘
    c2c = ret.shift(-1)                                        # D c2c
    low_o = (l.shift(-1) / o.shift(-1) - 1)                    # D 日内最深 vs 开盘
    ret5 = c.shift(-4) / c.shift(-1) - 1                       # D 收盘 -> D+4 收盘
    fade_D = ((g >= 0.03) & (giveback >= 0.7) & (~touch)).shift(-1)
    gD = g.shift(-1)

    pos = {s: i for i, s in enumerate(syms)}
    recs = []
    for _, row in picks.iterrows():
        t, s = row["list_date"], row["symbol"]
        if s not in pos or t not in dates:
            continue
        ti = dates.get_loc(t)
        if ti + 1 >= len(dates):
            continue  # 无执行日数据
        j = pos[s]
        recs.append({
            "list_date": t, "symbol": s, "board": row["board"],
            "fade_score": comp.iloc[ti, j], "fade_today": bool(fade_today.iloc[ti, j]),
            "o2c": o2c.iloc[ti, j], "c2c": c2c.iloc[ti, j], "low_o": low_o.iloc[ti, j],
            "ret5": ret5.iloc[ti, j], "fade_D": bool(fade_D.iloc[ti, j]) if not np.isnan(gD.iloc[ti + 1, j]) else np.nan,
            "g_D": gD.iloc[ti, j],
        })
    R = pd.DataFrame(recs)
    print(f"[join] 匹配 {len(R)} 只票, 覆盖清单日 {R['list_date'].nunique()}")

    print("=" * 76)
    print("[1] TOP10 票的体质: fade_score 分布 vs 全市场")
    uni = comp.iloc[-60:].stack()  # 近60日全市场分布
    print(f"  全市场分位参考: p50={uni.quantile(.5):.2f} p70={uni.quantile(.7):.2f} p80={uni.quantile(.8):.2f}")
    print(f"  TOP10 票 fade_score: p25={R['fade_score'].quantile(.25):.2f} p50={R['fade_score'].quantile(.5):.2f} "
          f"p75={R['fade_score'].quantile(.75):.2f} | >=0.7 占比 {(R['fade_score'] >= 0.7).mean():.1%} "
          f"| >=0.8 占比 {(R['fade_score'] >= 0.8).mean():.1%}")
    print(f"  清单日当天已冲高回落(fade_today)的票占比: {R['fade_today'].mean():.1%} ({R['fade_today'].sum()} 只)")
    print(f"  执行日 D 实际冲高回落占比: {R['fade_D'].mean():.1%} (n={R['fade_D'].notna().sum()})")

    print("=" * 76)
    print("[2] 状态型: 按 fade_score 分组 -> 执行日/5日表现 (均值% / 中位% / 上涨率%)")
    R["grp"] = pd.cut(R["fade_score"], [0, 0.6, 0.75, 1.01], labels=["低<0.6", "中0.6-0.75", "高>=0.75"])
    for m in ["o2c", "c2c", "low_o", "ret5", "g_D"]:
        t2 = R.groupby("grp", observed=True)[m].agg(
            [("均值", "mean"), ("中位", "median"), ("上涨率", lambda x: (x > 0).mean()), ("n", "size")])
        print(f"  -- {m}:")
        print((t2[["均值", "中位", "上涨率"]] * [100, 100, 100]).round(2).assign(n=t2["n"]).to_string())

    print("=" * 76)
    print("[3] 事件型: 清单日当天已冲高回落 vs 未 -> 执行日/5日表现")
    for flag in [True, False]:
        sub = R[R["fade_today"] == flag]
        if not len(sub):
            continue
        line = f"  fade_today={flag} (n={len(sub)}): "
        for m in ["o2c", "c2c", "low_o", "ret5"]:
            s = sub[m].dropna()
            line += f"{m} {s.mean() * 100:+.2f}/{s.median() * 100:+.2f}%  "
        print(line)
    print(f"  执行日再回落率: fade_today=True -> {R.loc[R['fade_today'], 'fade_D'].mean():.1%} | "
          f"False -> {R.loc[~R['fade_today'], 'fade_D'].mean():.1%}")

    print("=" * 76)
    print("[4] 执行日实际冲高回落(fade_D)的票事后表现 (检验'回落=坏'假设)")
    for flag in [True, False]:
        sub = R[R["fade_D"] == flag]
        if not len(sub):
            continue
        line = f"  fade_D={flag} (n={len(sub)}): "
        for m in ["o2c", "ret5"]:
            s = sub[m].dropna()
            line += f"{m} {s.mean() * 100:+.2f}/{s.median() * 100:+.2f}%  "
        print(line)

    print("=" * 76)
    print("[5] 分日明细 sanity (近8个清单日: 高体质/当日回落票数)")
    R["hi"] = R["fade_score"] >= 0.75
    t5 = R.groupby("list_date").agg(票数=("symbol", "size"), 高体质=("hi", "sum"),
                                    当日回落=("fade_today", "sum"), 高体质且当日=("hi", lambda x: 0))
    t5 = t5.tail(8)
    print(t5.to_string())


if __name__ == "__main__":
    main()
