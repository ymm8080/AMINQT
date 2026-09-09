# -*- coding: utf-8 -*-
"""上证指数冲高回落规律归纳 — 日线口径.

度量定义:
  g        = high/pre_close - 1          日内最大冲高幅度
  giveback = (high-close)/(high-pre_close) 冲高幅度吐回比例 (high>pre_close 时有效)
  close_pos= (close-low)/(high-low)      收盘在日内区间位置
  gap      = open/pre_close - 1          开盘跳空
  ret      = close/pre_close - 1         日收益

对比组: 冲高守住日 (同 g 档位, giveback 低) vs 冲高回落日 (giveback 高)
"""
import os

import numpy as np
import pandas as pd

OUT = r"data/processed/sh_index_daily_20260909.parquet"
START, END = "20050101", "20260909"


def fetch_index() -> pd.DataFrame:
    if os.path.exists(OUT):
        return pd.read_parquet(OUT)
    import tushare as ts

    token = os.getenv("TUSHARE_TOKEN") or ts.get_token()
    ts.set_token(token)
    pro = ts.pro_api()
    df = pro.index_daily(
        ts_code="000001.SH",
        start_date=START,
        end_date=END,
        fields="ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
    )
    df = df.sort_values("trade_date").reset_index(drop=True)
    df.to_parquet(OUT)
    print(f"[fetch] 上证指数 {len(df)} 行 -> {OUT}")
    return df


def build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values("trade_date").reset_index(drop=True).copy()
    d["date"] = pd.to_datetime(d["trade_date"])
    pc = d["pre_close"]
    d["g"] = d["high"] / pc - 1
    d["gap"] = d["open"] / pc - 1
    d["ret"] = d["close"] / pc - 1
    rng = (d["high"] - d["low"]).replace(0, np.nan)
    d["close_pos"] = (d["close"] - d["low"]) / rng
    up = (d["high"] - pc).where(d["high"] > pc)
    d["giveback"] = (d["high"] - d["close"]) / up
    d["vratio"] = d["vol"] / d["vol"].rolling(20).mean().shift(1)
    d["r5"] = d["close"].pct_change(5)
    d["r20"] = d["close"].pct_change(20)
    d["ma20"] = d["close"].rolling(20).mean()
    d["above_ma20"] = d["close"] / d["ma20"] - 1
    d["vol20"] = d["ret"].rolling(20).std() * np.sqrt(252)
    # 分钟级当日高点出现时间无从得知, 日线仅刻画形态
    return d


def summarize(d: pd.DataFrame) -> None:
    print("=" * 72)
    print("[0] 基础分布 (2005 至今)")
    print(f"  交易日 {len(d)}, 日期 {d['date'].iloc[0]:%Y-%m-%d} ~ {d['date'].iloc[-1]:%Y-%m-%d}")
    qs = d["g"].quantile([0.25, 0.5, 0.75, 0.9, 0.95]).mul(100).round(2)
    print(f"  日内最大涨幅 g 分位(%)  p25={qs[0.25]} p50={qs[0.5]} p75={qs[0.75]} p90={qs[0.9]} p95={qs[0.95]}")
    sub = d[d["g"] > 0]
    gb = sub["giveback"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).mul(100).round(1)
    print(f"  冲高日(g>0)吐回比例 giveback 分位(%) p10={gb[0.1]} p25={gb[0.25]} p50={gb[0.5]} p75={gb[0.75]} p90={gb[0.9]}")

    # ── 档位定义: 三档冲高 × 吐回线 ──
    print("=" * 72)
    print("[1] 冲高回落日频率 (主口径: g>=0.4% & giveback>=60%)")
    d["fade"] = (d["g"] >= 0.004) & (d["giveback"] >= 0.60)
    d["fade_strong"] = (d["g"] >= 0.007) & (d["giveback"] >= 0.70)
    for col, name in [("fade", "普通档"), ("fade_strong", "强档")]:
        grp = d.groupby(d["date"].dt.year)[col].agg(["mean", "sum"])
        print(f"  -- {name} 按年: 频率% / 次数")
        print("     " + "  ".join(f"{y}:{v * 100:.0f}%({int(n)})" for y, v, n in
                                  zip(grp.index[-8:], grp["mean"].tail(8), grp["sum"].tail(8))))
    recent = d[d["date"] >= "2026-03-01"]
    print(f"  2026-03 以来: 普通档 {recent['fade'].mean() * 100:.1f}% (历史全期 {d['fade'].mean() * 100:.1f}%), "
          f"强档 {recent['fade_strong'].mean() * 100:.1f}% (全期 {d['fade_strong'].mean() * 100:.1f}%)")

    # ── 簇集性 ──
    print("=" * 72)
    print("[2] 簇集性: 冲高回落日是否连续出现")
    f = d["fade"].astype(int)
    p0 = f.mean()
    cond = f.groupby(f.shift(1)).mean()
    m2 = d["fade"].rolling(2).sum().shift(1) >= 2
    cond2 = d.loc[m2, "fade"].mean()
    print(f"  无条件频率 {p0 * 100:.1f}% | 前一日回落 -> 今日再回落 {cond.get(1, np.nan) * 100:.1f}% "
          f"| 前一日守住 -> {cond.get(0, np.nan) * 100:.1f}%")
    print(f"  前2日连续回落 -> 今日再回落 {cond2 * 100:.1f}% (n={m2.sum()})")

    # ── 前置条件: 同冲高档位内, 守住 vs 吐回 ──
    print("=" * 72)
    print("[3] 前置条件对比 (样本=冲高日 g>=0.4%, 分组: 守住(giveback<40%) vs 回落(>=60%))")
    pool = d[d["g"] >= 0.004].copy()
    pool["grp"] = np.where(pool["giveback"] >= 0.60, "回落", np.where(pool["giveback"] < 0.40, "守住", "中间"))
    feats = ["gap", "vratio", "r5", "r20", "above_ma20", "vol20", "close_pos"]
    rows = []
    for ftr in feats:
        for g_ in ["守住", "回落"]:
            s = pool.loc[pool["grp"] == g_, ftr].dropna()
            rows.append({"特征": ftr, "组": g_, "均值": s.mean(), "中位": s.median()})
    t = pd.DataFrame(rows).pivot(index="特征", columns="组", values="均值")
    t["中位_守住"] = pool[pool["grp"] == "守住"][feats].median()
    t["中位_回落"] = pool[pool["grp"] == "回落"][feats].median()
    print(t.round(4).to_string())
    print(f"  样本量: 守住 {(pool['grp'] == '守住').sum()}, 回落 {(pool['grp'] == '回落').sum()}, 中间 {(pool['grp'] == '中间').sum()}")

    # gap 细分
    print("-" * 72)
    print("[3b] 开盘方式细分 (冲高日池内)")
    pool["开盘"] = pd.cut(pool["gap"], [-np.inf, -0.002, 0.002, 0.006, np.inf],
                          labels=["低开<-0.2%", "平开", "高开0.2-0.6%", "大幅高开>0.6%"])
    ct = pool.groupby("开盘", observed=True).apply(
        lambda x: pd.Series({
            "占比%": len(x) / len(pool) * 100,
            "回落率%": (x["giveback"] >= 0.6).mean() * 100,
            "日收益%": x["ret"].mean() * 100,
        }))
    print(ct.round(2).to_string())

    # ── 后果: 次日及5日表现 ──
    print("=" * 72)
    print("[4] 后果: 冲高回落日的次日/5日表现 (对比守住日)")
    d["ret1"] = d["ret"].shift(-1)
    d["ret5"] = d["close"].shift(-5) / d["close"] - 1
    pool2 = d[d["g"] >= 0.004].copy()
    pool2["grp"] = np.where(pool2["giveback"] >= 0.60, "回落", np.where(pool2["giveback"] < 0.40, "守住", "中间"))
    for h in ["ret1", "ret5"]:
        s = pool2.groupby("grp")[h].agg(["mean", "median", lambda x: (x > 0).mean()])
        s.columns = ["均值%", "中位%", "上涨率%"]
        print(f"  -- {h}:")
        print((s * [100, 100, 100]).round(2).to_string())

    # ── 全市场视角: 回落日 vs 全部交易日 ──
    print("=" * 72)
    print("[5] 回落日的次日表现 vs 全部交易日")
    base1 = d["ret1"].mean() * 100
    fade1 = d.loc[d["fade"], "ret1"].mean() * 100
    fade1_up = (d.loc[d["fade"], "ret1"] > 0).mean() * 100
    base_up = (d["ret1"] > 0).mean() * 100
    print(f"  次日均值: 全部 {base1:.2f}% vs 回落日 {fade1:.2f}% | 次日上涨率: {base_up:.1f}% vs {fade1_up:.1f}%")
    # 连续回落后的反弹概率
    d["f2"] = d["fade"].rolling(2).sum()
    for k in [1, 2]:
        m = d["f2"].shift(1) >= k
        if m.sum() > 30:
            print(f"  前{k if k > 1 else 1}日内出现>=2次回落 -> 次日均值 {d.loc[m, 'ret1'].mean() * 100:.2f}% "
                  f"上涨率 {(d.loc[m, 'ret1'] > 0).mean() * 100:.1f}% (n={m.sum()})")

    # ── 近期明细 ──
    print("=" * 72)
    print("[6] 2026-06 以来冲高回落日明细 (强档优先)")
    rec = d[(d["date"] >= "2026-06-01") & d["fade"]].copy()
    rec = rec.sort_values("date", ascending=False).head(30)
    cols = ["trade_date", "g", "gap", "ret", "giveback", "close_pos", "vratio", "r5"]
    show = rec[cols].copy()
    for c in ["g", "gap", "ret", "r5"]:
        show[c] = (show[c] * 100).round(2)
    show["giveback"] = (show["giveback"] * 100).round(0)
    show["close_pos"] = (show["close_pos"] * 100).round(0)
    print(show.to_string(index=False))


if __name__ == "__main__":
    df = fetch_index()
    d = build_metrics(df)
    summarize(d)
