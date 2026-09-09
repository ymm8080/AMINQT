# -*- coding: utf-8 -*-
"""误杀检验: fade_score 删线是否在大砍强势股. 回放同源数据, 切法:
  [1] 删除区 (>=0.75) 结果分布: 赢家多少/输家多少 (均值被谁拖动)
  [2] 大赢家泄漏: ret5>=5% / >=10% 的票里多少 fade_score>=0.75
  [3] 阈值扫描 0.75/0.80/0.85/0.90/0.95: 上侧组 o2c/ret5 + 删掉的大赢家数
  [4] 半窗稳定性: 删除区差表现是否集中于某一时间段
  [5] 用户点名3只 (300857/601869/603186) 的历史入选与结果
"""
import os
import re

import numpy as np
import pandas as pd

LISTDIR = r"D:/AMINQT/DAILY OPERATION/STOCK LIST"
PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"


def load_lists():
    rows = []
    for f in sorted(os.listdir(LISTDIR)):
        m = re.match(r"^legacy_stocklist_(\d{8})__.*\.csv$", f)
        if m:
            rows.append((m.group(1), f))
    best = {}
    for d, f in rows:
        best[d] = f
    out = []
    for d in sorted(best):
        df = pd.read_csv(os.path.join(LISTDIR, best[d]), dtype={"symbol": str})
        df["list_date"] = pd.to_datetime(d)
        out.append(df[["symbol", "board", "list_date"]])
    L = pd.concat(out, ignore_index=True).drop_duplicates(subset=["list_date", "symbol"])
    print(f"[lists] {L['list_date'].nunique()} 清单日 {len(L)} 票")
    return L


def main():
    picks = load_lists()
    df = pd.read_parquet(PANEL, columns=[
        "symbol", "date", "open", "high", "low", "close", "pre_close", "turnover_rate"])
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)
    c = df.pivot(index="date", columns="symbol", values="close").sort_index()
    o = df.pivot(index="date", columns="symbol", values="open").reindex_like(c)
    h = df.pivot(index="date", columns="symbol", values="high").reindex_like(c)
    pc = df.pivot(index="date", columns="symbol", values="pre_close").reindex_like(c)
    turn = df.pivot(index="date", columns="symbol", values="turnover_rate").reindex_like(c)
    syms, dates = c.columns, c.index

    ret = c / pc - 1
    r20 = c.pct_change(20, fill_method=None)
    vol20 = ret.rolling(20).std()
    turn20 = turn.rolling(20).mean()
    pos250 = c / h.rolling(250, min_periods=60).max() - 1
    comp = (vol20.rank(axis=1, pct=True) + turn20.rank(axis=1, pct=True)
            + r20.rank(axis=1, pct=True) + pos250.rank(axis=1, pct=True)) / 4

    o2c = c.shift(-1) / o.shift(-1) - 1
    ret5 = c.shift(-4) / c.shift(-1) - 1

    pos = {s: i for i, s in enumerate(syms)}
    recs = []
    for _, row in picks.iterrows():
        t, s = row["list_date"], row["symbol"]
        if s not in pos or t not in dates:
            continue
        ti = dates.get_loc(t)
        if ti + 1 >= len(dates):
            continue
        j = pos[s]
        recs.append({
            "list_date": t, "symbol": s, "fade_score": comp.iloc[ti, j],
            "o2c": o2c.iloc[ti, j], "ret5": ret5.iloc[ti, j],
        })
    R = pd.DataFrame(recs).dropna(subset=["fade_score"])
    print(f"[join] 匹配 {len(R)} 票 / {R['list_date'].nunique()} 日\n")

    K = R[R["fade_score"] >= 0.75]
    print("=" * 70)
    print(f"[1] 删除区 n={len(K)} 结果分布:")
    for m in ["o2c", "ret5"]:
        s = K[m].dropna()
        print(f"  {m}: 均值{s.mean()*100:+.2f}% 中位{s.median()*100:+.2f}% "
              f"上涨率{(s>0).mean():.1%} 亏>3%占{(s<-0.03).mean():.1%} 赚>3%占{(s>0.03).mean():.1%}")
    print(f"  ret5>=+5% (大赢家): {(K['ret5']>=0.05).sum()}/{len(K)} = {(K['ret5']>=0.05).mean():.1%}")
    print(f"  ret5<=-5% (大输家): {(K['ret5']<=-0.05).sum()}/{len(K)} = {(K['ret5']<=-0.05).mean():.1%}")

    print("=" * 70)
    print("[2] 大赢家泄漏 (被删线抓走的大赢家比例):")
    for thr_w in [0.05, 0.10]:
        W = R[R["ret5"] >= thr_w]
        print(f"  ret5>={thr_w:.0%} (n={len(W)}): fade>=0.75 占 "
              f"{(W['fade_score']>=0.75).mean():.1%} ({(W['fade_score']>=0.75).sum()}只), "
              f"fade>=0.80 占 {(W['fade_score']>=0.8).mean():.1%}")

    print("=" * 70)
    print("[3] 阈值扫描 (上侧组=会被删的):")
    for thr in [0.75, 0.80, 0.85, 0.90, 0.95]:
        up = R[R["fade_score"] >= thr]
        dn = R[R["fade_score"] < thr]
        if not len(up):
            print(f"  thr={thr}: n=0")
            continue
        nw = int((up["ret5"] >= 0.05).sum())
        print(f"  thr={thr}: 删{len(up)}只 o2c{up['o2c'].mean()*100:+.2f}% "
              f"ret5{up['ret5'].mean()*100:+.2f}% | 留存ret5{dn['ret5'].mean()*100:+.2f}% "
              f"| 删掉的大赢家(ret5>=5%) {nw}只")

    print("=" * 70)
    print("[4] 半窗稳定性 (删除区差表现是否集中某时段):")
    R2 = R.sort_values("list_date")
    half = len(R2) // 2
    for name, sub in [("前半", R2.iloc[:half]), ("后半", R2.iloc[half:])]:
        k = sub[sub["fade_score"] >= 0.75]
        r = sub[sub["fade_score"] < 0.75]
        if not len(k):
            print(f"  {name}: 删除区 0 只")
            continue
        print(f"  {name}: 删除区 {len(k)}只 o2c{k['o2c'].mean()*100:+.2f}% "
              f"ret5{k['ret5'].mean()*100:+.2f}% vs 留存 ret5{r['ret5'].mean()*100:+.2f}% "
              f"(日: {k['list_date'].min():%m-%d}..{k['list_date'].max():%m-%d})")

    print("=" * 70)
    print("[5] 点名3只历史入选记录:")
    for s in ["300857", "601869", "603186"]:
        sub = R[R["symbol"] == s]
        if not len(sub):
            print(f"  {s}: 无历史入选(回放窗)")
            continue
        for _, r in sub.iterrows():
            print(f"  {s} @{r['list_date']:%m-%d}: fade={r['fade_score']:.2f} "
                  f"o2c{r['o2c']*100:+.2f}% ret5{r['ret5']*100:+.2f}%")


if __name__ == "__main__":
    main()
