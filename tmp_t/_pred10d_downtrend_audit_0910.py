# -*- coding: utf-8 -*-
"""tmp research: 10d 头在下跌趋势股上的校准审计 (2026-09-10, 688228 案).

问题: legacy 10d 回归头把 4 连阴的 688228 预测成 +5.27%/10d (TOP8).
本审计问两件事 (candidates 全样本 + 交付清单两层):
  A. 排名层: 按 pred_ret_10d 排名键的头部票, 下跌趋势 (ret5<0) vs 其他 — 真实 o2c/ret5。
  B. 校准层: pred10d 分桶 × 下跌趋势 — 模型对下跌股是否系统性高估 (校准缺口)。
只读研究。面板只读 needed 列, 内存占用与 _gate_admission 探针同级 (重训进行中, 保持轻量)。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import glob

import pandas as pd

from config.settings import PANEL_V3_PATH

D0 = "2026-06-15"  # 08-03 前留 ~30 交易日历史 (ret5/连跌/ma5 用)


def load_panel():
    df = pd.read_parquet(
        PANEL_V3_PATH,
        columns=["symbol", "date", "open", "close", "pre_close"],
        filters=[("date", ">=", pd.Timestamp(D0))],
    )
    df["symbol"] = df["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
    df = df.drop_duplicates(["symbol", "date"])
    c = df.pivot(index="date", columns="symbol", values="close").sort_index()
    o = df.pivot(index="date", columns="symbol", values="open").reindex_like(c)
    pc = df.pivot(index="date", columns="symbol", values="pre_close").reindex_like(c)
    return c, o, pc


def load_candidates():
    frames = []
    for f in glob.glob("data/lists/candidates_2026*.parquet"):
        d = pd.read_parquet(f)
        if "pred_ret_10d" not in d.columns:
            continue
        if "date" not in d.columns:
            m = glob.glob and pd.Timestamp(
                f.split("candidates_")[-1][:8].replace("_", "")
            )
            d["date"] = pd.Timestamp(m)
        d = d[["symbol", "board", "date", "pred_ret_10d"]]
        d["list_date"] = pd.to_datetime(d["date"])
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def main():
    c, o, pc = load_panel()
    dates = c.index
    pos = {s: j for j, s in enumerate(c.columns)}
    r5 = c.pct_change(5, fill_method=None)  # ≤t 可知
    o2c = c.shift(-1) / o.shift(-1) - 1
    fwd5 = c.shift(-5) / c.shift(-1) - 1
    fwd10 = c.shift(-10) / c.shift(-1) - 1  # 与 label_10d 同视界 (毛口径, net≈-1.3%)

    cand = load_candidates()
    cand["symbol"] = cand["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
    rows = []
    for t, s, p10 in zip(
        cand["list_date"], cand["symbol"], cand["pred_ret_10d"]
    ):
        if s not in pos or t not in dates:
            continue
        ti = dates.get_loc(t)
        if ti + 10 >= len(dates):
            continue
        j = pos[s]
        rows.append(
            (t, s, p10, float(r5.iloc[ti, j]), float(o2c.iloc[ti, j]), float(fwd5.iloc[ti, j]), float(fwd10.iloc[ti, j]))
        )
    df = pd.DataFrame(
        rows, columns=["t", "s", "pred10", "r5", "o2c", "fwd5", "fwd10"]
    )
    df = df.dropna(subset=["pred10", "r5", "o2c"])
    # 每日按 pred10 排名键取头部 (双板合池近似交付头部池)
    df["rank"] = df.groupby("t")["pred10"].rank(ascending=False, method="first")
    df["down"] = df["r5"] < 0

    print(f"[cand] {df['t'].nunique()} days, {len(df)} scored rows (matured)")
    for pool, sub in (
        ("rank≤10 (排名键头部)", df[df["rank"] <= 10]),
        ("rank 11-30", df[(df["rank"] > 10) & (df["rank"] <= 30)]),
        ("rank>30 (池身位", df[df["rank"] > 30]),
    ):
        dn, up = sub[sub["down"]], sub[~sub["down"]]
        print(
            f"\n== {pool}: n={len(sub)} 下跌股 {len(dn)} ({len(dn)/max(len(sub),1):.0%})"
        )
        for name, g in (("下跌趋势", dn), ("非下跌", up)):
            if not len(g):
                continue
            print(
                f"  {name}: o2c {g['o2c'].mean():+.3%} | fwd5 {g['fwd5'].mean():+.3%} | "
                f"fwd5>0 {(g['fwd5']>0).mean():.1%} | fwd10 {g['fwd10'].mean():+.3%} (win {(g['fwd10']>0).mean():.0%}) | n={len(g)}"
            )

    # 校准层: pred10 分桶 × 下跌 — 高桶里模型承诺 vs 实得
    print("\n== 校准缺口 (全头部池 rank≤30): 承诺 pred10 vs 实得 fwd5")
    head = df[df["rank"] <= 30].copy()
    head["bucket"] = pd.cut(head["pred10"], [0, 0.02, 0.04, 0.06, 1.0])
    for (bkt, dn), g in head.groupby(["bucket", "down"], observed=True):
        gap = (g["fwd10"] - g["pred10"]).mean()
        print(
            f"  pred10∈{str(bkt):18s} {'下跌' if dn else '非下跌'}: n={len(g):4d} "
            f"承诺 {g['pred10'].mean():+.2%} 实得 {g['fwd10'].mean():+.2%} 缺口 {gap:+.2%}"
        )


if __name__ == "__main__":
    main()
