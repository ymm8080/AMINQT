# -*- coding: utf-8 -*-
"""THS 看涨/看跌回填信号信息量检查 — 事件研究 + TS IC (轻量, 只读).

背景: 生产面板 panel_full_enriched_v3.parquet 已并入 7 个 ths 信号列
(回填 0910)。ths_bull=1 正样本仅 2023-09-01~2024-08-12 (1649 股票日);
ths_bear_pool = 市场级每日计数 (广播列, 非个股信号); 看跌个股级回填未产出
(不等, 用户指令)。ths 11 特征已走 force_include 在池 — 本脚本验证信息量。

口径: 特征 t 日 vs 前瞻 10 交易日收益 fwd10 = close_hfq[t+10]/close_hfq[t]-1
(仅筛选用途, 非生产标签)。事件研究: ths_bull=1 日 vs 全样本基线的 fwd10 对比,
分年稳性; TS IC (个股时序 spearman); ths_bear_pool 为市场级 → 对全市场等权
日收益做时序相关 (它不是个股截面特征, IC 口径不适用)。
输出 WORM: diag/ths_signal_ic_{ts}.json
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import data_others_path  # noqa: E402

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
COLS = [
    "symbol", "date", "close_hfq",
    "ths_bull", "ths_bull_buy_sig_n", "ths_bull_tech_n", "ths_bear_pool",
]


def main() -> int:
    t0 = time.time()
    import pyarrow.parquet as pq

    names = set(pq.ParquetFile(PANEL).schema_arrow.names)
    use = [c for c in COLS if c in names]
    print(f"[load] {PANEL} cols={use}", flush=True)
    df = pd.read_parquet(PANEL, columns=use)
    df["symbol"] = df["symbol"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("symbol", sort=False)
    df["fwd10"] = g["close_hfq"].shift(-10) / df["close_hfq"] - 1.0

    rep = {"panel_rows": int(len(df)), "panel_span": [str(df['date'].min().date()), str(df['date'].max().date())]}

    # ── ths_bull 事件研究 ──
    ev = df[(df["ths_bull"] == 1.0) & df["fwd10"].notna()]
    base = df[df["fwd10"].notna()]
    rep["ths_bull_events"] = int(len(ev))
    rep["ths_bull_event_span"] = [str(ev["date"].min().date()), str(ev["date"].max().date())] if len(ev) else None
    rep["fwd10_mean_all"] = float(base["fwd10"].mean())
    rep["fwd10_mean_event"] = float(ev["fwd10"].mean()) if len(ev) else None
    rep["fwd10_median_event"] = float(ev["fwd10"].median()) if len(ev) else None
    rep["fwd10_win_rate_event"] = float((ev["fwd10"] > 0).mean()) if len(ev) else None
    rep["fwd10_win_rate_all"] = float((base["fwd10"] > 0).mean())
    by_year = {}
    for y, grp in ev.groupby(ev["date"].dt.year):
        by_year[int(y)] = {
            "n": int(len(grp)),
            "mean_fwd10": float(grp["fwd10"].mean()),
            "win_rate": float((grp["fwd10"] > 0).mean()),
        }
    rep["event_by_year"] = by_year
    # 同期基线 (事件窗口内) — 控制行情
    if len(ev):
        lo, hi = ev["date"].min(), ev["date"].max()
        base_win = base[(base["date"] >= lo) & (base["date"] <= hi)]
        rep["fwd10_mean_all_in_event_window"] = float(base_win["fwd10"].mean())
        rep["event_lift_pp"] = float((ev["fwd10"].mean() - base_win["fwd10"].mean()) * 100)

    # 事件日截面排名: 事件股当日 fwd10 在全市场截面中的分位 (1=最强)
    if len(ev):
        df["cs_rank"] = df.groupby("date")["fwd10"].rank(pct=True)
        rep["event_cs_rank_pctile_mean"] = float(df.loc[ev.index, "cs_rank"].mean())

    # TS IC (稀疏 0/1 旗标 IC 仅参考)
    m = df["ths_bull"].notna() & df["fwd10"].notna()
    sub = df[m]
    ics = []
    for _sym, grp in sub.groupby("symbol", sort=False):
        if len(grp) >= 100 and grp["ths_bull"].std() > 0:
            r = grp["ths_bull"].corr(grp["fwd10"], method="spearman")
            if np.isfinite(r):
                ics.append(r)
    rep["ths_bull_ts_ic_mean"] = float(np.mean(ics)) if ics else None
    rep["ths_bull_ts_ic_n"] = len(ics)

    # ── ths_bull_buy_sig_n / ths_bull_tech_n 分桶 ──
    for col in ("ths_bull_buy_sig_n", "ths_bull_tech_n"):
        bk = {}
        for v, grp in df[df[col].notna() & df["fwd10"].notna()].groupby(col):
            bk[str(v)] = {"n": int(len(grp)), "mean_fwd10": float(grp["fwd10"].mean())}
        rep[f"{col}_buckets"] = bk

    # ── ths_bear_pool (市场级): 与全市场等权 fwd10 均值的时序相关 ──
    mkt = df.groupby("date").agg(bear=("ths_bear_pool", "first"), mkt_fwd=("fwd10", "mean"))
    mkt = mkt.dropna()
    if len(mkt) > 60:
        rep["ths_bear_pool_corr_mktfwd10"] = float(mkt["bear"].corr(mkt["mkt_fwd"], method="spearman"))
        rep["ths_bear_pool_n_days"] = int(len(mkt))
        rep["ths_bear_pool_span"] = [str(mkt.index.min().date()), str(mkt.index.max().date())]

    # 688228 个案 (应在所有 ths 列上全 0/NaN)
    case = df[(df["symbol"] == "688228") & (df["date"] == pd.Timestamp("2026-09-09"))]
    rep["case_688228_row"] = {c: (None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.floating)) else str(v)))
                              for c, v in case.iloc[0].items()} if len(case) else None

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = data_others_path("diag") / f"ths_signal_ic_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k != "case_688228_row"},
                     ensure_ascii=False, indent=2, default=str), flush=True)
    print(f"[worm] {out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
