# -*- coding: utf-8 -*-
"""SLOW BULL 长持影子单 (2026-09-07 用户命名+拍板接线; 承接旧 SLOW_BULL_PAUSE 暂停模块
的长持产出位, 不推送).

规格 (SLOW_BULL): band[90,99.5)×grind(20日无单日≥8%)×pull10≥-10%×wr5闸(fail-open)
× 宽度闸(全市场20日动量>0占比 > 其60日滚动均值, 自适应基准)
× 格内mom分位(60%,90%]区 × 低波半格(vol20格内分位≤50%) — v4 极致确定性
(2026-09-07 用户"我要极致确定性"+"低波半"; 5.3只/日 赢59% 中位+4.1% 大亏7%)
× trail8/40日退出 (T+1收盘买).

用法: python _slowbull_list.py [--date YYYY-MM-DD]  (默认=面板最新日)
输出: DATA OTHERS/shadow/slowbull_list_<D>__v4.csv  (WORM; 沿革 grind_shadow_list_*:
v1=无半格88只, v2=高波半44只, v3=mom(70,90]区12只 — 高波/纯mom区规格已被 v4 替代)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import DATA_DIR, DATA_OTHERS_DIR, SLOW_BULL, PANEL_V3_PATH

CFG = SLOW_BULL
OUT_DIR = DATA_OTHERS_DIR / CFG["out_root"]
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2024-06-01")  # breadth MA60 预热 + 20日窗余量

ap = argparse.ArgumentParser()
ap.add_argument("--date", default=None, help="信号日 YYYY-MM-DD (默认面板最新日)")
args = ap.parse_args()

df = pd.read_parquet(
    PANEL_V3_PATH,
    columns=["symbol", "date", "close_hfq"],
    filters=[("date", ">=", START)],
)
df["symbol"] = (
    df["symbol"].astype(str).str.replace(r"\..*", "", regex=True).str.zfill(6)
)
df = df[~df["symbol"].str.startswith(("43", "83", "87", "92"))]
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

g = df.groupby("symbol")["close_hfq"]
df["ret1"] = g.pct_change()
df["mom"] = g.pct_change(20)
df["max1d20"] = df.groupby("symbol")["ret1"].rolling(20).max().reset_index(level=0, drop=True)
df["vol20"] = df.groupby("symbol")["ret1"].rolling(20).std().reset_index(level=0, drop=True)
df["pull10"] = df["close_hfq"] / df.groupby("symbol")["close_hfq"].transform(lambda s: s.rolling(10).max()) - 1
df["mom__r"] = df.groupby("date")["mom"].rank(pct=True)

# 宽度闸 (自适应基准 = 60日滚动均值)
breadth = (df["mom"] > 0).groupby(df["date"]).mean().astype(float)
b_ma = breadth.rolling(CFG["breadth_ma"], min_periods=CFG["breadth_ma"] // 2).mean()

if args.date:
    sig_d = pd.Timestamp(args.date)
else:
    sig_d = df["date"].max()
if sig_d not in set(df["date"].unique()):
    sys.exit(f"面板无 {sig_d.date()} 数据")

b_d = breadth.get(sig_d, np.nan)
b_ma_d = b_ma.get(sig_d, np.nan)
gate_breadth = bool(b_d > b_ma_d) if np.isfinite(b_d) and np.isfinite(b_ma_d) else False

# wr5 筹码闸 (fail-open: cyq 缺失不拦)
cq = pd.read_parquet(os.path.join(DATA_DIR, "cyq_panel.parquet"), columns=["symbol", "date", "winner_ratio"])
cq["symbol"] = cq["symbol"].astype(str).str.replace(r"\..*", "", regex=True).str.zfill(6)
cq = cq.sort_values(["symbol", "date"])
cq["wr5"] = cq.groupby("symbol")["winner_ratio"].diff(5)
wr_d = cq[cq["date"] <= sig_d].groupby("symbol").tail(1).set_index("symbol")["wr5"]

d = df[df["date"] == sig_d].set_index("symbol")
d["wr5"] = wr_d.reindex(d.index)
sel = d[
    (d["mom__r"] >= CFG["band_lo"])
    & (d["mom__r"] < CFG["band_hi"])
    & (d["max1d20"] < CFG["grind_max1d"])
    & (d["pull10"] >= CFG["pull_min"])
    & ~(d["wr5"] < 0)
]
if not gate_breadth:
    sel = sel.iloc[0:0]
if len(sel):
    # v4 格内选股 (两秩均在全闸格内算, 与回测口径一致, 自适应分位无固定值):
    # mom分位 (60,90] 确定性中段峰 × 低波半 (分位≤50%)
    zone = sel["mom__r"].rank(pct=True, method="first")
    vr = sel["vol20"].rank(pct=True, method="first")
    sel = sel[(zone > CFG["mom_lo"]) & (zone <= CFG["mom_hi"])]
    if CFG.get("vol_half") == "low":
        sel = sel[vr.reindex(sel.index) <= 0.5]

out = sel.reset_index()[["symbol", "mom__r", "max1d20", "pull10", "vol20", "wr5"]].sort_values("mom__r", ascending=False)
out["mom__r"] = out["mom__r"].round(4)
out["max1d20"] = out["max1d20"].round(4)
out["pull10"] = out["pull10"].round(4)
out["vol20"] = out["vol20"].round(4)
out["wr5"] = out["wr5"].round(4)

path = OUT_DIR / f"slowbull_list_{sig_d.date()}__v4.csv"
out.to_csv(path, index=False, encoding="utf-8-sig")

state = "开闸" if gate_breadth else "关闸(宽度低于60日均值)"
print(f"[slowbull] {sig_d.date()} 宽度闸:{state} (breadth={b_d:.3f} vs MA60={b_ma_d:.3f})")
print(f"[slowbull] 候选 {len(out)} 只 → {path}")
if len(out):
    print(out.to_string(index=False))
print("[slowbull] 规则: T+1收盘买入, trail 8% / 持有上限40交易日, 影子单不推送")
