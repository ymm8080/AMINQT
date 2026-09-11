# -*- coding: utf-8 -*-
"""0910 THS看涨候选规则全池回测 (逆向第二步: 共识→规则阶梯)
共识8案: x_ma10>0 与 r10>0 = 8/8同侧; 阳线7/8; 量比>1为5/8(分位6/8); KDJ金叉3日内5/8; MACD金叉仅3/8.
规则阶梯 (T日收盘可算): 核心松→紧, 报 FWD3/5, 日均触发, 8案召回.
近1年池(额>=5e7), 脏pctChg整窗剔除, FWD右删失(有效至0902). WORM.
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
PNL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
OUT = r"D:/AMINQT/Daily Operation/STOCK LIST/_ths_signal_rule_20260910.txt"
pd.set_option("display.width", 220)
L = []
def P(s=""):
    print(s); L.append(str(s))

CASES = [("002377", "2026-09-08"), ("002204", "2026-09-08"), ("002790", "2026-09-08"),
         ("301176", "2026-09-07"), ("000565", "2026-09-07"), ("002201", "2026-09-07"),
         ("600359", "2026-09-04"), ("000993", "2026-09-01")]

cols = ["symbol", "date", "close", "pctChg", "amount", "volume"]
df = pd.read_parquet(PNL, columns=cols)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
g = df.groupby("symbol")
for k in (3, 5):
    mat = pd.concat([g["pctChg"].shift(-i) for i in range(1, k + 1)], axis=1)
    valid = mat.notna().all(axis=1) & mat.abs().le(22).all(axis=1)
    df[f"f{k}"] = ((1 + mat / 100).prod(axis=1) - 1).where(valid)
df["ma10"] = g["close"].transform(lambda s: s.rolling(10).mean())
df["x_ma10"] = df["close"] / df["ma10"] - 1
df["r10"] = g["close"].transform(lambda s: s.pct_change(10))
df["vr"] = df["volume"] / g["volume"].transform(lambda s: s.rolling(5).mean().shift(1))
l9 = g["low"].transform(lambda s: s.rolling(9).min()) if "low" in df else None
df["is_lim"] = (df["pctChg"] >= 9.5)

# KDJ (需high/low, 单独补读)
if "low" not in df.columns:
    hl = pd.read_parquet(PNL, columns=["symbol", "date", "high", "low"])
    hl["date"] = pd.to_datetime(hl["date"])
    df = df.merge(hl, on=["symbol", "date"], how="left")
    g = df.groupby("symbol")
l9 = g["low"].transform(lambda s: s.rolling(9).min())
h9 = g["high"].transform(lambda s: s.rolling(9).max())
df["rsv"] = (df["close"] - l9) / (h9 - l9 + 1e-9) * 100
df["kv"] = g["rsv"].transform(lambda s: s.ewm(com=2, adjust=False).mean())
df["dv"] = g["kv"].transform(lambda s: s.ewm(com=2, adjust=False).mean())
kcut = (g["kv"].shift(1) <= g["dv"].shift(1)).fillna(False)
df["kd_x3"] = ((df["kv"] > df["dv"]) & kcut).astype(float)
df["kd_x3d"] = ((df["kd_x3"] > 0) |
                (g["kd_x3"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).max()) > 0))

d = df[(df["date"] >= df["date"].max() - pd.Timedelta(days=365)) &
       (df["amount"] >= 5e7)].dropna(subset=["f3", "f5"]).copy()
P(f"窗: {d['date'].min():%Y-%m-%d}..{d['date'].max():%Y-%m-%d}, 股日={len(d):,} (~{d['date'].nunique()}日)")
base3, base5 = d["f3"].mean(), d["f5"].mean()
P(f"池基线: FWD3={base3:+.4f} FWD5={base5:+.4f} hit5={(d['f3']>=0.05).mean():.1%} hit10={(d['f5']>=0.10).mean():.1%}")
ndays = d["date"].nunique()

# 8案召回: 信号日当日 或 前1交易日
dcase = df[df["amount"] >= 5e7][["symbol", "date", "pctChg", "x_ma10", "r10", "vr", "kd_x3d", "is_lim"]].copy()
def recall(mask_fn, lag=0):
    hits = 0
    for sym, ds in CASES:
        dt = pd.Timestamp(ds)
        dates = [dt - pd.Timedelta(days=0)] if lag == 0 else [dt - pd.Timedelta(days=3), dt]
        sub = dcase[(dcase["symbol"] == sym) & (dcase["date"].isin(
            pd.bdate_range(dates[0], dates[-1])))]
        if len(sub) and mask_fn(sub).any():
            hits += 1
    return hits

RULES = {
    "A 核心: x_ma10>0 & r10>0": lambda t: (t["x_ma10"] > 0) & (t["r10"] > 0),
    "B A+当日阳线": lambda t: (t["x_ma10"] > 0) & (t["r10"] > 0) & (t["pctChg"] > 0),
    "C B+量比>1": lambda t: (t["x_ma10"] > 0) & (t["r10"] > 0) & (t["pctChg"] > 0) & (t["vr"] > 1),
    "D B+KDJ金叉3日内": lambda t: (t["x_ma10"] > 0) & (t["r10"] > 0) & (t["pctChg"] > 0) & (t["kd_x3d"] == 1),
    "E C+KDJ金叉(最紧)": lambda t: (t["x_ma10"] > 0) & (t["r10"] > 0) & (t["pctChg"] > 0) & (t["vr"] > 1) & (t["kd_x3d"] == 1),
    "F A+MACD金叉(对照)": None,  # 跳过MACD(需dif/dea, 共识已排除)
    "G C & 信号日非涨停(可操作)": lambda t: (t["x_ma10"] > 0) & (t["r10"] > 0) & (t["pctChg"] > 0) & (t["vr"] > 1) & (~t["is_lim"]),
}
rows = []
for name, fn in RULES.items():
    if fn is None:
        continue
    m = fn(d)
    s = d[m]
    if not len(s):
        continue
    rows.append({"规则": name, "n": len(s), "日均触发": len(s) / ndays,
                 "FWD3均": s["f3"].mean(), "FWD3中位": s["f3"].median(),
                 "hit5%": (s["f3"] >= 0.05).mean(),
                 "FWD5均": s["f5"].mean(), "hit10%": (s["f5"] >= 0.10).mean(),
                 "超额FWD5": s["f5"].mean() - base5,
                 "超额FWD3": s["f3"].mean() - base3,
                 "8案召回(日)": recall(fn), "8案召回(含前1日)": recall(fn, lag=1)})
P("\n" + pd.DataFrame(rows).round(4).to_string(index=False))

# ---- D规则 × 额带 × 近10日板 (用户的带 + 簇票两态) ----
d["band"] = np.where(d["amount"] >= 8e7, ">=80M", "50-80M")
d["lim10"] = d.groupby("symbol")["is_lim"].transform(lambda s: s.astype(float).rolling(10).sum())
d["近10日有板"] = np.where(d["lim10"] > 0, "有板", "无板")
mD = (d["x_ma10"] > 0) & (d["r10"] > 0) & (d["pctChg"] > 0) & (d["kd_x3d"] == 1)
sD = d[mD]
P("\n=== D规则(站上MA10+r10>0+阳线+KDJ金叉) × 额带 × 板位 ===")
rows = []
for (b, pb), sub in sD.groupby(["band", "近10日有板"]):
    ctx = d[(d["band"] == b) & (d["近10日有板"] == pb)]
    rows.append({"band": b, "板位": pb, "n": len(sub), "日均": len(sub) / ndays,
                 "FWD3均": sub["f3"].mean(), "FWD5均": sub["f5"].mean(),
                 "hit10%": (sub["f5"] >= 0.10).mean(),
                 "同语境基线FWD5": ctx["f5"].mean(), "基线hit10%": (ctx["f5"] >= 0.10).mean(),
                 "超额FWD5": sub["f5"].mean() - ctx["f5"].mean()})
P(pd.DataFrame(rows).round(4).to_string(index=False))

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))
print(f"\nsaved -> {OUT}")
