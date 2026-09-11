# -*- coding: utf-8 -*-
"""0910 逆向THS看涨信号 (用户: 这些股票日都是同花顺标注看涨, 构建同样信号)
步骤:
 1. 案例表: 8个确定(symbol,信号日)对, 算~20个T日收盘可算技术指标, 转全池当日截面分位
 2. 共识: 各指标案例中位分位, 找一致极端(全体或9/10同侧)的条件
 3. 候选规则全池回测: FWD3/FWD5 vs 池基线 + 日均触发数 + 8案召回(信号日或前3日内触发)
指标全T日收盘可算, 无未来. 近1年池(额>=5e7). 脏pctChg整窗剔除. WORM.
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
PNL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
OUT = r"D:/AMINQT/Daily Operation/STOCK LIST/_ths_signal_reverse_20260910.txt"
pd.set_option("display.width", 250)
L = []
def P(s=""):
    print(s); L.append(str(s))

# 确定案例 (用户报的信号日; 600318/600876日期未指明→只做旁观)
CASES = [("002377", "2026-09-08"), ("002204", "2026-09-08"), ("002790", "2026-09-08"),
         ("301176", "2026-09-07"), ("000565", "2026-09-07"), ("002201", "2026-09-07"),
         ("600359", "2026-09-04"), ("000993", "2026-09-01")]

cols = ["symbol", "date", "open", "high", "low", "close", "pctChg", "amount", "volume"]
df = pd.read_parquet(PNL, columns=cols)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
g = df.groupby("symbol")
for k in (3, 5):
    mat = pd.concat([g["pctChg"].shift(-i) for i in range(1, k + 1)], axis=1)
    valid = mat.notna().all(axis=1) & mat.abs().le(22).all(axis=1)
    df[f"f{k}"] = ((1 + mat / 100).prod(axis=1) - 1).where(valid)

# ---- 特征电池 (T日收盘) ----
c, h, lo, v = df["close"], df["high"], df["low"], df["volume"]
tf = lambda s, n: s.transform if False else None  # noqa
ma = lambda n: g["close"].transform(lambda s: s.rolling(n).mean())
df["r5"] = g["close"].transform(lambda s: s.pct_change(5))
df["r10"] = g["close"].transform(lambda s: s.pct_change(10))
df["r20"] = g["close"].transform(lambda s: s.pct_change(20))
df["ma5"], df["ma10"], df["ma20"], df["ma60"] = ma(5), ma(10), ma(20), ma(60)
df["x_ma5"] = c / df["ma5"] - 1
df["x_ma10"] = c / df["ma10"] - 1
df["x_ma20"] = c / df["ma20"] - 1
df["x_ma60"] = c / df["ma60"] - 1
df["bull_ma"] = ((df["ma5"] > df["ma10"]) & (df["ma10"] > df["ma20"])).astype(float)
df["ma20_slope"] = df["ma20"] / g["ma20"].transform(lambda s: s.shift(5)) - 1
df["vol20"] = g["pctChg"].transform(lambda s: s.rolling(20).std())
hi20 = g["close"].transform(lambda s: s.rolling(20).max())
lo20c = g["close"].transform(lambda s: s.rolling(20).min())
df["range20"] = (hi20 - lo20c) / df["ma20"]
df["pos20"] = (c - lo20c) / (hi20 - lo20c + 1e-9)
df["dry5"] = g["volume"].transform(lambda s: s.rolling(5).mean()) / \
             g["volume"].transform(lambda s: s.rolling(20).mean())
df["vr"] = v / g["volume"].transform(lambda s: s.rolling(5).mean().shift(1))
# MACD 12/26/9
e12 = g["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
e26 = g["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
df["dif"] = e12 - e26
df["dea"] = g["dif"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
df["macd_bar"] = df["dif"] - df["dea"]
gcut = (g["dif"].shift(1) <= g["dea"].shift(1)).fillna(False)
df["macd_x3"] = ((df["dif"] > df["dea"]) & gcut).astype(float)
prev_x = g["macd_x3"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).max())
df["macd_x3d"] = ((df["macd_x3"] > 0) | (prev_x > 0)).astype(float)
df["dif_pos"] = (df["dif"] > 0).astype(float)
# KDJ 9
l9 = g["low"].transform(lambda s: s.rolling(9).min())
h9 = g["high"].transform(lambda s: s.rolling(9).max())
rsv = ((c - l9) / (h9 - l9 + 1e-9) * 100)
df["rsv"] = rsv
df["kv"] = g["rsv"].transform(lambda s: s.ewm(com=2, adjust=False).mean())
df["dv"] = g["kv"].transform(lambda s: s.ewm(com=2, adjust=False).mean())
df["jv"] = 3 * df["kv"] - 2 * df["dv"]
kcut = (g["kv"].shift(1) <= g["dv"].shift(1)).fillna(False)
df["kd_x3"] = ((df["kv"] > df["dv"]) & kcut).astype(float)
prev_kx = g["kd_x3"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).max())
df["kd_x3d"] = ((df["kd_x3"] > 0) | (prev_kx > 0)).astype(float)
df["is_lim"] = (df["pctChg"] >= 9.5).astype(float)
df["lim10"] = g["is_lim"].transform(lambda s: s.rolling(10).sum())
df["red"] = (df["pctChg"] > 0).astype(float)
df["band"] = np.where(df["amount"] >= 8e7, ">=80M", "50-80M")

# ---- 近1年池 + 截面分位 ----
d = df[(df["date"] >= df["date"].max() - pd.Timedelta(days=365)) &
       (df["amount"] >= 5e7)].copy()
FEATS = ["x_ma5", "x_ma10", "x_ma20", "x_ma60", "bull_ma", "ma20_slope",
         "r5", "r10", "r20", "vol20", "range20", "pos20", "dry5", "vr",
         "macd_bar", "macd_x3d", "dif_pos", "kd_x3d", "jv", "lim10", "red"]
for f in FEATS:
    d[f"p_{f}"] = d.groupby("date")[f].rank(pct=True)

# ---- 案例截面分位表 ----
P("=" * 100)
P("案例信号日截面分位 (全池额>=5e7当日; p_xxx=分位0-1, 括号=原始值)")
P("=" * 100)
rows = []
for sym, ds in CASES:
    r = d[(d["symbol"] == sym) & (d["date"] == ds)]
    if r.empty:
        P(f"{sym} {ds}: 不在池/无数据")
        continue
    r = r.iloc[0]
    row = {"案例": f"{sym}@{ds[5:]}"}
    for f in FEATS:
        pv = r.get(f"p_{f}", np.nan)
        row[f] = f"{pv:.2f}({r[f]:.3g})" if pd.notna(pv) else "NA"
    rows.append(row)
P(pd.DataFrame(rows).to_string(index=False))

# ---- 共识: 案例分位中位数, 找一致极端 ----
P("\n" + "=" * 100)
P("共识表 (8案分位中位; |中位-0.5|越大越有区分力)")
P("=" * 100)
cons = []
for f in FEATS:
    pvs = []
    for sym, ds in CASES:
        r = d[(d["symbol"] == sym) & (d["date"] == ds)]
        if not r.empty and pd.notna(r.iloc[0].get(f"p_{f}")):
            pvs.append(r.iloc[0][f"p_{f}"])
    if len(pvs) >= 6:
        cons.append({"特征": f, "案例中位分位": np.median(pvs), "同侧率": max(np.mean(np.array(pvs) > 0.5),
                                                                     np.mean(np.array(pvs) < 0.5)),
                     "n案": len(pvs)})
cons = pd.DataFrame(cons).sort_values("案例中位分位", key=lambda s: (s - 0.5).abs(), ascending=False)
P(cons.round(3).to_string(index=False))

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))
print(f"\nsaved -> {OUT}")
