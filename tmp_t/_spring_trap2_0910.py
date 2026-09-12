# -*- coding: utf-8 -*-
"""0910 诱空假设精修 (用户: 诱空=上涨股正反馈正确形态)
第一轮教训: 戏剧性假破位(spring)总体负. 簇案真实形态=缩量阴跌不破位.
信号 (T日收盘可算):
  drift_dry  = pctChg∈[-6,0) 且 缩量(<0.6×20日均量) 且 low >= 前10日最低 (不破位)
  drift2_dry = 连续2日阴跌合计∈[-8,0) 且 末日缩量 且 2日内不破10日低
  趋势语境   = close > MA20 (用户"上涨股票"=已处上升趋势, 正反馈语境)
检验 FWD3/FWD5 vs 同语境基线. 近1年, 额>=5e7, 脏pctChg整窗剔除.
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
PNL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
OUT = r"D:/AMINQT/Daily Operation/STOCK LIST/_spring_trap2_20260910.txt"
pd.set_option("display.width", 220)
L = []
def P(s=""):
    print(s); L.append(str(s))

cols = ["symbol", "date", "low", "close", "pctChg", "amount", "volume"]
df = pd.read_parquet(PNL, columns=cols)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
g = df.groupby("symbol")
for k in (3, 5):
    mat = pd.concat([g["pctChg"].shift(-i) for i in range(1, k + 1)], axis=1)
    valid = mat.notna().all(axis=1) & mat.abs().le(22).all(axis=1)
    df[f"f{k}"] = ((1 + mat / 100).prod(axis=1) - 1).where(valid)
df["v_ma20"] = g["volume"].transform(lambda s: s.rolling(20).mean())
df["low10p"] = g["low"].transform(lambda s: s.shift(1).rolling(10).min())
df["ma20"] = g["close"].transform(lambda s: s.rolling(20).mean())
df["dry"] = df["volume"] < 0.6 * df["v_ma20"]
df["band"] = np.where(df["amount"] < 8e7, "50-80M", ">=80M")

df["drift_dry"] = df["pctChg"].between(-6, 0, inclusive="left") & df["dry"] & (df["low"] >= df["low10p"])
c1 = g["pctChg"].shift(0)
c0 = g["pctChg"].shift(1)
df["drift2_dry"] = ((c1 + c0).between(-8, 0) & (c1 < 0) & (c0 < 0)
                    & df["dry"] & (df["low"] >= df["low10p"]))
df["up_ctx"] = df["close"] > df["ma20"]

cutoff = df["date"].max() - pd.Timedelta(days=365)
d = df[(df["date"] >= cutoff) & (df["amount"] >= 5e7)].dropna(subset=["f3", "f5"]).copy()
P(f"窗: {d['date'].min():%Y-%m-%d}..{d['date'].max():%Y-%m-%d}, 股日={len(d):,}")

for ctx, cmask in (("全语境", pd.Series(True, index=d.index)),
                   ("趋势内(close>MA20)", d["up_ctx"])):
    sub_ctx = d[cmask]
    P(f"\n=== {ctx} === 基线: FWD3={sub_ctx['f3'].mean():+.4f} FWD5={sub_ctx['f5'].mean():+.4f} "
      f"hit10={(sub_ctx['f5']>=0.10).mean():.1%} (n={len(sub_ctx):,})")
    rows = []
    for sig, name in ((d["drift_dry"], "缩量阴跌不破位(单日)"),
                      (d["drift2_dry"], "连2阴缩量不破位"),
                      (d["drift_dry"] & d["up_ctx"], "  └ 限趋势内"),
                      (d["drift2_dry"] & d["up_ctx"], "  └ 限趋势内")):
        s = d[sig & cmask]
        if not len(s):
            continue
        rows.append({"信号": name, "n": len(s), "FWD3均": s["f3"].mean(),
                     "FWD3中位": s["f3"].median(), "hit5%": (s["f3"] >= 0.05).mean(),
                     "FWD5均": s["f5"].mean(), "FWD5中位": s["f5"].median(),
                     "hit10%": (s["f5"] >= 0.10).mean(),
                     "超额FWD5": s["f5"].mean() - sub_ctx["f5"].mean()})
    P(pd.DataFrame(rows).round(4).to_string(index=False))

P("\n=== 趋势内×缩量阴跌 × 额带 ===")
s = d[d["drift_dry"] & d["up_ctx"]]
rows = []
for b, sub in s.groupby("band"):
    bctx = d[(d["band"] == b) & d["up_ctx"]]
    rows.append({"band": b, "n": len(sub), "FWD3均": sub["f3"].mean(), "FWD5均": sub["f5"].mean(),
                 "hit10%": (sub["f5"] >= 0.10).mean(),
                 "同语境基线FWD5": bctx["f5"].mean(), "基线hit10%": (bctx["f5"] >= 0.10).mean(),
                 "超额": sub["f5"].mean() - bctx["f5"].mean()})
P(pd.DataFrame(rows).round(4).to_string(index=False))

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))
print(f"\nsaved -> {OUT}")
