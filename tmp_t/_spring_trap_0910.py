# -*- coding: utf-8 -*-
"""0910 诱空(洗盘/假破位)假设回测 (用户: 通常上涨前都有诱空行为)
信号定义 (T日收盘可算, 无未来):
  spring10  = T日low < 前10日最低low 且 T日close > 前10日最低low (盘中假破位收回)
  spring20  = 同上, 20日口径
  缩量      = volume < 0.6 × 20日均量
  变体      = spring10×缩量 (真洗盘: 破位无量)
检验: 信号日→FWD3/FWD5 (close-close) vs 池基线; 分层: 额带(50-80M vs ≥80M) × 首板前
脏pctChg(|x|>22)整窗剔除; WORM输出带日期后缀.
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
PNL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
OUT = r"D:/AMINQT/Daily Operation/STOCK LIST/_spring_trap_20260910.txt"
pd.set_option("display.width", 220)
L = []
def P(s=""):
    print(s); L.append(str(s))

cols = ["symbol", "date", "open", "high", "low", "close", "pctChg", "amount", "volume"]
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
df["low20p"] = g["low"].transform(lambda s: s.shift(1).rolling(20).min())
df["spring10"] = (df["low"] < df["low10p"]) & (df["close"] > df["low10p"])
df["spring20"] = (df["low"] < df["low20p"]) & (df["close"] > df["low20p"])
df["dry"] = df["volume"] < 0.6 * df["v_ma20"]
df["band"] = np.where(df["amount"] < 5e7, "<5e7",
              np.where(df["amount"] < 8e7, "50-80M", ">=80M"))
df["is_lim"] = (df["pctChg"] >= 9.5).astype(int)
df["prev_lim5"] = g["is_lim"].transform(lambda s: s.shift(1).rolling(5).sum())
df["pre_board"] = np.where(df["prev_lim5"] > 0, "5日内有板", "无板基底")

cutoff = df["date"].max() - pd.Timedelta(days=365)
d = df[(df["date"] >= cutoff) & (df["amount"] >= 5e7)].copy()
P(f"窗: {d['date'].min():%m-%d}..{d['date'].max():%m-%d}, 池(额>=5e7)股日={len(d):,}")

base = d.dropna(subset=["f3", "f5"])
P(f"池基线: FWD3={base['f3'].mean():+.4f} (中位{base['f3'].median():+.4f}) hit(F3>=5%)={(base['f3']>=0.05).mean():.1%}"
  f" | FWD5={base['f5'].mean():+.4f} hit(F5>=10%)={(base['f5']>=0.10).mean():.1%}")

rows = []
for sig, name in ((d["spring10"], "spring10 假破位收回"),
                  (d["spring20"], "spring20 假破位收回"),
                  (d["spring10"] & d["dry"], "spring10×缩量"),
                  (d["spring20"] & d["dry"], "spring20×缩量"),
                  (d["dry"], "仅缩量(对照)")):
    s = d[sig].dropna(subset=["f3", "f5"])
    rows.append({"信号": name, "n": len(s), "FWD3均": s["f3"].mean(), "FWD3中位": s["f3"].median(),
                 "hit5%": (s["f3"] >= 0.05).mean(), "FWD5均": s["f5"].mean(),
                 "hit10%": (s["f5"] >= 0.10).mean(),
                 "超额FWD5": s["f5"].mean() - base["f5"].mean()})
P("\n=== 总体: 信号日→FWD3/FWD5 ===")
P(pd.DataFrame(rows).round(4).to_string(index=False))

P("\n=== spring10×缩量 × 额带 × 板位 ===")
s = d[(d["spring10"]) & (d["dry"])].dropna(subset=["f3", "f5"])
rows = []
for (b, pb), sub in s.groupby(["band", "pre_board"]):
    rows.append({"band": b, "板位": pb, "n": len(sub), "FWD3均": sub["f3"].mean(),
                 "FWD5均": sub["f5"].mean(), "hit10%": (sub["f5"] >= 0.10).mean(),
                 "基线FWD5": base[(base["band"] == b) & (base["pre_board"] == pb)]["f5"].mean(),
                 "基线hit10%": (base[(base["band"] == b) & (base["pre_board"] == pb)]["f5"] >= 0.10).mean()})
P(pd.DataFrame(rows).round(4).to_string(index=False))

# ================= 案例: 10簇票 0901-0910 逐日行为 =================
P("\n=== 案例: THS簇10只 08-25..09-10 (pctChg/额/spring10/缩量) ===")
CASE = ["002377", "002204", "002790", "301176", "000565",
        "002201", "600359", "000993", "600318", "600876"]
w = df[(df["date"] >= "2026-08-25") & (df["symbol"].isin(CASE))].copy()
for sym, sub in w.groupby("symbol"):
    sub = sub.sort_values("date")
    marks = []
    for _, r in sub.iterrows():
        m = ""
        if r["spring10"]:
            m += "S10"
        if r["spring20"]:
            m += "S20"
        if r["dry"]:
            m += "缩"
        if r["pctChg"] >= 9.5:
            m += "板"
        marks.append(m or "-")
    bar = " ".join(f"{r['date']:%m-%d}:{r['pctChg']:+.1f}{m}"
                   for (_, r), m in zip(sub.iterrows(), marks))
    P(f"{sym}: {bar}")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))
print(f"\nsaved -> {OUT}")
