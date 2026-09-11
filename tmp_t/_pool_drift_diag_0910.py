# -*- coding: utf-8 -*-
"""0910 池漂移诊断: 000565/002790 等 0904/0907 池消失 = 金额墙(设计) 还是 数据缺口(bug)
判据: 面板当日行存在 + amount>=5e7 + 非停牌 → 池里却没有 = 运行时面板缺口(可疑)
      amount<5e7 → 5000万墙(设计内) | 面板无行 → 入库缺口
"""
import os, glob
import pandas as pd
import sys
sys.stdout.reconfigure(encoding="utf-8")

SYMS = ["002377", "002204", "002201", "301176", "600318", "600359",
        "000993", "000565", "600876", "002790"]
PNL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
LST = r"D:/AMINQT/Daily Operation/STOCK LIST"
pd.set_option("display.width", 260)

# ---- 1) 各日两线池成员集合 ----
ldates = sorted(set(os.path.basename(f)[17:25] for f in glob.glob(f"{LST}/legacy_preds_raw_202609*.csv")))
pool = {}
print("========== POOL SIZES ==========")
for dstr in ldates:
    lf = glob.glob(f"{LST}/legacy_preds_raw_{dstr}__M*.csv")
    pf = glob.glob(f"{LST}/parallel_preds_raw_{dstr}__M*.csv")
    leg = set(pd.read_csv(lf[0], dtype={"symbol": str})["symbol"]) if lf else set()
    par = set(pd.read_csv(pf[0], dtype={"symbol": str})["symbol"]) if pf else set()
    pool[dstr] = (leg, par)
    print(f"{dstr}: legacy n={len(leg)}, parallel n={len(par)}")

# ---- 2) 10只 × 各日 × 两线: 在池? ----
print("\n========== POOL MEMBERSHIP (L=legacy, P=parallel, .=in, x=OUT) ==========")
hdr = "sym    | " + " | ".join(f"{d[4:]}L{d[2:4]}P" for d in ldates)
print(hdr)
for s in SYMS:
    row = []
    for dstr in ldates:
        leg, par = pool[dstr]
        row.append(("." if s in leg else "x") + ("." if s in par else "x"))
    print(f"{s} | " + "  | ".join(row))

# ---- 3) 面板当日行情: amount vs 5e7 墙 + 行存在性 ----
print("\n========== PANEL: amount vs 5000万 WALL ==========")
cols = ["symbol", "date", "close", "pctChg", "amount", "turnover_rate"]
df = pd.read_parquet(PNL, columns=cols)
df["date"] = pd.to_datetime(df["date"])
sub = df[df["symbol"].isin(SYMS) & (df["date"] >= "2026-08-28")].copy()
sub["wall_ok"] = sub["amount"] >= 5e7
pv = sub.pivot_table(index="symbol", columns="date", values="amount").round(0) / 1e6
print("amount (百万元):")
print(pv.to_string())
wv = sub.pivot_table(index="symbol", columns="date", values="wall_ok")
print("\n>=5000万墙 (True=过墙):")
print(wv.to_string())

# ---- 4) 关键股停牌/缺行检查: 交易日历对照 ----
print("\n========== KEY STOCKS row presence ==========")
for s in ["000565", "002790", "002377"]:
    d = sub[sub["symbol"] == s].sort_values("date")
    dates_have = set(d["date"].dt.strftime("%Y%m%d"))
    print(f"{s}: panel rows {sorted(dates_have)}")
