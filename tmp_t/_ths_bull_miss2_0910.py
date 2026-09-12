# -*- coding: utf-8 -*-
"""0910 探针v4: 新增600876/002790 + 000565停牌验证 + 低价/破净/市值假设"""
import os, glob
import pandas as pd

SYMS = ["002377", "002204", "002201", "301176", "600318", "600359",
        "000993", "000565", "600876", "002790"]
PNL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
LST = r"D:/AMINQT/Daily Operation/STOCK LIST"
pd.set_option("display.width", 250)
import sys; sys.stdout.reconfigure(encoding="utf-8")

# ---- 1) 行情: 重点000565/600876/002790 全窗 + 其余尾部 ----
cols = ["symbol", "date", "close", "pctChg", "amount", "volume_ratio", "turnover_rate", "chip_skew_dist"]
df = pd.read_parquet(PNL, columns=cols)
df["date"] = pd.to_datetime(df["date"])
sub = df[df["symbol"].isin(SYMS) & (df["date"] >= "2026-08-24")].sort_values(["symbol", "date"])
for s in ["000565", "600876", "002790", "002201", "000993"]:
    d = sub[sub["symbol"] == s].copy()
    if d.empty:
        print(f"\n### {s}: NO PANEL DATA"); continue
    print(f"\n### {s}")
    print(d[["date", "close", "pctChg", "amount", "volume_ratio", "turnover_rate"]].round(3).to_string(index=False))

# ---- 2) 排名深度: 全部可用日期, 只打关键行 ----
print("\n========== RANK DEPTH v2 ==========")
ldates = sorted(set(os.path.basename(f)[17:25] for f in glob.glob(f"{LST}/legacy_preds_raw_202609*.csv")))
for dstr in ldates:
    lfile = glob.glob(f"{LST}/legacy_preds_raw_{dstr}__M*.csv")
    pfile = glob.glob(f"{LST}/parallel_preds_raw_{dstr}__M*.csv")
    if not (lfile and pfile):
        continue
    leg = pd.read_csv(lfile[0], dtype={"symbol": str}).sort_values("pred_ret_10d", ascending=False).reset_index(drop=True)
    par = pd.read_csv(pfile[0], dtype={"symbol": str})
    par["rk_score"] = par.groupby("board")["score"].rank(ascending=False).astype(int)
    par["rk_mag5"] = par.groupby("board")["pred_mag_5d"].rank(ascending=False).astype(int)
    print(f"\n--- {dstr}: leg n={len(leg)} par n={len(par)}")
    for s in SYMS:
        lr = leg.loc[leg.symbol == s, "pred_ret_10d"]
        pr = par.loc[par.symbol == s, ["board", "rk_score", "rk_mag5", "pred_mag_5d", "pred_prob_5d", "pred_mag_10d"]]
        ltxt = f"#{(leg.index[leg.symbol==s][0]+1)}/{len(leg)} mag10={float(lr.iloc[0]):+.4f}" if len(lr) else "NOT-IN-POOL"
        ptxt = pr.to_string(index=False, header=False) if len(pr) else "NOT-IN-POOL"
        print(f"  {s}: LEG {ltxt} | PAR[brd,rkScore,rkMag5,mag5,prob5,mag10] {ptxt}")

# ---- 3) 假设检验: 低价/破净/市值 (0905 daily_basic) ----
print("\n========== PB / MV / PRICE (20260905) ==========")
import tushare as ts
tok = os.getenv("TUSHARE_TOKEN") or ts.get_token()
pro = ts.pro_api(tok)
suf = {s: (".SZ" if s.startswith(("0", "3")) else ".SH") for s in SYMS}
tscodes = ",".join(s + suf[s] for s in SYMS)
db = pro.daily_basic(trade_date="20260905",
                     fields="ts_code,close,turnover_rate,pb,total_mv,circ_mv")
db = db[db.ts_code.isin(set(tscodes.split(",")))].copy()
db["close"] = db["close"]; db["total_mv亿"] = db["total_mv"] / 1e4; db["circ_mv亿"] = db["circ_mv"] / 1e4
print(db[["ts_code", "close", "pb", "total_mv亿", "circ_mv亿"]].sort_values("ts_code").to_string(index=False))
basic = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry,market")
b = basic[basic.ts_code.isin(set(tscodes.split(",")))].sort_values("ts_code")
print("\n", b.to_string(index=False))
# 全市场对照: 0905 主板低价破净占比
allb = pro.daily_basic(trade_date="20260905", fields="ts_code,close,pb,circ_mv")
allb = allb[~allb.ts_code.str.startswith(("688", "300", "301"))]
low_pb_lowp = allb[(allb.close < 13) & (allb.pb < 1.5) & (allb.circ_mv < 8e5)]
print(f"\n全市场主板 0905: n={len(allb)}, 低价(<13元)&PB<1.5&流通<80亿 = {len(low_pb_lowp)} ({len(low_pb_lowp)/len(allb):.1%})")
