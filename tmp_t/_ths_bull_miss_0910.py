# -*- coding: utf-8 -*-
"""0910 探针v3: THS看多信号集群漏抓归因 (8只)
002377/002204(0908) 301176/000565/002201(0907) 600318 600359(0904) 000993(0901)
全部于09-10涨停(+10%), 301176 +15.3% — 集群事件
输出: 行情验证 / 各日两线排名深度+短名单截位差 / 行业 / 市场面
"""
import os, glob
import pandas as pd

SYMS = ["002377", "002204", "002201", "301176", "600318", "600359", "000993", "000565"]
PNL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
LST = r"D:/AMINQT/Daily Operation/STOCK LIST"
pd.set_option("display.width", 250)

cols = ["symbol", "date", "close", "pctChg", "amount", "volume_ratio",
        "turnover_rate", "chip_skew_dist", "chip_gini"]
df = pd.read_parquet(PNL, columns=cols)
df["date"] = pd.to_datetime(df["date"])

sub = df[df["symbol"].isin(SYMS) & (df["date"] >= "2026-08-27")].sort_values(["symbol", "date"])
for s in SYMS:
    d = sub[sub["symbol"] == s].copy()
    if d.empty:
        print(f"\n### {s}: NO PANEL DATA"); continue
    d["cum"] = (1 + d["pctChg"] / 100).cumprod().where(d["date"] >= "2026-09-04") - 1
    print(f"\n### {s}")
    print(d[["date", "close", "pctChg", "cum", "amount", "volume_ratio",
             "turnover_rate", "chip_skew_dist"]].round(3).to_string(index=False))

# ---- 排名深度: 全部可用日期 ----
print("\n\n========== RANK DEPTH ==========")
ldates = sorted(set(os.path.basename(f)[17:25] for f in glob.glob(f"{LST}/legacy_preds_raw_202609*.csv")))
for dstr in ldates:
    lfile = glob.glob(f"{LST}/legacy_preds_raw_{dstr}__M*.csv")
    pfile = glob.glob(f"{LST}/parallel_preds_raw_{dstr}__M*.csv")
    slfile = glob.glob(f"{LST}/parallel_shortlist_{dstr}__M*.csv")
    if not (lfile and pfile):
        print(f"\n--- {dstr}: raw missing, skip"); continue
    leg = pd.read_csv(lfile[0], dtype={"symbol": str}).sort_values("pred_ret_10d", ascending=False).reset_index(drop=True)
    leg["rk10"] = (leg.index + 1).astype(str) + "/" + str(len(leg))
    par = pd.read_csv(pfile[0], dtype={"symbol": str})
    n_by_brd = par.groupby("board").size().to_dict()
    par["rk_score"] = par.groupby("board")["score"].rank(ascending=False).astype(int)
    par["rk_mag5"] = par.groupby("board")["pred_mag_5d"].rank(ascending=False).astype(int)
    sl = pd.read_csv(slfile[0], dtype={"symbol": str}) if slfile else pd.DataFrame()
    sl_last_score = float(sl["score"].min()) if len(sl) else float("nan")
    print(f"\n--- {dstr}: legacy n={len(leg)}, parallel n={len(par)} {n_by_brd}, shortlist n={len(sl)} cutoff_score={sl_last_score:.4f}")
    for s in SYMS:
        lr = leg.loc[leg.symbol == s, ["rk10", "pred_ret_10d", "prob_up_10d", "pred_ret_5d", "prob_up_5d"]].round(4)
        pr = par.loc[par.symbol == s, ["board", "rk_score", "rk_mag5", "score", "pred_mag_5d", "pred_prob_5d",
                                       "pred_mag_10d", "pred_prob_10d"]].round(4)
        ltxt = lr.to_string(index=False, header=False) if not lr.empty else "NOT-IN-POOL"
        ptxt = pr.to_string(index=False, header=False) if not pr.empty else "NOT-IN-POOL"
        in_sl = "YES" if (len(sl) and s in set(sl["symbol"])) else ""
        print(f"  {s} {in_sl}")
        print(f"    LEG[rk10|mag10|prob10|mag5|prob5] {ltxt}")
        print(f"    PAR[brd|rkScore|rkMag5|score|mag5|prob5|mag10|prob10] {ptxt}")

# ---- 市场面 + 名称行业 ----
print("\n\n========== MARKET CONTEXT / NAMES ==========")
import tushare as ts
tok = os.getenv("TUSHARE_TOKEN") or ts.get_token()
pro = ts.pro_api(tok)
suf = {s: (".SZ" if s.startswith(("0", "3")) else ".SH") for s in SYMS}
tscodes = ",".join(s + suf[s] for s in SYMS)
basic = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry,market")
b = basic[basic.ts_code.isin(set(tscodes.split(",")))].sort_values("ts_code")
print(b.to_string(index=False))
for td in ["20260909", "20260910"]:
    d = pro.daily(trade_date=td)
    main10 = d[(d.pct_chg >= 9.7) & (~d.ts_code.str.startswith(("688", "300", "301")))]
    lim20 = d[d.pct_chg >= 19.5]
    print(f"{td}: up={(d.pct_chg > 0).sum()} down={(d.pct_chg < 0).sum()} "
          f"limit10+={len(main10)} limit20+={len(lim20)} total={len(d)}")
    hit = d[d.ts_code.isin(set(tscodes.split(",")))]
    print(hit[["ts_code", "close", "pct_chg", "amount"]].to_string(index=False))
for tc in ["000001.SH", "399006.SZ"]:
    idx = pro.index_daily(ts_code=tc, start_date="20260901", end_date="20260910").sort_values("trade_date")
    print(idx[["trade_date", "pct_chg"]].to_string(index=False))
