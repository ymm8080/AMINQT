# -*- coding: utf-8 -*-
"""Full alt data evaluation — all Tushare APIs + feature engine + IC."""

import sys
import os
import time
import logging
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ==== 1. Load panel ====
df = pd.read_parquet("data/panel_full_enriched.parquet")
rng = np.random.RandomState(42)
syms = rng.choice(df["symbol"].unique(), 500, replace=False)
df = df[df["symbol"].isin(syms)].sort_values(["symbol", "date"]).reset_index(drop=True)
logger.info("Panel: %d stocks, %d rows", df["symbol"].nunique(), len(df))

from app.pipeline1.data_supply import DataSupplyChain
from app.pipeline1.cleaning_pipeline import board_of, get_limit_pct

supply = DataSupplyChain()
pro = supply._tushare_pro()

# Industry from Tushare
basic = pro.stock_basic(exchange="", list_status="L", fields="ts_code,industry")
basic["symbol"] = basic["ts_code"].str.replace(".SZ", "").str.replace(".SH", "")
ind_map = dict(zip(basic["symbol"], basic["industry"].fillna("综合")))
df["industry"] = df["symbol"].map(ind_map).fillna("综合")
df["board"] = df["symbol"].map(board_of)
logger.info("Industry: %d unique", df["industry"].nunique())

start = df["date"].min().strftime("%Y%m%d")
end = df["date"].max().strftime("%Y%m%d")

# ==== 2. Fetch alt data ====

# 2a. fina_indicator — per-stock, sample 100 stocks for proof of concept
t0 = time.time()
fin_frames = []
test_syms = list(df["symbol"].unique())[:100]
for sym in test_syms:
    try:
        ts_code = sym + (".SZ" if sym.startswith(("0", "3", "1")) else ".SH")
        raw = pro.fina_indicator(
            ts_code=ts_code,
            fields="ts_code,ann_date,end_date,roe,roe_dt,roa,np_margin,gross_margin,eps_yoy,or_yoy,profit_yoy,cf_sales,debt_to_assets,current_ratio,assets_turn,inv_turn,ocf_to_or",
        )
        if raw is not None and len(raw) > 0:
            raw["symbol"] = sym
            fin_frames.append(raw)
    except:
        pass
    time.sleep(0.05)
if fin_frames:
    fin = pd.concat(fin_frames, ignore_index=True)
    fin = fin.rename(
        columns={
            "ann_date": "announce_date",
            "end_date": "report_period",
            "np_margin": "net_margin",
            "or_yoy": "rev_yoy",
            "cf_sales": "op_cf_ratio",
            "debt_to_assets": "debt_ratio",
            "assets_turn": "asset_turnover",
            "inv_turn": "inventory_turnover",
        }
    )
    fin = fin.loc[:, ~fin.columns.duplicated()]  # dedupe
    for c in [
        "roe",
        "roe_dt",
        "roa",
        "gross_margin",
        "net_margin",
        "eps_yoy",
        "rev_yoy",
        "profit_yoy",
        "op_cf_ratio",
        "debt_ratio",
        "current_ratio",
        "asset_turnover",
        "inventory_turnover",
        "ocf_to_or",
    ]:
        if c in fin.columns:
            fin[c] = pd.to_numeric(fin[c], errors="coerce")
    if "announce_date" in fin.columns:
        fin["announce_date"] = pd.to_datetime(
            fin["announce_date"], format="%Y%m%d", errors="coerce"
        )
        fin_cols = [
            c
            for c in fin.columns
            if c
            not in ("symbol", "ts_code", "report_period", "ann_date", "announce_date")
        ]
        f = (
            fin[["symbol", "announce_date"] + fin_cols]
            .dropna(subset=["announce_date"])
            .sort_values("announce_date")
        )
        df = df.sort_values("date")
        df = pd.merge_asof(
            df,
            f,
            left_on="date",
            right_on="announce_date",
            by="symbol",
            direction="backward",
        )
logger.info(
    "fina_indicator: %d/%d stocks, %.1fs",
    len(fin_frames),
    len(test_syms),
    time.time() - t0,
)

# 2b. holdernumber
t0 = time.time()
hn_all = []
for sym in df["symbol"].unique():
    try:
        ts_code = sym + (".SZ" if sym.startswith(("0", "3", "1")) else ".SH")
        raw = pro.stk_holdernumber(ts_code=ts_code, start_date=start, end_date=end)
        if raw is not None and len(raw) > 0:
            raw["symbol"] = sym
            raw["announce_date"] = pd.to_datetime(
                raw.get("ann_date", raw.get("end_date")),
                format="%Y%m%d",
                errors="coerce",
            )
            raw["holder_count"] = pd.to_numeric(
                raw.get("holder_num", 0), errors="coerce"
            )
            hn_all.append(raw[["symbol", "announce_date", "holder_count"]])
    except:
        pass
    time.sleep(0.03)
if hn_all:
    hn = (
        pd.concat(hn_all, ignore_index=True)
        .dropna(subset=["announce_date"])
        .sort_values("announce_date")
    )
    df = df.sort_values("date")
    df = pd.merge_asof(
        df,
        hn,
        left_on="date",
        right_on="announce_date",
        by="symbol",
        direction="backward",
    )
logger.info("holdernumber: %d stocks, %.1fs", len(hn_all), time.time() - t0)

# 2c. margin (sampled trading days to avoid rate limit)
t0 = time.time()
mg_frames = []
sample_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="BMS")[:8]
for d in sample_dates:
    dt = d.strftime("%Y%m%d")
    try:
        raw = pro.margin_detail(trade_date=dt)
        if raw is not None and len(raw) > 0:
            raw["symbol"] = raw["ts_code"].str.replace(".SZ", "").str.replace(".SH", "")
            raw["date"] = pd.to_datetime(d)
            for c in ["rzye", "rqye", "rzmre", "rqmcl"]:
                if c in raw.columns:
                    raw[c] = pd.to_numeric(raw[c], errors="coerce")
            mg_frames.append(raw[["symbol", "date", "rzye", "rqye", "rzmre", "rqmcl"]])
    except:
        pass
    time.sleep(0.3)
if mg_frames:
    mg = pd.concat(mg_frames, ignore_index=True).rename(
        columns={
            "rzye": "margin_balance",
            "rqye": "short_balance",
            "rzmre": "margin_buy_amt",
            "rqmcl": "short_sell_vol",
        }
    )
    df = df.merge(mg, on=["symbol", "date"], how="left")
logger.info("margin: %d days, %.1fs", len(mg_frames), time.time() - t0)

# 2d. moneyflow (Tushare)
t0 = time.time()
mf_frames = []
for d in sample_dates:
    dt = d.strftime("%Y%m%d")
    try:
        raw = pro.moneyflow(trade_date=dt)
        if raw is not None and len(raw) > 0:
            raw["symbol"] = raw["ts_code"].str.replace(".SZ", "").str.replace(".SH", "")
            raw["date"] = pd.to_datetime(d)
            for c in [
                "net_mf_amount",
                "buy_lg_vol",
                "sell_lg_vol",
                "buy_elg_vol",
                "sell_elg_vol",
            ]:
                if c in raw.columns:
                    raw[c] = pd.to_numeric(raw[c], errors="coerce")
            mf_frames.append(
                raw[
                    [
                        "symbol",
                        "date",
                        "net_mf_amount",
                        "buy_lg_vol",
                        "sell_lg_vol",
                        "buy_elg_vol",
                        "sell_elg_vol",
                    ]
                ]
            )
    except:
        pass
    time.sleep(0.3)
if mf_frames:
    mf = pd.concat(mf_frames, ignore_index=True)
    mf["main_money_flow"] = mf.get("net_mf_amount", 0)
    mf["super_large_order_net"] = mf.get("buy_elg_vol", 0) - mf.get("sell_elg_vol", 0)
    df = df.merge(
        mf[["symbol", "date", "main_money_flow", "super_large_order_net"]],
        on=["symbol", "date"],
        how="left",
    )
logger.info("moneyflow: %d days, %.1fs", len(mf_frames), time.time() - t0)

# Coverage report
print("\n=== Data Coverage ===")
for c in ["roe", "holder_count", "margin_balance", "main_money_flow"]:
    if c in df.columns:
        pct = (1 - df[c].isna().mean()) * 100
        print(f"  {c:<25s}: {pct:.1f}%")

# ==== 3. Build features ====
df["is_st"] = False
df["is_suspended"] = False
df["list_days"] = df.groupby("symbol").cumcount() + 1
df["limit_pct"] = [get_limit_pct(b, d) for b, d in zip(df["board"], df["date"])]

from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import LabelEngine

t0 = time.time()
fe = FeatureEngineV35()
df = fe.build(df)
df = LabelEngine.build_path_labels(df)
df = LabelEngine.build_labels(df)
df = LabelEngine.mask_suspension(df)
df = LabelEngine.mask_recent_days(df, days=6)
logger.info("Features: %d cols in %.1fs", len(df.columns), time.time() - t0)

# ==== 4. IC evaluation ====
label_1d = "label_1d_net" if "label_1d_net" in df.columns else "label_1d"
label_3d = "label_3d_net" if "label_3d_net" in df.columns else "label_3d"


def rank_ic(df, factor, label):
    sub = df[[factor, "date", label]].dropna()
    if len(sub) < 500:
        return 0.0
    ics = []
    for _, g in sub.groupby("date"):
        if len(g) >= 10:
            ic = g[[factor, label]].corr(method="spearman").iloc[0, 1]
            if not np.isnan(ic):
                ics.append(ic)
    return float(np.mean(np.abs(ics))) if ics else 0.0


# Test each new dimension
dims_features = {
    "dim22_finPIT": [
        "roe_zscore",
        "roa_zscore",
        "margin_composite",
        "growth_composite",
        "quality_score",
        "efficiency_score",
        "roe_stability",
        "margin_trend",
    ],
    "dim23_shareholder": [
        "holder_count_log",
        "holder_count_qoq",
        "holder_count_yoy",
        "holder_qoq_accel",
        "avg_shares_log",
        "avg_shares_qoq",
        "avg_shares_yoy",
        "holder_concentration_zscore",
    ],
    "dim24_margin": [
        "margin_balance_chg_1d",
        "margin_balance_chg_5d",
        "short_balance_ratio",
        "margin_buy_ratio",
        "margin_balance_ma20_dev",
        "margin_balance_yoy",
        "margin_pressure_score",
    ],
    "dim25_northbound": [
        "north_net_buy_5d",
        "north_net_buy_20d",
        "north_net_buy_streak",
        "north_buy_ratio",
        "north_sh_sz_divergence",
        "north_momentum_5d",
        "north_flow_zscore",
    ],
    "dim26_lhb": [
        "lhb_inst_net_buy_5d",
        "lhb_inst_net_buy_20d",
        "lhb_inst_count_5d",
        "lhb_inst_buy_ratio",
        "lhb_abnormal_score",
    ],
    "dim27_indflow": [
        "ind_north_flow_5d",
        "ind_north_flow_rank",
        "ind_margin_balance_chg",
        "ind_margin_rank",
        "ind_holder_qoq_mean",
        "ind_holder_rank",
        "ind_lhb_net_buy_5d",
        "ind_lhb_activity",
        "ind_flow_composite",
    ],
    "dim28_sector": [
        "sw_ret_1d",
        "sw_ret_5d",
        "sw_ret_20d",
        "sw_vol_20d",
        "sw_relative_strength",
        "sw_rotation_position",
        "sw_momentum_accel",
        "sw_turnover_anomaly",
    ],
}

print()
print("=" * 100)
print(
    f"{'Dim':<20s} {'Factor':<30s} {'IC_1d':<8s} {'IC_3d':<8s} {'NaN%':<7s} {'Signal'}"
)
print("-" * 100)

results = {}
for dim, feats in dims_features.items():
    best_ic, best_f, best_nan = 0, "", 100
    dim_results = []
    for f in feats:
        if f not in df.columns:
            continue
        nan_r = df[f].isna().mean()
        if nan_r > 0.95:
            continue
        ic1 = rank_ic(df, f, label_1d)
        ic3 = rank_ic(df, f, label_3d)
        best = max(ic1, ic3)
        if best > 0.001:
            sig = (
                "STRONG"
                if best >= 0.03
                else ("OK" if best >= 0.02 else ("weak" if best >= 0.01 else "-"))
            )
            print(
                f"{dim:<20s} {f:<30s} {ic1:<8.4f} {ic3:<8.4f} {nan_r * 100:<7.1f} {sig}"
            )
        if best > best_ic:
            best_ic, best_f, best_nan = best, f, nan_r
        dim_results.append(
            {
                "factor": f,
                "ic_1d": round(ic1, 5),
                "ic_3d": round(ic3, 5),
                "nan_pct": round(nan_r * 100, 1),
            }
        )
    verdict = "INCLUDE" if best_ic >= 0.02 else ("WATCH" if best_ic >= 0.01 else "SKIP")
    results[dim] = {
        "best_factor": best_f,
        "best_ic": round(best_ic, 5),
        "nan_pct": round(best_nan * 100, 1),
        "verdict": verdict,
        "factors": dim_results,
    }

print("-" * 100)
print()
print("=== FINAL VERDICT ===")
for dim, r in results.items():
    print(
        f"  {dim:<20s}: {r['best_factor']:<30s} IC={r['best_ic']:.5f}  NaN={r['nan_pct']:.1f}%  --> {r['verdict']}"
    )

# Save
os.makedirs("data/factor_registry", exist_ok=True)
out = {"timestamp": pd.Timestamp.now().isoformat(), "results": results}
with open(
    f"data/factor_registry/alt_full_eval_{pd.Timestamp.now():%Y%m%d_%H%M}.json", "w"
) as fh:
    json.dump(out, fh, ensure_ascii=False, indent=2)
logger.info("Saved results")
