"""Full alt data fetch + IC evaluation + PIPELINE1 integration.
Step 1: Fetch all alt data (margin, moneyflow, holdernumber, northbound, lhb)
Step 2: Merge into panel
Step 3: Build features (all 26 dims)
Step 4: IC per dim -> verdict
Step 5: Save results + factor registry
"""

import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

from app.pipeline1.cleaning_pipeline import board_of, get_limit_pct  # noqa: E402
from app.pipeline1.data_supply import DataSupplyChain  # noqa: E402

supply = DataSupplyChain()
pro = supply._tushare_pro()

CACHE = "data/supply_cache/alt_data"

# ====== STEP 1: Load panel + industry ======
logger.info("=== STEP 1: Load panel ===")
df = pd.read_parquet("data/panel_full_enriched.parquet")
df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
logger.info(
    "Panel: %d stocks, %d rows, %d dates",
    df["symbol"].nunique(),
    len(df),
    df["date"].nunique(),
)

# Industry
basic = pro.stock_basic(exchange="", list_status="L", fields="ts_code,industry")
basic["symbol"] = basic["ts_code"].str.replace(".SZ", "").str.replace(".SH", "")
ind_map = dict(zip(basic["symbol"], basic["industry"].fillna("综合"), strict=False))
df["industry"] = df["symbol"].map(ind_map).fillna("综合")
df["board"] = df["symbol"].map(board_of)
logger.info("Industry: %d unique", df["industry"].nunique())

all_dates = sorted(df["date"].dropna().unique())
all_symbols = sorted(df["symbol"].unique())
logger.info("Dates: %d, Symbols: %d", len(all_dates), len(all_symbols))

# ====== STEP 2: Fetch margin (every 10th date = ~62 dates) ======
logger.info("=== STEP 2: Fetch margin_detail ===")
margin_dates = all_dates[::10]  # ~62 dates
mg_path = os.path.join(CACHE, "margin_panel.parquet")
if os.path.exists(mg_path):
    mg_panel = pd.read_parquet(mg_path)
    logger.info("Margin from cache: %d rows", len(mg_panel))
else:
    mg_frames = []
    for i, d in enumerate(margin_dates):
        dt = d.strftime("%Y%m%d")
        try:
            raw = pro.margin_detail(trade_date=dt)
            if raw is not None and len(raw) > 0:
                raw["symbol"] = (
                    raw["ts_code"].str.replace(".SZ", "").str.replace(".SH", "")
                )
                raw["date"] = pd.to_datetime(d)
                for c in ["rzye", "rqye", "rzmre", "rqmcl"]:
                    if c in raw.columns:
                        raw[c] = pd.to_numeric(raw[c], errors="coerce")
                mg_frames.append(
                    raw[["symbol", "date", "rzye", "rqye", "rzmre", "rqmcl"]]
                )
        except Exception as e:
            if i < 3:
                logger.warning("margin %s: %s", dt, str(e)[:60])
        if (i + 1) % 20 == 0:
            logger.info("  margin: %d/%d", i + 1, len(margin_dates))
        time.sleep(0.15)
    if mg_frames:
        mg_panel = pd.concat(mg_frames, ignore_index=True)
        mg_panel = mg_panel.rename(
            columns={
                "rzye": "margin_balance",
                "rqye": "short_balance",
                "rzmre": "margin_buy_amt",
                "rqmcl": "short_sell_vol",
            }
        )
        os.makedirs(os.path.dirname(mg_path), exist_ok=True)
        mg_panel.to_parquet(mg_path, index=False)
    logger.info(
        "Margin: %d days, %d rows", len(mg_frames), len(mg_panel) if mg_frames else 0
    )

# Merge margin
if "mg_panel" in dir() and len(mg_panel) > 0:
    df = df.merge(mg_panel, on=["symbol", "date"], how="left")
    logger.info(
        "Margin merged: cov=%.1f%%", (1 - df["margin_balance"].isna().mean()) * 100
    )

# ====== STEP 3: Fetch moneyflow (every 10th date) ======
logger.info("=== STEP 3: Fetch moneyflow ===")
mf_frames = []
for i, d in enumerate(all_dates[::10]):
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
    except Exception:  # noqa: E722
        pass
    if (i + 1) % 20 == 0:
        logger.info("  moneyflow: %d/%d", i + 1, len(all_dates[::10]))
    time.sleep(0.15)
if mf_frames:
    mf = pd.concat(mf_frames, ignore_index=True)
    mf["main_money_flow"] = mf["net_mf_amount"]
    mf["super_large_order_net"] = mf["buy_elg_vol"].fillna(0) - mf[
        "sell_elg_vol"
    ].fillna(0)
    df = df.merge(
        mf[
            [
                "symbol",
                "date",
                "main_money_flow",
                "super_large_order_net",
                "buy_lg_vol",
                "sell_lg_vol",
                "buy_elg_vol",
                "sell_elg_vol",
            ]
        ],
        on=["symbol", "date"],
        how="left",
    )
logger.info(
    "Moneyflow: %d days, cov=%.1f%%",
    len(mf_frames),
    (1 - df["main_money_flow"].isna().mean()) * 100
    if "main_money_flow" in df.columns
    else 0,
)

# ====== STEP 4: Fetch holdernumber (all stocks) ======
logger.info("=== STEP 4: Fetch holdernumber ===")
hn_path = os.path.join(CACHE, "holdernumber_all.parquet")
if os.path.exists(hn_path):
    hn_all = pd.read_parquet(hn_path)
    logger.info("Holdernumber from cache: %d rows", len(hn_all))
else:
    hn_frames = []
    for i, sym in enumerate(all_symbols):
        try:
            ts_code = sym + (".SZ" if sym.startswith(("0", "3", "1")) else ".SH")
            raw = pro.stk_holdernumber(ts_code=ts_code)
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
                hn_frames.append(raw[["symbol", "announce_date", "holder_count"]])
        except Exception:  # noqa: E722
            pass
        if (i + 1) % 500 == 0:
            logger.info("  holdernumber: %d/%d", i + 1, len(all_symbols))
        time.sleep(0.05)
    if hn_frames:
        hn_all = pd.concat(hn_frames, ignore_index=True)
        os.makedirs(os.path.dirname(hn_path), exist_ok=True)
        hn_all.to_parquet(hn_path, index=False)
    logger.info(
        "Holdernumber: %d/%d stocks, %d rows",
        len(hn_frames),
        len(all_symbols),
        len(hn_all) if hn_frames else 0,
    )

# Merge holdernumber (PIT via announce_date)
if "hn_all" in dir() and len(hn_all) > 0:
    hn_all = hn_all.dropna(subset=["holder_count", "announce_date"]).sort_values(
        "announce_date"
    )
    df = df.sort_values("date")
    df = pd.merge_asof(
        df,
        hn_all[["symbol", "announce_date", "holder_count"]],
        left_on="date",
        right_on="announce_date",
        by="symbol",
        direction="backward",
    )
    logger.info(
        "Holdernumber merged: cov=%.1f%%", (1 - df["holder_count"].isna().mean()) * 100
    )

# ====== STEP 5: Fetch northbound (market-level, sample dates) ======
logger.info("=== STEP 5: Fetch northbound (market-level) ===")
# moneyflow_hsgt is MARKET-level, not per-stock
# We'll create a simple northbound feature from the market data
# For stock-level northbound: hk_hold (沪深港通持股明细)
nb_frames = []
for i, d in enumerate(all_dates[::5][:100]):  # Sample ~100 dates
    dt = d.strftime("%Y%m%d")
    try:
        raw = pro.hk_hold(trade_date=dt)
        if raw is not None and len(raw) > 0:
            raw["symbol"] = raw["ts_code"].str.replace(".SZ", "").str.replace(".SH", "")
            raw["date"] = pd.to_datetime(d)
            for c in ["vol", "ratio"]:
                if c in raw.columns:
                    raw[c] = pd.to_numeric(raw[c], errors="coerce")
            nb_frames.append(raw[["symbol", "date", "vol", "ratio"]])
    except Exception:  # noqa: E722
        pass
    if (i + 1) % 50 == 0:
        logger.info("  northbound: %d/%d", i + 1, min(100, len(all_dates[::5])))
    time.sleep(0.2)
if nb_frames:
    nb = pd.concat(nb_frames, ignore_index=True)
    nb = nb.rename(columns={"vol": "north_hold_vol", "ratio": "north_hold_pct"})
    # Add a net buy proxy (not directly available from hk_hold, use vol as proxy)
    df = df.merge(nb, on=["symbol", "date"], how="left")
    df["north_net_buy"] = df["north_hold_vol"] - df.groupby("symbol")[
        "north_hold_vol"
    ].shift(1)
logger.info("Northbound: %d days", len(nb_frames))

# ====== STEP 6: Fetch LHB (sample dates) ======
logger.info("=== STEP 6: Fetch LHB ===")
lhb_frames = []
for _i, d in enumerate(all_dates[::3][:50]):  # Sample ~50 dates
    dt = d.strftime("%Y%m%d")
    try:
        raw = pro.top_list(trade_date=dt)
        if raw is not None and len(raw) > 0:
            raw["symbol"] = raw["ts_code"].str.replace(".SZ", "").str.replace(".SH", "")
            raw["date"] = pd.to_datetime(d)
            for c in ["net_buy", "buy_amount", "sell_amount"]:
                if c in raw.columns:
                    raw[c] = pd.to_numeric(raw[c], errors="coerce")
            lhb_frames.append(
                raw[["symbol", "date", "net_buy", "buy_amount", "sell_amount"]]
            )
    except Exception:  # noqa: E722
        pass
    time.sleep(0.2)
if lhb_frames:
    lhb_full = pd.concat(lhb_frames, ignore_index=True)
    lhb_full = lhb_full.rename(
        columns={
            "net_buy": "lhb_net_buy",
            "buy_amount": "lhb_buy_amt",
            "sell_amount": "lhb_sell_amt",
        }
    )
    df = df.merge(lhb_full, on=["symbol", "date"], how="left")
logger.info("LHB: %d days", len(lhb_frames))

# ====== Coverage report ======
print("\n=== Data Coverage ===")
for c in [
    "margin_balance",
    "main_money_flow",
    "holder_count",
    "north_hold_vol",
    "lhb_net_buy",
    "roe",
]:
    pct = (1 - df[c].isna().mean()) * 100 if c in df.columns else 0
    print(f"  {c:<25s}: {pct:.1f}%")

# ====== STEP 7: Build features ======
logger.info("=== STEP 7: Build features ===")
df["is_st"] = False
df["is_suspended"] = False
df["list_days"] = df.groupby("symbol").cumcount() + 1
df["limit_pct"] = [get_limit_pct(b, d) for b, d in zip(df["board"], df["date"], strict=False)]

from app.pipeline1.feature_engine_v35 import FeatureEngineV35  # noqa: E402
from app.pipeline1.label_engine import LabelEngine  # noqa: E402

t0 = time.time()
fe = FeatureEngineV35()
df = fe.build(df)
df = LabelEngine.build_path_labels(df)
df = LabelEngine.build_labels(df)
df = LabelEngine.mask_suspension(df)
df = LabelEngine.mask_recent_days(df, days=6)
logger.info("Features: %d cols in %.1fs", len(df.columns), time.time() - t0)

# ====== STEP 8: IC evaluation ======
logger.info("=== STEP 8: IC evaluation ===")
label_1d = "label_1d_net" if "label_1d_net" in df.columns else "label_1d"
label_3d = "label_3d_net" if "label_3d_net" in df.columns else "label_3d"


def rank_ic(df, factor, label):
    sub = df[[factor, "date", label]].dropna()
    if len(sub) < 500:
        return 0.0
    ics = [
        g[[factor, label]].corr(method="spearman").iloc[0, 1]
        for _, g in sub.groupby("date")
        if len(g) >= 10
    ]
    ics = [i for i in ics if not np.isnan(i)]
    return float(np.mean(ics)) if ics else 0.0


dims_features = {
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
    "dim24_margin": [
        "margin_balance_chg_1d",
        "margin_balance_chg_5d",
        "short_balance_ratio",
        "margin_buy_ratio",
        "margin_balance_ma20_dev",
        "margin_balance_yoy",
        "margin_pressure_score",
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
        "ind_margin_chg_5d",
        "ind_margin_accel",
        "ind_holder_trend_20d",
        "ind_north_chg_5d",
        "ind_lhb_net_flow_5d",
        "ind_capital_flow",
    ],
    "dim22_finPIT": [
        "roe_qoq",
        "roa_qoq",
        "margin_chg",
        "growth_accel",
        "profit_accel",
        "debt_leveraging",
        "efficiency_chg",
        "ocf_stability",
        "roe_trend_4q",
        "margin_trend_4q",
        "rev_yoy_trend",
        "quality_momentum",
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
    verdict = "INCLUDE" if best_ic >= 0.02 else ("WATCH" if best_ic >= 0.01 else "SKIP")
    results[dim] = {
        "best_factor": best_f,
        "best_ic": round(best_ic, 5),
        "nan_pct": round(best_nan * 100, 1),
        "verdict": verdict,
    }

print("-" * 100)
print()
print("=== FINAL VERDICT ===")
include_dims = []
for dim, r in results.items():
    tag = ">> INCLUDE <<" if r["verdict"] == "INCLUDE" else ""
    print(
        f"  {dim:<20s}: {r['best_factor']:<30s} IC={r['best_ic']:.5f}  NaN={r['nan_pct']:.1f}%  --> {r['verdict']} {tag}"
    )
    if r["verdict"] == "INCLUDE":
        include_dims.append(dim)

# Save
os.makedirs("data/factor_registry", exist_ok=True)
out_path = f"data/factor_registry/alt_full_eval_{pd.Timestamp.now():%Y%m%d_%H%M}.json"
with open(out_path, "w") as fh:
    json.dump(
        {
            "timestamp": pd.Timestamp.now().isoformat(),
            "include_dims": include_dims,
            "results": results,
        },
        fh,
        ensure_ascii=False,
        indent=2,
    )
logger.info("Saved: %s", out_path)

print(f"\nDim to include in PIPELINE1: {include_dims}")
print(f"Results saved: {out_path}")
