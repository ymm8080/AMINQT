# -*- coding: utf-8 -*-
"""Fast evaluation of ALL columns: evaluate raw 107 cols + core dim features
WITHOUT the slow _chgN/_pct_chgN generation. Those are linear transforms.

Evaluates ~500 core features in <2 min (vs ~3,500 in 10+ min).
"""

import json, logging, os, sys, numpy as np, pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PANEL = "data/panel_full_enriched_v3.parquet"
N_STOCKS = 200
SEED = 42
REGISTRY_DIR = "data/factor_registry"


def eval_all_fast(board_name, board_df):
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.label_engine import LabelEngine
    from app.utils.daily_rank_ic import daily_rank_ic_series, mean_rank_ic

    fe = FeatureEngineV35()
    csr = board_name != "main"

    # Latest ~250d
    max_d = board_df["date"].max()
    cutoff = max_d - pd.Timedelta(days=400)
    board_df = board_df[board_df["date"] >= cutoff].copy()

    # --- FAST BUILD: all dims, skip _add_time_series_changes ---
    logger.info("[%s] Building features (fast, no _chgN/_pct_chgN)...", board_name)
    t0 = datetime.now()
    df = board_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df = fe.dim01_price_volume(df)
    df = fe.dim02_volatility(df)
    df = fe.dim03_fundamentals(df)
    df = fe.dim07_limit_gene(df)
    df = fe.dim04_sector_effect(df)
    df = fe.dim05_turnover_liquidity(df)
    df = fe.dim06_valuation_size(df)
    df = fe.dim_active_pit(df)
    df = fe.dim08_calendar_month(df)
    df = fe.dim09_custom_formulas(df)
    df = fe.dim10_money_flow(df)
    df = fe.dim11_float_limits(df)
    df = fe.dim12_ma_system(df)
    df = fe.dim13_holiday(df)
    df = fe.dim14_market_sentiment(df)
    df = fe.dim15_alpha_factors(df)
    df = fe.dim16_candlestick(df)
    df = fe.dim17_extended_factors(df)
    df = fe.dim20_short_horizon(df)
    df = fe.dim18_lhb(df)
    df = fe.dim19_amihud(df)
    df = fe.dim21_chip_tushare(df)
    df = fe.dim22_fundamental_pit(df)
    df = fe.dim23_shareholder_structure(df)
    df = fe.dim24_margin_trading(df)
    df = fe.dim25_northbound(df)
    df = fe.dim26_lhb_enhanced(df)
    df = fe.dim27_industry_flow(df)
    df = fe.dim28_sector_index(df)
    df = fe.dim29_holdertrade(df)
    df = fe.dim30_kline_geometry(df)
    df = fe.industry_neutralize(df)
    df = fe.add_missingness_flags(df)
    # SKIP: _add_time_series_changes (slow, variants of core features)
    # SKIP: _add_cross_sectional_ranks (only for dual anyway)
    df = df.replace([np.inf, -np.inf], np.nan)
    elapsed = (datetime.now() - t0).total_seconds()
    logger.info("[%s] Built %d cols in %.0fs", board_name, len(df.columns), elapsed)

    # Labels + masks
    df = LabelEngine.build_labels(df)
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=6)

    feature_cols = FeatureEngineV35.feature_columns(df)
    feature_cols = [c for c in feature_cols if c in df.columns and df[c].dtype != object]
    feature_cols = [c for c in feature_cols if df[c].isna().mean() < 0.95]

    label_col = "label_1d_net" if "label_1d_net" in df.columns else "label_1d"
    logger.info("[%s] Evaluating %d features vs %s...", board_name, len(feature_cols), label_col)

    results = []
    for i, col in enumerate(feature_cols):
        valid = df[[col, label_col, "date"]].dropna()
        if len(valid) < 50:
            continue
        try:
            ic_mean = mean_rank_ic(valid, col, label_col)
            ic_abs = mean_rank_ic(valid, col, label_col, abs_mean=True)
            ic_series = daily_rank_ic_series(valid, col, label_col)
            ic_std = float(np.nanstd(ic_series.values)) if len(ic_series) >= 5 else 0.0
            icir = ic_mean / ic_std if ic_std > 0 else 0.0
            pos_ratio = float((ic_series > 0).mean())
            nan_rate = float(board_df[col].isna().mean()) if col in board_df.columns else 1.0
            results.append({
                "factor": col, "ic_mean": round(float(ic_mean), 6),
                "ic_abs": round(float(ic_abs), 6), "ic_std": round(float(ic_std), 6),
                "icir": round(float(icir), 4), "pos_ratio": round(float(pos_ratio), 4),
                "nan_rate": round(float(nan_rate), 4), "n_dates": len(ic_series),
            })
        except Exception:
            pass
        if (i + 1) % 200 == 0:
            logger.info("[%s] %d/%d (%d results)", board_name, i+1, len(feature_cols), len(results))

    if not results:
        logger.error("[%s] NO results", board_name)
        return {"board": board_name, "n_features": 0}

    rdf = pd.DataFrame(results).sort_values("ic_abs", ascending=False)
    top_n = 60
    top = rdf.head(top_n)

    print(f"\n{'=' * 100}")
    print(f"  {board_name.upper()} — ALL CORE FEATURES ranked by |IC| (top {top_n})")
    print(f"  {board_df['symbol'].nunique()} stocks, {len(board_df)} rows, {len(feature_cols)} features")
    print(f"{'=' * 100}")
    print(f"{'Rank':<5s} {'Factor':<50s} {'IC_mean':>8s} {'|IC|':>8s} {'ICIR':>7s} {'Pos%':>7s} {'NaN%':>7s}")
    print("-" * 100)
    for idx, (_, r) in enumerate(top.iterrows(), 1):
        print(f"{idx:<5d} {r['factor']:<50s} {r['ic_mean']:>+8.4f} {r['ic_abs']:>8.4f} {r['icir']:>7.2f} {r['pos_ratio']:>7.1%} {r['nan_rate']:>7.1%}")

    strong = (rdf["ic_abs"] >= 0.05).sum()
    weak = ((rdf["ic_abs"] >= 0.02) & (rdf["ic_abs"] < 0.05)).sum()
    noise = (rdf["ic_abs"] < 0.02).sum()
    print(f"\n  Total: {len(rdf)} | Strong(|IC|>=0.05): {strong} | Weak(0.02-0.05): {weak} | Noise(<0.02): {noise}")
    print(f"  IC range: [{rdf['ic_mean'].min():+.4f}, {rdf['ic_mean'].max():+.4f}]")

    return {
        "board": board_name, "n_features": len(rdf),
        "n_strong": int(strong), "n_weak": int(weak), "n_noise": int(noise),
        "ic_range": [float(rdf["ic_mean"].min()), float(rdf["ic_mean"].max())],
        "top_50": rdf.head(50).to_dict(orient="records"),
    }


def main():
    from app.pipeline1.cleaning_pipeline import board_of

    logger.info("Loading panel...")
    panel = pd.read_parquet(PANEL)
    if "board" not in panel.columns:
        panel["board"] = panel["symbol"].map(board_of)

    rng = np.random.RandomState(SEED)
    main_syms = panel[panel["board"]=="main"]["symbol"].unique()
    main_pick = rng.choice(main_syms, min(N_STOCKS, len(main_syms)), replace=False)
    main_df = panel[panel["symbol"].isin(main_pick)].copy()
    main_df = main_df[~main_df["is_st"].astype(bool) & (main_df["list_days"] >= 250)]

    dual_mask = panel["board"].isin(["GEM", "STAR"])
    dual_syms = panel[dual_mask]["symbol"].unique()
    dual_pick = rng.choice(dual_syms, min(N_STOCKS, len(dual_syms)), replace=False)
    dual_df = panel[panel["symbol"].isin(dual_pick)].copy()
    dual_df = dual_df[~dual_df["is_st"].astype(bool) & (dual_df["list_days"] >= 250)]

    logger.info("Main: %d stocks | Dual: %d stocks", main_df["symbol"].nunique(), dual_df["symbol"].nunique())

    output = {"timestamp": datetime.now().isoformat(), "boards": {}}
    for name, bdf in [("main", main_df), ("dual", dual_df)]:
        output["boards"][name] = eval_all_fast(name, bdf)

    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(REGISTRY_DIR, f"feature_eval_fast_{tag}.json")
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
