# -*- coding: utf-8 -*-
"""Dual board 1-year train + predict (compact, no data fetch).

- Panel: last 250 trading days from panel_full_enriched_v3.parquet
- Training window patched to 250 days (1 calendar year ≈ 250 trading days)
- Prediction: today's candidates → data/lists/_list_dual_1y_{date}.parquet
"""
from __future__ import annotations

import logging, os, sys, time, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dual_1y")

import pandas as pd
import numpy as np

# ── Config ──────────────────────────────────────────
PANEL_PATH = "data/panel_full_enriched_v3.parquet"
MODEL_DIR = "models/pipeline1"
REGISTRY_PATH = "data/factor_registry"
LOOKBACK_DAYS = 250  # 1 calendar year ≈ 250 A-share trading days
MASK_RECENT_DAYS = 6
BOARD = "dual"

# ── Load & trim panel to 1 year ────────────────────
t_total = time.time()
logger.info("Loading panel: %s", PANEL_PATH)
panel = pd.read_parquet(PANEL_PATH)
dates = sorted(panel["date"].unique())
panel_1y = panel[panel["date"].isin(dates[-LOOKBACK_DAYS:])].copy()
logger.info(
    "Trimmed to %d days (%s → %s), %d stocks, %d rows",
    panel_1y["date"].nunique(),
    dates[-LOOKBACK_DAYS:][0].date() if hasattr(dates[-LOOKBACK_DAYS:][0], "date") else str(dates[-LOOKBACK_DAYS:][0])[:10],
    dates[-1].date() if hasattr(dates[-1], "date") else str(dates[-1])[:10],
    panel_1y["symbol"].nunique(),
    len(panel_1y),
)
del panel, dates

# ── Patch trainer window to 1 year ──────────────────
import app.pipeline1.dual_track_trainer as dtt
_orig_total = dtt.WINDOW_TOTAL
_orig_transition = dtt.WINDOW_TRANSITION
dtt.WINDOW_TOTAL = LOOKBACK_DAYS
dtt.WINDOW_TRANSITION = LOOKBACK_DAYS
logger.info("Patched WINDOW_TOTAL=%d WINDOW_TRANSITION=%d (orig: %d/%d)",
            LOOKBACK_DAYS, LOOKBACK_DAYS, _orig_total, _orig_transition)

# ── Clean → dual only ──────────────────────────────
from app.pipeline1.cleaning_pipeline import CleaningPipeline

t0 = time.time()
cleaner = CleaningPipeline()
main_df, dual_df = cleaner.run_train(panel_1y, board=BOARD)
logger.info("Cleaning: dual=%d rows (%.1fs)", len(dual_df), time.time() - t0)
del main_df, panel_1y

if len(dual_df) == 0:
    logger.error("No dual board samples after cleaning")
    raise SystemExit(1)

# ── Features + Labels ──────────────────────────────
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import LabelEngine

t0 = time.time()
features = FeatureEngineV35()
df = features.build(dual_df, cross_sectional_rank=True, registry=None)
logger.info("Features: %d cols (%.1fs)", len(df.columns), time.time() - t0)

t0 = time.time()
df = LabelEngine.build_path_labels(df)
df = LabelEngine.build_labels(df)
df = LabelEngine.mask_suspension(df)
df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)
logger.info("Labels: %d rows after masking (%.1fs)", len(df), time.time() - t0)

# ── Feature Selection ──────────────────────────────
from app.pipeline1.feature_selector import FeatureSelector, BruteForceGenerator

t0 = time.time()
try:
    selector = FeatureSelector(registry_dir=REGISTRY_PATH)
    logger.info("FeatureSelector: %s", selector.config.get(BOARD, {}).get("pipeline", "?"))
    selected = selector.select(df, BOARD)
    missing = [f for f in selected if f not in df.columns]
    if missing:
        gen = BruteForceGenerator()
        raw_cols = gen._eligible(df)
        new_feats = gen.generate(df, raw_cols=raw_cols)
        keep_cols = [c for c in missing if c in new_feats.columns]
        if keep_cols:
            df = df.join(new_feats[keep_cols])
    picked = [f for f in selected if f in df.columns]
    logger.info("FeatureSelection: %d/%d selected (%.0fs)", len(picked), len(selected), time.time() - t0)
except Exception as exc:
    logger.warning("FeatureSelector failed (%s), using all features", exc)
    picked = FeatureEngineV35.feature_columns(df)

# ── Train ──────────────────────────────────────────
from app.pipeline1.dual_track_trainer import DualTrackTrainer

tag = time.strftime("%GW%V") + "_1y"
logger.info("Training dual board: tag=%s features=%d rows=%d", tag, len(picked), len(df))

t0 = time.time()
trainer = DualTrackTrainer(model_dir=MODEL_DIR)
results = trainer.weekly_retrain({BOARD: df}, {BOARD: picked}, tag)
res = results.get(BOARD, {})
oos_1d = res.get("oos", {}).get("ics", {}).get("1d_reg", 0.0)
logger.info("DONE [%s] OOS_IC(1d)=%.4f switched=%s path=%s (train %.0fs)",
            BOARD, oos_1d, res.get("switched", False),
            res.get("path", "?"), time.time() - t0)

trained_path = res.get("path", "")

# ── Restore trainer window ─────────────────────────
dtt.WINDOW_TOTAL = _orig_total
dtt.WINDOW_TRANSITION = _orig_transition

# ── Predict ────────────────────────────────────────
if not trained_path or not os.path.exists(trained_path):
    logger.error("No trained model found at %s", trained_path)
    raise SystemExit(1)

logger.info("Loading model: %s", trained_path)
bundle = DualTrackTrainer.load(trained_path)
logger.info("Model features: %d", len(bundle.get("feature_cols", [])))

# Reload panel for inference (250-day lookback)
panel_full = pd.read_parquet(PANEL_PATH)
all_dates = sorted(panel_full["date"].unique())
panel_infer = panel_full[panel_full["date"].isin(all_dates[-LOOKBACK_DAYS:])].copy()
del panel_full, all_dates

t0 = time.time()
cleaner = CleaningPipeline()
main_df, dual_df, valve = cleaner.run_inference(panel_infer)
logger.info("Inference cleaning: main=%d dual=%d valve=%s (%.1fs)",
            len(main_df), len(dual_df), valve, time.time() - t0)

if valve == "empty" or len(dual_df) == 0:
    logger.error("No data after inference cleaning")
    raise SystemExit(1)

# Features for inference
t0 = time.time()
engine = FeatureEngineV35()
feat_dual = engine.build(dual_df, None, cross_sectional_rank=False,
                         inference_cols=bundle["feature_cols"])
logger.info("Inference features: %s (%.1fs)", feat_dual.shape, time.time() - t0)

# Today's stocks
latest_date = dual_df["date"].max()
today_syms = set(dual_df[dual_df["date"] == latest_date]["symbol"])
feat_today = feat_dual[feat_dual["symbol"].isin(today_syms)]
logger.info("Today (%s): %d stocks", latest_date.date(), len(feat_today))

if len(feat_today) == 0:
    logger.error("No stocks for prediction")
    raise SystemExit(1)

# Predict
from app.pipeline1.predictor import V35Predictor

predictor = V35Predictor({BOARD: trained_path})
preds = predictor.predict(feat_today, BOARD)
preds["board"] = BOARD
logger.info("Predictions: %d candidates", len(preds))

# List generation
from app.pipeline1.list_generator import ListGenerator

result = ListGenerator().emit(preds, env=None, market_state="range")
logger.info("List: mode=%s empty=%s cap=%s",
            result.get("mode"), result.get("empty"), result.get("cap_position", "N/A"))

if not result.get("empty") and len(result.get("list", pd.DataFrame())):
    lst = result["list"]
    trade_date = latest_date.strftime("%Y%m%d") if hasattr(latest_date, "strftime") else str(latest_date)[:10].replace("-", "")
    out_path = f"data/lists/_list_dual_1y_{trade_date}.parquet"
    os.makedirs("data/lists", exist_ok=True)
    lst.to_parquet(out_path, index=False)

    show_cols = ["symbol", "board", "composite_score", "pred_ret_1d", "pred_ret_3d", "pred_ret_5d", "prob_up"]
    avail = [c for c in show_cols if c in lst.columns]
    top = lst.nlargest(min(25, len(lst)), "composite_score")
    print(f"\n=== TOP 25 DUAL 1Y PICKS ({len(lst)} total) ===")
    print(top[avail].to_string())
    if "board" in lst.columns:
        print(f"\nBoard distribution:\n{lst['board'].value_counts().to_string()}")
    if "prob_up" in lst.columns:
        print(f"\nProb_up stats:\n{lst['prob_up'].describe().to_string()}")
    print(f"\nSaved: {out_path}")
else:
    print("\n=== EMPTY LIST ===")

logger.info("Total: %.0fs", time.time() - t_total)
