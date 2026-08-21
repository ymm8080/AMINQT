"""_diag_300911_818_20260820.py — 用 20260820 模块重放 8/18 cutoff, 300911 是否入选 (2026-08-20).

用户问题: 最新模块 (20260820, 08-20 19:39 重训落盘) 下, 300911 在 8/18 cutoff
(缩量洗盘日, 次日 8/19 涨停 +20%) 会不会进股票清单?

口径: 逐字镜像生产链 daily_pipeline.run() 的 legacy 段 (2026-08-20 生效配置):
  panel(≤8/18, 末300交易日) → cleaner.run_inference(pool_blend=True, 全谱)
  → features.build(inference_cols) → V35Predictor(dual_20260820) → pool_blend_cut
  → compute_scores → entry_filter(E7 生产闸: prob_margin dual+0.08 / pain_max dual 0.4)
  → pred_ret_10d 降序排名.
外加: LEGACY_PROB_GATE 概率头 (≤8/20 最新 bundle) 复核 (prob > base_rate + 0.08).

跳过 (非闸门因素): pred_smoothing EMA, holding bonus, D18 env — 不影响过闸判定,
只影响 score 排名微幅.

注意: 20260820 模块训练数据含 8/19 (300911 涨停日) — 诊断口径, 非 PIT 重放.

输出: BACKTEST_RESULT_DIR/_diag_300911_818_20260820_<ts>/report.json + stdout.
用法: python scripts/_diag_300911_818_20260820.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.list_generator import ListGenerator
from app.pipeline1.predictor import V35Predictor
from config.settings import BACKTEST_RESULT_DIR, DATA_DIR, LEGACY_PROB_GATE, PANEL_V3_PATH

TARGET = "300911"
TRADE_DATE = pd.Timestamp("2026-08-18")
BUNDLE_DUAL = "models/pipeline1/dual_20260820.pkl"
TAIL_DAYS = 300  # 与 _gen_legacy_list.py 同款内存切片 (特征等价已验证)

report: dict = {"target": TARGET, "cutoff": str(TRADE_DATE.date()), "bundle": BUNDLE_DUAL}


def log(msg: str) -> None:
    print(msg, flush=True)


def ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9
    except Exception:
        return -1.0


def load_prob_bundle_live_on(model_dir: Path, board: str, as_of: pd.Timestamp):
    """该目录中 trained_through <= as_of 的最新 WORM bundle."""
    import joblib

    cands = sorted(model_dir.glob(f"{board}_prob_*.joblib"))
    picked = None
    for p in cands:
        try:
            b = joblib.load(p)
        except Exception:
            continue
        tt = b.get("trained_through")
        if tt is None:
            continue
        if pd.Timestamp(tt) <= as_of:
            if picked is None or pd.Timestamp(tt) > pd.Timestamp(picked[1]["trained_through"]):
                picked = (p, b)
        del b
    return picked


def main() -> int:
    t0 = time.time()
    log(f"[ram] available {ram_gb():.1f} GB")
    if ram_gb() < 2.5:
        log("[FATAL] 内存 < 2.5GB, 拒绝启动 (重训/其他重活占用?)")
        return 3

    # ---------- 1. 面板: 双创板 ≤ 8/18, 末 300 交易日 ----------
    log("[1] 读面板 (GEM/STAR, ≤8/18, 末300交易日) ...")
    d = ds.dataset(str(PANEL_V3_PATH))
    df = d.to_table(
        filter=(ds.field("board").isin(["GEM", "STAR"])) & (ds.field("date") <= TRADE_DATE),
    ).to_pandas()
    df["symbol"] = df["symbol"].astype(str)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    dates = sorted(df["date"].unique())
    cut = dates[-TAIL_DAYS]
    df = df[df["date"] >= cut].sort_values(["symbol", "date"]).reset_index(drop=True)
    log(f"[1] {len(df):,} 行 {cut.date()}..{dates[-1].date()}")
    report["panel_rows"] = int(len(df))

    # ---------- 2. 清洗 (生产口径, pool_blend=True → 全谱双创) ----------
    cleaner = CleaningPipeline()
    _m, dual, _state = cleaner.run_inference(df, pool_blend=True)
    del df
    gc.collect()
    day = dual[pd.to_datetime(dual["date"]) == TRADE_DATE]
    log(f"[2] 清洗后 dual 全谱 {len(dual):,} 行; 8/18 截面 {len(day):,} 只")
    report["pool_full_spectrum_818"] = int(len(day))
    has911 = TARGET in set(day["symbol"])
    log(f"[2] 8/18 截面含 300911: {has911}")
    report["in_pool_full_spectrum"] = bool(has911)

    # ---------- 3. 特征 + 预测 (20260820 模块) ----------
    predictor = V35Predictor({"dual": BUNDLE_DUAL})
    feat = FeatureEngineV35().build(
        dual, None, inference_cols=predictor.bundles["dual"]["feature_cols"],
        cross_sectional_rank=True,
    )
    log(f"[3] 特征帧 {len(feat):,} 行 × {len(feat.columns)} 列")
    today_feat = feat[feat["date"] == feat["date"].max()].copy()
    pred = predictor.predict(today_feat, "dual")
    log(f"[3] 预测 {len(pred)} 只")
    report["n_pred"] = int(len(pred))

    # ---------- 4. pool_blend_cut (生产链原样调用; 2026-08-20 实证 fail-open) ----------
    has_ls = "liquidity_score" in pred.columns
    log(f"[4] candidates 含 liquidity_score: {has_ls} → blend 切池 {'生效' if has_ls else 'FAIL-OPEN 不切'}")
    report["blend_cut_active"] = bool(has_ls)
    candidates = cleaner.pool_blend_cut(pred)
    report["n_after_blend_cut"] = int(len(candidates))

    # ---------- 5. E7 生产闸 ----------
    lister = ListGenerator()
    scored = lister.compute_scores(candidates)
    passed = lister.entry_filter(scored, market_state="range")
    ranked = passed.sort_values("pred_ret_10d", ascending=False).reset_index(drop=True)
    log(f"[5] E7 过闸 {len(ranked)} 只 (20260820 模块, 8/18 cutoff)")
    report["n_passed"] = int(len(ranked))

    # ---------- 6. 300911 全链路判定 ----------
    r = scored[scored["symbol"] == TARGET]
    if len(r):
        x = r.iloc[0]
        base = x["base_rate"]
        margin = 0.08  # LEGACY_ENTRY_GATE.prob_margin dual
        pain_max = 0.4  # LEGACY_ENTRY_GATE.pain_max dual
        checks = {
            "闸1 compound_prob > base_rate+0.08": (
                float(x["compound_prob"]), float(base + margin),
                bool(x["compound_prob"] > base + margin)),
            "闸2 pred_ret_10d > 0": (float(x["pred_ret_10d"]), 0.0, bool(x["pred_ret_10d"] > 0)),
            "闸3a pred_q50_3d > 0": (float(x["pred_q50_3d"]), 0.0, bool(x["pred_q50_3d"] > 0)),
            "闸3b pred_q50_5d > 0": (float(x["pred_q50_5d"]), 0.0, bool(x["pred_q50_5d"] > 0)),
            "E2  pain_prob <= 0.4": (float(x["pain_prob"]), pain_max, bool(x["pain_prob"] <= pain_max)),
        }
        log(f"\n== 300911 8/18 (20260820 模块) 闸门逐项 ==")
        for k, (v, th, ok) in checks.items():
            log(f"  {k}: {v:+.4f} vs {th:+.4f} → {'PASS' if ok else 'FAIL'}")
        report["gates"] = {k: {"value": v, "threshold": th, "pass": ok} for k, (v, th, ok) in checks.items()}
        in_list = TARGET in set(ranked["symbol"])
        rk = (ranked["symbol"] == TARGET).idxmax() if in_list else None
        log(f"\n  → 300911 最终{'入选' if in_list else '未入选'} 8/18 清单"
            + (f", 排名 {int(rk)+1}/{len(ranked)}" if in_list else ""))
        report["in_list"] = bool(in_list)
        if in_list:
            report["rank"] = int(rk) + 1
        # 对照 top10
        log("  top10 (pred_ret_10d 降序):")
        for i, row in ranked.head(10).iterrows():
            log(f"    {int(i)+1:>2} {row['symbol']}  ret={row['pred_ret_10d']:+.4f}  prob={row['compound_prob']:.3f}")
        report["top10"] = [
            {"rank": int(i) + 1, "symbol": s, "pred_ret_10d": float(p)}
            for i, (s, p) in enumerate(zip(ranked.head(10)["symbol"], ranked.head(10)["pred_ret_10d"]))
        ]
    else:
        log("[6] 300911 不在预测输出 (被清洗层剔除)")
        report["in_list"] = False
        report["gates"] = None

    # ---------- 7. LEGACY_PROB_GATE 概率头复核 (≤8/20 最新 bundle) ----------
    if len(r):
        pb = load_prob_bundle_live_on(Path(DATA_DIR / "prob_head_legacy"), "dual", pd.Timestamp("2026-08-20"))
        if pb is None:
            log("[7] 找不到 ≤8/20 legacy 概率头 bundle → 跳过")
        else:
            log(f"[7] legacy 概率头: {pb[0].name} (trained_through={pb[1]['trained_through']})")
            b = pb[1]
            X = today_feat[b["feat_cols"]].to_numpy(dtype="float32")
            prob = b["model"].predict_proba(X)[:, 1]
            p911 = float(prob[today_feat["symbol"] == TARGET][0])
            gate = base + LEGACY_PROB_GATE["margin"]
            ok = p911 > gate
            log(f"[7] 300911 概率头 prob={p911:.3f} vs 闸={gate:.3f} → {'PASS' if ok else 'FAIL'}")
            report["prob_gate"] = {"prob": p911, "gate": gate, "pass": bool(ok)}
            del X, prob

    # ---------- 8. 对照: 8/18 实际交付清单 ----------
    actual_path = Path(DATA_DIR) / "lists" / "list_20260818.parquet"
    if actual_path.exists():
        act = pd.read_parquet(actual_path)
        act["symbol"] = act["symbol"].astype(str)
        report["actual_818_list"] = {"n": int(len(act)), "symbols": act["symbol"].tolist()}
        ov = set(act["symbol"]) & set(ranked["symbol"]) if len(ranked) else set()
        log(f"[8] 8/18 实际清单 {len(act)} 只 (module 20260818); 与 20260820 模块重放交集 {len(ov)} 只")
        report["overlap_with_actual"] = sorted(ov)
    else:
        log(f"[8] 无 {actual_path} (8/18 实际清单缺失)")

    # ---------- 落盘 ----------
    out_dir = Path(BACKTEST_RESULT_DIR) / ("_diag_300911_818_20260820_" + time.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, default=str)
    log(f"\n[done] WORM -> {out_dir / 'report.json'} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
