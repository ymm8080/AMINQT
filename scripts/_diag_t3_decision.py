"""诊断: 平滑 (输出级 EMA) 是否伤及 T+3 决策质量 (2026-08-06).

自包含版: 依赖的 app/pipeline1/pred_smoothing.py 源码已丢失 (未接生产, 仅实验),
故平滑直接用内嵌 ema_series (与生产 EMA 同公式: w_k=α(1-α)^k, 归一化, gap-robust)
对整个 30 日窗口逐股重放, 不依赖丢失模块.

核心问题: 用户定案 T+3 是选股决策唯一视界, "不能承受 T+3 质量变差".
上一轮 _diag_smoothing_backtest.py 显示 shortlist 范围 3d IC 平滑后 0.043→−0.023,
且逐日几乎全负 → 可能是真信号而非噪声. 这里用 30 日窗口裁决:

  1. 全池逐日 (raw | smooth) 推理, 落盘全表.
  2. 按生产 score_w 口径 (SHORTLIST_SCORE 权重, T+3 主视界) 每日排序, 取 top-N,
     对比 raw vs smooth 名单的已实现 3d 收益.
  3. 逐日 shortlist 范围 3d IC (raw vs smooth) → 判定退化是否稳定.
  4. alpha 敏感性: 提高 α (更信任今日) 能否在不牺牲稳定性的同时救回 3d IC.

结果 WORM → {BACKTEST_RESULT_DIR}/t3_decision_<ts>/
用法: python scripts/_diag_t3_decision.py [window_days=30]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.predictor import V35Predictor
from config.settings import BACKTEST_RESULT_DIR, PANEL_V3_PATH, SHORTLIST_SCORE

WINDOW = int(sys.argv[1]) if len(sys.argv) > 1 else 30
CLS_THRESHOLD = 0.005
BUNDLES = {
    "main": "models/pipeline1/main_current.pkl",
    "dual": "models/pipeline1/dual_current.pkl",
}
HORIZONS = (1, 2, 3, 5)
HW = SHORTLIST_SCORE["horizon_w"]  # {"2d":0.25,"3d":0.40,"5d":0.25,"10d":0.10}
GW, PW = SHORTLIST_SCORE["gain_w"], SHORTLIST_SCORE["prob_w"]
ALPHA, SMOOTH_K = 0.35, 12


def ema_series(
    df: pd.DataFrame, col: str, alpha: float, k: int = SMOOTH_K
) -> pd.Series:
    """每股 forecast 列的 EMA 重放: 每个交易日的值 = [当日]+近 k-1 日 raw 的衰减加权均值."""
    w = np.array([alpha * (1 - alpha) ** j for j in range(k)])
    w /= w.sum()
    out = np.empty(len(df))
    out[:] = np.nan
    for sym in df["symbol"].unique():
        idx = df["symbol"] == sym
        sub = df.loc[idx, ["date", col]].sort_values("date")
        for i, (dt, v) in enumerate(zip(sub["date"], sub[col], strict=False)):
            if not np.isfinite(v):
                continue
            prev = sub.loc[sub["date"] < dt, col]
            prev = prev[prev.notna()].tail(k - 1)
            vals = [v] + prev.tolist()
            ww = w[: len(vals)]
            ww /= ww.sum()
            out[sub.index[i]] = float(np.dot(vals, ww))
    return pd.Series(out, index=df.index)


def collect_shortlist_union() -> set[str]:
    syms: set[str] = set()
    roots = {
        "stocklist": Path("D:/AMINQT/DAILY OPERATION/STOCK LIST"),
        "lists": Path("D:/AMINQT/AMINQT CODES/data/lists"),
    }
    pats = [
        "legacy_stocklist_2026080*.csv",
        "STOCK LIST 2026080*.xlsx",
        "parallel_shortlist_2026080*.csv",
        "list_2026080*.parquet",
    ]
    for root in roots.values():
        for pat in pats:
            for fp in sorted(root.glob(pat)):
                try:
                    if fp.suffix == ".csv":
                        df = pd.read_csv(fp, dtype={"symbol": str})
                    elif fp.suffix == ".xlsx":
                        df = pd.read_excel(fp, dtype={"symbol": str})
                    else:
                        df = pd.read_parquet(fp)
                    if "symbol" in df.columns:
                        syms |= {str(s) for s in df["symbol"].dropna().tolist()}
                except Exception as e:  # noqa: BLE001
                    print(f"  [skip] {fp.name}: {e}")
    return syms


def realized_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel[["symbol", "date", "close_hfq"]].sort_values(["symbol", "date"]).copy()
    g = out.groupby("symbol")["close_hfq"]
    for h in HORIZONS:
        out[f"ret_{h}d"] = g.shift(-h) / out["close_hfq"] - 1
        out[f"up_{h}d"] = (out[f"ret_{h}d"] > CLS_THRESHOLD).astype(float)
    return out[
        ["symbol", "date"]
        + [f"ret_{h}d" for h in HORIZONS]
        + [f"up_{h}d" for h in HORIZONS]
    ]


def daily_ic_mean(df: pd.DataFrame, pred: str, real: str, min_n: int = 5) -> float:
    vals = []
    for _d, g in df.dropna(subset=[pred, real]).groupby("date"):
        if len(g) >= min_n and g[real].nunique() > 1:
            r = spearmanr(g[pred], g[real])
            if r.statistic == r.statistic:
                vals.append(r.statistic)
    return float(np.mean(vals)) if vals else float("nan")


def score_w(df: pd.DataFrame) -> pd.DataFrame:
    """生产口径: 每股 score_w = Σ_h hw[h] × (gw×norm_g + pw×norm_p), 横截面 min-max.

    legacy V35Predictor 只有 1d/2d/3d/5d 视界 (无 10d 列), 故只用 2d/3d/5d;
    与生产并行 add_score 相比少乘 0.9 常数, 不影响排名.
    """
    out = df.copy()
    for h in ("2d", "3d", "5d"):
        gh = f"norm_g_{h}"
        ph = f"norm_p_{h}"
        g = df[f"pred_ret_{h}"]
        p = df[f"prob_up_{h}"]
        glo, ghi = g.min(), g.max()
        plo, phi = p.min(), p.max()
        out[gh] = ((g - glo) / (ghi - glo)).fillna(0.0) if ghi > glo else 0.0
        out[ph] = ((p - plo) / (phi - plo)).fillna(0.0) if phi > plo else 0.0
    out["score_w"] = sum(
        HW[h] * (GW * out[f"norm_g_{h}"] + PW * out[f"norm_p_{h}"])
        for h in ("2d", "3d", "5d")
    )
    return out


def main() -> None:
    t0 = time.time()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = BACKTEST_RESULT_DIR / f"t3_decision_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[universe] 收集短名单股票...", flush=True)
    shortlist_union = collect_shortlist_union()
    print(f"[universe] 短名单并集 {len(shortlist_union)} 只", flush=True)

    print("[panel] 加载面板...", flush=True)
    panel = pd.read_parquet(str(PANEL_V3_PATH))
    panel["date"] = pd.to_datetime(panel["date"])
    all_dates = sorted(panel["date"].unique())
    cut = all_dates[-300]
    panel = panel[panel["date"] >= cut].reset_index(drop=True)
    print(
        f"[panel] {len(panel):,}r  {cut.date()}..{all_dates[-1].date()} ({time.time() - t0:.0f}s)",
        flush=True,
    )

    cleaner = CleaningPipeline()
    main_df, dual_df, _valve = cleaner.run_inference(panel)
    predictor = V35Predictor(BUNDLES)
    feats: dict[str, pd.DataFrame] = {}
    for board, df in (("main", main_df), ("dual", dual_df)):
        if len(df) == 0 or board not in predictor.bundles:
            continue
        cols = predictor.bundles[board]["feature_cols"]
        feats[board] = FeatureEngineV35().build(
            df, None, inference_cols=cols, cross_sectional_rank=(board == "dual")
        )
        print(
            f"[feat] {board}: {len(feats[board]):,}r ({time.time() - t0:.0f}s)",
            flush=True,
        )

    days = [d for d in all_dates if d >= all_dates[-WINDOW]]
    raw_frames = []
    for D in days:
        frames = []
        for board, feat in feats.items():
            fD = feat[feat["date"] == D]
            if len(fD) == 0:
                continue
            fD = fD.sort_values("date").groupby("symbol").tail(1)
            frames.append(predictor.predict(fD, board))
        if not frames:
            continue
        day_raw = pd.concat(frames, ignore_index=True)
        day_raw["date"] = D
        raw_frames.append(day_raw)
    raw = pd.concat(raw_frames, ignore_index=True)
    print(
        f"[raw] {len(raw):,} 预测行 ({len(raw['date'].unique())} 交易日) ({time.time() - t0:.0f}s)",
        flush=True,
    )

    # ── 平滑: 内嵌 EMA (与生产同公式) 逐 forecast 列重放 ──
    from app.pipeline1.model_meta import load_modules, module_id

    mod = module_id(load_modules())
    forecast_cols = [
        c
        for c in raw.columns
        if c.startswith("pred_ret_")
        or c.startswith("prob_up")
        or c.startswith("pred_q50")
    ]
    smooth = raw.copy()
    for col in forecast_cols:
        smooth[col] = ema_series(raw, col, ALPHA, SMOOTH_K)
    print(
        f"[smooth] 完成, {len(forecast_cols)} 列 ({time.time() - t0:.0f}s)", flush=True
    )

    # 落盘全表 (WORM)
    raw.to_parquet(out_dir / "raw.parquet")
    smooth.to_parquet(out_dir / "smooth.parquet")

    realized = realized_from_panel(panel)
    raw = raw.merge(realized, on=["symbol", "date"], how="left")
    smooth = smooth.merge(realized, on=["symbol", "date"], how="left")

    summary: dict = {
        "ts": ts,
        "window_days": WINDOW,
        "module": mod,
        "n_days": len(days),
        "shortlist_union_n": len(shortlist_union),
    }

    # ── 1. 逐日 shortlist 范围 3d IC (退化是否稳定) ──
    ic_rows = []
    for tag, df in (("raw", raw), ("smooth", smooth)):
        sl = df[df["symbol"].isin(shortlist_union)].dropna(
            subset=["pred_ret_3d", "ret_3d"]
        )
        for _d, g in sl.groupby("date"):
            if len(g) < 5 or g["ret_3d"].nunique() < 2:
                continue
            r = spearmanr(g["pred_ret_3d"], g["ret_3d"])
            ic_rows.append(
                {
                    "tag": tag,
                    "date": _d,
                    "n": len(g),
                    "ic_3d": r.statistic if r.statistic == r.statistic else np.nan,
                }
            )
    ic_df = pd.DataFrame(ic_rows)
    pivot = ic_df.pivot_table(index="date", columns="tag", values="ic_3d")
    pivot["delta"] = pivot.get("smooth", np.nan) - pivot.get("raw", np.nan)
    pivot = pivot.sort_index()
    pivot.to_csv(out_dir / "shortlist_daily_ic_3d.csv")
    n_neg = int((pivot["delta"] < -1e-9).sum())
    n_tot = int(pivot["delta"].notna().sum())
    print(f"[IC] shortlist 3d IC: 逐日 delta<0 天数 {n_neg}/{n_tot}")
    print(pivot.round(4).to_string())

    # ── 2. 全池 top-N 决策对比 (生产 score_w, 按已实现 3d) ──
    dec_rows = []
    for N in (5, 10, 14, 30):
        for tag, df in (("raw", raw), ("smooth", smooth)):
            s = score_w(df).dropna(subset=["score_w", "ret_3d"])
            vals = []
            for _d, g in s.groupby("date"):
                if len(g) < max(N, 5):
                    continue
                top = g.nlargest(N, "score_w")
                vals.append(
                    {
                        "date": _d,
                        "top_n": N,
                        "tag": tag,
                        "ret3d_top": float(top["ret_3d"].mean()),
                        "n": int(len(top)),
                    }
                )
            ddf = pd.DataFrame(vals)
            dec_rows.append(ddf)
    dec = pd.concat(dec_rows, ignore_index=True)
    dec.to_csv(out_dir / "decision_topn_ret3d.csv")
    print("\n--- 全池 top-N 已实现 T+3 (生产 score_w 排序) ---")
    for N in (5, 10, 14, 30):
        sub = dec[dec["top_n"] == N]
        pivot2 = sub.pivot_table(index="date", columns="tag", values="ret3d_top")
        pivot2["delta"] = pivot2["smooth"] - pivot2["raw"]
        print(
            f"TOP-{N}: raw_mean={pivot2['raw'].mean():+.4f} "
            f"smooth_mean={pivot2['smooth'].mean():+.4f} "
            f"delta={pivot2['delta'].mean():+.4f} "
            f"(delta<0 天数 {int((pivot2['delta'] < 0).sum())}/{len(pivot2)})"
        )
        summary[f"top{N}"] = {
            "raw": float(pivot2["raw"].mean()),
            "smooth": float(pivot2["smooth"].mean()),
            "delta": float(pivot2["delta"].mean()),
            "neg_days": int((pivot2["delta"] < 0).sum()),
            "n_days": int(len(pivot2)),
        }

    # ── 3. alpha 敏感性 (shortlist 3d IC) ──
    alpha_rows = []
    for alpha in (ALPHA, 0.5, 0.65, 0.8):
        raw2 = raw.copy()
        raw2["p3"] = ema_series(raw2, "pred_ret_3d", alpha)
        sl = raw2[raw2["symbol"].isin(shortlist_union)]
        ic = daily_ic_mean(sl, "p3", "ret_3d")
        alpha_rows.append({"alpha": alpha, "shortlist_ic_3d": ic})
    adf = pd.DataFrame(alpha_rows)
    adf.to_csv(out_dir / "alpha_sensitivity.csv")
    print("\n--- alpha 敏感性 (shortlist 3d IC) ---")
    print(adf.round(4).to_string())
    summary["alpha_sensitivity"] = adf.to_dict("records")

    summary["shortlist_daily_ic_3d_neg"] = f"{n_neg}/{n_tot}"
    summary["out_dir"] = str(out_dir)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[done] {time.time() - t0:.0f}s → {out_dir}")


if __name__ == "__main__":
    main()
