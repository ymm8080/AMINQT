"""_diag_pred_decomp.py — 单股预测逐日剧变的归因分解 (2026-08-06).

问题: 同一只股票, 相邻交易日预测(预期涨幅/达到概率)剧烈变化.
本脚本对 FULL 窗记录 (3y, 逐股历史更密) 量化两大驱动源:
  1. 校准器逐日漂移 (cross-sectional OLS/Platt 随 OOS 滚动窗口每日重拟合 → 同日 score 也会变预测)
  2. score 逐日抖动 (截面分位排名, 同股相邻交易日 score 变化)
以及 per-stock 拟合的实际使用率 (≥PER_STOCK_MIN_N 观测的股票占比).

对每 (board, system, horizon):
  - 模拟生产 OOS 滚动: 在末 ~150 个交易日, 每天用 trailing 130 交易日拟合 cross OLS(score→mfe)
    与 Platt(score→P(mfe≥ABS_TARGET)), 记相邻日 Δslope/Δintercept/Δprob(score=0.85).
  - 校准器单独导致的 Δpred(score 固定 0.85) = |Δslope·0.85 + Δintercept|.
  - score 抖动: 同股相邻交易日 |Δscore|, 映射到 Δpred = |slope_latest·Δscore|.
结果落盘 WORM 到 DATA_OTHERS_DIR/_diag_pred_decomp_<ts>.json (不只 print).

用法: python scripts/_diag_pred_decomp.py
"""

import json
import os
import sys
from datetime import datetime
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from app.pipeline_parallel.config import HORIZONS
from config.settings import DATA_OTHERS_DIR

FULLRUN_DIR = DATA_OTHERS_DIR / "BACKTESTING RESULT" / "20260806_144240"
ABS_TARGET = {"2d": 0.02, "3d": 0.03, "5d": 0.04, "10d": 0.06}
CAL_WINDOW = 130  # 校准窗口 (对齐 PER_STOCK_WINDOW, 生产 OOS≈126 交易日)
CAL_MIN_N = 30  # 拟合最少样本 (对齐 PER_STOCK_MIN_N)
PROBE_SCORE = 0.85  # 入选股代表分位 (打分进入排名的分位区域)
SCORE_DATES = 150  # 模拟滚动校准的末 N 个交易日


def load_full(board: str, sysname: str) -> pd.DataFrame:
    d = pd.read_csv(
        FULLRUN_DIR / f"stocks_{board}_{sysname}_full.csv", dtype={"symbol": str}
    )
    d = d.rename(columns={f"label_mfe_{h}_net": f"mfe_{h}" for h in HORIZONS})
    return d


def rolling_fits(sub: pd.DataFrame, h: str) -> list[dict]:
    """末 SCORE_DATES 个交易日, 逐日 trailing 窗口拟合 cross 校准器 → 漂移序列."""
    sub = sub[["date", "score", f"mfe_{h}"]].dropna()
    dates = np.array(sorted(sub["date"].unique()))
    if len(dates) < CAL_WINDOW + 2:
        return []
    dates = dates[-SCORE_DATES:]
    recs: list[dict] = []
    for i, d in enumerate(dates):
        lo = dates[max(0, i - CAL_WINDOW)]
        win = sub[(sub["date"] >= lo) & (sub["date"] <= d)]
        if len(win) < CAL_MIN_N:
            continue
        x = win[["score"]].to_numpy()
        y = win[f"mfe_{h}"].to_numpy()
        ols = LinearRegression().fit(x, y)
        platt = LogisticRegression().fit(x, (y >= ABS_TARGET[h]).astype(int))
        recs.append(
            {
                "date": str(d),
                "slope": float(ols.coef_[0]),
                "intercept": float(ols.intercept_),
                "prob085": float(platt.predict_proba([[PROBE_SCORE]])[0, 1]),
                "n": int(len(win)),
            }
        )
    return recs


def score_jitter(sub: pd.DataFrame) -> float:
    """同股相邻交易日 |Δscore| 均值 (score 列逐日抖动)."""
    sub = sub[["symbol", "date", "score"]].drop_duplicates().dropna()
    dates = np.array(sorted(sub["date"].unique()))
    diffs: list[float] = []
    for i in range(1, len(dates)):
        s_prev = sub[sub["date"] == dates[i - 1]].set_index("symbol")["score"]
        cur = sub[sub["date"] == dates[i]]
        cur = cur[cur["symbol"].isin(s_prev.index)]
        if cur.empty:
            continue
        diffs.append(float((cur["score"] - cur["symbol"].map(s_prev)).abs().mean()))
    return float(np.mean(diffs)) if diffs else float("nan")


def main() -> int:
    out: dict = {"probe_score": PROBE_SCORE, "cal_window": CAL_WINDOW, "boards": {}}
    for board, sysname in product(("main", "dual"), ("sniper", "fusion")):
        full = load_full(board, sysname)
        key = f"{board}/{sysname}"
        per_stock_n = full.groupby("symbol").size()
        n_used = int((per_stock_n >= CAL_MIN_N).sum())
        out["boards"][key] = {
            "n_stocks": int(full["symbol"].nunique()),
            "n_stocks_ge30obs": n_used,
            "per_stock_fit_usage_pct": round(
                100.0 * n_used / full["symbol"].nunique(), 2
            ),
            "horizons": {},
        }
        sj = score_jitter(full)
        for h in HORIZONS:
            recs = rolling_fits(full, h)
            if not recs:
                out["boards"][key]["horizons"][h] = {"n_days": 0}
                continue
            dsl = np.diff([r["slope"] for r in recs])
            dint = np.diff([r["intercept"] for r in recs])
            dprob = np.diff([r["prob085"] for r in recs])
            slope = recs[-1]["slope"]
            # 校准器单独导致的 Δpred (score 固定 PROBE_SCORE)
            cal_dpred = np.abs(dsl * PROBE_SCORE + dint)
            # score 抖动导致的 Δpred (校准器固定为最新斜率)
            score_dpred = np.abs(slope) * (sj if np.isfinite(sj) else 0.0)
            out["boards"][key]["horizons"][h] = {
                "n_days": len(recs),
                "slope": round(slope, 4),
                "mean_abs_dslope": round(float(np.mean(np.abs(dsl))), 5),
                "mean_abs_dintercept": round(float(np.mean(np.abs(dint))), 5),
                "mean_abs_dprob085": round(float(np.mean(np.abs(dprob))), 5),
                "calibrator_only_abs_dpred": round(float(np.mean(cal_dpred)), 5),
                "score_jitter_abs_dpred": round(float(score_dpred), 5),
                "score_jitter_abs_dscore": round(float(sj), 5)
                if np.isfinite(sj)
                else None,
            }
        print(
            f"[{key}] stocks={full['symbol'].nunique()} per-stock≥30obs={n_used} "
            f"({out['boards'][key]['per_stock_fit_usage_pct']}%) score_jitter_|Δscore|="
            f"{sj:.4f}"
            if np.isfinite(sj)
            else f"[{key}] stocks={full['symbol'].nunique()} per-stock≥30obs={n_used}",
            flush=True,
        )
        for h, v in out["boards"][key]["horizons"].items():
            if v.get("n_days"):
                print(
                    f"  h={h}  |Δslope|={v['mean_abs_dslope']} |Δint|={v['mean_abs_dintercept']} "
                    f"|Δprob@0.85|={v['mean_abs_dprob085']} → calib Δpred={v['calibrator_only_abs_dpred']} "
                    f"vs score Δpred={v['score_jitter_abs_dpred']}",
                    flush=True,
                )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fp = DATA_OTHERS_DIR / f"_diag_pred_decomp_{ts}.json"
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {fp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
