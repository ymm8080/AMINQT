"""接续 30 天 T+3 决策分析: 复用已落盘 raw/smooth, 补 top-N 实得 + alpha 敏感性 (2026-08-07).

原 _diag_t3_decision.py 在 score_w 处崩 (legacy 无 prob_up_10d 列, 已修), 但 raw.parquet /
smooth.parquet / shortlist_daily_ic_3d.csv 已 WORM 落盘. 本脚本加载已落盘产物, 补全:
  2. 全池 top-N 已实现 T+3 (生产 score_w 口径, legacy 只有 2d/3d/5d)
  3. alpha 敏感性 (shortlist 3d IC)
并汇总 IC 判语 → summary.json. 结果 WORM → 同 out_dir/.

用法: python scripts/_diag_t3_finish.py <out_dir>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config.settings import BACKTEST_RESULT_DIR, PANEL_V3_PATH, SHORTLIST_SCORE

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    BACKTEST_RESULT_DIR / "t3_decision_20260807_002008"
)
CLS_THRESHOLD = 0.005
HORIZONS = (1, 2, 3, 5)
HW = SHORTLIST_SCORE["horizon_w"]
GW, PW = SHORTLIST_SCORE["gain_w"], SHORTLIST_SCORE["prob_w"]
ALPHA, SMOOTH_K = 0.35, 12


def ema_series(df: pd.DataFrame, col: str, alpha: float, k: int = SMOOTH_K) -> pd.Series:
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
    pats = ["legacy_stocklist_2026080*.csv", "STOCK LIST 2026080*.xlsx",
            "parallel_shortlist_2026080*.csv", "list_2026080*.parquet"]
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
    return out[["symbol", "date"] + [f"ret_{h}d" for h in HORIZONS] + [f"up_{h}d" for h in HORIZONS]]


def score_w(df: pd.DataFrame) -> pd.DataFrame:
    """生产口径 (legacy 2d/3d/5d): score_w = Σ_h hw[h] × (gw×norm_g + pw×norm_p), 横截面 min-max."""
    out = df.copy()
    for h in ("2d", "3d", "5d"):
        gh, ph = f"norm_g_{h}", f"norm_p_{h}"
        g, p = df[f"pred_ret_{h}"], df[f"prob_up_{h}"]
        glo, ghi = g.min(), g.max()
        plo, phi = p.min(), p.max()
        out[gh] = ((g - glo) / (ghi - glo)).fillna(0.0) if ghi > glo else 0.0
        out[ph] = ((p - plo) / (phi - plo)).fillna(0.0) if phi > plo else 0.0
    out["score_w"] = sum(
        HW[h] * (GW * out[f"norm_g_{h}"] + PW * out[f"norm_p_{h}"]) for h in ("2d", "3d", "5d")
    )
    return out


def main() -> None:
    print(f"[out] {OUT_DIR}")
    raw = pd.read_parquet(OUT_DIR / "raw.parquet")
    smooth = pd.read_parquet(OUT_DIR / "smooth.parquet")
    for df in (raw, smooth):
        df["date"] = pd.to_datetime(df["date"])
    print(f"[load] raw {len(raw):,}r / smooth {len(smooth):,}r")

    shortlist_union = collect_shortlist_union()
    print(f"[universe] 短名单并集 {len(shortlist_union)} 只")

    panel = pd.read_parquet(str(PANEL_V3_PATH))
    panel["date"] = pd.to_datetime(panel["date"])
    all_dates = sorted(panel["date"].unique())
    panel = panel[panel["date"] >= all_dates[-300]].reset_index(drop=True)
    realized = realized_from_panel(panel)
    raw = raw.merge(realized, on=["symbol", "date"], how="left")
    smooth = smooth.merge(realized, on=["symbol", "date"], how="left")

    summary: dict = {
        "out_dir": str(OUT_DIR),
        "shortlist_union_n": len(shortlist_union),
        "module": "legacy main/dual (V35Predictor)",
    }

    # ── IC 汇总判语 (来自已落盘 shortlist_daily_ic_3d.csv) ──
    ic = pd.read_csv(OUT_DIR / "shortlist_daily_ic_3d.csv")
    ic["delta"] = ic["smooth"] - ic["raw"]
    n_neg = int((ic["delta"] < -1e-9).sum())
    n_tot = int(ic["delta"].notna().sum())
    summary["shortlist_ic_3d"] = {
        "mean_delta": float(ic["delta"].mean()),
        "neg_days": n_neg,
        "n_days": n_tot,
        "raw_mean": float(ic["raw"].mean()),
        "smooth_mean": float(ic["smooth"].mean()),
    }
    print(f"[IC] shortlist 3d IC: raw_mean={ic['raw'].mean():+.4f} "
          f"smooth_mean={ic['smooth'].mean():+.4f} delta_mean={ic['delta'].mean():+.4f} "
          f"neg {n_neg}/{n_tot}")

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
                vals.append({"date": _d, "top_n": N, "tag": tag,
                             "ret3d_top": float(top["ret_3d"].mean()), "n": int(len(top))})
            dec_rows.append(pd.DataFrame(vals))
    dec = pd.concat(dec_rows, ignore_index=True)
    dec.to_csv(OUT_DIR / "decision_topn_ret3d.csv", index=False)
    print("\n--- 全池 top-N 已实现 T+3 (生产 score_w 排序, legacy 2d/3d/5d) ---")
    for N in (5, 10, 14, 30):
        sub = dec[dec["top_n"] == N]
        pivot2 = sub.pivot_table(index="date", columns="tag", values="ret3d_top")
        pivot2["delta"] = pivot2["smooth"] - pivot2["raw"]
        print(f"TOP-{N}: raw_mean={pivot2['raw'].mean():+.4f} "
              f"smooth_mean={pivot2['smooth'].mean():+.4f} "
              f"delta={pivot2['delta'].mean():+.4f} "
              f"(delta<0 天数 {int((pivot2['delta'] < 0).sum())}/{len(pivot2)})")
        summary[f"top{N}"] = {
            "raw": float(pivot2["raw"].mean()), "smooth": float(pivot2["smooth"].mean()),
            "delta": float(pivot2["delta"].mean()),
            "neg_days": int((pivot2["delta"] < 0).sum()), "n_days": int(len(pivot2)),
        }

    # ── 3. alpha 敏感性 (shortlist 3d IC) ──
    alpha_rows = []
    for alpha in (ALPHA, 0.5, 0.65, 0.8):
        raw2 = raw.copy()
        raw2["p3"] = ema_series(raw2, "pred_ret_3d", alpha)
        sl = raw2[raw2["symbol"].isin(shortlist_union)]
        vals = []
        for _d, g in sl.dropna(subset=["p3", "ret_3d"]).groupby("date"):
            if len(g) >= 5 and g["ret_3d"].nunique() > 1:
                r = spearmanr(g["p3"], g["ret_3d"])
                if r.statistic == r.statistic:
                    vals.append(r.statistic)
        ic3 = float(np.mean(vals)) if vals else float("nan")
        alpha_rows.append({"alpha": alpha, "shortlist_ic_3d": ic3})
    adf = pd.DataFrame(alpha_rows)
    adf.to_csv(OUT_DIR / "alpha_sensitivity.csv", index=False)
    print("\n--- alpha 敏感性 (shortlist 3d IC) ---")
    print(adf.round(4).to_string())
    summary["alpha_sensitivity"] = adf.to_dict("records")

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[done] → {OUT_DIR}")


if __name__ == "__main__":
    main()
