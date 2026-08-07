"""诊断: T+3 决策质量 vs 输出级 EMA 平滑 α 敏感性 (2026-08-06).

输入: _diag_t3_decision.py 落盘的 raw.parquet / smooth.parquet (最新 t3_decision_* 目录).
做法 (与生产完全同路径):
  1. 校验: ema_series(raw, pred_ret_3d, α=0.35) vs 生产 smooth_preds 输出的 pred_ret_3d,
     应逐格相等 (同一 EMA 公式). 若偏差>1e-9 → 校验失败, 禁止下结论.
  2. α 扫描: 对每个 α, monkeypatch pred_smoothing.ALPHA → 逐日 persist_raw_preds + smooth_preds
     (生产同语义), 得到该 α 的平滑全表.
  3. 指标 (每个 α):
       - shortlist 范围 3d IC (逐日 spearman 均值)   ← T+3 主视界
       - full 范围 3d IC (参考, 全市场受益度)
       - 稳定度: 相邻日 |Δ| pred_ret_3d 均值 (越低越稳)
       - 方向: 已实现 ret_3d>0.5% 占比下, smooth 排名 top-N 命中 (可选简化)
  4. 结论规则: 选 α 使 (shortlist 3d IC 不劣于 raw) 且 (稳定度不劣于 α=0.35).

结果 WORM → {BACKTEST_RESULT_DIR}/t3_alpha_sens_<ts>/
用法: python scripts/_diag_t3_alpha_sens.py [alpha_csv]
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config.settings import BACKTEST_RESULT_DIR, PANEL_V3_PATH

CLS_THRESHOLD = 0.005
HORIZONS = (3,)
ALPHAS = (0.35, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8)


def newest_t3_dir() -> Path:
    cands = sorted(BACKTEST_RESULT_DIR.glob("t3_decision_*"))
    if not cands:
        raise SystemExit("no t3_decision_* dir found")
    return cands[-1]


def load_shortlist_union() -> set[str]:
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


def ema_series(df: pd.DataFrame, col: str, alpha: float, k: int = 12) -> pd.Series:
    """校验用重放: 与生产 smooth_preds 同 EMA 公式 (today 取 w0, 最多 K-1 个旧日)."""
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


def daily_ic(df: pd.DataFrame, pred: str, real: str, min_n: int = 5) -> float:
    vals = []
    for _d, g in df.dropna(subset=[pred, real]).groupby("date"):
        if len(g) >= min_n and g[real].nunique() > 1:
            r = spearmanr(g[pred], g[real])
            if r.statistic == r.statistic:
                vals.append(r.statistic)
    return float(np.mean(vals)) if vals else float("nan")


def mean_abs_delta(df: pd.DataFrame, col: str) -> float:
    d = df[["symbol", "date", col]].dropna().sort_values(["symbol", "date"])
    d["prev"] = d.groupby("symbol")[col].shift(1)
    d = d.dropna(subset=["prev"])
    return float((d[col] - d["prev"]).abs().mean()) if len(d) else float("nan")


def realized_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel[["symbol", "date", "close_hfq"]].sort_values(["symbol", "date"]).copy()
    g = out.groupby("symbol")["close_hfq"]
    out["ret_3d"] = g.shift(-3) / out["close_hfq"] - 1
    return out[["symbol", "date", "ret_3d"]]


def main() -> None:
    t0 = time.time()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = BACKTEST_RESULT_DIR / f"t3_alpha_sens_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    src = newest_t3_dir()
    summary_json = json.loads((src / "summary.json").read_text(encoding="utf-8"))
    mod = summary_json["module"]
    raw = pd.read_parquet(src / "raw.parquet")
    smooth = pd.read_parquet(src / "smooth.parquet")
    raw["date"] = pd.to_datetime(raw["date"])
    smooth["date"] = pd.to_datetime(smooth["date"])
    days = sorted(raw["date"].unique())
    print(f"[src] {src.name} | module={mod} | {len(days)} 交易日 | raw {len(raw):,}r", flush=True)

    shortlist_union = load_shortlist_union()
    print(f"[universe] 短名单并集 {len(shortlist_union)} 只", flush=True)

    print("[panel] 加载面板取已实现 ret_3d...", flush=True)
    panel = pd.read_parquet(str(PANEL_V3_PATH))
    panel["date"] = pd.to_datetime(panel["date"])
    realized = realized_from_panel(panel)
    raw = raw.merge(realized, on=["symbol", "date"], how="left")
    smooth = smooth.merge(realized, on=["symbol", "date"], how="left")

    # ── 1. 校验 ema_series vs 生产 smooth_preds ──
    probe = raw[["symbol", "date", "pred_ret_3d", "prob_up_3d"]].copy()
    replay = ema_series(probe, "pred_ret_3d", 0.35)
    prod = smooth.set_index(["symbol", "date"])["pred_ret_3d"]
    merged = pd.concat([replay.rename("replay"), raw[["pred_ret_3d"]]], axis=1)
    merged["prod"] = prod.reindex(raw.set_index(["symbol", "date"]).index).values
    valid = merged.dropna(subset=["prod"])
    diff = (valid["replay"] - valid["prod"]).abs().max()
    print(f"[validate] ema_series(α=0.35) vs 生产 smooth_preds: max|Δ|={diff:.3e} "
          f"(n={len(valid):,})", flush=True)
    if diff > 1e-9:
        print("[validate] FAIL → ema_series 与生产不一致, 禁止下结论", flush=True)
        (out_dir / "VALIDATION_FAIL.txt").write_text(
            f"max|Δ|={diff}\nproduction smooth dir={src}\n", encoding="utf-8"
        )
        return

    # ── 2. α 扫描 (生产路径) ──
    from app.pipeline1 import pred_smoothing

    orig_alpha = pred_smoothing.ALPHA
    rows = []
    for alpha in ALPHAS:
        pred_smoothing.ALPHA = alpha
        tmpdir = tempfile.mkdtemp(prefix="alpha_bt_")
        pred_smoothing.STOCK_LIST_DIR = Path(tmpdir)
        sm_frames = []
        for D in days:
            day_raw = raw[raw["date"] == D]
            if len(day_raw) == 0:
                continue
            dstr = D.strftime("%Y%m%d")
            pred_smoothing.persist_raw_preds(day_raw, dstr, mod)
            sm = pred_smoothing.smooth_preds(day_raw, dstr, mod)
            sm["date"] = D
            sm_frames.append(sm)
        pred_smoothing.ALPHA = orig_alpha
        sm = pd.concat(sm_frames, ignore_index=True)
        sm = sm.merge(realized, on=["symbol", "date"], how="left")

        sl = sm[sm["symbol"].isin(shortlist_union)]
        ic_sl = daily_ic(sl, "pred_ret_3d", "ret_3d")
        ic_full = daily_ic(sm, "pred_ret_3d", "ret_3d")
        stab = mean_abs_delta(sm, "pred_ret_3d")
        rows.append({"alpha": alpha, "shortlist_ic_3d": ic_sl,
                     "full_ic_3d": ic_full, "stab_mean_abs_delta": stab})
        print(f"[α={alpha:.2f}] shortlist 3d IC={ic_sl:+.4f} | full 3d IC={ic_full:+.4f} "
              f"| 稳定度 mean|Δ|={stab:.5f}", flush=True)

    adf = pd.DataFrame(rows)
    raw_ic_sl = daily_ic(raw[raw["symbol"].isin(shortlist_union)], "pred_ret_3d", "ret_3d")
    raw_stab = mean_abs_delta(raw, "pred_ret_3d")
    adf.to_csv(out_dir / "alpha_sensitivity_prod.csv", index=False)
    summary = {
        "ts": ts, "source_dir": str(src), "module": mod,
        "validation_max_abs_delta": float(diff), "validation_n": int(len(valid)),
        "raw_shortlist_ic_3d": raw_ic_sl, "raw_stab_mean_abs_delta": raw_stab,
        "alpha_table": adf.to_dict("records"),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n--- α 敏感性汇总 (raw 基准: shortlist 3d IC={raw_ic_sl:+.4f}, "
          f"stab={raw_stab:.5f}) ---")
    print(adf.round(4).to_string())
    print(f"\n[done] {time.time()-t0:.0f}s → {out_dir}")


if __name__ == "__main__":
    main()
