"""2×2 平滑评估: 全市场/短名单 × 一个月/一周, raw vs smooth (2026-08-07 用户).

用户: "I NEED EVALUATION ON ALL STOCK AND STOCK LIST FOR ONE MONTH AND ONE WEEK".
复用 t3_decision_20260807_002008 已落盘 raw/smooth.parquet (30 交易日全候选预测, 06-26→08-06),
不重跑特征构建. 指标:
  1. 逐日 IC (spearman pred_ret_h vs 已实现 ret_h), 均值: scope(全池|短名单) × period(月|周) × horizon(1/2/3/5d) × tag(raw|smooth)
  2. top-N 已实现收益 (score_w 排序, legacy 2d/3d/5d), scope × period × N(5/10/14) × tag, 主看 3d
周期口径: 月=全部 30 交易日; 周=该视界有有效已实现收益的最近 5 个交易日.
结果 WORM → BACKTEST_RESULT_DIR/t3_eval_2x2_<ts>/
用法: python scripts/_diag_eval_2x2.py [src_dir]
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

from config.settings import BACKTEST_RESULT_DIR, PANEL_V3_PATH, SHORTLIST_SCORE

SRC_DIR = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else (BACKTEST_RESULT_DIR / "t3_decision_20260807_002008")
)
CLS_THRESHOLD = 0.005
HORIZONS = (1, 2, 3, 5)
HORIZON_LABEL = {1: "1d", 2: "2d", 3: "3d", 5: "5d"}
HW = SHORTLIST_SCORE["horizon_w"]
GW, PW = SHORTLIST_SCORE["gain_w"], SHORTLIST_SCORE["prob_w"]


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
    return out[["symbol", "date"] + [f"ret_{h}d" for h in HORIZONS]]


def score_w(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for h in ("2d", "3d", "5d"):
        gh, ph = f"norm_g_{h}", f"norm_p_{h}"
        g, p = df[f"pred_ret_{h}"], df[f"prob_up_{h}"]
        glo, ghi = g.min(), g.max()
        plo, phi = p.min(), p.max()
        out[gh] = ((g - glo) / (ghi - glo)).fillna(0.0) if ghi > glo else 0.0
        out[ph] = ((p - plo) / (phi - plo)).fillna(0.0) if phi > plo else 0.0
    out["score_w"] = sum(
        HW[h] * (GW * out[f"norm_g_{h}"] + PW * out[f"norm_p_{h}"])
        for h in ("2d", "3d", "5d")
    )
    return out


def daily_ic_series(df: pd.DataFrame, pred: str, real: str, min_n: int = 5) -> dict:
    vals = {}
    for _d, g in df.dropna(subset=[pred, real]).groupby("date"):
        if len(g) >= min_n and g[real].nunique() > 1:
            r = spearmanr(g[pred], g[real])
            if r.statistic == r.statistic:
                vals[_d] = r.statistic
    return vals


def mean_of(vals: dict, last_n: int | None) -> tuple[float, int]:
    if not vals:
        return float("nan"), 0
    if last_n is not None:
        dates = sorted(vals)[-last_n:]
        vals = {d: vals[d] for d in dates}
    return float(np.mean(list(vals.values()))), len(vals)


def main() -> None:
    t0 = time.time()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = BACKTEST_RESULT_DIR / f"t3_eval_2x2_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(SRC_DIR / "raw.parquet")
    smooth = pd.read_parquet(SRC_DIR / "smooth.parquet")
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

    scopes = {"all": "全市场", "list": "短名单"}
    periods = {"month": "一个月(30日)", "week": "一周(5日)"}
    summary: dict = {"src": str(SRC_DIR), "shortlist_union_n": len(shortlist_union)}

    # ── 1. 逐日 IC ──
    ic_rows = []
    for scope_key, scope_label in scopes.items():
        for tag, df in (("raw", raw), ("smooth", smooth)):
            base = df if scope_key == "all" else df[df["symbol"].isin(shortlist_union)]
            for h in HORIZONS:
                lab = HORIZON_LABEL[h]
                icd = daily_ic_series(base, f"pred_ret_{lab}", f"ret_{h}d")
                for period, pn in (("month", None), ("week", 5)):
                    m, n = mean_of(icd, pn)
                    ic_rows.append(
                        {
                            "scope": scope_key,
                            "scope_label": scope_label,
                            "period": period,
                            "horizon": lab,
                            "tag": tag,
                            "ic_mean": m,
                            "n_days": n,
                        }
                    )
    ic_df = pd.DataFrame(ic_rows)
    ic_df.to_csv(out_dir / "ic_2x2.csv", index=False)
    print("\n========== 逐日 IC 均值 (raw→smooth) ==========")
    for scope_key in scopes:
        for period in periods:
            print(f"\n[{scopes[scope_key]} / {periods[period]}]")
            sub = ic_df[(ic_df["scope"] == scope_key) & (ic_df["period"] == period)]
            piv = sub.pivot_table(index="horizon", columns="tag", values="ic_mean")
            piv["Δ"] = piv["smooth"] - piv["raw"]
            piv["更差天数"] = ""
            for h in piv.index:
                r0 = ic_df[
                    (ic_df.scope == scope_key)
                    & (ic_df.period == period)
                    & (ic_df.horizon == h)
                    & (ic_df.tag == "raw")
                ]["n_days"].iloc[0]
                piv.loc[h, "更差天数"] = "-"
            print(piv.round(4).to_string())

    # ── 2. top-N 已实现 3d (score_w 排序) ──
    topn_rows = []
    for scope_key, scope_label in scopes.items():
        for tag, df in (("raw", raw), ("smooth", smooth)):
            base = df if scope_key == "all" else df[df["symbol"].isin(shortlist_union)]
            s = score_w(base)
            for N in (5, 10, 14):
                by_day = {}
                for _d, g in s.dropna(subset=["score_w", "ret_3d"]).groupby("date"):
                    if len(g) < max(N, 5):
                        continue
                    by_day[_d] = float(g.nlargest(N, "score_w")["ret_3d"].mean())
                for period, pn in (("month", None), ("week", 5)):
                    m, n = mean_of(by_day, pn)
                    topn_rows.append(
                        {
                            "scope": scope_key,
                            "scope_label": scope_label,
                            "period": period,
                            "top_n": N,
                            "tag": tag,
                            "ret3d_mean": m,
                            "n_days": n,
                        }
                    )
    topn_df = pd.DataFrame(topn_rows)
    topn_df.to_csv(out_dir / "topn_ret3d_2x2.csv", index=False)
    print("\n========== top-N 已实现 3d 收益均值 (raw→smooth) ==========")
    for scope_key in scopes:
        for period in periods:
            print(f"\n[{scopes[scope_key]} / {periods[period]}]")
            sub = topn_df[
                (topn_df["scope"] == scope_key) & (topn_df["period"] == period)
            ]
            piv = sub.pivot_table(index="top_n", columns="tag", values="ret3d_mean")
            piv["Δ"] = piv["smooth"] - piv["raw"]
            print(piv.round(4).to_string())

    # ── 落盘 summary.json (WORM) ──
    ic_piv = {}
    for scope_key in scopes:
        for period in periods:
            sub = ic_df[(ic_df["scope"] == scope_key) & (ic_df["period"] == period)]
            for h in ("1d", "2d", "3d", "5d"):
                row = sub[sub["horizon"] == h]
                if len(row):
                    r0 = row[row["tag"] == "raw"]["ic_mean"].iloc[0]
                    s0 = row[row["tag"] == "smooth"]["ic_mean"].iloc[0]
                    ic_piv[f"{scope_key}/{period}/{h}"] = {
                        "raw": float(r0),
                        "smooth": float(s0),
                        "delta": float(s0 - r0),
                    }
    tn_piv = {}
    for scope_key in scopes:
        for period in periods:
            sub = topn_df[
                (topn_df["scope"] == scope_key) & (topn_df["period"] == period)
            ]
            for N in (5, 10, 14):
                row = sub[sub["top_n"] == N]
                if len(row):
                    r0 = row[row["tag"] == "raw"]["ret3d_mean"].iloc[0]
                    s0 = row[row["tag"] == "smooth"]["ret3d_mean"].iloc[0]
                    tn_piv[f"{scope_key}/{period}/top{N}"] = {
                        "raw": float(r0),
                        "smooth": float(s0),
                        "delta": float(s0 - r0),
                    }
    summary["ic"] = ic_piv
    summary["topn_ret3d"] = tn_piv
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[done] {time.time() - t0:.0f}s → {out_dir}")


if __name__ == "__main__":
    main()
