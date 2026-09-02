"""_diag_gate3_arms.py — E7 闸3 (q50>0 符号闸) 三臂对比 250d OOS (2026-09-01).

背景: q50 头整体有信号 (test 段日度 IC +0.07~+0.13) 但闸3 只用符号 (>0), 把连续
信号砍成 1 bit 且判在信号最弱的贴零带 — main 拦掉 60% 候选 / 54% 真赢家也被拦
(002295 案). q50 seed 集成已判死 (q50_ensemble_ab_20260901_*, 贴零带无方向).

三臂 (基线 = 生产闸去 q50: prob>base+margin_board & compound_ret>0 & pain<=pain_max):
  sign   — 现行: q50_3d>0 & q50_5d>0 (缺列 fillna compound_ret, 与生产一致)
  q{τ}   — 相对分位闸: q50_3d/5d 各 ≥ 当日(board)截面 τ 分位 (τ∈{0.5,0.6,0.7};
           τ=0.6 过闸率≈现行 40%, 同池量比成员)
  off    — 撤闸

评估: 每 (date,board) 按 pred_ret_10d 降序 top-5/top-10, 实得 = T+10 c2c 净
(D+1 close 买 / D+11 close 卖, 扣 COST). 指标: 命中/实得/中位/≥5%/≥10%/出票日 +
winner_capture (当日池实得前 10% 赢家出现在臂 top-N 的比例均值 — 直接回应
"不要烂掉真赢家"). 4 子窗纪律: 臂 vs sign 的实得差 ≥3/4 季度非负才考虑换.

回放口径同 _diag_legacy_hitrate_topn (特征一次构建后按日切片, 无前瞻; base_rate
滚动 20 日预热; 用当前生产 bundle 重放 — 绝对水平含记忆偏, 只比臂间相对).

WORM: DATA OTHERS/diag/gate3_arms_<ts>.csv/.json

用法:
  python scripts/_diag_gate3_arms.py                 # 全量 --slice 420 --eval 250
  python scripts/_diag_gate3_arms.py --slice 120 --eval 20   # 冒烟
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.list_generator import ListGenerator
from app.pipeline1.predictor import V35Predictor
from config.settings import (
    LEGACY_ENTRY_GATE,
    PANEL_V3_PATH,
    data_others_path,
)

BUNDLES = {
    "main": "models/pipeline1/main_current.pkl",
    "dual": "models/pipeline1/dual_current.pkl",
}
COST = 0.0020
TOP_DEPTHS = [5, 10]
QTILES = [0.5, 0.6, 0.7]
REALIZED_BUY_LAG = 1
REALIZED_SELL_LAG = 11
WINNER_Q = 0.90  # 当日(board)池实得前 10% = 赢家


def _build_realized_pivot(panel: pd.DataFrame):
    cal = np.sort(np.unique(pd.to_datetime(panel["date"]).dt.normalize().to_numpy()))
    pivot = (
        panel.assign(dt=pd.to_datetime(panel["date"]).dt.normalize())
        .pivot_table(index="symbol", columns="dt", values="close_hfq", aggfunc="last")
        .sort_index()
    )
    pivot = pivot.reindex(columns=pd.to_datetime(cal)).ffill(axis=1)
    return pivot, cal


def _realized_net(pivot, cal, i: int, symbol: str) -> float:
    buy_dt = pd.Timestamp(cal[i + REALIZED_BUY_LAG])
    sell_dt = pd.Timestamp(cal[i + REALIZED_SELL_LAG])
    try:
        pb = float(pivot.at[symbol, buy_dt])
        ps = float(pivot.at[symbol, sell_dt])
    except KeyError:
        return float("nan")
    if not (np.isfinite(pb) and np.isfinite(ps)) or pb <= 0:
        return float("nan")
    return ps / pb - 1.0 - COST


def _base_mask(scored: pd.DataFrame, board: str) -> pd.Series:
    """生产闸去 q50: prob 边际 / ret>0 / pain 上限 (LEGACY_ENTRY_GATE 定案档)."""
    ok = pd.Series(True, index=scored.index)
    cp = (
        scored["compound_prob"]
        if "compound_prob" in scored.columns
        else scored["prob_up"]
    )
    cr = (
        scored["compound_ret"]
        if "compound_ret" in scored.columns
        else scored["pred_ret_10d"]
    )
    ok &= cp > scored["base_rate"] + LEGACY_ENTRY_GATE["prob_margin"].get(board, 0.0)
    ok &= cr > 0
    if "pain_prob" in scored.columns:
        ok &= scored["pain_prob"].fillna(0) <= LEGACY_ENTRY_GATE["pain_max"].get(
            board, 1.0
        )
    return ok


def _q50_filled(scored: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    cr = (
        scored["compound_ret"]
        if "compound_ret" in scored.columns
        else scored["pred_ret_10d"]
    )
    q3 = scored["pred_q50_3d"].fillna(cr) if "pred_q50_3d" in scored.columns else cr
    q5 = scored["pred_q50_5d"].fillna(cr) if "pred_q50_5d" in scored.columns else cr
    return q3, q5


def _arm_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """三臂 mask (作用在基线过闸池上, 按 (date,board) 截面)."""
    masks: dict[str, pd.Series] = {}
    masks["sign"] = (df["q50_3d"] > 0) & (df["q50_5d"] > 0)
    for tau in QTILES:
        t3 = df.groupby(["date", "board"])["q50_3d"].transform(
            lambda s, q=tau: s.quantile(q)
        )
        t5 = df.groupby(["date", "board"])["q50_5d"].transform(
            lambda s, q=tau: s.quantile(q)
        )
        masks[f"q{tau:.1f}"] = (df["q50_3d"] >= t3) & (df["q50_5d"] >= t5)
    masks["off"] = pd.Series(True, index=df.index)
    return masks


def _stats(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {
            "n_days": 0,
            "picks": 0,
            "avg_picks": 0.0,
            "hit": float("nan"),
            "mean": float("nan"),
            "med": float("nan"),
            "ge5": float("nan"),
            "ge10": float("nan"),
            "win_capture": float("nan"),
        }
    r = sub["realized_net"].dropna()
    # winner_capture: 各日 |top-N ∩ 当日赢家| / |当日赢家| 的均值
    caps = []
    for _d, g in sub.groupby("date"):
        winners = set(g.loc[g["is_winner"], "symbol"])
        if winners:
            caps.append(len(winners & set(g["symbol"])) / len(winners))
    return {
        "n_days": int(sub["date"].nunique()),
        "picks": int(len(sub)),
        "avg_picks": float(len(sub) / max(1, sub["date"].nunique())),
        "hit": float((r > 0).mean()) if len(r) else float("nan"),
        "mean": float(r.mean()) if len(r) else float("nan"),
        "med": float(r.median()) if len(r) else float("nan"),
        "ge5": float((r >= 0.05).mean()) if len(r) else float("nan"),
        "ge10": float((r >= 0.10).mean()) if len(r) else float("nan"),
        "win_capture": float(np.mean(caps)) if caps else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=420)
    ap.add_argument("--eval", type=int, default=250)
    args = ap.parse_args()

    t0 = time.time()
    predictor = V35Predictor(BUNDLES)
    features = FeatureEngineV35()
    lister = ListGenerator()

    print(f"[load] panel {PANEL_V3_PATH}", flush=True)
    panel = pd.read_parquet(str(PANEL_V3_PATH))
    dates = sorted(pd.unique(pd.to_datetime(panel["date"])))
    cut = dates[-args.slice]
    panel = panel[pd.to_datetime(panel["date"]) >= cut].reset_index(drop=True)
    print(
        f"[slice] {pd.Timestamp(cut).date()}.. {len(panel):,}r ({time.time() - t0:.0f}s)",
        flush=True,
    )

    pivot, cal = _build_realized_pivot(panel)
    all_cal = pd.to_datetime(cal)
    i_of = {d: i for i, d in enumerate(all_cal)}
    print(
        f"[pivot] symbols={len(pivot)} days={len(cal)} ({time.time() - t0:.0f}s)",
        flush=True,
    )

    main_df, dual_df, _state = CleaningPipeline().run_inference(panel)
    del panel
    gc.collect()
    print(f"[clean] done ({time.time() - t0:.0f}s)", flush=True)

    detail: list[dict] = []
    for board, dfb, csr in (("main", main_df, False), ("dual", dual_df, True)):
        cols = predictor.bundles[board]["feature_cols"]
        feat = features.build(dfb, None, inference_cols=cols, cross_sectional_rank=csr)
        del dfb
        gc.collect()
        print(f"[feat:{board}] {len(feat):,}r ({time.time() - t0:.0f}s)", flush=True)

        day_dates = sorted(pd.unique(pd.to_datetime(feat["date"])))
        eval_days = [
            d
            for d in day_dates
            if d in i_of and i_of[d] + REALIZED_SELL_LAG < len(all_cal)
        ][-args.eval :]

        # base_rate 滚动 20 日预热 (与生产每日链路等价)
        for d in [d for d in day_dates if d < eval_days[0]]:
            day_feat = feat[pd.to_datetime(feat["date"]) == d]
            if day_feat.empty:
                continue
            try:
                pred = predictor.predict(day_feat, board)
                if not pred.empty:
                    lister.compute_scores(pred)
            except Exception:
                pass
        print(
            f"[{board}] warm {len([d for d in day_dates if d < eval_days[0]])}d, eval {len(eval_days)}d",
            flush=True,
        )

        for k, d in enumerate(eval_days):
            di = i_of[d]
            day_feat = feat[pd.to_datetime(feat["date"]) == d]
            if day_feat.empty:
                continue
            try:
                pred = predictor.predict(day_feat, board)
            except Exception as exc:
                print(
                    f"[{board}] {pd.Timestamp(d).date()} predict err: {exc}", flush=True
                )
                continue
            if pred.empty:
                continue
            scored = lister.compute_scores(pred)
            keep = _base_mask(scored, board)
            q3, q5 = _q50_filled(scored)
            cr = (
                scored["compound_ret"]
                if "compound_ret" in scored.columns
                else scored["pred_ret_10d"]
            )
            cp = (
                scored["compound_prob"]
                if "compound_prob" in scored.columns
                else scored["prob_up"]
            )
            for idx in scored.index[keep]:
                row = scored.loc[idx]
                detail.append(
                    {
                        "date": str(pd.Timestamp(d).date()),
                        "board": board,
                        "symbol": row["symbol"],
                        "pred_ret_10d": float(cr.loc[idx]),
                        "prob": float(cp.loc[idx]),
                        "q50_3d": float(q3.loc[idx]),
                        "q50_5d": float(q5.loc[idx]),
                        "realized_net": _realized_net(pivot, cal, di, row["symbol"]),
                    }
                )
            if (k + 1) % 50 == 0 or k == len(eval_days) - 1:
                print(
                    f"[{board}] {k + 1}/{len(eval_days)} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
        del feat
        gc.collect()

    if not detail:
        print("无任何过闸候选", flush=True)
        return 1
    df = pd.DataFrame(detail)
    df["date"] = pd.to_datetime(df["date"])
    # 赢家 = 当日(board)基线池实得前 WINNER_Q 分位 (在全池上定, 非 top-N 子集)
    win_q = df.groupby(["date", "board"])["realized_net"].transform(
        lambda s: s.quantile(WINNER_Q)
    )
    df["is_winner"] = df["realized_net"] >= win_q
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    df.to_csv(out_dir / f"gate3_arms_{ts}.csv", index=False)

    summary: list[dict] = []
    for board in ("main", "dual"):
        bdf = df[df["board"] == board]
        days = sorted(bdf["date"].unique())
        quarters = np.array_split(days, 4)
        masks = _arm_masks(bdf)
        print(
            f"\n===== {board} | {len(days)} 已实现日 | 基线池 {len(bdf)} 行 =====",
            flush=True,
        )
        print(
            f"  {'臂':<8}{'过闸率':>7}  top-N  {'命中':>7}{'实得':>9}{'中位':>9}{'≥5%':>8}{'≥10%':>8}{'赢家捕获':>9}  4子窗实得(臂-sign,pp)",
            flush=True,
        )
        for arm, m in masks.items():
            sub = bdf[m]
            for n in TOP_DEPTHS:
                topn = (
                    sub.sort_values(["date", "pred_ret_10d"], ascending=[True, False])
                    .groupby("date", sort=False)
                    .head(n)
                )
                s = _stats(topn)
                # 4 子窗: 臂 - sign 的日均实得差 (pp)
                qd = []
                if arm != "sign":
                    sign_topn = (
                        bdf[masks["sign"]]
                        .sort_values(["date", "pred_ret_10d"], ascending=[True, False])
                        .groupby("date", sort=False)
                        .head(n)
                    )
                    for qdays in quarters:
                        a = topn[topn["date"].isin(qdays)]
                        b = sign_topn[sign_topn["date"].isin(qdays)]
                        ra, rb = a["realized_net"].mean(), b["realized_net"].mean()
                        qd.append(
                            (ra - rb) * 100
                            if np.isfinite(ra) and np.isfinite(rb)
                            else float("nan")
                        )
                qtxt = (
                    "/".join(f"{v:+.2f}" if np.isfinite(v) else "--" for v in qd)
                    if qd
                    else "基准"
                )
                print(
                    f"  {arm:<8}{m.mean():>7.1%}  top-{n:<3d} {s['hit']:>7.1%}{s['mean']:>+9.2%}{s['med']:>+9.2%}"
                    f"{s['ge5']:>8.1%}{s['ge10']:>8.1%}{s['win_capture']:>9.1%}  {qtxt}",
                    flush=True,
                )
                summary.append(
                    {
                        "board": board,
                        "arm": arm,
                        "top_n": n,
                        "pass_rate": float(m.mean()),
                        **s,
                        "q_deltas_pp": qd,
                    }
                )
        # 赢家误杀口径 (池级): 实得前10%赢家被各臂拦掉的比例
        pool_win = bdf["is_winner"]
        for arm, m in masks.items():
            kill = float((~m[pool_win]).mean()) if pool_win.any() else float("nan")
            print(f"  [池级赢家误杀] {arm}: {kill:.1%}", flush=True)
            summary.append(
                {
                    "board": board,
                    "arm": arm,
                    "metric": "pool_winner_kill",
                    "value": kill,
                }
            )

    (out_dir / f"gate3_arms_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "slice": args.slice,
                "eval": args.eval,
                "cost": COST,
                "qtiles": QTILES,
                "summary": summary,
                "n_rows": len(df),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    print(
        f"\n[saved] {out_dir}/gate3_arms_{ts}.csv/.json ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
