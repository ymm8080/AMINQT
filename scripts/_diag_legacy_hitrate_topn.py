"""_diag_legacy_hitrate_topn.py — legacy 命中率 top-N / 闸门 walk-forward 诊断 (2026-08-14).

背景: 并行入选收紧 top-5 (main 57.8→61.2% / dual 64.4→68.1% 命中) 后, 用户问 legacy
可否同样提命中率. 本脚本用当前 legacy 模型 (main/dual_current.pkl) 重放末 ~250
已实现交易日:

  1) 生产基准闸 (entry_filter): prob>base_rate / pred_ret_10d>0 / pred_q50_3d&5d>0 /
     pain<=0.5, 按 pred_ret_10d 降序 top-N∈{5,8,10,15} 的 命中率(>0)/实得/≥5%/≥10%.
  2) 闸门变体叠加基准闸: prob 边际 +0.02/+0.05/+0.08; pred_ret_10d 阈值 +0.5%/+1%/+2%;
     pain 闸 (生产 0.5 无回测依据) 更严档 ≤0.3/≤0.4 + 关闸 (含只被 pain 拦下的行).
     看命中率提升 vs 出票量(每日过闸票数) 的权衡.

实得 = T+10 close-to-close 净收益 (买 D+1 close / 卖 D+11 close, 扣 COST).
无前瞻: 特征构建一次后按日切片 — 已审计全部 rank 为逐日截面(按 date / date+board)或
rolling, 无负 shift / 无整窗全局统计, 与生产 daily 推理同构. base_rate 用单实例
ListGenerator 滚动 20 日 (与生产同).

WORM: DATA_OTHERS/diag/legacy_hitrate_topn_<ts>.json + <ts>.csv (逐日逐票).

用法:
  python scripts/_diag_legacy_hitrate_topn.py            # 全量: --slice 420 --eval 250
  python scripts/_diag_legacy_hitrate_topn.py --slice 120 --eval 20   # 冒烟
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gc

import numpy as np
import pandas as pd

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.list_generator import ListGenerator
from app.pipeline1.predictor import V35Predictor
from config.settings import PANEL_V3_PATH, data_others_path

BUNDLES = {
    "main": "models/pipeline1/main_current.pkl",
    "dual": "models/pipeline1/dual_current.pkl",
}
COST = 0.0020  # 往返成本: 佣金 0.025%x2 + 印花税 0.05% + 滑点 0.05%x2 ≈ 0.2%
TOP_DEPTHS = [5, 8, 10, 15]
PROB_MARGINS = [0.02, 0.05, 0.08]
RET_THRESHOLDS = [0.005, 0.01, 0.02]
PAIN_STRICTER = [0.3, 0.4]  # 08-14 pain 闸变体: 生产 0.5 无回测依据 → 更严档 + 关闸
REALIZED_BUY_LAG = 1  # 买 D+1 close
REALIZED_SELL_LAG = 11  # 卖 D+11 close (10 交易日持有)


def _build_realized_pivot(panel: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """symbol×date → close_hfq (每股 ffill 处理停牌), 返回 (宽表 pivot, 全局交易日历)."""
    cal = np.unique(pd.to_datetime(panel["date"].to_numpy()).normalize().to_numpy())
    cal = np.sort(cal)
    pivot = (
        panel.assign(dt=pd.to_datetime(panel["date"]).dt.normalize())
        .pivot_table(index="symbol", columns="dt", values="close_hfq", aggfunc="last")
        .sort_index()
    )
    pivot = pivot.reindex(columns=pd.to_datetime(cal)).ffill(axis=1)
    return pivot, cal


def _realized_net(pivot: pd.DataFrame, cal: np.ndarray, i: int, symbol: str) -> float:
    """决策日 cal[i] 的 T+10 净实得: buy=cal[i+buy_lag] close, sell=cal[i+sell_lag] close."""
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


def _gate_mask(
    scored: pd.DataFrame,
    prob_margin: float = 0.0,
    ret_thresh: float = 0.0,
    pain_thresh: float = 0.5,
):
    """生产 entry_filter 非 bear 口径 + 变体 (margin 抬 prob 闸, thresh 抬 ret 闸,
    pain_thresh=None 关 pain 闸)."""
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
    ok &= cp > scored["base_rate"] + prob_margin
    ok &= cr > ret_thresh
    if all(c in scored.columns for c in ("pred_q50_3d", "pred_q50_5d")):
        ok &= (scored["pred_q50_3d"].fillna(cr) > 0) & (
            scored["pred_q50_5d"].fillna(cr) > 0
        )
    if "pain_prob" in scored.columns and pain_thresh is not None:
        ok &= scored["pain_prob"].fillna(0) <= pain_thresh
    return ok


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
        }
    r = sub["realized_net"].dropna()
    return {
        "n_days": int(sub["date"].nunique()),
        "picks": int(len(sub)),
        "avg_picks": float(len(sub) / max(1, sub["date"].nunique())),
        "hit": float((r > 0).mean()) if len(r) else float("nan"),
        "mean": float(r.mean()) if len(r) else float("nan"),
        "med": float(r.median()) if len(r) else float("nan"),
        "ge5": float((r >= 0.05).mean()) if len(r) else float("nan"),
        "ge10": float((r >= 0.10).mean()) if len(r) else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--slice", type=int, default=420, help="面板切片交易日数 (含前温 + 已实现余量)"
    )
    ap.add_argument(
        "--eval", type=int, default=250, help="评估的已实现决策日数 (取末 N)"
    )
    args = ap.parse_args()

    t0 = time.time()
    predictor = V35Predictor(BUNDLES)
    cleaner = CleaningPipeline()
    features = FeatureEngineV35()
    lister = ListGenerator()

    print(f"[load] panel {PANEL_V3_PATH}", flush=True)
    panel = pd.read_parquet(str(PANEL_V3_PATH))
    print(
        f"[load] {len(panel):,}r max={panel['date'].max()} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    dates = sorted(pd.unique(pd.to_datetime(panel["date"])))
    cut = dates[-args.slice]
    panel = panel[pd.to_datetime(panel["date"]) >= cut].reset_index(drop=True)
    print(
        f"[slice] {pd.Timestamp(cut).date()}.. {len(panel):,}r ({time.time() - t0:.0f}s)",
        flush=True,
    )

    pivot, cal = _build_realized_pivot(panel)
    print(
        f"[pivot] symbols={len(pivot)} days={len(cal)} ({time.time() - t0:.0f}s)",
        flush=True,
    )

    main_df, dual_df, state = cleaner.run_inference(panel)
    print(
        f"[clean] valve={state} main={len(main_df):,} dual={len(dual_df):,} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    del panel
    gc.collect()

    # 全局交易日历 (与 realized pivot 对齐)
    all_cal = pd.to_datetime(cal)
    i_of = {d: i for i, d in enumerate(all_cal)}

    detail: list[dict] = []
    for board, dfb, csr in (("main", main_df, False), ("dual", dual_df, True)):
        cols = predictor.bundles[board]["feature_cols"]
        feat = features.build(dfb, None, inference_cols=cols, cross_sectional_rank=csr)
        print(
            f"[feat:{board}] {len(feat):,}r {len(feat.columns)}c ({time.time() - t0:.0f}s)",
            flush=True,
        )
        del dfb
        gc.collect()

        day_dates = sorted(pd.unique(pd.to_datetime(feat["date"])))
        # 只评估有已实现 T+10 的决策日 (i+sell_lag < len(cal)), 取末 args.eval 天
        eval_days = [
            d
            for d in day_dates
            if d in i_of and i_of[d] + REALIZED_SELL_LAG < len(all_cal)
        ][-args.eval :]
        print(
            f"[{board}] eval days {len(eval_days)} ({pd.Timestamp(eval_days[0]).date()}..{pd.Timestamp(eval_days[-1]).date()})",
            flush=True,
        )

        # base_rate 预热: lister 内部滚动 20 日窗口须在首个 eval 日前吃满历史
        # (2026-08-14 冒烟结论: 不预热则首 ~20 天 base_rate 用部分窗口, 与生产每日链路不等价)
        warm_days = [d for d in day_dates if d < eval_days[0]]
        for _k, d in enumerate(warm_days):
            day_feat = feat[pd.to_datetime(feat["date"]) == d]
            if day_feat.empty:
                continue
            try:
                pred = predictor.predict(day_feat, board)
                if not pred.empty:
                    lister.compute_scores(pred)
            except Exception:
                pass
        print(f"[{board}] base_rate 预热 {len(warm_days)} 天", flush=True)

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
            scored["date"] = d
            scored["board"] = board
            # 基准闸行 + 只被 pain 闸拦下的行 (08-14 pain 变体: 关闸/更严档 需这两类)
            base_mask = _gate_mask(scored)
            pain_mask = _gate_mask(scored, pain_thresh=None)
            record = scored[base_mask | pain_mask].copy()
            record["pain_excluded"] = ~base_mask[record.index]
            for _, row in record.iterrows():
                detail.append(
                    {
                        "date": str(pd.Timestamp(d).date()),
                        "board": board,
                        "symbol": row["symbol"],
                        "pred_ret_10d": float(
                            row.get("compound_ret", row.get("pred_ret_10d", np.nan))
                        ),
                        "prob": float(
                            row.get("compound_prob", row.get("prob_up", np.nan))
                        ),
                        "base_rate": float(row.get("base_rate", np.nan)),
                        "pain_prob": float(row.get("pain_prob", np.nan))
                        if "pain_prob" in row
                        else np.nan,
                        "pain_excluded": bool(row["pain_excluded"]),
                        "realized_net": _realized_net(pivot, cal, di, row["symbol"]),
                    }
                )
            if (k + 1) % 25 == 0 or k == len(eval_days) - 1:
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
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    df.to_csv(out_dir / f"legacy_hitrate_topn_{ts}.csv", index=False)

    summary: list[dict] = []
    for board in ("main", "dual"):
        # 现有变体全部只作用于基准闸行 (pain_excluded 行仅 pain 变体使用)
        sub = df[(df["board"] == board) & (~df["pain_excluded"])]
        print(f"\n===== {board} | {sub['date'].nunique()} 已实现日 =====", flush=True)
        print(
            f"  {'变体':<26}{'出票':>5}{'票/日':>6} {'命中':>7} {'实得':>8} {'中位':>8} {'≥5%':>7} {'≥10%':>7}",
            flush=True,
        )
        # 基准 top-N
        for n in TOP_DEPTHS:
            topn = (
                sub.sort_values(["date", "pred_ret_10d"], ascending=[True, False])
                .groupby("date", sort=False)
                .head(n)
            )
            s = _stats(topn)
            print(
                f"  {'基准 top-' + str(n):<26}{s['picks']:>5}{s['avg_picks']:>6.1f} {s['hit']:>7.1%} "
                f"{s['mean']:>+8.2%} {s['med']:>+8.2%} {s['ge5']:>7.1%} {s['ge10']:>7.1%}",
                flush=True,
            )
            summary.append({"board": board, "variant": f"base_top{n}", **s})
        # prob 边际
        for m in PROB_MARGINS:
            mask = sub["prob"] > sub["base_rate"] + m
            v = sub[mask]
            topn = (
                v.sort_values(["date", "pred_ret_10d"], ascending=[True, False])
                .groupby("date", sort=False)
                .head(5)
            )
            s = _stats(topn)
            print(
                f"  {'prob>base+' + f'{m:.2f}' + ' top-5':<26}{s['picks']:>5}{s['avg_picks']:>6.1f} {s['hit']:>7.1%} "
                f"{s['mean']:>+8.2%} {s['med']:>+8.2%} {s['ge5']:>7.1%} {s['ge10']:>7.1%}",
                flush=True,
            )
            summary.append(
                {"board": board, "variant": f"prob_margin_{m:.2f}_top5", **s}
            )
        # ret 阈值
        for t in RET_THRESHOLDS:
            mask = sub["pred_ret_10d"] > t
            v = sub[mask]
            topn = (
                v.sort_values(["date", "pred_ret_10d"], ascending=[True, False])
                .groupby("date", sort=False)
                .head(5)
            )
            s = _stats(topn)
            print(
                f"  {'ret>' + f'{t:.1%}' + ' top-5':<26}{s['picks']:>5}{s['avg_picks']:>6.1f} {s['hit']:>7.1%} "
                f"{s['mean']:>+8.2%} {s['med']:>+8.2%} {s['ge5']:>7.1%} {s['ge10']:>7.1%}",
                flush=True,
            )
            summary.append({"board": board, "variant": f"ret_thresh_{t:.3f}_top5", **s})
        # pain 闸变体 (08-14: 生产 0.5 无回测依据 → 更严档 + 关闸)
        bdf = df[df["board"] == board]
        for pt in [None, *PAIN_STRICTER]:
            if pt is None:
                v, name = bdf, "pain_off"
            else:
                v = bdf[bdf["pain_prob"].fillna(0) <= pt]
                name = f"pain_le_{pt:.1f}"
            topn = (
                v.sort_values(["date", "pred_ret_10d"], ascending=[True, False])
                .groupby("date", sort=False)
                .head(5)
            )
            s = _stats(topn)
            print(
                f"  {name + '-top5':<26}{s['picks']:>5}{s['avg_picks']:>6.1f} {s['hit']:>7.1%} "
                f"{s['mean']:>+8.2%} {s['med']:>+8.2%} {s['ge5']:>7.1%} {s['ge10']:>7.1%}",
                flush=True,
            )
            summary.append({"board": board, "variant": f"{name}_top5", **s})

    (out_dir / f"legacy_hitrate_topn_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "slice": args.slice,
                "eval": args.eval,
                "cost": COST,
                "summary": summary,
                "n_detail": len(df),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    print(
        f"\n[saved] {out_dir}/legacy_hitrate_topn_{ts}.csv/.json ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
