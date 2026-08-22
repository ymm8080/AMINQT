"""宇宙扩张后 legacy 概率闸 margin 重扫 (生产概率头 + top-N 重点, 2026-08-17).

背景: LEGACY_PROB_GATE margin=0.08 定案于 3000 只宇宙 (并行 walk-forward);
V3 扩张后 4960 只, 分布变化 (新增更小/更不流动/收益更低) → 用户问是否按
top-10 重扫 (记忆 v3-universe-distribution-shift-params 也安排 legacy 闸重扫).

与 _diag_legacy_prob_head_replay_20260816_220029 (已跑, 扩张宇宙, 但其新头 =
walk-forward 自训头, 与生产 bundle 分布不符: main keep 57% vs 生产 15.3%,
dual keep 34% vs 生产 90%) 的差异:
- 概率 = 生产 bundle (prob_head_legacy/*_prob_*.joblib, 固定头, 与生产闸同构)
- base_rate = 生产 _base_rate 口径 (全板尾 20 日 mfe 达标率, 无前瞻)
- 评估 = top-5/10/15 (用户重点 top-10) × margin {0.00,0.04,...,0.16}
  × 3 子窗稳定性 (参数扫描铁律), 实得含成本
- 不做 walk-forward 重训 → 运行 ~40min (回放 ~90min 大头在重训)

闸语义 (镜像 apply_prob_gate): 保留 ⇔ pred_prob > base_rate + margin,
个股缺失/当日 base NaN → fail-open 保留.

WORM: data/diag/legacy_prob_gate_margin_sweep_<ts>.json/.csv
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "pipeline1"))

from app.pipeline1 import prob_head
from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import LabelEngine
from app.pipeline1.list_generator import ListGenerator
from app.pipeline1.predictor import V35Predictor
from config.settings import (
    LEGACY_PROB_GATE,
    PANEL_V3_PATH,
    data_others_path,
)

BUNDLES = {
    "main": "models/pipeline1/main_current.pkl",
    "dual": "models/pipeline1/dual_current.pkl",
}
ABS_TARGET = LEGACY_PROB_GATE["abs_target"]
REALIZED_BUY_LAG = 1
REALIZED_SELL_LAG = 11
COST = 0.003  # 分层滑点, 同回放
BASE_TAIL_DAYS = LEGACY_PROB_GATE["base_rate_days"] + 14
# 08-22 定案网格 (用户拍板): 125d 窗口探主板上沿 0.12/0.16/0.18/0.20,
# 另含 0.08 生产基线 (同窗对照) — margin 数不影响耗时 (均从同一 detail 行重算).
MARGINS = [0.08, 0.12, 0.16, 0.18, 0.20]
TOPN = [5, 10, 15]


def _build_realized_pivot(panel: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
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


def _gate_mask(scored: pd.DataFrame, pain_thresh: float = 0.5):
    """生产 entry_filter 非 bear 口径 (同回放/同 _diag_legacy_hitrate_topn)."""
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
    ok &= cp > scored["base_rate"] + 0.0
    ok &= cr > 0.0
    if all(c in scored.columns for c in ("pred_q50_3d", "pred_q50_5d")):
        ok &= (scored["pred_q50_3d"].fillna(cr) > 0) & (
            scored["pred_q50_5d"].fillna(cr) > 0
        )
    if "pain_prob" in scored.columns and pain_thresh is not None:
        ok &= scored["pain_prob"].fillna(0) <= pain_thresh
    return ok


def _build_raw_labels(dfb: pd.DataFrame) -> pd.DataFrame:
    """清洗帧 → 小 raw 帧 (symbol/date/close_hfq/high_hfq/low_hfq/adv20/mfe_3d/label_pain)."""
    from app.pipeline1.label_engine import _ensure_sorted

    if "adv20" not in dfb.columns:
        if "amount" not in dfb.columns:
            raise ValueError("清洗帧缺 amount (无法现算 adv20 打标签)")
        dfb = _ensure_sorted(dfb)
        dfb["adv20"] = (
            dfb.groupby("symbol")["amount"]
            .rolling(20, min_periods=20)
            .mean()
            .reset_index(level=0, drop=True)
        )
    raw = dfb[["symbol", "date", "close_hfq", "high_hfq", "low_hfq", "adv20"]].copy()
    raw["symbol"] = raw["symbol"].astype(str)
    raw = prob_head._add_mfe_3d(raw)
    pain = LabelEngine.build_path_labels(raw)["label_pain"]
    if "is_suspended" in dfb.columns:
        rs = (
            dfb.groupby("symbol")["is_suspended"]
            .rolling(5)
            .sum()
            .reset_index(level=0, drop=True)
        )
        vals = rs.values
        susp = np.zeros(len(vals), dtype=bool)
        if len(vals) > 4:
            susp[: len(vals) - 4] = vals[4:] > 0
        pain = pain.where(~pd.Series(susp, index=rs.index), np.nan)
    raw["label_pain"] = pain
    return raw


def _topn(sub: pd.DataFrame, n: int, rank_col: str = "pred_ret_10d") -> pd.DataFrame:
    return (
        sub.sort_values(["date", rank_col], ascending=[True, False])
        .groupby("date", sort=False)
        .head(n)
    )


def _stats3(sub: pd.DataFrame) -> dict:
    """3 子窗稳定性 (参数扫描铁律). 每窗 win/hit10/mean10."""
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
            "sub_windows": [],
        }
    r = sub["realized_net"].dropna()
    days = sorted(sub["date"].unique())
    n_sub = 3
    step = max(1, len(days) // n_sub)
    subs = []
    for i in range(n_sub):
        s0, s1 = i * step, len(days) if i == n_sub - 1 else (i + 1) * step
        seg = r[sub["date"].isin(days[s0:s1])]
        subs.append(
            {
                "win": f"{i + 1}/{n_sub}",
                "hit10": float((seg > 0).mean()) if len(seg) else float("nan"),
                "mean10": float(seg.mean()) if len(seg) else float("nan"),
            }
        )
    return {
        "n_days": int(sub["date"].nunique()),
        "picks": int(len(sub)),
        "avg_picks": float(len(sub) / max(1, sub["date"].nunique())),
        "hit": float((r > 0).mean()) if len(r) else float("nan"),
        "mean": float(r.mean()) if len(r) else float("nan"),
        "med": float(r.median()) if len(r) else float("nan"),
        "ge5": float((r >= 0.05).mean()) if len(r) else float("nan"),
        "ge10": float((r >= 0.10).mean()) if len(r) else float("nan"),
        "sub_windows": subs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=420, help="面板切片交易日数")
    ap.add_argument("--eval", type=int, default=250, help="评估的已实现决策日数")
    ap.add_argument(
        "--board",
        default="both",
        choices=("main", "dual", "both"),
        help="只扫指定板块 (08-22: dual 撤闸后仅 main)",
    )
    args = ap.parse_args()

    t0 = time.time()
    predictor = V35Predictor(BUNDLES)
    cleaner = CleaningPipeline()
    features = FeatureEngineV35()
    lister = ListGenerator()
    prob_bundles = {b: prob_head.load_latest(b) for b in ("main", "dual")}

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

    all_cal = pd.to_datetime(cal)
    i_of = {d: i for i, d in enumerate(all_cal)}

    detail: list[dict] = []
    _boards = {
        "main": (main_df, False),
        "dual": (dual_df, True),
        "both": None,
    }
    if args.board == "both":
        board_iter = (("main", main_df, False), ("dual", dual_df, True))
    else:
        board_iter = ((args.board,) + _boards[args.board],)
    for board, dfb, csr in board_iter:
        b = prob_bundles[board]
        if b is None:
            print(f"[{board}] 概率头 bundle 缺失 -> skip", flush=True)
            continue
        reg_cols = predictor.bundles[board]["feature_cols"]
        prob_cols = list(b["feat_cols"])
        union = list(dict.fromkeys([*reg_cols, *prob_cols]))
        feat = features.build(dfb, None, inference_cols=union, cross_sectional_rank=csr)
        print(
            f"[feat:{board}] {len(feat):,}r {len(feat.columns)}c "
            f"(union reg {len(reg_cols)} + prob {len(prob_cols)}) ({time.time() - t0:.0f}s)",
            flush=True,
        )

        day_dates = sorted(pd.unique(pd.to_datetime(feat["date"])))
        eval_days = [
            d
            for d in day_dates
            if d in i_of and i_of[d] + REALIZED_SELL_LAG < len(all_cal)
        ][-args.eval :]
        print(
            f"[{board}] eval days {len(eval_days)} "
            f"({pd.Timestamp(eval_days[0]).date()}..{pd.Timestamp(eval_days[-1]).date()})",
            flush=True,
        )

        # base_prod 逐日序列 (生产 _base_rate 口径)
        raw = _build_raw_labels(dfb)
        del dfb
        gc.collect()
        board_dates_arr = np.array(pd.to_datetime(day_dates))
        base_map: dict[pd.Timestamp, float] = {}
        for d in eval_days:
            pos = int(np.searchsorted(board_dates_arr, np.datetime64(d)))
            lo = max(0, pos - BASE_TAIL_DAYS + 1)
            tail = raw[
                (raw["date"] >= pd.Timestamp(board_dates_arr[lo]))
                & (raw["date"] <= pd.Timestamp(board_dates_arr[pos - 4]))
            ]
            bv = prob_head._base_rate(tail)
            base_map[pd.Timestamp(d)] = bv if bv is not None else np.nan
        n_ok = sum(1 for v in base_map.values() if np.isfinite(v))
        print(f"[{board}] base_prod {n_ok}/{len(eval_days)} 日可用", flush=True)

        # 预热 base_rate (compute_scores 20 日滚动)
        warm_days = [d for d in day_dates if d < eval_days[0]]
        for d in warm_days:
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
            base_mask = _gate_mask(scored)
            pain_mask = _gate_mask(scored, pain_thresh=None)
            record = scored[base_mask | pain_mask].copy()
            record["pain_excluded"] = ~base_mask[record.index]
            if record.empty:
                continue
            try:
                prob = prob_head.predict(b, day_feat[prob_cols])
            except Exception as exc:
                print(
                    f"[{board}] {pd.Timestamp(d).date()} prob_head predict err: {exc}",
                    flush=True,
                )
                continue
            prob_by_symbol = pd.Series(
                prob.to_numpy(), index=day_feat["symbol"].astype(str)
            )
            record["pred_prob"] = record["symbol"].astype(str).map(prob_by_symbol)
            base = base_map.get(pd.Timestamp(d), np.nan)
            for _, row in record.iterrows():
                detail.append(
                    {
                        "date": str(pd.Timestamp(d).date()),
                        "board": board,
                        "symbol": str(row["symbol"]),
                        "pred_ret_10d": float(
                            row.get("compound_ret", row.get("pred_ret_10d", np.nan))
                        ),
                        "pred_prob": float(row.get("pred_prob", np.nan)),
                        "base_prod": float(base) if np.isfinite(base) else np.nan,
                        "pain_excluded": bool(row["pain_excluded"]),
                        "realized_net": _realized_net(
                            pivot, cal, di, str(row["symbol"])
                        ),
                    }
                )
            if (k + 1) % 25 == 0 or k == len(eval_days) - 1:
                print(
                    f"[{board}] detail {k + 1}/{len(eval_days)} ({time.time() - t0:.0f}s)",
                    flush=True,
                )

        del feat, raw
        gc.collect()

    if not detail:
        print("无任何过闸候选", flush=True)
        return 1

    df = pd.DataFrame(detail)
    df["date"] = pd.to_datetime(df["date"])
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)

    summary: list[dict] = []
    for board in ("main", "dual"):
        sub = df[(df["board"] == board) & (~df["pain_excluded"])].copy()
        print(f"\n===== {board} | {sub['date'].nunique()} 已实现日 =====", flush=True)
        print(
            f"  {'变体':<24}{'出票':>5}{'票/日':>6} {'命中':>7} {'实得':>8} "
            f"{'≥5%':>7} {'≥10%':>7}  {'子窗 hit/实得'}",
            flush=True,
        )

        def _report(
            name: str, v: pd.DataFrame, n: int, current_board: str = board
        ) -> None:
            s = _stats3(_topn(v, n))
            summary.append({"board": current_board, "variant": name, "topn": n, **s})
            sub_s = "  ".join(
                f"{w['win']}:{w['hit10']:.0%}/{w['mean10']:+.2%}"
                for w in s["sub_windows"]
            )
            print(
                f"  {name:<24}{s['picks']:>5}{s['avg_picks']:>6.1f} {s['hit']:>7.1%} "
                f"{s['mean']:>+8.2%} {s['ge5']:>7.1%} {s['ge10']:>7.1%}  {sub_s}",
                flush=True,
            )

        for n in TOPN:
            print(f"  -- top-{n} --", flush=True)
            _report(f"基准 top-{n} (无概率闸)", sub, n)
            for m in MARGINS:
                keep = (
                    (sub["pred_prob"] > sub["base_prod"] + m)
                    | sub["pred_prob"].isna()
                    | sub["base_prod"].isna()
                )
                _report(f"prob>base+{m:.2f}", sub[keep], n)

    csv_path = out_dir / f"legacy_prob_gate_margin_sweep_{ts}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path = out_dir / f"legacy_prob_gate_margin_sweep_{ts}.json"
    import json

    json.dump(
        {
            "ts": ts,
            "slice": args.slice,
            "eval": args.eval,
            "cost": COST,
            "abs_target": ABS_TARGET,
            "margin": LEGACY_PROB_GATE["margin"],
            "note": (
                "生产概率头 + 生产 base_rate 口径 + top-5/10/15 × 3 子窗; "
                "margin 0.08 现状钉住对照"
            ),
            "summary": summary,
            "n_detail": len(df),
        },
        open(json_path, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
    print(
        f"\n[WORM] {json_path.name}\n[WORM] {csv_path.name} "
        f"(总 {time.time() - t0:.0f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
