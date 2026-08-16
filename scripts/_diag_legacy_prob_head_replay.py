"""_diag_legacy_prob_head_replay.py — legacy 新 GBM 概率头 250d walk-forward replay (2026-08-16).

问题 (用户 08-16): legacy 新概率头由 parallel 移植而来, 但无 legacy 自己的回测证据 —
新头对 legacy 到底有没有增益? 本脚本重放末 ~250 已实现交易日, 在同一候选池上对比:
  A. 基准闸 top-5 (生产旧闸 entry_filter, 无新头) = "none" 排名键
  B. 旧头边际闸 top-5 (prob_up > base_rate + m, m ∈ 0.02/0.05/0.08)
  C. 新头边际闸 top-5 (walk-forward GBM pred_prob > base_rate_prod + m,
     m ∈ 0.04/0.06/0.08/0.10, 生产 fail-open: pred NaN → 保留)
  D. 排名键 blend (用户 08-16 问): pred_ret_10d × prob —
     旧头 blend (prob_up) 与新头 blend (pred_prob_new) 各一档

新头 walk-forward 与生产训练同配方 (prob_head.LGB_PARAMS, mfe_3d>=3% 二分类,
ok = mfe/pain 非 NaN, 每 21 交易日扩窗重训); base_rate 用生产 prob_head._base_rate
(尾 35 日切片, 只观测 <=D-4, 剔 NaN). 候选池/实得与 _diag_legacy_hitrate_topn
完全一致 (基准闸行, T+10 c2c 净收益扣 0.2% 成本) — B 组应逐字复现其数字 (自检).

检查点: data/_diag_legacy_wf_pred_<board>_e<eval>.parquet (walk-forward 预测落盘,
崩溃后重跑免重训; 特征构建不落盘, 与 parallel replay 同权衡).
WORM: DATA_OTHERS/diag/legacy_prob_head_replay_<ts>.csv/.json

用法:
  python scripts/_diag_legacy_prob_head_replay.py                        # 全量
  python scripts/_diag_legacy_prob_head_replay.py --slice 120 --eval 20  # 冒烟
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
from lightgbm import LGBMClassifier

from app.pipeline1 import prob_head
from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import LabelEngine, _ensure_sorted
from app.pipeline1.list_generator import ListGenerator
from app.pipeline1.predictor import V35Predictor
from config.settings import DATA_DIR, LEGACY_PROB_GATE, PANEL_V3_PATH, data_others_path

BUNDLES = {
    "main": "models/pipeline1/main_current.pkl",
    "dual": "models/pipeline1/dual_current.pkl",
}
COST = 0.0020  # 往返成本: 佣金 0.025%x2 + 印花税 0.05% + 滑点 0.05%x2 ≈ 0.2%
OLD_MARGINS = [0.02, 0.05, 0.08]
NEW_MARGINS = [0.04, 0.06, 0.08, 0.10]
REFIT_EVERY = LEGACY_PROB_GATE["refit_every_days"]
ABS_TARGET = LEGACY_PROB_GATE["abs_target"]
BASE_TAIL_DAYS = 35  # 生产 _base_rate 调用方传入的尾切片长度 (~base_rate_days+14)
REALIZED_BUY_LAG = 1
REALIZED_SELL_LAG = 11


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
    """生产 entry_filter 非 bear 口径 (同 _diag_legacy_hitrate_topn)."""
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
            "sub_windows": [],
        }
    r = sub["realized_net"].dropna()
    days = sorted(sub["date"].unique())
    n_sub = 4
    step = len(days) // n_sub
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


def _build_raw_labels(dfb: pd.DataFrame) -> pd.DataFrame:
    """清洗帧 → 小 raw 帧 (symbol/date/close/high/low/adv20/mfe_3d/label_pain).

    adv20 从 amount 现算 (同 _train_legacy_prob_head._attach_labels);
    mfe_3d/pain 用生产函数, 停牌遮蔽镜像 mask_suspension mdd_3d 分支.
    """
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


def _top5(sub: pd.DataFrame, rank_col: str = "pred_ret_10d") -> pd.DataFrame:
    return (
        sub.sort_values(["date", rank_col], ascending=[True, False])
        .groupby("date", sort=False)
        .head(5)
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=420, help="面板切片交易日数")
    ap.add_argument("--eval", type=int, default=250, help="评估的已实现决策日数")
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

    all_cal = pd.to_datetime(cal)
    i_of = {d: i for i, d in enumerate(all_cal)}

    detail: list[dict] = []
    base_maps: dict[str, dict] = {}
    for board, dfb, csr in (("main", main_df, False), ("dual", dual_df, True)):
        cols = predictor.bundles[board]["feature_cols"]
        feat = features.build(dfb, None, inference_cols=cols, cross_sectional_rank=csr)
        print(
            f"[feat:{board}] {len(feat):,}r {len(feat.columns)}c ({time.time() - t0:.0f}s)",
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

        # ---- 1) 候选池: 基准闸行 + 只被 pain 拦下的行 (同 rescan, B 组自检) ----
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
            base_mask = _gate_mask(scored)
            pain_mask = _gate_mask(scored, pain_thresh=None)
            record = scored[base_mask | pain_mask].copy()
            record["pain_excluded"] = ~base_mask[record.index]
            for _, row in record.iterrows():
                detail.append(
                    {
                        "date": str(pd.Timestamp(d).date()),
                        "board": board,
                        "symbol": str(row["symbol"]),
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

        # ---- 2) 标签 + base_prod 逐日序列 (生产 _base_rate) ----
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
                & (raw["date"] <= pd.Timestamp(d))
            ]
            b = prob_head._base_rate(tail)
            base_map[pd.Timestamp(d)] = b if b is not None else np.nan
        base_maps[board] = base_map
        n_ok = sum(1 for v in base_map.values() if np.isfinite(v))
        print(
            f"[{board}] base_prod {n_ok}/{len(eval_days)} 日可用 ({time.time() - t0:.0f}s)",
            flush=True,
        )

        # ---- 3) 新头 walk-forward (生产训练配方, 每 21 交易日扩窗重训) ----
        feat["symbol"] = feat["symbol"].astype(str)
        feat["date"] = pd.to_datetime(feat["date"])
        meta = feat[["symbol", "date"]].reset_index(drop=True)
        feat = feat.merge(
            raw[["symbol", "date", "mfe_3d", "label_pain"]],
            on=["symbol", "date"],
            how="left",
        )
        print(f"[{board}] 标签已合并 ({time.time() - t0:.0f}s)", flush=True)

        feat_cols = prob_head.feature_cols(feat)
        y = (feat["mfe_3d"] >= ABS_TARGET).astype(float)
        ok = y.notna() & feat["label_pain"].notna()
        x_all = feat[feat_cols].to_numpy(dtype="float32")
        idx = np.searchsorted(board_dates_arr, feat["date"].values)
        ok_arr = ok.to_numpy()
        del feat
        gc.collect()

        ckpt = DATA_DIR / f"_diag_legacy_wf_pred_{board}_e{args.eval}.parquet"
        if ckpt.exists():
            cp = pd.read_parquet(str(ckpt))
            print(f"[{board}] walk-forward 从检查点恢复 ({len(cp):,} 行)", flush=True)
        else:
            model = None
            wf_rows: list[pd.DataFrame] = []
            n_refits = 0
            for k, d in enumerate(eval_days):
                pos = int(np.searchsorted(board_dates_arr, np.datetime64(d)))
                if model is None or k % REFIT_EVERY == 0:
                    tr = (idx < pos) & ok_arr
                    model = LGBMClassifier(**prob_head.LGB_PARAMS)
                    model.fit(x_all[tr], y.loc[tr].to_numpy())
                    n_refits += 1
                te = idx == pos
                if not te.any():
                    continue
                p = model.predict_proba(x_all[te])[:, 1]
                wf_rows.append(meta.loc[te].assign(pred=p).reset_index(drop=True))
                if (k + 1) % 25 == 0 or k == len(eval_days) - 1:
                    print(
                        f"[{board}] wf {k + 1}/{len(eval_days)} "
                        f"(refits={n_refits}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
            pd.concat(wf_rows, ignore_index=True).to_parquet(ckpt)
            print(
                f"[{board}] walk-forward 完成: {n_refits} 次重训 → {ckpt.name}",
                flush=True,
            )

        del x_all, raw, meta
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
        ckpt = DATA_DIR / f"_diag_legacy_wf_pred_{board}_e{args.eval}.parquet"
        wf = pd.read_parquet(str(ckpt))
        wf["date"] = pd.to_datetime(wf["date"])
        sub = sub.merge(wf, on=["symbol", "date"], how="left").rename(
            columns={"pred": "pred_prob_new"}
        )
        sub["base_prod"] = sub["date"].map(base_maps[board])
        sub["blend_old"] = sub["pred_ret_10d"] * sub["prob"]
        sub["blend_new"] = sub["pred_ret_10d"] * sub["pred_prob_new"]
        print(f"\n===== {board} | {sub['date'].nunique()} 已实现日 =====", flush=True)
        print(
            f"  {'变体':<24}{'出票':>5}{'票/日':>6} {'命中':>7} {'实得':>8} "
            f"{'≥5%':>7} {'≥10%':>7}  {'子窗 hit/实得'}",
            flush=True,
        )

        def _report(
            name: str,
            v: pd.DataFrame,
            rank_col: str = "pred_ret_10d",
            board_name: str = board,
        ) -> None:
            s = _stats(_top5(v, rank_col))
            summary.append({"board": board_name, "variant": name, **s})
            sub_s = "  ".join(
                f"{w['win']}:{w['hit10']:.0%}/{w['mean10']:+.2%}"
                for w in s["sub_windows"]
            )
            print(
                f"  {name:<24}{s['picks']:>5}{s['avg_picks']:>6.1f} {s['hit']:>7.1%} "
                f"{s['mean']:>+8.2%} {s['ge5']:>7.1%} {s['ge10']:>7.1%}  {sub_s}",
                flush=True,
            )

        _report("基准 top-5 (无新头)", sub, board_name=board)
        for m in OLD_MARGINS:
            _report(
                f"旧头 prob>base+{m:.2f}",
                sub[sub["prob"] > sub["base_rate"] + m],
                board_name=board,
            )
        for m in NEW_MARGINS:
            keep = (
                (sub["pred_prob_new"] > sub["base_prod"] + m)
                | sub["pred_prob_new"].isna()
                | sub["base_prod"].isna()
            )
            _report(f"新头 prob>base+{m:.2f}", sub[keep], board_name=board)
        _report("旧头 blend (ret×prob_up)", sub, rank_col="blend_old", board_name=board)
        _report(
            "新头 blend (ret×pred_prob_new)",
            sub,
            rank_col="blend_new",
            board_name=board,
        )

    df.to_csv(out_dir / f"legacy_prob_head_replay_{ts}.csv", index=False)
    (out_dir / f"legacy_prob_head_replay_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "slice": args.slice,
                "eval": args.eval,
                "cost": COST,
                "abs_target": ABS_TARGET,
                "refit_every": REFIT_EVERY,
                "note": "基准=生产旧闸无新头; 旧头=cls prob_up vs lister base_rate; "
                "新头=walk-forward GBM (生产配方) vs 生产 _base_rate, fail-open; "
                "blend_old=pred_ret_10d×prob_up; blend_new=pred_ret_10d×pred_prob_new",
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
        f"\n[saved] {out_dir}/legacy_prob_head_replay_{ts}.csv/.json "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
