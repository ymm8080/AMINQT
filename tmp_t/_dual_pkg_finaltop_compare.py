# -*- coding: utf-8 -*-
"""_dual_pkg_finaltop_compare.py — 多模型包"终版交付清单"质量对拍 (影子回放).

对每个指定包, 在 TODAY 面板上重放 label-matured 窗口内逐日的最终交付清单口径
(复用 tmp_t/_q90_slot_eval.py 已验证的闸重建/排名/真赢家口径, 2026-09-02 对拍):
  当日截面全池预测 (生产 V35Predictor 路径, 含 brute 列/超额加回/reg 残差概率)
  → E7 闸 (当期 LEGACY_ENTRY_GATE: prob>base+margin, ret10>0, pain<=max;
    q50 符号闸按 --q50-gate 复现旧环境, 默认撤=今日 config)
  → pred_ret_10d 降序 TOP10 (板内) + 跨板合并 TOP10/TOP15 (交付口径)
  → net3/net10 实得 (buy D+1 close / sell D+1+h close − COST, 同 q90 回放).

口径警告 (影子回放):
- 包内含训练窗数据 → 绝对水平偏乐观; 各臂共享偏差, 配对 Δ 仍可比.
- 闸 = 当期 config 静态 margin; 生产 LEGACY_PROB_GATE 自适应 margin / 行业上限 /
  FINAL STOCK SCAN 未重建 (各臂一致省略).
- base_rate 20 日滚动基准用 --warmup 天预热近似生产历史 (>=20 天即精确收敛).

WORM: DATA_OTHERS/diag/_dual_pkg_finaltop_compare_<ts>.json + .parquet

用法:
  # 全窗: dual A/B/C + main A/B
  python tmp_t/_dual_pkg_finaltop_compare.py
  # 冒烟 (指定日, 仅 A 臂, q50 闸复现旧环境) + 与交付 CSV 对拍
  python tmp_t/_dual_pkg_finaltop_compare.py --only-dates 2026-08-19,2026-08-28,2026-08-31 --bundles a --q50-gate
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import gc

import numpy as np
import pandas as pd

from app.pipeline1.cleaning_pipeline import CleaningConfig, CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.list_generator import ListGenerator
from app.pipeline1.predictor import V35Predictor
from config.settings import PANEL_V3_PATH, data_others_path
from scripts._diag_q90_slot_replay import COST, _net_vec, _pivots
from scripts._run_guard import find_conflicts
from tmp_t._q90_slot_eval import TOPN, gate_mask

DEFAULT_MAIN = [
    "models/pipeline1/main_current.pkl",       # A: tag 20260903
    "models/pipeline1/main_20260902.pkl",      # B
]
DEFAULT_DUAL = [
    "models/pipeline1/dual_current.pkl",       # A: tag 20260830excessfix
    "models/pipeline1/dual_20260902.pkl",      # B
    "models/pipeline1/dual_20260903.pkl",      # C: 09-01 build
]
KEEP_PRED = [
    "symbol",
    "pred_ret_10d",
    "prob_up",
    "base_rate",
    "pain_prob",
    "pred_q50_3d",
    "pred_q50_5d",
]
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def replay_arm(predictor: V35Predictor, board: str, groups: dict,
               eval_days: list, warmup_days: list) -> pd.DataFrame:
    """单包单板: 预热 base_rate 20 日历史 → 逐日预测+评分, 返回全池行."""
    lister = ListGenerator()
    rows: list[pd.DataFrame] = []
    for phase, days in (("warm", warmup_days), ("eval", eval_days)):
        for d in days:
            day_feat = groups.get(d)
            if day_feat is None or day_feat.empty:
                continue
            try:
                pred = predictor.predict(day_feat, board)
                if pred.empty:
                    continue
                scored = lister.compute_scores(pred)
            except Exception as exc:
                log(f"[{board}] {pd.Timestamp(d).date()} predict err: {exc}")
                continue
            if phase == "warm":
                continue
            if "compound_ret" in scored.columns:
                scored["pred_ret_10d"] = scored["compound_ret"]
            if "compound_prob" in scored.columns:
                scored["prob_up"] = scored["compound_prob"]
            have = [c for c in KEEP_PRED if c in scored.columns]
            sub = scored[have].copy()
            sub["symbol"] = sub["symbol"].astype(str).str.zfill(6)
            rows.append(sub.assign(date=str(pd.Timestamp(d).date())))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def attach_nets(df: pd.DataFrame, px: pd.DataFrame, all_cal: pd.DatetimeIndex,
                i_of: dict, amt: pd.DataFrame) -> pd.DataFrame:
    """net_3d/net_10d + anchor amount (同 q90 回放口径); 视界不足 → NaN."""
    out = []
    for d, g in df.groupby("date"):
        di = i_of[pd.Timestamp(d)]
        syms = g["symbol"]
        g = g.copy()
        g["net_3d"] = (
            _net_vec(px, syms, all_cal[di + 1], all_cal[di + 4])
            if di + 4 < len(all_cal) else np.nan
        )
        g["net_10d"] = (
            _net_vec(px, syms, all_cal[di + 1], all_cal[di + 11])
            if di + 11 < len(all_cal) else np.nan
        )
        dts = pd.Timestamp(d)
        g["amount"] = (
            amt[dts].reindex(syms).to_numpy(dtype=float)
            if dts in amt.columns else np.nan
        )
        out.append(g)
    return pd.concat(out, ignore_index=True)


def daily_top(pool: pd.DataFrame, n: int) -> dict[str, list[str]]:
    """日→TOP n symbol 列表 (pred_ret_10d 降序, 平局按 symbol 稳定)."""
    out: dict[str, list[str]] = {}
    for d, g in pool.groupby("date"):
        top = g.sort_values(["pred_ret_10d", "symbol"], ascending=[False, True]).head(n)
        out[d] = list(top["symbol"])
    return out


def winners_of(df: pd.DataFrame, floor: float) -> dict[str, set[str]]:
    """真赢家: 当日池 amount>=floor 且 net3 非 NaN 的 net3 TOP10 (drop_duplicates)."""
    ok = (df["amount"] >= floor) & df["net_3d"].notna()
    u = df[ok].drop_duplicates(["date", "symbol"])
    return {d: set(g.nlargest(TOPN, "net_3d")["symbol"]) for d, g in u.groupby("date")}


def day_net_of(df: pd.DataFrame) -> dict[str, tuple[dict, dict]]:
    return {
        d: (dict(zip(g["symbol"], g["net_3d"])), dict(zip(g["symbol"], g["net_10d"])))
        for d, g in df.groupby("date")
    }


def arm_stats(picks: dict[str, list[str]], day_net: dict, winners: dict) -> dict:
    """日度指标 + 汇总 (net3/net10/hit3/cov, 与 _q90_slot_eval.arm_stats 同口径)."""
    daily = []
    for d, syms in picks.items():
        n3, n10 = day_net.get(d, ({}, {}))
        v3 = [n3[s] for s in syms if s in n3 and pd.notna(n3[s])]
        v10 = [n10[s] for s in syms if s in n10 and pd.notna(n10[s])]
        daily.append({
            "date": d,
            "net3": float(np.mean(v3)) if v3 else np.nan,
            "net10": float(np.mean(v10)) if v10 else np.nan,
            "hit3": float(np.mean([v > 0 for v in v3])) if v3 else np.nan,
            "cov": len(set(syms) & winners.get(d, set())),
        })
    dd = pd.DataFrame(daily)
    return {
        "daily": dd,
        "days": int(dd["net3"].notna().sum()) if len(dd) else 0,
        "net3": float(dd["net3"].mean()) if len(dd) else None,
        "net10": float(dd["net10"].mean()) if len(dd) else None,
        "hit3": float(dd["hit3"].mean()) if len(dd) else None,
        "cov": float(dd["cov"].mean()) if len(dd) else None,
    }


def paired(ref: dict, other: dict) -> dict:
    """配对 Δ = other − ref (net3), 全窗 + 两半窗 + 胜负日."""
    common = sorted(set(ref["daily"]["date"]) & set(other["daily"]["date"]))
    if not common:
        return {"days": 0}
    r = ref["daily"].set_index("date")["net3"]
    o = other["daily"].set_index("date")["net3"]
    dif = (o[common] - r[common]).dropna()
    h = len(dif) // 2
    return {
        "days": int(len(dif)),
        "d3_full": float(dif.mean()) if len(dif) else None,
        "d3_h1": float(dif.iloc[:h].mean()) if h else None,
        "d3_h2": float(dif.iloc[h:].mean()) if len(dif) - h else None,
        "win_days": int((dif > 0).sum()),
        "lose_days": int((dif < 0).sum()),
    }


def expect_check(replay_days: list, merged_picks15: dict[str, list[str]],
                 merged_rows: pd.DataFrame, csv_paths: list) -> list[dict]:
    """与交付 CSV 对拍: symbol 集重叠 + 重叠票 pred_ret_10d 值差 (诚实呈现)."""
    out = []
    for p in csv_paths:
        m = re.search(r"(\d{8})", os.path.basename(p))
        if not m:
            log(f"[expect] 无法解析日期: {p}")
            continue
        cands = [d for d in replay_days if pd.Timestamp(d) <= pd.Timestamp(m.group(1))]
        if not cands:
            log(f"[expect] {m.group(1)}: 回放无更早交易日, 跳过")
            continue
        d = max(cands)
        try:
            csv = pd.read_csv(p, dtype=str)
        except Exception as exc:
            log(f"[expect] 读取失败 {p}: {exc}")
            continue
        csv["symbol"] = csv["symbol"].astype(str).str.zfill(6)
        want_syms = set(csv["symbol"])
        got15 = set(merged_picks15.get(d, []))
        hit = want_syms & got15
        val_diff = None
        rows_d = merged_rows[merged_rows["date"] == d]
        if len(rows_d) and hit:
            rr = rows_d.set_index("symbol")["pred_ret_10d"].astype(float)
            cv = csv.set_index("symbol")["pred_ret_10d"].astype(float)
            both = [s for s in hit if s in rr.index and s in cv.index and pd.notna(rr[s])]
            if both:
                val_diff = float(np.median(np.abs(rr[both] - cv[both])))
        rec = {
            "csv": os.path.basename(p), "csv_date": m.group(1), "replay_date": str(d),
            "csv_syms": len(want_syms), "overlap": len(hit),
            "csv_only": sorted(want_syms - got15),
            "replay_only": sorted(got15 - want_syms),
            "pred10_medabsdiff": val_diff,
        }
        out.append(rec)
        vd = "n/a" if val_diff is None else f"{val_diff:.2e}"
        log(f"[expect] {os.path.basename(p)} → replay {d}: 重叠 {len(hit)}/{len(want_syms)}"
            f" | 重叠票 pred10 中位差 {vd}")
        log(f"[expect]   csv独有: {rec['csv_only']}")
        log(f"[expect]   replay独有: {rec['replay_only']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-bundles", default=",".join(DEFAULT_MAIN))
    ap.add_argument("--dual-bundles", default=",".join(DEFAULT_DUAL))
    ap.add_argument("--boards", default="main,dual")
    ap.add_argument("--bundles", default="all", help="臂选择: a / a,b / a,b,c")
    ap.add_argument("--slice", type=int, default=380)
    ap.add_argument("--eval-days", type=int, default=48)
    ap.add_argument("--warmup", type=int, default=21)
    ap.add_argument("--amount-floor", type=float, default=3e7)
    ap.add_argument("--only-dates", default=None, help="逗号分隔 YYYY-MM-DD, 冒烟用")
    ap.add_argument("--q50-gate", action="store_true", help="复现旧环境 q50 符号闸")
    ap.add_argument("--expect-csv", action="append", default=[])
    ap.add_argument(
        "--guard-exclude-pid",
        type=int,
        action="append",
        default=[],
        help="守卫豁免 PID (retrain 等宿主进程内子进程调用时传宿主 PID, 否则被自家宿主误杀)",
    )
    args = ap.parse_args()

    others = find_conflicts(exclude_pids={p for p in args.guard_exclude_pid})
    if others:
        for c in others:
            log(f"[guard] 冲突: {c['sentinel']} (PID {c['pid']})")
        log(f"[guard] 已有 {len(others)} 个重活进程, 本实例退出 (rc=3)")
        return 3

    boards = [b.strip() for b in args.boards.split(",")]
    bundles: dict[str, list[str]] = {}
    for board in boards:
        paths = (args.main_bundles if board == "main" else args.dual_bundles).split(",")
        bundles[board] = [p.strip() for p in paths if p.strip()]
    all_labels = [chr(ord("A") + i) for i in range(max(len(v) for v in bundles.values()))]
    if args.bundles != "all":
        sel = {c.strip().upper() for c in args.bundles.split(",")}
        keep = [i for i, lb in enumerate(all_labels) if lb in sel]
    else:
        keep = list(range(len(all_labels)))
    labels = [all_labels[i] for i in keep]
    for board in bundles:
        bundles[board] = [bundles[board][i] for i in keep if i < len(bundles[board])]
    log(f"[arms] boards={boards} labels={labels}")
    for b, v in bundles.items():
        log(f"[arms]   {b}: {dict(zip(labels, v))}")

    log(f"[load] {PANEL_V3_PATH}")
    dates_all = pd.read_parquet(str(PANEL_V3_PATH), columns=["date"])["date"]
    ds = sorted(pd.unique(dates_all))
    cut = ds[-args.slice]
    del dates_all
    panel = pd.read_parquet(str(PANEL_V3_PATH), filters=[("date", ">=", pd.Timestamp(cut))])
    log(f"[slice] {pd.Timestamp(cut).date()}..{pd.Timestamp(ds[-1]).date()} -> {len(panel):,}r")

    px, amt, cal = _pivots(panel)
    all_cal = pd.to_datetime(cal)
    i_of = {d: i for i, d in enumerate(all_cal)}
    log(f"[pivot] symbols={len(px)} days={len(cal)}")

    cleaned = CleaningPipeline(CleaningConfig()).run_inference(panel)
    main_df, dual_df, state = cleaned[0], cleaned[1], cleaned[2]
    log(f"[clean] valve={state} main={len(main_df):,} dual={len(dual_df):,}")
    del panel
    gc.collect()

    board_frames = {"main": (main_df, False), "dual": (dual_df, True)}
    features = FeatureEngineV35()
    rows_by_board: dict[str, dict[str, pd.DataFrame]] = {}
    eval_days: list = []
    for board in boards:
        dfb, csr = board_frames[board]
        predictors = {}
        for lb, path in zip(labels, bundles[board]):
            pred = V35Predictor({board: path})
            if board not in pred.bundles:
                raise SystemExit(f"[fatal] 包加载失败: {path}")
            predictors[lb] = pred
        cols = sorted({c for p in predictors.values()
                       for c in p.bundles[board]["feature_cols"]})
        log(f"[feat:{board}] inference_cols={len(cols)} (union of {len(predictors)} 包)")
        feat = features.build(dfb, None, inference_cols=cols, cross_sectional_rank=csr)
        del dfb
        gc.collect()
        log(f"[feat:{board}] {len(feat):,}r {len(feat.columns)}c")
        groups = {d: g for d, g in feat.groupby("date")}
        day_dates = sorted(groups)
        if args.only_dates:
            want = [pd.Timestamp(x).normalize() for x in args.only_dates.split(",")]
            eval_days = [d for d in day_dates if d in want]
            missing = set(want) - set(eval_days)
            if missing:
                log(f"[warn] only-dates 不在面板: {sorted(str(x.date()) for x in missing)}")
            if not eval_days:
                raise SystemExit("[fatal] 冒烟日期全部不在面板")
            first_i = min(day_dates.index(d) for d in eval_days)
            warmup_days = day_dates[:first_i][-args.warmup:]
        else:
            matured = [d for d in day_dates if i_of[d] + 11 < len(all_cal)]
            eval_days = matured[-args.eval_days:]
            warmup_days = day_dates[: day_dates.index(eval_days[0])][-args.warmup:]
        log(f"[{board}] warmup={len(warmup_days)}d | eval={len(eval_days)}d "
            f"{pd.Timestamp(eval_days[0]).date()}..{pd.Timestamp(eval_days[-1]).date()}")
        rows_by_board[board] = {}
        for lb in labels:
            if lb not in predictors:
                continue
            df = replay_arm(predictors[lb], board, groups, eval_days, warmup_days)
            if df.empty:
                raise SystemExit(f"[fatal] {board}:{lb} 回放 0 行")
            rows_by_board[board][lb] = attach_nets(df, px, all_cal, i_of, amt)
            log(f"[{board}:{lb}] rows={len(rows_by_board[board][lb]):,}")
        del feat, groups, predictors
        gc.collect()
    del main_df, dual_df
    gc.collect()

    # ── 评估: 闸重建 + 板内 TOP10 + 合并交付口径 + 真赢家 + 配对 Δ ────────
    result = {
        "ts": pd.Timestamp.now().isoformat(),
        "bundles": {b: dict(zip(labels, v)) for b, v in bundles.items()},
        "args": {k: str(v) for k, v in vars(args).items()},
        "cost": COST,
        "q50_sign_gate": bool(args.q50_gate),
        "eval_days": [str(pd.Timestamp(d).date()) for d in eval_days],
    }
    daily_records: list[dict] = []
    gated_by_board: dict[str, dict[str, pd.DataFrame]] = {}

    for board in boards:
        arms = rows_by_board[board]
        ref_lb = labels[0]
        ref = arms[ref_lb]
        winners = winners_of(ref, args.amount_floor)
        day_net = day_net_of(ref)
        result[board] = {
            "gate": f"E7@config (prob_margin/ret>0/pain_max, q50={args.q50_gate})",
            "pool_mean_per_day": float(len(ref) / max(ref["date"].nunique(), 1)),
        }
        board_gated: dict[str, pd.DataFrame] = {}
        stats: dict[str, dict] = {}
        for lb, df in arms.items():
            gated = df[gate_mask(df, board, args.q50_gate)].copy()
            board_gated[lb] = gated
            picks = daily_top(gated, TOPN)
            st = arm_stats(picks, day_net, winners)
            stats[lb] = st
            result[board][lb] = {k: v for k, v in st.items() if k != "daily"}
            for _, r in st["daily"].iterrows():
                daily_records.append({
                    "board": board, "arm": lb, "date": r["date"], "net3": r["net3"],
                    "net10": r["net10"], "hit3": r["hit3"], "cov": r["cov"],
                    "picks": ";".join(picks[r["date"]]),
                })
            log(f"[eval] {board}:{lb} days={st['days']} net3={st['net3']:+.5f} "
                f"net10={st['net10']:+.5f} hit3={st['hit3']:.3f} cov={st['cov']:.3f}")
        for lb in labels[1:]:
            if lb in stats and stats[lb]["days"] and stats[ref_lb]["days"]:
                result[board][f"delta_{lb}_vs_{ref_lb}"] = paired(stats[ref_lb], stats[lb])
        gated_by_board[board] = board_gated

    common_labels = [lb for lb in labels
                     if all(lb in rows_by_board[b] for b in rows_by_board)]
    if len(rows_by_board) > 1 and common_labels:
        ref_lb = common_labels[0]
        merged_rows = pd.concat(
            [rows_by_board[b][ref_lb] for b in rows_by_board], ignore_index=True
        )
        mw = winners_of(merged_rows, args.amount_floor)
        mdn = day_net_of(merged_rows)
        result["merged"] = {}
        mstats = {}
        merged_picks15: dict[str, list[str]] = {}
        for lb in common_labels:
            mg = pd.concat([gated_by_board[b][lb] for b in rows_by_board],
                           ignore_index=True)
            picks10 = daily_top(mg, TOPN)
            if lb == ref_lb:
                merged_picks15 = daily_top(mg, 15)
            st = arm_stats(picks10, mdn, mw)
            mstats[lb] = st
            result["merged"][lb] = {k: v for k, v in st.items() if k != "daily"}
            for _, r in st["daily"].iterrows():
                daily_records.append({
                    "board": "merged", "arm": lb, "date": r["date"], "net3": r["net3"],
                    "net10": r["net10"], "hit3": r["hit3"], "cov": r["cov"],
                    "picks": ";".join(picks10[r["date"]]),
                    "picks15": ";".join(merged_picks15.get(r["date"], [])),
                })
            log(f"[eval] merged:{lb} days={st['days']} net3={st['net3']:+.5f} "
                f"hit3={st['hit3']:.3f} cov={st['cov']:.3f}")
        for lb in common_labels[1:]:
            result["merged"][f"delta_{lb}_vs_{ref_lb}"] = paired(mstats[ref_lb], mstats[lb])
        if args.expect_csv:
            replay_days = [str(pd.Timestamp(d).date()) for d in eval_days]
            result["expect"] = expect_check(replay_days, merged_picks15,
                                            merged_rows, args.expect_csv)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    json_path = out_dir / f"_dual_pkg_finaltop_compare_{ts}.json"
    json_path.write_text(
        json.dumps({k: v for k, v in result.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pq_path = out_dir / f"_dual_pkg_finaltop_compare_{ts}.parquet"
    pd.DataFrame(daily_records).to_parquet(pq_path, index=False)
    log(f"[saved] {json_path}")
    log(f"[saved] {pq_path}")
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
