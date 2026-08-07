"""池分 vs LEGACY prob 正交性 + 叠加 OOS 双头 (2026-08-05).

用户: "池是因子不是模型, LEGACY 是唯一训练模型. 池分与 LEGACY prob 是否正交?
叠加 final_score 是否真赢过单用任一? pv_corr_5 进池后在叠加口径还有没有增量?"

OOS 6m 末窗, 每板块:
  - 正交性: 每日期截面 Spearman(池分, prob_up_3d) 均值/std/min/max
  - 四排名逐视界双头 (TOP-N = sniper 5 / fusion 10):
      a. 纯池分            pool_score(spec.pool)        [含 pv_corr_5, 已入 config]
      b. 纯 LEGACY prob    prob_up_3d
      c. 叠加              0.5*池分 + 0.5*prob           (overlay 默认权重)
      d. 叠加去 pv_corr_5   池不含 pv_corr_5 的叠加       (因子增量检验)
  - 自检: 池分方法输出应与 run_system 完全一致.

LEGACY prob 对每个 OOS 交易日全截面推理 (V35Predictor.predict), 面板已含 feature_cols.
输出 (WORM): data/_diag_overlay_orthogonality_<ts>.json
用法: python scripts/_diag_overlay_orthogonality.py [--window 6m] [--board main] [--out ...]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline_parallel.backtest import add_mfe_labels, run_system, tradability_gate
from app.pipeline_parallel.config import (
    ALL_HORIZON_INTS,
    BOARD_THRESHOLDS,
    FUSION,
    OOS_WINDOWS,
    PANEL,
    SNIPER,
)
from app.pipeline_parallel.scoring import (
    dual_head_ok,
    measure_dual_head,
    pool_score,
    select_topn,
)

MODEL_DIR = os.path.join("models", "pipeline1")
W_POOL, W_PROB = 0.5, 0.5  # overlay 默认权重 (legacy_overlay.rerank)
# 分类器优先级 (k, models 键), 与 overlay._pick_prob_col 一致: 3d > 5d > 2d > 1d
_PROB_PRIORITY_MODEL = ((3, "3d_cls"), (5, "5d_cls"), (2, "2d_cls"), (1, "1d_cls"))
SYSTEMS = {"sniper": SNIPER, "fusion": FUSION}

# 轻量列集 (池特征 ∪ MFE 标签输入); 全 548 列载入会 OOM (44GB commit)
_NEEDED = (
    "symbol",
    "date",
    "close_hfq",
    "high_hfq",
    "adv20",
    "volume",
    "amihud_illiq",
    "small_mv_premium",
    "amihud_illiquidity",
    "down_gap_pct",
    "VAR51",
    "ret_reversal_5d",
    "limit_dist_pct",
)


def _groll(s: pd.Series, key: pd.Series, n: int) -> pd.Series:
    """组内滚动均值 (对齐生产 indicators._groll 口径)."""
    return (
        s.groupby(key, sort=False)
        .rolling(n, min_periods=n)
        .mean()
        .reset_index(level=0, drop=True)
    )


def _pv_corr_5(df: pd.DataFrame) -> pd.DataFrame:
    """生产口径 pv_corr_5 (indicators.py:145-154): ret 与 volume 变化率 5 日 Pearson 相关."""
    key = df["symbol"]
    ret = df["close_hfq"].groupby(key, sort=False).pct_change()
    volp = df["volume"].groupby(key, sort=False).pct_change()
    xy = (ret * volp).replace([np.inf, -np.inf], np.nan)
    mxy = _groll(xy, key, 5)
    mx = _groll(ret, key, 5)
    my = _groll(volp, key, 5)
    varx = _groll(ret * ret, key, 5) - mx * mx
    vary = _groll(volp * volp, key, 5) - my * my
    corr = (mxy - mx * my) / np.sqrt(np.maximum(varx * vary, 0.0)).replace(
        [np.inf, -np.inf], np.nan
    )
    df["pv_corr_5"] = corr.fillna(0.0)  # 生产 prepare_adx 同款: NaN→0 不毒化 pool_score
    del xy, mxy, mx, my, varx, vary
    gc.collect()
    return df


def load_board(board: str) -> pd.DataFrame:
    """OOS 轻量行集: 池特征 + MFE 净标签 + 可交易性门 + pv_corr_5 + board."""
    ckpt = PANEL.main_checkpoint if board == "main" else PANEL.dual_checkpoint
    df = pd.read_parquet(ckpt, columns=list(_NEEDED))
    df = add_mfe_labels(df, horizons=ALL_HORIZON_INTS)
    df, _ = tradability_gate(df)
    df = _pv_corr_5(df)
    df["board"] = board
    return df


def load_probs(board: str, predictor, oos_start) -> tuple[pd.DataFrame, str]:
    """对 OOS 窗全截面一次性推理 LEGACY prob (冻结模型, PIT 特征), 返回 (date,symbol,prob_up) 帧 + 列名.

    快速路径: 只用分类器 + 对应校准器 (跳过 reg/quantile/pain/rank 等 ~40 个非必要模型),
    已验证与 V35Predictor.predict 的 prob_up_{k}d 完全一致 (corr=1.0, max|Δ|=0.0).
    注意: 历史截面要 per-(date,symbol) prob, 非 predict() 的每 symbol 最新一行 — 故不用 tail(1).
    """
    ckpt = PANEL.main_checkpoint if board == "main" else PANEL.dual_checkpoint
    bundle = predictor.bundles[board]
    feature_cols = list(bundle["feature_cols"])
    models = bundle["models"]
    calibrators = bundle.get("calibrators", {})
    k, kind = next(
        ((k, kind) for k, kind in _PROB_PRIORITY_MODEL if kind in models), (None, None)
    )
    if kind is None:
        raise RuntimeError(f"[{board}] 模型包无 {_PROB_PRIORITY_MODEL}")
    df = pd.read_parquet(
        ckpt,
        columns=feature_cols + ["symbol", "date"],
        filters=[("date", ">=", pd.Timestamp(oos_start))],
    )
    df = df.sort_values(["symbol", "date"], ignore_index=True)
    X = np.nan_to_num(
        df[feature_cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    raw = models[kind][0].predict_proba(X)[:, 1]
    cal = calibrators.get(k) if calibrators else None
    prob = cal.predict_proba(raw) if cal is not None else raw
    out = pd.DataFrame(
        {"symbol": df["symbol"].values, "date": df["date"].values, "prob_up": prob}
    )
    del df, X, raw, prob
    gc.collect()
    return out, f"prob_up_{k}d"


def orthogonality(sub: pd.DataFrame) -> dict:
    """每日期截面 Spearman(池分, prob) 的均值/std/min/max."""
    s = sub[["date", "pool_score", "prob_up"]].dropna()
    rs = []
    for _day, g in s.groupby("date"):
        if len(g) < 10:
            continue
        res = spearmanr(g["pool_score"], g["prob_up"])
        r = res.statistic if hasattr(res, "statistic") else res[0]
        rs.append(float(r))
    return {
        "n_days": len(rs),
        "spearman_avg": round(float(np.mean(rs)), 4) if rs else None,
        "spearman_std": round(float(np.std(rs)), 4) if rs else None,
        "spearman_min": round(float(np.min(rs)), 4) if rs else None,
        "spearman_max": round(float(np.max(rs)), 4) if rs else None,
    }


def measure(sub: pd.DataFrame, score: pd.Series, spec, bcrit) -> dict:
    """按 score 逐日期取 TOP-N, 逐视界量双头."""
    picks = select_topn(sub, score, spec.top_n)
    if picks.empty:
        return {"top_n": spec.top_n, "n_picks": 0, "per_horizon": {}}
    picks = picks.merge(
        sub[["date", "symbol"] + list(spec.labels)], on=["date", "symbol"], how="left"
    )
    per = {}
    for h, lab in zip(spec.horizons, spec.labels):
        m = measure_dual_head(picks, lab)
        per[h] = {
            "mag": m["mag"],
            "winrate": m["winrate"],
            "n": m["n"],
            "ok": bool(dual_head_ok(m, *bcrit)),
        }
    return {"top_n": spec.top_n, "n_picks": int(len(picks)), "per_horizon": per}


def _fmt(v, pct=False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v:+.1%}" if pct else f"{v:.3f}"


def _print_methods(methods: dict) -> None:
    print(
        f"    {'方法':<16}{'3d_wr':>8}{'3d_mag':>9}{'5d_wr':>8}{'5d_mag':>9}"
        f"{'10d_wr':>8}{'10d_mag':>9}{'2d_wr':>8}"
    )
    for name, m in methods.items():
        ph = m["per_horizon"]
        row = f"    {name:<16}"
        for h in ("3d", "5d", "10d", "2d"):
            r = ph.get(h, {})
            row += f"{_fmt(r.get('winrate'), True):>8}{_fmt(r.get('mag'), True):>9}"
        print(row)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="池分 vs LEGACY prob 正交性 + 叠加 OOS 双头"
    )
    ap.add_argument("--window", default="6m", choices=list(OOS_WINDOWS))
    ap.add_argument(
        "--lag",
        type=int,
        default=0,
        help="将 OOS 窗整体前移 N 个交易日 (0=末尾窗; 3m+lag63=前一半, 稳定性检查用)",
    )
    ap.add_argument("--board", default=None, help="main/dual, 默认两者")
    ap.add_argument("--out", default=None, help="WORM JSON 路径")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from app.pipeline1.predict_runner import resolve_current_bundles
    from app.pipeline1.predictor import V35Predictor

    bundles = resolve_current_bundles(MODEL_DIR)
    if not bundles:
        print(f"无模型包: {MODEL_DIR}")
        return 1
    predictor = V35Predictor(bundles)
    print("模型包:", {b: os.path.basename(p) for b, p in bundles.items()})

    d = OOS_WINDOWS[args.window]
    out: dict = {
        "ts": "2026-08-05",
        "type": "overlay_orthogonality",
        "window": args.window,
        "trading_days": d,
        "weights": {"w_pool": W_POOL, "w_prob": W_PROB},
        "criteria": {b: t for b, t in BOARD_THRESHOLDS.items()},
        "boards": {},
    }
    for board in ["main", "dual"] if not args.board else [args.board]:
        print(f"\n========== 板块 [{board}] ==========", flush=True)
        sub = load_board(board)
        dates = np.sort(sub["date"].unique())
        start_idx = len(dates) - d - args.lag
        end_idx = len(dates) - args.lag
        oos_start, oos_end = dates[start_idx], dates[end_idx - 1]
        mask = (sub["date"].values >= oos_start) & (sub["date"].values <= oos_end)
        sub = sub[mask].copy()
        bcrit = (
            BOARD_THRESHOLDS[board]["min_winrate"],
            BOARD_THRESHOLDS[board]["min_mag"],
        )
        print(
            f"OOS{args.window} {pd.Timestamp(oos_start).date()} → "
            f"{pd.Timestamp(sub['date'].max()).date()} ({d} 交易日) | "
            f"行 {len(sub):,} | 阈值 wr>={bcrit[0]} mag>{bcrit[1]}",
            flush=True,
        )

        prob, pc = load_probs(board, predictor, oos_start)
        sub = sub.merge(prob, on=["date", "symbol"], how="left")
        cover = float(sub["prob_up"].notna().mean())
        print(f"LEGACY prob 列 = {pc} | OOS 截面覆盖 {cover:.1%}", flush=True)
        del prob
        gc.collect()

        board_out: dict = {"prob_col": pc, "prob_coverage": cover, "systems": {}}
        for name, spec in SYSTEMS.items():
            print(f"\n── 系统 [{name}] TOP-{spec.top_n} ──")
            s = sub.copy()
            s["pool_score"] = pool_score(s, spec.pool)
            pool_nopv = tuple(p for p in spec.pool if p != "pv_corr_5")
            s["pool_score_nopv"] = pool_score(s, pool_nopv)
            prob_f = s["prob_up"].fillna(0.0)
            s["combined"] = W_POOL * s["pool_score"] + W_PROB * prob_f
            s["combined_nopv"] = W_POOL * s["pool_score_nopv"] + W_PROB * prob_f

            # 自检: 池分方法 == run_system (sub 已是 OOS 切片 → mask=None)
            ref = run_system(sub, spec, spec.top_n, None, bcrit)
            mine = measure(s, s["pool_score"], spec, bcrit)
            ok = all(
                mine["per_horizon"].get(h, {}).get("n")
                == ref["per_horizon"].get(h, {}).get("n")
                and abs(
                    (mine["per_horizon"].get(h, {}).get("mag") or 0)
                    - (ref["per_horizon"].get(h, {}).get("mag") or 0)
                )
                < 1e-9
                for h in spec.horizons
            )
            print(f"  自检 池分==run_system: {'✓' if ok else '✗ 不一致!'}", flush=True)

            methods = {
                "pool": measure(s, s["pool_score"], spec, bcrit),
                "prob": measure(s, s["prob_up"], spec, bcrit),
                "combined": measure(s, s["combined"], spec, bcrit),
                "combined_nopv": measure(s, s["combined_nopv"], spec, bcrit),
            }
            ortho = orthogonality(s)
            print(
                f"  正交性 Spearman(池分, {pc}): avg={ortho['spearman_avg']} "
                f"std={ortho['spearman_std']} "
                f"[{ortho['spearman_min']}, {ortho['spearman_max']}] "
                f"({ortho['n_days']} 日)"
            )
            _print_methods(methods)

            # 增量汇总 (combined vs pool / vs prob / vs nopv) 只看主视界 3d/5d
            d3 = methods["combined"]["per_horizon"].get("3d", {})
            p3 = methods["pool"]["per_horizon"].get("3d", {})
            b3 = methods["prob"]["per_horizon"].get("3d", {})
            n3 = methods["combined_nopv"]["per_horizon"].get("3d", {})
            print(
                f"  3d: 叠加 vs 纯池 Δwr {_fmt(_safe_delta(d3.get('winrate'), p3.get('winrate')), True)} "
                f"| 叠加 vs 纯prob Δwr {_fmt(_safe_delta(d3.get('winrate'), b3.get('winrate')), True)} "
                f"| 含pv_corr_5 vs 去pv Δwr {_fmt(_safe_delta(d3.get('winrate'), n3.get('winrate')), True)}",
                flush=True,
            )
            board_out["systems"][name] = {
                "top_n": spec.top_n,
                "orthogonality": ortho,
                "selfcheck_pool_eq_run_system": bool(ok),
                "methods": methods,
            }
            del s
            gc.collect()

        out["boards"][board] = {
            "latest": str(sub["date"].max()),
            "systems": board_out["systems"],
        }
        del sub
        gc.collect()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print(f"\nWORM 落盘: {args.out}")
    return 0


def _safe_delta(a, b) -> float | None:
    if a is None or b is None:
        return None
    return float(a - b)


if __name__ == "__main__":
    raise SystemExit(main())
