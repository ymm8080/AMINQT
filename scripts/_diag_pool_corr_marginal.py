# -*- coding: utf-8 -*-
"""池内相关矩阵 + 相关因子 OOS 边际测试 (2026-08-05).

两段:
  1. 池内相关矩阵: 每板块 OOS 6m 窗, 对 sniper/fusion 池特征算 Pearson 相关,
     诊断共线性/单维度集中度 (6 特征 ≈ 几个独立维度?).
  2. OOS 边际: 基池 vs 基池+相关因子 (pv_corr_5 / pv_sync_direct_5d|20d /
     pv_sync_5d|20d), 逐视界双头对比 — 加相关因子是否带来 OOS 边际增益.

只测 sniper/fusion (slow_bull 的 ADX 因子已是量价相关族, 不需要边际). 轻量加载:
复用检查点 → _finalize_slice → add_mfe_labels → tradability_gate, 不做 prepare_adx.

输出 (WORM): data/_diag_pool_corr_marginal_<ts>.json
用法: python scripts/_diag_pool_corr_marginal.py [--window 6m] [--board main]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import replace

from app.pipeline_parallel.backtest import (add_mfe_labels, run_system,
                                            tradability_gate)
from app.pipeline_parallel.config import (ALL_HORIZON_INTS, BOARD_THRESHOLDS,
                                          FUSION, OOS_WINDOWS, PANEL, SNIPER)

POOL_SYSTEMS = {"sniper": SNIPER, "fusion": FUSION}
# 相关因子候选 (量价相关族, PIT)
CORR_FACTORS = ("pv_corr_5", "pv_sync_direct_5d", "pv_sync_direct_20d",
                "pv_sync_5d", "pv_sync_20d")
HIGH_CORR = 0.5  # |r| 高于此视为强共线 (冗余)

# 本测试只需的列 (池特征 ∪ 相关因子 ∪ MFE 标签输入); 全 548 列载入会 OOM
_NEEDED = (
    "symbol", "date", "close_hfq", "high_hfq", "adv20", "volume",
    "amihud_illiq", "small_mv_premium", "amihud_illiquidity",
    "down_gap_pct", "VAR51", "ret_reversal_5d", "limit_dist_pct",
    "pv_sync_direct_5d", "pv_sync_direct_20d", "pv_sync_5d", "pv_sync_20d",
)


def load_board(board: str) -> pd.DataFrame:
    """轻量加载单板块行集 (只读所需列 + MFE 标签 + 可交易性门 + board 列).

    不复用 _finalize_slice (其生产 10d 标签非本测试所需), 直接 add_mfe_labels —
    add_mfe_labels 自含 adv20 滑点成本, 与生产口径一致.
    """
    ckpt = PANEL.main_checkpoint if board == "main" else PANEL.dual_checkpoint
    df = pd.read_parquet(ckpt, columns=list(_NEEDED))
    df = add_mfe_labels(df, horizons=ALL_HORIZON_INTS)
    df, _ = tradability_gate(df)
    df["board"] = board
    df = _add_pv_corr_5(df)
    return df


def _add_pv_corr_5(df: pd.DataFrame) -> pd.DataFrame:
    """量价相关 5 日 (ret 与 vol 变化率的滚动 Pearson 相关), PIT, 逐股 groupby."""
    df = df.sort_values(["symbol", "date"])
    ret = df["close_hfq"].groupby(df["symbol"], sort=False).pct_change()
    vol = df["volume"].groupby(df["symbol"], sort=False).pct_change()
    d = pd.concat([ret.rename("ret"), vol.rename("vol")], axis=1)
    # 逐股滚动 5 日相关; .corr() 返回 (symbol, date, col, col) 多级索引矩阵
    cc = d.groupby(df["symbol"], sort=False).rolling(
        5, min_periods=5).corr()
    # 每个窗口取 (ret, vol) 对的相关系数: 行索引奇数行(col 对全矩阵) 取第 1 列
    pv = cc.iloc[0::2, 1]                      # ret-vol 对
    pv = pv.reset_index(level=[0, 2], drop=True)
    df["pv_corr_5"] = pv
    del cc, pv
    gc.collect()
    return df


def pool_corr_matrix(sub: pd.DataFrame, pool: tuple[str, ...]) -> dict:
    """池特征在子窗 (OOS) 上的 Pearson 相关矩阵 + 集中度指标."""
    cols = [c for c in pool if c in sub.columns]
    corr = sub[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    pairs = corr.where(mask).stack()
    pairs = pairs[np.isfinite(pairs)]
    return {
        "pool": list(pool),
        "corr": {f"{i}|{j}": round(float(corr.loc[i, j]), 4)
                 for i in cols for j in cols if i != j},
        "avg_abs_pair": round(float(pairs.abs().mean()), 4) if len(pairs) else None,
        "max_abs_pair": round(float(pairs.abs().max()), 4) if len(pairs) else None,
        "high_corr_pairs": sorted(
            [f"{i}~{j}={corr.loc[i, j]:.2f}"
             for i in cols for j in cols if i != j
             and abs(corr.loc[i, j]) >= HIGH_CORR]),
        "n_eff": round(float(1 / (1 + 2 * sum(
            corr.loc[i, j] ** 2 for i in cols for j in cols if i != j))), 3)
        if len(cols) else None,
    }


def marginal(sub: pd.DataFrame, spec, factor: str, oos_mask,
             bcrit: tuple[float, float]) -> dict:
    """基池 vs 基池+factor 逐视界双头对比 (OOS)."""
    base = run_system(sub, spec, spec.top_n, oos_mask, bcrit)
    aug = run_system(sub, replace(spec, pool=spec.pool + (factor,)),
                     spec.top_n, oos_mask, bcrit)
    rows = {}
    for h in spec.horizons:
        b, a = base["per_horizon"].get(h, {}), aug["per_horizon"].get(h, {})
        rows[h] = {
            "base": {"mag": b.get("mag"), "winrate": b.get("winrate"),
                     "n": b.get("n"), "ok": b.get("ok")},
            "aug": {"mag": a.get("mag"), "winrate": a.get("winrate"),
                    "n": a.get("n"), "ok": a.get("ok")},
        }
        if b.get("n") and a.get("n"):
            rows[h]["delta_wr"] = round(float(a["winrate"] - b["winrate"]), 4)
            rows[h]["delta_mag"] = round(float(a["mag"] - b["mag"]), 6)
            rows[h]["n_drop"] = int(b["n"] - a["n"])
    return {
        "factor": factor,
        "base_pool": list(spec.pool),
        "base_passed": base["passed"],
        "aug_passed": aug["passed"],
        "new_pass": sorted(set(aug["passed"]) - set(base["passed"])),
        "lost_pass": sorted(set(base["passed"]) - set(aug["passed"])),
        "base_npicks": base["n_picks"],
        "aug_npicks": aug["n_picks"],
        "per_horizon": rows,
    }


def _fmt(v: float, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v:+.1%}" if pct else f"{v:.3f}"


def _print_marginal(m: dict) -> None:
    print(f"\n  +因子 {m['factor']}: 基池通过 {m['base_passed'] or '无'} "
          f"→ 加后通过 {m['aug_passed'] or '无'} "
          f"(新增 {m['new_pass'] or '无'}, 丢失 {m['lost_pass'] or '无'})")
    print(f"    {'视界':<5}{'基mag':>9}{'加mag':>9}{'Δmag':>9}"
          f"{'基wr':>8}{'加wr':>8}{'Δwr':>8}{'n基':>6}{'n加':>6}")
    for h, r in m["per_horizon"].items():
        b, a = r["base"], r["aug"]
        print(f"    {h:<5}{_fmt(b['mag'],True):>9}{_fmt(a['mag'],True):>9}"
              f"{_fmt(r.get('delta_mag'),True):>9}{_fmt(b['winrate'],True):>8}"
              f"{_fmt(a['winrate'],True):>8}{_fmt(r.get('delta_wr'),True):>8}"
              f"{b['n'] if b['n'] else '-':>6}{a['n'] if a['n'] else '-':>6}")


def main() -> int:
    ap = argparse.ArgumentParser(description="池内相关矩阵 + 相关因子 OOS 边际")
    ap.add_argument("--window", default="6m", choices=list(OOS_WINDOWS))
    ap.add_argument("--board", default=None, help="main/dual, 默认两者")
    ap.add_argument("--out", default=None, help="WORM JSON 路径")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    d = OOS_WINDOWS[args.window]
    out: dict = {"ts": "2026-08-05", "type": "pool_corr_marginal",
                 "window": args.window, "trading_days": d,
                 "criteria": {b: t for b, t in BOARD_THRESHOLDS.items()},
                 "boards": {}}
    for board in (["main", "dual"] if not args.board else [args.board]):
        print(f"\n========== 板块 [{board}] ==========", flush=True)
        sub = load_board(board)
        dates = np.sort(sub["date"].unique())
        oos_mask = sub["date"].values >= dates[-d]
        bcrit = (BOARD_THRESHOLDS[board]["min_winrate"],
                 BOARD_THRESHOLDS[board]["min_mag"])
        print(f"行 {len(sub):,} | OOS{args.window} "
              f"{pd.Timestamp(dates[-d]).date()} → "
              f"{pd.Timestamp(sub['date'].max()).date()} "
              f"({d} 交易日) | 阈值 wr>={bcrit[0]} mag>{bcrit[1]}", flush=True)

        corr_out: dict = {}
        marginal_out: dict = {}
        for name, spec in POOL_SYSTEMS.items():
            print(f"\n── 系统 [{name}] 池相关矩阵 (OOS) ──")
            cm = pool_corr_matrix(sub[oos_mask], spec.pool)
            corr_out[name] = cm
            cols = list(cm["corr"])
            print(f"  avg|r|={cm['avg_abs_pair']} max|r|={cm['max_abs_pair']} "
                  f"n_eff≈{cm['n_eff']} | 强共线: "
                  f"{cm['high_corr_pairs'] or '无'}")
            # 打印矩阵 (唯一列名)
            uniq = list(dict.fromkeys([c.split('|')[0] for c in cols]))
            mat = sub[oos_mask][uniq].corr()
            print("  " + "  ".join(f"{c:>14}" for c in uniq))
            for c in uniq:
                row = mat[c]
                print(f"  {c:>14}" + "".join(
                    f"{row[o]:>14.3f}" for o in uniq))

            print(f"\n── 系统 [{name}] 相关因子 OOS 边际 (TOP-{spec.top_n}) ──")
            marg_out = []
            for factor in CORR_FACTORS:
                if factor not in sub.columns:
                    print(f"  (跳过 {factor}: 面板无此列)")
                    continue
                m = marginal(sub, spec, factor, oos_mask, bcrit)
                marg_out.append(m)
                _print_marginal(m)
            marginal_out[name] = marg_out

        out["boards"][board] = {
            "rows": int(len(sub)), "latest": str(sub["date"].max()),
            "pool_corr": corr_out, "marginal": marginal_out}
        del sub
        gc.collect()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print(f"\nWORM 落盘: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
