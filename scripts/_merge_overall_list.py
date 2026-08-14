"""_merge_overall_list.py — 并行+慢牛 全模块整体名单 (模块标注 + 双模块共荐).

合并每日两份交付文件:
  - parallel_shortlist_<date>__<module>.csv    (狙击/融合, _shortlist_t5_t10 产物)
  - slowbull_pool_*_<date>__slow_bull_<ver>.csv (慢牛, pipeline_parallel.runner 产物)

每股标注命中模块 (sniper / fusion / slow_bull, 组合用 + 连接), 并给出
「并行∩慢牛 双模块共荐」标记列 both. 输出 WORM:
  STOCK_LIST_DIR/overall_shortlist_<date>__parallel+slow_bull.csv

排序: 双模块共荐股在前, 其后并行短名单按 pred_mag_10d 降序, 最后纯慢牛按 rk.
用法: python scripts/_merge_overall_list.py [YYYYMMDD, 默认最新 parallel_shortlist]
"""

from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import STOCK_LIST_DIR

GATE_COLS = ("过门", "制度门", "gate_pass")


def _latest_date() -> str:
    pats = sorted(glob.glob(str(STOCK_LIST_DIR / "parallel_shortlist_*.csv")))
    if not pats:
        sys.exit("无 parallel_shortlist_*.csv, 先跑 _shortlist_t5_t10.py")
    # 文件名 parallel_shortlist_YYYYMMDD__<module>.csv
    return os.path.basename(pats[-1]).split("_", 2)[2].split("__", 1)[0]


def load_parallel(date: str) -> pd.DataFrame:
    pats = glob.glob(str(STOCK_LIST_DIR / f"parallel_shortlist_{date}__*.csv"))
    if not pats:
        sys.exit(f"无 parallel_shortlist_{date}__*.csv")
    sl = pd.read_csv(pats[0], dtype={"symbol": str})
    # 2026-08-14 入选收紧至每板块 TOP-5 → 交付 CSV 仅 cut=T-5 行; 按 board+symbol 去重兜底
    sl = sl.drop_duplicates(subset=["board", "symbol"]).copy()
    sl["module_parallel"] = (
        sl.get("systems").fillna("") if "systems" in sl.columns else ""
    )
    return sl


def load_slowbull(date: str) -> pd.DataFrame:
    pats = glob.glob(str(STOCK_LIST_DIR / f"slowbull_pool_*_{date}__slow_bull_*.csv"))
    if not pats:
        return pd.DataFrame(columns=["board", "symbol"])
    sb = pd.concat(
        [pd.read_csv(p, dtype={"symbol": str}) for p in pats],
        ignore_index=True,
    )
    return sb[["board", "symbol"]].drop_duplicates().copy()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    date = sys.argv[1] if len(sys.argv) > 1 else _latest_date()
    sl = load_parallel(date)
    sb = load_slowbull(date)
    sb_keys = set(zip(sb["board"], sb["symbol"]))

    rows = []
    for _, r in sl.iterrows():
        key = (r["board"], r["symbol"])
        in_sb = key in sb_keys
        mod = (r["module_parallel"] or "").strip("+")
        if not mod:
            mod = "parallel"  # 并行短名单候选, 但非 sniper/fusion 旧 top-N
        module = "+".join(filter(None, (mod, "slow_bull"))) if in_sb else mod
        rows.append(
            {
                "date": date,
                "board": r["board"],
                "symbol": r["symbol"],
                "module": module,
                "both": bool(in_sb),
                "score": r.get("score"),
                "pred_mag_3d": r.get("pred_mag_3d"),
                "pred_prob_3d": r.get("pred_prob_3d"),
                "pred_mag_5d": r.get("pred_mag_5d"),
                "pred_prob_5d": r.get("pred_prob_5d"),
                "pred_mag_10d": r.get("pred_mag_10d"),
                "pred_prob_10d": r.get("pred_prob_10d"),
                "slow_bull_rk": np.nan,
                "parallel_rk": r.get("rank"),
                "gate": r.get(_existing_gate(r)),
            }
        )
    for _, r in sb.iterrows():
        key = (r["board"], r["symbol"])
        if key in set(zip(sl["board"], sl["symbol"])):
            continue  # 已在并行行合并
        rows.append(
            {
                "date": date,
                "board": r["board"],
                "symbol": r["symbol"],
                "module": "slow_bull",
                "both": False,
                "score": np.nan,
                "pred_mag_3d": np.nan,
                "pred_prob_3d": np.nan,
                "pred_mag_5d": np.nan,
                "pred_prob_5d": np.nan,
                "pred_mag_10d": np.nan,
                "pred_prob_10d": np.nan,
                "slow_bull_rk": np.nan,
                "parallel_rk": np.nan,
                "gate": np.nan,
            }
        )
    out = pd.DataFrame(rows)
    # 排序: 双模块共荐在前 → 并行按 pred_mag_10d 降序 → 纯慢牛按池内 rk
    both_df = out[out["both"]]
    par_df = out[~out["both"] & out["pred_mag_10d"].notna()].sort_values(
        "pred_mag_10d", ascending=False
    )
    sb_df = out[~out["both"] & out["pred_mag_10d"].isna()]
    out = pd.concat([both_df, par_df, sb_df], ignore_index=True)
    out["rank"] = range(1, len(out) + 1)
    # 慢牛 rk 从 slowbull 池带过来 (并行行置 NaN; 纯慢牛按池 rk)
    sb_rk = pd.concat(
        [
            pd.read_csv(p, dtype={"symbol": str})
            for p in glob.glob(
                str(STOCK_LIST_DIR / f"slowbull_pool_*_{date}__slow_bull_*.csv")
            )
        ],
        ignore_index=True,
    )[["board", "symbol", "rk"]]
    rk_map = dict(zip(zip(sb_rk["board"], sb_rk["symbol"]), sb_rk["rk"]))
    out["slow_bull_rk"] = out.apply(
        lambda r: rk_map.get((r["board"], r["symbol"]), np.nan), axis=1
    )
    cols = [
        "date",
        "rank",
        "board",
        "symbol",
        "module",
        "both",
        "score",
        "pred_mag_3d",
        "pred_prob_3d",
        "pred_mag_5d",
        "pred_prob_5d",
        "pred_mag_10d",
        "pred_prob_10d",
        "slow_bull_rk",
        "parallel_rk",
        "gate",
    ]
    out = out[[c for c in cols if c in out.columns]]
    fp = STOCK_LIST_DIR / f"overall_shortlist_{date}__parallel+slow_bull.csv"
    out.to_csv(fp, index=False, encoding="utf-8-sig")
    n_both = int(out["both"].sum())
    print(f"[saved] {fp}")
    print(
        f"[overall] 共 {len(out)} 只 = 并行 {len(sl)} + 慢牛 {len(sb)} | "
        f"双模块共荐 {n_both} 只"
    )
    if n_both:
        print(
            "[both]",
            ", ".join(
                f"{r['board']}:{r['symbol']}" for _, r in out[out["both"]].iterrows()
            ),
        )
    return


def _existing_gate(r) -> str | None:
    for c in GATE_COLS:
        if c in r.index and not pd.isna(r.get(c)):
            return c
    return None


if __name__ == "__main__":
    main()
