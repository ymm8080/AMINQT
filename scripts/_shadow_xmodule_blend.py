"""_shadow_xmodule_blend.py — 跨模块影子排名 (legacy × parallel 合池混排, 2026-08-26).

用户批准的零风险实验 (module-diag: 两模块交集≈0, legacy-rank-transfer: 排名信息
主板互补): 每板块把两模块交付清单并池, 各自在自己板内名次百分位归一
(键 = 各模块生产排序键: legacy prob_up / parallel rank_blend), 加权混排取 TOP-N.
影子只落盘不交付, 积累数周后与两模块各自 TOP-10 对比前瞻收益定去留.

数据源 (全部 WORM 已有文件, 无新生产依赖):
  legacy   = DATA_DIR/lists/list_<D>*.parquet (含 _dual 变体, 同 symbol keep-last)
  parallel = STOCK_LIST_DIR/parallel_shortlist_<D>__*.csv (多版本排序后 keep-last;
             bare parallel_shortlist_<D>.csv 也算, 无 rank_blend 列的旧版跳过)
板组映射: legacy GEM/STAR → dual, main → main; parallel 本就 main/dual.

输出: DATA OTHERS/shadow/xmodule_blend_<D>__<ts>.csv (WORM)
  列: date, board, symbol, in_legacy, in_parallel, legacy_pct, parallel_pct,
      blend, shadow_rank, source (both/legacy/parallel)
用法: python scripts/_shadow_xmodule_blend.py [--date YYYYMMDD] [--backfill]
  --date   指定清单日 (默认今日, 自动前找最近有清单日)
  --backfill 回填所有双源齐备的历史清单日 (跳过已有影子文件的日期)
"""

from __future__ import annotations

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import (
    DATA_DIR,
    STOCK_LIST_DIR,
    XMODULE_SHADOW,
    data_others_path,
)

LEGACY_BOARD = {"main": "main", "GEM": "dual", "STAR": "dual"}

SHADOW_COLS = [
    "date", "board", "symbol", "in_legacy", "in_parallel",
    "legacy_pct", "parallel_pct", "blend", "shadow_rank", "source",
]


def load_legacy(date: str) -> pd.DataFrame | None:
    """list_<D>*.parquet → (symbol, board_group, key); 无文件 → None."""
    fps = sorted(glob.glob(str(DATA_DIR / "lists" / f"list_{date}*.parquet")))
    if not fps:
        return None
    parts = [
        pd.read_parquet(
            fp, columns=["symbol", "board", "prob_up"]
        ).assign(symbol=lambda d: d["symbol"].astype(str).str.zfill(6))
        for fp in fps
    ]
    d = pd.concat(parts, ignore_index=True).dropna(subset=["prob_up"])
    d = d.drop_duplicates(subset=["symbol"], keep="last")
    if d.empty:
        return None
    d["board_group"] = d["board"].map(LEGACY_BOARD)
    return d[["symbol", "board_group", "prob_up"]].rename(columns={"prob_up": "key"})


def load_parallel(date: str) -> pd.DataFrame | None:
    """parallel_shortlist_<D>__*.csv (含 bare 版) → (symbol, board, key=rank_blend)."""
    fps = sorted(
        glob.glob(str(STOCK_LIST_DIR / f"parallel_shortlist_{date}__*.csv"))
        + glob.glob(str(STOCK_LIST_DIR / f"parallel_shortlist_{date}.csv"))
    )
    parts = []
    for fp in fps:
        d = pd.read_csv(fp, dtype={"symbol": str})
        if "rank_blend" not in d.columns:
            continue
        parts.append(d[["symbol", "board", "rank_blend"]])
    if not parts:
        return None
    d = pd.concat(parts, ignore_index=True)
    d["symbol"] = d["symbol"].str.zfill(6)
    d = d.dropna(subset=["rank_blend"]).drop_duplicates(
        subset=["symbol"], keep="last"
    )
    if d.empty:
        return None
    return d[["symbol", "board", "rank_blend"]].rename(
        columns={"board": "board_group", "rank_blend": "key"}
    )


def build_shadow(
    legacy: pd.DataFrame | None,
    parallel: pd.DataFrame | None,
    weights: dict,
    top_n: int,
) -> pd.DataFrame:
    """双源清单 → 影子 TOP-N (纯函数). 缺席模块该侧 pct=0; 双缺 → 空.

    成员身份 (in_legacy/in_parallel) 由 symbol 集合判定, 与 pct 解耦 —
    板内末名 pct=0.0 不等于"不在该模块清单".
    """
    frames = []
    member = {
        "legacy": set(legacy["symbol"]) if legacy is not None and len(legacy) else set(),
        "parallel": set(parallel["symbol"]) if parallel is not None and len(parallel) else set(),
    }
    for name, src in (("legacy", legacy), ("parallel", parallel)):
        if src is None or src.empty:
            continue
        g = src.copy()
        r = g.groupby("board_group")["key"].rank(ascending=False, method="first")
        n = g.groupby("board_group")["key"].transform("size")
        g["pct"] = np.where(n == 1, 1.0, (n - r) / (n - 1))
        g["module"] = name
        frames.append(g[["symbol", "board_group", "pct", "module"]])
    if not frames:
        return pd.DataFrame(columns=SHADOW_COLS[1:])
    long = pd.concat(frames, ignore_index=True)
    wl, wp = float(weights["legacy"]), float(weights["parallel"])
    wide = long.pivot_table(
        index=["symbol", "board_group"], columns="module", values="pct"
    ).reset_index()
    for col in ("legacy", "parallel"):
        if col not in wide.columns:
            wide[col] = 0.0
    wide = wide.rename(columns={"legacy": "legacy_pct", "parallel": "parallel_pct"})
    wide["legacy_pct"] = wide["legacy_pct"].fillna(0.0)
    wide["parallel_pct"] = wide["parallel_pct"].fillna(0.0)
    wide["blend"] = wl * wide["legacy_pct"] + wp * wide["parallel_pct"]
    wide["in_legacy"] = wide["symbol"].isin(member["legacy"])
    wide["in_parallel"] = wide["symbol"].isin(member["parallel"])
    wide["source"] = np.where(
        wide["in_legacy"] & wide["in_parallel"], "both",
        np.where(wide["in_legacy"], "legacy", "parallel"),
    )
    out = []
    for board, g in wide.groupby("board_group"):
        g = g.sort_values(["blend", "symbol"], ascending=[False, True]).head(top_n)
        g = g.assign(board=board, shadow_rank=range(1, len(g) + 1))
        out.append(g)
    res = pd.concat(out, ignore_index=True)
    return res[SHADOW_COLS[1:]]


def run_date(date: str, ts: str, cfg: dict | None = None) -> str:
    """写一份影子清单 (WORM); 双源全缺 → 返回 ''. 非致命, 永不抛."""
    cfg = cfg or XMODULE_SHADOW
    try:
        legacy = load_legacy(date)
        parallel = load_parallel(date)
    except Exception as e:  # noqa: BLE001 — 影子步骤绝不挡自动化
        print(f"[shadow {date}] 读取失败: {e}", flush=True)
        return ""
    if legacy is None and parallel is None:
        print(f"[shadow {date}] 双源清单均缺, 跳过", flush=True)
        return ""
    res = build_shadow(legacy, parallel, cfg["weights"], int(cfg["top_n"]))
    if res.empty:
        print(f"[shadow {date}] 合池为空, 跳过", flush=True)
        return ""
    res.insert(0, "date", date)
    out_dir = data_others_path(cfg["out_root"])
    os.makedirs(str(out_dir), exist_ok=True)
    fn = f"xmodule_blend_{date}__{ts}.csv"
    res.to_csv(out_dir / fn, index=False, encoding="utf-8-sig")
    n_both = int((res["source"] == "both").sum())
    print(
        f"[shadow {date}] legacy {0 if legacy is None else len(legacy)} / "
        f"parallel {0 if parallel is None else len(parallel)} 票, 影子 {len(res)} 行 "
        f"(交集 {n_both}) → {fn}",
        flush=True,
    )
    return fn


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="跨模块影子排名 (纯记录)")
    ap.add_argument("--date", default=None, help="清单日 YYYYMMDD (默认今日)")
    ap.add_argument("--backfill", action="store_true", help="回填全部双源齐备历史日")
    args = ap.parse_args()
    cfg = XMODULE_SHADOW
    if not cfg.get("enable"):
        print("[shadow] 开关关闭, 退出", flush=True)
        return 0
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    if args.backfill:
        leg_days = {
            os.path.basename(p)[len("list_"):][:8]
            for p in glob.glob(str(DATA_DIR / "lists" / "list_*.parquet"))
            if os.path.basename(p)[len("list_"):][:8].isdigit()
        }
        par_days = {
            os.path.basename(p).split("__")[0][len("parallel_shortlist_"):]
            for p in glob.glob(str(STOCK_LIST_DIR / "parallel_shortlist_*.csv"))
        }
        done = {
            os.path.basename(p).split("__")[0][len("xmodule_blend_"):]
            for p in glob.glob(str(data_others_path(cfg["out_root"]) / "xmodule_blend_*.csv"))
        }
        days = sorted((leg_days & par_days) - done)
        print(f"[shadow] 回填 {len(days)} 日: {days[0] if days else '—'}..{days[-1] if days else '—'}", flush=True)
        for d in days:
            run_date(d, f"bf{ts}", cfg)
        return 0
    date = args.date or pd.Timestamp.now().strftime("%Y%m%d")
    run_date(date, ts, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
