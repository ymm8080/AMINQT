"""_verify_overlay_track.py — 前向跟踪验证: overlay 快照 × 已实现 MFE (2026-08-05).

用户批准轻量前向跟踪 ("改动正不正两周后给实盘答案"): 每次 legacy_overlay 交付时
把当日叠加再排名单快照落盘 (STOCK_LIST_DIR/overlay_track_{date}__{module}.csv,
列集见 legacy_overlay.SNAPSHOT_COLS, 含 w_pool/w_prob 实际应用权重). 本脚本收集
全部快照, 按 (symbol, date) join 面板已实现 MFE 净标签 (label_mfe_{2,3,5,10}d_net),
逐 (board, date) 检验: final_score/prob 高排名是否真对应更高已实现 MFE 与上涨率.

  - main (0.2/0.8, prob 主导) vs dual (0.5/0.5, 对半) 是两个自然实验组 —
    快照已 stamp w_pool/w_prob, 可分组分开看.
  - 距选股日不足 h+1 交易日的行 label 为 NaN (未来价不存在, add_mfe_labels
    skipna=False), 天数不足自动标 n/a.

用法: python scripts/_verify_overlay_track.py [--snap-dir STOCK_LIST_DIR] [--out ...]
输出 (WORM): data/_overlay_track_verify_<ts>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import DATA_DIR, STOCK_LIST_DIR

# pyarrow / backtest / PANEL 惰性导入 (load_realized 内部) — 依赖链较重,
# 测试只调 per_date_summary / load_snapshots, 避免收集时 ImportError.

HORIZONS = ("2d", "3d", "5d", "10d")
MFE_COL = {h: f"label_mfe_{h}_net" for h in HORIZONS}
_MFE_NEEDED = ("symbol", "date", "close_hfq", "high_hfq", "adv20")


def _m(v) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), 6)


def _spearman(a: pd.Series | None, b: pd.Series | None, min_n: int = 10):
    if a is None or b is None:
        return None
    s = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(s) < min_n:
        return None
    r = spearmanr(s["a"], s["b"])
    stat = r.statistic if hasattr(r, "statistic") else r[0]
    return round(float(stat), 4) if np.isfinite(stat) else None


def load_snapshots(snap_dir: Path) -> pd.DataFrame:
    """收集 STOCK_LIST_DIR/overlay_track_*.csv 全部快照."""
    files = sorted(snap_dir.glob("overlay_track_*.csv"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(f, dtype={"symbol": str}) for f in files]
    snap = pd.concat(frames, ignore_index=True)
    snap["date"] = snap["date"].astype(str)
    return snap


def load_realized(snap: pd.DataFrame, board: str) -> pd.DataFrame:
    """快照 × 已实现 MFE: 只读该板块检查点中快照涉及 symbol 的行 (轻量列读).

    对 (symbol, date) join; 未来价不足 → label NaN (add_mfe_labels skipna=False).
    """
    import pyarrow.parquet as pq

    from app.pipeline_parallel.backtest import add_mfe_labels
    from app.pipeline_parallel.config import PANEL

    syms = list(snap.loc[snap["board"] == board, "symbol"].unique())
    if not syms:
        return pd.DataFrame()
    ckpt = PANEL.main_checkpoint if board == "main" else PANEL.dual_checkpoint
    t = pq.read_table(
        str(ckpt), columns=list(_MFE_NEEDED), filters=[("symbol", "in", syms)]
    )
    t = t.to_pandas()
    t["symbol"] = t["symbol"].astype(str)
    t["date"] = t["date"].astype(str)
    t = add_mfe_labels(t, horizons=(2, 3, 5, 10))
    keep = ["symbol", "date"] + list(MFE_COL.values())
    t = t[keep]
    sub = snap[snap["board"] == board].copy()
    merged = sub.merge(t, on=["symbol", "date"], how="left")
    return merged


def per_date_summary(merged: pd.DataFrame) -> dict:
    """逐 (board, date): final_score 分上下半区, 对比已实现 MFE 与上涨率.

    另给整表 3d Spearman(final_score / prob_up, 已实现 MFE) 作排名-收益一致性.
    """
    rows: dict = {}
    for (board, date), g in merged.groupby(["board", "date"], sort=True):
        g = g.sort_values("final_score", ascending=False).reset_index(drop=True)
        n = len(g)
        half = max(1, n // 2)
        top, bot = g.iloc[:half], g.iloc[half:]
        per: dict = {}
        for h in HORIZONS:
            tv, bv = top[MFE_COL[h]].dropna(), bot[MFE_COL[h]].dropna()
            per[h] = {
                "n": int(len(tv) + len(bv)),
                "top_mfe": _m(tv.mean() if len(tv) else None),
                "bot_mfe": _m(bv.mean() if len(bv) else None),
                "top_wr": _m((tv > 0).mean() if len(tv) else None),
                "bot_wr": _m((bv > 0).mean() if len(bv) else None),
                "top_minus_bot": _m(
                    (tv.mean() - bv.mean()) if len(tv) and len(bv) else None
                ),
            }
        rows[f"{board}|{date}"] = {
            "n": n,
            "half": half,
            "w_pool": _m(float(g["w_pool"].iloc[0])),
            "w_prob": _m(float(g["w_prob"].iloc[0])),
            "prob_col": str(g["prob_col"].iloc[0]),
            "per_horizon": per,
            "spearman_3d": {
                # 单日期 n≈10-15, 门槛 5 即可当方向参考 (非统计检验)
                "final_score": _spearman(g["final_score"], g[MFE_COL["3d"]], min_n=5),
                "prob_up": _spearman(g["prob_up"], g[MFE_COL["3d"]], min_n=5),
            },
        }
    return rows


def board_pooled(merged: pd.DataFrame) -> dict:
    """板块级聚合 (跨日期合并): 排名-已实现 MFE 一致性 + 上/下半区净差."""
    out = {}
    for board, g in merged.groupby("board", sort=True):
        g = g.sort_values("final_score", ascending=False)
        half = max(1, len(g) // 2)
        top, bot = g.iloc[:half], g.iloc[half:]
        tv3, bv3 = top[MFE_COL["3d"]].dropna(), bot[MFE_COL["3d"]].dropna()
        out[board] = {
            "n": int(len(g)),
            "n_realized_3d": int(tv3.notna().sum() + bv3.notna().sum()),
            "top_mfe_3d": _m(tv3.mean() if len(tv3) else None),
            "bot_mfe_3d": _m(bv3.mean() if len(bv3) else None),
            "top_wr_3d": _m((tv3 > 0).mean() if len(tv3) else None),
            "top_minus_bot_3d": _m(
                (tv3.mean() - bv3.mean()) if len(tv3) and len(bv3) else None
            ),
            "spearman_3d": {
                "final_score": _spearman(g["final_score"], g[MFE_COL["3d"]]),
                "prob_up": _spearman(g["prob_up"], g[MFE_COL["3d"]]),
            },
        }
    return out


def fmt_rows(rows: dict, pooled: dict) -> list[str]:
    hdr = (
        f"{'board|date':<26}{'n':>4}{'w_pool':>7}{'w_prob':>7}"
        + "  ".join(f"{'T+' + h[:-1]:>12}" for h in HORIZONS)
        + f" {'spF':>6} {'spP':>6}"
    )
    lines = [hdr]
    for key, r in rows.items():
        cells = []
        for h in HORIZONS:
            p = r["per_horizon"][h]
            seg = "-" if p["n"] == 0 else f"{p['top_mfe']:+.1%}/{p['bot_mfe']:+.1%}"
            cells.append(f"{seg:>12}")
        lines.append(
            f"{key:<26}{r['n']:>4}{r['w_pool']:>7}{r['w_prob']:>7}"
            + "".join(cells)
            + f" {str(r['spearman_3d']['final_score']):>6}"
            f" {str(r['spearman_3d']['prob_up']):>6}"
        )
    for board, p in pooled.items():
        f_t, f_b = _m(p["top_mfe_3d"]), _m(p["bot_mfe_3d"])
        f_d = _m(p["top_minus_bot_3d"])
        f_w = _m(p["top_wr_3d"])
        seg = (
            "-"
            if f_t is None
            else f"top vs bot 3d MFE {f_t:+.1%} vs {f_b:+.1%} "
            f"(Δ{f_d:+.1%}, wr {f_w:.0%})"
        )
        lines.append(
            f"[{board}] 合并 n={p['n']} (3d 已实现 {p['n_realized_3d']}) {seg} "
            f"sp(F,MFE)={p['spearman_3d']['final_score']} "
            f"sp(P,MFE)={p['spearman_3d']['prob_up']}"
        )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description="overlay 快照 × 已实现 MFE 前向验证")
    ap.add_argument("--snap-dir", default=str(STOCK_LIST_DIR), help="快照目录")
    ap.add_argument("--out", default=None, help="WORM JSON 路径")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    snap = load_snapshots(Path(args.snap_dir))
    if snap.empty:
        print("[info] 无 overlay_track_*.csv 快照 (legacy_overlay 交付后才有数据)")
        return 0
    merged = pd.concat(
        [load_realized(snap, b) for b in ("main", "dual")], ignore_index=True
    )
    rows = per_date_summary(merged)
    pooled = board_pooled(merged)
    out = {
        "ts": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
        "type": "overlay_track_verify",
        "n_snapshot_days": int(snap["date"].nunique()),
        "n_rows": int(len(snap)),
        "per_date": rows,
        "board_pooled": pooled,
    }
    if args.out is None:
        args.out = str(DATA_DIR / f"_overlay_track_verify_{out['ts']}.json")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"WORM 落盘: {args.out}")
    print("\n".join(fmt_rows(rows, pooled)))
    print(
        "\n说明: top/bot 为 final_score 上/下半区; 每格 = top_mfe/bot_mfe (T+ 视界); "
        "spF/spP = Spearman(final_score|prob_up, 已实现3d MFE); 距选股日不足 h+1 "
        "交易日 → 该视界 n/a ('-'). main(0.2/0.8) vs dual(0.5/0.5) 两组自然对照."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
