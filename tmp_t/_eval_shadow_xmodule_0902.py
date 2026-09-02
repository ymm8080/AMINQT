# -*- coding: utf-8 -*-
"""_eval_shadow_xmodule_0902.py — 跨模块影子排名积累样本终审 (2026-09-02).

对比三臂 per (date, board) 的 T+3 close-to-close 实现收益均值:
  shadow  = xmodule_blend_<D>__*.csv 的 blend TOP10 (0.5*legacy_pct + 0.5*parallel_pct)
  legacy  = in_legacy 行按 legacy_pct (生产键 prob_up) TOP10
  parallel= in_parallel 行按 parallel_pct (生产键 rank_blend) TOP10
同一候选池三臂配对, 差值可归因于排名键. 不足 3 个后续交易日的清单日跳过.

WORM: data/others/_shadow_xmodule_eval_<ts>.json
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config.settings import data_others_path

PANEL = os.getenv("PANEL_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")
SHADOW_DIR = data_others_path("shadow")
TOPN = 10
H = 3  # 前瞻交易日 (与交付主视界 T+3 一致)


def load_fwd() -> tuple[pd.DataFrame, list, dict]:
    hist = pq.read_table(PANEL, columns=["symbol", "date", "close_hfq"]).to_pandas()
    hist["symbol"] = hist["symbol"].astype(str).str.zfill(6)
    days = sorted(hist["date"].unique())
    pos = {d: i for i, d in enumerate(days)}
    fwd = {}
    for d, g in hist.groupby("date"):
        i = pos[d]
        if i + H < len(days):
            base = g.set_index("symbol")["close_hfq"]
            tgt = hist[hist["date"] == days[i + H]].set_index("symbol")["close_hfq"]
            r = (tgt / base - 1).dropna()
            fwd[d] = r
    return hist, days, fwd


def arm_top(df: pd.DataFrame, src_col: str, key_col: str) -> set:
    sub = df[df[src_col] == True].nlargest(TOPN, key_col)  # noqa: E712
    return set(sub["symbol"])


def main() -> int:
    files = sorted(glob.glob(str(SHADOW_DIR / "xmodule_blend_*.csv")))
    # 同日多版本 keep-last
    by_date: dict[str, str] = {}
    for fp in files:
        d = os.path.basename(fp).split("__")[0][len("xmodule_blend_") :]
        by_date[d] = fp  # sorted 顺序 → 后者覆盖

    _, _, fwd = load_fwd()
    rows = []
    for d, fp in sorted(by_date.items()):
        df = pd.read_csv(fp, dtype={"symbol": str})
        df["symbol"] = df["symbol"].str.zfill(6)
        dt = pd.Timestamp(d)
        if dt not in fwd:
            continue
        r = fwd[dt]
        for board, g in df.groupby("board"):
            shadow = set(g.nlargest(TOPN, "blend")["symbol"])
            legacy = arm_top(g, "in_legacy", "legacy_pct")
            parallel = arm_top(g, "in_parallel", "parallel_pct")
            rec = {"date": d, "board": board}
            for name, picks in (("shadow", shadow), ("legacy", legacy), ("parallel", parallel)):
                vals = [r[s] for s in picks if s in r.index]
                rec[name] = float(np.mean(vals)) if vals else np.nan
                rec[f"{name}_n"] = len(vals)
            rec["both_in_shadow"] = int(
                len([s for s in shadow if s in r.index and
                     g.set_index("symbol").at[s, "source"] == "both"])
            )
            rows.append(rec)

    ev = pd.DataFrame(rows)
    out: dict = {"ts": pd.Timestamp.now().isoformat(), "days": int(ev["date"].nunique()),
                 "horizon": f"T+{H} c2c"}
    for board, g in ev.groupby("board"):
        arms = {}
        for name in ("shadow", "legacy", "parallel"):
            s = g.dropna(subset=[name])["date"].tolist()
            vals = g.dropna(subset=[name]).set_index("date")[name]
            half = len(vals) // 2
            arms[name] = {
                "days": int(len(vals)),
                "mean": float(vals.mean()),
                "h1": float(vals.iloc[:half].mean()),
                "h2": float(vals.iloc[half:].mean()),
                "hit": float((vals > 0).mean()),
            }
        # 配对差 (同日同板可比)
        for a, b in (("shadow", "legacy"), ("shadow", "parallel")):
            common = g.dropna(subset=[a, b]).set_index("date")
            dif = common[a] - common[b]
            half = len(dif) // 2
            arms[f"d_{a}-{b}"] = {
                "days": int(len(dif)),
                "mean": float(dif.mean()),
                "h1": float(dif.iloc[:half].mean()),
                "h2": float(dif[half:].mean()),
                "win": float((dif > 0).mean()),
            }
        arms["shadow_both_cnt"] = float(g["both_in_shadow"].mean())
        out[str(board)] = arms

    out_path = data_others_path(
        f"_shadow_xmodule_eval_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {out_path}")
    for board in ("main", "dual"):
        if board not in out:
            continue
        print(f"\n== {board} ==")
        for name, st in out[board].items():
            if isinstance(st, float):
                print(f"  both_in_shadow/日 = {st:.2f}")
                continue
            print(f"  {name:16s} days={st['days']:3d} mean={st['mean']:+.4f} "
                  f"(h1 {st['h1']:+.4f} / h2 {st['h2']:+.4f})"
                  + (f" win={st['win']:.2f}" if "win" in st else f" hit={st['hit']:.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
