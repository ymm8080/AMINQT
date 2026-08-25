"""诊断: 测量 legacy E7 准入在 p_reg 新概率下的逐板块通过数.

与 _gen_legacy_list.py 同构 (本地面板 + 300 日切片 + 当前 bundle), 但在
entry_filter 之前打印逐板块概率分布 / base_rate / 阈值 / 过闸数.
不修改生产代码. 用法: python scripts/_diag_e7_gate_counts.py [YYYYMMDD]
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from app.pipeline1.daily_pipeline import DailySelectionPipeline
from app.pipeline1.data_supply import DataSupplyChain
from config.settings import PANEL_V3_PATH

BUNDLES = {
    "main": "models/pipeline1/main_current.pkl",
    "dual": "models/pipeline1/dual_current.pkl",
}


def _instrument(pipe):
    orig = pipe.lister.entry_filter

    def wrapped(df, *a, **kw):
        if len(df):
            df = df.copy()
            if "board" in df.columns and "prob_up" in df.columns:
                cp = df.get("compound_prob", df["prob_up"])
                base = df["base_rate"] if "base_rate" in df.columns else cp.mean()
                for b, g in df.groupby("board"):
                    thr = base.loc[g.index].iloc[0] if hasattr(base, "loc") else base
                    margin = 0.08 if str(b) == "dual" else 0.0
                    gcp = cp.loc[g.index]
                    n_pass_prob = int((gcp > thr + margin).sum())
                    print(
                        f"[gate] board={b} n={len(g)} "
                        f"prob_up10d mean={gcp.mean():.4f} std={gcp.std():.4f} "
                        f"min={gcp.min():.4f} max={gcp.max():.4f} "
                        f"base_rate={thr:.4f} thr(base+margin={margin})={thr + margin:.4f} "
                        f"n_pass_prob={n_pass_prob} ({n_pass_prob / len(g):.1%})",
                        flush=True,
                    )
            else:
                print(f"[gate] df {len(df)}r 缺 board/prob_up 列", flush=True)
        return orig(df, *a, **kw)

    pipe.lister.entry_filter = wrapped


def main():
    trade_date = sys.argv[1] if len(sys.argv) > 1 else "20260804"
    t0 = time.time()
    panel = pd.read_parquet(str(PANEL_V3_PATH))
    print(f"[panel] {len(panel):,}r max={panel['date'].max():%Y-%m-%d} ({time.time() - t0:.0f}s)", flush=True)
    dates = sorted(panel["date"].unique())
    cut = dates[-300]
    panel = panel[panel["date"] >= cut]
    print(f"[slice] {cut.date()}..{dates[-1].date()} -> {len(panel):,}r", flush=True)
    pipe = DailySelectionPipeline(supply=DataSupplyChain(), bundle_paths=BUNDLES)
    _instrument(pipe)
    res = pipe.run(trade_date, panel=panel)
    lst = res.get("list")
    print(f"[done] mode={res.get('mode')} n={0 if lst is None else len(lst)}", flush=True)
    if lst is not None and len(lst):
        print("[list board counts]", lst["board"].value_counts().to_dict(), flush=True)
        print(lst.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
