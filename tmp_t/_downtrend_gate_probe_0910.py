# -*- coding: utf-8 -*-
"""tmp research: downtrend-kill gate admission probe (688228 case 2026-09-10).

688228 四连阴缩量下跌 -9.3% 却入 09-09 legacy 清单 #8 (pred_10d=+5.34%).
按闸准入协议 (GATE_ADMISSION) 测多个下跌趋势 kill 规则在交付清单回放上的判词.
复用 scripts/_gate_admission.py 的 load_lists/build_records/evaluate_gate.
只读研究, 不接线.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.settings import GATE_ADMISSION
from scripts._gate_admission import (
    _CAL_LOOKBACK,
    _pivots,
    build_records,
    evaluate_gate,
    load_lists,
)

VARIANTS = {
    "ret5<0": lambda pv: pv["c"].pct_change(5, fill_method=None) < 0,
    "close<ma5": lambda pv: pv["c"] < pv["c"].rolling(5, min_periods=3).mean(),
    "ret5<0 & close<ma5": lambda pv: (pv["c"].pct_change(5, fill_method=None) < 0)
    & (pv["c"] < pv["c"].rolling(5, min_periods=3).mean()),
    "3连阴": lambda pv: (pv["c"].pct_change(fill_method=None) < 0).rolling(3).sum() == 3,
    "ret5<-0.03": lambda pv: pv["c"].pct_change(5, fill_method=None) < -0.03,
}


def main():
    L = load_lists(lines=("legacy", "parallel"))
    print(f"[lists] {L['list_date'].nunique()} days, {len(L)} picks")
    d0 = pd.Timestamp(L["list_date"].min()) - pd.Timedelta(days=_CAL_LOOKBACK)
    from config.settings import PANEL_V3_PATH

    pv = _pivots(PANEL_V3_PATH, d0)
    print(f"[panel] {pv['c'].shape}")
    for name, fn in VARIANTS.items():
        factor = fn(pv).astype(float)
        recs = build_records(L, pv, factor)
        recs["killed"] = recs["factor"] == 1.0
        rep = evaluate_gate(recs, GATE_ADMISSION)
        print(
            f"\n=== {name}: {rep['verdict']} "
            f"(日{rep['n_days']} 删{rep['n_killed']} 删率{rep['kill_rate']:.1%})"
        )
        print(
            f"  全窗差 {rep['edge_full_pp']:+.2f}pp | "
            f"前半 {rep['edge_front_pp']:+.2f}pp / 后半 {rep['edge_back_pp']:+.2f}pp | "
            f"删区ret5均值 {rep['killed_ret5_mean']:+.3%} vs 留 {rep['kept_ret5_mean']:+.3%}"
        )
        wl = rep["winner_leak"].get("10%")
        if wl and wl["enrich"] is not None:
            print(f"  大赢家误杀富集 {wl['enrich']:.2f}x (leak {wl['leak']:.1%})")
        for rs in rep["reasons"]:
            print(f"  [reason] {rs}")


if __name__ == "__main__":
    main()
