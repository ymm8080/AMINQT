"""验证诚实验收闸 (胜率≥0.50 且 幅度>1%/1.5%) 在历史 run 上的判定."""

import glob
import json
import os

BASE = "D:/AMINQT/DATA OTHERS/BACKTESTING RESULT"
RUNS = ["20260810_125705", "20260808_072512", "20260806_144240"]
MAG = {"main": 0.01, "dual": 0.015}


def find(p):
    h = glob.glob(os.path.join(BASE, p, "backtest.json")) + glob.glob(
        os.path.join(BASE, p, "**", "backtest.json"), recursive=True
    )
    return h[0] if h else None


for run in RUNS:
    p = find(run)
    if not p:
        print(f"== {run}: NOT FOUND")
        continue
    d = json.load(open(p, encoding="utf-8"))
    print(f"\n===== {run} =====")
    for b, bd in d.get("boards", {}).items():
        for sname, s in bd.get("systems", {}).items():
            if not s.get("enabled"):
                continue
            for lab, oos in s.get("oos", {}).items():
                pr = oos.get("primary") or {}
                passed = []
                for h, r in (pr.get("per_horizon") or {}).items():
                    ok = bool(
                        r.get("n", 0) >= 5
                        and r.get("winrate", 0) >= 0.50
                        and r.get("mag", 0) > MAG[b]
                    )
                    if ok:
                        passed.append(f"{h}(wr{r['winrate']:.1%}/m{r['mag']:+.1%})")
                tag = passed if passed else "-"
                print(f"  {b}/{sname}/{lab}: {tag}")
