# -*- coding: utf-8 -*-
"""002377 生产排名键位次实测 (用户: "我记得是在TOP100里做选择的, 002377应该上榜")
生产漏斗: raw preds 全池 → prob闸 → 每板块TOP-10 (键=pred_mag_10d × pred_prob_10d)
→ 滞留行 → amt_agree删 → 交付CSV (≤20+滞留) → THS推送=rank≤10
对照三种键: score(模型分) / mag10(纯幅度键) / blend(生产键)
"""
import glob
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
LST = r"D:/AMINQT/Daily Operation/STOCK LIST"
SYM = "002377"
hdr = f"{'date':<10}{'板':<6}{'池':>6}{'score排':>8}{'mag10排':>8}{'blend排':>8}{'mag10':>9}{'prob10':>8}{'blend':>9}"
print(hdr)
for dstr in ["20260903", "20260904", "20260905", "20260906", "20260907", "20260908"]:
    fs = sorted(glob.glob(f"{LST}/parallel_preds_raw_{dstr}__M*.csv"))
    if not fs:
        continue
    d = pd.read_csv(fs[-1], dtype={"symbol": str})
    d["blend"] = d["pred_mag_10d"] * d["pred_prob_10d"]
    sub = d[d["symbol"] == SYM]
    for _, r in sub.iterrows():
        b = d[d["board"] == r["board"]]
        rs = int((b["score"] > r["score"]).sum() + 1)
        rm = int((b["pred_mag_10d"] > r["pred_mag_10d"]).sum() + 1)
        rb = int((b["blend"] > r["blend"]).sum() + 1)
        print(f"{dstr:<10}{r['board']:<6}{len(b):>6}{rs:>8}{rm:>8}{rb:>8}"
              f"{r['pred_mag_10d']:>9.3%}{r['pred_prob_10d']:>8.3f}{r['blend']:>9.3%}")
# 对照: 002790 同表 (已被确认三次前三)
print("\n002790 对照:")
print(hdr)
for dstr in ["20260903", "20260904", "20260905", "20260906", "20260907", "20260908"]:
    fs = sorted(glob.glob(f"{LST}/parallel_preds_raw_{dstr}__M*.csv"))
    if not fs:
        continue
    d = pd.read_csv(fs[-1], dtype={"symbol": str})
    d["blend"] = d["pred_mag_10d"] * d["pred_prob_10d"]
    sub = d[d["symbol"] == "002790"]
    for _, r in sub.iterrows():
        b = d[d["board"] == r["board"]]
        rs = int((b["score"] > r["score"]).sum() + 1)
        rm = int((b["pred_mag_10d"] > r["pred_mag_10d"]).sum() + 1)
        rb = int((b["blend"] > r["blend"]).sum() + 1)
        print(f"{dstr:<10}{r['board']:<6}{len(b):>6}{rs:>8}{rm:>8}{rb:>8}"
              f"{r['pred_mag_10d']:>9.3%}{r['pred_prob_10d']:>8.3f}{r['blend']:>9.3%}")
