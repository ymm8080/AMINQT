# -*- coding: utf-8 -*-
"""_sniper_acceptance.py — 狙击系统特征验收 (2026-08-04).

并行系统架构第 1 轨: 狙击系统 每日输出 3-5 只, T+1 买.
持有视界: **T+3 主 (2026-08-04 用户: "我倾向他+3天的"), T+2 附**.
本脚本对全部面板特征做 **双头 TOP-N 绝对验收** (3d 主 + 2d 附):
  - 每日期截面按特征值降序取 TOP_N (主 TOP-5, 附 TOP-3);
  - 双头: 平均净收益(幅度)>0 且 胜率>=55%, 缺一不过 (2026-08-04 用户双头更正);
  - 对照无条件基准 (全截面 胜率/幅度), 报告 Δ;
  - 与融合轨 (5d/10d TOP-10) 结果并排, 支撑双系统特征分配.

行集 = 快速路径 (复用 main/dual 3y 检查点 + _finalize_slice 补 10d 标签, 不重建).
输出 (WORM): data/_sniper_acceptance_<ts>.json + .log
"""
import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd

from scripts._reclassify_all_features import (MAIN_CHECKPOINT, DUAL_CHECKPOINT,
                                              _finalize_slice, feature_cols)
from scripts._classify_freq_analog import family_of

# 狙击口径 (2026-08-04 用户: 每日 3-5 只, T+1买; 持有 T+2/T+3/T+5 任一视界
# 累计涨幅高+确定性高即保留; 双头 幅度>0 且 胜率>=55%)
HORIZONS = [("2d", "label_pm_2d_net"), ("3d", "label_pm_3d_net"),
            ("5d", "label_pm_5d_net")]
# 裁决/展示优先级: 3d > 2d > 5d (2026-08-04 用户明示); 任一过即保留
PRIORITY = ["3d", "2d", "5d"]
MIN_WINRATE = 0.55
MIN_MAG = 0.0
TOP_N = 5          # 主验收: 每日输出上限 5 只
TOP_N_ALT = 3      # 附: 每日输出 3 只 (更严)


def measure_topn_h(work: pd.DataFrame, col: str, top_n: int) -> dict:
    """双头 TOP-N (高值端, 每日期截面), 按 HORIZONS 逐视界."""
    if col not in work.columns:
        return {"missing": True}
    base = work[["symbol", "date", col] +
                [lab for _, lab in HORIZONS if lab in work.columns]]
    base = base.dropna(subset=[col])
    if len(base) < top_n:
        return {"insufficient": len(base)}
    top = base.sort_values([col], ascending=False).groupby(
        "date", group_keys=False).head(top_n)
    out = {}
    for name, lab in HORIZONS:
        if lab not in top.columns:
            out[name] = {"mag": None, "winrate": None, "n": 0, "ok": False}
            continue
        v = top[lab].dropna()
        if len(v) < 5:
            out[name] = {"mag": None, "winrate": None, "n": int(len(v)),
                         "ok": False, "reason": "few"}
            continue
        mag = float(v.mean())
        winrate = float((v > 0).mean())
        out[name] = {
            "mag": mag, "winrate": winrate, "n": int(len(v)),
            "ok": (mag >= MIN_MAG) and (winrate >= MIN_WINRATE),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="只测前 N 列 (调试用, 默认全量)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    def prog(msg):
        print(msg, flush=True)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # ── 1. 快速路径行集 ──
    slices = []
    for ckpt in (MAIN_CHECKPOINT, DUAL_CHECKPOINT):
        df = _finalize_slice(pd.read_parquet(ckpt))
        slices.append(df)
        del df
        gc.collect()
    work = pd.concat(slices, ignore_index=True).sort_values(
        ["symbol", "date"], ignore_index=True)
    del slices
    gc.collect()
    cols = feature_cols(work)
    if args.limit:
        cols = cols[: args.limit]
    prog(f"行集 rows={len(work):,} stocks={work['symbol'].nunique():,} "
         f"特征 {len(cols)} | latest={work['date'].max():%Y-%m-%d}")

    # ── 2. 无条件基准 (逐视界 3d/2d) ──
    baselines = {}
    for name, lab in HORIZONS:
        v = work[lab].dropna()
        g = work.dropna(subset=[lab]).groupby("date")[lab]
        baselines[name] = {
            "overall": {"mag": float(v.mean()), "winrate": float((v > 0).mean()),
                        "n": int(len(v))},
            "per_date": {"mag": float(g.mean().mean()), "winrate": float((g.mean() > 0).mean())},
        }
        b = baselines[name]["overall"]
        prog(f"\nT+{name} 无条件基准: 全截面 幅度={b['mag']:+.4f} 胜率={b['winrate']:.1%} "
             f"n={b['n']:,} | 日期等权 幅度={baselines[name]['per_date']['mag']:+.4f} "
             f"胜率={baselines[name]['per_date']['winrate']:.1%}")
    base_wr = {k: v["overall"]["winrate"] for k, v in baselines.items()}

    # ── 3. 逐特征 双头 TOP-N 验收 (3d 主 / 2d 附) ──
    header = ("=" * 100 +
              "\n  狙击验收: 双头 (幅度>0 且 胜率>=55%) | 高值端每日期截面 | "
              f"视界 T+2/3/5 任一过即保留 (裁决优先级 3d>2d>5d) | "
              f"TOP-{TOP_N} 主 / TOP-{TOP_N_ALT} 附" +
              "\n=" * 100)
    prog(header)
    summary = []
    n_pass5 = n_pass3 = 0
    for i, col in enumerate(cols):
        r5 = measure_topn_h(work, col, TOP_N)
        r3 = measure_topn_h(work, col, TOP_N_ALT)
        ok5_h = {k: bool(v.get("ok")) for k, v in r5.items()}
        ok3_h = {k: bool(v.get("ok")) for k, v in r3.items()}
        ok5 = any(ok5_h.values())
        ok3 = any(ok3_h.values())
        # 裁决视界: 任一过即保留; 展示优先级 3d > 5d > 2d
        def _pick(ok_h):
            for h in PRIORITY:
                if ok_h.get(h):
                    return h
            return None
        h5 = _pick(ok5_h) if ok5 else None
        h3 = _pick(ok3_h) if ok3 else None
        verdict = h5 if ok5 else h3
        passed5 = [h for h in PRIORITY if ok5_h.get(h)]
        passed3 = [h for h in PRIORITY if ok3_h.get(h)]
        if ok5:
            n_pass5 += 1
        if ok3:
            n_pass3 += 1
        mark = "✓" if (ok5 or ok3) else "✗"
        def _fmt(d):
            if not d:
                return "  -- "
            return (f"{d.get('mag', float('nan')):+.2%}/"
                    f"{d.get('winrate', float('nan')):>5.1%} "
                    f"(Δwr{((d.get('winrate') or 0) - base_wr[verdict or '3d']):+.1%})")
        prog(f"[{i+1}/{len(cols)}] {mark} {col:<26} "
             f"TOP-{TOP_N} 过 {passed5 or ['-']} | TOP-{TOP_N_ALT} 过 {passed3 or ['-']} "
             f"| 裁决 T+{verdict or '-'} | {family_of(col)}")
        summary.append({
            "col": col, "family": family_of(col),
            "ok_top5": ok5, "ok_top3": ok3,
            "passed_horizons_top5": passed5, "passed_horizons_top3": passed3,
            "verdict_horizon": verdict,
            "top5": r5, "top3": r3,
        })

    prog("-" * 100)
    prog(f"TOP-{TOP_N} 通过 {n_pass5} / TOP-{TOP_N_ALT} 通过 {n_pass3} / 合计 {len(cols)}")

    # ── 4. WORM 落盘 ──
    out = {
        "ts": ts,
        "system": "sniper",
        "rules": "T+1 买, 持有 T+2/T+3/T+5 任一, 每日输出 3-5 只",
        "note": "任一视界累计涨幅高+确定性高即保留 (2026-08-04 用户)",
        "window": {"end": str(work["date"].max()), "years": 3},
        "rows": len(work), "stocks": int(work["symbol"].nunique()),
        "n_features": len(cols),
        "meta": {"top_n": TOP_N, "top_n_alt": TOP_N_ALT, "per": "date",
                 "end": "high_only", "labels": [lab for _, lab in HORIZONS],
                 "accept": "任一视界 T+2/3/5 双头过即保留",
                 "verdict_priority": PRIORITY,
                 "min_winrate": MIN_WINRATE,
                 "min_mag": MIN_MAG, "dual_head": True},
        "baseline": baselines,
        "n_pass_top5": n_pass5, "n_pass_top3": n_pass3,
        "features": summary,
    }
    p = os.path.join("data", f"_sniper_acceptance_{ts}.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    text = "\n".join([header] +
                     [f"[{'PASS5' if s['ok_top5'] else ('PASS3' if s['ok_top3'] else '---')}] "
                      f"{s['col']:<26} T+{s.get('verdict_horizon') or '-'} {s['family']}"
                      for s in summary])
    log = os.path.join("data", f"_sniper_acceptance_{ts}.log")
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    prog(f"\n落盘: {p}\n落盘: {log}")


if __name__ == "__main__":
    main()
