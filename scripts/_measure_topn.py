# -*- coding: utf-8 -*-
"""_measure_topn.py — 用户裁决口径: TOP-N 绝对上涨幅度 + 上涨概率 (2026-08-04).

rank IC 是相对排序, 会把"跌得少"当正信号 → 误导 (LHB 教训). 用户明确:
裁决指标 = 按特征选出 TOP-N 股票后, 买入的「绝对上涨幅度 + 上涨概率」要高.
本工具对任意特征列, 在给定工作 df (含 label_pm_{k}d_net 净收益标签) 上:
  - 每股/每日 按特征值 rank, 取 top_n 只 (默认 10, 支持 pct 前十分位);
  - 测这些股票在 T+2/3/5/10/20 的 平均绝对收益(幅度) + 上涨胜率(概率);
  - 输出逐特征逐视界表 + 裁决 (幅度正 & 胜率>=threshold 才算通过).
用法: 被 _reclassify_all_features.py 导入; 也可独立跑单列诊断.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import pandas as pd

# 用户裁决阈值 (可调; 用户口径 "买进绝对能赚钱" → 胜率要高, 平均收益为正且显著)
MIN_WINRATE = 0.55   # 上涨胜率阈值
MIN_MAG = 0.0        # 平均绝对收益 > 0
# 验收视界 (2026-08-04 用户定稿, 我拍板): 持有至多 5 天可复利 → 5d/10d 累计.
# 5d=基础持有, 10d=趋势持有复利段 ("一直小幅涨就一直拿"). 10d 净标签由
# _reclassify_all_features.add_label_pm_10d_net 按生产公式补算.
HORIZONS = (5, 10)

# 视界用 label 列: label_pm_{k}d_net (净收益, 扣成本+滑点)
LABEL_COLS = {k: f"label_pm_{k}d_net" for k in HORIZONS}


def measure_topn(
    work: pd.DataFrame,
    col: str,
    top_n: int = 10,
    per: str = "date",   # "date"=每日截面取top10; "stock"=每股时序取top段
    ascending: bool = False,   # False=取特征值最高topN; True=取最低topN (双向裁决)
    winrate_threshold: float = MIN_WINRATE,
    mag_threshold: float = MIN_MAG,
) -> dict:
    """对单个特征测 TOP-N 绝对上涨幅度 + 概率.

    per="date": 每个交易日横截面按特征 rank 取 top_n → 汇总这些 (symbol,date) 的
    T+k 净收益幅度与胜率 (用户 TOP10 选股口径).
    返回: {horizon: {mag, winrate, n, ok}} 及聚合裁决.
    """
    res = {}
    if col not in work.columns:
        return {"missing": True}
    base = work[["symbol", "date", col] + [LABEL_COLS[k] for k in HORIZONS if LABEL_COLS[k] in work.columns]]
    base = base.dropna(subset=[col])
    if len(base) < top_n:
        return {"insufficient": len(base)}

    if per == "date":
        # 每日横截面取 top_n (ascending=False=取高值端; True=取低值端, 负向因子用)
        top = base.sort_values([col], ascending=ascending).groupby("date", group_keys=False).head(top_n)
    else:
        # 每股时序取后 top_n 行 (TS 口径: 个股自身信号最强的时段)
        top = base.sort_values(["symbol", col], ascending=ascending).groupby("symbol", group_keys=False).head(top_n)

    for k in HORIZONS:
        lab = LABEL_COLS[k]
        if lab not in top.columns:
            res[k] = {"mag": None, "winrate": None, "n": 0, "ok": False, "reason": "label_missing"}
            continue
        v = top[lab].dropna()
        if len(v) < 5:
            res[k] = {"mag": None, "winrate": None, "n": int(len(v)), "ok": False, "reason": "few"}
            continue
        mag = float(v.mean())
        winrate = float((v > 0).mean())
        ok = (mag >= mag_threshold) and (winrate >= winrate_threshold)
        res[k] = {"mag": mag, "winrate": winrate, "n": int(len(v)), "ok": ok}

    # 聚合裁决: 至少一个视界达标, 且 T+2 短视界优先 (可操作持有)
    passed = [k for k in HORIZONS if res[k].get("ok")]
    best_h = None
    if passed:
        # 优先短视界 (T+2 最可操作); 同达标取幅度*胜率综合最高
        best_h = min(passed)
    res["_verdict"] = {
        "col": col,
        "top_n": top_n,
        "per": per,
        "best_horizon": best_h,
        "passed_horizons": passed,
        "notes": "胜率>={winrate:.0%} 且 平均净收益>0".format(winrate=winrate_threshold),
    }
    return res


def measure_topdecile(work: pd.DataFrame, col: str, per: str = "date") -> dict:
    """按特征值取前 10% (每日期截面), 测绝对幅度+胜率 — 样本更稳."""
    if col not in work.columns:
        return {"missing": True}
    base = work[["symbol", "date", col] + [LABEL_COLS[k] for k in HORIZONS if LABEL_COLS[k] in work.columns]].dropna(subset=[col])
    if len(base) < 20:
        return {"insufficient": len(base)}
    if per == "date":
        top = base.sort_values([col], ascending=False).groupby("date", group_keys=False).apply(
            lambda g: g.head(max(1, int(len(g) * 0.10)))
        )
    else:
        top = base.sort_values(["symbol", col], ascending=False).groupby("symbol", group_keys=False).apply(
            lambda g: g.head(max(1, int(len(g) * 0.10)))
        )
    out = {}
    for k in HORIZONS:
        lab = LABEL_COLS[k]
        v = top[lab].dropna() if lab in top.columns else pd.Series(dtype=float)
        if len(v) < 20:
            out[k] = {"mag": None, "winrate": None, "n": int(len(v))}
            continue
        out[k] = {"mag": float(v.mean()), "winrate": float((v > 0).mean()), "n": int(len(v))}
    return out


def _f(v, w=7):
    return f"{v:+.4f}" if isinstance(v, float) else ("   nan" if v is None else f"{v}")


def _fmt_mag(v):
    return f"{v:+.2%}" if isinstance(v, float) else "   nan"


def report_topn(work: pd.DataFrame, cols, top_n=10, per="date", out=None, prog=None) -> list:
    """对多列批量测 TOP-N, 输出 幅度/胜率 表, 返回裁决清单."""
    lines = out if out is not None else []
    lines.append("=" * 92)
    lines.append(f"  TOP{top_n} 绝对上涨裁决 ({per}截面 | 净收益标签 label_pm_*d_net | "
                 f"胜率>={MIN_WINRATE:.0%} 且 平均>0 为通过)")
    lines.append("=" * 92)
    hdr = f"{'feature':<26}" + "".join(f"{'T+%d'%k:>16}" for k in HORIZONS) + "  裁决"
    lines.append(hdr)
    lines.append("-" * 92)
    summary = []
    for c in cols:
        r = measure_topn(work, c, top_n=top_n, per=per)
        if r.get("missing"):
            lines.append(f"{c:<26}  (列缺失)")
            continue
        if r.get("insufficient"):
            lines.append(f"{c:<26}  (样本不足 {r['insufficient']})")
            continue
        row = f"{c:<26}"
        for k in HORIZONS:
            v = r.get(k, {})
            row += f"{_fmt_mag(v.get('mag'))}/{_fmt_winrate(v.get('winrate')):>7}  "
        verdict = r["_verdict"]
        mark = "✓" if verdict["best_horizon"] else "✗"
        row += f"  {mark}{verdict['best_horizon'] or ''}"
        lines.append(row)
        summary.append({"col": c, "verdict": verdict, "detail": {k: r[k] for k in HORIZONS}})
        if prog:
            prog(f"    done {c}")
    return summary


def _fmt_winrate(v):
    return f"{v:>6.1%}" if isinstance(v, float) else "  nan"


if __name__ == "__main__":
    # 单列快速演示/自检
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("导入成功. measure_topn / measure_topdecile / report_topn 可用.")
