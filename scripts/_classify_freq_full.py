# -*- coding: utf-8 -*-
"""_classify_freq_full.py — 全特征集 6格判定 + 事件组事件池归属 (用户 2026-08-04).

目标: 对 feature_engine 全部候选特征, 按功能分组, 判定每组特征在
日/周/月 × TS(个股时序)/XS(日截面) 六格中的归属 → 三频模型特征分配表.

非事件组: 每列构建 1/5/20 日 pct_change 窗口代理日/周/月, 算 per-stock TSIC
          (向量化) + 日截面 rank IC, 6格取 |IC| 最强格为判定.
事件组 (LHB/HOLDER/BT): 走事件池 — 跨事件对齐 [-20,+20], 事件特征对事件后
          T+2/3/5/10/20 收益的跨事件 rank IC (事件定义须用未被 ffill 的稀疏列).

内存友好: 不把窗口列 join 进面板; 每特征即时构建 3 窗口列, 算完即弃;
labels 排名只预排一次. 逐特征打印进度 (flush).

输出: 落盘 data/_classify_freq_full_<ts>.log (WORM). 全文数据×3年.
用法: python scripts/_classify_freq_full.py
"""
import gc
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH
from app.pipeline1.label_engine import LabelEngine
from scripts._diag_column_feed import (
    LABELS,
    MASK_RECENT_DAYS,
    WEIGHTS,
)

logging.disable(logging.CRITICAL)
np.seterr(all="ignore")

WINDOWS = {"日(1)": 1, "周(5)": 5, "月(20)": 20}
MIN_OBS = 60  # 个股时序: 每股至少 60 观测
MIN_CROSS = 10  # 日截面: 每日至少 10 只

# ── 功能组: 非事件 (6格判定) ──
GROUPS_TSXS = {
    "chip 筹码": [
        "pct_90_con", "pct_90_high", "weight_avg", "chip_entropy", "chip_gini",
        "chip_skew_dist", "conc_trend_20d", "conc_90_industry_rank", "peak_price",
        "peak_roc_5d", "peak_roc_20d", "cost_bias", "resistance_dist", "support_dist",
    ],
    "cost 成本线": ["cost_50pct", "cost_95pct"],
    "price 价格": ["close_hfq"],
    "vol 量": [
        "volume", "amount", "turnover_rate", "free_float_turnover_rate",
        "volume_ratio", "ma_vol_ratio_5_20", "vol_surge", "amt_surge",
    ],
    "ma 均线乖离": [
        "bias_5", "bias_10", "bias_20", "bias_60", "bias_120", "bias_250",
        "bias_5_20_cross", "bias_20_60_cross",
    ],
    "volatility 波动": ["amplitude_5d", "intraday_range", "winner_ratio", "pctChg"],
    "valuation 估值市值": ["pe_ttm", "pb", "ps_ttm", "dv_ratio", "total_mv", "circ_mv"],
    "margin 两融": ["margin_balance", "short_balance", "margin_buy_amt", "short_sell_vol"],
    "fundamental 基本面": [
        "roe", "roe_deducted", "roa", "gross_margin", "rev_yoy", "debt_ratio",
        "current_ratio", "asset_turnover", "ar_turnover", "inventory_turnover",
        "ocf_to_or", "net_margin", "eps_yoy", "profit_yoy", "ocfps", "revenue_ps",
        "bps", "eps", "dt_eps", "roe_yoy", "q_roe", "q_ocf_to_sales",
    ],
}

# ── 事件组 (事件池归属; 稀疏列, 未被 ffill 污染) ──
GROUPS_EVT = {
    "LHB 龙虎榜": {
        "cols": [
            "lhb_net_buy", "lhb_buy_amt", "lhb_sell_amt", "lhb_inst_buy",
            "lhb_inst_sell", "lhb_top_buy", "lhb_top_sell", "lhb_quant_buy",
            "lhb_quant_sell", "lhb_retail_buy", "lhb_retail_sell",
        ],
        "mask": "lhb_net_buy",
    },
    "HOLDER 增减持": {
        # 事件定义只用稀疏 ratio 列 (未被 dim29 ffill 污染);
        # sh_change_amt_total 等被 ffill 51.9% 污染, 不得作事件定义, 只在其真实事件池内评 IC.
        "cols": [
            "sh_net_ratio", "sh_g_ratio", "sh_p_ratio", "sh_c_ratio",
            "sh_change_vol", "sh_change_amt_total", "sh_net_change_sign", "sh_net_sign",
        ],
        "mask": "sh_net_ratio",
        "evt_only": True,
    },
    "BT 大宗": {
        "cols": ["bt_count", "bt_disc_raw", "bt_inst_absorb", "bt_amt_ratio_float_mv"],
        "mask": "bt_count",
    },
}


def _load() -> pd.DataFrame:
    read_cols = list(
        dict.fromkeys(
            ["date", "symbol", "is_suspended", "close_hfq"]
            + [c for cols in GROUPS_TSXS.values() for c in cols]
            + [c for g in GROUPS_EVT.values() for c in g["cols"]]
        )
    )
    df = pd.read_parquet(PANEL_V3_PATH, columns=read_cols)
    df = LabelEngine.build_labels(df, session="PM")
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def _f(v):
    return f"{v:+.4f}" if v == v else "   nan"


def _wtsic(per):
    """按 LABEL_WEIGHTS (2/3/5d) 加权 IC; 仅用非 NaN 视界, 权重按占比归一."""
    acc, wsum = 0.0, 0.0
    for k, w in WEIGHTS.items():
        v = per[f"label_pm_{k}d_net"]
        if v == v:
            acc += w * v
            wsum += w
    return acc / wsum if wsum > 0 else np.nan


def group_spearman(rf, rl, g, min_obs):
    """按组平均 Spearman. rf/rl 为已排名的 Series, g 为组键 Series (对齐 rf.index)."""
    m = rf.notna() & rl.notna()
    if m.sum() < 10:
        return np.nan
    gm = g[m]
    fc = rf.where(m) - rf.where(m).groupby(gm).transform("mean")
    lc = rl.where(m) - rl.where(m).groupby(gm).transform("mean")
    num = (fc * lc).groupby(gm).sum()
    f2 = (fc * fc).groupby(gm).sum()
    l2 = (lc * lc).groupby(gm).sum()
    n = m.groupby(g).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = num / np.sqrt(f2 * l2)
    corr = corr[n.reindex(corr.index).fillna(0.0) >= min_obs]
    v = corr.mean()
    return float(v) if v == v else np.nan


def report_tsxs(work, out, prog):
    out.append("=" * 82)
    out.append("  PART A. 非事件组 6格判定 (TS=个股时序IC / XS=日截面rankIC; 日/周/月=1/5/20变化窗口)")
    out.append("=" * 82)
    g_sym = work["symbol"]
    g_date = work["date"]

    # 预排 labels: 每股 (TS) / 每日 (XS)
    lab_sym = {lab: work.groupby("symbol")[lab].rank() for lab in LABELS}
    lab_date = {lab: work.groupby("date")[lab].rank() for lab in LABELS}

    summary = []
    for gname, cols in GROUPS_TSXS.items():
        out.append(f"\n  [{gname}]")
        out.append(
            f"{'feature':<26}{'TS日':>8}{'TS周':>8}{'TS月':>8}"
            f"{'XS日':>8}{'XS周':>8}{'XS月':>8}  ← 判定"
        )
        out.append("-" * 82)
        g_grp = work.groupby("symbol")
        for c in cols:
            if c not in work.columns:
                out.append(f"{c:<26}   (列缺失)")
                continue
            base = work[c]
            wins = {}
            for w in WINDOWS.values():
                wins[f"{c}_p{w}"] = (base / g_grp[c].shift(w) - 1.0).astype("float64")
            tsic = {}  # {f: {lab: val}}
            xic = {}
            for f, wc in wins.items():
                wr_sym = wc.groupby(g_sym.values).rank()
                wr_date = wc.groupby(g_date.values).rank()
                tsic[f] = {lab: group_spearman(wr_sym, lab_sym[lab], g_sym, MIN_OBS)
                           for lab in LABELS}
                xic[f] = {lab: group_spearman(wr_date, lab_date[lab], g_date, MIN_CROSS)
                          for lab in LABELS}
            ts = {w: _wtsic(tsic[f"{c}_p{w}"]) for w in (1, 5, 20)}
            xs = {w: _wtsic(xic[f"{c}_p{w}"]) for w in (1, 5, 20)}
            cells = {
                "TS日": ts[1], "TS周": ts[5], "TS月": ts[20],
                "XS日": xs[1], "XS周": xs[5], "XS月": xs[20],
            }
            best = max(cells, key=lambda k: abs(cells[k]))
            out.append(
                f"{c:<26}{_f(cells['TS日']):>8}{_f(cells['TS周']):>8}{_f(cells['TS月']):>8}"
                f"{_f(cells['XS日']):>8}{_f(cells['XS周']):>8}{_f(cells['XS月']):>8}"
                f"  ← {best} ({abs(cells[best]):.4f})"
            )
            summary.append(
                {"group": gname, "feat": c, "freq": best[2:], "type": best[:2],
                 "ic": abs(cells[best])}
            )
            del wins, tsic, xic
            prog(f"    done {c} ({gname})")
        out.append("")
        del g_grp
        gc.collect()
    return summary


def build_window(df, ev_mask, W=20):
    df2 = df[["symbol", "date", "close_hfq"]].copy()
    df2["ridx"] = df2.groupby("symbol").cumcount()
    events = df.loc[ev_mask, ["symbol", "date"]].copy()
    events["base_ridx"] = df2.loc[df.index[ev_mask], "ridx"].values
    off = pd.DataFrame({"off": np.arange(-W, W + 1)})
    ev_long = events.merge(off, how="cross")
    ev_long["trg"] = ev_long["base_ridx"] + ev_long["off"]
    win = ev_long.merge(
        df2[["symbol", "ridx", "close_hfq"]],
        left_on=["symbol", "trg"],
        right_on=["symbol", "ridx"],
        how="left",
        suffixes=("", "_x"),
    )
    base = win.loc[win["off"] == 0, ["symbol", "date", "close_hfq"]].rename(
        columns={"close_hfq": "base"}
    )
    win = win.merge(base[["symbol", "date", "base"]], on=["symbol", "date"], how="left")
    win["rel"] = win["close_hfq"] / win["base"] - 1.0
    return win.pivot_table(index=["symbol", "date"], columns="off", values="rel")


def report_evt(work, out, prog):
    out.append("\n" + "=" * 82)
    out.append("  PART B. 事件组 — 事件池归属 (跨事件 rank IC)")
    out.append("=" * 82)
    for gname, spec in GROUPS_EVT.items():
        mc = spec["mask"]
        if mc not in work.columns:
            out.append(f"\n  [{gname}] 列缺失, 跳过")
            continue
        if spec["mask"] == "lhb_net_buy":
            ev_mask = work["lhb_net_buy"].fillna(0) != 0
        elif spec.get("evt_only"):
            # 事件定义只用稀疏 ratio 列 (未被 ffill 污染), 其余列仅在其真实事件池内评 IC
            ev_mask = work[["sh_net_ratio", "sh_g_ratio", "sh_p_ratio", "sh_c_ratio"]].notna().any(axis=1)
        else:
            ev_mask = work[spec["cols"]].notna().any(axis=1)
        n_ev = int(ev_mask.sum())
        out.append(f"\n  [{gname}] 事件数 = {n_ev:,} (覆盖率 {n_ev / len(work):.2%})")
        if n_ev < 200:
            out.append("  样本过少, 跳过")
            continue
        piv = build_window(work, ev_mask)
        out.append(f"  {'feature':<24}{'T+2':>9}{'T+3':>9}{'T+5':>9}{'T+10':>9}{'T+20':>9}")
        for c in spec["cols"]:
            if c not in work.columns:
                continue
            xf = work.loc[ev_mask, ["symbol", "date", c]].set_index(["symbol", "date"])[c]
            row = [f"{c:<24}"]
            for h in (2, 3, 5, 10, 20):
                if h not in piv.columns:
                    row.append(f"{'   nan':>9}")
                    continue
                y = piv[h]
                xx = xf.reindex(y.index)
                m = xx.notna().to_numpy() & y.notna().to_numpy()
                if m.sum() < 30:
                    row.append(f"{'   nan':>9}")
                    continue
                r = xx[m].rank()
                ry = y[m].rank()
                row.append(f"{_f(r.corr(ry)):>9}")
            out.append("  ".join(row))
            prog(f"    done {c} ({gname})")
        out.append("")
        del piv
        gc.collect()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    def prog(msg):
        print(msg, flush=True)

    out = []
    df = _load()
    latest = df["date"].max()
    cutoff = latest - pd.DateOffset(years=3)
    work = df[df["date"] >= cutoff].reset_index(drop=True)
    out.append(
        f"--- 全市场×3年 | rows={len(work):,} stocks={work['symbol'].nunique()} "
        f"| {latest:%Y-%m-%d} 往前3年 ---"
    )

    summary = report_tsxs(work, out, prog)
    ts_run = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join("data", f"_classify_freq_full_summary_{ts_run}.json"),
              "w", encoding="utf-8") as fh:
        import json as _json
        _json.dump({"ts": ts_run, "rows": len(work), "features": summary},
                   fh, indent=2, ensure_ascii=False)
    report_evt(work, out, prog)

    # ── 汇总: 三频模型特征分配 ──
    out.append("\n" + "=" * 82)
    out.append("  汇总 — 三频模型特征分配 (月频/周频/日频模型分别取对应特征)")
    out.append("=" * 82)
    out.append(f"{'功能组':<26}{'判定频率':>8}{'判定类型':>8}{'最强|IC|':>10}{'样本数':>6}")
    out.append("-" * 60)
    grp_freq = {}
    for s in summary:
        grp_freq.setdefault(s["group"], []).append(s)
    for gname, rows in grp_freq.items():
        freq_acc = {}
        type_acc = {}
        for r in rows:
            freq_acc.setdefault(r["freq"], []).append(r["ic"])
            type_acc.setdefault(r["type"], []).append(r["ic"])
        best_freq = max(freq_acc, key=lambda k: np.mean(freq_acc[k]))
        best_type = max(type_acc, key=lambda k: np.mean(type_acc[k]))
        ic = max(np.mean(v) for v in freq_acc.values())
        out.append(
            f"{gname:<26}{best_freq:>8}{best_type:>8}{ic:>10.4f}{len(rows):>6}"
        )
    out.append("")
    out.append("事件组: 走事件池 (跨事件对齐研究), 独立事件模块, 不进个股TS/日截面常规模.")
    out.append("事件组衰减视界决定其归属模型: LHB/HOLDER 均为短视界事件信号, 归 日/事件 模型 (LHB T+2 rankIC +0.156 → T+20 +0.045 衰减).")
    out.append("注: rankIC 正 ≠ 绝对上涨 — LHB 净买>0 绝对净收益口径 T+2 -0.56%/胜率41.5%, 不达'买进能赚'裁决.")
    out.append("BT bt_disc_raw 一致负 -0.037 → 风控/负向信号, 不进正向 alpha 模型.")

    text = "\n".join(out)
    print(text, flush=True)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    p = os.path.join("data", f"_classify_freq_full_{ts}.log")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\n落盘: {p}", flush=True)

    del df, work
    gc.collect()


if __name__ == "__main__":
    main()
