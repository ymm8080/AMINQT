"""_diag_chain_crossdata.py — 收集→洗盘→试盘 完整链路 × 两套系统数据交叉检验 (2026-08-19).

用户假设: LEGACY 用 PARALLEL 数据 / PARALLEL 用 LEGACY 数据, 形成完整
"收集筹码→试盘→洗盘"过程链路, 或能预测 300911 类股.

本脚本两部分:
  A. 300911 在 dual 检查点 (LEGACY 特征) + prepare_adx (PARALLEL 独有列) 的链路值,
     看两套系统各自看到了什么.
  B. 879d dual 池三阶段链路统计检验 (与 ambush_breakout 同口径):
       收集: ent_chg20 < -8% (决策日 D 前 20 日筹码熵降, 筹码集中)
       洗盘: D-3..D-1 缩量 (日量/20日均量 < 0.8)
       试盘: D-3..D-1 内出现 vr>1.5 & 0<ret<15% (放量试盘未涨停)
       目标: D+1..D+10 max_ret >= 7% (突破) 与 label_pm_10d_net (c2c T+10 净收益)
     对比 基准 / 收集 alone / 试盘 alone / 试盘+洗盘 / 完整链路, 4 子窗稳定.

用法: python scripts/_diag_chain_crossdata.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from app.pipeline_parallel.indicators import prepare_adx
from config.settings import PANEL_V3_PATH, data_others_path

SYMBOL = "300911"
SLICE = 879
CKPT = os.path.join("data", "_diag_stage_dual_3y.parquet")


# ── Part A: 300911 双系统链路值 ──────────────────────────────────────────
def part_a() -> None:
    need = [
        "symbol",
        "date",
        "close_hfq",
        "high_hfq",
        "low_hfq",
        "volume",
        "turnover_rate",
        "chip_entropy",
        "chip_gini",
        "vol_ratio_ma5",
        "upper_shadow_pct",
        "lower_shadow_pct",
        "吸筹_density_10d",
        "洗盘_density_5d",
        "close_position",
        "label_pm_10d_net",
    ]
    t = pd.read_parquet(CKPT, columns=need)
    t = t[t["symbol"] == SYMBOL].copy()
    t = t.sort_values("date")
    t = prepare_adx(
        t
    )  # vol_ratio / rps_60 / pct_70_con / adx / ma_tightness / sharpe_20 / pv_corr_5
    t["dt"] = pd.to_datetime(t["date"]).dt.strftime("%m-%d")
    t["ent_chg20"] = t["chip_entropy"].pct_change(20)
    t.groupby("symbol")
    t["vol_ma20"] = (
        t.groupby("symbol")["volume"].transform(
            lambda v: v.where(v > 0).rolling(20, min_periods=10).mean()
        )
        if "volume" in t.columns
        else np.nan
    )
    show = t.tail(12)[
        [
            "dt",
            "close_hfq",
            "chip_entropy",
            "ent_chg20",
            "chip_gini",
            "vol_ratio_ma5",
            "upper_shadow_pct",
            "洗盘_density_5d",
            "vol_ratio",
            "rps_60",
            "pct_70_con",
            "ma_tightness",
            "label_pm_10d_net",
        ]
    ]
    print("\n=== A. 300911 链路特征 (dual 检查点 + parallel prepare_adx 列) ===")
    print(show.round(3).to_string(index=False))
    print(
        "   列来源: chip_*/vol_ratio_ma5/shadow/吸筹/洗盘/close_position = LEGACY 特征;"
    )
    print("   vol_ratio/rps_60/pct_70_con/ma_tightness = PARALLEL prepare_adx 独有列")


# ── Part B: 三阶段链路统计 (时序: 收集→试盘S→洗盘S+1..S+3→目标S+1..S+10) ──
def part_b() -> None:
    p = pd.read_parquet(
        str(PANEL_V3_PATH),
        columns=["symbol", "date", "close_hfq", "low_hfq", "volume", "chip_entropy"],
    )
    p = p[p["symbol"].str.startswith(("30", "68"))].copy()
    p = p.sort_values(["symbol", "date"]).reset_index(drop=True)
    p["dt"] = pd.to_datetime(p["date"]).dt.normalize()
    dates = sorted(pd.unique(p["dt"]))
    p = p[p["dt"] >= dates[-SLICE]]

    g = p.groupby("symbol", group_keys=False)
    p["ret_1d"] = g["close_hfq"].pct_change(fill_method=None)
    p["vol_ma20"] = p.groupby("symbol")["volume"].transform(
        lambda v: v.where(v > 0).rolling(20, min_periods=10).mean()
    )
    p["vr"] = p["volume"] / p["vol_ma20"]
    p["ent_chg20"] = g["chip_entropy"].pct_change(20, fill_method=None)

    piv_c = p.pivot_table(
        index="symbol", columns="dt", values="close_hfq", aggfunc="last"
    )
    piv_l = p.pivot_table(
        index="symbol", columns="dt", values="low_hfq", aggfunc="last"
    )
    cp = piv_c.reindex(columns=pd.to_datetime(dates)).ffill(axis=1)
    lp = piv_l.reindex(columns=pd.to_datetime(dates)).ffill(axis=1)
    ret_piv = cp.pct_change(axis=1, fill_method=None)

    def fwd_rolling(df: pd.DataFrame, win: int, fn) -> pd.DataFrame:
        return fn(df.iloc[:, ::-1].rolling(win, min_periods=1, axis=1)).iloc[:, ::-1]

    fut10_max_ret = fwd_rolling(ret_piv, 10, lambda r: r.max())  # S+1..S+10 max
    fut3_max_ret = fwd_rolling(ret_piv, 3, lambda r: r.max())  # S+1..S+3 max
    fut3_min_low = fwd_rolling(lp, 3, lambda r: r.min())  # S+1..S+3 min low

    def join(name: str, df: pd.DataFrame) -> pd.DataFrame:
        out = df.stack().rename(name).reset_index()
        out.columns = ["symbol", "dt", name]
        return p.merge(out, on=["symbol", "dt"], how="left")

    p = join("fut10_max_ret", fut10_max_ret)
    p = join("fut3_max_ret", fut3_max_ret)
    p = join("fut3_min_low", fut3_min_low)

    ok = p["fut10_max_ret"].notna() & p["ent_chg20"].notna()
    probe = ok & (p["vr"] > 1.5) & (p["ret_1d"] > 0) & (p["ret_1d"] < 0.15)  # 试盘日 S
    wash = (
        ok & (p["fut3_max_ret"] < 0.07) & (p["fut3_min_low"] >= p["low_hfq"])
    )  # S 后洗盘
    collect = ok & (p["ent_chg20"] < -0.08)  # S 前 20 日筹码集中

    def stats(sub: pd.DataFrame, label: str) -> None:
        if len(sub) < 30:
            print(f"{label:<30} n={len(sub):>6} (样本过少)")
            return
        print(
            f"{label:<30} n={len(sub):>6} 10天突破≥7%={float((sub['fut10_max_ret'] >= 0.07).mean()):>6.2%}"
        )

    print(
        f"\n=== B. 三阶段链路 (879d dual 池, 截至 {pd.Timestamp(dates[-1]).date()}) ==="
    )
    print(
        "   时序: 收集(ent_chg20<-8% 于 S) → 试盘日 S(vr>1.5 & 0<ret<15%) → 洗盘(S+1..S+3)"
    )
    print("   目标: S+1..S+10 max_ret>=7%")
    stats(p[ok], "全池基准")
    stats(p[collect], "收集 alone")
    stats(p[probe], "试盘日 alone")
    stats(p[probe & wash], "试盘 + 洗盘")
    stats(p[probe & ~wash], "试盘 + 未洗盘 (对照)")
    stats(p[collect & probe & wash], "完整链路 收集+试盘+洗盘")
    stats(p[collect & probe & ~wash], "收集+试盘 未洗盘 (对照)")

    print()
    seg = pd.cut(p["dt"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"])
    for name, m in [
        ("基准", ok),
        ("试盘 alone", probe),
        ("试盘+洗盘", probe & wash),
        ("完整链路", collect & probe & wash),
    ]:
        print(f"{name} 子窗口:")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            stats(p[m & (seg == q)], f"  {q}")

    out = {
        "as_of": pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"),
        "window": f"{SLICE}d dual pool",
        "collect_def": "ent_chg20 < -8% (试盘日 S)",
        "probe_def": "S: vr>1.5 & 0<ret<15%",
        "wash_def": "S+1..S+3 max ret<7% & min low>=S low",
        "target": "S+1..S+10 max_ret >= 7%",
        "base_rate": round(float(p.loc[ok, "fut10_max_ret"].ge(0.07).mean()), 4),
        "probe_rate": round(float(p.loc[probe, "fut10_max_ret"].ge(0.07).mean()), 4),
        "probe_wash_rate": round(
            float(p.loc[probe & wash, "fut10_max_ret"].ge(0.07).mean()), 4
        ),
        "chain_rate": round(
            float(p.loc[collect & probe & wash, "fut10_max_ret"].ge(0.07).mean()), 4
        ),
    }
    out_path = os.path.join(data_others_path("diag"), "chain_crossdata_20260819.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n报告: {out_path}")


def main() -> int:
    part_a()
    part_b()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
