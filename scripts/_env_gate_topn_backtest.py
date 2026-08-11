"""环境闸对 TOP-N 预测质量的 A/B 回测 (2026-08-10).

回答用户: "加环境闸对 TOPN 预测质量起正面作用了没?" — 用并行行集实得净标签说话.

方法 (同一打分规则, 唯一差别 = 哪些交易日选股):
  - 基线:  每个 OOS 交易日都选 TOP-N (现状: market_state 恒 range)
  - 闸A:   跳过极端冰点否决日 (is_veto, 空清单日)
  - 闸B:   跳过全部冰点日 (ice → 最强 bear 收紧形态, 不开仓)
  对比 TOP-N 实得 c2c 净收益 (label_pm_{h}d_net) 的 命中率/幅度.

环境状态 PIT (无前视): 每日涨停/跌停家数按 board 分档近似, 常态基线 = 前 60 交易日
(shift(1) 排除当日), 阈值与 market_environment.py 完全一致.

验收: 只看 OOS (末 250 交易日, 另报 126d 6m). 随机种子无关 (确定性打分).
"""

import gc
import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from app.pipeline1.market_environment import (
    _limit_mask,
    classify_market_state,
    is_veto,
)
from app.pipeline_parallel.backtest import (
    BOARD_THRESHOLDS,
    load_panel,
    run_system,
)
from app.pipeline_parallel.config import FUSION, SNIPER

OOS_DAYS_LIST = [250, 126]


def pit_env_states(work: pd.DataFrame) -> dict:
    """逐日 PIT 环境状态: {date: {'state':..., 'veto': bool}}. 无前视."""
    up, dn = _limit_mask(work)
    cnt = (
        work[["date"]]
        .assign(_u=up.astype(int), _d=dn.astype(int))
        .groupby("date")
        .agg(u=("_u", "sum"), d=("_d", "sum"))
        .sort_index()
    )
    # 常态基线 = 前 60 交易日均值 (shift(1) 排除当日, 防前视)
    cnt["base"] = cnt["u"].shift(1).rolling(60, min_periods=20).mean()
    states = {}
    for d, r in cnt.iterrows():
        sent = {"count_limit_up": int(r["u"]), "count_limit_down": int(r["d"])}
        if pd.notna(r["base"]):
            hist = pd.DataFrame({"count_limit_up": np.full(60, float(r["base"]))})
            st = classify_market_state(sent, hist)
        else:
            st = classify_market_state(sent)
        states[d] = {"state": st, "veto": is_veto(st, sent)}
    return states


def _mask_by_date(sub, dates, oos_days, states, mode):
    oos = sub["date"].values >= dates[-oos_days]
    st = np.array([states[d]["state"] for d in sub["date"]])
    vt = np.array([states[d]["veto"] for d in sub["date"]])
    if mode == "baseline":
        return oos
    if mode == "gate_veto":
        return oos & ~vt
    if mode == "gate_ice":
        return oos & (st != "ice")
    raise ValueError(mode)


def day_discrimination(work, states):
    """验证分类器前提: 冰点/高潮日 全截面实得前向收益是否系统性差/好."""
    st = np.array([states[d]["state"] for d in work["date"]])
    cols = [f"label_pm_{h}d_net" for h in (5, 10)]
    cols = [c for c in cols if c in work.columns]
    rows = {"date": work["date"].values, "state": st}
    for c in cols:
        rows[c] = work[c].values
    g = pd.DataFrame(rows).dropna(subset=cols)
    out = []
    for s in ("ice", "range", "hot"):
        sub = g[g["state"] == s]
        if len(sub) == 0:
            continue
        r = {"state": s, "days": int(sub["date"].nunique()), "rows": len(sub)}
        for c in cols:
            r[f"{c}_mean"] = round(float(sub[c].mean()) * 100, 3)
            r[f"{c}_winrate"] = round(float((sub[c] > 0).mean()) * 100, 1)
        out.append(r)
    return out


def main():
    print("加载行集 (3y 检查点 + c2c/MFE 净标签)...", flush=True)
    work = load_panel()
    print(
        f"行集 rows={len(work):,} stocks={work['symbol'].nunique():,} "
        f"days={work['date'].nunique()} latest={work['date'].max():%Y-%m-%d}",
        flush=True,
    )
    states = pit_env_states(work)
    print(
        "\n[0] 环境状态分布 (全窗 PIT): "
        + ", ".join(
            f"{s}={sum(1 for v in states.values() if v['state'] == s)}日"
            for s in ("ice", "range", "hot")
        ),
        flush=True,
    )
    veto_days = [d for d, v in states.items() if v["veto"]]
    print(f"    极端冰点否决日: {len(veto_days)} 日", flush=True)

    print("\n[1] 分类器前提 — 全截面实得前向收益 by 环境状态 (含所有股票, 非选股):")
    disc = day_discrimination(work, states)
    dcols = [
        f"label_pm_{h}d_net" for h in (5, 10) if f"label_pm_{h}d_net" in work.columns
    ]
    hdr = f"{'状态':<6s} {'天数':>5s} {'行数':>8s} " + " ".join(
        f"{c}_均%({c}_涨率%)".ljust(20) for c in dcols
    )
    print("  " + hdr)
    for r in disc:
        cells = [f"{r['state']:<6s}", f"{r['days']:>5d}", f"{r['rows']:>8d}"]
        for h in (5, 10):
            c = f"label_pm_{h}d_net"
            if c + "_mean" in r:
                cells.append(
                    f"{r[c + '_mean']:+.3f}({r[c + '_winrate']:.1f}%)".ljust(20)
                )
        print("  " + " ".join(cells))

    print("\n[2] TOP-N 质量 A/B (同一打分 run_system, 唯一差别 = 选股日):")
    for b in ("main", "dual"):
        sub = work[work["board"] == b]
        if len(sub) == 0:
            continue
        dates = np.sort(sub["date"].unique())
        crit = (BOARD_THRESHOLDS[b]["min_winrate"], BOARD_THRESHOLDS[b]["min_mag"])
        print(f"\n  === 板块 {b.upper()} (行 {len(sub):,} 日 {len(dates)}) ===")
        for spec in (SNIPER, FUSION):
            for oos_days in OOS_DAYS_LIST:
                if oos_days >= len(dates) - 60:
                    continue
                print(
                    f"\n  -- {spec.name.upper()} TOP-{spec.top_n} | OOS {oos_days}d --"
                )
                results = {}
                for mode in ("baseline", "gate_veto", "gate_ice"):
                    m = _mask_by_date(sub, dates, oos_days, states, mode)
                    r = run_system(sub, spec, spec.top_n, m, crit=crit)
                    ph = {
                        h: {"winrate": x["winrate"], "mag": x["mag"], "n": x["n"]}
                        for h, x in r["per_horizon"].items()
                        if x.get("n")
                    }
                    results[mode] = ph
                # 打印: 行=视界, 列=mode
                for h in spec.horizons:
                    if h not in results["baseline"]:
                        continue
                    cells = []
                    for mode in ("baseline", "gate_veto", "gate_ice"):
                        ph = results[mode]
                        if h not in ph:
                            cells.append(f"{mode:<9s} -")
                            continue
                        v = ph[h]
                        cells.append(
                            f"{mode:<9s} 涨率={v['winrate'] * 100:5.1f}% 均={v['mag'] * 100:+6.2f}% n={v['n']:>5d}"
                        )
                    print(f"    T+{h}: " + " | ".join(cells))
            gc.collect()
    print("\nDONE")


if __name__ == "__main__":
    main()
