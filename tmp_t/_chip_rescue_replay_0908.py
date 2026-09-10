# -*- coding: utf-8 -*-
"""筹码派发闸(wr5<0)救回条件回放 09-08 (研究, 不改生产).

口径 (照 09-05 tmp_t/_top10_chipdir_replay_0905.py 重建):
- 回放池: LEGACY scored 检查点 e125 (main+dual), 每 (板,日) 按 pred_ret_10d 降序 top10.
- wr5: cyq_panel 按 symbol 前第 5 行 winner_ratio 差 (≤T 最近一行, 与生产
  apply_wr5_gate 同口径, scripts/_prob10_density_shadow.py 只读参考).
- 净收益: T+1 收盘买 → T+4 收盘卖, (sell/buy-1)*100 - 0.2. 赢=净>=5, 大亏=净<=-10.
- 救回因子全部只用 T 日及以前数据 (shift 正向), 无 look-ahead.
- 面板无 moneyflow/大单净流入列 (lhb_=龙虎榜事件, bt_=大宗事件, margin_=两融,
  非逐日大单), 用 turnover_rate T/前20日均值比 + amount 250日分位替代.
"""
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = r"D:\AMINQT\AMINQT CODES"
CYQ_PATH = os.path.join(REPO, "data", "cyq_panel.parquet")
PANEL_PATH = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
SCORED = {
    "main": os.path.join(REPO, "data", "_diag_rankkey_scored_main_e125.parquet"),
    "dual": os.path.join(REPO, "data", "_diag_rankkey_scored_dual_e125.parquet"),
}
OUT_DIR = os.path.join(REPO, "data", "others")

TOPN = 10
COST_PP = 0.2
WIN_TH, BIGLOSS_TH = 5.0, -10.0
PANEL_START = "2024-10-01"  # 250d 回看全覆盖
PCT_MIN = 60  # rolling rank min_periods


def build_picks() -> pd.DataFrame:
    parts = []
    for b, f in SCORED.items():
        d = pd.read_parquet(f, columns=["date", "board", "symbol", "pred_ret_10d"])
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df = df.sort_values(
        ["board", "date", "pred_ret_10d", "symbol"],
        ascending=[True, True, False, True],
    )
    picks = df.groupby(["board", "date"], sort=False).head(TOPN).reset_index(drop=True)
    return picks


def chip_features() -> pd.DataFrame:
    """每 (symbol, date): wr5 = winner_ratio - 前5行; cost_chg5 = avg_cost 5行涨幅."""
    cq = pd.read_parquet(CYQ_PATH, columns=["symbol", "date", "winner_ratio", "avg_cost"])
    cq["symbol"] = cq["symbol"].astype(str).str.zfill(6)
    cq = cq.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"])
    g = cq.groupby("symbol", sort=False)
    cq["wr5"] = g["winner_ratio"].shift(0) - g["winner_ratio"].shift(5)
    cq["cost_chg5"] = g["avg_cost"].pct_change(periods=5)
    return cq[["symbol", "date", "wr5", "cost_chg5"]]


def panel_features() -> pd.DataFrame:
    """每 (symbol, date): 前向收益 + T 日可知因子 (全部 groupby+shift/rolling 向量化)."""
    cols = ["symbol", "date", "close", "turnover_rate", "amount"]
    t = pq.read_table(
        PANEL_PATH, columns=cols, filters=[("date", ">=", pd.Timestamp(PANEL_START))]
    )
    p = t.to_pandas()
    p["symbol"] = p["symbol"].astype(str).str.zfill(6)
    p = p.drop_duplicates(["symbol", "date"], keep="last").sort_values(["symbol", "date"])
    p = p.reset_index(drop=True)
    g = p.groupby("symbol", sort=False)

    c = g["close"]
    p["buy_px"] = c.shift(-1)   # T+1 收盘买
    p["sell_px"] = c.shift(-4)  # T+4 收盘卖
    p["net_pp"] = (p["sell_px"] / p["buy_px"] - 1.0) * 100.0 - COST_PP
    for k in (5, 20, 60, 250):
        p[f"r{k}"] = p["close"] / g["close"].shift(k) - 1.0

    # 价格/成交额 250 日分位 (rolling rank pct, 含 T 当日)
    p["pct250"] = g["close"].transform(
        lambda s: s.rolling(250, min_periods=PCT_MIN).rank(pct=True)
    )
    p["amt_pct250"] = g["amount"].transform(
        lambda s: s.rolling(250, min_periods=PCT_MIN).rank(pct=True)
    )
    # 量能: T 日换手率 / 前 20 日换手率均值 (不含 T)
    p["to20"] = p["turnover_rate"] / (
        g["turnover_rate"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    )
    return p


def mstats(df: pd.DataFrame, denom: float) -> dict:
    """组指标: n, 只/日, 赢率, 均值净pp, 大亏率, 最差, 真赢家/日."""
    n = len(df)
    if n == 0:
        return {"n": 0, "per_day": 0.0, "win": np.nan, "mean": np.nan,
                "bigloss": np.nan, "worst": np.nan, "win_per_day": 0.0}
    w = (df["net_pp"] >= WIN_TH).sum()
    return {
        "n": int(n),
        "per_day": round(n / denom, 3),
        "win": round(100.0 * (df["net_pp"] >= WIN_TH).mean(), 1),
        "mean": round(df["net_pp"].mean(), 2),
        "bigloss": round(100.0 * (df["net_pp"] <= BIGLOSS_TH).mean(), 1),
        "worst": round(df["net_pp"].min(), 1),
        "win_per_day": round(w / denom, 3),
    }


def main():
    picks = build_picks()
    n_days = picks["date"].nunique()
    boards = sorted(picks["board"].unique())
    denom_pool = n_days * len(boards)  # 板日数 (池分母)

    # --- wr5 (merge_asof backward: ≤T 最近 cyq 行) ---
    cq = chip_features()
    pk = picks.rename(columns={"date": "T"}).sort_values("T")
    cq_s = cq.sort_values("date")
    pk = pd.merge_asof(
        pk, cq_s, left_on="T", right_on="date", by="symbol", direction="backward"
    )
    pk["cyq_lag_days"] = (pk["T"] - pk["date"]).dt.days
    pk = pk.drop(columns=["date"])

    # --- 面板因子 + 前向收益 (精确匹配 symbol+T) ---
    pf = panel_features()
    pk = pk.merge(
        pf[["symbol", "date", "net_pp", "r5", "r20", "r60", "r250",
            "pct250", "to20", "amt_pct250", "close"]],
        left_on=["symbol", "T"], right_on=["symbol", "date"], how="left",
    ).drop(columns=["date"])

    pk["marked"] = pk["wr5"] < 0  # NaN<0 = False → 保留 (与生产 fail-open 一致)
    pk["wr5_nan"] = pk["wr5"].isna()
    # div_pc 代理 (09-05 家族: 价格 5 日跑输成本上移)
    pk["divpc"] = pk["r5"] - pk["cost_chg5"]

    n_nomatch_panel = int(pk["close"].isna().sum())
    n_nonet = int(pk["net_pp"].isna().sum())  # 窗尾无 T+4
    ev = pk[pk["net_pp"].notna() & pk["close"].notna()].copy()  # 可评测集

    res = {
        "window": {
            "dates": [str(picks["date"].min().date()), str(picks["date"].max().date())],
            "n_days": int(n_days), "boards": boards,
            "pool_denom_boarddays": int(denom_pool),
            "picks_total": int(len(picks)),
            "cyq_exact_T_rate": round(100.0 * (pk["cyq_lag_days"] == 0).mean(), 1),
            "wr5_nan_picks": int(pk["wr5_nan"].sum()),
            "panel_no_match": n_nomatch_panel,
            "no_T4_exit": n_nonet,
            "evaluable": int(len(ev)),
        }
    }

    # ===== T0: 被标记组 vs 保留组 (基线对照, 分板+池) =====
    t0 = {}
    for scope, d in [("pool", ev)] + [(f"board={b}", ev[ev["board"] == b]) for b in boards]:
        dd = denom_pool if scope == "pool" else n_days
        t0[scope] = {
            "ALL": mstats(d, dd),
            "marked_wr5<0": mstats(d[d["marked"]], dd),
            "retained": mstats(d[~d["marked"]], dd),
            "marked_share_pct": round(100.0 * d["marked"].mean(), 1),
        }
    res["T0_marked_vs_retained"] = t0

    # ===== 因子分组表 (被标记组内; 保留组池基线做对照) =====
    def bins_table(d_mark: pd.DataFrame, col, edges, labels) -> list:
        rows = []
        for lab, lo, hi in labels:
            sub = d_mark[(d_mark[col] >= lo) & (d_mark[col] < hi)]
            s = mstats(sub, denom_pool)
            s.update({"factor": col, "bucket": lab})
            rows.append(s)
        return rows

    mk = ev[ev["marked"]].copy()
    ret_pool = mstats(ev[~ev["marked"]], denom_pool)
    factor_rows = []
    factor_rows += bins_table(mk, "pct250", None, [
        ("low<0.3", 0.0, 0.3), ("mid0.3-0.7", 0.3, 0.7), ("high>=0.7", 0.7, 1.01)])
    factor_rows += bins_table(mk, "amt_pct250", None, [
        ("low<0.3", 0.0, 0.3), ("mid0.3-0.7", 0.3, 0.7), ("high>=0.7", 0.7, 1.01)])
    factor_rows += bins_table(mk, "to20", None, [
        ("shrink<0.8", 0.0, 0.8), ("flat0.8-1.5", 0.8, 1.5), ("surge>=1.5", 1.5, 100.0)])
    for col in ("r5", "r20", "r60", "r250", "divpc"):
        factor_rows += bins_table(mk, col, None, [
            ("neg<0", -1e9, 0.0), ("pos>=0", 0.0, 1e9)])
    res["T1_factor_buckets_marked"] = {
        "retained_baseline": ret_pool, "marked_overall": mstats(mk, denom_pool),
        "rows": factor_rows,
    }

    # ===== 救回规则评估 =====
    RULES = {
        "R_low_p30": mk["pct250"] <= 0.3,
        "R_low_p50": mk["pct250"] <= 0.5,
        "R_r20pos": mk["r20"] > 0,
        "R_r60pos": mk["r60"] > 0,
        "R_r5pos": mk["r5"] > 0,
        "R_r250pos": mk["r250"] > 0,
        "R_to_surge": mk["to20"] >= 1.2,
        "R_low50_r20": (mk["pct250"] <= 0.5) & (mk["r20"] > 0),
        "R_low30_r20": (mk["pct250"] <= 0.3) & (mk["r20"] > 0),
        "R_low50_r60": (mk["pct250"] <= 0.5) & (mk["r60"] > 0),
        "R_low30_or_r20": (mk["pct250"] <= 0.3) | (mk["r20"] > 0),
    }
    # 组合模拟分母固定 denom_pool; 池 C = 全部票 - (marked & ~rule)
    poolA = ev
    poolB = ev[~ev["marked"]]
    rule_eval = {}
    for name, mask in RULES.items():
        rescued = mk[mask.fillna(False)]
        deleted = mk[~mask.fillna(False)]
        poolC = pd.concat([ev[~ev["marked"]], rescued], ignore_index=True)
        rule_eval[name] = {
            "rescued": mstats(rescued, denom_pool),
            "still_deleted": mstats(deleted, denom_pool),
            "poolC_rescue_gate": mstats(poolC, denom_pool),
        }
    res["T2_rescue_rules"] = {
        "poolA_no_gate": mstats(poolA, denom_pool),
        "poolB_current_gate": mstats(poolB, denom_pool),
        "rules": rule_eval,
    }

    # ===== 双半窗稳定性 (前 62 / 后 63 日) =====
    dsorted = np.sort(ev["T"].unique())
    split = dsorted[len(dsorted) // 2]
    halves = {"h1_first": ev[ev["T"] < split], "h2_second": ev[ev["T"] >= split]}
    dd_half = {k: v["T"].nunique() * len(boards) for k, v in halves.items()}
    # 半窗按显式条件函数重算 (规则 mask 与全窗同一逻辑)
    def rule_fn(df, name):
        c = {
            "R_low_p30": lambda d: d["pct250"] <= 0.3,
            "R_low_p50": lambda d: d["pct250"] <= 0.5,
            "R_r20pos": lambda d: d["r20"] > 0,
            "R_r60pos": lambda d: d["r60"] > 0,
            "R_r5pos": lambda d: d["r5"] > 0,
            "R_r250pos": lambda d: d["r250"] > 0,
            "R_to_surge": lambda d: d["to20"] >= 1.2,
            "R_low50_r20": lambda d: (d["pct250"] <= 0.5) & (d["r20"] > 0),
            "R_low30_r20": lambda d: (d["pct250"] <= 0.3) & (d["r20"] > 0),
            "R_low50_r60": lambda d: (d["pct250"] <= 0.5) & (d["r60"] > 0),
            "R_low30_or_r20": lambda d: (d["pct250"] <= 0.3) | (d["r20"] > 0),
        }[name]
        return c(df).fillna(False)
    stab = {}
    for name in RULES:
        row = {}
        for hname, hd in halves.items():
            mk_h = hd[hd["marked"]]
            m = rule_fn(mk_h, name)
            row[hname] = {
                "rescued": mstats(mk_h[m], dd_half[hname]),
                "still_deleted": mstats(mk_h[~m], dd_half[hname]),
                "poolB_vs_C_mean": [
                    mstats(hd[~hd["marked"]], dd_half[hname])["mean"],
                    mstats(
                        pd.concat([hd[~hd["marked"]], mk_h[m]]), dd_half[hname]
                    )["mean"],
                ],
            }
        stab[name] = row
    res["T3_half_window"] = {
        "split_date": str(pd.Timestamp(split).date()),
        "half_boarddays": dd_half,
        "rules": stab,
    }

    # ===== 002881 案例核对 (09-02~09-04 生产标记, 09-07 +9.99 / 09-08 +3.01) =====
    case = {}
    sym = "002881"
    cq1 = pd.read_parquet(CYQ_PATH, columns=["symbol", "date", "winner_ratio", "avg_cost"],
                          filters=[("symbol", "=", sym)])
    cq1 = cq1.drop_duplicates(["symbol", "date"]).sort_values("date").reset_index(drop=True)
    cq1["wr5"] = cq1["winner_ratio"].diff(5)
    pf1 = pf[(pf["symbol"] == sym) & (pf["date"] >= "2026-08-25")].set_index("date")
    for d in ("2026-09-02", "2026-09-03", "2026-09-04"):
        ts = pd.Timestamp(d)
        crow = cq1[cq1["date"] == ts]
        prow = pf1.loc[ts] if ts in pf1.index else None
        case[d] = {
            "wr5": round(float(crow["wr5"].iloc[0]), 4) if len(crow) else None,
            "pct250": round(float(prow["pct250"]), 2) if prow is not None else None,
            "r20": round(float(prow["r20"]), 3) if prow is not None else None,
            "r60": round(float(prow["r60"]), 3) if prow is not None else None,
            "r5": round(float(prow["r5"]), 3) if prow is not None else None,
            "to20": round(float(prow["to20"]), 2) if prow is not None else None,
            "rescued_by_low50_r20": bool(
                prow is not None and prow["pct250"] <= 0.5 and prow["r20"] > 0
            ),
            "rescued_by_low30_r20": bool(
                prow is not None and prow["pct250"] <= 0.3 and prow["r20"] > 0
            ),
            "rescued_by_low_p50": bool(prow is not None and prow["pct250"] <= 0.5),
        }
    res["T4_case_002881"] = case

    # ===== 落盘 (WORM) =====
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(OUT_DIR, f"chip_rescue_replay_20260908_{ts}")
    ev_out = ev.copy()
    ev_out["T"] = ev_out["T"].astype(str)
    ev_out.to_parquet(base + ".parquet", index=False)
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)

    # ===== 控制台摘要 =====
    print("== window ==", res["window"])
    print("\n== T0 pool ==")
    print(json.dumps(t0["pool"], ensure_ascii=False, indent=1))
    for b in boards:
        print(f"-- {b}: marked_share={t0[f'board={b}']['marked_share_pct']}% "
              f"marked={t0[f'board={b}']['marked_wr5<0']} "
              f"retained={t0[f'board={b}']['retained']}")
    print("\n== retained baseline ==", ret_pool)
    print("== marked overall ==", mstats(mk, denom_pool))
    fr = pd.DataFrame(factor_rows)
    print("\n== T1 factor buckets (marked group) ==")
    print(fr.to_string(index=False))
    print("\n== T2 rules: rescued / still_deleted / poolC ==")
    for name, r in rule_eval.items():
        print(f"{name}: rescued(n={r['rescued']['n']}, win={r['rescued']['win']}, "
              f"mean={r['rescued']['mean']}, big={r['rescued']['bigloss']}) | "
              f"del(n={r['still_deleted']['n']}, win={r['still_deleted']['win']}, "
              f"mean={r['still_deleted']['mean']}, big={r['still_deleted']['bigloss']}) | "
              f"poolC(mean={r['poolC_rescue_gate']['mean']}, win={r['poolC_rescue_gate']['win']}, "
              f"wd={r['poolC_rescue_gate']['win_per_day']})")
    print("\npoolA(no gate):", res["T2_rescue_rules"]["poolA_no_gate"])
    print("poolB(current):", res["T2_rescue_rules"]["poolB_current_gate"])
    print("\n== T3 half-window (rescued vs deleted mean) ==")
    for name in ("R_low_p50", "R_low50_r20", "R_low30_r20", "R_r20pos", "R_low_p30",
                 "R_low50_r60", "R_low30_or_r20", "R_to_surge"):
        r = stab[name]
        print(f"{name}: h1 resc={r['h1_first']['rescued']['mean']} del={r['h1_first']['still_deleted']['mean']}"
              f" | h2 resc={r['h2_second']['rescued']['mean']} del={r['h2_second']['still_deleted']['mean']}"
              f" | h1 B/C={r['h1_first']['poolB_vs_C_mean']} h2 B/C={r['h2_second']['poolB_vs_C_mean']}")
    print("\n== T4 case 002881 ==")
    print(json.dumps(case, ensure_ascii=False, indent=1))
    print("\nOUT:", base + ".parquet / .json")


if __name__ == "__main__":
    main()
