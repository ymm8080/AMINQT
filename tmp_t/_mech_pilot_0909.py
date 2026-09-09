# -*- coding: utf-8 -*-
"""机制分离试点 Phase-A: 规则分桶 × 标签/因子交互诊断 (任务#5, 09-09).

问题: 单模型对所有 (stock,day) 一刀切, 机制不同 (超跌反弹/连板动量/温和走平/
放量冲高) 因子方向可能翻转 — 09-08 行情路由终审 "环境价值在因子交互层" 的
个股层版本.

设计 (全 t-1 可观测, 禁 look-ahead):
  桶 = 规则分类, 互斥优先级 other 兜底:
    board_momentum   近3日有涨停 (按前缀 10/20/30% 阈值)
    oversold_reb     mom20 <= -15%
    surge_highvol    mom20 >= +15% 或 20日已实现波动top三成
    grind_quiet      |mom20| < 5% 且波动 bottom 三成
    other
  诊断1: 桶 × (label_pm_1d_net / label_pm_10d_net) 均值/胜率/样本数 — 标签分离度
  诊断2: 桶 × 头部因子 TS IC (ovd_wdist, winner_ratio, mom20, mf_lg_net_ratio,
        eop_vol... mf/bs5 若在) — IC 翻号 = 交互层价值证据
判据 (预注册): 任一桶 |label均值差| >= 2x 全样本 且 >=2 因子桶间 IC 差 >= 0.03
  → 双头路由 Phase-B 立项; 否则试点判死 (方法条件性档案留存).

运行时: 只读面板少数列 (~1GB), 排在面板重写队列之后 (predecessor-only 守卫).
终态: MECH_PILOT_DONE + tmp_t/_mech_pilot_result.json
"""
import json
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = r"D:\AMINQT\AMINQT CODES"
sys.path.insert(0, ROOT)
PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
MF = os.path.join(ROOT, "tmp_min", "_l2_mf_feats.parquet")
BS5 = os.path.join(ROOT, "tmp_min", "_l2_bs5_feats_pilot.parquet")
RESULT = os.path.join(ROOT, "tmp_t", "_mech_pilot_result.json")
OOS = 250


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    from scripts._run_guard import find_conflicts
    PRED = ("_ohlc_repair_v3_0909.py", "_ovd_backfill_v3_0909.py",
            "_cyq_recompute_v3_0909.py")
    conflicts = find_conflicts(sentinels=PRED)
    attempt = 0
    while conflicts and attempt < 60:
        sens = [c.get("sentinel", "?") for c in conflicts]
        log(f"run-guard 冲突, 等10min重试 ({attempt + 1}/60): {sens}")
        time.sleep(600)
        attempt += 1
        conflicts = find_conflicts(sentinels=PRED)
    if conflicts:
        log("MECH_PILOT_DONE status=aborted reason=guard_conflict")
        return 2

    t0 = time.time()
    need = ["symbol", "date", "pre_close", "close", "high", "low", "amount",
            "turnover_rate", "winner_ratio"]
    d = pq.read_table(PANEL, columns=need).to_pandas()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values(["symbol", "date"]).reset_index(drop=True)
    log(f"panel rows={len(d):,} stocks={d['symbol'].nunique()}")

    # ovd (若面板已回补) / mf / bs5 特征 join
    extra = {}
    if "ovd_wdist" in pq.ParquetFile(PANEL).schema_arrow.names:
        ov = pq.read_table(PANEL, columns=["symbol", "date", "ovd_wdist",
                                           "ovd_15p"]).to_pandas()
        ov["date"] = pd.to_datetime(ov["date"])
        extra["ovd"] = ov
    if os.path.exists(MF):
        mf = pd.read_parquet(MF)
        mf["date"] = pd.to_datetime(mf["date"])
        extra["mf"] = mf
    if os.path.exists(BS5):
        bs5 = pd.read_parquet(BS5)
        bs5["date"] = pd.to_datetime(bs5["date"])
        extra["bs5"] = bs5
    for k, v in extra.items():
        d = d.merge(v, on=["symbol", "date"], how="left")
        log(f"join {k}: +{len(v.columns) - 2} cols")

    # t-1 机制特征 (禁 look-ahead: 桶判定只用截至 t-1 收盘的信息, 全部 shift(1))
    g = d.groupby("symbol", sort=False)
    pct = d["close"] / d["pre_close"] - 1.0
    limit_th = pd.Series(np.where(d["symbol"].str.startswith(("30", "68")), 0.195,
                          np.where(d["symbol"].str.startswith("8"), 0.295, 0.095)),
                         index=d.index)
    d["is_limit"] = (pct >= limit_th).astype(float)
    d["mom20_raw"] = g["close"].transform(lambda s: s / s.shift(20) - 1.0)
    d["rv20_raw"] = g["close"].transform(
        lambda s: s.pct_change().rolling(20).std())
    d["mom20"] = g["mom20_raw"].shift(1)
    d["rv20"] = g["rv20_raw"].shift(1)
    d["limit3"] = g["is_limit"].transform(lambda s: s.rolling(3).max()).shift(1)

    # 标签 (前瞻, 只用于评估不用于桶)
    d["label_1d"] = g["close"].shift(-1) / d["close"] - 1.0
    d["label_10d"] = g["close"].shift(-10) / d["close"] - 1.0
    cut = d["date"].max() - pd.Timedelta(days=int(OOS * 1.5))

    # 波动阈值只用 OOS 之前的数据定 (诊断层同样禁前视)
    train_rv = d.loc[d["date"] < cut, "rv20"].dropna()
    rv70, rv30 = train_rv.quantile(0.7), train_rv.quantile(0.3)
    conds = [
        (d["limit3"] > 0),
        (d["mom20"] <= -0.15),
        ((d["mom20"] >= 0.15) | (d["rv20"] >= rv70)),
        ((d["mom20"].abs() < 0.05) & (d["rv20"] <= rv30)),
    ]
    names = ["board_momentum", "oversold_reb", "surge_highvol", "grind_quiet"]
    d["mech"] = "other"
    # 顺序申请, 先到先得 → 列表序即优先级 (board 最高)
    for c, n in zip(conds, names):
        d.loc[c & (d["mech"] == "other"), "mech"] = n
    oos = d[d["date"] >= cut].copy()
    log(f"OOS窗 rows={len(oos):,} ({oos['date'].min().date()}~) "
        f"桶分布: {oos['mech'].value_counts().to_dict()}")

    # 诊断1: 桶 × 标签
    lab = oos.groupby("mech").agg(
        n=("label_1d", "size"),
        mean_1d=("label_1d", "mean"), win_1d=("label_1d", lambda s: (s > 0).mean()),
        mean_10d=("label_10d", "mean"), win_10d=("label_10d", lambda s: (s > 0).mean()),
    ).round(5)
    lab["n"] = lab["n"].astype(int)
    log("桶 × 标签:\n" + lab.to_string())

    # 诊断2: 桶 × 因子 TS IC (因子先 shift(1) 用 t-1 值对 t 收益)
    factors = {}
    for src, cols in {
        "panel": ["winner_ratio", "mom20", "rv20"],
        "ovd": ["ovd_wdist", "ovd_15p"],
        "mf": ["mf_lg_net_ratio", "mf_elg_net_ratio", "mf_net_ratio"],
        "bs5": ["eop_vol_ratio", "intraday_recovery", "vwap_dev_close"],
    }.items():
        for c in cols:
            if c in oos.columns:
                factors[c] = c
    ic_rows = []
    for c in factors:
        f = g[c].shift(1).reindex(oos.index)
        for lab_col in ("label_1d", "label_10d"):
            m = f.notna() & oos[lab_col].notna()
            if m.sum() < 500:
                continue
            row = {"factor": c, "label": lab_col, "all": np.corrcoef(
                f[m], oos.loc[m, lab_col])[0, 1]}
            for bucket in names + ["other"]:
                mb = m & (oos["mech"] == bucket)
                row[bucket] = (np.corrcoef(f[mb], oos.loc[mb, lab_col])[0, 1]
                               if mb.sum() >= 200 else np.nan)
            ic_rows.append(row)
    ic = pd.DataFrame(ic_rows).round(4)
    log("因子 × 桶 IC:\n" + ic.to_string(index=False))

    # 预注册判据
    base10 = oos["label_10d"].mean()
    lab_sep = (lab["mean_10d"].drop("other", errors="ignore").sub(base10)
               .abs().max()) if "mean_10d" in lab else 0
    ic_spread = 0.0
    for _, r in ic.iterrows():
        vals = [r[b] for b in names if b in r and r[b] == r[b]]
        if len(vals) >= 2:
            ic_spread = max(ic_spread, max(vals) - min(vals))
    verdict = ("GO-PhaseB" if (lab_sep >= 2 * abs(base10) and ic_spread >= 0.03)
               else "KILL")
    log(f"判据: 桶标签分离度={lab_sep:.5f} (基线10d={base10:+.5f}, 阈 {2 * abs(base10):.5f}); "
        f"桶间IC极差={ic_spread:.4f} (阈 0.03) → {verdict}")

    out = {
        "status": "ok", "verdict": verdict,
        "label_table": lab.reset_index().to_dict("records"),
        "ic_table": ic.to_dict("records"),
        "lab_separation": round(float(lab_sep), 6),
        "ic_spread": round(float(ic_spread), 4),
        "oos_rows": int(len(oos)),
        "elapsed_s": round(time.time() - t0, 1),
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    log(f"MECH_PILOT_DONE status=ok verdict={verdict} -> {RESULT}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        log("MECH_PILOT_DONE status=aborted")
        sys.exit(1)
