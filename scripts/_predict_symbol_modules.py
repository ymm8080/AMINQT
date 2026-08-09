#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对指定 symbols 用 legacy(main/dual) + 并行(sniper/fusion/slow_bull) 全模块做价格预测.

- legacy:   V35Predictor 逐股 pred_ret_1d/2d/3d/5d + prob_up + pain_prob (切片 V3 面板末 300 交易日).
- parallel: merged 短名单 (sniper/fusion 池分 → mag_10d 校准 T+10 预期幅度) + slow_bull 池
            (ADX 硬门槛 → 排名键 → 信号), 用生产 load_panel() 行集, 与 runner 同源.

用法:
  python scripts/_predict_symbol_modules.py 301373 300049 603082 002281 300911
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from config.settings import PANEL_V3_PATH

LEGACY_BUNDLES = {
    "main": "models/pipeline1/main_current.pkl",
    "dual": "models/pipeline1/dual_current.pkl",
}

LEGACY_SHOW = [
    "symbol",
    "board",
    "industry",
    "close",
    "day_change",
    "pred_ret_1d",
    "pred_ret_2d",
    "pred_ret_3d",
    "pred_ret_5d",
    "prob_up",
    "prob_up_2d",
    "prob_up_3d",
    "prob_up_5d",
    "pred_q50",
    "pain_prob",
    "rank_score",
    "composite_score",
]


def _z6(s) -> str:
    return str(s).zfill(6)


def predict_legacy(symbols):
    """legacy main/dual 逐股预测. 返回长表含 module 列 (legacy_main/legacy_dual)."""
    from app.pipeline1.daily_pipeline import DailySelectionPipeline
    from app.pipeline1.data_supply import DataSupplyChain

    want = {_z6(s) for s in symbols}
    t0 = time.time()
    dates = pq.read_table(str(PANEL_V3_PATH), columns=["date"]).to_pandas()["date"]
    uniq = np.unique(dates.values)
    cut = pd.Timestamp(uniq[-300])
    panel = pq.read_table(
        str(PANEL_V3_PATH), filters=[("date", ">=", cut)]
    ).to_pandas()
    print(
        f"[legacy] 切片 {cut.date()}..{pd.Timestamp(uniq[-1]).date()} "
        f"({time.time()-t0:.0f}s)",
        flush=True,
    )

    pipe = DailySelectionPipeline(supply=DataSupplyChain(), bundle_paths=LEGACY_BUNDLES)
    main_df, dual_df, valve = pipe.cleaner.run_inference(panel)
    del panel
    print(
        f"[legacy] clean main={len(main_df):,} dual={len(dual_df):,} "
        f"valve={valve}",
        flush=True,
    )

    out = []
    for board, dfb, csr in (("main", main_df, False), ("dual", dual_df, True)):
        if len(dfb) == 0 or board not in pipe.predictor.bundles:
            continue
        cols = pipe.predictor.bundles[board]["feature_cols"]
        feat = pipe.features.build(
            dfb,
            pipe.float_shares_map,
            inference_cols=cols,
            cross_sectional_rank=csr,
        )
        sub = feat[feat["symbol"].astype(str).str.zfill(6).isin(want)].copy()
        if len(sub) == 0:
            continue
        pred = pipe.predictor.predict(sub, board)
        pred["symbol"] = pred["symbol"].astype(str).str.zfill(6)
        pred["module"] = f"legacy_{board}"
        latest = (
            dfb[dfb["symbol"].astype(str).str.zfill(6).isin(want)]
            .sort_values("date")
            .groupby("symbol")
            .tail(1)
        )
        for col in ("close", "pre_close", "ATR_pct", "adv20", "amount", "turnover_rate"):
            if col in latest.columns:
                pred[col] = (
                    latest.set_index("symbol").reindex(pred["symbol"])[col].values
                )
        out.append(pred)
    print(f"[legacy] done ({time.time()-t0:.0f}s)", flush=True)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def predict_parallel(symbols):
    """并行 sniper/fusion merged 短名单 + slow_bull 池. 返回两组长表含 module 列."""
    from app.pipeline_parallel import signals
    from app.pipeline_parallel.backtest import build_merged_shortlist, load_panel
    from app.pipeline_parallel.config import SLOW_BULL

    want = {_z6(s) for s in symbols}
    t0 = time.time()
    work = load_panel()
    work["symbol"] = work["symbol"].astype(str).str.zfill(6)
    latest = work["date"].max()
    print(
        f"[parallel] 行集 {len(work):,}r stocks={work['symbol'].nunique():,} "
        f"latest={latest:%Y-%m-%d} ({time.time()-t0:.0f}s)",
        flush=True,
    )

    # ── merged 短名单 (sniper/fusion): 全池 mag_10d 校准 → 每股 score/mag/rk ──
    merged = build_merged_shortlist(work, top_n=999999)
    merged["symbol"] = merged["symbol"].astype(str).str.zfill(6)
    m = merged[merged["symbol"].isin(want)].copy()
    if not m.empty:
        m = m.sort_values("date").groupby("symbol", as_index=False).tail(1)
        m["module"] = "parallel_sniper/fusion"
        m["in_top10"] = m["rk"] <= 10
    print(f"[parallel] merged 校准 done ({time.time()-t0:.0f}s)", flush=True)

    # ── slow_bull: 逐股最新行 (gate + 信号 + 排名) + 是否进池 ──
    pool_members = set()
    for b in ("main", "dual"):
        p = signals.daily_slowbull_pool(work, None, b, SLOW_BULL, SLOW_BULL.top_n)
        if p.empty:
            continue
        p["symbol"] = p["symbol"].astype(str).str.zfill(6)
        pool_members.update(p["symbol"].tolist())
    w = work[work["symbol"].isin(want)].copy()
    sb_rows = (
        w.sort_values("date").groupby("symbol", as_index=False).tail(1)
    )
    print(f"[parallel] slow_bull 池含目标 {sorted(pool_members & want)} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return m, sb_rows, pool_members, latest


def main() -> int:
    args = sys.argv[1:]
    symbols = [a for a in args if a.strip()] or ["300049"]
    want = [_z6(s) for s in symbols]
    print(f"预测目标: {want}  (多模块: legacy main/dual + parallel sniper/fusion/slow_bull)", flush=True)

    # ── legacy ──
    leg = predict_legacy(want)
    show_legacy = [c for c in LEGACY_SHOW if c in leg.columns] + ["module"] if not leg.empty else []
    if not leg.empty:
        print("\n" + "=" * 130)
        print("【legacy 模块】逐股预测 (pred_ret=预期涨幅, prob_up=上涨概率, pain_prob=回撤风险)")
        print("=" * 130)
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        print(leg[show_legacy].to_string(index=False), flush=True)
        missing_leg = sorted(set(want) - set(leg["symbol"]))
        if missing_leg:
            print(f"[legacy][miss] {missing_leg} 未被 legacy 预测 (清洗剔除或缺模型)", flush=True)

    # ── parallel ──
    merged, sb_rows, pool_members, latest = predict_parallel(want)
    if not merged.empty:
        print("\n" + "=" * 130)
        print(f"【并行 merged 短名单】sniper/fusion 池分→mag_10d 校准 (T+10 预期 close-to-close 幅度), "
              f"最新交易日 {latest:%Y-%m-%d}")
        print("=" * 130)
        cols = [c for c in
                ("symbol", "board", "module", "score", "mag", "rk", "in_top10",
                 "systems", "co_occur") if c in merged.columns]
        print(merged[cols].to_string(index=False), flush=True)
        not_top = merged.loc[~merged["in_top10"], "symbol"].tolist()
        if not_top:
            print(f"[parallel] 未进 TOP-10: {not_top} (仍给出 score/mag 供参考)", flush=True)
        missing_m = sorted(set(want) - set(merged["symbol"]))
        if missing_m:
            print(f"[parallel][miss] {missing_m} 不在 merged 行集 (无 label_pm_10d_net 或缺特征)", flush=True)

    if not sb_rows.empty:
        print("\n" + "=" * 130)
        print(f"【并行 slow_bull 池】ADX 硬门槛+排名键, 最新交易日 {latest:%Y-%m-%d}")
        print("=" * 130)
        sb_cols = [c for c in
                   ("symbol", "board", "close_cont", "ma5", "ma20", "adx", "pdi", "mdi",
                    "adx_rise5", "turnover_rate", "vol_ratio", "dev5",
                    "gate_slow_bull", "slow_bull_regime", "rps_60", "pv_corr_5",
                    "pullback_ma5", "pullback_ma10", "shrink_vol", "chase_high",
                    "vol_spike_up", "adx_falling", "below_ma20", "adx_broken",
                    "big_drop", "below_ma5_3d", "turnover_spike", "tp_80_div",
                    "trail8_dd", "trail8_trigger") if c in sb_rows.columns]
        out = sb_rows.copy()
        out["in_pool"] = out["symbol"].isin(pool_members)
        print(out[sb_cols + ["in_pool"]].to_string(index=False), flush=True)
        miss_sb = sorted(set(want) - set(sb_rows["symbol"]))
        if miss_sb:
            print(f"[parallel][miss] {miss_sb} 不在 slow_bull 行集 (可交易性门剔除/停牌?)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
