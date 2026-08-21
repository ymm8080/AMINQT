"""_diag_300911_today.py — 今天 (末面板日) 300911 在 legacy 生产链路的位置诊断 (2026-08-19).

用当前生产配置 (E6=10%) 复现逐日推理: cleaner → features → predict → compute_scores → 生产闸,
输出 300911 的池内 amount rank / 闸门通过 / pred_ret_10d 排名位置 (top10 视角).
用法: python scripts/_diag_300911_today.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.list_generator import ListGenerator
from app.pipeline1.predictor import V35Predictor
from config.settings import PANEL_V3_PATH

SYMBOL = "300911"


def main() -> int:
    cleaner = CleaningPipeline()
    features = FeatureEngineV35()
    lister = ListGenerator()
    predictor = V35Predictor({"dual": "models/pipeline1/dual_current.pkl"})

    panel = pd.read_parquet(str(PANEL_V3_PATH))
    today = pd.to_datetime(panel["date"]).max()
    print(f"[panel] max={pd.Timestamp(today).date()} rows={len(panel):,}")

    main_df, dual_df, state = cleaner.run_inference(panel)
    del panel
    print(f"[clean] valve={state} dual_rows={len(dual_df):,}")

    # ---- 池内位置: E6 切人后 300911 是否幸存 ----
    day_dual = dual_df[pd.to_datetime(dual_df["date"]) == today]
    print(f"[today] {pd.Timestamp(today).date()} dual_pool={len(day_dual)}")
    s = day_dual[day_dual["symbol"] == SYMBOL]
    if s.empty:
        # 区分: 被 E6 切 vs 未进 top-200 池 — 查 E6 前池内 amount rank
        pre = _pool_before_e6(panel, today)
        if SYMBOL in pre.index:
            print(
                f"[E6] {SYMBOL} 被 E6=10% 切掉: E6 前池内 amount rank_pct={pre.loc[SYMBOL]:.3f} (<0.10)"
            )
        else:
            print(f"[E6] {SYMBOL} 未进 top-200 池 (E6 前已无此票)")
        return 0
    print(f"[E6] {SYMBOL} 幸存, amount={s.iloc[0]['amount']:.2e}")
    pool_rank = day_dual["amount"].rank(pct=True)
    print(f"     amount rank_pct={pool_rank[s.index[0]]:.3f} (E6=10% 切 <0.10)")

    # ---- 过闸 + 排名 ----
    feat = features.build(
        day_dual,
        None,
        inference_cols=predictor.bundles["dual"]["feature_cols"],
        cross_sectional_rank=True,
    )
    pred = predictor.predict(feat, "dual")
    if pred.empty:
        print("[pred] empty")
        return 0
    scored = lister.compute_scores(pred)
    if SYMBOL in scored["symbol"].values:
        r = scored[scored["symbol"] == SYMBOL].iloc[0]
        prob_key = "prob_up" if "prob_up" in scored else "compound_prob"
        ret_key = "pred_ret_10d" if "pred_ret_10d" in scored else "compound_ret"
        print(
            f"[300911 gates] prob={r[prob_key]:.3f} base_rate={r['base_rate']:.3f} "
            f"pred_ret_10d={r[ret_key]:+.3f} pain_prob={r.get('pain_prob', float('nan')):.3f}"
        )
    base_mask = _gate_mask(scored)
    rows = scored[base_mask].copy()
    rows = rows.sort_values("pred_ret_10d", ascending=False).reset_index(drop=True)
    print(f"[gates] 过闸 {len(rows)} 只 (top10 视角)")
    if SYMBOL in rows["symbol"].values:
        rk = rows.index[rows["symbol"] == SYMBOL][0] + 1
        r = rows.iloc[rk - 1]
        print(
            f"   {SYMBOL} rank {rk}/{len(rows)}  pred_ret_10d={r['pred_ret_10d']:+.3f} prob={r.get('prob_up', r.get('compound_prob', float('nan'))):.3f}"
        )
        print("   top10:")
        for i, row in rows.head(10).iterrows():
            print(
                f"    {i + 1:>2} {row['symbol']}  pred_ret_10d={row['pred_ret_10d']:+.3f}"
            )
    else:
        print(f"   {SYMBOL} 未过闸 (被 entry_filter 拦下)")
    return 0


def _gate_mask(scored: pd.DataFrame) -> pd.Series:
    """生产 entry_filter 非 bear 口径 (与 _diag_legacy_hitrate_topn 一致)."""
    m = pd.Series(True, index=scored.index)
    if "prob_up" in scored:
        m &= scored["prob_up"] > scored["base_rate"]
    elif "compound_prob" in scored:
        m &= scored["compound_prob"] > scored["base_rate"]
    if "pred_ret_10d" in scored:
        m &= scored["pred_ret_10d"] > 0
    elif "compound_ret" in scored:
        m &= scored["compound_ret"] > 0
    if "pred_q50" in scored:
        m &= scored["pred_q50"] > 0
    if "pain_prob" in scored:
        m &= scored["pain_prob"].fillna(0) <= 0.5
    return m


def _pool_before_e6(panel: pd.DataFrame, today) -> pd.Series:
    """E6 前 (step0-4 后) 的 dual 池内 amount rank_pct, 按 symbol."""
    cleaner = CleaningPipeline()
    main, dual = cleaner.step0_board_split(panel)
    dual = cleaner.step3_extreme(
        cleaner.step2_liquidity(cleaner.step1_base_state(dual), apply_top_n=True)
    )
    both = pd.concat([main, dual], ignore_index=True)
    both, _ = cleaner.step4_tradability(both, inference_only=True)
    day = both[pd.to_datetime(both["date"]) == today]
    day = day[day["symbol"].str.startswith(("30", "68"))]  # GEM+STAR = dual
    return day.set_index("symbol")["amount"].rank(pct=True)


if __name__ == "__main__":
    raise SystemExit(main())
