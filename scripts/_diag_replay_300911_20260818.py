"""_diag_replay_300911_20260818.py — 若 8/18 300911 在候选池, 生产模型 (dual_20260818)
会给什么预测? (2026-08-19, 08-20 修正: 生产 bundle + 强制加回 8/18 当日行)

背景: 300911 8/12 在池且入选, 8/13 试盘放量, 8/14-8/18 生产池无它 (cleaner
liquidity top-200 截断 / 换手不稳罚分). 本脚本强制把 300911 的 8/18 行加回
cleaner 输出 → 全池截面特征 → 8/18 生产模型推理, 看模型是否"本来就该抓它".

修正 (2026-08-20): 原脚本 BUNDLE 误用 20260819 (8/19 训练含 8/18 数据 = 前瞻),
成员检查只看全历史且 iloc[0] 取最早行 — 结论无效. 现在用 8/18 生产 bundle
dual_20260818.pkl, 只打印 8/18 当日预测行.

用法: python scripts/_diag_replay_300911_20260818.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.filterwarnings("ignore")

import pandas as pd

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.dual_track_trainer import DualTrackTrainer
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.predictor import V35Predictor
from config.settings import PANEL_V3_PATH

TARGET = "300911"
TRADE_DATE = pd.Timestamp("2026-08-18")
BUNDLE = r"models/pipeline1/dual_20260818.pkl"


def main() -> int:
    p = pd.read_parquet(str(PANEL_V3_PATH), filters=[("board", "in", ["GEM", "STAR"])])
    p["symbol"] = p["symbol"].astype(str)
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    p = (
        p[p["date"] <= TRADE_DATE]
        .sort_values(["symbol", "date"])
        .reset_index(drop=True)
    )
    print(f"[panel] {len(p):,} 行 dual 历史 (截至 8/18)", flush=True)

    cleaner = CleaningPipeline()
    _m, dual, state = cleaner.run_inference(p)
    dual["symbol"] = dual["symbol"].astype(str)
    in_pool_818 = TARGET in set(dual.loc[dual["date"] == TRADE_DATE, "symbol"])
    print(
        f"[cleaner] dual 输出 {len(dual):,} 行, 8/18 截面含 300911: {in_pool_818}",
        flush=True,
    )

    p911 = p[p["symbol"] == TARGET].copy()
    if "turnover_stability_5" not in p911.columns:
        g = p911.groupby("symbol")["turnover_rate"]
        std5 = g.rolling(5, min_periods=1).std().reset_index(level=0, drop=True)
        mean5 = g.rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        p911["turnover_stability_5"] = (std5 / mean5.replace(0, pd.NA)).fillna(0.0)
        p911["churn_suspect"] = (p911["turnover_stability_5"] > 0.5).astype(int)
    dual = pd.concat([dual, p911], ignore_index=True)
    dual = dual.drop_duplicates(["symbol", "date"], keep="last")
    print(
        f"[replay] 300911 全行强制加回 (8/18 当日 {TARGET in set(dual.loc[dual['date'] == TRADE_DATE, 'symbol'])})"
    )

    bundle = DualTrackTrainer.load(BUNDLE)
    feat = FeatureEngineV35().build(
        dual,
        float_shares_map=None,
        inference_cols=bundle["feature_cols"],
        cross_sectional_rank=True,
    )
    print(f"[feature] 特征面板 {len(feat):,} 行 × {len(feat.columns)} 列", flush=True)

    pred = V35Predictor({"dual": BUNDLE}).predict(feat, "dual")
    # predict 输出每 symbol 最新一行 (无 date 列); 300911 最新行日期应=8/18 (强制加回)
    r = pred[pred["symbol"] == TARGET]
    print("\n== 8/18 生产模型 (dual_20260818) 对 300911 的预测 ==")
    if len(r):
        x = r.iloc[0]
        last_date = feat.loc[feat["symbol"] == TARGET, "date"].max()
        print(f"  (预测行日期: {last_date.date()})")
        for c in r.columns:
            if c == "symbol":
                continue
            print(f"  {c}: {x[c]}")
        # 对照: 8/18 池内全体预测分布 (中位)
        print("\n== 对照: 8/18 池内全体预测分布 (中位) ==")
        med = pred.select_dtypes("number").median(numeric_only=True)
        for c in ["pred_ret_3d", "pred_ret_5d", "pred_ret_10d", "prob_up"]:
            if c in med:
                print(f"  {c}: 300911={x[c]:+.4f}  池中位={med[c]:+.4f}")
        # 300911 在池内分位 (pred_ret_10d)
        print("\n== 300911 在 8/18 池内分位 (pred_ret_10d) ==")
        rk = pred["pred_ret_10d"].rank(pct=True).loc[x.name]
        print(f"  pred_ret_10d 分位: {rk:.3f}")
    else:
        print("  300911 无预测行!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
