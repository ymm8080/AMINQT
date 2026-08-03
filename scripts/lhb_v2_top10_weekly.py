# -*- coding: utf-8 -*-
"""KIMI LHB v2.0 TOP10 周频选股能力评估 — 测试模型能否可靠选出"值得买"的标的.

设计: 两个训练目标 × 两个实际持有期, 全面测模型能力:
  训练目标: r1 (现有系统, 预测次日) / r5 (周频优化, 预测 5 日)
  实际持有: r3 (T+3 收盘卖出) / r5 (T+5 收盘卖出)
  每日按模型预测在 LHB 上榜池取 top-N, 逐组合报告三类精度:

  预测精度 (Forecast, 对模型自身训练目标): bias/MAE/RMSE — 涨幅预测值偏不偏
  概率精度 (Probability, 对实际持有期): 命中率 P(ret>0) vs 池基准率, 跑赢池中位数率
  排名精度 (Ranking, 对实际持有期): 日频 rankIC, 入选股实现收益分位 (0.5=随机),
              超额 vs 池均值
  回测 (对实际持有期): t+1 开盘买入 → t+h 收盘卖出, 佣金+印花+滑点, 净周收益与周胜率

标签已由 lhb_v2_train_eval.add_targets 处理 hfq 重定基伪跳 (r3/r5 跨 2026-07-27
作废), 保证测试期标签干净.

输出 (WORM, DATA_OTHERS + 日期后缀):
  lhb_v2_top10_<ts>.xlsx      精度摘要 (2 模型 × 2 持有期)
  lhb_v2_top10_picks_<ts>.parquet  测试期全部入选明细

用法:
  python scripts/lhb_v2_top10_weekly.py
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import DATA_OTHERS_DIR, LHB_V2_EVAL, LHB_V2_TOP10  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("lhb_v2_top10_weekly")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "lhb_te", os.path.join(_SCRIPT_DIR, "lhb_v2_train_eval.py")
)
_lhb_te = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lhb_te)

load_panel = _lhb_te.load_panel
add_targets = _lhb_te.add_targets
run_dim34 = _lhb_te.run_dim34
FEATURES = _lhb_te.FEATURES


def train_predict(
    tr: pd.DataFrame, te: pd.DataFrame, av: list[str], target: str, cfg: dict
) -> pd.DataFrame:
    """训练目标模型并预测测试池 (仅保留自身训练目标有效的行)."""
    d = tr.dropna(subset=av + [target])
    m = lgb.LGBMRegressor(verbose=-1, **cfg["lgb"])
    m.fit(d[av], d[target])
    te = te.copy()
    te["pred"] = m.predict(te[av])
    te = te.dropna(subset=[target])
    return te, m


def pick_top10(sub: pd.DataFrame, pred_col: str, top_n: int) -> pd.DataFrame:
    """每日按预测取 top-n (保留 r3/r5 两个实现收益列)."""
    sub = sub[sub["r3"].notna() & sub["r5"].notna()].copy()
    sub = sub.sort_values(["date", pred_col], ascending=[True, False])
    blocks = []
    for d, g in sub.groupby("date"):
        g = g.head(top_n).copy()
        g["pick_rank"] = np.arange(len(g))
        blocks.append(g)
    return pd.concat(blocks, ignore_index=False) if blocks else pd.DataFrame()


def _daily_spearman(
    sub: pd.DataFrame, pred_col: str, label: str, cfg: dict
) -> float | None:
    ics = []
    for _, g in sub.groupby("date"):
        v = g[[pred_col, label]].dropna()
        if len(v) < cfg["min_ic_n"]:
            continue
        r, _ = spearmanr(v[pred_col], v[label])
        if not np.isnan(r):
            ics.append(r)
    if len(ics) < cfg["min_ic_obs"]:
        return None
    return float(np.mean(ics))


def evaluate(
    picks: pd.DataFrame,
    sub: pd.DataFrame,
    cfg: dict,
    top10: dict,
    model: str,
    label: str,
    own_label: str,
) -> dict:
    """对给定实际持有期 label 计算三类精度 + 成本后回测.

    sub: 该模型预测的 r3/r5 有效池; own_label: 模型自身训练目标 (预测精度基准).
    """
    if len(picks) == 0:
        return {"model": model, "label": label, "n_picks": 0}

    # 预测精度: 模型自身目标 (pool 级)
    bias = float((sub["pred"] - sub[own_label]).mean())
    mae = float((sub["pred"] - sub[own_label]).abs().mean())
    rmse = float(np.sqrt(((sub["pred"] - sub[own_label]) ** 2).mean()))

    # 成本
    buy_cost = top10["cost_commission"] + top10["cost_slippage"]
    sell_cost = top10["cost_commission"] + top10["cost_stamp"] + top10["cost_slippage"]

    # 入选股对 label 的池统计
    sub_l = sub[sub[label].notna()].copy()
    pkl = picks[picks[label].notna()].copy()
    if len(pkl) == 0:
        return {"model": model, "label": label, "n_picks": 0}
    pool_daily = (
        sub_l.groupby("date")[label]
        .agg(
            pool_mean="mean",
            pool_med="median",
            base_rate=lambda x: float((x > 0).mean()),
        )
        .reset_index()
    )
    pkl = pkl.merge(pool_daily, on="date", how="left")
    pct = sub_l.groupby("date")[label].rank(pct=True)
    pkl["pct_rank"] = pct.reindex(pkl.index)
    pkl["net"] = (1 + pkl[label]) * (1 - sell_cost) / (1 + buy_cost) - 1

    return {
        "model": model,
        "label": label,
        "n_picks": int(len(pkl)),
        "n_dates": int(pkl["date"].nunique()),
        # 预测精度 (自身目标, pool 级)
        "forecast_bias": bias,
        "forecast_mae": mae,
        "forecast_rmse": rmse,
        # 概率精度 (实际持有期, 入选股)
        "hit_rate": float((pkl[label] > 0).mean()),
        "base_rate": float(pkl["base_rate"].mean()),
        "hit_lift": float((pkl[label] > 0).mean() - pkl["base_rate"].mean()),
        "beat_pool_med": float((pkl[label] > pkl["pool_med"]).mean()),
        # 排名精度 (实际持有期)
        "rank_ic": _daily_spearman(sub_l, "pred", label, cfg),
        "avg_pct_rank": float(pkl["pct_rank"].mean()),
        "excess_vs_pool": float((pkl[label] - pkl["pool_mean"]).mean()),
        # 成本后回测
        "pool_week": float(pkl["pool_mean"].mean()),
        "gross_week": float(pkl[label].mean()),
        "net_week": float(pkl["net"].mean()),
        "net_excess": float((pkl["net"] - pkl["pool_mean"]).mean()),
        "week_win_rate": float((pkl.groupby("date")["net"].mean() > 0).mean()),
    }


def main():
    cfg = LHB_V2_EVAL
    top10 = LHB_V2_TOP10

    df = load_panel()
    df = add_targets(df)
    df = run_dim34(df)

    pool = df[df["listed"]].copy()
    if top10.get("exclude_st", True):
        pool = pool[~pool["is_st"].fillna(False)]
    dates = sorted(pool["date"].unique())
    cutoff = dates[int(len(dates) * cfg["split_ratio"])]
    tr = pool[pool["date"] < cutoff].copy()
    te = pool[pool["date"] >= cutoff].copy()
    logger.info(
        "切分: 训练 %s ~ %s, 测试 %s ~ %s",
        tr["date"].min().strftime("%Y%m%d"),
        cutoff.strftime("%Y%m%d"),
        cutoff.strftime("%Y%m%d"),
        te["date"].max().strftime("%Y%m%d"),
    )

    av = [f for f in FEATURES if f in te.columns]
    te_clean = te.dropna(subset=av).copy()

    rows = []
    all_picks = []
    for target in ("r1", "r5"):
        te_pred, _ = train_predict(tr, te_clean, av, target, cfg)
        picks = pick_top10(te_pred, "pred", top10["top_n"])
        picks["model"] = target
        all_picks.append(picks)
        for label in ("r3", "r5"):
            res = evaluate(picks, te_pred, cfg, top10, target, label, target)
            rows.append(res)
            logger.info(
                "模型 %s | 持有 %s: 命中率 %.1f%% (池 %.1f%%), 净周收益 %.2f%% "
                "(池 %.2f%%), 实现分位 %.3f, rankIC %s",
                target,
                label,
                100 * res["hit_rate"],
                100 * res["base_rate"],
                100 * res["net_week"],
                100 * res["pool_week"],
                res["avg_pct_rank"],
                f"{res['rank_ic']:.4f}" if res["rank_ic"] is not None else "NA",
            )

    summary = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx_path = DATA_OTHERS_DIR / f"lhb_v2_top10_{ts}.xlsx"
    pick_path = DATA_OTHERS_DIR / f"lhb_v2_top10_picks_{ts}.parquet"
    summary.to_excel(str(xlsx_path), index=False)
    logger.info("写回: %s", xlsx_path)
    pd.concat(all_picks, ignore_index=True).to_parquet(pick_path, index=False)
    logger.info("写回: %s", pick_path)

    print("\n──── LHB2 TOP10 选股能力评估 (2 模型 × 2 持有期) ────")
    cols = [
        "model",
        "label",
        "n_picks",
        "n_dates",
        "hit_rate",
        "base_rate",
        "hit_lift",
        "beat_pool_med",
        "avg_pct_rank",
        "rank_ic",
        "excess_vs_pool",
        "net_week",
        "net_excess",
        "week_win_rate",
        "forecast_bias",
        "forecast_mae",
        "forecast_rmse",
    ]
    print(summary[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
