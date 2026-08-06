# -*- coding: utf-8 -*-
"""KIMI LHB v2.0 训练/预测/评估 — 仅龙虎榜上榜股票池 (spec §5.3).

工作流:
  1. 加载 V3 面板 (席位列 lhb_*_buy/sell 已由 _backfill_lhb_seats_v2.py 回填).
  2. 运行 dim34_lhb_v2 → 14 个 lhb2_* 特征 (EWMA 记忆, PIT ≤t).
  3. 上榜池 = 当日任一 lhb 基础列非空的行 (与 dim34 的 I(List) 同定义).
  4. 标签 (spec §6.1 T+1 开盘买入, hfq 价):
       r1 = close_{t+1}/open_{t+1} − 1
       r3 = close_{t+3}/open_{t+1} − 1
       r5 = close_{t+5}/open_{t+1} − 1
  5. 时间切分: 前 80% 上榜日训练, 后 20% 评估 (仅在榜池内).
  6. 评估: 单特征 IC / LGBM rank IC·ICIR·t / 多空收益差 / 特征重要性.

输出 (WORM, DATA_OTHERS + 日期后缀):
  lhb_v2_eval_<ts>.xlsx       3 sheets: 模型摘要 / 特征 IC 表 / 特征重要性
  lhb_v2_preds_<ts>.parquet   测试期 (symbol, date, pred, r1, r3, r5)

用法:
  python scripts/lhb_v2_train_eval.py
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline1.feature_engine_v35 import FeatureEngineV35 as FE  # noqa: E402
from config.settings import BACKTEST_RESULT_DIR, LHB_V2_EVAL, PANEL_V3_PATH  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("lhb_v2_train_eval")

FEATURES = [
    "lhb2_inst_flow",
    "lhb2_inst_shock",
    "lhb2_top_flow",
    "lhb2_quant_flow",
    "lhb2_retail_flow",
    "lhb2_sell_pressure",
    "lhb2_sell_buy_ratio",
    "lhb2_list_count_5d",
    "lhb2_conboard_mem",
    "lhb2_inst_strength",
    "lhb2_inst_resolve",
    "lhb2_inst_conboard",
    "lhb2_inst_premium",
    "lhb2_inst_lock",
]
SEAT_COLS = [
    "lhb_inst_buy",
    "lhb_inst_sell",
    "lhb_top_buy",
    "lhb_top_sell",
    "lhb_quant_buy",
    "lhb_quant_sell",
    "lhb_retail_buy",
    "lhb_retail_sell",
]
BASE_LHB = ["lhb_buy_amt", "lhb_sell_amt", "lhb_net_buy"]

LOAD_COLS = (
    [
        "date",
        "symbol",
        "open",
        "close",
        "high",
        "low",
        "open_hfq",
        "close_hfq",
        "up_limit_raw",
        "down_limit_raw",
        "circ_mv",
    ]
    + SEAT_COLS
    + BASE_LHB
)

# hfq 后复权重定基伪跳检测 (标签作废依据):
#   HFQ_GLITCH_FACTOR       单股孤立因子巨跳阈值. 真除权 ≤~5x (10送30 上限);
#                           数据商重定基可达 544x (600602). 超过判定为伪跳.
#   HFQ_REBASE_FAC_CHG      计数用因子跳变阈值 (真除权现金分红即可 >1.02).
#   HFQ_REBASE_MIN_STOCKS   单日发生因子跳变的股票数 ≥ 此值 → 该日为数据商重定基日
#                           (真除权日最多百余只; 2026-07-27 重定基日 2831 只).
#                           重定基后 hfq 不在同一因子基, 跨它的标签收益是伪值.
HFQ_GLITCH_FACTOR = 20.0
HFQ_REBASE_FAC_CHG = 1.02
HFQ_REBASE_MIN_STOCKS = 500


def load_panel() -> pd.DataFrame:
    logger.info("加载面板: %s", PANEL_V3_PATH)
    df = pd.read_parquet(PANEL_V3_PATH, columns=LOAD_COLS)
    missing = [c for c in LOAD_COLS if c not in df.columns]
    if missing:
        logger.error("面板缺少列: %s (需先跑 _backfill_lhb_seats_v2.py)", missing)
        raise SystemExit(1)
    logger.info(
        "面板: %d 行, %d symbols, %d 列, %s ~ %s",
        len(df),
        df["symbol"].nunique(),
        len(df.columns),
        df["date"].min().strftime("%Y%m%d"),
        df["date"].max().strftime("%Y%m%d"),
    )
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """上榜示性 + T+1 开盘买入标签 (hfq, 对齐 symbol 未来行)."""
    df["listed"] = df[BASE_LHB].notna().any(axis=1)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", group_keys=False)
    o1 = g["open_hfq"].shift(-1)
    for h in (1, 3, 5):
        df[f"r{h}"] = g["close_hfq"].shift(-h) / o1 - 1

    # hfq 后复权重定基伪跳 → 跨过它的标签作废. 数据商重定基在单日让大量股票同时
    # 发生因子跳变 (2026-07-27 有 2831 只, 真除权日最多百余只), 重定基后 hfq 不在
    # 同一因子基, 收益是伪值; 单股孤立巨跳 (>HFQ_GLITCH_FACTOR) 也判定伪跳.
    # 真除权日 hfq 价连续 (总收益), 不误伤. 窗口 t+2..t+h 内任一行为伪跳行 →
    # r{h} 作废; r1 为同日 open→close, 开收盘同因子, 天然不受污染; 伪跳在入口行
    # t+1 则开收盘同因子, 收益一致, 不误伤.
    fac = df["close_hfq"] / df["close"].replace(0, np.nan)
    fac_chg = fac / fac.groupby(df["symbol"]).shift(1)
    jump_cnt = (
        ((fac_chg > HFQ_REBASE_FAC_CHG) | (fac_chg < 1 / HFQ_REBASE_FAC_CHG))
        .groupby(df["date"])
        .sum()
    )
    rebase_day = jump_cnt[jump_cnt >= HFQ_REBASE_MIN_STOCKS].index
    glitch = df["date"].isin(rebase_day) | (
        (fac_chg > HFQ_GLITCH_FACTOR) | (fac_chg < 1 / HFQ_GLITCH_FACTOR)
    )
    cgt = glitch.groupby(df["symbol"]).cumsum()
    for h in (3, 5):
        # 符号内窗口 t+2..t+h 的伪跳数 = cum_glitch[t+h] − cum_glitch[t+1] > 0
        bad = (
            cgt.groupby(df["symbol"]).shift(-h) - cgt.groupby(df["symbol"]).shift(-1)
        ) > 0
        df[f"r{h}"] = df[f"r{h}"].mask(bad)
    return df


def run_dim34(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("运行 dim34_lhb_v2 (14 特征)...")
    df = FE.dim34_lhb_v2(df)
    return df


def feature_ic(sub: pd.DataFrame, labels: list[str], cfg: dict) -> pd.DataFrame:
    """单特征日频 spearman IC (仅上榜池)."""
    rows = []
    for f in FEATURES:
        if f not in sub.columns:
            continue
        for lab in labels:
            ics = []
            for _, g in sub.groupby("date"):
                v = g[[f, lab]].dropna()
                if len(v) < cfg["min_ic_n"]:
                    continue
                r, _ = spearmanr(v[f], v[lab])
                if not np.isnan(r):
                    ics.append(r)
            nn = len(ics)
            if nn < cfg["min_ic_obs"]:
                rows.append(
                    {
                        "feature": f,
                        "label": lab,
                        "ic": np.nan,
                        "icir": np.nan,
                        "t": np.nan,
                        "n_dates": nn,
                    }
                )
                continue
            a = np.array(ics)
            m, s = a.mean(), a.std(ddof=1)
            rows.append(
                {
                    "feature": f,
                    "label": lab,
                    "ic": round(m, 5),
                    "icir": round(m / s, 4) if s > 0 else 0.0,
                    "t": round(m / (s / np.sqrt(nn)), 2) if s > 0 else 0.0,
                    "n_dates": nn,
                }
            )
    return pd.DataFrame(rows)


def rank_ic_by_date(te: pd.DataFrame, label: str, cfg: dict) -> tuple:
    ics = []
    for _, g in te.groupby("date"):
        v = g[["pred", label]].dropna()
        if len(v) < cfg["min_ic_n"]:
            continue
        r, _ = spearmanr(v["pred"], v[label])
        if not np.isnan(r):
            ics.append(r)
    if len(ics) < cfg["min_ic_obs"]:
        return np.nan, np.nan, np.nan, len(ics)
    a = np.array(ics)
    m, s = a.mean(), a.std(ddof=1)
    return (
        m,
        (m / s if s > 0 else 0.0),
        (m / (s / np.sqrt(len(a))) if s > 0 else 0.0),
        len(a),
    )


def long_short(te: pd.DataFrame, label: str, cfg: dict) -> tuple:
    """每日期内按 pred 分位, top/bottom 20% 多空均值差 (跨日期等权平均)."""
    spreads = []
    for _, g in te.groupby("date"):
        v = g[["pred", label]].dropna()
        if len(v) < 2 * cfg["min_ic_n"]:
            continue
        q_hi = v["pred"].quantile(1 - cfg["quantile"])
        q_lo = v["pred"].quantile(cfg["quantile"])
        long = v.loc[v["pred"] >= q_hi, label].mean()
        short = v.loc[v["pred"] <= q_lo, label].mean()
        spreads.append(long - short)
    if not spreads:
        return np.nan, 0
    return float(np.mean(spreads)), len(spreads)


def run_model(tr: pd.DataFrame, te: pd.DataFrame, labels: list[str], cfg: dict) -> dict:
    av = [f for f in FEATURES if f in tr.columns]
    tr = tr.dropna(subset=av + ["r1"])
    te = te.dropna(subset=av + ["r1"])
    logger.info(
        "样本: 训练 %d 行 (%d 日), 测试 %d 行 (%d 日)",
        len(tr),
        tr["date"].nunique(),
        len(te),
        te["date"].nunique(),
    )
    if len(tr) < 100 or len(te) < 50:
        logger.error("样本过少, 中止")
        raise SystemExit(1)

    m = lgb.LGBMRegressor(verbose=-1, **cfg["lgb"])
    m.fit(tr[av], tr["r1"])
    imp = pd.DataFrame(
        {
            "feature": av,
            "gain": m.booster_.feature_importance(importance_type="gain"),
        }
    )
    imp["gain_pct"] = (imp["gain"] / imp["gain"].sum() * 100).round(2)
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)

    te = te.copy()
    te["pred"] = m.predict(te[av])

    res = {"importance": imp, "test": te, "model": m}
    for lab in labels:
        ric, ricir, t, nn = rank_ic_by_date(te, lab, cfg)
        spread, n_days = long_short(te, lab, cfg)
        res[f"rank_ic_{lab}"] = ric
        res[f"icir_{lab}"] = ricir
        res[f"t_{lab}"] = t
        res[f"n_{lab}"] = nn
        res[f"ls_{lab}"] = spread
        res[f"ls_days_{lab}"] = n_days
        logger.info(
            "LGBM pred vs %s: rankIC=%.4f ICIR=%.4f t=%.2f (n=%d) LS=%.4f%% (n=%d)",
            lab,
            ric,
            ricir,
            t,
            nn,
            100 * spread,
            n_days,
        )
    return res


def main():
    cfg = LHB_V2_EVAL
    labels = [f"r{h}" for h in cfg["horizons"]]

    df = load_panel()
    df = add_targets(df)
    df = run_dim34(df)

    pool = df[df["listed"]].copy()
    logger.info(
        "上榜池: %d 行, %d symbols, %d 个上榜日",
        len(pool),
        pool["symbol"].nunique(),
        pool["date"].nunique(),
    )

    # 时间切分 (按上榜日)
    dates = sorted(pool["date"].unique())
    cutoff = dates[int(len(dates) * cfg["split_ratio"])]
    tr = pool[pool["date"] < cutoff].copy()
    te = pool[pool["date"] >= cutoff].copy()
    logger.info(
        "切分: 训练 %s ~ %s (%d 日), 测试 %s ~ %s (%d 日)",
        dates[0].strftime("%Y%m%d"),
        cutoff.strftime("%Y%m%d"),
        tr["date"].nunique(),
        cutoff.strftime("%Y%m%d"),
        dates[-1].strftime("%Y%m%d"),
        te["date"].nunique(),
    )

    ic_tbl = feature_ic(te, labels, cfg)
    logger.info(
        "单特征 IC (测试期, %d 特征 × %d 标签):",
        ic_tbl["feature"].nunique(),
        ic_tbl["label"].nunique(),
    )

    res = run_model(tr, te, labels, cfg)

    # ── 摘要 ──
    rows = []
    for lab in labels:
        rows.append(
            {
                "label": lab,
                "rank_ic": round(res[f"rank_ic_{lab}"], 5)
                if not pd.isna(res[f"rank_ic_{lab}"])
                else np.nan,
                "icir": round(res[f"icir_{lab}"], 4)
                if not pd.isna(res[f"icir_{lab}"])
                else np.nan,
                "t": round(res[f"t_{lab}"], 2)
                if not pd.isna(res[f"t_{lab}"])
                else np.nan,
                "n_dates": res[f"n_{lab}"],
                "long_short_pct": round(100 * res[f"ls_{lab}"], 4)
                if not pd.isna(res[f"ls_{lab}"])
                else np.nan,
                "ls_dates": res[f"ls_days_{lab}"],
            }
        )
    summary = pd.DataFrame(rows)
    n_tr = len(res["test"])
    summary.insert(0, "n_test", n_tr)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BACKTEST_RESULT_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = run_dir / f"lhb_v2_eval_{ts}.xlsx"
    pred_path = run_dir / f"lhb_v2_preds_{ts}.parquet"
    imp = res["importance"]

    pred_out = res["test"][["date", "symbol", "pred"] + labels]
    pred_out.to_parquet(pred_path, index=False)

    # 汇总写入 xlsx 多 sheet (模型摘要 + 特征 IC 表 + 特征重要性)
    with pd.ExcelWriter(str(xlsx_path), engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="model_summary", index=False)
        ic_tbl.to_excel(writer, sheet_name="feature_ic", index=False)
        imp.to_excel(writer, sheet_name="importance", index=False)
    logger.info("写回: %s", xlsx_path)
    logger.info("写回: %s", pred_path)

    logger.info("──── 模型摘要 ────")
    print(summary.to_string(index=False))
    logger.info("──── 特征重要性 (top 8) ────")
    print(imp.head(8).to_string(index=False))
    logger.info("──── 特征 IC (r1, top 8) ────")
    top_ic = ic_tbl[ic_tbl["label"] == "r1"].sort_values("ic", key=abs, ascending=False)
    print(top_ic.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
