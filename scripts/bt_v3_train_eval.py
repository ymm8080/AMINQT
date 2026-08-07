"""大宗交易 v3 (dim33) 训练/评估 — 仅大宗事件相关股票池 (用户定案 2026-08-03).

与 lhb_v2_train_eval.py 同协议 (spec §5.3 选择性偏差: 仅上榜池) — 稀疏事件特征
在全市场面板上评估会被 98.8% 的零值行稀释, 因此在"相关数据集"(大宗事件池)内
训练与评估. 事件池 = 当日任一 bt_* 原始列非空的行 (与 dim33 的上游同定义).

工作流:
  1. 加载 V3 面板 (bt_count/bt_disc_raw/bt_inst_absorb/bt_amt_ratio_float_mv + OHLCV).
  2. 运行 dim33_block_trade → 4 个 bt_*_ewma 特征 (EWMA 记忆, PIT ≤t).
  3. 事件池 = 4 个 bt_ 原始列任一非空的行 (与上游同定义).
  4. 标签 (同 LHB v2 §6.1 T+1 开盘买入, hfq 价):
       r1 = close_{t+1}/open_{t+1} − 1
       r3 = close_{t+3}/open_{t+1} − 1
       r5 = close_{t+5}/open_{t+1} − 1
  5. 时间切分: 前 80% 事件日训练, 后 20% 评估 (仅在事件池内).
  6. 评估: 单特征 IC / LGBM rank IC·ICIR·t / 多空收益差 / 特征重要性.
  7. 对照: 测试期全市场 vs 事件池单特征 IC (暴露"事件池是否浓缩信号"的证据).

输出 (WORM, DATA_OTHERS + 日期后缀):
  bt_v3_eval_<ts>.xlsx   4 sheets: 模型摘要 / 特征 IC 表 / 全市场vs事件池 IC / 特征重要性
  bt_v3_preds_<ts>.parquet   测试期 (symbol, date, pred, r1, r3, r5)

用法:
  python scripts/bt_v3_train_eval.py
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
from config.settings import BACKTEST_RESULT_DIR, BT_V3_EVAL, PANEL_V3_PATH  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("bt_v3_train_eval")

# 4 个 dim33 EWMA 特征 ← 4 个上游原始列 (V3 面板只保留这 4 列, 2026-08-03 精简定案)
FEATURES = ["bt_act_ewma", "bt_disc_ewma", "bt_inst_abs_ewma", "bt_mv_ratio_ewma"]
BT_COLS = ["bt_count", "bt_disc_raw", "bt_inst_absorb", "bt_amt_ratio_float_mv"]

LOAD_COLS = [
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
] + BT_COLS

# hfq 后复权重定基伪跳检测 (标签作废依据) — 与 lhb_v2_train_eval.py 相同:
#   数据商重定基在单日让大量股票同时发生因子跳变 (2026-07-27 有 2831 只),
#   重定基后 hfq 不在同一因子基, 跨它的标签收益是伪值; 单股孤立巨跳也判伪跳.
HFQ_GLITCH_FACTOR = 20.0
HFQ_REBASE_FAC_CHG = 1.02
HFQ_REBASE_MIN_STOCKS = 500


def load_panel() -> pd.DataFrame:
    logger.info("加载面板: %s", PANEL_V3_PATH)
    df = pd.read_parquet(PANEL_V3_PATH, columns=LOAD_COLS)
    missing = [c for c in LOAD_COLS if c not in df.columns]
    if missing:
        logger.error("面板缺少列: %s (需先回填 4 个 bt_* 原始列)", missing)
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
    """T+1 开盘买入标签 (hfq, 对齐 symbol 未来行)."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", group_keys=False)
    o1 = g["open_hfq"].shift(-1)
    for h in (1, 3, 5):
        df[f"r{h}"] = g["close_hfq"].shift(-h) / o1 - 1

    # hfq 后复权重定基伪跳 → 跨过它的标签作废 (同 lhb_v2_train_eval.py).
    # 窗口 t+2..t+h 内任一行为伪跳行 → r{h} 作废; r1 为同日 open→close 天然免疫.
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
        bad = (
            cgt.groupby(df["symbol"]).shift(-h) - cgt.groupby(df["symbol"]).shift(-1)
        ) > 0
        df[f"r{h}"] = df[f"r{h}"].mask(bad)
    return df


def run_dim33(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("运行 dim33_block_trade (4 EWMA 特征)...")
    return FE.dim33_block_trade(df)


def feature_ic(sub: pd.DataFrame, labels: list[str], cfg: dict) -> pd.DataFrame:
    """单特征日频 spearman IC (仅事件池)."""
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


def full_vs_pool_ic(
    te_full: pd.DataFrame, te_pool: pd.DataFrame, labels: list[str], cfg: dict
) -> pd.DataFrame:
    """测试期 全市场 vs 事件池 单特征日度 mean rank IC 对照.

    稀疏特征在事件池内是否浓缩信号, 由该对照直接暴露 (实测: 全市场 |IC| 反而略大,
    事件池并未浓缩 — 见运行日志). 供未来 selection 决策参考.
    """
    rows = []
    for f in FEATURES:
        for lab in labels:
            for tag, sub in (("full", te_full), ("pool", te_pool)):
                ics = []
                for _, g in sub.groupby("date"):
                    v = g[[f, lab]].dropna()
                    if len(v) < cfg["min_ic_n"]:
                        continue
                    r, _ = spearmanr(v[f], v[lab])
                    if not np.isnan(r):
                        ics.append(r)
                nn = len(ics)
                rows.append(
                    {
                        "feature": f,
                        "label": lab,
                        "scope": tag,
                        "ic": round(float(np.mean(ics)), 5)
                        if nn >= cfg["min_ic_obs"]
                        else np.nan,
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
        "样本: 训练 %d 行 (%d 事件日), 测试 %d 行 (%d 事件日)",
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
    cfg = BT_V3_EVAL
    labels = [f"r{h}" for h in cfg["horizons"]]

    df = load_panel()
    df = add_targets(df)
    df = run_dim33(df)

    is_event = df[BT_COLS].notna().any(axis=1)
    pool = df[is_event].copy()
    logger.info(
        "大宗事件池: %d 行, %d symbols, %d 个事件日",
        len(pool),
        pool["symbol"].nunique(),
        pool["date"].nunique(),
    )

    # 时间切分 (按事件日)
    dates = sorted(pool["date"].unique())
    cutoff = dates[int(len(dates) * cfg["split_ratio"])]
    tr = pool[pool["date"] < cutoff].copy()
    te_pool = pool[pool["date"] >= cutoff].copy()
    te_full = df[df["date"] >= cutoff].copy()  # 全市场对照 (仅测试期)
    logger.info(
        "切分: 训练 %s ~ %s (%d 事件日), 测试 %s ~ %s (%d 事件日)",
        dates[0].strftime("%Y%m%d"),
        cutoff.strftime("%Y%m%d"),
        tr["date"].nunique(),
        cutoff.strftime("%Y%m%d"),
        dates[-1].strftime("%Y%m%d"),
        te_pool["date"].nunique(),
    )

    ic_tbl = feature_ic(te_pool, labels, cfg)
    ic_contrast = full_vs_pool_ic(te_full, te_pool, labels, cfg)
    logger.info(
        "单特征 IC (测试期事件池, %d 特征 × %d 标签):",
        ic_tbl["feature"].nunique(),
        ic_tbl["label"].nunique(),
    )

    res = run_model(tr, te_pool, labels, cfg)

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
    xlsx_path = run_dir / f"bt_v3_eval_{ts}.xlsx"
    pred_path = run_dir / f"bt_v3_preds_{ts}.parquet"
    imp = res["importance"]

    pred_out = res["test"][["date", "symbol", "pred"] + labels]
    pred_out.to_parquet(pred_path, index=False)

    with pd.ExcelWriter(str(xlsx_path), engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="model_summary", index=False)
        ic_tbl.to_excel(writer, sheet_name="feature_ic", index=False)
        ic_contrast.to_excel(writer, sheet_name="ic_full_vs_pool", index=False)
        imp.to_excel(writer, sheet_name="importance", index=False)
    logger.info("写回: %s", xlsx_path)
    logger.info("写回: %s", pred_path)

    logger.info("──── 模型摘要 ────")
    print(summary.to_string(index=False))
    logger.info("──── 特征重要性 ────")
    print(imp.to_string(index=False))
    logger.info("──── 特征 IC (r1, 事件池, top 8) ────")
    top_ic = ic_tbl[ic_tbl["label"] == "r1"].sort_values("ic", key=abs, ascending=False)
    print(top_ic.head(8).to_string(index=False))
    logger.info("──── 全市场 vs 事件池 IC 对照 (r1) ────")
    print(
        ic_contrast[ic_contrast["label"] == "r1"]
        .pivot(index="feature", columns="scope", values="ic")
        .reindex(FEATURES)
        .to_string()
    )


if __name__ == "__main__":
    main()
