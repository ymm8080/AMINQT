# -*- coding: utf-8 -*-
"""A/B: top-10 剔除 vs 不剔除近 N 日 LHB — 按用户目标口径 (5 日净收益) 决策验证.

背景: 用户目标 = 每日多头选出 5 日内最值得买的 top-10. 生产模型不吃 LHB 特征
(main=76 CYQ + dual=30 板块动量, 均 0 LHB 特征); LHB 上榜池 3-5 日反转
(r3 rankIC -0.025, r5 -0.038). 用户设计: 近期有 LHB 的股票不进预测 (规则剔除,
不进模型). 本脚本验证该剔除是否真的提升 top-10 净收益.

方法 (与生产 daily_pipeline 同链路):
  panel(截断最近 LOOKBACK+TEST 交易日) → cleaner.run_inference →
  FE.build(inference_cols=bundle feature_cols, dual 开 cross_sectional_rank) →
  每测试日横截面 pred_ret_5d (bundle["models"]["5d_reg"][0]) →
  两臂按 pred_ret_5d 取 top-10:
    Arm A (基线): 全候选池.
    Arm B (剔除): 先剔除近 N 交易日有 LHB 上榜的股票 (lhb_buy_amt/sell/net 任一
                 非空即上榜; 上榜当日晚间公布, 次日开盘可行动, 窗口含当日).
  实现收益 r5 = close_hfq_{t+5}/open_hfq_{t+1} − 1 (T+1 开盘买入, T+5 收盘卖出),
  含成本 (buy=万2.5+滑点0.10%, sell=+印花0.05%), hfq 重定基伪跳窗口作废
  (掩码逻辑复刻 lhb_v2_train_eval.add_targets, 2026-07-27 重定基).

冒烟输出: 每板块模型特征列有多少可在当前 FE 复现 (缺列 → 补 0, 与生产 predict 一致).
输出 (WORM): DATA_OTHERS/lhb_ab_exclude_<ts>.xlsx + lhb_ab_picks_<ts>.parquet

用法: python scripts/ab_lhb_exclude_top10.py
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline1.cleaning_pipeline import CleaningPipeline  # noqa: E402
from app.pipeline1.dual_track_trainer import DualTrackTrainer  # noqa: E402
from app.pipeline1.feature_engine_v35 import FeatureEngineV35  # noqa: E402
from app.pipeline1.predict_runner import find_bundles  # noqa: E402
from config.settings import DATA_OTHERS_DIR, PANEL_V3_PATH  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("ab_lhb_exclude")

TOP_N = 10
TEST_DAYS = 120
LOOKBACK_DAYS = 300
LHB_WINDOWS = (3, 5, 10)
MODEL_DIR = "models/pipeline1"

# 成本 (与 LHB_V2_TOP10 一致)
BUY_COST = 0.00025 + 0.0010  # 佣金万2.5 + 滑点0.10%
SELL_COST = 0.00025 + 0.0005 + 0.0010  # 佣金 + 印花0.05% + 滑点

# hfq 重定基伪跳阈值 (复刻 lhb_v2_train_eval.py 常量)
HFQ_GLITCH_FACTOR = 20.0
HFQ_REBASE_FAC_CHG = 1.02
HFQ_REBASE_MIN_STOCKS = 500

BASE_LHB = ["lhb_buy_amt", "lhb_sell_amt", "lhb_net_buy"]


def add_r5_labels(df: pd.DataFrame) -> pd.DataFrame:
    """r5 = close_hfq_{t+5}/open_hfq_{t+1} − 1 (T+1 开盘买入), hfq 伪跳窗口作废."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", group_keys=False)
    df["r5"] = g["close_hfq"].shift(-5) / g["open_hfq"].shift(-1) - 1

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
    bad = (
        cgt.groupby(df["symbol"]).shift(-5) - cgt.groupby(df["symbol"]).shift(-1)
    ) > 0
    df["r5"] = df["r5"].mask(bad)
    return df


def load_panel_window() -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """读面板最近 LOOKBACK+TEST 交易日 (labels 需延后至面板末). 返回 (面板, 交易日历)."""
    dates = pd.read_parquet(PANEL_V3_PATH, columns=["date"])["date"].sort_values()
    dates = pd.Index(dates.unique())
    n = min(len(dates), LOOKBACK_DAYS + TEST_DAYS + 5)
    start = dates[-n]
    logger.info("面板窗口: %s ~ %s (保留 %d 交易日)", start.date(), dates[-1].date(), n)
    df = pd.read_parquet(PANEL_V3_PATH, filters=[("date", ">=", start)])
    logger.info("面板: %d 行, %d symbols", len(df), df["symbol"].nunique())
    return df, list(dates)


def main() -> None:
    bundles = find_bundles(model_dir=MODEL_DIR)
    if not bundles:
        logger.error("无可用模型包")
        raise SystemExit(1)
    logger.info("模型包: %s", bundles)

    panel, calendar = load_panel_window()
    missing_lhb = [c for c in BASE_LHB if c not in panel.columns]
    if missing_lhb:
        logger.error("面板缺 LHB 基础列: %s", missing_lhb)
        raise SystemExit(1)

    panel = add_r5_labels(panel)

    # ── 生产同链路清洗 ──
    cleaner = CleaningPipeline()
    main_df, dual_df, valve = cleaner.run_inference(panel)
    logger.info("清洗: main=%d, dual=%d, valve=%s", len(main_df), len(dual_df), valve)

    # ── 特征构建 (冒烟: 统计缺列) ──
    feats: dict[str, pd.DataFrame] = {}
    for board, df in (("main", main_df), ("dual", dual_df)):
        if board not in bundles or len(df) == 0:
            logger.warning("[%s] 无模型或无数据, 跳过", board)
            continue
        path = bundles[board]
        bundle = DualTrackTrainer.load(path)
        cols = bundle["feature_cols"]
        f = FeatureEngineV35().build(
            df,
            None,
            inference_cols=cols,
            cross_sectional_rank=(board == "dual"),
        )
        miss = [c for c in cols if c not in f.columns]
        for c in miss:
            f[c] = 0.0
        logger.info(
            "[%s] 特征: 复现 %d/%d, 缺列补0: %d (%s)",
            board,
            len(cols) - len(miss),
            len(cols),
            len(miss),
            miss[:5],
        )
        if miss:
            print(f"[SMOKE] {board} 缺列 → 补0: {len(miss)}: {miss[:10]}")
        else:
            print(f"[SMOKE] {board} 全部 {len(cols)} 个模型特征列可复现 ✔")
        f = f.sort_values(["symbol", "date"]).reset_index(drop=True)
        feats[board] = f

    if not feats:
        raise SystemExit(1)

    # ── 每测试日横截面预测 ──
    cal_pos = {d: i for i, d in enumerate(calendar)}

    # 有效 r5 计数 → 测试日 (自动排除 07-20+ 伪跳窗口与面板末无未来行)
    r5_ok = panel[panel["r5"].notna()]
    valid_count = r5_ok.groupby("date").size()
    valid_dates = [d for d in calendar if valid_count.get(d, 0) >= 30]
    test_dates = valid_dates[-TEST_DAYS:]
    logger.info(
        "测试期: %s ~ %s (%d 个交易日, 每日均 ≥30 只有效 r5)",
        test_dates[0].strftime("%Y%m%d"),
        test_dates[-1].strftime("%Y%m%d"),
        len(test_dates),
    )

    pred_rows = []
    for board, f in feats.items():
        bundle = DualTrackTrainer.load(bundles[board])
        cols = bundle["feature_cols"]
        m5 = bundle["models"]["5d_reg"][0]
        sub = f[f["date"].isin(test_dates)].copy()
        sub["pred_ret_5d"] = m5.predict(
            np.nan_to_num(sub[cols].to_numpy(dtype=float), nan=0.0)
        )
        pred_rows.append(sub[["symbol", "date", "board", "pred_ret_5d"]])
    pred = pd.concat(pred_rows, ignore_index=True)

    pred = pred.merge(
        panel[["symbol", "date", "r5"]], on=["symbol", "date"], how="left"
    )
    # 候选池均值基准 (该日全部可买股票的实现 r5, 成本前)
    pool_mean = pred.groupby("date")["r5"].mean().rename("pool_r5")
    pred = pred.merge(pool_mean, left_on="date", right_index=True, how="left")

    # ── LHB 近 N 日上榜掩码 (日历位置窗口) ──
    evt = panel[BASE_LHB].notna().any(axis=1)
    sym_pos = panel["date"].map(cal_pos)
    M = (
        pd.DataFrame({"symbol": panel["symbol"], "pos": sym_pos, "evt": evt})
        .groupby(["symbol", "pos"])["evt"]
        .max()
        .unstack(fill_value=False)
    )
    cum = M.cumsum(axis=1)
    recent_mask = {w: cum.sub(cum.shift(w, axis=1).fillna(0)) > 0 for w in LHB_WINDOWS}

    net = lambda r: (1 + r) * (1 - SELL_COST) / (1 + BUY_COST) - 1  # noqa: E731

    def daily_mean(s: pd.Series, date_key: pd.Series) -> float:
        return float(s.groupby(date_key).mean().mean())

    rows, pick_list = [], []
    for w in LHB_WINDOWS:
        excl = recent_mask[w]
        for arm, use_excl in (("A_noexcl", False), ("B_excl", True)):
            picks = []
            for d in test_dates:
                p = cal_pos[d]
                sub = pred[pred["date"] == d].copy()
                if use_excl:
                    banned = set(excl.index[excl[p]])
                    sub = sub[~sub["symbol"].isin(banned)]
                top = sub.sort_values("pred_ret_5d", ascending=False).head(TOP_N).copy()
                top["arm"] = arm
                top["w"] = w
                picks.append(top)
            picks = pd.concat(picks, ignore_index=True)
            ok = picks[picks["r5"].notna()].copy()
            ok["net"] = net(ok["r5"])
            daily = ok.groupby("date").agg(d_net=("net", "mean"), d_r5=("r5", "mean"))
            daily = daily.merge(
                ok[["date", "pool_r5"]].drop_duplicates(), on="date", how="left"
            )
            daily["pool_net"] = net(daily["pool_r5"])
            rows.append(
                {
                    "w": w,
                    "arm": arm,
                    "n_dates": ok["date"].nunique(),
                    "n_picks": len(ok),
                    "n_no_r5": len(picks) - len(ok),
                    "mean_net": ok["net"].mean(),
                    "daily_net": daily["d_net"].mean(),
                    "win_rate": float((ok["net"] > 0).mean()),
                    "hit_r5": float((ok["r5"] > 0).mean()),
                    "daily_pool_net": daily["pool_net"].mean(),
                    "daily_excess": float((daily["d_net"] - daily["pool_net"]).mean()),
                    "mean_pred": ok["pred_ret_5d"].mean(),
                }
            )
            pick_list.append(ok)

        # Arm A 实际选中的近期-LHB 股 (剔除的直接损失): 实现净收益 + 与 B 的清单差异
        a = pd.concat(pick_list, ignore_index=True)
        a = a[(a["arm"] == "A_noexcl") & (a["w"] == w)].copy()
        a["is_lhb"] = False
        for d in test_dates:
            p = cal_pos[d]
            banned = set(excl.index[excl[p]])
            sel = a["date"] == d
            a.loc[sel, "is_lhb"] = a.loc[sel, "symbol"].isin(banned)
        a_lhb = a[a["is_lhb"] & a["r5"].notna()].copy()
        a_lhb["net"] = net(a_lhb["r5"])
        b = pd.concat(pick_list, ignore_index=True)
        b = b[(b["arm"] == "B_excl") & (b["w"] == w)]
        # 逐日对比两臂 top-10 符号集合
        a_sym = a.groupby("date")["symbol"].apply(set)
        b_sym = b.groupby("date")["symbol"].apply(set)
        diff_days = sum(bool(a_sym[d] != b_sym[d]) for d in a_sym.index)
        rows.append(
            {
                "w": w,
                "arm": "A_LHB_kicked",
                "n_dates": a_lhb["date"].nunique(),
                "n_picks": len(a_lhb),
                "mean_net": a_lhb["net"].mean(),
                "daily_net": daily_mean(a_lhb["net"], a_lhb["date"]),
                "win_rate": float((a_lhb["net"] > 0).mean()),
                "hit_r5": float((a_lhb["r5"] > 0).mean()),
                "daily_pool_net": np.nan,
                "daily_excess": np.nan,
                "mean_pred": a_lhb["pred_ret_5d"].mean(),
            }
        )
        rows.append(
            {
                "w": w,
                "arm": "list_diff_days",
                "n_dates": diff_days,
                "n_picks": int(a["is_lhb"].sum()),
                "mean_net": np.nan,
                "daily_net": np.nan,
                "win_rate": np.nan,
                "hit_r5": np.nan,
                "daily_pool_net": np.nan,
                "daily_excess": np.nan,
                "mean_pred": np.nan,
            }
        )

    summary = pd.DataFrame(rows).round(4)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx = DATA_OTHERS_DIR / f"lhb_ab_exclude_{ts}.xlsx"
    parquet = DATA_OTHERS_DIR / f"lhb_ab_picks_{ts}.parquet"
    summary.to_excel(str(xlsx), index=False)
    pd.concat(pick_list, ignore_index=True).to_parquet(parquet, index=False)
    logger.info("写回: %s", xlsx)
    logger.info("写回: %s", parquet)

    print("\n──── A/B: top-10 剔除 vs 不剔除近 N 日 LHB (r5 净收益) ────")
    cols = [
        "w",
        "arm",
        "n_dates",
        "n_picks",
        "mean_net",
        "daily_net",
        "win_rate",
        "hit_r5",
        "daily_pool_net",
        "daily_excess",
    ]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
