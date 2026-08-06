# -*- coding: utf-8 -*-
"""LHB 上榜池前向收益测量: T+2/T+3/T+5/T+10 (T+1 开盘买入, hfq).

上榜池 = 当日任一 lhb_buy_amt/lhb_sell_amt/lhb_net_buy 非空的行 (dim34 I(List) 同定义).
收益: r{h} = close_hfq(T+h) / open_hfq(T+1) - 1  (LHB 收盘后公布, T+1 开盘最早可行动).
hfq 重定基伪跳作废逻辑复用 lhb_v2_train_eval (2026-07-27 数据商重定基日等).
基准 = 全面板 (V3 已剔 ST/次新) 同口径前向收益, 对比上榜池是否跑赢市场.

输出 (WORM): BACKTEST_RESULT_DIR/<ts>/lhb_fwd_ret_<ts>.{parquet,csv}
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import BACKTEST_RESULT_DIR  # noqa: E402
from scripts.lhb_v2_train_eval import (  # noqa: E402
    HFQ_GLITCH_FACTOR,
    HFQ_REBASE_FAC_CHG,
    HFQ_REBASE_MIN_STOCKS,
    load_panel,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("measure_lhb_fwd_ret")

BASE_LHB = ["lhb_buy_amt", "lhb_sell_amt", "lhb_net_buy"]
HORIZONS = (2, 3, 5, 10, 15, 20)
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


def add_targets(df: pd.DataFrame, horizons: tuple = HORIZONS) -> pd.DataFrame:
    df["listed"] = df[BASE_LHB].notna().any(axis=1)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", group_keys=False)
    o1 = g["open_hfq"].shift(-1)
    for h in horizons:
        df[f"r{h}"] = g["close_hfq"].shift(-h) / o1 - 1

    # hfq 后复权重定基伪跳 → 窗口 t+2..t+h 含伪跳行则作废 (同 lhb_v2_train_eval).
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
    for h in horizons:
        bad = (
            cgt.groupby(df["symbol"]).shift(-h) - cgt.groupby(df["symbol"]).shift(-1)
        ) > 0
        df[f"r{h}"] = df[f"r{h}"].mask(bad)
    return df


def summarize(sub: pd.DataFrame, horizons: tuple) -> pd.DataFrame:
    rows = []
    for h in horizons:
        v = sub[f"r{h}"].dropna()
        if len(v) == 0:
            continue
        rows.append(
            {
                "horizon": f"T+{h}",
                "n": len(v),
                "mean_pct": round(100 * v.mean(), 4),
                "median_pct": round(100 * v.median(), 4),
                "std_pct": round(100 * v.std(), 4),
                "hit_rate": round(float((v > 0).mean()), 4),
                "p25_pct": round(100 * v.quantile(0.25), 4),
                "p75_pct": round(100 * v.quantile(0.75), 4),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = load_panel()
    df = add_targets(df)

    pool = df[df["listed"]].copy()
    logger.info(
        "上榜池: %d 行, %d symbols, %d 个上榜日",
        len(pool),
        pool["symbol"].nunique(),
        pool["date"].nunique(),
    )

    sum_lhb = summarize(pool, HORIZONS).assign(pool="LHB 上榜池")
    sum_mkt = summarize(df, HORIZONS).assign(pool="全市场 (V3)")
    out = pd.concat([sum_lhb, sum_mkt], ignore_index=True)
    out = out.sort_values(["horizon", "pool"]).reset_index(drop=True)

    print("\n==== LHB 上榜池 vs 全市场 前向收益 (T+1 开盘买入, hfq) ====")
    print(out.to_string(index=False))

    # ── 条件拆解: 什么情况下 LHB 后仍上涨 ──
    cond_rows = []

    def _cond(name, mask):
        sub = pool[mask].copy()
        for h in (2, 5, 10, 20):
            v = sub[f"r{h}"].dropna()
            if not len(v):
                continue
            cond_rows.append(
                {
                    "subset": name,
                    "horizon": f"T+{h}",
                    "n": len(v),
                    "mean_pct": round(100 * v.mean(), 4),
                    "median_pct": round(100 * v.median(), 4),
                    "hit_rate": round(float((v > 0).mean()), 4),
                }
            )

    if "up_limit_raw" in pool.columns:
        tol = 0.005
        _cond("涨停上榜 (close>=up_limit)", pool["close"] >= pool["up_limit_raw"] * (1 - tol))
        _cond("非涨停上榜", pool["close"] < pool["up_limit_raw"] * (1 - tol))

    if "lhb_inst_buy" in pool.columns and "lhb_inst_sell" in pool.columns:
        inst_net = pool["lhb_inst_buy"] - pool["lhb_inst_sell"]
        inst_pres = pool[["lhb_inst_buy", "lhb_inst_sell"]].notna().all(axis=1)
        _cond("有机构席位 & 机构净买", inst_pres & (inst_net > 0))
        _cond("有机构席位 & 机构净卖", inst_pres & (inst_net < 0))
        _cond("无机构席位", ~inst_pres)

    if "lhb_net_buy" in pool.columns:
        _cond("龙虎榜净买 (lhb_net_buy>0)", pool["lhb_net_buy"] > 0)
        _cond("龙虎榜净卖 (lhb_net_buy<0)", pool["lhb_net_buy"] < 0)

    cond = pd.DataFrame(cond_rows)
    print("\n==== 条件拆解: 上榜后的前向收益 ====")
    print(cond.to_string(index=False))
    out2 = cond

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = BACKTEST_RESULT_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(run_dir / f"lhb_fwd_ret_{ts}.csv", index=False, encoding="utf-8-sig")
    out2.to_csv(run_dir / f"lhb_fwd_cond_{ts}.csv", index=False, encoding="utf-8-sig")
    cols = ["date", "symbol", "listed"] + [f"r{h}" for h in HORIZONS]
    df[cols].to_parquet(run_dir / f"lhb_fwd_ret_{ts}.parquet", index=False)
    logger.info("写回: %s", run_dir / f"lhb_fwd_ret_{ts}.csv")


if __name__ == "__main__":
    main()
