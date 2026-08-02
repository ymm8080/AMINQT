# -*- coding: utf-8 -*-
"""Pipeline-1 卖出信号评估器 (预测驱动 + 价格硬止损)
================================================================
输入: 持仓股当天的预测行 (V35Predictor.predict 输出), 每 symbol 一行.
输出: 每行追加 sell_signal + sell_reason 两列.

信号级别 (预测驱动, 优先级 红 > 橙 > 黄 > 绿):
  hold         绿/持有: 无任何预警
  watch        黄/警戒: prob_up<0.5 / pred_ret_1d<0 / pain_prob>=0.4
  sell         橙/卖出: pred_ret_1d<=-0.5% / prob_up<0.45 / pain_prob>=0.5
  strong_sell  红/强卖: pred_ret_1d<=-1.5% / (pred_ret_3d<=-2% 且 1d<0)
                        / pain_prob>=0.6 / pred_q10<=-3%
价格硬止损 (用户 2026-08-02 裁决): pnl (现价/买入成本-1) <= -6% → 红/强卖,
覆盖任何预测信号. 提供 pnl_col 时才启用.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGNAL_LEVELS = ["hold", "watch", "sell", "strong_sell"]

# 预测驱动阈值 (默认; 可调)
PROB_WATCH = 0.50
PROB_SELL = 0.45
RET1D_SELL = -0.005
RET1D_STRONG = -0.015
RET3D_STRONG = -0.02
PAIN_WATCH = 0.40
PAIN_SELL = 0.50
PAIN_STRONG = 0.60
Q10_STRONG = -0.03

# 价格硬止损 (相对买入成本)
PRICE_HARD_STOP = -0.06


def evaluate_sell_signal(
    pred: pd.DataFrame,
    pnl_col: str | None = None,
    price_hard_stop: float = PRICE_HARD_STOP,
) -> pd.DataFrame:
    """对预测行计算卖出信号级别 + 原因 (向量化).

    Args:
        pred: predict() 输出, 含 pred_ret_1d/3d/5d, prob_up 等;
              缺失列按中性值处理 (prob_up→0.5, 其余→0)
        pnl_col: 列名, 内容为当前相对买入成本收益率 (现价/成本-1).
                 存在时 pnl <= price_hard_stop → 红/强卖 (覆盖预测信号)
        price_hard_stop: 价格硬止损阈值 (默认 -6%)

    Returns:
        pred + sell_signal (hold/watch/sell/strong_sell) + sell_reason (str)
    """
    out = pred.copy()
    n = len(out)
    level = np.zeros(n, dtype=int)  # 0=hold 1=watch 2=sell 3=strong_sell
    reason = np.full(n, "", dtype=object)

    def col(name: str, default: float) -> np.ndarray:
        if name in out.columns:
            return out[name].astype(float).fillna(default).to_numpy()
        return np.full(n, default, dtype=float)

    r1 = col("pred_ret_1d", 0.0)
    r3 = col("pred_ret_3d", 0.0)
    prob = col("prob_up", 0.5)
    pain = col("pain_prob", 0.0)
    q10 = col("pred_q10", 0.0)

    def hit(mask: np.ndarray, lv: int, make_reason) -> None:
        """mask 命中的行若当前级别低于 lv, 升级到 lv 并写原因 (同级别首个保留)."""
        up = np.asarray(mask, dtype=bool) & (level < lv)
        if not up.any():
            return
        idx = np.flatnonzero(up)
        level[up] = lv
        reason[up] = make_reason(idx)

    # ---- 红/强卖 ----
    def _r1_strong(idx):
        return [f"次日预期{r1[i] * 100:+.1f}%≤{RET1D_STRONG * 100:.0f}%" for i in idx]

    def _r3_strong(idx):
        return [f"3日预期{r3[i] * 100:+.1f}%且次日为负" for i in idx]

    def _pain_strong(idx):
        return [f"浮亏预警{pain[i]:.2f}≥{PAIN_STRONG:.2f}" for i in idx]

    def _q10_strong(idx):
        return [f"下行分位{q10[i] * 100:.1f}%≤{Q10_STRONG * 100:.0f}%" for i in idx]

    hit(r1 <= RET1D_STRONG, 3, _r1_strong)
    hit((r3 <= RET3D_STRONG) & (r1 < 0), 3, _r3_strong)
    hit(pain >= PAIN_STRONG, 3, _pain_strong)
    hit(q10 <= Q10_STRONG, 3, _q10_strong)

    # ---- 橙/卖出 ----
    def _r1_sell(idx):
        return [f"次日预期{r1[i] * 100:+.1f}%≤{RET1D_SELL * 100:.1f}%" for i in idx]

    def _prob_sell(idx):
        return [f"胜率{prob[i]:.2f}<{PROB_SELL:.2f}" for i in idx]

    def _pain_sell(idx):
        return [f"浮亏预警{pain[i]:.2f}≥{PAIN_SELL:.2f}" for i in idx]

    hit(r1 <= RET1D_SELL, 2, _r1_sell)
    hit(prob < PROB_SELL, 2, _prob_sell)
    hit(pain >= PAIN_SELL, 2, _pain_sell)

    # ---- 黄/警戒 ----
    def _prob_watch(idx):
        return [f"胜率{prob[i]:.2f}<{PROB_WATCH:.2f}" for i in idx]

    def _r1_watch(idx):
        return [f"次日预期转负{r1[i] * 100:.2f}%" for i in idx]

    def _pain_watch(idx):
        return [f"浮亏预警{pain[i]:.2f}≥{PAIN_WATCH:.2f}" for i in idx]

    hit(prob < PROB_WATCH, 1, _prob_watch)
    hit(r1 < 0.0, 1, _r1_watch)
    hit(pain >= PAIN_WATCH, 1, _pain_watch)

    # ---- 价格硬止损 (覆盖一切预测信号) ----
    if pnl_col and pnl_col in out.columns:
        pnl = out[pnl_col].astype(float).to_numpy()

        def _stop(idx):
            return [
                f"硬止损{pnl[i] * 100:+.1f}%≤{price_hard_stop * 100:.0f}%"
                for i in idx
            ]

        hit(pnl <= price_hard_stop, 3, _stop)

    out["sell_signal"] = [SIGNAL_LEVELS[l] for l in level]
    out["sell_reason"] = [r if r else "持有" for r in reason]
    return out
