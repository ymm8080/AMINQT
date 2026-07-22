# -*- coding: utf-8 -*-
"""
回测引擎 — V3.5 回测协议 (DESIGN §14.5, 实施计划 P9/P14.4)
================================================================
协议 (不可违背):
  成交价: 晚盘 14:55 + 滑点0.05% / 早盘 09:35 VWAP + 滑点0.05%; 严禁裸收盘价
          [日K近似] 日线数据无 14:55/VWAP, 买入用 T+1 open (早盘口径) 或 T close (晚盘口径),
          统一加固定滑点 — 文档化近似, 上分钟数据后替换.
  涨跌停: T+1 一字涨停买单放弃; 跌停卖单顺延至下一可交易日
  资金: 等权 1/Top_N, 单票 <= 10%, 行业 <= 4 只 (数量约束)
  成本: 佣金万2.5(双边) + 印花税0.05%(卖出) + 滑点0.05%(双边) ≈ round trip 0.13%
  验收: 扣费后净超额 vs 基准 (默认中证1000, 可注入任意基准序列)
持仓约束: 最多 max_hold_days 个交易日 (可调参).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.pipeline1.cleaning_pipeline import board_of, get_limit_pct

logger = logging.getLogger(__name__)

COMMISSION = 0.00025  # 万2.5 双边
STAMP_TAX = 0.0005  # 印花税 0.05% 仅卖出
SLIPPAGE = 0.0005  # 固定滑点 0.05% 双边


@dataclass
class BacktestProtocol:
    """V3.5 回测协议参数."""

    exec_session: str = "AM"  # "AM" 早盘 T+1 open / "PM" 晚盘 T close
    slippage: float = SLIPPAGE
    commission: float = COMMISSION
    stamp_tax: float = STAMP_TAX
    top_n: int = 15
    single_max: float = 0.10
    max_per_industry: int = 4
    max_hold_days: int = 3  # [TUNABLE] 持仓上限
    # 日线近似退出规则 (与 rule_engine_v2 Config 对齐, 调参目标)
    hard_stop: float = -0.04  # [TUNABLE] 持仓浮亏硬止损 (相对成本)
    trailing_drawdown: float = 0.04  # [TUNABLE] 移动止盈: 高点回撤
    prob_exit: float = 0.50  # [TUNABLE] 概率衰减退出
    exec_price_col: str = "open"  # AM 口径


@dataclass
class DailyBar:
    """单日组合状态快照."""

    date: pd.Timestamp
    nav: float
    cash: float
    n_positions: int


class BacktestEngineV35:
    """日频回测: 每日清单(调用方提供) → 模拟成交 → 绩效.

    Args:
        panel: 全市场日线面板 (symbol/date/open/high/low/close/pre_close/board/industry/amount)
        protocol: 回测协议参数
    """

    def __init__(self, panel: pd.DataFrame, protocol: BacktestProtocol | None = None):
        self.cfg = protocol or BacktestProtocol()
        self.panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
        self.dates = sorted(self.panel["date"].unique())
        self._by_date = {
            d: g.set_index("symbol") for d, g in self.panel.groupby("date")
        }

    # ---------------- 工具 ----------------
    def _bar(self, date, symbol) -> pd.Series | None:
        g = self._by_date.get(date)
        if g is None or symbol not in g.index:
            return None
        return g.loc[symbol]

    def _exec_buy_price(self, date, symbol) -> float | None:
        """成交价 (协议 §1): AM = T+1 open × (1+滑点); 一字涨停放弃 (协议 §2)."""
        bar = self._bar(date, symbol)
        if bar is None:
            return None
        limit = get_limit_pct(board_of(symbol), pd.Timestamp(date))
        lu = round(bar["pre_close"] * (1 + limit), 2)
        open_px = bar["open"] if self.cfg.exec_session == "AM" else bar["close"]
        if abs(open_px - lu) < 0.01:
            return None  # 一字涨停, 买单放弃
        return open_px * (1 + self.cfg.slippage)

    def _exec_sell_price(self, date, symbol) -> float | None:
        """跌停顺延: 返回 None 表示当日不可卖."""
        bar = self._bar(date, symbol)
        if bar is None:
            return None
        limit = get_limit_pct(board_of(symbol), pd.Timestamp(date))
        ld = round(bar["pre_close"] * (1 - limit), 2)
        px = bar["open"] if self.cfg.exec_session == "AM" else bar["close"]
        if abs(px - ld) < 0.01:
            return None  # 跌停, 卖单顺延
        return px * (1 - self.cfg.slippage)

    @staticmethod
    def _costs(buy_amount: float, sell_amount: float, cfg: BacktestProtocol) -> float:
        return buy_amount * cfg.commission + sell_amount * (
            cfg.commission + cfg.stamp_tax
        )

    # ---------------- 主回测 ----------------
    def run(
        self,
        daily_lists: dict,
        benchmark: pd.Series | None = None,
        initial_capital: float = 1_000_000,
    ) -> dict:
        """执行回测.

        Args:
            daily_lists: {date: DataFrame(symbol, score[, prob_up, industry])}
                         每日候选清单 (由 ListGenerator 或 mock 提供), T 日清单 T+1 执行
            benchmark:   基准日收益序列 (index=date), 默认 0 (绝对收益)
            initial_capital: 初始资金

        Returns:
            {nav_curve, trades, metrics}
        """
        cfg = self.cfg
        cash, positions = initial_capital, {}
        nav_hist, trades = [], []
        bench = benchmark if benchmark is not None else pd.Series(0.0, index=self.dates)

        for i, date in enumerate(self.dates):
            self._by_date[date]

            # ---- 1. 持仓估值 + 退出裁决 (止损/移动止盈/概率衰减/到期) ----
            for sym in list(positions):
                pos = positions[sym]
                bar = self._bar(date, sym)
                if bar is None:
                    continue
                pos["hold_days"] += 1
                pos["high"] = max(pos["high"], bar["high"])
                px = bar["close"]
                pnl = px / pos["cost"] - 1
                dd_high = px / pos["high"] - 1
                prob = pos.get("prob_up", 1.0)
                reason = None
                if pnl <= cfg.hard_stop:
                    reason = f"硬止损{pnl:.1%}"
                elif pos["high"] > pos["cost"] and dd_high <= -cfg.trailing_drawdown:
                    reason = f"移动止盈{dd_high:.1%}"
                elif prob < cfg.prob_exit:
                    reason = f"概率衰减{prob:.2f}"
                elif pos["hold_days"] >= cfg.max_hold_days:
                    reason = f"持仓满{cfg.max_hold_days}日"
                if reason:
                    sell_px = self._exec_sell_price(date, sym)
                    if sell_px is not None:
                        amount = pos["shares"] * sell_px
                        cash += amount - self._costs(0, amount, cfg)
                        trades.append(
                            {
                                "date": date,
                                "symbol": sym,
                                "side": "sell",
                                "price": sell_px,
                                "reason": reason,
                                "pnl": sell_px / pos["cost"] - 1,
                            }
                        )
                        del positions[sym]

            # ---- 2. 执行昨日清单买入 (T+1) ----
            prev_date = self.dates[i - 1] if i > 0 else None
            lst = daily_lists.get(prev_date) if prev_date is not None else None
            if lst is not None and len(lst):
                lst = lst[~lst["symbol"].isin(positions)]
                # 行业数量约束
                ind_count = {}
                for sym in positions:
                    ind = positions[sym]["industry"]
                    ind_count[ind] = ind_count.get(ind, 0) + 1
                picks = []
                for _, row in lst.sort_values("score", ascending=False).iterrows():
                    ind = row.get("industry", "UNKNOWN")
                    if ind_count.get(ind, 0) >= cfg.max_per_industry:
                        continue
                    picks.append(row)
                    ind_count[ind] = ind_count.get(ind, 0) + 1
                    if len(picks) >= cfg.top_n - len(positions):
                        break
                # 等权 1/top_n, 单票 <= 10%
                nav_now = cash + sum(
                    p["shares"]
                    * (
                        self._bar(date, s)["close"]
                        if self._bar(date, s) is not None
                        else p["cost"]
                    )
                    for s, p in positions.items()
                )
                budget = min(nav_now * min(1 / cfg.top_n, cfg.single_max), cash)
                for row in picks:
                    px = self._exec_buy_price(date, row["symbol"])
                    if px is None or budget < nav_now * 0.03:
                        continue
                    shares = int(budget / px / 100) * 100
                    if shares <= 0:
                        continue
                    amount = shares * px
                    cash -= amount + self._costs(amount, 0, cfg)
                    positions[row["symbol"]] = {
                        "cost": px,
                        "shares": shares,
                        "high": px,
                        "hold_days": 0,
                        "industry": row.get("industry", "UNKNOWN"),
                        "prob_up": row.get("prob_up", 1.0),
                    }
                    trades.append(
                        {
                            "date": date,
                            "symbol": row["symbol"],
                            "side": "buy",
                            "price": px,
                            "reason": f"清单score={row['score']:.4f}",
                            "pnl": np.nan,
                        }
                    )

            # ---- 3. 日终净值 ----
            nav = cash + sum(
                p["shares"]
                * (
                    self._bar(date, s)["close"]
                    if self._bar(date, s) is not None
                    else p["cost"]
                )
                for s, p in positions.items()
            )
            nav_hist.append(DailyBar(date, nav, cash, len(positions)))

        return {
            "nav_curve": pd.DataFrame([vars(b) for b in nav_hist]),
            "trades": pd.DataFrame(
                trades, columns=["date", "symbol", "side", "price", "reason", "pnl"]
            ),
            "metrics": self._metrics(nav_hist, bench, initial_capital),
        }

    # ---------------- 绩效 ----------------
    def _metrics(
        self, nav_hist: list[DailyBar], bench: pd.Series, initial: float
    ) -> dict:
        nav = pd.Series({b.date: b.nav for b in nav_hist})
        ret = nav.pct_change().dropna()
        bench = bench.reindex(nav.index).fillna(0.0)
        ann = (nav.iloc[-1] / initial) ** (252 / max(len(nav), 1)) - 1
        bench_ann = (
            (1 + bench).prod() ** (252 / max(len(bench), 1)) - 1
            if bench.abs().sum() > 0
            else 0.0
        )
        excess_series = ret - bench.pct_change().fillna(bench).reindex(
            ret.index
        ).fillna(0)
        excess_ann = float(excess_series.mean() * 252)
        dd = (nav / nav.cummax() - 1).min()
        sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
        return {
            "total_return": float(nav.iloc[-1] / initial - 1),
            "annual_return": float(ann),
            "benchmark_annual": float(bench_ann),
            "net_excess_annual": excess_ann,  # 扣费后净超额 (验收口径)
            "max_drawdown": float(dd),
            "sharpe": sharpe,
            "n_days": len(nav),
        }
