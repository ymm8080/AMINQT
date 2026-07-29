"""
回测引擎 — V3.8 回测协议 (DESIGN §14.5, 实施计划 P9/P14.4, PIPELINE1_V3.8 §四 bis)
================================================================================
协议 (不可违背):
  成交价: 晚盘 14:55 + 分层滑点 [E5] / 早盘 09:35 VWAP + 分层滑点; 严禁裸收盘价
          [日K近似] 日线数据无 14:55/VWAP, 买入用 T+1 open (早盘口径) 或 T close (晚盘口径),
          统一加分层滑点 — 文档化近似, 上分钟数据后替换.
  [E5] 滑点分层 (按 ADV20): >5亿→0.05% / 1~5亿→0.10% / <1亿→0.15% (双边计入);
       panel 无 adv20 列时回退固定滑点. slippage_multiplier=2 即 2 倍滑点敏感性测试
       (E5 强制门禁: 2倍滑点下净年化超额 ≥5% 方算稳健).
  涨跌停: T+1 一字涨停买单放弃; 跌停卖单顺延至下一可交易日
  资金: 等权 1/Top_N, 单票 <= 10%, 行业 <= 4 只 (数量约束)
  成本: 佣金万2.5(双边) + 印花税0.05%(卖出) + 分层滑点 [E5]
  验收: 扣费后净超额 vs 基准 (默认中证1000, 可注入任意基准序列); Sortino 为主目标 [E11]
  [E9/E11] daily_multiplier 钩子: 波动率熔断/熊市协议的仓位乘数按日注入;
           空仓期现金按 cash_yield_annual 计逆回购收益 (E11 熊市协议回测口径).
持仓约束: 最多 max_hold_days 个交易日 (可调参).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.pipeline1.cleaning_pipeline import board_of, get_limit_pct
from app.pipeline1.label_engine import slippage_tier

logger = logging.getLogger(__name__)


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法: 分母为零时返回 default, 避免 ZeroDivisionError."""
    if abs(denominator) < 1e-8:
        return default
    return numerator / denominator

COMMISSION = 0.00025  # 万2.5 双边
STAMP_TAX = 0.0005  # 印花税 0.05% 仅卖出
SLIPPAGE = 0.0005  # 固定滑点 0.05% 双边 (panel 无 adv20 时回退)


@dataclass
class BacktestProtocol:
    """V3.8 回测协议参数."""

    exec_session: str = "AM"  # "AM" 早盘 T+1 open / "PM" 晚盘 T close
    slippage: float = SLIPPAGE
    tiered_slippage: bool = True  # [E5] 按 ADV20 分层 (panel 需含 adv20 列)
    slippage_multiplier: float = 1.0  # [E5] 2.0 = 2倍滑点敏感性测试 (上线门禁)
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
    cash_yield_annual: float = 0.0  # [E11] 空仓资金逆回购/货基年化 (≈2%)


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

    def _slippage_for(self, bar: pd.Series) -> float:
        """[E5] 分层滑点 (ADV20 三档) × 敏感性乘数; 无 adv20 列回退固定滑点."""
        if self.cfg.tiered_slippage and "adv20" in bar.index:
            return slippage_tier(bar["adv20"]) * self.cfg.slippage_multiplier
        return self.cfg.slippage * self.cfg.slippage_multiplier

    def _exec_buy_price(self, date, symbol) -> float | None:
        """成交价 (协议 §1): AM = T+1 open × (1+滑点); 一字涨停放弃 (协议 §2)."""
        bar = self._bar(date, symbol)
        if bar is None:
            return None
        limit = get_limit_pct(board_of(symbol), pd.Timestamp(date))
        lu = round(bar["pre_close"] * (1 + limit), 2)
        open_px = bar["open"] if self.cfg.exec_session == "AM" else bar["close"]
        if abs(open_px - lu) < max(0.01, lu * 0.001):
            return None  # 一字涨停, 买单放弃
        return open_px * (1 + self._slippage_for(bar))

    def _exec_sell_price(self, date, symbol) -> float | None:
        """跌停顺延: 返回 None 表示当日不可卖."""
        bar = self._bar(date, symbol)
        if bar is None:
            return None
        limit = get_limit_pct(board_of(symbol), pd.Timestamp(date))
        ld = round(bar["pre_close"] * (1 - limit), 2)
        px = bar["open"] if self.cfg.exec_session == "AM" else bar["close"]
        if abs(px - ld) < max(0.01, ld * 0.001):
            return None  # 跌停, 卖单顺延
        return px * (1 - self._slippage_for(bar))

    @staticmethod
    def _px_close(bar: pd.Series) -> float:
        """估值用收盘价: 优先 close_hfq (总回报含分红), 缺省回退 close.

        close_hfq 消除除权除息日的价格断层, PnL/NAV 基于此计算
        才能反映实盘总回报 (价格+分红), 避免除权日假亏损误触止损。
        执行价仍用原始 open/close (实际成交价), 不受此影响。
        """
        return float(bar.get("close_hfq", bar["close"]))

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
        daily_multiplier: dict | None = None,
    ) -> dict:
        """执行回测.

        Args:
            daily_lists: {date: DataFrame(symbol, score[, prob_up, industry])}
                         每日候选清单 (由 ListGenerator 或 mock 提供), T 日清单 T+1 执行
            benchmark:   基准日收益序列 (index=date), 默认 0 (绝对收益)
            initial_capital: 初始资金
            daily_multiplier: {date: 仓位乘数} [E9 波动率熔断 / E11 熊市协议按日注入];
                              缺省 1.0, 0.0 = 当日不开新仓

        Returns:
            {nav_curve, trades, metrics}
        """
        cfg = self.cfg
        cash, positions = initial_capital, {}
        nav_hist, trades = [], []
        bench = benchmark if benchmark is not None else pd.Series(0.0, index=self.dates)
        daily_multiplier = daily_multiplier or {}
        cash_yield_daily = cfg.cash_yield_annual / 252

        for i, date in enumerate(self.dates):
            self._by_date[date]
            cash *= 1 + cash_yield_daily  # [E11] 空仓资金逆回购收益计入

            # ---- 1. 持仓估值 + 退出裁决 (止损/移动止盈/概率衰减/到期) ----
            for sym in list(positions):
                pos = positions[sym]
                bar = self._bar(date, sym)
                if bar is None:
                    continue
                pos["hold_days"] += 1
                pos["high"] = max(pos["high"], bar["high"])
                px = self._px_close(bar)
                pnl = _safe_divide(px, pos["cost_hfq"]) - 1
                dd_high = _safe_divide(px, pos["high_hfq"]) - 1
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
            multiplier = float(daily_multiplier.get(date, 1.0))  # [E9/E11]
            if lst is not None and len(lst) and multiplier > 0:
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
                        self._px_close(self._bar(date, s))
                        if self._bar(date, s) is not None
                        else p["cost_hfq"]
                    )
                    for s, p in positions.items()
                )
                budget = min(nav_now * min(1 / cfg.top_n, cfg.single_max), cash)
                budget *= multiplier  # [E9/E11] 仓位乘数 (熔断/熊市协议)
                for row in picks:
                    px = self._exec_buy_price(date, row["symbol"])
                    if px is None or budget < nav_now * 0.03:
                        continue
                    shares = int(budget / px / 100) * 100
                    if shares <= 0:
                        continue
                    amount = shares * px
                    cash -= amount + self._costs(amount, 0, cfg)
                    buy_bar = self._bar(date, row["symbol"])
                    cost_hfq = self._px_close(buy_bar) if buy_bar is not None else px
                    positions[row["symbol"]] = {
                        "cost": px,
                        "cost_hfq": cost_hfq,
                        "shares": shares,
                        "high": px,
                        "high_hfq": cost_hfq,
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
                            "score": float(row.get("score", 0.5)),
                        }
                    )

            # ---- 3. 日终净值 ----
            nav = cash + sum(
                p["shares"]
                * (
                    self._px_close(self._bar(date, s))
                    if self._bar(date, s) is not None
                    else p["cost_hfq"]
                )
                for s, p in positions.items()
            )
            nav_hist.append(DailyBar(date, nav, cash, len(positions)))

        trades_df = pd.DataFrame(
            trades,
            columns=["date", "symbol", "side", "price", "reason", "pnl", "score"],
        )
        return {
            "nav_curve": pd.DataFrame([vars(b) for b in nav_hist]),
            "trades": trades_df,
            "metrics": self._metrics(nav_hist, bench, initial_capital, trades_df),
        }

    # ---------------- 绩效 ----------------
    def _metrics(
        self,
        nav_hist: list[DailyBar],
        bench: pd.Series,
        initial: float,
        trades_df: pd.DataFrame | None = None,
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
        # [E11] Sortino (主目标): 只惩罚下行波动
        downside = ret[ret < 0]
        sortino = (
            float(ret.mean() / downside.std() * np.sqrt(252))
            if len(downside) > 1 and downside.std() > 0
            else 0.0
        )

        # ---- 赚钱指标: 胜率/盈亏比/期望/最大连亏 ----
        win_rate, pl_ratio, expectancy, max_consec_loss, total_trades, avg_holding = (
            0.0,
            0.0,
            0.0,
            0,
            0,
            0.0,
        )
        if trades_df is not None and len(trades_df):
            sells = trades_df[trades_df["side"] == "sell"]
            if len(sells):
                pnls = sells["pnl"].dropna().values
                total_trades = len(pnls)
                wins = pnls[pnls > 0]
                losses = pnls[pnls <= 0]
                win_rate = float(len(wins) / total_trades) if total_trades else 0.0
                avg_win = float(wins.mean()) if len(wins) else 0.0
                avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
                pl_ratio = float(avg_win / avg_loss) if avg_loss > 0 else 0.0
                expectancy = float(pnls.mean())
                streak = 0
                for p in pnls:
                    streak = streak + 1 if p <= 0 else 0
                    max_consec_loss = max(max_consec_loss, streak)
                # 每笔卖出匹配最临近的前一笔买入 (同一 symbol, 严格早于卖出日)
                buys = trades_df[trades_df["side"] == "buy"].sort_values("date")
                hold_days = []
                for _, sell in sells.iterrows():
                    sell_dt = pd.to_datetime(sell["date"])
                    sym_buys = buys[
                        (buys["symbol"] == sell["symbol"])
                        & (pd.to_datetime(buys["date"]) < sell_dt)
                    ]
                    if len(sym_buys):
                        entry_dt = pd.to_datetime(sym_buys.iloc[-1]["date"])
                        hold_days.append((sell_dt - entry_dt).days)
                avg_holding = float(np.mean(hold_days)) if hold_days else 0.0

        # ---- OOS Rank IC: 预测排序 vs 实际收益 — 核心判断"模型是否有效" ----
        oos_rank_ic, _ic_rolling_20d, ic_daily = 0.0, 0.0, []
        if trades_df is not None and len(trades_df):
            sells = trades_df[trades_df["side"] == "sell"]
            if len(sells) >= 5:
                from scipy.stats import spearmanr

                # Match each sell with its buy score (if available via the buy trade)
                pnl_scores = []
                for _, sell in sells.iterrows():
                    buy = trades_df[
                        (trades_df["side"] == "buy")
                        & (trades_df["symbol"] == sell["symbol"])
                    ]
                    if len(buy):
                        pnl_scores.append((sell["pnl"], buy.iloc[-1].get("score", 0.5)))
                if len(pnl_scores) >= 5:
                    pnls_arr = np.array([p[0] for p in pnl_scores], dtype=float)
                    scores_arr = np.array([p[1] for p in pnl_scores], dtype=float)
                    valid = ~(np.isnan(pnls_arr) | np.isnan(scores_arr))
                    pnls_arr = pnls_arr[valid]
                    scores_arr = scores_arr[valid]
                    if len(pnls_arr) >= 5 and np.std(scores_arr) > 1e-9:
                        r = spearmanr(scores_arr, pnls_arr)
                        oos_rank_ic = (
                            float(r.correlation) if not np.isnan(r.correlation) else 0.0
                        )

        return {
            "total_return": float(nav.iloc[-1] / initial - 1),
            "annual_return": float(ann),
            "benchmark_annual": float(bench_ann),
            "net_excess_annual": excess_ann,
            "max_drawdown": float(dd),
            "sharpe": sharpe,
            "sortino": sortino,
            "n_days": len(nav),
            # 赚钱指标
            "win_rate": round(win_rate, 4),
            "pl_ratio": round(pl_ratio, 4),
            "expectancy": round(expectancy, 5),
            "total_trades": total_trades,
            "max_consecutive_loss": max_consec_loss,
            "avg_holding_days": round(avg_holding, 1),
            # OOS IC — 模型预测排序有效性
            "oos_rank_ic": round(oos_rank_ic, 4),
            "ic_daily": [round(float(x), 4) for x in ic_daily[::-1][:20]],
        }
