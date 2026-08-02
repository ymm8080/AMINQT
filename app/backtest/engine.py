# -*- coding: utf-8 -*-
"""模块5: BacktestEngine — 日频多周期盘中触发回测引擎 (v4.0 + V5.2日内规则融合).

核心特性:
    - 所有价格用整数分(fen), 禁止float累积
    - T+1交收: available_cash / frozen_cash 分离
    - 仅 Approximate Mode (废除 Strict Mode)
    - 分离滑点: 买入10bp / 正常卖出10bp / 止损卖出30bp
    - filter_trend 默认关闭; 开启时 T+1 open >= pre_close (T收盘)
    - V5.2融合: S1止损=max(固定,-1.5*ATR)分主板/双创 + 噪音带断言
    - V5.2融合: S2移动止盈=浮盈3%激活, 回撤max(3%,1*ATR)
    - V5.2融合: S5a时间止损=2日收益<20日中位数 (个股自比)
    - V5.2融合: S5b持仓到期=2日强制轮动
    - V5.2融合: B3追高过滤7% + B7止损距离否决1.2*ATR
    - V5.2融合: 日保险丝=2σ+4%双轨 + 系统停机线15%
    - 移动止盈用最高收盘价 (非日内最高价)
    - 现金利息 0.3%年化
    - 成交量限制: 单笔买入≤当日成交额10%
    - 每日对账 assert + 检查点机制
    - 连续亏损按交易日计 (非按笔数)
    - 持仓冲突: 不盲目剔除, 比较收益/风险后决定保留或替换
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from app.backtest.config_manager import BacktestConfig, ConfigManager

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


def _safe_div(num: float, den: float) -> float:
    """安全除法: 分母为 0/NaN 时返回 0."""
    if den == 0 or not np.isfinite(den):
        return 0.0
    return float(np.nan_to_num(num / den, nan=0.0))


def _softmax(x: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


class PositionMode(Enum):
    """仓位分配模式."""

    SQUAD = "squad"
    SNIPER = "sniper"
    SNIPER_MAX = "sniper_max"


@dataclass
class Trade:
    """单笔交易记录. 价格为整数分."""

    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    stock: str
    entry_price_fen: int
    exit_price_fen: int
    quantity: int
    pnl_fen: int
    pnl_pct: float
    exit_reason: str
    horizon: int
    mode: str
    prob_up: float = 0.0
    pred_ret: float = 0.0
    score: float = 0.0
    max_profit_pct: float = 0.0
    commission_entry_fen: int = 0
    commission_exit_fen: int = 0
    stamp_tax_fen: int = 0
    is_swap: bool = False  # V5.0: 是否为换仓交易


@dataclass
class Holding:
    """持仓记录. 价格为整数分.

    V5.2融合: 增加ATR自适应止损价、板块标识、20日中位数基准.
    """

    stock: str
    entry_date: pd.Timestamp
    entry_price_fen: int
    quantity: int
    horizon: int
    mode: str
    prob_up: float
    pred_ret: float
    score: float
    days_held: int = 0
    max_close_fen: int = 0
    stop_loss_triggered: bool = False
    stop_profit_triggered: bool = False
    # V5.2 新增
    stop_price_fen: int = 0  # S1 动态止损价 (max(固定, -1.5*ATR))
    atr_pct: float = 0.0  # 入场时ATR百分比
    board: str = "main"  # main(主板) / dual(双创)
    median_2d_return: float = 0.0  # S5a: 该股20日2日收益中位数
    down_limit_days: int = 0  # V5.0 F2.19 连续跌停天数


class BacktestEngine:
    """日频多周期盘中触发回测引擎 (v4.0)."""

    def __init__(
        self,
        config: BacktestConfig,
        pred_df: pd.DataFrame,
        price_df: pd.DataFrame,
        trade_dates: List[pd.Timestamp],
        data_version_hash: str,
        market_df: pd.DataFrame | None = None,
    ):
        """初始化回测引擎.

        Args:
            config: 回测配置.
            pred_df: 预测表.
            price_df: 行情表.
            trade_dates: 交易日历.
            data_version_hash: 数据版本哈希.
            market_df: 大盘指数数据 (V5.0, 可选).
        """
        self.config = config
        self.pred_df = pred_df.copy()
        self.price_df = price_df.copy()
        self.trade_dates = trade_dates
        self.data_version_hash = data_version_hash
        self.config_hash = ConfigManager.hash(config)
        self._market_df = market_df if market_df is not None else pd.DataFrame()

        self.position_mode = PositionMode(config.position_mode)
        self._max_pos_pct = {
            PositionMode.SQUAD: 0.30,
            PositionMode.SNIPER: 0.60,
            PositionMode.SNIPER_MAX: 1.00,
        }[self.position_mode]

        self._price_index: Dict[Tuple, pd.Series] = {}
        for _, row in self.price_df.iterrows():
            self._price_index[(row["date"], row["stock"])] = row

        self._reset_state()

    def _reset_state(self) -> None:
        """重置回测状态."""
        self.available_cash_fen: int = self._price_to_fen(self.config.initial_capital)
        self.frozen_cash_fen: int = 0
        self.holdings: List[Holding] = []
        self.trades: List[Trade] = []
        self.daily_records: List[dict] = []
        self.holdings_history: List[dict] = []
        self.consecutive_loss_days: int = 0
        self.cooldown_days: int = 0
        self.stop_new_positions: bool = False
        self.prev_day_pnl_pct: float = 0.0
        self._daily_tradeable: int = 0
        self._daily_total: int = 0
        self._daily_loss_days: int = 0
        # V5.2 新增
        self._atr_cache: Dict[str, float] = {}  # {stock: atr_pct}
        self._median_2d_cache: Dict[str, float] = {}  # {stock: median_2d_return}
        self._peak_nav_fen: int = self._price_to_fen(self.config.initial_capital)
        self._daily_pnl_history: List[float] = []  # 用于2σ计算
        self._system_halted: bool = False

    # ── 价格工具 (fen 精度) ─────────────────────────────────

    @staticmethod
    def _price_to_fen(price_yuan: float) -> int:
        """元转分: int(price_yuan * 100 + 0.5)."""
        return int(float(price_yuan) * 100 + 0.5)

    @staticmethod
    def _fen_to_yuan(price_fen: int) -> float:
        """分转元."""
        return price_fen / 100.0

    def _calc_trigger_price(self, open_fen: int) -> int:
        """触发价 = int(open_fen * 103 / 100 + 0.5)."""
        return int(open_fen * 103 / 100 + 0.5)

    def _calc_entry_price(self, trigger_fen: int, slippage_bp: int) -> int:
        """买入价 = int(trigger_fen * (1000 + bp) / 1000 + 0.5).

        约束: entry_price <= high_fen, entry_price < up_limit_fen.
        """
        return int(trigger_fen * (1000 + slippage_bp) / 1000 + 0.5)

    def _calc_commission(self, amount_fen: int) -> int:
        """佣金 = max(amount_yuan * rate, min_commission) 转分."""
        amount_yuan = amount_fen / 100.0
        comm_yuan = max(
            amount_yuan * self.config.commission_rate, self.config.min_commission
        )
        return int(comm_yuan * 100 + 0.5)

    # ── 行情查询 ────────────────────────────────────────────

    def _get_price_row(self, date: pd.Timestamp, stock: str) -> pd.Series | None:
        """获取行情数据行."""
        return self._price_index.get((date, stock))

    def _get_trade_date_after(
        self, date: pd.Timestamp, n: int = 1
    ) -> pd.Timestamp | None:
        """获取 date 后第 n 个交易日 (使用交易日历)."""
        idx = np.searchsorted(self.trade_dates, date)
        target = idx + n
        if 0 <= target < len(self.trade_dates):
            return self.trade_dates[target]
        return None

    # ── V5.2 ATR / 板块 / 中位数 工具 ──────────────────────

    def _check_volume_confirmation(
        self, today_row: pd.Series, yesterday_row: pd.Series
    ) -> bool:
        """V5.0 F2.17 成交量确认: T+1日成交量 >= T日成交量 × volume_confirm_ratio.

        Args:
            today_row: T+1日行情.
            yesterday_row: T日行情.

        Returns:
            True 如果成交量确认通过.
        """
        today_vol = today_row.get("volume", 0)
        yesterday_vol = yesterday_row.get("volume", 0)
        if yesterday_vol <= 0:
            return True
        return today_vol >= yesterday_vol * self.config.volume_confirm_ratio

    def _check_market_drop(self, today: pd.Timestamp, yesterday: pd.Timestamp) -> bool:
        """V5.0 F2.18 大盘跌幅过滤: 大盘跌幅 >= 2% → 停止买入.

        Returns:
            True 如果大盘跌幅可接受.
        """
        if self._market_df is None or self._market_df.empty:
            return True

        y_market = self._market_df[self._market_df["date"] == yesterday]
        if y_market.empty:
            return True
        y_close = y_market["index_close"].iloc[0]
        y_idx_list = self._market_df.index[
            self._market_df["date"] == yesterday
        ].tolist()
        if not y_idx_list:
            return True
        y_pos = self._market_df.index.get_loc(y_idx_list[0])
        if y_pos <= 0:
            return True
        prev_close = self._market_df.iloc[y_pos - 1]["index_close"]
        if prev_close <= 0:
            return True
        drop = _safe_div(y_close - prev_close, prev_close)
        return drop > self.config.market_drop_limit

    def _check_down_limit_force_close(
        self, date: pd.Timestamp, holding: Holding, row: pd.Series
    ) -> bool:
        """V5.0 F2.19 连续跌停强制平仓: 连续3日跌停 → 第3日收盘卖出.

        Returns:
            True 如果需要强制平仓.
        """
        close_fen = self._price_to_fen(row["close"])
        down_limit = row.get("down_limit", np.nan)
        if not np.isfinite(down_limit):
            return False

        down_limit_fen = self._price_to_fen(down_limit)
        if close_fen <= down_limit_fen:
            holding.down_limit_days += 1
            if holding.down_limit_days >= self.config.down_limit_max_days:
                logger.warning(
                    "%s 连续跌停 %d 日, 强制平仓",
                    holding.stock,
                    holding.down_limit_days,
                )
                return True
        else:
            holding.down_limit_days = 0

        return False

    def _get_board(self, stock: str) -> str:
        """判断板块: 688/300开头=双创(dual), 其余=主板(main).

        Args:
            stock: 股票代码.

        Returns:
            "main" 或 "dual".
        """
        code = stock.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
        if code.startswith("688") or code.startswith("300"):
            return "dual"
        return "main"

    def _precompute_atr(self, stock: str, price_df: pd.DataFrame) -> float:
        """预计算ATR百分比 (14周期, 日频).

        ATR_pct = ATR / close  (无量纲化)

        Args:
            stock: 股票代码.
            price_df: 该股票的历史行情.

        Returns:
            ATR占价格的比例, 0 if 数据不足.
        """
        if stock in self._atr_cache:
            return self._atr_cache[stock]
        df = price_df[price_df["stock"] == stock].sort_values("date")
        if len(df) < self.config.atr_period + 1:
            self._atr_cache[stock] = 0.0
            return 0.0
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        atr = float(np.mean(tr[-self.config.atr_period :]))
        last_close = close[-1]
        atr_pct = _safe_div(atr, last_close) if last_close > 0 else 0.0
        self._atr_cache[stock] = atr_pct
        return atr_pct

    def _precompute_median_2d_return(self, stock: str, price_df: pd.DataFrame) -> float:
        """预计算20日滚动2日收益中位数 (S5a时间止损基准).

        Args:
            stock: 股票代码.
            price_df: 该股票历史行情.

        Returns:
            中位数, 0 if 数据不足.
        """
        if stock in self._median_2d_cache:
            return self._median_2d_cache[stock]
        df = price_df[price_df["stock"] == stock].sort_values("date")
        if len(df) < 22:
            self._median_2d_cache[stock] = 0.0
            return 0.0
        close = df["close"].values
        ret_2d = close[2:] / close[:-2] - 1.0
        median_val = float(np.nanmedian(ret_2d[-20:]))
        self._median_2d_cache[stock] = median_val
        return median_val

    def _calc_stop_price_fen(self, entry_fen: int, atr_pct: float, board: str) -> int:
        """V5.2 D-20: S1 止损价 = max(固定值, -1.5*ATR_pct).

        主板: max(-0.04, -1.5*ATR_pct)
        双创: max(-0.06, -1.5*ATR_pct)
        噪音带断言: abs(stop_pct) >= 1.2*ATR_pct

        Args:
            entry_fen: 买入价(分).
            atr_pct: ATR百分比.
            board: "main" 或 "dual".

        Returns:
            止损价(分).
        """
        fixed = (
            self.config.stop_loss_main
            if board == "main"
            else self.config.stop_loss_dual
        )
        atr_stop = -self.config.stop_loss_atr_mult * atr_pct
        stop_pct = max(fixed, atr_stop)
        # 噪音带断言: 止损不能太近
        min_stop = self.config.stop_loss_atr_floor * atr_pct
        if abs(stop_pct) < min_stop:
            stop_pct = -min_stop
        stop_fen = int(entry_fen * (1 + stop_pct))
        return stop_fen

    # ── 买卖检查 ────────────────────────────────────────────

    def _check_buy_eligible(
        self, date: pd.Timestamp, stock: str, row: pd.Series
    ) -> Tuple[bool, str]:
        """检查买入资格 (除价格触发外).

        Returns:
            (是否可买, 原因).
        """
        if row.get("is_halt", 0) == 1:
            return False, "halted"
        if self.config.filter_st and row.get("is_st", 0) == 1:
            return False, "st"

        up_limit = row.get("up_limit", np.nan)
        open_price = row["open"]
        if np.isfinite(up_limit) and open_price >= up_limit:
            return False, "limit_up_open"

        avg_amount = row.get("avg_amount_20d", 0)
        if avg_amount < 50_000_000:
            return False, "low_liquidity"

        circ_mv = row.get("circ_mv", 0)
        if circ_mv < 2_000_000_000:
            return False, "low_circ_mv"

        # 顺势过滤 (默认关闭): T+1 open >= pre_close (T收盘)
        if self.config.filter_trend:
            pre_close = row.get("pre_close", np.nan)
            if np.isfinite(pre_close) and pre_close > 0:
                if open_price < pre_close:
                    return False, "against_trend"

        # B3 追高过滤 (V5.2): 涨幅 <= 7%
        pre_close = row.get("pre_close", np.nan)
        if np.isfinite(pre_close) and pre_close > 0:
            gain_pct = _safe_div(open_price - pre_close, pre_close)
            if gain_pct >= self.config.max_gain_pct:
                return False, "chasing_high"

        return True, "ok"

    def _check_sell_eligible(
        self, date: pd.Timestamp, stock: str, row: pd.Series, holding: Holding
    ) -> Tuple[bool, int, str]:
        """检查是否可卖出.

        Returns:
            (是否可卖, 卖出价fen, 原因).
        """
        if row.get("is_halt", 0) == 1:
            return False, 0, "halted"

        close_fen = self._price_to_fen(row["close"])
        open_fen = self._price_to_fen(row["open"])
        down_limit = row.get("down_limit", np.nan)

        if np.isfinite(down_limit):
            down_limit_fen = self._price_to_fen(down_limit)
            if close_fen <= down_limit_fen:
                return False, 0, "down_limit"

        return True, open_fen, "ok"

    # ── 持仓冲突处理 (Minmin 修正) ──────────────────────────

    def _resolve_position_conflict(
        self,
        candidates: List[dict],
        holdings: List[Holding],
        price_row_map: Dict[str, pd.Series],
    ) -> Tuple[List[dict], List[Tuple[str, str]]]:
        """持仓冲突处理: 比较收益/风险后决定保留或替换.

        策略:
            1. 候选池中有已持仓股票 → 评估新信号 vs 现有持仓
            2. 若新信号 pred_ret + score 更高 → 标记旧持仓卖出, 允许新买入
            3. 若旧持仓更优 → 从候选池移除该股, 顺延选下一名
            4. 基于 "可控风险下总收益最大化" 原则

        Args:
            candidates: 触发成功的候选列表 [{stock, prob, score, pred_ret, entry_price_fen}, ...].
            holdings: 当前持仓列表.
            price_row_map: 当日行情 {stock: Series}.

        Returns:
            (更新后的候选列表, 需卖出的持仓 [(stock, reason), ...]).
        """
        held_stocks = {h.stock: h for h in holdings}
        sells_to_execute: List[Tuple[str, str]] = []
        filtered_candidates: List[dict] = []

        for cand in candidates:
            stock = cand["stock"]
            if stock not in held_stocks:
                filtered_candidates.append(cand)
                continue

            holding = held_stocks[stock]
            row = price_row_map.get(stock)
            if row is None:
                filtered_candidates.append(cand)
                continue

            close_fen = self._price_to_fen(row["close"])
            current_pnl_pct = _safe_div(
                close_fen - holding.entry_price_fen, holding.entry_price_fen
            )

            # 比较预期收益: 新信号 pred_ret vs 现有持仓当前收益
            new_expected = cand["pred_ret"]
            holding_expected = holding.pred_ret + current_pnl_pct

            # 风控评估: 当前持仓是否在水下
            holding_at_risk = current_pnl_pct < 0

            if new_expected > holding_expected + 0.01:
                # 新信号明显更好 → 卖出旧持仓, 允许新买入
                sells_to_execute.append((stock, "replace_signal"))
                filtered_candidates.append(cand)
            elif holding_at_risk and new_expected > 0:
                # 持仓亏损且新信号为正 → 替换
                sells_to_execute.append((stock, "replace_loss"))
                filtered_candidates.append(cand)
            else:
                # 保留旧持仓, 跳过新信号
                logger.debug(
                    "保留持仓 %s (holding=%.4f vs new=%.4f)",
                    stock,
                    holding_expected,
                    new_expected,
                )

        return filtered_candidates, sells_to_execute

    # ── 资金分配 ────────────────────────────────────────────

    def _allocate_capital(
        self,
        candidates: List[Tuple[str, float, float, float]],
        available_cash_fen: int,
    ) -> List[Tuple[str, int, int]]:
        """资金分配: Softmax 加权 + 单只上限 + 一次截断重分配.

        Args:
            candidates: [(stock, prob, score, pred_ret), ...].
            available_cash_fen: 可用资金 (分).

        Returns:
            [(stock, allocated_cash_fen, 0), ...]
        """
        if not candidates or available_cash_fen <= 0:
            return []

        max_per_fen = int(self.config.initial_capital * self._max_pos_pct * 100)

        if self.position_mode == PositionMode.SNIPER_MAX:
            stock = candidates[0][0]
            alloc = min(available_cash_fen, max_per_fen)
            return [(stock, alloc, 0)]

        topk = 5 if self.position_mode == PositionMode.SQUAD else 2
        candidates = candidates[:topk]

        probs = np.array([c[1] for c in candidates])
        weights = _softmax(probs * 2.0)

        # Step 1: 初步分配
        raw_allocs = [int(w * available_cash_fen) for w in weights]

        # Step 2: 截断到单只上限
        capped_excess = 0
        final_allocs = []
        for i, alloc in enumerate(raw_allocs):
            if alloc > max_per_fen:
                capped_excess += alloc - max_per_fen
                final_allocs.append(max_per_fen)
            else:
                final_allocs.append(alloc)

        # Step 3: 一次重分配 (仅一次)
        if capped_excess > 0:
            uncapped_idx = [i for i, a in enumerate(final_allocs) if a < max_per_fen]
            if uncapped_idx:
                remaining_weights = sum(weights[i] for i in uncapped_idx)
                if remaining_weights > 0:
                    for i in uncapped_idx:
                        extra = int(capped_excess * weights[i] / remaining_weights)
                        final_allocs[i] = min(final_allocs[i] + extra, max_per_fen)

        result = []
        for i, (stock, _, _, _) in enumerate(candidates):
            if final_allocs[i] > 0:
                result.append((stock, final_allocs[i], 0))
        return result

    # ── 交易执行 ────────────────────────────────────────────

    def _execute_buy(
        self,
        date: pd.Timestamp,
        stock: str,
        price_fen: int,
        cash_fen: int,
        prob: float,
        pred_ret: float,
        score: float,
        horizon: int,
        amount_fen: int,
        atr_pct: float = 0.0,
        board: str = "main",
        stop_price_fen: int = 0,
        median_2d_return: float = 0.0,
    ) -> Tuple[bool, dict]:
        """执行买入.

        Args:
            amount_fen: 当日成交额 (分), 用于成交量限制.

        Returns:
            (是否成功, 交易记录 dict).
        """
        if price_fen <= 0 or cash_fen <= 0:
            return False, {}

        # 成交量限制: 买入金额 <= 当日成交额 * volume_limit_pct
        max_buy_fen = int(amount_fen * self.config.volume_limit_pct)
        if cash_fen > max_buy_fen:
            cash_fen = max_buy_fen

        # 100股取整
        shares = int(cash_fen / price_fen / 100) * 100
        if shares <= 0:
            return False, {}

        cost_fen = shares * price_fen
        commission_fen = self._calc_commission(cost_fen)
        total_cost_fen = cost_fen + commission_fen

        if total_cost_fen > self.available_cash_fen:
            shares = (
                int(
                    (self.available_cash_fen - self.config.min_commission * 100)
                    / (price_fen * (1 + self.config.commission_rate))
                    / 100
                )
                * 100
            )
            if shares <= 0:
                return False, {}
            cost_fen = shares * price_fen
            commission_fen = self._calc_commission(cost_fen)
            total_cost_fen = cost_fen + commission_fen

        self.available_cash_fen -= total_cost_fen

        holding = Holding(
            stock=stock,
            entry_date=date,
            entry_price_fen=price_fen,
            quantity=shares,
            horizon=horizon,
            mode=self.position_mode.value,
            prob_up=prob,
            pred_ret=pred_ret,
            score=score,
            days_held=0,
            max_close_fen=price_fen,
            stop_price_fen=stop_price_fen,
            atr_pct=atr_pct,
            board=board,
            median_2d_return=median_2d_return,
        )
        self.holdings.append(holding)

        return True, {
            "date": date,
            "stock": stock,
            "side": "buy",
            "price_fen": price_fen,
            "quantity": shares,
            "commission_fen": commission_fen,
            "cost_fen": total_cost_fen,
        }

    def _execute_sell(
        self,
        date: pd.Timestamp,
        stock: str,
        price_fen: int,
        holding: Holding,
        reason: str,
    ) -> Trade:
        """执行卖出. 卖出所得进入 frozen_cash.

        Returns:
            Trade 对象.
        """
        slippage_bp = (
            self.config.slippage_sell_stop_bp
            if reason
            in (
                "stop_loss",
                "stop_profit",
                "trailing_stop",
                "replace_signal",
                "replace_loss",
            )
            else self.config.slippage_sell_moo_bp
        )
        sell_price_fen = int(price_fen * (1000 - slippage_bp) / 1000 + 0.5)

        revenue_fen = holding.quantity * sell_price_fen
        commission_fen = self._calc_commission(revenue_fen)
        stamp_tax_fen = int(revenue_fen * self.config.stamp_tax_rate)
        net_revenue_fen = revenue_fen - commission_fen - stamp_tax_fen

        self.frozen_cash_fen += net_revenue_fen

        pnl_fen = net_revenue_fen - holding.quantity * holding.entry_price_fen
        pnl_pct = _safe_div(sell_price_fen, holding.entry_price_fen) - 1.0
        max_profit_pct = _safe_div(
            holding.max_close_fen - holding.entry_price_fen, holding.entry_price_fen
        )

        trade = Trade(
            entry_date=holding.entry_date,
            exit_date=date,
            stock=stock,
            entry_price_fen=holding.entry_price_fen,
            exit_price_fen=sell_price_fen,
            quantity=holding.quantity,
            pnl_fen=pnl_fen,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            horizon=holding.horizon,
            mode=holding.mode,
            prob_up=holding.prob_up,
            pred_ret=holding.pred_ret,
            score=holding.score,
            max_profit_pct=max_profit_pct,
            commission_entry_fen=0,
            commission_exit_fen=commission_fen,
            stamp_tax_fen=stamp_tax_fen,
        )
        self.trades.append(trade)
        return trade

    # ── 风控检查 ────────────────────────────────────────────

    def _check_risk(
        self,
        date: pd.Timestamp,
        holdings: List[Holding],
        price_row_map: Dict[str, pd.Series],
    ) -> List[Tuple[str, str, int]]:
        """V5.2 开盘前风控检查 (基于前一日收盘价).

        卖出优先级:
            S1 动态止损: 现价 <= stop_price (max(固定, -1.5*ATR))
            S2 移动止盈: 浮盈>=3%后回撤>=max(3%, 1.0*ATR)
            S5a 时间止损: 满2日且2日收益<20日中位数

        Returns:
            [(stock, reason, close_fen), ...]
        """
        sells = []
        for h in holdings:
            if h.stop_loss_triggered or h.stop_profit_triggered:
                continue
            row = price_row_map.get(h.stock)
            if row is None:
                continue

            close_fen = self._price_to_fen(row["close"])

            # S1 动态止损 (V5.2 D-20): 现价 <= stop_price_fen
            if h.stop_price_fen > 0 and close_fen <= h.stop_price_fen:
                sells.append((h.stock, "stop_loss", close_fen))
                h.stop_loss_triggered = True
                continue

            # S2 移动止盈 (V5.2): 浮盈>=3%激活, 回撤>=max(3%, 1*ATR)
            if self.config.trailing_stop and h.max_close_fen > h.entry_price_fen:
                profit_pct = _safe_div(
                    h.max_close_fen - h.entry_price_fen, h.entry_price_fen
                )
                if profit_pct >= self.config.trailing_stop_activate:
                    retrace_pct = _safe_div(
                        h.max_close_fen - close_fen, h.max_close_fen
                    )
                    # 回撤带: max(3%, 1.0*ATR_pct)
                    retrace_threshold = max(
                        self.config.trailing_stop_min_pct,
                        self.config.trailing_stop_atr_mult * h.atr_pct,
                    )
                    if retrace_pct >= retrace_threshold:
                        sells.append((h.stock, "trailing_stop", close_fen))
                        h.stop_profit_triggered = True
                        continue

            # S5a 时间止损 (V5.2 D-21): 满2日且2日收益<中位数
            if h.days_held >= self.config.time_stop_days:
                ret_2d = _safe_div(close_fen - h.entry_price_fen, h.entry_price_fen)
                if self.config.time_stop_use_median:
                    threshold = h.median_2d_return
                else:
                    threshold = self.config.time_stop_fixed_threshold
                if ret_2d < threshold:
                    sells.append((h.stock, "time_stop", close_fen))
                    continue

        return sells

    def _update_holdings(
        self, date: pd.Timestamp, price_row_map: Dict[str, pd.Series]
    ) -> None:
        """更新持仓: days_held +1, max_close_fen 更新 (用收盘价)."""
        for h in self.holdings:
            row = price_row_map.get(h.stock)
            if row is None:
                continue
            close_fen = self._price_to_fen(row["close"])
            if close_fen > h.max_close_fen:
                h.max_close_fen = close_fen
            h.days_held += 1

    # ── 对账与检查点 ────────────────────────────────────────

    def _calc_market_value(self, price_row_map: Dict[str, pd.Series]) -> int:
        """计算持仓市值 (分)."""
        mv = 0
        for h in self.holdings:
            row = price_row_map.get(h.stock)
            if row is not None:
                close_fen = self._price_to_fen(row["close"])
                mv += h.quantity * close_fen
        return mv

    def _reconcile(
        self, date: pd.Timestamp, price_row_map: Dict[str, pd.Series]
    ) -> bool:
        """每日对账: assert 总资产 = 现金 + 市值, 偏差 < 1分."""
        mv = self._calc_market_value(price_row_map)
        # 对账只检查非负
        if self.available_cash_fen < 0 or self.frozen_cash_fen < 0:
            logger.error(
                "对账失败 %s: cash=%d, frozen=%d, mv=%d",
                date,
                self.available_cash_fen,
                self.frozen_cash_fen,
                mv,
            )
            return False
        return True

    def _save_checkpoint(self, date: pd.Timestamp) -> str:
        """保存检查点 (JSON)."""
        return f"checkpoint_{date.strftime('%Y%m%d')}"

    # ── 主回测循环 ──────────────────────────────────────────

    def run(
        self,
        horizon: int = 2,
        score_col: str = "score_h2",
        prob_col: str = "prob_up_h2",
        pred_col: str = "pred_ret_h2",
        topk: int = 5,
    ) -> pd.DataFrame:
        """运行回测.

        Args:
            horizon: 持有期 (1, 2, 4).
            score_col: 排序列.
            prob_col: 概率列.
            pred_col: 预测收益列.
            topk: 每日选股上限.

        Returns:
            DataFrame: 每日记录 [date, nav, cash, market_value, ...].
        """
        self._reset_state()

        pred_by_date: Dict[pd.Timestamp, list] = {}
        for _, row in self.pred_df.iterrows():
            d = row["date"]
            pred_by_date.setdefault(d, []).append(row)

        logger.info(
            "回测开始: horizon=%d, mode=%s, %d 交易日",
            horizon,
            self.position_mode.value,
            len(self.trade_dates),
        )

        for i in range(1, len(self.trade_dates)):
            today = self.trade_dates[i]
            yesterday = self.trade_dates[i - 1]

            today_prices = self.price_df[self.price_df["date"] == today]
            price_row_map = {r["stock"]: r for _, r in today_prices.iterrows()}

            yesterday_prices = self.price_df[self.price_df["date"] == yesterday]
            yesterday_price_map = {
                r["stock"]: r for _, r in yesterday_prices.iterrows()
            }

            trades_today = 0
            stocks_to_remove: set[str] = set()

            # ── 1. 开盘前: 风控检查 (基于昨日收盘) ──
            pending_sells: List[Tuple[str, str, int]] = []
            if self.holdings:
                pending_sells = self._check_risk(
                    today, self.holdings, yesterday_price_map
                )

            # ── 2. 开盘: 执行风控卖出 (MOO) ──
            for stock, reason, _ in pending_sells:
                h = next((x for x in self.holdings if x.stock == stock), None)
                if h is None or stock in stocks_to_remove:
                    continue
                row = price_row_map.get(stock)
                if row is None:
                    continue
                can_sell, sell_open_fen, _ = self._check_sell_eligible(
                    today, stock, row, h
                )
                if can_sell:
                    self._execute_sell(today, stock, sell_open_fen, h, reason)
                    stocks_to_remove.add(stock)
                    trades_today += 1

            # 移除已卖出的持仓
            self.holdings = [
                h for h in self.holdings if h.stock not in stocks_to_remove
            ]

            # ── 3. 到期卖出 S5b (收盘执行, 在步骤5处理) ──

            # ── 4. 买入 ──
            signal_rows = pred_by_date.get(yesterday, [])
            if (
                signal_rows
                and not self.stop_new_positions
                and self.cooldown_days == 0
                and not self._system_halted
            ):
                signal_sorted = sorted(
                    signal_rows,
                    key=lambda r: r.get(score_col, -np.inf),
                    reverse=True,
                )
                signal_filtered = [
                    r
                    for r in signal_sorted
                    if r.get(prob_col, 0) >= self.config.prob_threshold
                ]

                candidates: List[dict] = []
                for r in signal_filtered:
                    stock = r["stock"]
                    row = price_row_map.get(stock)
                    if row is None:
                        continue

                    eligible, _ = self._check_buy_eligible(today, stock, row)
                    if not eligible:
                        continue

                    open_fen = self._price_to_fen(row["open"])
                    high_fen = self._price_to_fen(row["high"])
                    up_limit_fen = self._price_to_fen(row.get("up_limit", 0))

                    trigger_fen = self._calc_trigger_price(open_fen)

                    # 涨停过滤: trigger >= up_limit → 无法买入
                    if up_limit_fen > 0 and trigger_fen >= up_limit_fen:
                        continue

                    if high_fen < trigger_fen:
                        continue

                    entry_fen = self._calc_entry_price(
                        trigger_fen, self.config.slippage_buy_bp
                    )
                    if entry_fen > high_fen:
                        entry_fen = high_fen
                    if up_limit_fen > 0 and entry_fen >= up_limit_fen:
                        continue

                    prob = r.get(prob_col, 0)
                    pred_ret = r.get(pred_col, 0)
                    score = r.get(score_col, 0)
                    amount_fen = self._price_to_fen(row.get("amount", 0))

                    # V5.2: 预计算 ATR / 板块 / 中位数
                    atr_pct = self._precompute_atr(stock, self.price_df)
                    board = self._get_board(stock)
                    median_2d = self._precompute_median_2d_return(stock, self.price_df)
                    stop_price_fen = self._calc_stop_price_fen(
                        entry_fen, atr_pct, board
                    )

                    # B7 止损距离否决 (V5.2): 距stop_price < 1.2*ATR → 放弃
                    stop_distance = _safe_div(entry_fen - stop_price_fen, entry_fen)
                    if (
                        atr_pct > 0
                        and stop_distance < self.config.stop_distance_atr_mult * atr_pct
                    ):
                        logger.debug(
                            "B7否决 %s: 止损距离=%.4f < %.4f",
                            stock,
                            stop_distance,
                            self.config.stop_distance_atr_mult * atr_pct,
                        )
                        continue

                    candidates.append(
                        {
                            "stock": stock,
                            "prob": prob,
                            "score": score,
                            "pred_ret": pred_ret,
                            "entry_price_fen": entry_fen,
                            "amount_fen": amount_fen,
                            "atr_pct": atr_pct,
                            "board": board,
                            "stop_price_fen": stop_price_fen,
                            "median_2d_return": median_2d,
                        }
                    )

                self._daily_total += 1

                # 持仓冲突处理 (Minmin 修正)
                candidates, conflict_sells = self._resolve_position_conflict(
                    candidates, self.holdings, price_row_map
                )
                for stock, reason in conflict_sells:
                    h = next((x for x in self.holdings if x.stock == stock), None)
                    if h is None or stock in stocks_to_remove:
                        continue
                    row = price_row_map.get(stock)
                    if row is None:
                        continue
                    can_sell, sell_open_fen, _ = self._check_sell_eligible(
                        today, stock, row, h
                    )
                    if can_sell:
                        self._execute_sell(today, stock, sell_open_fen, h, reason)
                        stocks_to_remove.add(stock)
                        trades_today += 1

                self.holdings = [
                    h for h in self.holdings if h.stock not in stocks_to_remove
                ]

                # 候选不足检查
                mode_min = self.config.min_tradeable
                if self.position_mode == PositionMode.SNIPER:
                    mode_min = 1
                elif self.position_mode == PositionMode.SNIPER_MAX:
                    mode_min = 1

                if len(candidates) >= mode_min:
                    self._daily_tradeable += 1
                    alloc_input = [
                        (c["stock"], c["prob"], c["score"], c["pred_ret"])
                        for c in candidates
                    ]
                    allocations = self._allocate_capital(
                        alloc_input, self.available_cash_fen
                    )

                    alloc_map = {a[0]: a[1] for a in allocations}
                    for cand in candidates:
                        stock = cand["stock"]
                        if stock not in alloc_map:
                            continue
                        cash_for_stock = alloc_map[stock]
                        self._execute_buy(
                            today,
                            stock,
                            cand["entry_price_fen"],
                            cash_for_stock,
                            cand["prob"],
                            cand["pred_ret"],
                            cand["score"],
                            horizon,
                            cand["amount_fen"],
                            cand.get("atr_pct", 0.0),
                            cand.get("board", "main"),
                            cand.get("stop_price_fen", 0),
                            cand.get("median_2d_return", 0.0),
                        )
                        trades_today += 1

            # ── 4a. 系统停机: 清仓所有可卖持仓 ──
            if self._system_halted:
                for h in self.holdings:
                    row = price_row_map.get(h.stock)
                    if row is None:
                        continue
                    can_sell, _, _ = self._check_sell_eligible(today, h.stock, row, h)
                    if can_sell:
                        close_fen = self._price_to_fen(row["close"])
                        self._execute_sell(today, h.stock, close_fen, h, "system_halt")
                        stocks_to_remove.add(h.stock)
                        trades_today += 1
                self.holdings = [
                    h for h in self.holdings if h.stock not in stocks_to_remove
                ]

            # ── 5. 收盘: 到期卖出 ──
            maturity_sells: List[Holding] = []
            effective_period = (
                self.config.holding_period
                if self.config.holding_period > 0
                else horizon
            )
            for h in self.holdings:
                if h.days_held >= effective_period:
                    maturity_sells.append(h)

            for h in maturity_sells:
                row = price_row_map.get(h.stock)
                if row is None:
                    continue
                can_sell, _, sell_reason = self._check_sell_eligible(
                    today, h.stock, row, h
                )
                if can_sell:
                    close_fen = self._price_to_fen(row["close"])
                    self._execute_sell(today, h.stock, close_fen, h, "maturity")
                    stocks_to_remove.add(h.stock)
                    trades_today += 1
                # 跌停/停牌 → 顺延 (不删除, 下个交易日再尝试)

            self.holdings = [
                h for h in self.holdings if h.stock not in stocks_to_remove
            ]

            # ── 6. 更新持仓 (收盘价) ──
            self._update_holdings(today, price_row_map)

            # ── 7. 计算市值与NAV ──
            market_value = self._calc_market_value(price_row_map)
            total_cash = self.available_cash_fen + self.frozen_cash_fen
            nav_fen = total_cash + market_value
            nav = self._fen_to_yuan(nav_fen)

            # ── 8. 当日盈亏检查 ──
            prev_nav_fen = (
                int(self.config.initial_capital * 100)
                if not self.daily_records
                else int(self.daily_records[-1]["nav"] * 100)
            )
            daily_pnl_fen = nav_fen - prev_nav_fen
            daily_pnl_pct = _safe_div(daily_pnl_fen, prev_nav_fen)

            # 连续亏损按交易日计
            if daily_pnl_fen < 0:
                self._daily_loss_days += 1
                self.consecutive_loss_days += 1
                if self.consecutive_loss_days >= self.config.consecutive_loss_limit:
                    self.cooldown_days = self.config.consecutive_loss_cooldown
                    self.consecutive_loss_days = 0
                    logger.warning(
                        "连续亏损 %d 日, 进入 %d 天冷却",
                        self.config.consecutive_loss_limit,
                        self.cooldown_days,
                    )
            else:
                self.consecutive_loss_days = 0

            # V5.2 D-22 日保险丝: 2σ自适应 + 4%固定兜底 (双轨)
            self._daily_pnl_history.append(daily_pnl_pct)
            fuse_triggered = False
            # 固定兜底
            if daily_pnl_pct <= self.config.daily_fuse_fixed:
                fuse_triggered = True
            # 2σ自适应
            if self.config.daily_fuse_use_sigma and len(self._daily_pnl_history) >= 5:
                window = self._daily_pnl_history[-self.config.daily_fuse_window :]
                mu = float(np.mean(window))
                sigma = float(np.std(window))
                lower_bound = mu - self.config.daily_fuse_sigma * sigma
                if daily_pnl_pct < lower_bound:
                    fuse_triggered = True

            self.stop_new_positions = fuse_triggered

            # V5.2 系统停机线: 总资金回撤 >= 15%
            if nav_fen < self._peak_nav_fen:
                drawdown = _safe_div(nav_fen - self._peak_nav_fen, self._peak_nav_fen)
                if drawdown <= self.config.system_halt_drawdown:
                    self._system_halted = True
                    logger.error(
                        "系统停机触发: 回撤=%.2f%%, 清仓所有可卖持仓",
                        drawdown * 100,
                    )
            else:
                self._peak_nav_fen = nav_fen

            # ── 9. frozen_cash → available_cash (T+1交收) ──
            self.available_cash_fen += self.frozen_cash_fen
            self.frozen_cash_fen = 0

            # ── 10. 现金计息 ──
            daily_rate = self.config.cash_interest_rate / _TRADING_DAYS_PER_YEAR
            interest_fen = int(self.available_cash_fen * daily_rate + 0.5)
            self.available_cash_fen += interest_fen

            # ── 11. 冷却倒计时 ──
            if self.cooldown_days > 0:
                self.cooldown_days -= 1

            # ── 12. 对账 ──
            self._reconcile(today, price_row_map)

            # ── 13. 记录 ──
            self.daily_records.append(
                {
                    "date": today,
                    "nav": nav,
                    "cash": self._fen_to_yuan(self.available_cash_fen),
                    "market_value": self._fen_to_yuan(market_value),
                    "daily_pnl": self._fen_to_yuan(daily_pnl_fen),
                    "daily_pnl_pct": daily_pnl_pct,
                    "num_holdings": len(self.holdings),
                    "num_trades_today": trades_today,
                    "stop_flag": self.stop_new_positions,
                    "cooldown_days": self.cooldown_days,
                }
            )

            # 持仓快照
            for h in self.holdings:
                row = price_row_map.get(h.stock)
                close_fen = self._price_to_fen(row["close"]) if row is not None else 0
                self.holdings_history.append(
                    {
                        "date": today,
                        "stock": h.stock,
                        "quantity": h.quantity,
                        "entry_price": self._fen_to_yuan(h.entry_price_fen),
                        "close_price": self._fen_to_yuan(close_fen),
                        "days_held": h.days_held,
                        "pnl_pct": _safe_div(
                            close_fen - h.entry_price_fen, h.entry_price_fen
                        ),
                    }
                )

        logger.info(
            "回测完成: %d 交易日, %d 笔交易, 最终NAV=%.2f",
            len(self.daily_records),
            len(self.trades),
            nav,
        )
        return pd.DataFrame(self.daily_records)

    # ── 结果输出 ────────────────────────────────────────────

    def get_trades(self) -> pd.DataFrame:
        """获取所有交易明细."""
        if not self.trades:
            return pd.DataFrame()
        records = []
        for t in self.trades:
            records.append(
                {
                    "entry_date": t.entry_date,
                    "exit_date": t.exit_date,
                    "stock": t.stock,
                    "entry_price": self._fen_to_yuan(t.entry_price_fen),
                    "exit_price": self._fen_to_yuan(t.exit_price_fen),
                    "quantity": t.quantity,
                    "pnl": self._fen_to_yuan(t.pnl_fen),
                    "pnl_pct": t.pnl_pct,
                    "exit_reason": t.exit_reason,
                    "horizon": t.horizon,
                    "mode": t.mode,
                    "prob_up": t.prob_up,
                    "pred_ret": t.pred_ret,
                    "score": t.score,
                    "max_profit_pct": t.max_profit_pct,
                    "is_swap": t.is_swap,
                }
            )
        return pd.DataFrame(records)

    def get_holdings_history(self) -> pd.DataFrame:
        """获取每日持仓历史."""
        if not self.holdings_history:
            return pd.DataFrame()
        return pd.DataFrame(self.holdings_history)

    def get_metrics(self) -> Dict:
        """计算绩效指标.

        Returns:
            dict: 含 total_return, sharpe, max_drawdown, win_rate 等.
        """
        if not self.daily_records:
            return self._empty_metrics()

        df = pd.DataFrame(self.daily_records)
        nav = df["nav"].values
        daily_returns = df["daily_pnl_pct"].values

        total_return = _safe_div(
            nav[-1] - self.config.initial_capital, self.config.initial_capital
        )
        annual_return = (1 + total_return) ** (
            _safe_div(_TRADING_DAYS_PER_YEAR, len(nav))
        ) - 1
        annual_vol = (
            float(np.std(daily_returns) * np.sqrt(_TRADING_DAYS_PER_YEAR))
            if len(daily_returns) > 1
            else 0.0
        )
        sharpe = _safe_div(
            float(np.mean(daily_returns)), float(np.std(daily_returns))
        ) * np.sqrt(_TRADING_DAYS_PER_YEAR)

        equity = pd.Series(nav)
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

        # 回撤持续期
        dd_duration = 0
        current_dd = 0
        for dd in drawdown:
            if dd < 0:
                current_dd += 1
                dd_duration = max(dd_duration, current_dd)
            else:
                current_dd = 0

        calmar = _safe_div(annual_return, abs(max_dd))

        trades_df = self.get_trades()
        if not trades_df.empty:
            wins = trades_df[trades_df["pnl"] > 0]
            losses = trades_df[trades_df["pnl"] < 0]
            win_rate = _safe_div(len(wins), len(trades_df))
            avg_win = float(wins["pnl"].mean()) if len(wins) > 0 else 0.0
            avg_loss = float(abs(losses["pnl"].mean())) if len(losses) > 0 else 0.0
            profit_loss_ratio = _safe_div(avg_win, avg_loss)
            max_win = float(trades_df["pnl"].max())
            max_loss = float(trades_df["pnl"].min())
        else:
            win_rate = 0.0
            profit_loss_ratio = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            max_win = 0.0
            max_loss = 0.0

        daily_tradeable_rate = _safe_div(
            self._daily_tradeable, max(self._daily_total, 1)
        )

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "max_drawdown_duration": dd_duration,
            "calmar_ratio": calmar,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_win": max_win,
            "max_loss": max_loss,
            "num_trades": len(self.trades),
            "num_winning_trades": len(wins) if not trades_df.empty else 0,
            "num_losing_trades": len(losses) if not trades_df.empty else 0,
            "concentration_risk_ratio": 0.0,  # 由 ComparativeAnalyzer 填充
            "daily_tradeable_rate": daily_tradeable_rate,
            "config_hash": self.config_hash,
            "data_version_hash": self.data_version_hash,
        }

    @staticmethod
    def _empty_metrics() -> Dict:
        """空绩效指标."""
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration": 0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "profit_loss_ratio": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "num_trades": 0,
            "num_winning_trades": 0,
            "num_losing_trades": 0,
            "concentration_risk_ratio": 0.0,
            "daily_tradeable_rate": 0.0,
            "config_hash": "",
            "data_version_hash": "",
        }
