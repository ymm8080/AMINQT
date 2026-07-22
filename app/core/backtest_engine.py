# -*- coding: utf-8 -*-
"""回测引擎 (P9, ARCH §7).

职责:
1. 模型/因子/规则的历史回测 (Train/Val/Test/OOS 严格按时间切分)
2. 输出 GAP 指标供 AdaptiveEngine 自适应调整 (selection/trading/factor/risk GAP)
3. 控盘比例专项: ctrl_ratio_ic / 周线上升组合收益 / 多空收益 (ARCH §5.13.8.C)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    """回测结果."""

    total_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    ic: float = 0.0
    icir: float = 0.0
    # 控盘比例专项 (ARCH §5.13.8.C)
    ctrl_ratio_ic: float = 0.0
    ctrl_ratio_icir: float = 0.0
    ctrl_ratio_weekly_rising_return: float = 0.0
    ctrl_ratio_weekly_falling_return: float = 0.0
    ctrl_ratio_long_short_return: float = 0.0
    # GAP 输出 (供 AdaptiveEngine, ARCH §5.17)
    gaps: Dict[str, float] = field(default_factory=dict)


def _safe_div(numerator: float, denominator: float) -> float:
    """安全除法: 分母为 0/NaN 时返回 0."""
    if denominator == 0 or not np.isfinite(denominator):
        return 0.0
    return float(np.nan_to_num(numerator / denominator, nan=0.0,
                               posinf=0.0, neginf=0.0))


class BacktestEngine:
    """回测引擎 (简单做多日线回测).

    输入: {symbol: DataFrame}, 含 close 列, 可选 open / signal(布尔) /
    score(评分) / ctrl_ratio / ctrl_weekly_rising(布尔) 列, 索引或
    date 列为交易日期。

    规则:
        - 买入: signal 列存在时按 signal=True, 否则按 score > score_threshold
        - 成交: 信号日 close 买入 (execution="close", 默认) 或次日 open
          买入 (execution="next_open"); 持有 holding_days 日后 close 卖出
        - T+1: holding_days < 1 时强制为 1 (A 股当日买入不可当日卖出)
        - 组合: 每日等权持有所有信号股, 日收益为信号股前向收益均值
        - IC: 每个交易日截面 score 与 holding_days 日前向收益的 Spearman
          相关 (pandas rank corr, 不依赖 scipy); 前向收益为标签, 不构成
          因子未来函数
    """

    def __init__(self, config: dict = None) -> None:
        """加载回测配置.

        Args:
            config: 配置字典, 支持键:
                holding_days (int, 默认 1) / score_threshold (float, 默认 0.5)
                score_col / signal_col / ctrl_col / ctrl_rising_col (列名)
                execution ("close"/"next_open")
                targets (dict: {target_sharpe, target_total_return,
                                target_ic, target_max_drawdown})
                data (dict, 可选): {symbol: DataFrame} 供 run() 缺省使用。
        """
        self.config = config or {}
        self.holding_days = max(1, int(self.config.get("holding_days", 1)))
        self.score_threshold = float(self.config.get("score_threshold", 0.5))
        self.score_col = self.config.get("score_col", "score")
        self.signal_col = self.config.get("signal_col", "signal")
        self.ctrl_col = self.config.get("ctrl_col", "ctrl_ratio")
        self.ctrl_rising_col = self.config.get(
            "ctrl_rising_col", "ctrl_weekly_rising")
        self.execution = self.config.get("execution", "close")
        self.targets: Dict[str, float] = self.config.get("targets", {})
        logger.info("BacktestEngine 初始化: holding_days=%d, execution=%s",
                    self.holding_days, self.execution)

    # ── 数据准备 ────────────────────────────────────────────

    def _build_panel(self, data: Dict[str, pd.DataFrame],
                     start: str, end: str) -> pd.DataFrame:
        """构建长表面板: date/symbol/close/fwd_ret/score/signal/ctrl 列.

        前向收益 fwd_ret = close[t+holding_days] / close[t] - 1 (标签)。
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        frames: List[pd.DataFrame] = []
        for symbol, df in data.items():
            if df is None or df.empty or "close" not in df.columns:
                logger.warning("跳过无 close 列的数据: %s", symbol)
                continue
            d = df.copy()
            if "date" in d.columns:
                d["date"] = pd.to_datetime(d["date"])
            else:
                d["date"] = pd.to_datetime(d.index)
            d = d.sort_values("date").reset_index(drop=True)
            d = d[(d["date"] >= start_ts) & (d["date"] <= end_ts)]
            if d.empty:
                continue
            d["symbol"] = symbol
            # 标签: holding_days 日前向收益 (允许使用未来价格 — 仅作标签)
            d["fwd_ret"] = d["close"].shift(-self.holding_days) / d["close"] - 1.0
            if self.signal_col in d.columns:
                d["_signal"] = d[self.signal_col].fillna(False).astype(bool)
            elif self.score_col in d.columns:
                d["_signal"] = d[self.score_col] > self.score_threshold
            else:
                d["_signal"] = False
            if self.score_col not in d.columns:
                d[self.score_col] = np.nan
            frames.append(d)
        if not frames:
            return pd.DataFrame()
        panel = pd.concat(frames, ignore_index=True)
        return panel

    # ── 绩效指标 ────────────────────────────────────────────

    @staticmethod
    def _portfolio_metrics(daily_ret: pd.Series) -> Dict[str, float]:
        """由日收益序列计算 total_return/sharpe/max_drawdown."""
        daily_ret = daily_ret.dropna()
        if daily_ret.empty:
            return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
        equity = (1.0 + daily_ret).cumprod()
        total_return = float(equity.iloc[-1] - 1.0)
        sharpe = _safe_div(float(daily_ret.mean()), float(daily_ret.std())) \
            * np.sqrt(_TRADING_DAYS_PER_YEAR)
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
        return {
            "total_return": float(np.nan_to_num(total_return, nan=0.0)),
            "sharpe": float(np.nan_to_num(sharpe, nan=0.0)),
            "max_drawdown": max_drawdown,
        }

    @staticmethod
    def _ic_series(panel: pd.DataFrame, factor_col: str) -> pd.Series:
        """逐日截面 Spearman IC (factor vs fwd_ret, pandas rank corr)."""
        ics: Dict[pd.Timestamp, float] = {}
        for date, grp in panel.groupby("date"):
            sub = grp[[factor_col, "fwd_ret"]].dropna()
            if len(sub) < 2 or sub[factor_col].nunique() < 2 \
                    or sub["fwd_ret"].nunique() < 2:
                continue
            corr = sub[factor_col].corr(sub["fwd_ret"], method="spearman")
            if np.isfinite(corr):
                ics[date] = float(corr)
        return pd.Series(ics, dtype=float)

    @classmethod
    def _ic_icir(cls, panel: pd.DataFrame, factor_col: str):
        """返回 (IC 均值, ICIR)."""
        ics = cls._ic_series(panel, factor_col)
        if ics.empty:
            return 0.0, 0.0
        ic = float(ics.mean())
        icir = _safe_div(ic, float(ics.std()))
        return ic, icir

    # ── 主流程 ──────────────────────────────────────────────

    def run(self, strategy: str, symbols: List[str],
            start: str, end: str,
            data: Optional[Dict[str, pd.DataFrame]] = None) -> BacktestResult:
        """执行回测.

        Args:
            strategy: 策略名 (selection/trading/factor/rule), 仅作记录。
            symbols: 回测标的。
            start: 开始日期 YYYY-MM-DD。
            end: 结束日期 YYYY-MM-DD。
            data: {symbol: DataFrame}; None 时用 config["data"]。

        Returns:
            BacktestResult 含收益/IC/GAP 指标。
        """
        data = data if data is not None else self.config.get("data", {})
        subset = {s: data[s] for s in symbols if s in data}
        logger.info("回测开始: strategy=%s, %d/%d 标的, %s ~ %s",
                    strategy, len(subset), len(symbols), start, end)
        result = BacktestResult()
        panel = self._build_panel(subset, start, end)
        if panel.empty:
            logger.warning("回测面板为空, 返回零指标")
            result.gaps = self.compute_gaps(result)
            return result

        # 组合日收益: 信号股前向收益等权均值 (无信号日收益 0)
        signaled = panel[panel["_signal"]]
        daily_ret = signaled.groupby("date")["fwd_ret"].mean()
        all_dates = pd.Series(panel["date"].unique())
        daily_ret = daily_ret.reindex(all_dates, fill_value=0.0)
        metrics = self._portfolio_metrics(daily_ret)
        result.total_return = metrics["total_return"]
        result.sharpe = metrics["sharpe"]
        result.max_drawdown = metrics["max_drawdown"]

        # 因子 IC/ICIR
        if panel[self.score_col].notna().any():
            result.ic, result.icir = self._ic_icir(panel, self.score_col)

        # 控盘比例专项 (ARCH §5.13.8.C)
        if self.ctrl_col in panel.columns and panel[self.ctrl_col].notna().any():
            result.ctrl_ratio_ic, result.ctrl_ratio_icir = \
                self._ic_icir(panel, self.ctrl_col)
        if self.ctrl_rising_col in panel.columns:
            labeled = panel.dropna(subset=["fwd_ret"])
            rising = labeled.loc[labeled[self.ctrl_rising_col] == True, "fwd_ret"]  # noqa: E712
            falling = labeled.loc[labeled[self.ctrl_rising_col] == False, "fwd_ret"]  # noqa: E712
            rising_ret = float(rising.mean()) if len(rising) else 0.0
            falling_ret = float(falling.mean()) if len(falling) else 0.0
            result.ctrl_ratio_weekly_rising_return = float(
                np.nan_to_num(rising_ret, nan=0.0))
            result.ctrl_ratio_weekly_falling_return = float(
                np.nan_to_num(falling_ret, nan=0.0))
            # 多空: 做多周线上升 / 做空周线下降
            result.ctrl_ratio_long_short_return = (
                result.ctrl_ratio_weekly_rising_return
                - result.ctrl_ratio_weekly_falling_return)

        result.gaps = self.compute_gaps(result)
        logger.info("回测完成: total_return=%.4f, sharpe=%.4f, "
                    "max_drawdown=%.4f, ic=%.4f",
                    result.total_return, result.sharpe,
                    result.max_drawdown, result.ic)
        return result

    def compute_gaps(self, result: BacktestResult) -> Dict[str, float]:
        """计算回测 GAP (实际 vs 目标的差距), 供 AdaptiveEngine 调参.

        GAP 统一为 actual - target (正=优于目标, 负=低于目标);
        max_drawdown 取绝对值后比较 (回撤越小越好, 故 GAP 取
        target - |actual|)。

        Args:
            result: 回测结果。

        Returns:
            {"sharpe_gap": ..., "total_return_gap": ..., "ic_gap": ...,
             "max_drawdown_gap": ...} (仅含 config.targets 中配置的项)。
        """
        mapping = {
            "target_sharpe": ("sharpe_gap", result.sharpe, 1),
            "target_total_return": ("total_return_gap", result.total_return, 1),
            "target_ic": ("ic_gap", result.ic, 1),
            "target_max_drawdown": (
                "max_drawdown_gap", abs(result.max_drawdown), -1),
        }
        gaps: Dict[str, float] = {}
        for target_key, (gap_key, actual, sign) in mapping.items():
            if target_key not in self.targets:
                continue
            target = float(self.targets[target_key])
            gap = (actual - target) * sign
            gaps[gap_key] = float(np.nan_to_num(gap, nan=0.0))
        return gaps
