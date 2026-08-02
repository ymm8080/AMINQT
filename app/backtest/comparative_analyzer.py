# -*- coding: utf-8 -*-
"""模块6: ComparativeAnalyzer — 对比分析器.

对比 Squad vs Sniper 模式的绩效差异.
计算集中度风险系数、Jensen Alpha 等指标.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


def _safe_div(num: float, den: float) -> float:
    """安全除法."""
    if den == 0 or not np.isfinite(den):
        return 0.0
    return float(np.nan_to_num(num / den, nan=0.0))


class ComparativeAnalyzer:
    """对比分析器: Squad vs Sniper 模式绩效对比."""

    def __init__(
        self,
        squad_result: pd.DataFrame,
        sniper_result: pd.DataFrame,
        squad_trades: pd.DataFrame,
        sniper_trades: pd.DataFrame,
        squad_metrics: Dict,
        sniper_metrics: Dict,
        benchmark_df: pd.DataFrame | None = None,
    ):
        """初始化对比分析器.

        Args:
            squad_result: Squad 模式每日 NAV 记录.
            sniper_result: Sniper 模式每日 NAV 记录.
            squad_trades: Squad 交易明细.
            sniper_trades: Sniper 交易明细.
            squad_metrics: Squad 绩效指标.
            sniper_metrics: Sniper 绩效指标.
            benchmark_df: 基准数据 [date, hs300_close, csi1000_close].
        """
        self.squad_result = squad_result
        self.sniper_result = sniper_result
        self.squad_trades = squad_trades
        self.sniper_trades = sniper_trades
        self.squad_metrics = squad_metrics
        self.sniper_metrics = sniper_metrics
        self.benchmark_df = benchmark_df

    def calc_concentration_risk_ratio(self) -> float:
        """集中度风险系数 = Sniper 最大回撤 / Squad 最大回撤.

        Returns:
            比值. >1 表示 Sniper 风险更高.
        """
        squad_dd = abs(self.squad_metrics.get("max_drawdown", 0))
        sniper_dd = abs(self.sniper_metrics.get("max_drawdown", 0))
        if squad_dd == 0:
            return 0.0 if sniper_dd == 0 else float("inf")
        return _safe_div(sniper_dd, squad_dd)

    def calc_jensen_alpha(self, result: pd.DataFrame, benchmark_col: str = "csi1000") -> float:
        """计算 Jensen Alpha (相对基准).

        Alpha = 策略年化收益 - (无风险利率 + Beta × (基准年化收益 - 无风险利率))

        Args:
            result: 每日 NAV 记录.
            benchmark_col: 基准列名 (hs300 / csi1000).

        Returns:
            Jensen Alpha.
        """
        if result.empty or self.benchmark_df is None or self.benchmark_df.empty:
            return 0.0

        nav = result["nav"].values
        if len(nav) < 2:
            return 0.0

        strategy_returns = np.diff(nav) / nav[:-1]

        # 基准收益
        bench_col_map = {"hs300": "hs300_close", "csi1000": "csi1000_close"}
        bench_col = bench_col_map.get(benchmark_col, "csi1000_close")
        if bench_col not in self.benchmark_df.columns:
            return 0.0

        bench = self.benchmark_df.sort_values("date").reset_index(drop=True)
        bench_nav = bench[bench_col].values
        if len(bench_nav) < 2:
            return 0.0
        bench_returns = np.diff(bench_nav) / bench_nav[:-1]

        # 对齐长度
        min_len = min(len(strategy_returns), len(bench_returns))
        if min_len < 2:
            return 0.0

        strategy_returns = strategy_returns[:min_len]
        bench_returns = bench_returns[:min_len]

        # Beta
        cov_matrix = np.cov(strategy_returns, bench_returns)
        beta = _safe_div(cov_matrix[0, 1], cov_matrix[1, 1]) if cov_matrix[1, 1] != 0 else 0.0

        # 年化收益
        strategy_annual = (1 + np.mean(strategy_returns)) ** _TRADING_DAYS_PER_YEAR - 1
        bench_annual = (1 + np.mean(bench_returns)) ** _TRADING_DAYS_PER_YEAR - 1
        rf = 0.02  # 无风险利率 2%

        alpha = strategy_annual - (rf + beta * (bench_annual - rf))
        return float(alpha)

    def compare_nav_curves(self) -> str:
        """对比 Squad vs Sniper 资金曲线 (不绘图, 返回数据描述).

        Returns:
            "squad_final_nav=X, sniper_final_nav=Y, benchmark=Z"
        """
        squad_final = float(self.squad_result["nav"].iloc[-1]) if not self.squad_result.empty else 0.0
        sniper_final = float(self.sniper_result["nav"].iloc[-1]) if not self.sniper_result.empty else 0.0
        bench_final = 0.0
        if self.benchmark_df is not None and not self.benchmark_df.empty:
            if "csi1000_close" in self.benchmark_df.columns:
                bench_final = float(self.benchmark_df["csi1000_close"].iloc[-1])

        return f"squad_final_nav={squad_final:.2f}, sniper_final_nav={sniper_final:.2f}, benchmark={bench_final:.2f}"

    def generate_comparison_report(self) -> Dict:
        """生成对比报告.

        Returns:
            dict: {
                'concentration_risk_ratio': float,
                'squad_sharpe': float,
                'sniper_sharpe': float,
                'squad_max_dd': float,
                'sniper_max_dd': float,
                'squad_total_return': float,
                'sniper_total_return': float,
                'jensen_alpha_squad': float,
                'jensen_alpha_sniper': float,
                'recommendation': str  # 'concentrated' / 'diversified' / 'tighten_stop'
            }
        """
        concentration_ratio = self.calc_concentration_risk_ratio()
        alpha_squad = self.calc_jensen_alpha(self.squad_result)
        alpha_sniper = self.calc_jensen_alpha(self.sniper_result)

        squad_sharpe = self.squad_metrics.get("sharpe_ratio", 0.0)
        sniper_sharpe = self.sniper_metrics.get("sharpe_ratio", 0.0)
        squad_dd = self.squad_metrics.get("max_drawdown", 0.0)
        sniper_dd = self.sniper_metrics.get("max_drawdown", 0.0)
        squad_ret = self.squad_metrics.get("total_return", 0.0)
        sniper_ret = self.sniper_metrics.get("total_return", 0.0)

        # 建议
        if concentration_ratio > 2.0 and sniper_sharpe < squad_sharpe:
            recommendation = "diversified"  # 分散更优
        elif sniper_sharpe > squad_sharpe * 1.2 and concentration_ratio < 1.5:
            recommendation = "concentrated"  # 集中更优
        else:
            recommendation = "tighten_stop"  # 收紧止损

        return {
            "concentration_risk_ratio": concentration_ratio,
            "squad_sharpe": squad_sharpe,
            "sniper_sharpe": sniper_sharpe,
            "squad_max_dd": squad_dd,
            "sniper_max_dd": sniper_dd,
            "squad_total_return": squad_ret,
            "sniper_total_return": sniper_ret,
            "jensen_alpha_squad": alpha_squad,
            "jensen_alpha_sniper": alpha_sniper,
            "recommendation": recommendation,
        }
