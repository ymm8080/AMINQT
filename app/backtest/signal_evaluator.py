# -*- coding: utf-8 -*-
"""模块4: SignalEvaluator — 信号质量评估.

不交易, 只验证模型信号本身是否有预测力.
v4.0 变更:
    - 接受 config + trade_dates 参数
    - calc_rank_ic 增加 simulate_trigger 参数
    - 新增 calc_gap_risk: 隔夜跳空风险
    - 信号评估与回测一致性: 分母用触发价 (开盘×1.03)
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from app.backtest.config_manager import BacktestConfig

logger = logging.getLogger(__name__)


def _safe_div(numerator: float, denominator: float) -> float:
    """安全除法: 分母为 0/NaN 时返回 0."""
    if denominator == 0 or not np.isfinite(denominator):
        return 0.0
    return float(np.nan_to_num(numerator / denominator, nan=0.0))


class SignalEvaluator:
    """信号质量评估. 不交易, 只验证模型信号本身是否有预测力."""

    def __init__(
        self,
        pred_df: pd.DataFrame,
        price_df: pd.DataFrame,
        trade_dates: List[pd.Timestamp],
        config: BacktestConfig,
    ):
        """初始化信号评估器.

        Args:
            pred_df: 预测表.
            price_df: 行情表.
            trade_dates: 交易日历.
            config: 回测配置.
        """
        self.pred_df = pred_df.copy()
        self.price_df = price_df.copy()
        self.trade_dates = trade_dates
        self.config = config
        self._price_index = {}
        for _, row in self.price_df.iterrows():
            self._price_index[(row["date"], row["stock"])] = row

    def _get_price(self, date: pd.Timestamp, stock: str) -> pd.Series | None:
        """获取指定日期和股票的行情数据."""
        return self._price_index.get((date, stock))

    def _get_trade_date_after(
        self, date: pd.Timestamp, offset: int
    ) -> pd.Timestamp | None:
        """获取 date 后第 offset 个交易日 (使用交易日历)."""
        idx = np.searchsorted(self.trade_dates, date)
        target = idx + offset
        if 0 <= target < len(self.trade_dates):
            return self.trade_dates[target]
        return None

    def _calc_actual_returns(
        self, score_col: str, horizon: int, simulate_trigger: bool = True
    ) -> pd.DataFrame:
        """计算每个预测对应的实际收益.

        simulate_trigger=True 时, 分母用触发价 (开盘×1.03)×(1+滑点).
        未触发样本 actual_ret = NaN.

        Returns:
            DataFrame: [date, stock, score, prob_up, pred_ret, actual_ret,
                        triggered, open_triggered, gap_return]
        """
        records = []
        trigger_pct = self.config.trigger_pct
        slippage_buy = 1 + self.config.slippage_buy_bp / 10000.0

        for _, row in self.pred_df.iterrows():
            signal_date = row["date"]
            stock = row["stock"]
            score = row.get(score_col, np.nan)
            prob_up = row.get(f"prob_up_h{horizon}", np.nan)
            pred_ret = row.get(f"pred_ret_h{horizon}", np.nan)

            entry_date = self._get_trade_date_after(signal_date, 1)
            if entry_date is None:
                continue
            entry_row = self._get_price(entry_date, stock)
            if entry_row is None:
                continue

            open_price = entry_row["open"]
            high_price = entry_row["high"]
            close_t = entry_row.get("pre_close", np.nan)

            exit_date = self._get_trade_date_after(signal_date, 1 + horizon)
            if exit_date is None:
                continue
            exit_row = self._get_price(exit_date, stock)
            if exit_row is None:
                continue

            close_price = exit_row["close"]
            trigger_price = open_price * (1 + trigger_pct)
            triggered = high_price >= trigger_price

            if np.isfinite(close_t) and close_t > 0:
                open_triggered = open_price >= close_t * (1 + trigger_pct)
                gap_return = open_price / close_t - 1.0
            else:
                open_triggered = False
                gap_return = np.nan

            if simulate_trigger:
                if triggered:
                    entry_price = trigger_price * slippage_buy
                    actual_ret = (
                        close_price / entry_price - 1.0 if entry_price > 0 else np.nan
                    )
                else:
                    actual_ret = np.nan
            else:
                actual_ret = (
                    close_price / open_price - 1.0 if open_price > 0 else np.nan
                )

            records.append(
                {
                    "date": signal_date,
                    "stock": stock,
                    "score": score,
                    "prob_up": prob_up,
                    "pred_ret": pred_ret,
                    "actual_ret": actual_ret,
                    "triggered": triggered,
                    "open_triggered": open_triggered,
                    "gap_return": gap_return,
                }
            )

        return pd.DataFrame(records)

    def calc_rank_ic(
        self,
        score_col: str = "score_h2",
        horizon: int = 2,
        simulate_trigger: bool = True,
    ) -> pd.DataFrame:
        """计算每日 Top5 内部 Rank IC.

        Returns:
            DataFrame: [date, rank_ic].
        """
        df = self._calc_actual_returns(score_col, horizon, simulate_trigger)
        if df.empty:
            return pd.DataFrame(columns=["date", "rank_ic"])
        if simulate_trigger:
            df = df.dropna(subset=["actual_ret"])

        records = []
        for date, grp in df.groupby("date"):
            top5 = grp.nlargest(5, "score")
            valid = top5[["score", "actual_ret"]].dropna()
            if len(valid) < 2:
                continue
            if valid["score"].nunique() < 2 or valid["actual_ret"].nunique() < 2:
                continue
            corr, _ = spearmanr(valid["score"], valid["actual_ret"])
            if np.isfinite(corr):
                records.append({"date": date, "rank_ic": float(corr)})
        return pd.DataFrame(records)

    def calc_topk_hit_rate(
        self,
        score_col: str = "score_h2",
        prob_col: str = "prob_up_h2",
        horizon: int = 2,
        k: int = 5,
    ) -> Dict:
        """TopK 命中率统计."""
        df = self._calc_actual_returns(
            score_col, horizon, self.config.signal_simulate_trigger
        )
        if df.empty:
            return self._empty_hit_rate()

        df = df[df["prob_up"] >= self.config.prob_threshold]
        df = df.dropna(subset=["actual_ret"])

        hit_rates: dict[str, list] = {}
        top5_returns: list[float] = []
        top1_returns: list[float] = []

        for _, grp in df.groupby("date"):
            top = grp.nlargest(k, "score")
            for i in [1, 2, 3, 5]:
                top_i = top.head(i)
                valid = top_i["actual_ret"].dropna()
                if len(valid) > 0:
                    key = f"top{i}_hit_rate"
                    hit_rates.setdefault(key, []).append(
                        float((valid > 0).sum()) / len(valid)
                    )
            top5_valid = top.head(5)["actual_ret"].dropna()
            if len(top5_valid) > 0:
                top5_returns.append(float(top5_valid.mean()))
            top1_valid = top.head(1)["actual_ret"].dropna()
            if len(top1_valid) > 0:
                top1_returns.append(float(top1_valid.iloc[0]))

        return {
            "top1_hit_rate": float(np.mean(hit_rates["top1_hit_rate"]))
            if "top1_hit_rate" in hit_rates
            else 0.0,
            "top2_hit_rate": float(np.mean(hit_rates["top2_hit_rate"]))
            if "top2_hit_rate" in hit_rates
            else 0.0,
            "top3_hit_rate": float(np.mean(hit_rates["top3_hit_rate"]))
            if "top3_hit_rate" in hit_rates
            else 0.0,
            "top5_hit_rate": float(np.mean(hit_rates["top5_hit_rate"]))
            if "top5_hit_rate" in hit_rates
            else 0.0,
            "top5_avg_return": float(np.mean(top5_returns)) if top5_returns else 0.0,
            "top1_avg_return": float(np.mean(top1_returns)) if top1_returns else 0.0,
        }

    @staticmethod
    def _empty_hit_rate() -> Dict:
        """空命中率结果."""
        return {
            "top1_hit_rate": 0.0,
            "top2_hit_rate": 0.0,
            "top3_hit_rate": 0.0,
            "top5_hit_rate": 0.0,
            "top5_avg_return": 0.0,
            "top1_avg_return": 0.0,
        }

    def calc_trigger_stats(
        self,
        score_col: str = "score_h2",
        prob_col: str = "prob_up_h2",
        horizon: int = 2,
        k: int = 5,
    ) -> Dict:
        """触发率统计."""
        df = self._calc_actual_returns(score_col, horizon, simulate_trigger=False)
        if df.empty:
            return self._empty_trigger_stats()

        df = df[df["prob_up"] >= self.config.prob_threshold]
        total = 0
        triggered_count = 0
        open_triggered_count = 0
        triggered_returns: list[float] = []
        missed_returns: list[float] = []

        for _, grp in df.groupby("date"):
            top = grp.nlargest(k, "score")
            for _, row in top.iterrows():
                total += 1
                if row["triggered"]:
                    triggered_count += 1
                    if pd.notna(row["actual_ret"]):
                        triggered_returns.append(float(row["actual_ret"]))
                else:
                    if pd.notna(row["actual_ret"]):
                        missed_returns.append(float(row["actual_ret"]))
                if row["open_triggered"]:
                    open_triggered_count += 1

        return {
            "trigger_rate": _safe_div(triggered_count, total),
            "open_trigger_rate": _safe_div(open_triggered_count, total),
            "intraday_trigger_rate": _safe_div(
                triggered_count - open_triggered_count, total
            ),
            "triggered_avg_return": float(np.mean(triggered_returns))
            if triggered_returns
            else 0.0,
            "missed_avg_return": float(np.mean(missed_returns))
            if missed_returns
            else 0.0,
        }

    @staticmethod
    def _empty_trigger_stats() -> Dict:
        """空触发统计结果."""
        return {
            "trigger_rate": 0.0,
            "open_trigger_rate": 0.0,
            "intraday_trigger_rate": 0.0,
            "triggered_avg_return": 0.0,
            "missed_avg_return": 0.0,
        }

    def calc_prediction_bias(
        self, pred_col: str = "pred_ret_h2", horizon: int = 2
    ) -> Dict:
        """预测偏差分析. Returns: {mae, rmse, wmape, bias, correlation}."""
        score_col = f"score_h{horizon}"
        df = self._calc_actual_returns(
            score_col, horizon, self.config.signal_simulate_trigger
        )
        if df.empty:
            return {
                "mae": 0.0,
                "rmse": 0.0,
                "wmape": 0.0,
                "bias": 0.0,
                "correlation": 0.0,
            }

        valid = df[["pred_ret", "actual_ret"]].dropna()
        if len(valid) < 2:
            return {
                "mae": 0.0,
                "rmse": 0.0,
                "wmape": 0.0,
                "bias": 0.0,
                "correlation": 0.0,
            }

        pred = valid["pred_ret"].values
        actual = valid["actual_ret"].values
        errors = pred - actual
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))
        wmape = _safe_div(float(np.sum(np.abs(errors))), float(np.sum(np.abs(actual))))
        bias = float(np.mean(errors))
        correlation = (
            float(np.corrcoef(pred, actual)[0, 1])
            if np.std(pred) > 0 and np.std(actual) > 0
            else 0.0
        )

        return {
            "mae": mae,
            "rmse": rmse,
            "wmape": wmape,
            "bias": bias,
            "correlation": correlation,
        }

    def calc_prob_calibration(
        self, prob_col: str = "prob_up_h2", horizon: int = 2
    ) -> pd.DataFrame:
        """概率校准: Prob 分桶 vs 实际上涨频率.

        Returns:
            DataFrame: [prob_bin, count, actual_up_rate, avg_return].
        """
        score_col = f"score_h{horizon}"
        df = self._calc_actual_returns(
            score_col, horizon, self.config.signal_simulate_trigger
        )
        if df.empty:
            return pd.DataFrame(
                columns=["prob_bin", "count", "actual_up_rate", "avg_return"]
            )

        valid = df[["prob_up", "actual_ret"]].dropna()
        if valid.empty:
            return pd.DataFrame(
                columns=["prob_bin", "count", "actual_up_rate", "avg_return"]
            )

        bins = [0.0, 0.50, 0.55, 0.60, 0.70, 1.01]
        labels = ["<0.50", "0.50-0.55", "0.55-0.60", "0.60-0.70", "0.70+"]
        valid = valid.copy()
        valid["prob_bin"] = pd.cut(
            valid["prob_up"], bins=bins, labels=labels, right=False
        )

        records = []
        for label in labels:
            subset = valid[valid["prob_bin"] == label]
            if len(subset) == 0:
                records.append(
                    {
                        "prob_bin": label,
                        "count": 0,
                        "actual_up_rate": 0.0,
                        "avg_return": 0.0,
                    }
                )
            else:
                records.append(
                    {
                        "prob_bin": label,
                        "count": len(subset),
                        "actual_up_rate": float((subset["actual_ret"] > 0).mean()),
                        "avg_return": float(subset["actual_ret"].mean()),
                    }
                )
        return pd.DataFrame(records)

    def calc_gap_risk(self, score_col: str = "score_h2", k: int = 5) -> float:
        """隔夜跳空风险 (Gap Risk).

        Avg_Gap_Return = mean( (T+1 open / T close) - 1 ) for TopK stocks.

        Returns:
            平均隔夜跳空收益率.
        """
        df = self._calc_actual_returns(score_col, horizon=1, simulate_trigger=False)
        if df.empty:
            return 0.0

        gap_returns: list[float] = []
        for _, grp in df.groupby("date"):
            top = grp.nlargest(k, "score")
            valid = top["gap_return"].dropna()
            if len(valid) > 0:
                gap_returns.append(float(valid.mean()))
        return float(np.mean(gap_returns)) if gap_returns else 0.0

    def calc_volume_confirmation_rate(
        self, score_col: str = "score_h2", k: int = 5
    ) -> float:
        """V5.0 成交量确认率: 触发股票中成交量>=1.5倍的比例.

        Args:
            score_col: 排序列.
            k: TopK 数量.

        Returns:
            成交量确认率 (0.0-1.0).
        """
        ratio = self.config.volume_confirm_ratio
        confirmed_count = 0
        total_count = 0

        for date, grp in self.pred_df.groupby("date"):
            top = grp.nlargest(k, score_col)
            for _, row in top.iterrows():
                stock = row["stock"]
                entry_date = self._get_trade_date_after(date, 1)
                if entry_date is None:
                    continue
                entry_row = self._get_price(entry_date, stock)
                if entry_row is None:
                    continue
                signal_row = self._get_price(date, stock)
                if signal_row is None:
                    continue

                today_vol = entry_row.get("volume", 0)
                yesterday_vol = signal_row.get("volume", 0)

                if yesterday_vol > 0:
                    total_count += 1
                    if today_vol >= yesterday_vol * ratio:
                        confirmed_count += 1

        return _safe_div(float(confirmed_count), float(total_count))

    def run_full_report(self) -> Dict:
        """运行完整信号评估, 输出所有指标.

        对 H1/H2/H4 三个周期分别计算所有指标.

        Returns:
            dict: {h1: {...}, h2: {...}, h4: {...}}
        """
        report: Dict = {}
        for horizon in self.config.signal_horizons:
            score_col = f"score_h{horizon}"
            prob_col = f"prob_up_h{horizon}"
            pred_col = f"pred_ret_h{horizon}"

            logger.info("评估信号质量: H%d", horizon)

            rank_ic_df = self.calc_rank_ic(
                score_col, horizon, self.config.signal_simulate_trigger
            )
            hit_rate = self.calc_topk_hit_rate(score_col, prob_col, horizon)
            trigger_stats = self.calc_trigger_stats(score_col, prob_col, horizon)
            pred_bias = self.calc_prediction_bias(pred_col, horizon)
            calibration = self.calc_prob_calibration(prob_col, horizon)
            gap_risk = self.calc_gap_risk(score_col)

            rank_ic_mean = (
                float(rank_ic_df["rank_ic"].mean()) if not rank_ic_df.empty else 0.0
            )
            rank_ic_std = (
                float(rank_ic_df["rank_ic"].std()) if not rank_ic_df.empty else 0.0
            )
            rank_ir = _safe_div(rank_ic_mean, rank_ic_std)

            report[f"h{horizon}"] = {
                "rank_ic_mean": rank_ic_mean,
                "rank_ic_std": rank_ic_std,
                "rank_ir": rank_ir,
                "hit_rate": hit_rate,
                "trigger_stats": trigger_stats,
                "prediction_bias": pred_bias,
                "calibration": calibration.to_dict("records"),
                "avg_gap_return": gap_risk,
                "volume_confirmation_rate": self.calc_volume_confirmation_rate(
                    score_col
                ),
            }

        logger.info("信号评估完成")
        return report
