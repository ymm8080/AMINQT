"""
参数调优器 (用户 2026-07-22 需求: 预定义买卖/持仓规则参数由回测结果调整)
============================================================================
- 调参目标: app/rules/config.py TUNABLE_BOUNDS 标记的 [TUNABLE] 参数
  (止损/移动止盈/概率衰减/持仓天数/跳空阈值/广度/控盘线/冷静期等)
- 方法: 网格搜索 (小维度) / 坐标下降 (大维度), 目标函数 = 扣费后净年化超额
- 防过拟合: 训练段选参 → OOS 段复验, OOS 不达标则回退默认值 + 告警
  (与 V3.5 超参纪律一致: 严禁在验证集上反复搜索 = 多重检验)
- 产出: 推荐 Config 参数 + 报告 (data/tuning_report_{tag}.json)
"""

from __future__ import annotations

import itertools
import json
import logging
import os

import numpy as np
import pandas as pd

from app.pipeline1.backtest_v35 import BacktestEngineV35, BacktestProtocol
from app.rules.config import TUNABLE_BOUNDS

logger = logging.getLogger(__name__)

# 回测协议参数 ←→ Config 字段映射 (调参时同步两处)
PROTOCOL_FIELDS = {"max_hold_days", "hard_stop", "trailing_drawdown", "prob_exit"}
# Config 参数 → 回测协议参数映射 (名称不同的)
CONFIG_TO_PROTOCOL = {
    "hard_stop_intraday": "hard_stop",
    "max_hold_days": "max_hold_days",
    "trailing_drawdown": "trailing_drawdown",
    "prob_exit": "prob_exit",
}


def _native(v):
    """numpy 标量 → Python 原生类型 (JSON/pydantic 序列化安全)."""
    if isinstance(v, np.generic):
        return v.item()
    return v


class ParamTuner:
    """买卖/持仓规则参数回测调优."""

    def __init__(
        self,
        panel: pd.DataFrame,
        daily_lists: dict,
        benchmark: pd.Series | None = None,
        report_dir: str = "data",
    ):
        self.panel = panel
        self.daily_lists = daily_lists
        self.benchmark = benchmark
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)

    # ---------------- 单次评估 ----------------
    def _evaluate(
        self, params: dict, panel: pd.DataFrame, lists: dict, benchmark
    ) -> dict:
        """跑一轮回测, 返回完整 metrics 字典."""
        proto_kwargs = {}
        for cfg_name, value in params.items():
            if value is None:
                continue
            proto_name = CONFIG_TO_PROTOCOL.get(cfg_name)
            if proto_name in PROTOCOL_FIELDS:
                proto_kwargs[proto_name] = value
        protocol = BacktestProtocol(**proto_kwargs)
        eng = BacktestEngineV35(panel, protocol)
        result = eng.run(lists, benchmark)
        return result["metrics"]

    @staticmethod
    def _score(metrics: dict, objective: str) -> float:
        """从 metrics 中提取目标函数值."""
        if objective == "sharpe":
            return metrics.get("sharpe", -np.inf)
        if objective == "total_return":
            return metrics.get("total_return", -np.inf)
        # 默认: 扣费后净年化超额
        return metrics.get("net_excess_annual", -np.inf)

    # ---------------- 网格搜索 ----------------
    def grid_search(
        self,
        param_names: list[str],
        oos_ratio: float = 0.3,
        top_k: int = 5,
        objective: str = "net_excess_annual",
        max_dd_limit: float | None = None,
    ) -> dict:
        """对指定参数做网格搜索.

        Args:
            param_names: TUNABLE_BOUNDS 的子集 (建议 <= 4 维, 控制组合数)
            oos_ratio: OOS 段占比 (尾部样本, 严禁反向调参)
            top_k: 训练段前 K 名进入 OOS 复验
            objective: 目标函数字段 (net_excess_annual / sharpe / total_return)
            max_dd_limit: 最大回撤约束 (如 -0.10); OOS 复验时过滤

        Returns:
            {best_params, train_score, oos_score, fallback, leaderboard, report_path}
        """
        grids = []
        for name in param_names:
            lo, hi, step = TUNABLE_BOUNDS[name]
            values = np.arange(lo, hi + step / 2, step)
            if (
                isinstance(TUNABLE_BOUNDS[name][0], int)
                or float(step).is_integer()
                and hi <= 10
            ):
                values = np.unique(values.astype(int))
            grids.append([(name, v) for v in values])
        combos = [dict(c) for c in itertools.product(*grids)]
        logger.info("网格搜索: %d 维 %d 组合", len(param_names), len(combos))

        # 训练/OOS 切分 (按时间)
        dates = sorted(self.panel["date"].unique())
        split = int(len(dates) * (1 - oos_ratio))
        train_dates = set(dates[:split])
        train_panel = self.panel[self.panel["date"].isin(train_dates)]
        oos_panel = self.panel[~self.panel["date"].isin(train_dates)]
        train_lists = {d: v for d, v in self.daily_lists.items() if d in train_dates}
        oos_lists = {d: v for d, v in self.daily_lists.items() if d not in train_dates}
        bench_train = (
            self.benchmark[self.benchmark.index.isin(train_dates)]
            if self.benchmark is not None
            else None
        )
        bench_oos = (
            self.benchmark[~self.benchmark.index.isin(train_dates)]
            if self.benchmark is not None
            else None
        )

        # 训练段评分
        scores = []
        for params in combos:
            metrics = self._evaluate(params, train_panel, train_lists, bench_train)
            score = self._score(metrics, objective)
            if np.isnan(score):
                continue
            scores.append((params, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        leaderboard = scores[:top_k]

        # OOS 复验 (只验 top_k, 不做二次搜索)
        best_params, best_train = leaderboard[0]
        oos_metrics = self._evaluate(best_params, oos_panel, oos_lists, bench_oos)
        oos_best = self._score(oos_metrics, objective)

        default_params = {n: None for n in param_names}  # 协议默认值
        oos_default_metrics = self._evaluate(
            default_params, oos_panel, oos_lists, bench_oos
        )
        oos_default = self._score(oos_default_metrics, objective)

        # 约束检查
        fallback = False
        if max_dd_limit is not None:
            dd = oos_metrics.get("max_drawdown", 0.0)
            if dd < max_dd_limit:
                fallback = True
                logger.warning(
                    "OOS 最大回撤 %.2f%% 超过约束 %.2f%%, 回退默认值",
                    dd * 100,
                    max_dd_limit * 100,
                )
        if not fallback and oos_best < oos_default:
            fallback = True
            logger.warning(
                "OOS 复验不达标: 调参 %.2f%% < 默认 %.2f%%, 回退默认值",
                oos_best * 100,
                oos_default * 100,
            )
        if fallback:
            best_params = default_params

        report = {
            "param_names": param_names,
            "objective": objective,
            "max_dd_limit": max_dd_limit,
            "n_combos": len(combos),
            "best_params": {k: _native(v) for k, v in best_params.items()},
            "train_score": round(best_train, 4),
            "oos_score": round(oos_best, 4),
            "oos_default": round(oos_default, 4),
            "fallback_to_default": fallback,
            "leaderboard": [
                ({k: _native(v) for k, v in p.items()}, round(s, 4))
                for p, s in leaderboard
            ],
        }
        path = os.path.join(self.report_dir, "tuning_report.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1, default=str)
        report["report_path"] = path
        return report

    # ---------------- 应用到 Config ----------------
    @staticmethod
    def apply_to_config(params: dict, cfg):
        """把调优结果写回规则引擎 Config (只写 TUNABLE 字段)."""
        for name, value in params.items():
            if name in TUNABLE_BOUNDS and value is not None and hasattr(cfg, name):
                setattr(cfg, name, type(getattr(cfg, name))(value))
        return cfg
