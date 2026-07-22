# -*- coding: utf-8 -*-
"""全自适应引擎 (P10.12, ARCH §5.17, DESIGN_V1 §9 #4).

取消所有固定比例: 选股评分混合/交易信号混合/因子影响力/风控阈值等
全部由回测 GAP 动态调整, 配置仅保留边界和初始值。
"""

import copy
import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config", "adaptive_config.yaml",
)
DEFAULT_STATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "adaptive_state.json",
)

# 单次调整步长占边界区间宽度的比例
_STEP_FRACTION = 0.10


class AdaptiveEngine:
    """全自适应引擎 — 基于回测 GAP 动态计算所有比例/阈值.

    调整规则 (ARCH §5.17.2):
        - GAP > 0 (实际优于目标) → 沿正方向调整 (适度放松)
        - GAP < 0 (实际低于目标) → 沿负方向调整 (收紧)
        - new = clamp(current + step * sign(gap), bounds)
        - step = 10% * (upper - lower)
        - candidates 型参数 (离散候选) 不自动调整, 保持当前值
        - 每次调整前压入历史快照, 支持 rollback; 状态持久化到
          data/adaptive_state.json
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH,
                 state_path: str = DEFAULT_STATE_PATH) -> None:
        """加载边界与初始值配置.

        Args:
            config_path: adaptive_config.yaml 路径 (initial + bounds/candidates)。
            state_path: 自适应状态持久化路径 (当前值 + 历史快照栈)。
        """
        self.config_path = config_path
        self.state_path = state_path
        with open(config_path, "r", encoding="utf-8") as f:
            self._spec: Dict = yaml.safe_load(f) or {}
        # 当前值: 优先取持久化状态, 否则取 initial
        persisted = self._load_state()
        self._current: Dict[str, Dict[str, float]] = {}
        for section, params in self._spec.items():
            if not isinstance(params, dict):
                continue
            self._current[section] = {}
            for name, spec in params.items():
                if not isinstance(spec, dict) or "initial" not in spec:
                    continue
                value = persisted.get("current", {}).get(section, {}).get(
                    name, spec["initial"])
                self._current[section][name] = value
        self._history: List[Dict] = persisted.get("history", [])
        logger.info("AdaptiveEngine 初始化: %d 个配置段, 历史快照 %d 条",
                    len(self._current), len(self._history))

    # ── 持久化 ──────────────────────────────────────────────

    def _load_state(self) -> Dict:
        """从磁盘加载自适应状态."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as exc:
                logger.warning("自适应状态加载失败 (%s), 使用 initial", exc)
        return {}

    def _save_state(self) -> None:
        """持久化当前值 + 历史快照栈."""
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        payload = {"current": self._current, "history": self._history}
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # ── 内部调整原语 ────────────────────────────────────────

    def _snapshot(self) -> None:
        """调整前压入当前值快照."""
        self._history.append(copy.deepcopy(self._current))

    def _adjust(self, section: str, name: str, gap: float):
        """按 GAP 调整单个数值参数, 返回新值 (无边界定义则不变).

        Args:
            section: 配置段名 (如 scoring_mix)。
            name: 参数名 (如 model_weight)。
            gap: 回测 GAP (实际 - 目标); 符号决定调整方向。

        Returns:
            调整后的新值 (clamp 到 bounds)。
        """
        spec = self._spec.get(section, {}).get(name, {})
        bounds = spec.get("bounds")
        current = self._current.get(section, {}).get(name)
        if bounds is None or current is None:
            logger.debug("跳过无边界参数: %s.%s", section, name)
            return current
        if not isinstance(current, (int, float)) or isinstance(current, bool):
            logger.debug("跳过非数值参数: %s.%s", section, name)
            return current
        if gap == 0 or not np.isfinite(gap):
            return current
        lower, upper = float(bounds[0]), float(bounds[1])
        step = _STEP_FRACTION * (upper - lower)
        new_value = float(np.clip(current + step * np.sign(gap), lower, upper))
        new_value = float(np.nan_to_num(new_value, nan=current))
        self._current[section][name] = new_value
        logger.info("自适应调整 %s.%s: %s -> %s (gap=%.4f, bounds=[%s, %s])",
                    section, name, current, new_value, gap, lower, upper)
        return new_value

    # ── 公开接口 ────────────────────────────────────────────

    def compute_selection_mix(self, backtest_gaps: Dict[str, float]) -> float:
        """基于选股 GAP 计算评分混合比例 (模型分 vs 指标分).

        Args:
            backtest_gaps: 回测 GAP 字典, 读 "selection_ic_gap"
                           (缺省 0.0 → 不调整)。

        Returns:
            调整后的 scoring_mix.model_weight。
        """
        self._snapshot()
        value = self._adjust(
            "scoring_mix", "model_weight",
            float(backtest_gaps.get("selection_ic_gap", 0.0)))
        self._save_state()
        return value

    def compute_trading_mix(self, backtest_gaps: Dict[str, float]) -> float:
        """基于交易 GAP 计算信号混合比例.

        Args:
            backtest_gaps: 回测 GAP 字典, 读 "trading_win_rate_gap"。

        Returns:
            调整后的 trading_mix.model_weight。
        """
        self._snapshot()
        value = self._adjust(
            "trading_mix", "model_weight",
            float(backtest_gaps.get("trading_win_rate_gap", 0.0)))
        self._save_state()
        return value

    def compute_factor_influence(self, backtest_gaps: Dict[str, float]) -> float:
        """基于因子 GAP 计算训练因子影响力比例.

        Args:
            backtest_gaps: 回测 GAP 字典, 读 "factor_ic_gap"。

        Returns:
            调整后的 factor_influence.ths_boost。
        """
        self._snapshot()
        value = self._adjust(
            "factor_influence", "ths_boost",
            float(backtest_gaps.get("factor_ic_gap", 0.0)))
        self._save_state()
        return value

    def compute_risk_thresholds(self, backtest_gaps: Dict[str, float]) -> Dict[str, float]:
        """基于风控 GAP 计算风控阈值组.

        对 risk 段全部数值型有边界参数统一按 "risk_drawdown_gap"
        (或逐参数 "risk_<name>_gap") 调整; candidates 型参数跳过。

        Args:
            backtest_gaps: 回测 GAP 字典。

        Returns:
            {参数名: 调整后阈值}。
        """
        self._snapshot()
        default_gap = float(backtest_gaps.get("risk_drawdown_gap", 0.0))
        result: Dict[str, float] = {}
        for name in self._current.get("risk", {}):
            current = self._current["risk"][name]
            if not isinstance(current, (int, float)) or isinstance(current, bool):
                continue  # candidates 型 (离散) 参数跳过
            gap = float(backtest_gaps.get(f"risk_{name}_gap", default_gap))
            result[name] = self._adjust("risk", name, gap)
        self._save_state()
        return result

    def get_adaptive_config(self) -> dict:
        """返回当前全部自适应参数 (供各 Pipeline/组件读取).

        Returns:
            {section: {param: 当前值}} (深拷贝)。
        """
        return copy.deepcopy(self._current)

    def rollback(self) -> None:
        """回滚到上一版自适应参数 (回测恶化时).

        弹出最近一次调整前的快照并持久化; 无历史时为空操作。
        """
        if not self._history:
            logger.warning("无历史快照, 回滚为空操作")
            return
        self._current = self._history.pop()
        self._save_state()
        logger.info("已回滚到上一版自适应参数 (剩余快照 %d 条)", len(self._history))
