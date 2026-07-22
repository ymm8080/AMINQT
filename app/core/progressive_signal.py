# -*- coding: utf-8 -*-
"""渐进式交易信号推进 (P10.13, ARCH §5.18, DESIGN_V1 §9 #5).

交易信号不一次性生成: 种子 → 确认 → 触发 三阶段推进。
支持 auto (系统评判) / manual (用户确认) / hybrid 模式;
支持用户指定触发条件; 内置不连续买入约束。
"""

import logging
import operator
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}


class SignalStage(Enum):
    """信号推进阶段."""

    SEED = 1  # 种子: 初步条件满足
    CONFIRMATION = 2  # 确认: 多维条件复核
    TRIGGER = 3  # 触发: 可执行信号


class ProgressiveSignal:
    """渐进式交易信号推进器."""

    def __init__(self, config: dict = None) -> None:
        """加载配置 (trading_config.yaml: progressive_signal 段).

        含 mode (auto/manual/hybrid), prevent_consecutive_buy: true,
        user_trigger_conditions 等。

        Args:
            config: progressive_signal 配置段 (可为 None, 全部用默认值)。
        """
        self.config = config or {}
        self.mode = str(self.config.get("mode", "auto"))
        self.prevent_consecutive_buy = bool(
            self.config.get("prevent_consecutive_buy", True)
        )
        self.user_trigger_conditions: List[dict] = list(
            self.config.get("user_trigger_conditions", [])
        )
        # 每股票当前阶段 / 最近因子 / 最近买入记录
        self._stages: Dict[str, SignalStage] = {}
        self._last_factors: Dict[str, dict] = {}
        self._last_buy: Dict[str, float] = {}  # symbol -> 买入价

    # ── 配置读取辅助 ────────────────────────────────────────────────
    def _cfg(self, section: str, key: str, default):
        """读取嵌套配置, 缺省返回 default."""
        sec = self.config.get(section, {})
        if isinstance(sec, dict):
            return sec.get(key, default)
        return default

    @staticmethod
    def _to_score(value: float) -> float:
        """分数归一化到 0~100 (>1 视为已是百分制)."""
        v = float(value)
        return v if v > 1.0 else v * 100.0

    def get_stage(self, symbol: str) -> SignalStage:
        """查询股票当前所处阶段 (默认 SEED)."""
        return self._stages.get(symbol, SignalStage.SEED)

    def evaluate_seed(self, symbol: str, factors: dict) -> bool:
        """Stage 1 种子评估: 初步条件是否满足.

        种子分数 = 加权(model_score, factor_resonance, market_env) × 100,
        默认权重 0.4/0.4/0.2, 阈值 60。

        Args:
            symbol: 股票代码。
            factors: {model_score, factor_resonance, market_env} (0~1 或 0~100)。

        Returns:
            True = 种子分数达标。
        """
        weights = self._cfg("seed", "weights", {}) or {}
        w_model = float(weights.get("model_score", 0.4))
        w_res = float(weights.get("factor_resonance", 0.4))
        w_env = float(weights.get("market_env", 0.2))
        threshold = float(self._cfg("seed", "threshold", 60))

        score = (
            w_model * self._to_score(factors.get("model_score", 0.0))
            + w_res * self._to_score(factors.get("factor_resonance", 0.0))
            + w_env * self._to_score(factors.get("market_env", 0.0))
        )
        passed = score >= threshold
        logger.debug(
            "种子评估 %s: score=%.1f threshold=%.1f → %s",
            symbol,
            score,
            threshold,
            passed,
        )
        return passed

    def evaluate_confirmation(self, symbol: str, factors: dict) -> bool:
        """Stage 2 确认评估: 多维条件复核.

        默认检查项 (各占 25 分): daily_mark / market_breadth /
        flow_net_positive / ctrl_ratio, 阈值 75 → 至少过 3 项。

        Args:
            symbol: 股票代码。
            factors: {daily_mark: bool, market_breadth: float,
                      flow_net: float, ctrl_ratio: float}。

        Returns:
            True = 确认分数达标。
        """
        required = self._cfg("confirmation", "required_checks", None)
        if not required:
            required = [
                "daily_mark",
                "market_breadth",
                "flow_net_positive",
                "ctrl_ratio",
            ]
        threshold = float(self._cfg("confirmation", "threshold", 75))
        breadth_th = float(self._cfg("confirmation", "breadth_threshold", 0.6))
        ctrl_th = float(self._cfg("confirmation", "ctrl_ratio_threshold", 0.30))

        checks = {
            "daily_mark": bool(factors.get("daily_mark", False)),
            "market_breadth": float(factors.get("market_breadth", 0.0)) > breadth_th,
            "flow_net_positive": float(factors.get("flow_net", 0.0)) > 0.0,
            "ctrl_ratio": float(factors.get("ctrl_ratio", 0.0)) > ctrl_th,
        }
        total = len(required)
        passed_count = sum(1 for c in required if checks.get(c, False))
        score = 100.0 * passed_count / total if total > 0 else 0.0
        passed = score >= threshold
        logger.debug(
            "确认评估 %s: %d/%d 项通过 score=%.1f → %s",
            symbol,
            passed_count,
            total,
            score,
            passed,
        )
        return passed

    def evaluate_trigger(self, symbol: str, factors: dict) -> bool:
        """Stage 3 触发评估: 是否达到执行条件.

        优先级:
          1. 用户指定触发条件满足 → 触发
          2. auto_trigger 开启且 factors['opening_confirmed'] → 触发
        买入方向且 prevent_consecutive_buy 开启时, 受不连续买入约束拦截。

        Args:
            symbol: 股票代码。
            factors: 含 opening_confirmed / side / price 等。

        Returns:
            True = 达到执行条件。
        """
        self._last_factors[symbol] = dict(factors)
        triggered = False

        # 1. 用户指定触发条件
        conditions = self.user_trigger_conditions
        if conditions and self.check_user_trigger(symbol, factors, conditions):
            triggered = True
            logger.info("触发评估 %s: 用户指定条件达成", symbol)

        # 2. 系统自动触发 (开盘10分钟走势确认)
        if not triggered and bool(self._cfg("trigger", "auto_trigger", True)):
            if bool(factors.get("opening_confirmed", False)):
                triggered = True
                logger.info("触发评估 %s: 开盘走势确认, 自动触发", symbol)

        # 不连续买入约束 (仅买入方向)
        side = str(factors.get("side", "buy"))
        if triggered and side == "buy" and not self.check_consecutive_buy(symbol):
            logger.info("触发评估 %s: 被不连续买入约束拦截", symbol)
            return False
        return triggered

    def check_user_trigger(
        self, symbol: str, factors: dict, conditions: List[dict]
    ) -> bool:
        """用户指定触发条件检查 (达到用户指标即触发).

        Args:
            symbol: 股票代码。
            factors: 当前因子值 {factor_name: value}。
            conditions: [{factor, op, value}], 全部满足 (AND) 才触发。

        Returns:
            True = 全部条件满足。
        """
        if not conditions:
            return False
        for cond in conditions:
            factor = cond.get("factor")
            op_str = cond.get("op", ">")
            target = cond.get("value")
            op_fn = _OPS.get(op_str)
            if op_fn is None:
                logger.warning("check_user_trigger: 未知操作符 %r", op_str)
                return False
            actual = factors.get(factor)
            if actual is None:
                return False
            try:
                if not op_fn(float(actual), float(target)):
                    return False
            except (TypeError, ValueError):
                logger.warning(
                    "check_user_trigger: 无法比较 %s=%r %s %r",
                    factor,
                    actual,
                    op_str,
                    target,
                )
                return False
        return True

    def advance(self, symbol: str, factors: dict) -> Optional[dict]:
        """推进信号到下一阶段.

        每调用一次评估当前阶段:
          SEED 通过 → CONFIRMATION; CONFIRMATION 通过 → TRIGGER;
          TRIGGER 通过 → 返回执行信号并重置回 SEED。
        manual 模式: 每阶段推进需 factors['user_confirmed']=True;
        hybrid 模式: 仅 Stage 3 需用户确认; auto 模式全自动。

        Args:
            symbol: 股票代码。
            factors: 当前因子快照 (会留存用于约束检查)。

        Returns:
            触发时返回执行信号 dict, 否则 None。
        """
        self._last_factors[symbol] = dict(factors)
        stage = self.get_stage(symbol)
        user_confirmed = bool(factors.get("user_confirmed", False))

        # 用户确认门槛
        needs_confirm = self.mode == "manual" or (
            self.mode == "hybrid" and stage == SignalStage.TRIGGER
        )
        if needs_confirm and not user_confirmed:
            logger.debug(
                "推进 %s: %s 模式等待用户确认 (stage=%s)", symbol, self.mode, stage.name
            )
            return None

        if stage == SignalStage.SEED:
            if self.evaluate_seed(symbol, factors):
                self._stages[symbol] = SignalStage.CONFIRMATION
                logger.info("信号推进 %s: SEED → CONFIRMATION", symbol)
            return None

        if stage == SignalStage.CONFIRMATION:
            if self.evaluate_confirmation(symbol, factors):
                self._stages[symbol] = SignalStage.TRIGGER
                logger.info("信号推进 %s: CONFIRMATION → TRIGGER", symbol)
            return None

        # stage == TRIGGER
        if self.evaluate_trigger(symbol, factors):
            side = str(factors.get("side", "buy"))
            price = factors.get("price")
            if side == "buy" and price is not None:
                self._last_buy[symbol] = float(price)
            self._stages[symbol] = SignalStage.SEED  # 触发后重置
            signal = {
                "symbol": symbol,
                "signal": side,
                "stage": SignalStage.TRIGGER.name,
                "price": price,
                "reason": "渐进式信号三阶段全部通过",
            }
            logger.info("执行信号 %s: %s @ %s", symbol, side, price)
            return signal
        return None

    def check_consecutive_buy(self, symbol: str) -> bool:
        """不连续买入约束: 同一只股票上涨时不得连续两次买入 (可手工覆盖).

        判定: 上次买入过该股 且 当前价 > 上次买入价 (上涨中) → 拦截;
        下跌/持平允许补仓。配置 allow_manual_override 且当前因子快照
        含 user_confirmed=True 时放行。

        Args:
            symbol: 股票代码。

        Returns:
            True = 允许买入; False = 违反约束。
        """
        if not self.prevent_consecutive_buy:
            return True
        last_price = self._last_buy.get(symbol)
        if last_price is None:
            return True
        factors = self._last_factors.get(symbol, {})
        current_price = factors.get("price")
        if current_price is None:
            return True  # 无法判断, 不拦截
        if float(current_price) > float(last_price):
            if bool(
                self._cfg("consecutive_buy", "allow_manual_override", True)
            ) and bool(factors.get("user_confirmed", False)):
                logger.info("不连续买入约束 %s: 用户手动覆盖, 放行", symbol)
                return True
            logger.info(
                "不连续买入约束 %s: 上涨中重复买入 (%.3f > %.3f) → 拦截",
                symbol,
                current_price,
                last_price,
            )
            return False
        return True
