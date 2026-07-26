"""交易状态机 (P10, ARCH §8.3.2/§9.4).

独立双向控制: 自动买 / 自动卖 各自开关;
状态: RUNNING / PAUSED / STOPPED (暂停/恢复/停止全部)。

有效状态迁移:
    start:    STOPPED/PAUSED → RUNNING
    pause:    RUNNING        → PAUSED
    resume:   PAUSED         → RUNNING
    stop_all: RUNNING/PAUSED → STOPPED (并触发 on_stop_all 回调)
非法迁移 → 记录 warning, 不做任何操作 (no-op)。
"""

import logging
from collections.abc import Callable
from enum import Enum

logger = logging.getLogger(__name__)


class TradingState(Enum):
    """交易状态."""

    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


# 合法迁移表: {动作: (允许的源状态集合, 目标状态)}
_TRANSITIONS = {
    "start": ({TradingState.STOPPED, TradingState.PAUSED}, TradingState.RUNNING),
    "pause": ({TradingState.RUNNING}, TradingState.PAUSED),
    "resume": ({TradingState.PAUSED}, TradingState.RUNNING),
    "stop_all": ({TradingState.RUNNING, TradingState.PAUSED}, TradingState.STOPPED),
}


class TradingStateMachine:
    """交易状态机 — 自动买/卖独立开关 + 暂停/恢复.

    Attributes:
        state: 当前 TradingState。
        auto_buy_enabled: 自动买开关 (独立于自动卖)。
        auto_sell_enabled: 自动卖开关 (独立于自动买)。
    """

    def __init__(self, on_stop_all: Callable[[], None] | None = None) -> None:
        """初始化状态机 (初始 STOPPED, 双向自动开关均关闭).

        Args:
            on_stop_all: 可选回调, stop_all 时触发 (如撤销所有待确认委托)。
        """
        self.state = TradingState.STOPPED
        self.auto_buy_enabled = False  # 自动买开关 (独立)
        self.auto_sell_enabled = False  # 自动卖开关 (独立)
        self.on_stop_all = on_stop_all
        # 暂停记忆: pause 前保存的自动开关状态, resume 时恢复
        self._paused_auto_buy = False
        self._paused_auto_sell = False

    def _transition(self, action: str) -> bool:
        """执行状态迁移; 非法迁移记录 warning 并 no-op.

        Args:
            action: 动作名 (start/pause/resume/stop_all)。

        Returns:
            True 迁移成功, False 非法迁移 (no-op)。
        """
        allowed, target = _TRANSITIONS[action]
        if self.state not in allowed:
            logger.warning(
                "[TSM] 非法迁移 %s: 当前状态 %s, 忽略", action, self.state.value
            )
            return False
        logger.info("[TSM] %s: %s → %s", action, self.state.value, target.value)
        self.state = target
        return True

    def start(self) -> None:
        """启动 (STOPPED/PAUSED → RUNNING)."""
        self._transition("start")

    def pause(self) -> None:
        """暂停 (RUNNING → PAUSED), 记住当前自动开关状态."""
        if self._transition("pause"):
            self._paused_auto_buy = self.auto_buy_enabled
            self._paused_auto_sell = self.auto_sell_enabled
            logger.info("[TSM] 暂停, 记忆自动买/卖=%s/%s", self._paused_auto_buy, self._paused_auto_sell)

    def resume(self) -> None:
        """恢复 (PAUSED → RUNNING), 还原暂停前自动开关状态."""
        if self._transition("resume"):
            self.auto_buy_enabled = self._paused_auto_buy
            self.auto_sell_enabled = self._paused_auto_sell
            logger.info("[TSM] 恢复, 还原自动买/卖=%s/%s", self.auto_buy_enabled, self.auto_sell_enabled)

    def stop_all(self) -> None:
        """停止全部 (→ STOPPED), 并触发 on_stop_all 回调 (如撤销待确认委托)."""
        if self._transition("stop_all"):
            self.auto_buy_enabled = False
            self.auto_sell_enabled = False
            self._paused_auto_buy = False
            self._paused_auto_sell = False
            if self.on_stop_all is not None:
                try:
                    self.on_stop_all()
                except Exception:
                    logger.exception("[TSM] on_stop_all 回调异常")

    def set_auto_buy(self, enabled: bool) -> None:
        """自动买开关 (独立于自动卖)."""
        self.auto_buy_enabled = bool(enabled)
        logger.info("[TSM] 自动买 %s", "开启" if self.auto_buy_enabled else "关闭")

    def set_auto_sell(self, enabled: bool) -> None:
        """自动卖开关 (独立于自动买)."""
        self.auto_sell_enabled = bool(enabled)
        logger.info("[TSM] 自动卖 %s", "开启" if self.auto_sell_enabled else "关闭")

    def can_execute(self, side: str) -> bool:
        """是否可执行: state==RUNNING 且对应方向自动开关已开.

        Args:
            side: "buy" 或 "sell"。

        Returns:
            True 当且仅当 RUNNING 且对应方向开关开启; 未知 side 返回 False。
        """
        if self.state is not TradingState.RUNNING:
            return False
        if side == "buy":
            return self.auto_buy_enabled
        if side == "sell":
            return self.auto_sell_enabled
        logger.warning("[TSM] can_execute 未知方向: %s", side)
        return False

    # ── IMPLEMENTATION PLAN v3.1 命名别名 ─────────────────────────

    def enable_auto_buy(self) -> None:
        """启动自动买入 (P10 命名)."""
        self.set_auto_buy(True)

    def disable_auto_buy(self) -> None:
        """停止自动买入 (P10 命名)."""
        self.set_auto_buy(False)

    def enable_auto_sell(self) -> None:
        """启动自动卖出 (P10 命名)."""
        self.set_auto_sell(True)

    def disable_auto_sell(self) -> None:
        """停止自动卖出 (P10 命名)."""
        self.set_auto_sell(False)

    def force_stop(self, reason: str = "风控强制停止") -> None:
        """风控强制停止全部自动交易 (→ STOPPED)."""
        logger.warning("[TSM] %s", reason)
        self.stop_all()

    def should_auto_execute(self, side: str) -> bool:
        """判断信号是否自动执行 (P10 命名, 同 can_execute)."""
        return self.can_execute(side)
