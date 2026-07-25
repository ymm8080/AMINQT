"""
生产配置 (V5.1 P20.3: prod_config 盘前生成, 盘中只读; 紧急开关为唯一人工入口)
====================================================================================
配置冻结审计: 盘前 (21:30 清单下发后) 生成当日 prod_config;
盘中任何读取走 frozen 快照, 修改即告警拒绝 (盘中绝对禁止热更新参数, V5.1 §7).
紧急开关 (唯一人工入口): 暂停 / 只平不开 / 强制平仓 — 全部 WORM 留痕.
"""

from __future__ import annotations

import copy
import logging

from .worm_logger import WormLogger

logger = logging.getLogger(__name__)

SWITCH_ACTIONS = ("pause", "close_only", "force_liquidate", "resume")


class ConfigFrozenError(Exception):
    """盘中修改冻结配置 → 拒绝."""


class ProdConfig:
    """盘前生成, 盘中只读的生产配置."""

    def __init__(self):
        self._frozen = False
        self._config: dict = {}

    def generate(self, config: dict) -> None:
        """盘前生成 (仅此入口可写)."""
        assert not self._frozen, "当日配置已冻结, 次日盘前方可重新生成"
        self._config = copy.deepcopy(config)
        self._frozen = True
        logger.info("prod_config 盘前生成并冻结: %d 键", len(config))

    def get(self, key: str, default=None):
        """盘中只读."""
        return self._config.get(key, default)

    def set(self, key: str, value) -> None:
        """盘中修改 → 拒绝 (配置冻结审计)."""
        if self._frozen:
            logger.error("盘中修改配置被拒绝: %s (冻结审计, 盘中禁止热更新)", key)
            raise ConfigFrozenError(f"配置已冻结, 拒绝修改: {key}")
        self._config[key] = value

    @property
    def frozen(self) -> bool:
        return self._frozen

    def new_day(self) -> None:
        """次日盘前解冻, 允许重新生成."""
        self._frozen = False
        self._config = {}


class EmergencySwitch:
    """紧急开关 (唯一人工入口, 全部 WORM 留痕).

    状态: NORMAL → pause (暂停全部信号) / close_only (只平不开)
          → force_liquidate (强制平仓所有可卖持仓) / resume (恢复).
    """

    def __init__(self, worm: WormLogger):
        self.state = "NORMAL"
        self.worm = worm

    def activate(
        self, trade_date: str, action: str, operator: str, reason: str
    ) -> None:
        """触发紧急动作 (人工, 需操作员与原因 — 留痕审计)."""
        assert action in SWITCH_ACTIONS, f"动作须为 {SWITCH_ACTIONS}"
        assert operator and reason, "紧急开关必须登记操作员与原因"
        self.state = "NORMAL" if action == "resume" else action
        self.worm.log(
            trade_date,
            "manual",
            {"emergency": action, "operator": operator, "reason": reason},
        )
        logger.critical("紧急开关: %s (操作员 %s, 原因: %s)", action, operator, reason)

    def allow_buy(self) -> bool:
        return self.state == "NORMAL"

    def allow_sell(self) -> bool:
        return True  # 任何状态都允许卖出 (pause 也只停买入信号)

    def must_liquidate(self) -> bool:
        return self.state == "force_liquidate"
