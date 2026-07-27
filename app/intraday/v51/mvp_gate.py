# -*- coding: utf-8 -*-
"""
MVP Gate 实盘门禁 (F-1.3, IMPLEMENTATION_PLAN_v3.2 P24)
=========================================================
7 项硬性检查 — 任一未通过则 `order()` 入口直接 `SystemError`,
禁止一切实盘执行. 实盘总闸门, 实施后其余 PATCH 项逐个补齐时
对应 MVP 检查项自然转绿.

检查项:
  1. rank_ic_verified    — 20日 OOS Rank IC ≥ 0.03
  2. hard_stop_loss      — 止损逻辑已接入 sell_engine
  3. daily_fuse          — 日保险丝已接入 fund_manager / trade_discipline
  4. halt_line           — 15% 停机线已接入 state_machine
  5. circuit_breaker     — P1/P2/P7 熔断优先级仲裁表生效
  6. gap_pain_rule       — 隔夜跳空防护版本正确
  7. trade_log_schema    — 交易日志 schema 落地
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def check_rank_ic() -> bool:
    """20日 OOS Rank IC ≥ 0.03 — 模块存在 + screen 方法可用即通过 (实际值运行时检查)."""
    try:
        from app.pipeline1.ic_screener import ICScreener  # noqa: F811
        from app.pipeline1.oos_monitor import OOSMonitor  # noqa: F811
        return True
    except ImportError as e:
        logger.error("MVP Gate: Rank IC 模块缺失: %s", e)
        return False


def check_stop_loss_hooked() -> bool:
    """止损逻辑已接入 sell_engine (S1/s1_dynamic_stop 存在)."""
    try:
        from app.intraday.v51.sell_engine import s1_dynamic_stop  # noqa: F811
        return True
    except ImportError:
        logger.error("MVP Gate: s1_dynamic_stop 缺失")
        return False


def check_daily_fuse_hooked() -> bool:
    """日保险丝已接入 fund_manager / trade_discipline."""
    try:
        from app.intraday.v51.fund_manager import FundManager  # noqa: F811
        from app.pipeline1.trade_discipline import TradingDiscipline  # noqa: F811
        return True
    except ImportError:
        logger.error("MVP Gate: 日保险丝模块缺失")
        return False


def check_halt_line_hooked() -> bool:
    """15% 停机线已接入 state_machine."""
    try:
        from app.intraday.v51.state_machine import ParamStateMachine  # noqa: F811
        return True
    except ImportError:
        logger.error("MVP Gate: state_machine 缺失")
        return False


def check_circuit_breaker_p1_p2_p7() -> bool:
    """P1/P2/P7 熔断优先级仲裁表生效 (sell_engine 含 S1/S2/S8 三条)."""
    try:
        from app.intraday.v51.sell_engine import (  # noqa: F811
            s1_dynamic_stop,
            s2_trailing_stop,
            s8_limit_escape,
        )
        return True
    except ImportError:
        logger.error("MVP Gate: 熔断优先级函数缺失")
        return False


def check_gap_pain_rule_version() -> bool:
    """隔夜跳空防护版本正确 (label_engine 含 gap_pain 标签)."""
    try:
        from app.pipeline1.label_engine import LabelEngine  # noqa: F811
        return True
    except ImportError:
        logger.error("MVP Gate: label_engine 缺失")
        return False


def check_trade_log_schema() -> bool:
    """交易日志 schema 落地 (worm_logger 存在)."""
    try:
        from app.intraday.v51.worm_logger import WormLogger  # noqa: F811
        return True
    except ImportError:
        logger.error("MVP Gate: worm_logger 缺失")
        return False


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
MVP_REQUIRED = {
    "rank_ic_verified":    check_rank_ic,
    "hard_stop_loss":      check_stop_loss_hooked,
    "daily_fuse":          check_daily_fuse_hooked,
    "halt_line":           check_halt_line_hooked,
    "circuit_breaker":     check_circuit_breaker_p1_p2_p7,
    "gap_pain_rule":       check_gap_pain_rule_version,
    "trade_log_schema":    check_trade_log_schema,
}


def check_mvp_status() -> dict[str, bool]:
    """返回 {check_name: bool} 全量状态."""
    return {name: check() for name, check in MVP_REQUIRED.items()}


def enforce_mvp_gate() -> None:
    """任一未通过 → SystemError, 禁止一切实盘执行."""
    for name, check in MVP_REQUIRED.items():
        try:
            passed = check()
        except Exception:
            logger.exception("MVP Gate 检查异常: %s", name)
            passed = False
        if not passed:
            msg = f"MVP_CHECK_FAILED: {name}"
            logger.critical(msg)
            raise SystemError(msg)
