"""
参数部署四态流转 (V5.1 §9, 检查清单 #15)
==================================================
candidate (季度寻优产出)
    ↓ 人工看板确认
shadow (2-4 周, 只记录不交易)
    ↓ 满足 4 条件: 扣费后超额>0 / 放弃率<20% / 无数据故障 / 回测与shadow逐笔一致率>99%
staging (3-5 个交易日, 半自动小规模验证)
    ↓ 表现无异常
active (生产参数)
任何阶段出现异常 → 回退 candidate 并书面归因.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

STATE_CANDIDATE = "candidate"
STATE_SHADOW = "shadow"
STATE_STAGING = "staging"
STATE_ACTIVE = "active"
STATES = (STATE_CANDIDATE, STATE_SHADOW, STATE_STAGING, STATE_ACTIVE)

# Shadow 门禁 (4 条件全满足)
SHADOW_MIN_EXCESS = 0.0  # 扣费后超额 > 0
SHADOW_MAX_VETO_RATE = 0.20  # 放弃率 < 20%
SHADOW_MIN_MATCH_RATE = 0.99  # 回测与 shadow 逐笔一致率 > 99%


def shadow_gate(report: dict) -> dict:
    """Shadow → staging 门禁 (4 条件全满足).

    report: {'excess_return', 'veto_rate', 'data_failures', 'match_rate'}
    """
    checks = {
        "excess_positive": report["excess_return"] > SHADOW_MIN_EXCESS,
        "veto_rate_ok": report["veto_rate"] < SHADOW_MAX_VETO_RATE,
        "no_data_failure": report["data_failures"] == 0,
        "match_rate_ok": report["match_rate"] > SHADOW_MIN_MATCH_RATE,
    }
    return {"pass": all(checks.values()), "checks": checks}


class ParamStateMachine:
    """单组参数的四态流转."""

    def __init__(self):
        self.state = STATE_CANDIDATE

    def confirm_by_human(self) -> str:
        """candidate → shadow (人工看板确认, 唯一入口)."""
        assert self.state == STATE_CANDIDATE
        self.state = STATE_SHADOW
        logger.info("参数流转: candidate → shadow (人工确认)")
        return self.state

    def promote_from_shadow(self, report: dict) -> str:
        """shadow → staging (门禁 4 条件) 或回退 candidate."""
        assert self.state == STATE_SHADOW
        gate = shadow_gate(report)
        if gate["pass"]:
            self.state = STATE_STAGING
            logger.info("参数流转: shadow → staging (门禁通过)")
        else:
            self.state = STATE_CANDIDATE
            logger.error(
                "参数回退: shadow → candidate (门禁失败 %s), 书面归因",
                [k for k, v in gate["checks"].items() if not v],
            )
        return self.state

    def promote_from_staging(self, abnormal: bool) -> str:
        """staging → active (无异常) 或回退 candidate."""
        assert self.state == STATE_STAGING
        if abnormal:
            self.state = STATE_CANDIDATE
            logger.error("参数回退: staging 异常 → candidate, 书面归因")
        else:
            self.state = STATE_ACTIVE
            logger.info("参数流转: staging → active (生产上线)")
        return self.state

    def demote(self, reason: str) -> str:
        """任何状态 → candidate (异常回退, 书面归因)."""
        logger.error("参数回退: %s → candidate (%s)", self.state, reason)
        self.state = STATE_CANDIDATE
        return self.state
