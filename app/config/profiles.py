"""
附录D 攻击参数档 (IMPLEMENTATION_PLAN_v3.2 P23.8, 分主板/双创)
==============================================================
定位: 与稳定档 (stable) 共用同一套引擎 — 数据管道、15维度特征、五模型矩阵、
训练排程、输出字段完全不变. 切档只改一组下游执行参数与三条硬规则.
目标函数: stable = Sortino 最大; aggressive = 盈亏比最大 (小资金收益弹性).

v3.2 变更:
  - 原 aggressive (cap 1.0) + aggressive_b (cap 0.75) → aggressive_main (cap 0.75)
  - 新增 aggressive_chinext (cap 0.50, _status: CANDIDATE_PENDING_D10)
  - 新增字段: stop_loss_atr_mult, daily_loss_limit → "2sigma_20d",
    time_stop_baseline, sector_min_return, trend_filter, gap_pain_prob_max

原则: 模型输出不变, 执行松紧度变. 改 bug 只改一处; 退回保守模式 = 改一行 PROFILE.
红线 (D.4, 不因资金小而豁免): 不上杠杆 / 不打板 / 熊市协议 (安全网#19)
对攻击档强制接管 (DEFENSE 只卖不买), 无豁免.
"""

from __future__ import annotations

COST = 0.0013  # round-trip 费用 (E5 口径, 与 list_generator 一致)

PROFILES: dict[str, dict] = {
    "stable": {
        "max_positions": 15,
        "single_cap": 0.10,
        "prob_entry": 0.60,
        "rotate_threshold": 2 * COST,
        "stop_loss": None,  # 交日内引擎默认规则
        "drawdown_limit": 0.15,
        "target_holding_days": (2.5, 3.5),
    },
    # 旧版 aggressive (cap 1.0) — 保留但标废弃, 生产禁选
    "aggressive": {
        "_status": "DEPRECATED",
        "_note": "v3.2 废弃, 用 aggressive_main 替代. C档(100%)待D.10解锁.",
        "max_positions": 1,
        "single_cap": 1.00,
        "grade_A_entry": {
            "prob_up_calibrated": 0.68,
            "rank_score_top": 2,
            "pain_prob_max": 0.15,
            "sector_resonance_top": 5,
            "main_board_only": True,
            "event_window_blacklist": True,
        },
        "rotate_threshold": 1.5 * COST,
        "stop_loss": -0.04,
        "drawdown_limit": 0.15,
        "daily_loss_limit": 0.04,
        "target_holding_days": (1, 2),
        "expected_idle_ratio": 0.75,
    },
    # B 档 (1只×75%, prob_entry 0.58): 旧版生产档 — 保留向后兼容
    "aggressive_b": {
        "max_positions": 1,
        "single_cap": 0.75,
        "prob_entry": 0.58,
        "rotate_threshold": 1.5 * COST,
        "stop_loss": -0.04,
        "drawdown_limit": 0.15,
        "daily_loss_limit": 0.03,
        "target_holding_days": (1, 2),
    },
    # ── v3.2 新版攻击档 ──
    "aggressive_main": {
        "board_type": "main",
        "max_positions": 1,
        "single_cap": 0.75,
        "grade_A_entry": {
            "prob_up_calibrated": 0.68,
            "rank_score_top": 2,
            "pain_prob_max": 0.15,
            "gap_pain_prob_max": 0.20,              # v3.2 新增
            "sector_resonance_top": 5,
            "sector_min_return": "median_250d",      # D-14, v3.2 新增
            "main_board_only": True,
            "event_window_blacklist": True,
            "trend_filter": "VWAP_SUPPORT",          # D-12, v3.2 新增
        },
        "stop_loss_fixed": -0.04,
        "stop_loss_atr_mult": 1.5,                   # D-20, v3.2 新增
        "daily_loss_limit": "2sigma_20d",            # D-22, v3.2 改
        "time_stop_baseline": "median_2d_return_20d",# D-21, v3.2 新增
        "drawdown_limit": 0.15,
        "target_holding_days": (1, 2),
        "leverage": 1.0,
    },
    "aggressive_chinext": {
        "_status": "CANDIDATE_PENDING_D10",          # 明确标注未解锁
        "board_type": "chinext",
        "max_positions": 1,
        "single_cap": 0.50,                          # 双创半仓
        "grade_A_entry": {
            "prob_up_calibrated": 0.72,              # 门槛提高
            "rank_score_top": 2,
            "pain_prob_max": 0.15,
            "gap_pain_prob_max": 0.20,
            "sector_resonance_top": 5,
            "sector_min_return": "median_250d",
            "main_board_only": False,
            "event_window_blacklist": True,
            "trend_filter": "VWAP_SUPPORT",
        },
        "stop_loss_fixed": -0.06,                    # 放宽
        "stop_loss_atr_mult": 1.5,
        "daily_loss_limit": "2sigma_20d",
        "time_stop_baseline": "median_2d_return_20d",
        "drawdown_limit": 0.15,
        "target_holding_days": (1, 2),
        "leverage": 1.0,
    },
}

# 切档的唯一入口 (D.1): 改这一行 = 全系统切档
# V3.8 → v3.2: 生产档 = aggressive_main (单票75%, 主板)
C_PROFILE_LOCKED = True
ACTIVE_PROFILE = "aggressive_main"

assert not (C_PROFILE_LOCKED and ACTIVE_PROFILE == "aggressive"), (
    "C档(100%)锁定: D.10裁决+40笔双闸门通过前不可启用为生产档"
)
assert ACTIVE_PROFILE != "aggressive_chinext" or not C_PROFILE_LOCKED, (
    "双创攻击档 agressive_chinext 待 D.10 裁决解锁, 不可作为生产档"
)


def get_profile(name: str | None = None, allow_deprecated: bool = True) -> dict:
    """取档位参数 (默认当前激活档). name 非法 → KeyError (失败要大声)."""
    key = name or ACTIVE_PROFILE
    if key not in PROFILES:
        raise KeyError(f"未知档位: {key} (可选: {list(PROFILES)})")
    profile = PROFILES[key]
    if profile.get("_status") == "DEPRECATED" and not allow_deprecated:
        raise KeyError(f"废弃档位禁止使用: {key}")
    if profile.get("_status") == "DEPRECATED":
        import logging
        logging.getLogger(__name__).warning("使用废弃档位: %s, 请迁移至 aggressive_main", key)
    if profile.get("_status") == "CANDIDATE_PENDING_D10":
        import logging
        logging.getLogger(__name__).warning("双创攻击档 %s 待 D.10 裁决解锁", key)
    return profile


def is_aggressive(name: str | None = None) -> bool:
    """是否攻击档 (stable 之外的狙击/回退档)."""
    return (name or ACTIVE_PROFILE) != "stable"


def _resolve_daily_loss_limit(value) -> float:
    """兼容 daily_loss_limit 的两种格式: 数值型 (固定阈值) 和字符串型 (如 '2sigma_20d')."""
    if isinstance(value, (int, float)):
        return float(value)
    # 字符串型 → 返回 0 表示使用自适应逻辑, 由 trade_discipline.py 的 daily_fuse() 处理
    return 0.03


def resolve_live_profile(journal=None, d10_c_approved: bool = False) -> str:
    """P21.3 实盘解锁双闸门 (D.10): 前 40 笔实盘按 B/aggressive_main 档执行;
    D.10 回测裁决选 C 且 40 笔样本满足 期望>+0.5%/笔 且 最大连亏≤5,
    方可切 C 档. 任一闸门不过 → "aggressive_main" (失败要大声, 默认保守).

    v3.2: aggressive_chinext 额外需要 d10_dual_approved=True (独立裁决).
    """
    if not d10_c_approved:
        return "aggressive_main"
    if journal is None:
        return "aggressive_main"
    return "aggressive" if journal.unlock_check()["unlock"] else "aggressive_main"


def resolve_board_type(profile_name: str | None = None) -> str:
    """从 profile 解析 board_type ('main' | 'chinext' | 'unknown')."""
    profile = get_profile(profile_name)
    return profile.get("board_type", "unknown")
