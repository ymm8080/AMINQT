"""
附录D 攻击参数档 (PIPELINE1_V3.8 附录D.1, 检查清单 D-1)
==========================================================
定位: 与稳定档 (stable) 共用同一套引擎 — 数据管道、15维度特征、五模型矩阵、
训练排程、输出字段完全不变. 切档只改一组下游执行参数与三条硬规则.
目标函数: stable = Sortino 最大; aggressive = 盈亏比最大 (小资金收益弹性).

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
    # 狙击满仓档 C档 (1只×100%): 锁定待 D.10 裁决 + 实盘40笔双闸门解锁 (P21.0)
    "aggressive": {
        "max_positions": 1,
        "single_cap": 1.00,  # 仅 A 级信号允许满仓
        "grade_A_entry": {  # 入场门槛 (全部满足, 缺一不可)
            "prob_up_calibrated": 0.68,  # 校准后概率 (Isotonic); >0.70 尾部失真不抬高
            "rank_score_top": 2,  # 数量闸门: 每日最多全市场前 2 名
            "pain_prob_max": 0.15,
            "sector_resonance_top": 5,  # 板块涨幅全市场前 5
            "main_board_only": True,  # 跌停 -10% 封顶
            "event_window_blacklist": True,  # 财报/解禁/待披露公告禁买
        },
        "rotate_threshold": 1.5 * COST,
        "stop_loss": -0.04,  # D.3 止损硬化: 单笔 -4% 无条件砍
        "drawdown_limit": 0.15,  # 停机线 -15% (用户定稿)
        "daily_loss_limit": 0.04,  # 100%×-4%=-4%, 一笔止损即当日收工
        "target_holding_days": (1, 2),
        "expected_idle_ratio": 0.75,  # 门槛 0.68+Top2 → 预计 75% 交易日空仓 (特性非故障)
    },
    # B 档 (1只×75%, prob_entry 0.58): 生产档 (V3.8 定稿); 兼 D.10 回退/验证档
    "aggressive_b": {
        "max_positions": 1,
        "single_cap": 0.75,
        "prob_entry": 0.58,
        "rotate_threshold": 1.5 * COST,
        "stop_loss": -0.04,
        "drawdown_limit": 0.15,  # 停机线与狙击档一致
        "daily_loss_limit": 0.03,
        "target_holding_days": (1, 2),
    },
}

# 切档的唯一入口 (D.1): 改这一行 = 全系统切档
# V3.8 定稿: 生产档 = B档 (单票75%); C档 (100%) 锁定待 D.10 裁决 +
# 实盘40笔双闸门 (期望>+0.5%/笔 且 最大连亏≤5) 解锁, 此前 ACTIVE_PROFILE
# 不得指向 "aggressive" (参数读取用于影子清单/D.10回测不受限).
C_PROFILE_LOCKED = True
ACTIVE_PROFILE = "aggressive_b"

assert not (
    C_PROFILE_LOCKED and ACTIVE_PROFILE == "aggressive"
), "C档(100%)锁定: D.10裁决+40笔双闸门通过前不可启用为生产档"


def get_profile(name: str | None = None) -> dict:
    """取档位参数 (默认当前激活档). name 非法 → KeyError (失败要大声)."""
    key = name or ACTIVE_PROFILE
    if key not in PROFILES:
        raise KeyError(f"未知档位: {key} (可选: {list(PROFILES)})")
    return PROFILES[key]


def is_aggressive(name: str | None = None) -> bool:
    """是否攻击档 (stable 之外的狙击/回退档)."""
    return (name or ACTIVE_PROFILE) != "stable"
