"""bigdrop 次日大跌模块单测 (2026-09-15).

覆盖两处 0915 改动:
  1. 第 10 条闸「自由流通换手 > 10%」—— 补 600814(20260910) 那类「处处偏高但
     无一极端」的中空形态 (原 9 条闸全是极端值阈值, 对其全盲)。
  2. 模型报警通道 should_alarm —— 规则 0 分而模型报警时必须举手, 且不重复报警。
"""

import numpy as np
import pandas as pd

from scripts.bigdrop_check import (
    CAPTURE_TIERS,
    MODEL_ALARM,
    capture_tier,
    effective_prob,
    rule_flags,
    should_alarm,
)

# build() 写进 bundle 的报警格兑现 (20260915 包)
ALARM = {"th": 0.10, "oos_n": 8390, "oos_prec": 0.09, "oos_lift": 1.74, "daily": 49.6}

# 各闸均不触发的安全底样
SAFE = {
    "bias_20": 0.0,
    "bias_60": 0.0,
    "turnover_rate": 1.0,
    "zt_break": 0,
    "conseq_zt": 0,
    "volume_ratio": 1.0,
    "pctChg": 0.0,
    "low60_gain": 0.0,
    "ret10": 0.0,
    "free_float_turnover_rate": 1.0,
}

# 600814 于 20260910 的实测值 (次日 -7.33%)
_S600814 = {
    "free_float_turnover_rate": 13.51,
    "ret10": 0.1235,
    "bias_20": 0.0902,
    "bias_60": 0.1531,
    "low60_gain": 0.2679,
    "turnover_rate": 4.54,
    "volume_ratio": 1.12,
    "pctChg": -3.17,
}


def _flags(**over):
    return rule_flags(pd.DataFrame([{**SAFE, **over}])).iloc[0]


def test_rule_count_is_ten():
    # 闸数变=得分档口径变, 必须是有意的
    assert len(rule_flags(pd.DataFrame([SAFE])).columns) == 10


def test_ff_turn_threshold_boundary():
    assert bool(_flags(free_float_turnover_rate=13.51)["hi_ff_turn"])
    assert not bool(_flags(free_float_turnover_rate=10.0)["hi_ff_turn"])
    assert not bool(_flags(free_float_turnover_rate=9.99)["hi_ff_turn"])


def test_ff_turn_nan_does_not_fire():
    # 面板缺值不得静默触发 (NaN 比较为 False)
    assert not bool(
        rule_flags(pd.DataFrame([{**SAFE, "free_float_turnover_rate": np.nan}])).iloc[
            0
        ]["hi_ff_turn"]
    )


def test_safe_row_scores_zero():
    assert int(rule_flags(pd.DataFrame([SAFE])).sum(axis=1).iloc[0]) == 0


def test_600814_profile_now_trips_exactly_one_rule():
    """0910 该股原 9 条闸全够不着 —— 新闸必须是唯一命中的那条。"""
    f = _flags(**_S600814)
    assert bool(f["hi_ff_turn"])
    assert int(f.sum()) == 1
    assert not bool(
        f[
            [
                "hi_bias20",
                "hi_bias60",
                "huge_turn",
                "vol_surge",
                "today_drop",
                "big_rise",
                "rise10",
            ]
        ].any()
    )


def test_alarm_fires_when_rules_silent():
    assert should_alarm(MODEL_ALARM, 0)
    assert should_alarm(0.30, 0)
    # 600814 当日模型值
    assert should_alarm(0.102, 0)


def test_alarm_low_model_prob_is_quiet():
    assert not should_alarm(MODEL_ALARM - 1e-6, 0)
    assert not should_alarm(0.02, 0)


def test_alarm_does_not_duplicate_rule_signal():
    # 规则已举手时规则表概率本身已抬升, 报警通道不重复
    assert not should_alarm(0.90, 1)
    assert not should_alarm(0.90, 3)


def test_effective_prob_uses_alarm_cell_when_rules_silent():
    """并集落到头条: 规则沉默而模型报警时, 概率取报警格兑现, 不取规则表低分档。"""
    p, warn = effective_prob(0.017, 0.125, 0, ALARM)
    assert warn
    assert p == ALARM["oos_prec"]


def test_effective_prob_keeps_rule_table_when_not_alarming():
    # 规则沉默但模型不够响 → 规则表概率就是对的, 不加戏
    p, warn = effective_prob(0.017, 0.03, 0, ALARM)
    assert not warn
    assert p == 0.017
    # 规则已举手 → 规则表概率本身已抬升, 不被报警格顶掉
    p, warn = effective_prob(0.31, 0.60, 2, ALARM)
    assert not warn
    assert p == 0.31


def test_effective_prob_without_alarm_block_falls_back():
    # 旧 bundle 没有 alarm 键时不得崩, 退回规则表值
    p, warn = effective_prob(0.017, 0.125, 0, {})
    assert not warn
    assert p == 0.017


# ---- 捕获阶梯 (0915 用户令: 抓住所有跌停股和大跌股) ----


def test_capture_tiers_are_nested_by_surface():
    """档越靠后表面积越大 —— 否则阶梯毫无意义。"""
    qs = [q for _, _, q in CAPTURE_TIERS]
    assert qs[0] is None  # T1 用绝对报警线, 其余用当日分位
    assert qs[1:] == sorted(qs[1:]), f"分位必须单调放宽, 实为 {qs}"


def test_capture_tier_rule_hit_lands_in_t1():
    # 规则够分即入 T1 报警档 (与模型无关)
    assert capture_tier(2, 0.001, 0.01)[0] == "T1 报警"
    assert capture_tier(1, 0.001, 0.01)[1] == 1


def test_capture_tier_model_alarm_lands_in_t1():
    # 规则 0 分但模型过 T1 绝对线 —— 并集口径, 仍入 T1
    assert capture_tier(0, MODEL_ALARM, 0.5)[0] == "T1 报警"


def test_capture_tier_widens_by_day_rank():
    """规则沉默时, 档位只由当日分位决定 —— T2/T3/未标逐级下降。"""
    assert capture_tier(0, 0.05, 0.85)[0] == "T2 警戒"  # 前 15% → 落在前20%档
    assert capture_tier(0, 0.05, 0.70)[0] == "T3 关注"  # 前 30% → 落在前40%档
    assert capture_tier(0, 0.01, 0.10)[0] == "T4 未标"


def test_capture_tier_boundary_is_inclusive_at_tier_edge():
    # 边界: 恰好前 20% 应当入 T2 (>=), 差一点点掉出
    assert capture_tier(0, 0.0, 0.80)[0] == "T2 警戒"
    assert capture_tier(0, 0.0, 0.7999)[0] == "T3 关注"


def test_capture_tier_never_returns_beyond_ladder():
    name, num = capture_tier(0, 0.0, 0.0)
    assert name == "T4 未标"
    assert num == len(CAPTURE_TIERS) + 1


# ---- 从 bundle 复读的只读视图 (不读面板, 因此可以直接跑) ----

_MIN_BUNDLE = {
    "tag": "20260915",
    "winner": "纯规则表",
    "oos_base": 0.033954,
    "oos_brier": {"纯规则表": 0.03015, "模型(isotonic校准)": 0.03080},
    "oos_cal_err": {"纯规则表": 0.0149, "模型(isotonic校准)": 0.0193},
    "oos_auc": 0.8554,
    "split": {"fit_end": "20250701", "cal_end": "20260101", "embargo": 5},
}


def test_show_compare_marks_winner_and_prints_every_caliber(capsys):
    """--compare 只读 bundle, 不得 KeyError; winner 必须被标出来。"""
    from scripts.bigdrop_check import show_compare

    show_compare(_MIN_BUNDLE)
    out = capsys.readouterr().out
    for n in _MIN_BUNDLE["oos_brier"]:
        assert n in out
    assert "← winner" in out
    assert out.count("← winner") == 1  # 有且只有一个赢家
    assert "0.8554" in out


def test_show_capture_degrades_on_old_bundle(capsys):
    """0915 之前的包没有 capture 块 —— 要给出提示而不是崩。"""
    from scripts.bigdrop_check import show_capture

    show_capture({"tag": "20260910"})
    assert "没有 capture 块" in capsys.readouterr().out
