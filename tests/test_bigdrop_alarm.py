"""bigdrop 次日大跌模块单测 (2026-09-15).

覆盖两处 0915 改动:
  1. 第 9 条闸「自由流通换手 > 10%」—— 补 600814(20260910) 那类「处处偏高但
     无一极端」的中空形态 (加它时原有 9 条闸全是极端值阈值, 对其全盲; 0915 撤
     连板高位后, 它排在 9 条闸的末位)。
  2. 模型报警通道 should_alarm —— 规则 0 分而模型报警时必须举手, 且不重复报警。
"""

import numpy as np
import pandas as pd
import pytest

from scripts.bigdrop_check import (
    BRANCH_DIRECTIONAL,
    BRANCH_NONE,
    BRANCH_VOLATILITY,
    CAPTURE_TIERS,
    MODEL_ALARM,
    capture_tier,
    effective_prob,
    rule_flags,
    should_alarm,
    signal_kind,
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


def test_rule_count_is_nine():
    # 闸数变=得分档口径变, 必须是有意的 (0915 撤连板高位: 10 -> 9)
    assert len(rule_flags(pd.DataFrame([SAFE])).columns) == 9


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


def test_score_frame_rejects_rule_count_mismatch(capsys):
    """闸数与包不一致必须当场炸 —— 0915 撤闸事故的守卫。

    rule_table 的键是**建包时**的得分刻度, 取用时拿当场新算的 sc 去索引: 10 闸的包
    配 9 闸的代码, 高档会静默读错 (缺键还回退 base_pre), 融合头同理, 而全文件没有
    任何断言。这里给一个 10 闸假包, 必须在碰 booster **之前** 退出 —— 所以包里故意
    不放 booster。
    """
    from scripts.bigdrop_check import score_frame

    d = pd.DataFrame([SAFE])
    with pytest.raises(SystemExit) as ei:
        score_frame(d, {"rules": [("x",)] * 10, "tag": "20260901"})
    assert ei.value.code == 2
    out = capsys.readouterr().out
    assert "闸数不一致" in out
    assert "包 10 闸" in out and "代码 9 闸" in out


def test_score_frame_passes_guard_when_counts_match():
    """正控: 闸数一致时守卫放行 —— 随后因缺 booster 才 KeyError, 证明确实越过了守卫。"""
    from scripts.bigdrop_check import score_frame

    with pytest.raises(KeyError):
        score_frame(pd.DataFrame([SAFE]), {"rules": [("x",)] * 9, "tag": "20260915"})


def test_score_frame_guard_skips_old_bundle_without_rules():
    """旧包没有 rules 键 —— 按既有惯例优雅降级 (同 alarm/branches/capture), 不炸。"""
    from scripts.bigdrop_check import score_frame

    with pytest.raises(KeyError):
        score_frame(pd.DataFrame([SAFE]), {"tag": "20260901"})


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


# ---- 信号分支 (0915: 报警拆成"带方向"与"不带方向"两种读法) ----


def test_signal_kind_model_branch_is_directional():
    """模型过线即带方向 —— 哪怕规则也吵得很凶 (都喊格次日均 -0.26%)。"""
    assert signal_kind(0, MODEL_ALARM) == (BRANCH_DIRECTIONAL, True)
    assert signal_kind(3, 0.30) == (BRANCH_DIRECTIONAL, True)
    assert signal_kind(0, 0.102) == (BRANCH_DIRECTIONAL, True)


def test_signal_kind_rules_only_branch_claims_no_direction():
    """规则喊而模型沉默 = 波动/跌停风险, 不得声称方向 (该格次日均 +0.06%)。"""
    name, has_dir = signal_kind(1, 0.001)
    assert name == BRANCH_VOLATILITY
    assert not has_dir
    assert signal_kind(4, 0.099)[0] == BRANCH_VOLATILITY


def test_signal_kind_quiet_row():
    assert signal_kind(0, 0.0) == (BRANCH_NONE, False)
    assert signal_kind(0, MODEL_ALARM - 1e-6) == (BRANCH_NONE, False)
    assert not signal_kind(0, 0.0)[1]


def test_signal_kind_threshold_is_inclusive_like_should_alarm():
    """与 should_alarm 同一条线, 不能一个 >= 一个 >。"""
    for mp in (MODEL_ALARM, 0.30):
        assert signal_kind(0, mp)[1] is should_alarm(mp, 0)
    assert signal_kind(0, MODEL_ALARM - 1e-6)[1] is should_alarm(MODEL_ALARM - 1e-6, 0)


def test_signal_kind_partitions_every_row():
    """三分支互斥且穷尽; 带方向的充要条件就是走模型支。"""
    seen = set()
    for sc in range(0, 11):
        for mp in (0.0, 0.05, 0.099, 0.10, 0.5):
            name, has_dir = signal_kind(sc, mp)
            seen.add(name)
            assert has_dir == (mp >= MODEL_ALARM), f"score={sc} model_p={mp}"
    assert seen == {BRANCH_DIRECTIONAL, BRANCH_VOLATILITY, BRANCH_NONE}


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


# ---- build() 建包路径冒烟 (纯函数单测够不到的那一段) ----


def _synthetic_panel(n_per_day: int = 800, seed: int = 7) -> "pd.DataFrame":
    """造一个三段齐备的小面板, 只为让 build() 的真实路径跑起来。

    规则列都用各自量纲造出厚尾 (冒烟只要求得分档铺得开, 不追求真实兑现率);
    fwd 强制掺入 5% 的 <= -9.5%, 否则涨跌停那条基准线样本为 0 会除零。
    """
    from scripts.bigdrop_check import FEATS

    rng = np.random.default_rng(seed)
    dates = (
        [f"2024010{d}" for d in range(2, 10)]  # FIT   < 20250701
        + [f"2025070{d}" for d in range(1, 10)]  # CALIB 20250701~20251231
        + [f"2026010{d}" for d in range(2, 10)]  # OOS   >= 20260101
    )
    frames = []
    for dt in dates:
        n = n_per_day
        f: dict = {c: rng.normal(0.0, 1.0, n) for c in FEATS}
        f["bias_20"] = rng.normal(0.10, 0.15, n)
        f["bias_60"] = rng.normal(0.20, 0.30, n)
        f["turnover_rate"] = rng.normal(8.0, 10.0, n)
        f["free_float_turnover_rate"] = rng.normal(6.0, 5.0, n)
        f["volume_ratio"] = rng.normal(1.2, 0.8, n)
        f["pctChg"] = rng.normal(0.0, 3.0, n)
        f["low60_gain"] = rng.normal(0.25, 0.25, n)
        f["ret10"] = rng.normal(0.10, 0.10, n)
        f["zt_break"] = (rng.random(n) < 0.06).astype(int)
        f["conseq_zt"] = rng.integers(0, 5, n)

        fwd = rng.normal(0.0, 0.04, n)
        tail = rng.random(n) < 0.05
        fwd[tail] = -rng.uniform(0.095, 0.14, int(tail.sum()))

        f["date"] = dt
        f["fwd"] = fwd
        frames.append(pd.DataFrame(f))

    d = pd.concat(frames, ignore_index=True)
    d["y"] = (d["fwd"] <= -0.05).astype(float)
    return d


def test_build_runs_end_to_end(tmp_path, monkeypatch):
    """build() 必须跑通并产出完整 bundle —— 纯函数单测覆盖不到的建包路径。

    回归: 捕获阶梯循环曾用 `m` 承接样本掩码, 覆盖了上面的 LGBM 模型变量,
    建包崩在 booster=m.booster_ 处, bundle 完全写不出来 —— 当时 19 条单测
    全绿也照样没拦住, 因为它们只碰纯函数。
    """
    from scripts import bigdrop_check as bd

    monkeypatch.setattr(bd, "load_frame", lambda: _synthetic_panel())
    monkeypatch.setattr(bd, "BUNDLE_DIR", tmp_path)

    b = bd.build()

    for key in (
        "booster",
        "iso",
        "lr",
        "rules",
        "rule_table",
        "alarm",
        "capture",
        "branches",
    ):
        assert key in b, f"bundle 缺 {key}"
    assert len(b["rules"]) == 9

    # 遮蔽 bug 的直接断言: 存进去的必须是模型对象, 不是被覆盖的 ndarray
    assert not isinstance(b["booster"], np.ndarray)
    assert hasattr(b["booster"], "predict")

    # 阶梯嵌套 => 召回随档位放宽单调不降
    assert b["capture"][-1]["name"] == "T4 未标"
    rec = [c["bigdrop_recall"] for c in b["capture"][:-1]]
    assert rec == sorted(rec)

    # 分支表: 三分支穷尽 OOS 面 (份额和 = 1), 且方向支的次日均收益必须存在
    brs = b["branches"]
    assert [x["name"] for x in brs] == [
        BRANCH_DIRECTIONAL,
        BRANCH_VOLATILITY,
        BRANCH_NONE,
    ]
    assert abs(sum(x["share"] for x in brs) - 1.0) < 1e-9
    assert all(np.isfinite(x["ret"]) for x in brs)

    # 落盘到 tmp_path, 绝不碰真 artifact (WORM)
    assert (tmp_path / "bundle_latest.joblib").exists()
    assert list(tmp_path.glob("bundle_*.joblib"))
