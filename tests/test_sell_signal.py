# -*- coding: utf-8 -*-
"""卖出信号评估器单元测试: 级别判定 + 优先级 + 价格硬止损 + 缺失列安全."""

import pandas as pd

from app.pipeline1.sell_signal import (
    PRICE_HARD_STOP,
    evaluate_sell_signal,
)


def _row(**kw):
    base = {
        "symbol": "600519",
        "pred_ret_1d": 0.01,
        "pred_ret_3d": 0.03,
        "pred_ret_5d": 0.05,
        "prob_up": 0.62,
        "pain_prob": 0.10,
        "pred_q10": -0.005,
    }
    base.update(kw)
    return base


def test_hold_when_all_positive():
    out = evaluate_sell_signal(pd.DataFrame([_row()]))
    assert out.loc[0, "sell_signal"] == "hold"
    assert out.loc[0, "sell_reason"] == "持有"


def test_watch_when_1d_turns_negative():
    # -0.3% 在黄色区间 (-0.5% < r1 < 0), 未到橙/红
    out = evaluate_sell_signal(pd.DataFrame([_row(pred_ret_1d=-0.003, prob_up=0.60)]))
    assert out.loc[0, "sell_signal"] == "watch"


def test_watch_when_prob_below_coin_flip():
    out = evaluate_sell_signal(pd.DataFrame([_row(prob_up=0.48)]))
    assert out.loc[0, "sell_signal"] == "watch"


def test_sell_when_prob_below_threshold():
    out = evaluate_sell_signal(pd.DataFrame([_row(prob_up=0.42)]))
    assert out.loc[0, "sell_signal"] == "sell"


def test_sell_when_1d_below_minus_half_percent():
    out = evaluate_sell_signal(pd.DataFrame([_row(pred_ret_1d=-0.006)]))
    assert out.loc[0, "sell_signal"] == "sell"


def test_strong_sell_when_1d_below_minus_1_5_percent():
    out = evaluate_sell_signal(pd.DataFrame([_row(pred_ret_1d=-0.02)]))
    assert out.loc[0, "sell_signal"] == "strong_sell"
    assert "次日预期" in out.loc[0, "sell_reason"]


def test_strong_sell_when_3d_negative_and_1d_negative():
    out = evaluate_sell_signal(
        pd.DataFrame([_row(pred_ret_1d=-0.005, pred_ret_3d=-0.025)])
    )
    assert out.loc[0, "sell_signal"] == "strong_sell"


def test_strong_sell_when_pain_high():
    out = evaluate_sell_signal(pd.DataFrame([_row(pain_prob=0.65)]))
    assert out.loc[0, "sell_signal"] == "strong_sell"


def test_strong_sell_when_q10_very_negative():
    out = evaluate_sell_signal(pd.DataFrame([_row(pred_q10=-0.04)]))
    assert out.loc[0, "sell_signal"] == "strong_sell"


def test_price_hard_stop_overrides_positive_prediction():
    # 预测全绿, 但已跌 7% > 6% → 硬止损红/强卖
    out = evaluate_sell_signal(pd.DataFrame([_row(pnl=-0.07)]), pnl_col="pnl")
    assert out.loc[0, "sell_signal"] == "strong_sell"
    assert "硬止损" in out.loc[0, "sell_reason"]


def test_price_hard_stop_boundary():
    # 恰好 -6% 触发, -5.9% 不触发
    out = evaluate_sell_signal(pd.DataFrame([_row(pnl=-0.06)]), pnl_col="pnl")
    assert out.loc[0, "sell_signal"] == "strong_sell"
    out = evaluate_sell_signal(pd.DataFrame([_row(pnl=-0.059)]), pnl_col="pnl")
    assert out.loc[0, "sell_signal"] != "strong_sell"


def test_missing_columns_safe():
    # 只有核心两列, 其余按中性值处理, 不应报错
    out = evaluate_sell_signal(
        pd.DataFrame([{"symbol": "600519", "pred_ret_1d": 0.01, "prob_up": 0.60}])
    )
    assert out.loc[0, "sell_signal"] in ("hold", "watch")
    assert out.loc[0, "sell_reason"]


def test_red_wins_over_orange_and_yellow():
    # 同票同时命中 红(pain 0.65) / 橙(pred_ret_1d -0.5%) / 黄(prob 0.48) → 红
    out = evaluate_sell_signal(
        pd.DataFrame([_row(pred_ret_1d=-0.005, prob_up=0.48, pain_prob=0.65)])
    )
    assert out.loc[0, "sell_signal"] == "strong_sell"


def test_multiple_symbols_vectorized():
    rows = [
        _row(symbol="600000", pred_ret_1d=-0.02),
        _row(symbol="600001", prob_up=0.43),
        _row(symbol="600002", pred_ret_1d=-0.003),
        _row(symbol="600003"),
    ]
    out = evaluate_sell_signal(pd.DataFrame(rows))
    got = dict(zip(out["symbol"], out["sell_signal"]))
    assert got == {
        "600000": "strong_sell",
        "600001": "sell",
        "600002": "watch",
        "600003": "hold",
    }
    assert len(out["sell_reason"]) == 4


def test_default_stop_is_minus_6():
    assert PRICE_HARD_STOP == -0.06
