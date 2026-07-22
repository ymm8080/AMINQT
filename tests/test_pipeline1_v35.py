# -*- coding: utf-8 -*-
"""Pipeline-1 V3.5 模块测试 (安全网 #0-#14 关键路径)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.pipeline1.cleaning_pipeline import (
    CleaningConfig,
    CleaningPipeline,
    board_of,
    get_limit_pct,
    is_limit_up,
    limit_up_price,
)
from app.pipeline1.label_engine import LabelEngine
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.ic_screener import ICScreener
from app.pipeline1.list_generator import (
    ListDeliveryGuard,
    ListGenerator,
    MarketEnv,
    SCHEMA_FIELDS,
    check_invalidation,
)
from app.pipeline1.oos_monitor import OOSMonitor
from app.pipeline1.prob_calibrator import ProbCalibrator


# ============================================================
# 合成数据
# ============================================================
def make_panel(
    symbols=("600519", "300750", "601318"), days=300, seed=42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=days)
    frames = []
    for i, sym in enumerate(symbols):
        board = board_of(sym)
        close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, days))
        open_ = close * (1 + rng.normal(0, 0.005, days))
        high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.005, days)))
        low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.005, days)))
        pre_close = np.concatenate([[close[0]], close[:-1]])
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates,
                    "board": board,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "close_hfq": close,
                    "high_hfq": high,
                    "low_hfq": low,
                    "open_hfq": open_,
                    "volume": rng.integers(1e6, 1e8, days).astype(float),
                    "amount": rng.uniform(6e7, 2e9, days),
                    "turnover_rate": rng.uniform(1, 10, days),
                    "free_float_turnover_rate": rng.uniform(1, 10, days),
                    "pre_close": pre_close,
                    "is_suspended": False,
                    "is_st": False,
                    "industry": "白酒"
                    if sym == "600519"
                    else ("电池" if sym == "300750" else "保险"),
                    "list_days": 1000,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# ============================================================
# 清洗管线
# ============================================================
class TestCleaning:
    def test_board_split(self):
        assert board_of("600519") == "main"
        assert board_of("300750") == "GEM"
        assert board_of("688981") == "STAR"

    def test_limit_pct_segmented(self):
        """安全网 #6: 创业板 2020-08-24 分段."""
        assert get_limit_pct("main", pd.Timestamp("2020-08-23")) == 0.10
        assert get_limit_pct("GEM", pd.Timestamp("2020-08-23")) == 0.10
        assert get_limit_pct("GEM", pd.Timestamp("2020-08-24")) == 0.20
        assert get_limit_pct("STAR", pd.Timestamp("2019-01-01")) == 0.20
        with pytest.raises(ValueError):
            get_limit_pct("BSE", pd.Timestamp("2024-01-01"))

    def test_limit_up_precise(self):
        """涨停价精确比对 round(pre_close*(1+pct),2), 容差 <0.01 (即±1分内算涨停)."""
        assert limit_up_price(10.03, 0.10) == 11.03
        assert is_limit_up(11.03, 10.03, 0.10)
        assert not is_limit_up(11.01, 10.03, 0.10)  # 差2分, 非涨停

    def test_step1_st_and_list_days(self):
        df = make_panel()
        df.loc[df["symbol"] == "600519", "is_st"] = True
        df.loc[df["symbol"] == "300750", "list_days"] = 100
        out = CleaningPipeline().step1_base_state(df)
        assert set(out["symbol"]) == {"601318"}

    def test_step2_liquidity_and_stability(self):
        """成交额下限 + Score 前N + D24 换手稳定性."""
        df = make_panel(days=60)
        df.loc[df["symbol"] == "601318", "amount"] = 1e6  # < 5000万
        cfg = CleaningConfig(liquidity_top_n=1)
        out = CleaningPipeline(cfg).step2_liquidity(df)
        assert "601318" not in set(out["symbol"])
        assert "turnover_stability_5" in out.columns
        assert out.groupby(["date", "board"])["symbol"].count().max() <= 1

    def test_step3_resume_first_day(self):
        """安全网 #11: 复牌首日剔除, 复牌次日纳入."""
        df = make_panel(symbols=("600519",), days=30)
        df.loc[10, "is_suspended"] = True
        out = CleaningPipeline().step3_extreme(df)
        dates = out["date"].tolist()
        assert df["date"].iloc[10] not in dates  # 停牌日
        assert df["date"].iloc[11] not in dates  # 复牌首日
        assert df["date"].iloc[12] in dates  # 复牌次日纳入

    def test_step4_one_word_and_valve(self):
        """安全网 #8: 一字涨停剔除 + 8000万安全阀 + <15 强制空清单."""
        df = make_panel(symbols=("600519", "300750"), days=5)
        d = df["date"].max()
        today = df[df["date"] == d].copy()
        # 600519 一字涨停
        row = today[today["symbol"] == "600519"].index[0]
        lu = round(today.loc[row, "pre_close"] * 1.10, 2)
        today.loc[row, ["open", "high", "low", "close"]] = lu
        cfg = CleaningConfig(abs_amount_floor=8e7, valve_full=50, valve_reduced=15)
        out, state = CleaningPipeline(cfg).step4_tradability(today, inference_only=True)
        assert "600519" not in set(out["symbol"])
        assert state == "empty"  # 剩 1 只 < 15

    def test_delisted_virtual_rows(self):
        """安全网 #14: 退市股虚拟 T+1 = 收盘×0.5."""
        df = make_panel(symbols=("600519",), days=30)
        out = CleaningPipeline().inject_delisted_virtual_rows(df, ["600519"])
        last = out[out["symbol"] == "600519"].iloc[-1]
        prev_close = df["close"].iloc[-1]
        assert last["close"] == pytest.approx(prev_close * 0.5)
        assert len(out) == 31


# ============================================================
# 标签引擎
# ============================================================
class TestLabels:
    def test_labels_groupby_no_cross_stock(self):
        """安全网 #5/#13: label 必须按 symbol 分组, 不串股."""
        df = make_panel(symbols=("600519", "300750"), days=30)
        out = LabelEngine.build_labels(df)
        for sym in ("600519", "300750"):
            sub = out[out["symbol"] == sym]
            raw = df[df["symbol"] == sym]["close_hfq"].values
            assert sub["label_1d"].iloc[0] == pytest.approx(raw[1] / raw[0] - 1)
        # cls 标签
        assert set(out["label_cls"].dropna().unique()) <= {0.0, 1.0}

    def test_am_session_labels(self):
        """早盘标签: open(T+1) 基准."""
        df = make_panel(symbols=("600519",), days=30)
        out = LabelEngine.build_labels(df, session="AM")
        raw_c = df["close_hfq"].values
        raw_o = df["open"].values
        assert out["label_1d"].iloc[0] == pytest.approx(raw_c[1] / raw_o[1] - 1)

    def test_winsorize(self):
        df = make_panel(symbols=("600519", "300750", "601318"), days=60)
        out = LabelEngine.build_labels(df)
        out.loc[0, "label_1d"] = 10.0  # 极端值
        clipped = LabelEngine.winsorize_cross_section(out)
        assert clipped["label_1d"].max() < 10.0

    def test_mask_suspension(self):
        df = make_panel(symbols=("600519",), days=30)
        out = LabelEngine.build_labels(df)
        out.loc[5, "is_suspended"] = True  # T+1 停牌 → T 的 label_1d 置 NaN
        masked = LabelEngine.mask_suspension(out)
        assert np.isnan(masked["label_1d"].iloc[4])

    def test_mask_recent_and_per_model_dropna(self):
        df = make_panel(symbols=("600519",), days=30)
        out = LabelEngine.build_labels(df)
        out = LabelEngine.mask_recent_days(out, days=5)
        assert out["label_5d"].tail(5).isna().all()
        sets = LabelEngine.per_model_dropna(out)
        # 30 行 - 5 行遮蔽 (与自然 NaN 尾部重叠) = 各 25 有效
        assert len(sets["1d"]) == 25
        assert len(sets["5d"]) == 25
        assert len(sets["cls"]) == 25


# ============================================================
# 特征引擎
# ============================================================
class TestFeatures:
    def test_dims_and_groupby(self):
        """14 维特征产出 + groupby(symbol) 无跨股泄漏."""
        df = make_panel(symbols=("600519", "300750"), days=300)
        eng = FeatureEngineV35()
        out = eng.build(df)
        for col in (
            "MACD",
            "RSI",
            "K",
            "ATR_pct",
            "BB_width",
            "bias_60",
            "limit_up_days_10",
            "consecutive_board",
            "month",
            "MA250_dist",
            "market_turnover",
            "turnover_stability_5" if "turnover_stability_5" in out else "MA5_dist",
        ):
            assert col in out.columns, col
        # groupby 检查: 每股 MA5_dist 独立计算, 首 4 日为 NaN
        sub = out[out["symbol"] == "600519"]
        assert sub["MA5_dist"].iloc[:4].isna().all()
        assert sub["MA5_dist"].iloc[4] == pytest.approx(
            sub["close_hfq"].iloc[4] / sub["close_hfq"].iloc[:5].mean() - 1
        )

    def test_missingness_flags(self):
        df = make_panel(symbols=("600519",), days=30)
        df["chip_concentration"] = np.nan
        out = FeatureEngineV35.add_missingness_flags(df, ["chip_concentration"])
        assert out["is_missing_chip_concentration"].sum() == 30

    def test_industry_neutralize(self):
        df = make_panel(days=30)
        df["PE_log"] = np.random.default_rng(1).normal(2, 1, len(df))
        out = FeatureEngineV35.industry_neutralize(df, ["PE_log"])
        assert "PE_log_industry_rank" in out.columns
        assert out["PE_log_industry_rank"].between(0, 1).all()


# ============================================================
# IC 筛选
# ============================================================
class TestICScreener:
    def test_rank_ic_perfect(self):
        dates = pd.bdate_range("2025-01-01", periods=80)
        rng = np.random.default_rng(7)
        rows = []
        for d in dates:
            f = rng.normal(size=30)
            for i in range(30):
                rows.append(
                    {"date": d, "factor": f[i], "label_1d": f[i] + rng.normal(0, 0.01)}
                )
        df = pd.DataFrame(rows)
        ic = ICScreener.rank_ic(df, "factor", "label_1d")
        assert ic > 0.9  # 完美相关 → IC≈1


# ============================================================
# 清单生成
# ============================================================
def make_candidates(n=20, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    boards = ["main", "GEM", "STAR"]
    inds = ["白酒", "电池", "保险", "半导体"]
    return pd.DataFrame(
        {
            "symbol": [f"60{i:04d}" for i in range(n)],
            "board": [boards[i % 3] for i in range(n)],
            "industry": [inds[i % 4] for i in range(n)],
            "pred_ret_1d": rng.uniform(-0.05, 0.05, n),
            "pred_ret_3d": rng.uniform(-0.08, 0.10, n),
            "pred_ret_5d": rng.uniform(-0.10, 0.15, n),
            "prob_up": rng.uniform(0.35, 0.65, n),
            "is_in_yesterday_list": [i % 2 for i in range(n)],
        }
    )


class TestListGenerator:
    def test_schema_and_top15(self):
        out = ListGenerator().emit(make_candidates())
        lst = out["list"]
        assert len(lst) <= 15
        assert list(lst.columns) == SCHEMA_FIELDS
        assert out["schema_version"] == "1.0"
        assert lst["prob_up"].iloc[0] == round(lst["prob_up"].iloc[0], 3)

    def test_industry_limit(self):
        cands = make_candidates(n=20)
        cands["industry"] = "白酒"  # 全部同行业 → 最多 4 只
        out = ListGenerator().emit(cands)
        assert len(out["list"]) == 4

    def test_momentum_firewall(self):
        """V3.4 陷阱修复: pred_1d=-8%, pred_3d=-2% → 旧规则误判 high, 新规则 low."""
        assert ListGenerator.compute_momentum(-0.08, -0.02, -0.01) == "low"
        assert ListGenerator.compute_momentum(-0.01, 0.05, 0.10) == "medium"
        assert ListGenerator.compute_momentum(0.0005, 0.001, 0.002) == "medium"
        assert ListGenerator.compute_momentum(0.02, 0.09, 0.20) == "high"  # 加速
        assert ListGenerator.compute_momentum(0.05, 0.10, 0.10) == "low"  # 3d 衰减
        assert ListGenerator.compute_momentum(0.03, 0.10, 0.12) == "medium"

    def test_holding_bonus(self):
        cands = make_candidates(n=2, seed=3)
        cands.loc[:, ["pred_ret_1d", "pred_ret_3d", "pred_ret_5d"]] = 0.02
        cands.loc[:, "prob_up"] = 0.5
        cands["is_in_yesterday_list"] = [1, 0]
        out = ListGenerator().emit(cands)
        # 昨日在清单内的票 +0.2 加成 → 排第一, 分差 ≈ 0.2
        assert out["list"].iloc[0]["symbol"] == "600000"
        assert out["list"].iloc[0]["score"] - out["list"].iloc[1][
            "score"
        ] == pytest.approx(0.2)

    def test_empty_triggers(self):
        """D18: 暴跌/跌停>50 → 空清单; 连跌3日 → 仅 Top 5."""
        out = ListGenerator().emit(make_candidates(), MarketEnv(hs300_drop_today=0.031))
        assert out["empty"] and len(out["list"]) == 0
        out = ListGenerator().emit(
            make_candidates(), MarketEnv(count_limit_down_market=51)
        )
        assert out["empty"]
        out = ListGenerator().emit(
            make_candidates(n=20), MarketEnv(hs300_consecutive_down=3)
        )
        assert not out["empty"] and len(out["list"]) <= 5 and out["cap_position"] == 0.3

    def test_delivery_guard(self):
        g = ListDeliveryGuard()
        assert g.on_failure()["mode"] == "reuse_yesterday"
        assert g.on_failure()["mode"] == "sell_only"
        assert g.on_failure()["mode"] == "manual_intervention"
        lst = pd.DataFrame({"symbol": ["600519"]})
        assert g.on_success(lst)["mode"] == "normal"
        assert g.consecutive_failures == 0

    def test_invalidation(self):
        assert check_invalidation(5.5, False, 0, 0) is not None
        assert check_invalidation(0, True, 0, 0) is not None
        assert check_invalidation(0, False, -3.5, 0) is not None
        assert check_invalidation(0, False, 0, 7.5) is not None
        assert check_invalidation(2.0, False, -1.0, 3.0) is None


# ============================================================
# 概率校准
# ============================================================
class TestCalibrator:
    def test_platt_changes_output(self):
        """严禁直接用原始 predict_proba: 校准后输出必须不同且可验收."""
        rng = np.random.default_rng(5)
        raw = rng.uniform(0.3, 0.7, 500)
        y = (raw + rng.normal(0, 0.15, 500) > 0.5).astype(float)
        cal = ProbCalibrator().fit(raw, y)
        prob = cal.predict_proba(raw)
        assert not np.allclose(prob, raw)
        rep = cal.reliability_report(y, prob)
        assert "brier" in rep and "reliability_offset" in rep


# ============================================================
# OOS 监控
# ============================================================
class TestOOSMonitor:
    def test_state_transitions(self):
        m = OOSMonitor()
        for _ in range(5):
            r = m.daily_check(0.04)
        assert r["state"] == "NORMAL"
        for _ in range(3):
            r = m.daily_check(0.005)
        assert r["state"] in ("YELLOW_REVIEW", "RED_SIMULATE")

    def test_halt(self):
        m = OOSMonitor()
        state = None
        for _ in range(5):
            state = m.daily_check(-0.02)["state"]
        assert state == "HALT"

    def test_kill_switch(self):
        m = OOSMonitor()
        for _ in range(40):
            m.daily_check(0.005)  # 连续 2 月 IC < 0.01
        r = m.kill_switch_check()
        assert r["retire"] and m.state == "RETIRED"
