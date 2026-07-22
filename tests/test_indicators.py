# -*- coding: utf-8 -*-
"""指标复刻层 (P16) + CompositeFeed + 双轨训练器 smoke 测试."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.indicators.zhuli_lasheng import zhuli_lasheng, had_accumulation_peak
from app.indicators.yimeng_dingdi import yimeng_dingdi, YimengFeed
from app.indicators.chip_distribution import ChipDistribution, ChipFeed
from app.indicators.faxian_niugu import faxian_niugu
from app.indicators.indicator_feed import CompositeFeed
from app.rules.rule_engine import Candidate, PortfolioState, RuleEngine


def make_ohlc(days=300, seed=11, trend=0.0008) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=days)
    close = 100 * np.cumprod(1 + rng.normal(trend, 0.02, days))
    open_ = close * (1 + rng.normal(0, 0.004, days))
    return pd.DataFrame(
        {
            "time": dates.strftime("%Y%m%d"),
            "date": dates,
            "open": open_,
            "high": np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.005, days))),
            "low": np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.005, days))),
            "close": close,
            "volume": rng.integers(1e6, 5e7, days).astype(float),
        }
    )


class TestZhuliLasheng:
    def test_outputs_and_no_nan_tail(self):
        df = zhuli_lasheng(make_ohlc())
        for col in (
            "主力轨迹",
            "MAZL",
            "吸筹",
            "洗盘",
            "拉高",
            "出货",
            "吸筹峰",
            "上方死叉出货",
        ):
            assert col in df.columns, col
        assert df["主力轨迹"].iloc[-1] == pytest.approx(
            df["主力轨迹"].iloc[-1]
        )  # 非 NaN

    def test_accumulation_peak_lookback(self):
        df = zhuli_lasheng(make_ohlc())
        assert isinstance(had_accumulation_peak(df, 20), bool)


class TestYimengDingdi:
    def test_three_lines_and_feed(self):
        df = yimeng_dingdi(make_ohlc())
        for col in ("短期线", "中期线", "长期线", "红蓝距离", "红在蓝上"):
            assert col in df.columns, col
        # 红蓝映射已确认: 红=长期线, 蓝=中期线 → 红蓝距离 = |长期-中期|
        assert df["红蓝距离"].iloc[-1] == pytest.approx(
            abs(df["长期线"].iloc[-1] - df["中期线"].iloc[-1])
        )
        feed = YimengFeed({"600519": df})
        assert isinstance(feed.red_above_blue_since_peak("600519"), bool)
        assert isinstance(feed.red_blue_distance_min("600519"), bool)

    def test_filter_causal(self):
        """FILTER 因果性: 顶部信号 N=4 日内最多 1 次."""
        df = yimeng_dingdi(make_ohlc())
        tops = df.index[df["顶部"]].tolist()
        for a, b in zip(tops, tops[1:]):
            assert b - a >= 4


class TestChipDistribution:
    def test_winner_bounds_and_feed(self):
        df = make_ohlc()
        df = ChipDistribution().build(df, float_shares=1e8)
        assert df["A04"].between(0, 100).all()
        assert df["获利盘"].between(0, 100).all()
        feed = ChipFeed({"600519": df})
        assert 0 <= feed.profit_chip_ratio("600519") <= 100
        assert isinstance(feed.red_bar_rising_and_majority("600519"), bool)

    def test_rally_profit_ratio_high(self):
        """稳定上涨 (低噪声) 后获利盘应接近 100% (方向锚点)."""
        dates = pd.bdate_range("2025-01-01", periods=120)
        close = np.linspace(100, 200, 120)
        df = pd.DataFrame(
            {
                "time": dates.strftime("%Y%m%d"),
                "date": dates,
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": np.full(120, 1e7),
            }
        )
        df = ChipDistribution().build(df, float_shares=1e8)
        assert df["获利盘"].iloc[-1] > 90


class TestFaxianNiugu:
    def test_ss_and_alignment(self):
        df = faxian_niugu(make_ohlc())
        assert "SS" in df.columns and "多头排列" in df.columns
        # SS 条件: EMA3 上穿 EMA20 + 收阳 + 涨 + ≥1.8%
        ss = df[df["SS"]]
        for _, r in ss.iterrows():
            assert r["close"] > r["open"]


class TestCompositeFeedIntegration:
    def _feed(self):
        df = make_ohlc()
        zdf = zhuli_lasheng(df.copy())
        ydf = yimeng_dingdi(df.copy())
        cdf = ChipDistribution().build(df.copy(), float_shares=1e8)
        return CompositeFeed(
            yimeng=YimengFeed({"600519": ydf}),
            chip=ChipFeed({"600519": cdf}),
            capital=None,
            zhuli_hist={"600519": zdf},
            daily_hist={"600519": df},
            prob_provider=lambda code: 0.60,
        ), zdf

    def test_all_protocol_methods(self):
        feed, _ = self._feed()
        for fn in (
            "control_ratio",
            "red_above_blue_since_peak",
            "red_blue_distance_min",
            "control_weekly_up",
            "bottom_breakout_volume",
            "recent_shadow_lines",
            "red_bar_rising_and_majority",
        ):
            assert isinstance(getattr(feed, fn)("600519"), (bool, float, np.bool_))
        assert feed.had_accumulation_peak("600519", 20) in (True, False)
        assert 0 <= feed.profit_chip_ratio("600519") <= 100
        assert feed.latest_prob_up("600519") == 0.60

    def test_degradation_no_crash(self):
        """子源缺失 → 保守默认值, 不崩溃."""
        feed = CompositeFeed()
        assert feed.control_ratio("600519") == 0.0
        assert feed.red_blue_distance_min("600519") is False
        assert feed.profit_chip_ratio("600519") == 100.0
        assert feed.latest_prob_up("600519") == 1.0

    def test_rule_engine_with_real_feed(self):
        """CompositeFeed 注入 RuleEngine: 盘后流程真实跑通."""
        feed, _ = self._feed()
        eng = RuleEngine(feed)
        pf = PortfolioState(cash=1e6, nav=1e6, peak_nav=1e6)
        c = Candidate(
            "600519",
            "白酒",
            0.03,
            0.60,
            daily_closes=[100, 101, 99, 102, 103],
            turnover_today=12,
        )
        r = eng.after_close(1, [c], pf)
        assert len(r["pool"]) == 1  # 进池正常, 不崩溃


class TestDualTrackTrainer:
    def test_split_window_segments(self):
        """720 窗口四段: 690/10/10/10, 校准与早停物理隔离."""
        from app.pipeline1.dual_track_trainer import DualTrackTrainer

        dates = pd.bdate_range("2023-01-02", periods=720)
        df = pd.DataFrame({"date": dates, "x": range(720)})
        segs = DualTrackTrainer.split_window(df)
        assert {k: len(v) for k, v in segs.items()} == {
            "train": 690,
            "es": 10,
            "calib": 10,
            "test": 10,
        }
        assert set(segs["es"]["date"]) & set(segs["calib"]["date"]) == set()

    def test_time_weights_decay(self):
        from app.pipeline1.dual_track_trainer import DualTrackTrainer

        df = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=500)})
        w = DualTrackTrainer.time_weights(df)
        assert w[-1] == pytest.approx(1.0)  # 最新权重 1
        assert w[0] == pytest.approx(0.5 ** (499 / 250), rel=1e-3)
        assert (np.diff(w) > 0).all()  # 单调递增

    def test_train_smoke(self):
        """LightGBM 双轨 smoke: 小样本小参数真实训练 + 预测 (打补丁缩小规模)."""
        import app.pipeline1.dual_track_trainer as dtt

        dtt.LGB_PARAMS_REG["n_estimators"] = 20
        dtt.LGB_PARAMS_CLS["n_estimators"] = 20
        dtt.ES_PATIENCE = 5
        rng = np.random.default_rng(2)
        dates = pd.bdate_range("2023-01-02", periods=720)
        f = rng.normal(size=720)
        df = pd.DataFrame(
            {
                "date": dates,
                "symbol": "600519",
                "f1": f,
                "f2": rng.normal(size=720),
                "label_1d": f * 0.01 + rng.normal(0, 0.01, 720),
                "label_3d": rng.normal(0, 0.02, 720),
                "label_5d": rng.normal(0, 0.03, 720),
            }
        )
        df["label_cls"] = (df["label_1d"] > 0.005).astype(float)
        trainer = dtt.DualTrackTrainer()
        trained = trainer.train_window(df, "main", ["f1", "f2"])
        assert set(trained["models"]) == {"1d_reg", "1d_cls", "3d_reg", "5d_reg"}
        pred = trained["models"]["1d_reg"][0].predict(df[["f1", "f2"]].tail(5))
        assert len(pred) == 5
        oos = trainer.validate_oos(trained)
        assert "1d_reg" in oos["ics"]
