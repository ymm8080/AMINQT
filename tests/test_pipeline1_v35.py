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
    limit_pct_series,
    limit_up_price,
)
from app.pipeline1.feature_engine_v35 import (
    _LHB_ALPHA,
    FeatureEngineV35,
    _apply_per_stock,
)
from app.pipeline1.ic_screener import ICScreener
from app.pipeline1.label_engine import LabelEngine
from app.pipeline1.list_generator import (
    SCHEMA_FIELDS,
    ListDeliveryGuard,
    ListGenerator,
    MarketEnv,
    ProvenanceTracker,
    check_invalidation,
)
from app.pipeline1.oos_monitor import OOSMonitor
from app.pipeline1.prob_calibrator import ProbCalibrator
from app.pipeline1.serenity_overlay import SerenityOverlay


# ============================================================
# 合成数据
# ============================================================
def make_panel(
    symbols=("600519", "300750", "601318"), days=300, seed=42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=days)
    frames = []
    for _i, sym in enumerate(symbols):
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
                    "industry": "白酒"
                    if sym == "600519"
                    else ("电池" if sym == "300750" else "保险"),
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

    def test_limit_pct_series_matches_scalar(self):
        """M4: 向量化 limit_pct_series 与逐行 get_limit_pct 全等 (跨板块/分段日期)."""
        boards = pd.Series(["main", "GEM", "star", "GEM", "main", "STAR", "gem"])
        dates = pd.to_datetime(
            [
                "2020-08-23",
                "2020-08-23",
                "2019-01-01",
                "2020-08-24",
                "2024-01-01",
                "2024-01-01",
                "2020-08-23",
            ]
        )
        got = limit_pct_series(boards, dates)
        want = [get_limit_pct(b, d) for b, d in zip(boards, dates)]
        assert list(got) == pytest.approx(want)
        with pytest.raises(ValueError):
            limit_pct_series(
                pd.Series(["main", "BSE"]), pd.to_datetime(["2024-01-01", "2024-01-01"])
            )

    def test_limit_up_precise(self):
        """涨停价精确比对 round(pre_close*(1+pct),2), B5 相对容差 max(0.01, lu*0.1%)."""
        assert limit_up_price(10.03, 0.10) == 11.03
        assert is_limit_up(11.03, 10.03, 0.10)
        assert not is_limit_up(11.01, 10.03, 0.10)  # 差2分, 非涨停

    def test_limit_up_relative_tol_high_price(self):
        """B5: 高价股容差 = lu*0.1% (>0.01), 如 100元股容差=0.11."""
        # pre_close=100, limit_up=110.00, tol=max(0.01, 0.11)=0.11
        assert is_limit_up(109.95, 100.0, 0.10)  # 差0.05 < 0.11 → 涨停
        assert not is_limit_up(109.80, 100.0, 0.10)  # 差0.20 > 0.11 → 非涨停

    def test_step1_pass_through(self):
        """ST/*ST 与次新股已在 V3 入库入口 (ingest gate) 过滤, step1 为 pass-through."""
        df = make_panel()
        out = CleaningPipeline().step1_base_state(df)
        assert out.equals(df)
        assert len(out) == len(df)

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
        """安全网 #14: 退市股虚拟 T+1 = 收盘×0.5; [B18] is_virtual 标记."""
        df = make_panel(symbols=("600519",), days=30)
        out = CleaningPipeline().inject_delisted_virtual_rows(df, ["600519"])
        last = out[out["symbol"] == "600519"].iloc[-1]
        prev_close = df["close"].iloc[-1]
        assert last["close"] == pytest.approx(prev_close * 0.5)
        assert len(out) == 31
        assert last["is_virtual"] == 1  # B18: 虚拟行标记
        assert (out.iloc[:-1]["is_virtual"] == 0).all()  # 真实行=0


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

    def test_label_pm_kd_b9(self):
        """B9: 晚盘验收标签 label_pm_kd = close_hfq(T+1+k)/price_1455(T+1) - 1."""
        df = make_panel(symbols=("600519",), days=30)
        df["price_1455"] = df["close"] * 0.99  # 模拟 14:55 价
        out = LabelEngine.build_labels(df)  # 默认 PM
        raw_c = df["close_hfq"].values
        raw_p = df["price_1455"].values
        # T=0: label_pm_1d = close(T+2)/price_1455(T+1) - 1
        assert out["label_pm_1d"].iloc[0] == pytest.approx(raw_c[2] / raw_p[1] - 1)
        assert out["label_pm_3d"].iloc[0] == pytest.approx(raw_c[4] / raw_p[1] - 1)
        # 研究口径 label_kd 仍保留 (close(T) 基准)
        assert out["label_1d"].iloc[0] == pytest.approx(raw_c[1] / raw_c[0] - 1)

    def test_label_pm_kd_b9_daily_proxy(self):
        """B9 日K近似: 无 price_1455 列时用 close(T+1) 代理."""
        df = make_panel(symbols=("600519",), days=30)
        out = LabelEngine.build_labels(df)
        raw_c = df["close_hfq"].values
        assert out["label_pm_1d"].iloc[0] == pytest.approx(raw_c[2] / raw_c[1] - 1)

    def test_winsorize_virtual_exemption_b18(self):
        """B18: is_virtual=1 退市虚拟样本豁免缩尾 (-50% 不被剪, 不参与分位)."""
        df = make_panel(symbols=("600519", "300750", "601318"), days=60)
        out = LabelEngine.build_labels(df)
        out["is_virtual"] = 0
        # 注入虚拟行: 某日期 label_1d = -0.5
        vrow = out.iloc[[0]].copy()
        vrow["label_1d"] = -0.5
        vrow["is_virtual"] = 1
        out = pd.concat([out, vrow], ignore_index=True)
        clipped = LabelEngine.winsorize_cross_section(out)
        virtual = clipped[clipped["is_virtual"] == 1]
        assert virtual["label_1d"].iloc[0] == pytest.approx(-0.5)  # 未被缩尾

    def test_winsorize(self):
        """B2: 缩尾 0.1%/99.9% 仅防数据错误, 保留尾部真实收益."""
        df = make_panel(symbols=("600519", "300750", "601318"), days=60)
        out = LabelEngine.build_labels(df)
        out.loc[0, "label_1d"] = 10.0  # 极端值
        clipped = LabelEngine.winsorize_cross_section(out)
        assert clipped["label_1d"].max() < 10.0
        # B2: 默认参数应为 0.001/0.999
        import inspect

        sig = inspect.signature(LabelEngine.winsorize_cross_section)
        assert sig.parameters["lower"].default == 0.001
        assert sig.parameters["upper"].default == 0.999

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

    def test_multihorizon_cls_labels(self):
        """多视界分类标签: label_{2,3,5}d_cls + PM 变体; AM 会话不报 KeyError."""
        df = make_panel(symbols=("600519",), days=30)
        out = LabelEngine.build_labels(df)  # 默认 PM
        for k in (2, 3, 5):
            assert f"label_{k}d_cls" in out.columns
            assert f"label_pm_{k}d_cls" in out.columns
            assert set(out[f"label_{k}d_cls"].dropna().unique()) <= {0.0, 1.0}
            assert set(out[f"label_pm_{k}d_cls"].dropna().unique()) <= {0.0, 1.0}
        out_am = LabelEngine.build_labels(df, session="AM")  # 无 label_pm_* → 不报错
        for k in (2, 3, 5):
            assert f"label_{k}d_cls" in out_am.columns

    def test_net_cls_labels(self):
        """E5: label_{2,3,5}d_cls_net / label_pm_{2,3,5}d_cls_net 基于净收益>0."""
        df = make_panel(symbols=("600519",), days=30)
        out = LabelEngine.add_net_labels(LabelEngine.build_labels(df))
        for k in (2, 3, 5):
            assert f"label_{k}d_cls_net" in out.columns
            assert f"label_pm_{k}d_cls_net" in out.columns
            assert set(out[f"label_{k}d_cls_net"].dropna().unique()) <= {0.0, 1.0}


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

    def test_dim29_spec_event_features(self):
        """GLM 大股东增减持 spec 三特征: 公告恐慌衰减/结束反弹/是否执行中.

        事件: 公告日 A=2025-01-01, 变动窗口 S=2025-01-15 ~ E=2025-04-15.
        """
        dates = pd.to_datetime(
            [
                "2024-12-20",
                "2025-01-01",
                "2025-01-10",
                "2025-02-01",
                "2025-04-15",
                "2025-05-01",
            ]
        )
        df = pd.DataFrame(
            {
                "symbol": ["600519"] * 6,
                "date": dates,
                "sh_net_change_sign": [np.nan, 1.0, np.nan, np.nan, np.nan, np.nan],
                "sh_change_amt_total": [np.nan, 1000.0, np.nan, np.nan, np.nan, np.nan],
                "sh_evt_start_date": pd.to_datetime(
                    [None, "2025-01-15", None, None, None, None]
                ),
                "sh_evt_end_date": pd.to_datetime(
                    [None, "2025-04-15", None, None, None, None]
                ),
            }
        )
        out = FeatureEngineV35.dim29_holdertrade(df)
        sub = out[out["symbol"] == "600519"].sort_values("date")
        ann = sub["sh_ann_decay"].tolist()
        end = sub["sh_end_decay"].tolist()
        exe = sub["sh_is_executing"].tolist()
        # 事件前: 无上下文 → NaN
        assert np.isnan(ann[0]) and np.isnan(end[0]) and np.isnan(exe[0])
        # 公告日 01-01: 恐慌最大 1.0; 未结束未开始
        assert ann[1] == pytest.approx(1.0)
        assert end[1] == 0.0 and exe[1] == 0.0
        # 01-10: 距公告 9 天 → 1/(1+9)=0.1; 未到开始日
        assert ann[2] == pytest.approx(0.1)
        assert end[2] == 0.0 and exe[2] == 0.0
        # 02-01: 执行期中 → exec=1; 未结束 → end=0
        assert ann[3] == pytest.approx(1 / 32)
        assert end[3] == 0.0 and exe[3] == 1.0
        # 结束日 04-15 (排他): 结束反弹=1.0 达峰值; 视为已结束 → exec=0 (spec 示例表)
        assert ann[4] == pytest.approx(1 / 105)
        assert end[4] == pytest.approx(1.0)
        assert exe[4] == 0.0
        # 05-01: 结束 16 天后 → end=1/17; 已出执行期
        assert ann[5] == pytest.approx(1 / 121)
        assert end[5] == pytest.approx(1 / 17)
        assert exe[5] == 0.0

    def test_dim29_spec_event_features_missing_cols(self):
        """无 evt 列 (AKShare 降级/旧数据): 三特征全 NaN, 不崩."""
        dates = pd.bdate_range("2025-01-06", periods=4)
        df = pd.DataFrame(
            {
                "symbol": ["600519"] * 4,
                "date": dates,
                "sh_net_change_sign": [np.nan, 1.0, np.nan, np.nan],
                "sh_change_amt_total": [np.nan, 1000.0, np.nan, np.nan],
            }
        )
        out = FeatureEngineV35.dim29_holdertrade(df)
        for c in ("sh_ann_decay", "sh_end_decay", "sh_is_executing"):
            assert out[c].isna().all()

    def test_dim32_glm_spec_features(self):
        """GLM 龙虎榜 spec: EWMA 衰减记忆 (h=5, α≈0.129) + 上榜频次.

        7 交易日, amount=1e9 恒定; 第 2 天与第 5 天上榜.
        上榜日 R 瞬间跃升 (F=α·R), 非上榜日乘 (1−α) 衰减.
        """
        dates = pd.bdate_range("2026-01-05", periods=7)
        df = pd.DataFrame(
            {
                "symbol": ["000001"] * 7,
                "date": dates,
                "amount": [1e9] * 7,
                "lhb_buy_amt": [np.nan, 1e8, np.nan, np.nan, 5e7, np.nan, np.nan],
                "lhb_sell_amt": [np.nan, 3e7, np.nan, np.nan, 1e7, np.nan, np.nan],
                "lhb_net_buy": [np.nan, 7e7, np.nan, np.nan, 4e7, np.nan, np.nan],
                "lhb_inst_buy": [np.nan, 2e7, np.nan, np.nan, 0.0, np.nan, np.nan],
                "lhb_inst_sell": [np.nan, 5e6, np.nan, np.nan, 0.0, np.nan, np.nan],
                "lhb_retail_buy": [np.nan, 4e7, np.nan, np.nan, 3e7, np.nan, np.nan],
                "lhb_retail_sell": [np.nan, 0.0, np.nan, np.nan, 0.0, np.nan, np.nan],
            }
        )
        out = FeatureEngineV35.dim32_lhb_glm(df).sort_values("date")
        inst = out["lhb_inst_flow"].tolist()
        retail = out["lhb_retail_flow"].tolist()
        sellp = out["lhb_sell_pressure"].tolist()
        cnt = out["lhb_list_count_5d"].tolist()

        # 未上榜日 (index 0): R=0 → EWMA 停在 0
        assert inst[0] == pytest.approx(0.0)
        assert cnt[0] == 0.0
        # 上榜日 (index 1): F = α·R — 机构 0.015, 散户 0.04, 抛压 0.03
        assert inst[1] == pytest.approx(_LHB_ALPHA * 0.015, rel=1e-3)
        assert retail[1] == pytest.approx(_LHB_ALPHA * 0.04, rel=1e-3)
        assert sellp[1] == pytest.approx(_LHB_ALPHA * 0.03, rel=1e-3)
        assert cnt[1] == 1.0
        # 非上榜日 (index 2): 纯衰减 ×(1−α)
        assert inst[2] == pytest.approx(inst[1] * (1 - _LHB_ALPHA), rel=1e-6)
        assert sellp[2] == pytest.approx(sellp[1] * (1 - _LHB_ALPHA), rel=1e-6)
        # 第二上榜日 (index 4): 机构买入=0 → R=0 仅衰减; 散户/抛压再跃升
        assert inst[4] == pytest.approx(inst[3] * (1 - _LHB_ALPHA), rel=1e-6)
        assert retail[4] == pytest.approx(
            _LHB_ALPHA * 0.03 + (1 - _LHB_ALPHA) * retail[3], rel=1e-3
        )
        assert sellp[4] == pytest.approx(
            _LHB_ALPHA * 0.01 + (1 - _LHB_ALPHA) * sellp[3], rel=1e-3
        )
        # 上榜频次: 近 5 交易日含当日 — index4 窗口含两次上榜
        assert cnt[4] == 2.0
        # index5 窗口 [1..5] 仍含两次上榜 → 2; index6 窗口 [2..6] 首日跌出 → 1
        assert cnt[5] == 2.0
        assert cnt[6] == 1.0

    def test_dim32_glm_missing_cols(self):
        """无席位列 (回填前): 机构/散户动能 NaN; 抛压/频次仍用基础 LHB 计算.
        完全无 LHB: 四特征全 NaN."""
        df = pd.DataFrame(
            {
                "symbol": ["000001"] * 3,
                "date": pd.bdate_range("2026-01-05", periods=3),
                "amount": [1e9] * 3,
                "lhb_buy_amt": [np.nan, 1e8, np.nan],
                "lhb_sell_amt": [np.nan, 3e7, np.nan],
                "lhb_net_buy": [np.nan, 7e7, np.nan],
            }
        )
        out = FeatureEngineV35.dim32_lhb_glm(df)
        assert out["lhb_inst_flow"].isna().all()
        assert out["lhb_retail_flow"].isna().all()
        assert out["lhb_sell_pressure"].iloc[1] == pytest.approx(
            _LHB_ALPHA * 0.03, rel=1e-3
        )
        assert out["lhb_list_count_5d"].iloc[1] == 1.0

        df2 = pd.DataFrame(
            {
                "symbol": ["000001"] * 2,
                "date": pd.bdate_range("2026-01-05", periods=2),
                "amount": [1e9] * 2,
            }
        )
        out2 = FeatureEngineV35.dim32_lhb_glm(df2)
        for c in (
            "lhb_inst_flow",
            "lhb_retail_flow",
            "lhb_sell_pressure",
            "lhb_list_count_5d",
        ):
            assert out2[c].isna().all()

    # ============================================================
    # dim34 KIMI LHB v2.0 spec 特征
    # ============================================================
    @staticmethod
    def _ha(h: float) -> float:
        return 1 - 2 ** (-1 / h)

    def test_dim34_lhb_v2_ewma_and_weights(self):
        """KIMI LHB v2.0: 修正分母净占比 + 情境权重 W_price + EWMA 衰减 + F_min 下限.

        7 交易日, 第 2/第 5 天上榜 (均涨停 → W=1.5). 机构净占比 0.6, 顶级游资 1.0,
        量化 −1.0, 散户 1.0; 抛压 3e7/1.3e8×1.5; 买卖比 0.3.
        """
        from config.settings import LHB_V2_SPEC

        a8 = self._ha(LHB_V2_SPEC["h_inst"])
        a6 = self._ha(LHB_V2_SPEC["h_top"])
        a4 = self._ha(LHB_V2_SPEC["h_quant"])
        a5 = self._ha(LHB_V2_SPEC["h_sell"])
        a3 = self._ha(LHB_V2_SPEC["h_conboard"])

        dates = pd.bdate_range("2026-01-05", periods=7)
        close = [10.0, 11.0, 11.0, 11.0, 12.1, 12.1, 12.1]
        pre = [10.0, 10.0, 11.0, 11.0, 11.0, 12.1, 12.1]
        df = pd.DataFrame(
            {
                "symbol": ["000001"] * 7,
                "date": dates,
                "amount": [1e9] * 7,
                "circ_mv": [1e5] * 7,  # 万元 → 1e9 元
                "close": close,
                "pre_close": pre,
                "high": [c * 1.03 for c in close],
                "low": [c * 0.97 for c in close],
                "up_limit_raw": [np.nan, 11.0, np.nan, np.nan, 12.1, np.nan, np.nan],
                "down_limit_raw": [np.nan] * 7,
                "lhb_buy_amt": [np.nan, 1e8, np.nan, np.nan, 5e7, np.nan, np.nan],
                "lhb_sell_amt": [np.nan, 3e7, np.nan, np.nan, 1e7, np.nan, np.nan],
                "lhb_net_buy": [np.nan, 7e7, np.nan, np.nan, 4e7, np.nan, np.nan],
                "lhb_inst_buy": [np.nan, 2e7, np.nan, np.nan, 0.0, np.nan, np.nan],
                "lhb_inst_sell": [np.nan, 5e6, np.nan, np.nan, 0.0, np.nan, np.nan],
                "lhb_top_buy": [np.nan, 1.5e7, np.nan, np.nan, 5e6, np.nan, np.nan],
                "lhb_top_sell": [np.nan, 0.0, np.nan, np.nan, 0.0, np.nan, np.nan],
                "lhb_quant_buy": [np.nan, 0.0, np.nan, np.nan, 0.0, np.nan, np.nan],
                "lhb_quant_sell": [np.nan, 1e7, np.nan, np.nan, 0.0, np.nan, np.nan],
                "lhb_retail_buy": [np.nan, 4e7, np.nan, np.nan, 3e7, np.nan, np.nan],
                "lhb_retail_sell": [np.nan, 0.0, np.nan, np.nan, 0.0, np.nan, np.nan],
            }
        )
        out = FeatureEngineV35.dim34_lhb_v2(df).sort_values("date")

        # ── 上榜日 (index 1): F = α·R ──
        assert out["lhb2_inst_flow"].iloc[1] == pytest.approx(a8 * 0.6, rel=1e-3)
        assert out["lhb2_inst_shock"].iloc[1] == pytest.approx(a8 * 0.015, rel=1e-3)
        assert out["lhb2_top_flow"].iloc[1] == pytest.approx(a6 * 1.0, rel=1e-3)
        assert out["lhb2_quant_flow"].iloc[1] == pytest.approx(
            -a4, rel=1e-3
        )  # 负值保留
        assert out["lhb2_retail_flow"].iloc[1] == pytest.approx(a4 * 1.0, rel=1e-3)
        # 抛压 = 卖出占比 × 涨停情境权重 1.5
        assert out["lhb2_sell_pressure"].iloc[1] == pytest.approx(
            a5 * (3e7 / 1.3e8) * 1.5, rel=1e-3
        )
        assert out["lhb2_sell_buy_ratio"].iloc[1] == pytest.approx(a5 * 0.3, rel=1e-3)
        assert out["lhb2_conboard_mem"].iloc[1] == pytest.approx(a3 * 1.0, rel=1e-3)

        # ── 未上榜日 (index 2): 纯衰减 ×(1−α), 不归零 ──
        assert out["lhb2_inst_flow"].iloc[2] == pytest.approx(
            out["lhb2_inst_flow"].iloc[1] * (1 - a8), rel=1e-6
        )
        assert out["lhb2_sell_pressure"].iloc[2] == pytest.approx(
            out["lhb2_sell_pressure"].iloc[1] * (1 - a5), rel=1e-6
        )
        # 未上榜日 F 保持前值衰减 (F_min 下限远低于此, 不绑定)
        assert out["lhb2_inst_flow"].iloc[2] > 0

        # ── 第二上榜日 (index 4): 机构买入=0 → R_inst=0 仅衰减; 抛压再跃升(×W=1.5) ──
        assert out["lhb2_inst_flow"].iloc[4] == pytest.approx(
            out["lhb2_inst_flow"].iloc[3] * (1 - a8), rel=1e-6
        )
        assert out["lhb2_top_flow"].iloc[4] == pytest.approx(
            a6 * 1.0 + (1 - a6) * out["lhb2_top_flow"].iloc[3], rel=1e-3
        )
        assert out["lhb2_retail_flow"].iloc[4] == pytest.approx(
            a4 * 1.0 + (1 - a4) * out["lhb2_retail_flow"].iloc[3], rel=1e-3
        )
        assert out["lhb2_sell_pressure"].iloc[4] == pytest.approx(
            a5 * (1e7 / 6e7) * 1.5 + (1 - a5) * out["lhb2_sell_pressure"].iloc[3],
            rel=1e-3,
        )

        # ── 上榜频次 (近 5 交易日含当日) ──
        cnt = out["lhb2_list_count_5d"].tolist()
        assert cnt == [0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0]

    def test_dim34_lhb_v2_interactions(self):
        """价格行为交互 (§4.1/4.2/4.3/4.4): strength/resolve/conboard/premium."""
        from config.settings import LHB_V2_SPEC

        dates = pd.bdate_range("2026-01-05", periods=7)
        close = [10.0, 11.0, 11.0, 11.0, 12.1, 12.1, 12.1]
        pre = [10.0, 10.0, 11.0, 11.0, 11.0, 12.1, 12.1]
        df = pd.DataFrame(
            {
                "symbol": ["000001"] * 7,
                "date": dates,
                "amount": [1e9] * 7,
                "circ_mv": [1e5] * 7,
                "close": close,
                "pre_close": pre,
                "high": [c * 1.03 for c in close],
                "low": [c * 0.97 for c in close],
                "up_limit_raw": [np.nan, 11.0, np.nan, np.nan, 12.1, np.nan, np.nan],
                "down_limit_raw": [np.nan] * 7,
                "lhb_buy_amt": [np.nan, 1e8, np.nan, np.nan, 5e7, np.nan, np.nan],
                "lhb_sell_amt": [np.nan, 3e7, np.nan, np.nan, 1e7, np.nan, np.nan],
                "lhb_net_buy": [np.nan, 7e7, np.nan, np.nan, 4e7, np.nan, np.nan],
                "lhb_inst_buy": [np.nan, 2e7, np.nan, np.nan, 0.0, np.nan, np.nan],
                "lhb_inst_sell": [np.nan, 5e6, np.nan, np.nan, 0.0, np.nan, np.nan],
                "lhb_top_buy": [np.nan, 1.5e7, np.nan, np.nan, 5e6, np.nan, np.nan],
                "lhb_top_sell": [np.nan, 0.0, np.nan, np.nan, 0.0, np.nan, np.nan],
                "lhb_quant_buy": [np.nan, 0.0, np.nan, np.nan, 0.0, np.nan, np.nan],
                "lhb_quant_sell": [np.nan, 1e7, np.nan, np.nan, 0.0, np.nan, np.nan],
                "lhb_retail_buy": [np.nan, 4e7, np.nan, np.nan, 3e7, np.nan, np.nan],
                "lhb_retail_sell": [np.nan, 0.0, np.nan, np.nan, 0.0, np.nan, np.nan],
            }
        )
        out = FeatureEngineV35.dim34_lhb_v2(df).sort_values("date")
        f_inst = out["lhb2_inst_flow"]

        # I_strength = F_inst × Ret (index1/4 涨停日 ret=0.10)
        assert out["lhb2_inst_strength"].iloc[1] == pytest.approx(
            f_inst.iloc[1] * 0.10, rel=1e-6
        )
        assert out["lhb2_inst_strength"].iloc[4] == pytest.approx(
            f_inst.iloc[4] * 0.10, rel=1e-6
        )
        # 非涨停日 ret=0 → I_strength=0
        assert out["lhb2_inst_strength"].iloc[3] == pytest.approx(0.0, abs=1e-12)

        # I_resolve = F_inst / (Amp + ε); Amp = (high−low)/low
        amp1 = (11.0 * 1.03 - 11.0 * 0.97) / (11.0 * 0.97)
        assert out["lhb2_inst_resolve"].iloc[1] == pytest.approx(
            f_inst.iloc[1] / (amp1 + LHB_V2_SPEC["eps"]), rel=1e-3
        )

        # I_conboard = F_inst × C_board (index1/4 连板1天)
        assert out["lhb2_inst_conboard"].iloc[1] == pytest.approx(
            f_inst.iloc[1], rel=1e-6
        )
        assert out["lhb2_inst_conboard"].iloc[4] == pytest.approx(
            f_inst.iloc[4], rel=1e-6
        )

        # I_premium = F_inst × F_top × (1−F_retail) × (1+Ret)
        ret1 = (11.0 - 10.0) / 10.0
        exp_prem1 = (
            f_inst.iloc[1]
            * out["lhb2_top_flow"].iloc[1]
            * (1 - out["lhb2_retail_flow"].iloc[1])
            * (1 + ret1)
        )
        assert out["lhb2_inst_premium"].iloc[1] == pytest.approx(exp_prem1, rel=1e-6)

        # 机构锁仓: F_inst 远低于 0.3 → 恒 0
        assert out["lhb2_inst_lock"].sum() == 0

        # 连板衰减记忆 F_conboard = EWMA(C_board×I(list), h=3)
        a3 = self._ha(LHB_V2_SPEC["h_conboard"])
        assert out["lhb2_conboard_mem"].iloc[4] == pytest.approx(
            a3 * 1.0 + (1 - a3) * out["lhb2_conboard_mem"].iloc[3], rel=1e-3
        )

    def test_dim34_lhb_v2_lock(self):
        """机构锁仓 (§4.5): F_inst>0.3 连续两日且收盘创新高 → I_lock=1.

        row0 为未上榜零行 (EWMA 冷启动 F_0=0); rows 1-8 每日机构净买入 (R_inst=1)
        抬高 F_inst; 上榜日仅 1,2,7,8 → 任何 5 日窗口最多 2 次上榜, 不触发过热惩罚.
        """
        from config.settings import LHB_V2_SPEC

        a8 = self._ha(LHB_V2_SPEC["h_inst"])
        dates = pd.bdate_range("2026-01-05", periods=9)
        close = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0]
        pre = [10.0, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5]
        buy_amt = [np.nan, 1e8, 1e8, np.nan, np.nan, np.nan, np.nan, 1e8, 1e8]
        df = pd.DataFrame(
            {
                "symbol": ["000001"] * 9,
                "date": dates,
                "amount": [1e9] * 9,
                "circ_mv": [1e5] * 9,
                "close": close,
                "pre_close": pre,
                "high": [c * 1.02 for c in close],
                "low": [c * 0.98 for c in close],
                "up_limit_raw": [np.nan] * 9,
                "down_limit_raw": [np.nan] * 9,
                "lhb_buy_amt": buy_amt,
                "lhb_inst_buy": [0.0] + [1e7] * 8,  # R_inst = 1 → F_inst 逼近 1
                "lhb_inst_sell": [0.0] * 9,
            }
        )
        out = FeatureEngineV35.dim34_lhb_v2(df).sort_values("date")
        f_inst = out["lhb2_inst_flow"]
        d8 = 1 - a8
        # R_inst=1 恒定: F_t = α·1 + (1−α)·F_{t−1}; row0 冷启动为 0
        assert f_inst.iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert f_inst.iloc[1] == pytest.approx(a8, rel=1e-6)
        assert f_inst.iloc[5] == pytest.approx(a8 + d8 * f_inst.iloc[4], rel=1e-6)
        # 锁仓: F_inst>0.3 连续两日 & close 创新高(>前5日收盘) — row6 起
        lock = out["lhb2_inst_lock"].tolist()
        assert lock == [0, 0, 0, 0, 0, 0, 1, 1, 1]
        # 连板基因 = F_inst × 连板天数 (本测试无涨停 → C_board=0)
        assert out["lhb2_inst_conboard"].abs().max() == 0

    def test_dim34_lhb_v2_overheat_penalty(self):
        """过热惩罚 (§5.2): C5d≥3 (主板) → 正向资金流特征 ×0.7; 反向特征不动."""
        from config.settings import LHB_V2_SPEC

        a5 = self._ha(LHB_V2_SPEC["h_sell"])
        dates = pd.bdate_range("2026-01-05", periods=7)
        # row0 为未上榜零行 (EWMA 冷启动 F_0=0); rows 1-6 每日上榜 R_inst=1/3
        df = pd.DataFrame(
            {
                "symbol": ["000001"] * 7,
                "date": dates,
                "amount": [1e9] * 7,
                "circ_mv": [1e5] * 7,
                "close": [10.0] * 7,
                "pre_close": [10.0] * 7,
                "high": [10.3] * 7,
                "low": [9.7] * 7,
                "up_limit_raw": [np.nan] * 7,
                "down_limit_raw": [np.nan] * 7,
                "lhb_buy_amt": [np.nan] + [1e8] * 6,  # C5d = [0,1,2,3,4,5,5]
                "lhb_sell_amt": [np.nan] + [2e7] * 6,
                "lhb_net_buy": [np.nan] + [8e7] * 6,
                "lhb_inst_buy": [0.0] + [2e7] * 6,  # R_inst = 1/3 恒定
                "lhb_inst_sell": [0.0] + [1e7] * 6,
                "lhb_quant_buy": [0.0] * 7,  # 反向特征: 量化净卖出
                "lhb_quant_sell": [0.0] + [1e7] * 6,
            }
        )
        out = FeatureEngineV35.dim34_lhb_v2(df).sort_values("date")

        a8 = self._ha(LHB_V2_SPEC["h_inst"])
        d8 = 1 - a8
        r = 1 / 3
        fs = [a8 * r]
        for _ in range(5):
            fs.append(a8 * r + d8 * fs[-1])
        # 未过热行 (row1/2, C5d<3): 不惩罚
        assert out["lhb2_inst_flow"].iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert out["lhb2_inst_flow"].iloc[1] == pytest.approx(fs[0], rel=1e-6)
        assert out["lhb2_inst_flow"].iloc[2] == pytest.approx(fs[1], rel=1e-6)
        # 过热行 (row3+, C5d≥3): ×0.7
        assert out["lhb2_inst_flow"].iloc[3] == pytest.approx(0.7 * fs[2], rel=1e-6)
        assert out["lhb2_inst_flow"].iloc[6] == pytest.approx(0.7 * fs[5], rel=1e-6)
        # 反向特征 (量化/抛压) 不受惩罚
        a4 = self._ha(LHB_V2_SPEC["h_quant"])
        d4 = 1 - a4
        assert out["lhb2_quant_flow"].iloc[3] == pytest.approx(
            -a4 + d4 * out["lhb2_quant_flow"].iloc[2], rel=1e-6
        )
        r_sell = 2e7 / 1.2e8  # W=1.0 (无涨跌停)
        fsell = [a5 * r_sell]
        for _ in range(5):
            fsell.append(a5 * r_sell + (1 - a5) * fsell[-1])
        assert out["lhb2_sell_pressure"].iloc[6] == pytest.approx(fsell[5], rel=1e-6)

    def test_dim34_lhb_v2_missing_cols(self):
        """无席位列 (回填前): 机构/游资/量化/散户及交互 NaN; 抛压/买卖比/频次仍算.
        完全无 LHB: 14 特征全 NaN."""
        from config.settings import LHB_V2_SPEC

        a5 = self._ha(LHB_V2_SPEC["h_sell"])
        df = pd.DataFrame(
            {
                "symbol": ["000001"] * 3,
                "date": pd.bdate_range("2026-01-05", periods=3),
                "amount": [1e9] * 3,
                "close": [10.0, 11.0, 11.0],
                "pre_close": [10.0, 10.0, 11.0],
                "high": [10.3, 11.33, 11.33],
                "low": [9.7, 10.67, 10.67],
                "up_limit_raw": [np.nan, 11.0, np.nan],
                "down_limit_raw": [np.nan] * 3,
                "lhb_buy_amt": [np.nan, 1e8, np.nan],
                "lhb_sell_amt": [np.nan, 3e7, np.nan],
                "lhb_net_buy": [np.nan, 7e7, np.nan],
            }
        )
        out = FeatureEngineV35.dim34_lhb_v2(df)
        for c in (
            "lhb2_inst_flow",
            "lhb2_inst_shock",
            "lhb2_top_flow",
            "lhb2_quant_flow",
            "lhb2_retail_flow",
            "lhb2_inst_strength",
            "lhb2_inst_resolve",
            "lhb2_inst_conboard",
            "lhb2_inst_premium",
        ):
            assert out[c].isna().all(), c
        assert out["lhb2_inst_lock"].sum() == 0  # 无机构数据 → 无锁仓信号 (0)
        assert out["lhb2_sell_pressure"].iloc[1] == pytest.approx(
            a5 * (3e7 / 1.3e8) * 1.5, rel=1e-3
        )
        assert out["lhb2_sell_buy_ratio"].iloc[1] == pytest.approx(a5 * 0.3, rel=1e-3)
        assert out["lhb2_list_count_5d"].iloc[1] == 1.0

        df2 = pd.DataFrame(
            {
                "symbol": ["000001"] * 2,
                "date": pd.bdate_range("2026-01-05", periods=2),
                "amount": [1e9] * 2,
                "close": [10.0, 10.0],
                "pre_close": [10.0, 10.0],
                "high": [10.3, 10.3],
                "low": [9.7, 9.7],
            }
        )
        out2 = FeatureEngineV35.dim34_lhb_v2(df2)
        all14 = [
            "lhb2_inst_flow",
            "lhb2_inst_shock",
            "lhb2_top_flow",
            "lhb2_quant_flow",
            "lhb2_retail_flow",
            "lhb2_sell_pressure",
            "lhb2_sell_buy_ratio",
            "lhb2_list_count_5d",
            "lhb2_conboard_mem",
            "lhb2_inst_strength",
            "lhb2_inst_resolve",
            "lhb2_inst_conboard",
            "lhb2_inst_premium",
            "lhb2_inst_lock",
        ]
        for c in all14:
            assert out2[c].isna().all(), c

    def test_pit_active_v17(self):
        """§14.2.2 安全网 #15: is_active PIT 标签, shift(1) 确保不含 T."""
        df = make_panel(symbols=("600519",), days=300)
        eng = FeatureEngineV35()
        out = eng.dim_active_pit(df)
        assert "is_active" in out.columns
        # 首行应为 0 (shift(1) 后无历史)
        assert out["is_active"].iloc[0] == 0
        # PIT: is_active 仅用 T-1 及更早数据, shift(1) 保证
        # 第 252 行后应有非零值 (252 日窗口满足)
        assert out["is_active"].iloc[-1] in (0, 1)

    def test_alpha_factors_dim15(self):
        """dim15: Alpha101 + GTJA191 精选因子产出 + groupby(symbol) 无跨股泄漏."""
        df = make_panel(symbols=("600519", "300750"), days=60)
        eng = FeatureEngineV35()
        out = eng.dim15_alpha_factors(df)
        for col in (
            "alpha006",
            "alpha012",
            "alpha041",
            "alpha042_ts",
            "alpha054_ts",
            "gtja_001_ts",
            "gtja_004",
        ):
            assert col in out.columns, col
        # groupby 检查: 每股 alpha006 首 9 日为 NaN (10 日窗口)
        sub = out[out["symbol"] == "600519"]
        assert sub["alpha006"].iloc[:9].isna().all()
        assert sub["alpha006"].iloc[9] == pytest.approx(
            sub["open"].iloc[:10].corr(sub["volume"].iloc[:10]) * -1
        )
        # gtja_004: 10 日窗口, 首 9 日 NaN
        assert sub["gtja_004"].iloc[:9].isna().all()

    def test_candlestick_dim16(self):
        """dim16: 6 个 K 线形态二值特征."""
        df = make_panel(symbols=("600519",), days=30)
        eng = FeatureEngineV35()
        out = eng.dim16_candlestick(df)
        for col in (
            "bullish_engulfing",
            "bearish_engulfing",
            "hammer",
            "shooting_star",
            "morning_star",
            "evening_star",
        ):
            assert col in out.columns, col
            assert set(out[col].dropna().unique()) <= {0, 1}

    def test_extended_factors_dim17(self):
        """dim17: Amihud + Fisher + FearGreed."""
        df = make_panel(symbols=("600519",), days=60)
        eng = FeatureEngineV35()
        out = eng.dim17_extended_factors(df)
        for col in ("amihud_illiq", "fisher_transform", "fear_greed"):
            assert col in out.columns, col
        # Amihud: 首 4 日 NaN (min_periods=5)
        assert out["amihud_illiq"].iloc[:4].isna().all()
        # Fisher: 首 19 日 NaN (20 日窗口)
        assert out["fisher_transform"].iloc[:19].isna().all()

    def test_lhb_dim18_no_data(self):
        """dim18: 无 LHB 数据时填 NaN."""
        df = make_panel(symbols=("600519",), days=30)
        eng = FeatureEngineV35()
        out = eng.dim18_lhb(df)
        for col in ("lhb_net_buy_5d", "lhb_inst_count_5d", "lhb_hot_money_5d"):
            assert col in out.columns, col
            assert out[col].isna().all()

    def test_lhb_dim18_with_data(self):
        """dim18: 有 LHB 数据时计算 5 日均值 + shift(1) PIT."""
        df = make_panel(symbols=("600519",), days=30)
        df["lhb_net_buy"] = 1e8
        eng = FeatureEngineV35()
        out = eng.dim18_lhb(df)
        assert "lhb_net_buy_5d" in out.columns
        assert pd.isna(out["lhb_net_buy_5d"].iloc[0])  # shift(1)
        assert not pd.isna(out["lhb_net_buy_5d"].iloc[1])


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

    def test_newey_west_b6(self):
        """B6: Newey-West HAC t 统计量, 自相关调整后 t 应低于原始 t."""
        dates = pd.bdate_range("2025-01-01", periods=80)
        rng = np.random.default_rng(7)
        rows = []
        for d in dates:
            f = rng.normal(size=30)
            for i in range(30):
                rows.append(
                    {"date": d, "factor": f[i], "label_3d": f[i] + rng.normal(0, 0.01)}
                )
        df = pd.DataFrame(rows)
        # lag=5 (3d 标签)
        t_nw = ICScreener.ic_t_stat_newey_west(df, "factor", "label_3d", lag=5)
        # 完美相关 → t 应显著 (>1.96)
        assert t_nw > 1.96
        # lag=0 (无调整) 应 ≥ lag=5 (自相关调整后 t 更保守)
        t_raw = ICScreener.ic_t_stat_newey_west(df, "factor", "label_3d", lag=0)
        assert t_raw >= t_nw


# ============================================================
# 清单生成
# ============================================================
# E7 准入闸门默认开启 (prob>0.60, ret>2×COST); 非闸门主题的测试用透传档位
GATE_OFF = {"entry_prob": 0.0, "entry_ret_mult": 0.0}


def make_candidates(n=20, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    boards = ["main", "GEM", "STAR"]
    inds = ["白酒", "电池", "保险", "半导体"]
    df = pd.DataFrame(
        {
            "symbol": [f"60{i:04d}" for i in range(n)],
            "board": [boards[i % 3] for i in range(n)],
            "industry": [inds[i % 4] for i in range(n)],
            "pred_ret_1d": rng.uniform(-0.05, 0.05, n),
            "pred_ret_2d": rng.uniform(-0.07, 0.12, n),
            "pred_ret_3d": rng.uniform(-0.08, 0.10, n),
            "pred_ret_5d": rng.uniform(-0.10, 0.15, n),
            "pred_ret_10d": rng.uniform(-0.12, 0.20, n),
            "prob_up": rng.uniform(0.35, 0.65, n),
            "is_in_yesterday_list": [i % 2 for i in range(n)],
        }
    )
    # 多视界概率列缺省 = prob_up (compound_prob 精确回退 == prob_up, 现有断言不变)
    for k in (2, 3, 5, 10):
        df[f"prob_up_{k}d"] = df["prob_up"]
    return df


class TestListGenerator:
    def test_schema_and_top15(self):
        out = ListGenerator(**GATE_OFF).emit(make_candidates())
        lst = out["list"]
        assert len(lst) <= 15
        assert list(lst.columns) == SCHEMA_FIELDS
        assert out["schema_version"] == "1.4"
        assert lst["prob_up"].iloc[0] == round(lst["prob_up"].iloc[0], 3)

    def test_industry_limit(self):
        cands = make_candidates(n=20)
        cands["industry"] = "白酒"  # 全部同行业 → 最多 4 只
        # 600019 在块交易扫描缓存里有近期大宗交易, FINAL STOCK SCAN 会剔除 →
        # 压出 top-4, 避免扫描耦合破坏行业上限断言 (2026-08-07 排名键改 10d 后暴露)
        cands.loc[cands["symbol"] == "600019", "pred_ret_10d"] = -1.0
        out = ListGenerator(**GATE_OFF).emit(cands)
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
        """向后兼容: 无 holding_day 列时, is_in_yesterday_list=1 视为 day1 (weight=1.0)."""
        cands = make_candidates(n=2, seed=3)
        # 新 COMPOUND_W 含 10d → 10d 也设平, 保证两票 base 分相等, 分差仅来自 bonus
        cands.loc[
            :,
            [
                "pred_ret_1d",
                "pred_ret_2d",
                "pred_ret_3d",
                "pred_ret_5d",
                "pred_ret_10d",
            ],
        ] = 0.02
        cands.loc[
            :, ["prob_up", "prob_up_2d", "prob_up_3d", "prob_up_5d", "prob_up_10d"]
        ] = 0.5
        cands["is_in_yesterday_list"] = [1, 0]
        out = ListGenerator(**GATE_OFF).emit(cands)
        # 昨日在清单内的票 +0.2 加成 → 排第一, 分差 ≈ 0.2
        assert out["list"].iloc[0]["symbol"] == "600000"
        assert out["list"].iloc[0]["score"] - out["list"].iloc[1][
            "score"
        ] == pytest.approx(0.2)

    def test_holding_bonus_decay_b3(self):
        """B3: Holding Bonus 按持仓天数衰减 day1=1.0/day2=0.5/day3=0.0."""
        cands = make_candidates(n=3, seed=3)
        # 新 COMPOUND_W 含 10d (权重最高), 必须连 10d 一起设平 → 三票 base 分等,
        # 只让 holding bonus (0.2/0.1/0.0) 产生差异
        cands.loc[
            :,
            [
                "pred_ret_1d",
                "pred_ret_2d",
                "pred_ret_3d",
                "pred_ret_5d",
                "pred_ret_10d",
            ],
        ] = 0.02
        cands.loc[
            :, ["prob_up", "prob_up_2d", "prob_up_3d", "prob_up_5d", "prob_up_10d"]
        ] = 0.5
        cands["is_in_yesterday_list"] = [1, 1, 1]
        cands["holding_day"] = [1, 2, 3]  # day1/day2/day3
        gen = ListGenerator(**GATE_OFF)
        out = gen.emit(cands)
        scores = out["list"].set_index("symbol")["score"].to_dict()
        # day1 > day2 > day3 (衰减: 0.2/0.1/0.0)
        day1 = scores["600000"]
        day2 = scores["600001"]
        day3 = scores["600002"]
        assert day1 > day2 > day3
        assert day1 - day2 == pytest.approx(0.1)  # 1.0→0.5 = 差0.1
        assert day2 - day3 == pytest.approx(0.1)  # 0.5→0.0 = 差0.1

    def test_base_rate_rolling_b4(self):
        """B4: base_rate 使用 20 日滚动均值, 非单日均值."""
        gen = ListGenerator()
        # 第一次 emit: 单日均值 = 滚动均值 (仅1天)
        c1 = make_candidates(n=5, seed=1)
        c1.loc[
            :, ["prob_up", "prob_up_2d", "prob_up_3d", "prob_up_5d", "prob_up_10d"]
        ] = 0.50
        gen.emit(c1)
        # 第二次 emit: 不同均值, base_rate 应为前2天平均
        c2 = make_candidates(n=5, seed=2)
        c2.loc[
            :, ["prob_up", "prob_up_2d", "prob_up_3d", "prob_up_5d", "prob_up_10d"]
        ] = 0.60
        df2 = gen.compute_scores(c2)
        # 20日滚动 = (0.50 + 0.60) / 2 = 0.55
        assert df2["base_rate"].iloc[0] == pytest.approx(0.55, abs=0.01)

    def test_cross_group_normalization_v17(self):
        """§14.2.3: 主板/双创 score 尺度不同, 组内 rank_pct 后合并排序."""
        cands = make_candidates(n=6, seed=5)
        # 主板 score 天然偏大
        cands.loc[cands["board"] == "main", "pred_ret_1d"] = 0.10
        cands.loc[cands["board"] == "GEM", "pred_ret_1d"] = 0.01
        cands.loc[:, ["pred_ret_2d", "pred_ret_3d", "pred_ret_5d"]] = 0.01
        cands.loc[:, ["prob_up", "prob_up_2d", "prob_up_3d", "prob_up_5d"]] = 0.5
        cands["industry"] = ["白酒", "电池", "半导体", "保险", "白酒", "电池"]
        out = ListGenerator(**GATE_OFF).emit(cands)
        lst = out["list"]
        # 跨组归一化后, 双创板 (低 score) 也有机会进 Top 15
        boards_in_list = set(lst["board"])
        assert "GEM" in boards_in_list

    def test_compound_prob_weighting(self):
        """多视界加权概率: compound_prob = w3*p3 + w5*p5 + w10*p10 (1d/2d 权重=0)."""
        n = 3
        cands = make_candidates(n=n, seed=7)
        cands.loc[
            :,
            [
                "prob_up",
                "prob_up_2d",
                "prob_up_3d",
                "prob_up_5d",
                "prob_up_10d",
            ],
        ] = [
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.4, 0.6, 0.7, 0.8, 0.9],
            [0.3, 0.2, 0.9, 1.0, 0.6],
        ]
        scored = ListGenerator(entry_prob=0.0, entry_ret_mult=0.0).compute_scores(cands)
        cp = scored["compound_prob"]
        w3, w5, w10 = 0.10, 0.40, 0.50
        assert cp.iloc[0] == pytest.approx(0.5)
        assert cp.iloc[1] == pytest.approx(w3 * 0.7 + w5 * 0.8 + w10 * 0.9)
        assert cp.iloc[2] == pytest.approx(w3 * 0.9 + w5 * 1.0 + w10 * 0.6)

    def test_compound_prob_fallback_when_columns_missing(self):
        """旧 bundle 缺 2/3/5d 概率列 → compound_prob 精确回退 prob_up."""
        cands = make_candidates(n=3, seed=7).drop(
            columns=["prob_up_2d", "prob_up_3d", "prob_up_5d"]
        )
        scored = ListGenerator(entry_prob=0.0, entry_ret_mult=0.0).compute_scores(cands)
        assert (scored["compound_prob"] == scored["prob_up"]).all()

    def test_empty_triggers(self):
        """D18: 暴跌/跌停>50 → 空清单; 连跌3日 → 仅 Top 5."""
        gen = ListGenerator(**GATE_OFF)
        out = gen.emit(make_candidates(), MarketEnv(hs300_drop_today=0.031))
        assert out["empty"] and len(out["list"]) == 0
        out = gen.emit(make_candidates(), MarketEnv(count_limit_down_market=51))
        assert out["empty"]
        out = gen.emit(make_candidates(n=20), MarketEnv(hs300_consecutive_down=3))
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

    def test_provenance_tracker(self):
        """ProvenanceTracker: 记录数据来源/模型版本."""
        tracker = ProvenanceTracker()
        tracker.record("600519", data_source="baostock", model_tag="main_v352")
        meta = tracker.get("600519")
        assert meta["data_source"] == "baostock"
        assert meta["model_tag"] == "main_v352"
        frame = tracker.to_frame()
        assert len(frame) == 1
        assert "symbol" in frame.columns


class TestSerenityOverlay:
    def test_score_computation(self):
        """SerenityOverlay: 加权因子 - 罚分×2."""
        overlay = SerenityOverlay()
        factors = {"demand_inflection": 4, "chokepoint_severity": 3}
        penalties = {"governance": 2}
        score = overlay.compute_score(factors, penalties)
        # 4*15 + 3*15 = 105; penalty = 2*2 = 4; score = 105 - 4 = 101
        assert score == pytest.approx(101)

    def test_apply_filters_low_score(self):
        """SerenityOverlay: < 40 分的票从清单移除."""
        lst = pd.DataFrame({"symbol": ["600000", "600001"], "score": [0.5, 0.3]})
        scorecard = {
            "600000": {"factors": {"demand_inflection": 4}, "penalties": {}},
            "600001": {"factors": {}, "penalties": {"governance": 5}},
        }
        overlay = SerenityOverlay()
        result = overlay.apply(lst, scorecard)
        # 600000: 4*15=60 >= 40 -> 保留
        assert "600000" in set(result["symbol"])
        # 600001: 0 - 5*2*2=-20 < 40 -> 移除
        assert "600001" not in set(result["symbol"])

    def test_apply_no_scorecard_passthrough(self):
        """无质化数据时原样返回."""
        lst = pd.DataFrame({"symbol": ["600000"]})
        overlay = SerenityOverlay()
        result = overlay.apply(lst, None)
        assert len(result) == 1


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


# ============================================================
# 数据供应链 [B11]
# ============================================================
class TestDataSupplyB11:
    def test_check_backfill_depth(self):
        """B11: ≥1250 交易日 → 720 窗口; 不足 → 540 过渡; 空面板 → False."""
        from app.pipeline1.data_supply import DataSupplyChain

        chain = DataSupplyChain()
        deep = make_panel(symbols=("600519",), days=1300)
        assert chain.check_backfill_depth(deep) is True
        shallow = make_panel(symbols=("600519",), days=300)
        assert chain.check_backfill_depth(shallow) is False
        assert chain.check_backfill_depth(pd.DataFrame()) is False

    def test_backfill_ohlcv_mock(self, tmp_path):
        """B11: backfill_ohlcv 逐股拉取 ≥5 年历史, 单股失败告警跳过不中断."""
        from app.pipeline1.data_supply import DataSupplyChain

        hist = make_panel(symbols=("600519",), days=1300).drop(columns=["board"])

        def mock_hist(symbol, start, end):
            if symbol == "300750":
                from app.pipeline1.data_supply import DataSupplyError

                raise DataSupplyError("mock 失败")
            return hist[hist["symbol"] == "600519"].reset_index(drop=True)

        chain = DataSupplyChain(cache_dir=str(tmp_path), fetcher_hist=mock_hist)
        panel = chain.backfill_ohlcv(["600519", "300750"], years=5, end="2026-07-24")
        assert set(panel["symbol"]) == {"600519"}  # 失败股被跳过
        assert chain.check_backfill_depth(panel) is True


# ═══════════════════════════════════════════════════════════════
# _apply_per_stock 内存优化 identity (安全网 #5 参考语义)
# ═══════════════════════════════════════════════════════════════
def _ref_apply_per_stock(df: pd.DataFrame, fn) -> pd.DataFrame:
    """旧实现参考: 全部 parts 驻留 + 一次性 pd.concat (安全网 #5 原语义)."""
    parts = [fn(g.copy()) for _, g in df.groupby("symbol")]
    return pd.concat(parts).sort_values(["symbol", "date"]).reset_index(drop=True)


def test_apply_per_stock_memory_lean_identity():
    """内存优化后 _apply_per_stock 与旧语义逐字节一致 (值/dtype/列序/索引).

    RAM 优化只改内部累积方式 (预分配单块帧替代 list-of-parts), 不得改变输出.
    """
    df = make_panel(symbols=("600519", "300750", "601318"), days=120)

    def per_stock(g):
        g = g.sort_values("date")
        g["ma5"] = g["close"].rolling(5, min_periods=1).mean()
        g["mom1"] = g["close"] / g["close"].shift(1) - 1
        return g

    out = _apply_per_stock(df, per_stock)
    ref = _ref_apply_per_stock(df, per_stock)
    assert out.equals(ref)
    assert list(out.columns) == list(ref.columns)
    assert out.dtypes.to_dict() == ref.dtypes.to_dict()
    assert out.index.equals(ref.index)


def test_apply_per_stock_zero_close_row_preserved():
    """零收盘价行不被过滤/丢失; 逐股应用保留全部行 (含除零数据行)."""
    df = make_panel(symbols=("600519", "300750"), days=30)
    df.loc[df["symbol"] == "600519", "close"] = 0.0

    def per_stock(g):
        g = g.sort_values("date")
        g["zero_flag"] = (g["close"] == 0).astype(int)
        return g

    out = _apply_per_stock(df, per_stock)
    assert len(out) == len(df)
    assert (out.loc[out["symbol"] == "600519", "zero_flag"] == 1).all()
    assert (out.loc[out["symbol"] == "300750", "zero_flag"] == 0).all()


def test_apply_per_stock_nan_int_dtype_no_crash():
    """跨股票混合填充: 首股 int 列, 后续股缺值 NaN → 不得 IntCastingNaNError.

    dtype 恢复只对全有限列生效; 含 NaN 的列保持 float 保留缺失
    (LightGBM 原生处理, 非静默丢弃) — 回归: audit 提交 08875218 后重训崩溃.
    """
    df = make_panel(symbols=("600519", "300750"), days=5)

    def per_stock(g):
        g = g.sort_values("date")
        if g["symbol"].iloc[0] == "600519":
            g["exp_days"] = np.arange(len(g), dtype="int64")
        else:
            g["exp_days"] = np.full(len(g), np.nan)
        return g

    out = _apply_per_stock(df, per_stock)
    assert out.loc[out["symbol"] == "300750", "exp_days"].isna().all()
    assert pd.api.types.is_float_dtype(out["exp_days"])


def test_apply_per_stock_row_dropping_fn_trim():
    """fn 过滤行 (非保行) 时, 输出与旧语义一致 (尾部裁剪)."""
    df = make_panel(symbols=("600519", "300750"), days=20)

    def per_stock(g):
        return g[g["close"] > g["close"].median()]

    out = _apply_per_stock(df, per_stock)
    ref = _ref_apply_per_stock(df, per_stock)
    assert out.equals(ref)
