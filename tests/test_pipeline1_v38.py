"""PIPELINE1 V3.8 (E1-E11) 单元测试.

覆盖: E1 分位数五模型+单调性 / E2 mdd标签+痛苦预警+排序惩罚 / E3 GT-Score /
E4 三色灯+L1 / E5 分层滑点(标签+回测) / E6 amihud+成交额后10%+liquidity_cap /
E7 动态准入 / E8 簇阻断 / E9 波动率熔断 / E11 熊市协议 / Isotonic 校准 / schema V1.2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.pipeline1.bear_protocol import BearProtocol
from app.pipeline1.cleaning_pipeline import CleaningConfig, CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.gt_score import gt_score
from app.pipeline1.label_engine import LabelEngine, slippage_tier
from app.pipeline1.list_generator import ListGenerator
from app.pipeline1.oos_monitor import OOSMonitor
from app.pipeline1.quantile_models import PainModel, QuantileModelSet
from app.pipeline1.risk_overlays import (
    apply_cluster_caps,
    cluster_block,
    liquidity_cap,
    vol_breaker_multiplier,
)


# ============================================================
# E5 滑点分层
# ============================================================
class TestSlippageTier:
    def test_tiers(self):
        assert slippage_tier(6e8) == 0.0005  # >5亿
        assert slippage_tier(2e8) == 0.0010  # 1~5亿
        assert slippage_tier(5e7) == 0.0015  # <1亿
        assert slippage_tier(np.nan) == 0.0015  # 未知按最差档 (保守)


def _label_panel(days=30) -> pd.DataFrame:
    """单股面板: close 恒定 100, low 已知序列 (便于手算 mdd)."""
    dates = pd.bdate_range("2025-01-01", periods=days)
    low = np.full(days, 99.0)
    low[3] = 90.0  # T=2 之后第二天的深坑 → 影响 T=0..2 的 mdd_3d
    return pd.DataFrame(
        {
            "symbol": "600519",
            "date": dates,
            "close_hfq": 100.0,
            "low_hfq": low,
            "amount": 1e9,  # adv20 = 1e9 → 滑点 0.05% 档
        }
    )


class TestNetLabels:
    def test_net_labels_deduct_cost_and_tiered_slippage(self):
        df = _label_panel()
        df = LabelEngine.build_labels(df)
        # adv20=1e9 > 5亿 → 滑点 0.0005; 净 = 毛 - 0.0013 - 2×0.0005
        assert "label_1d_net" in df.columns
        delta = df["label_1d"] - df["label_1d_net"]
        # adv20 成形后 (≥20 日): 0.05% 档
        assert delta.iloc[25] == pytest.approx(0.0013 + 2 * 0.0005)
        # adv20 未成形 (前 19 日): 保守按最差档 0.15%
        assert delta.iloc[0] == pytest.approx(0.0013 + 2 * 0.0015)

    def test_pm_net_labels(self):
        df = _label_panel()
        df = LabelEngine.build_labels(df, session="PM")
        assert "label_pm_1d_net" in df.columns
        delta = df["label_pm_1d"] - df["label_pm_1d_net"]
        assert delta.iloc[25] == pytest.approx(0.0023)


# ============================================================
# E2 路径依赖标签
# ============================================================
class TestPathLabels:
    def test_mdd_window(self):
        df = LabelEngine.build_path_labels(_label_panel())
        row0 = df.iloc[0]
        # close 恒定 100, exec_close = close[T+1] = 100
        # mdd_1d = min(low[1..2])/100-1 = 99/100-1
        assert row0["label_mdd_1d"] == pytest.approx(-0.01)
        # mdd_3d = min(low[1..4])/100-1 = 90/100-1 (low[3]=90)
        assert row0["label_mdd_3d"] == pytest.approx(-0.10)
        assert row0["label_pain"] == 1.0  # 3日浮亏>5%
        # T=5: 窗口 low[6..8] 全 99 → mdd_3d = -0.01 → pain 0
        assert df.iloc[5]["label_pain"] == 0.0
        # 尾部标签未生成 → NaN
        assert np.isnan(df.iloc[-1]["label_mdd_5d"])

    def test_mask_recent_days_covers_new_labels(self):
        df = LabelEngine.build_path_labels(_label_panel())
        df = LabelEngine.build_labels(df)
        df = LabelEngine.mask_recent_days(df, days=6)
        tail = df.tail(6)
        assert tail["label_mdd_3d"].isna().all()
        assert tail["label_pain"].isna().all()
        assert tail["label_1d_net"].isna().all()


# ============================================================
# E1 分位数五模型 + 单调性
# ============================================================
class TestQuantileModels:
    def _train(self, n=300):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(n, 3))
        y = X[:, 0] * 0.02 + rng.normal(0, 0.03, n)
        params = {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbosity": -1,
        }
        qset = QuantileModelSet(params).fit(X, y)
        return qset, X

    def test_monotonicity_enforced(self):
        qset, X = self._train()
        dist = qset.predict(X[:50])
        cols = ["pred_q10", "pred_q25", "pred_q50", "pred_q75", "pred_q90"]
        vals = dist[cols].values
        assert (np.diff(vals, axis=1) >= -1e-12).all(), "分位数必须单调不减"
        assert (dist["uncertainty_width"] >= -1e-12).all()

    def test_quantile_spread_covers_median(self):
        qset, X = self._train()
        dist = qset.predict(X[:100])
        # 大致: q90 > q50 中位 (上行为正态噪声)
        assert dist["pred_q90"].median() > dist["pred_q10"].median()

    def test_flat_es_refit_min_trees(self):
        # es 窗过平 (常数 y) → 早停 1 树 = 常数分位 (2026-08-14 M20260812 main 3d q50 复发),
        # 地板 (QUANTILE_MIN_TREES) 必须兜底重训 → q50 有真实截面区分度
        rng = np.random.default_rng(11)
        X = rng.normal(size=(300, 3))
        y = X[:, 0] * 0.02 + rng.normal(0, 0.03, 300)
        X_es = rng.normal(size=(80, 3))
        y_es = np.zeros(80)
        params = {
            "n_estimators": 10,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbosity": -1,
        }
        qset = QuantileModelSet(params).fit(X, y, eval_set=(X_es, y_es), es_patience=5)
        preds = qset.predict(X[:50])["pred_q50"].values
        assert preds.std() > 0, "地板重训后 q50 不应为常数"

    def test_es_caps_n_estimators_at_max(self):
        # 上限 (QUANTILE_MAX_TREES): q0.75/q0.90 pinball 不早停会跑到 1000 树
        # (08-17 实测 100-350 树 / 1.4M 行), 上限只兜底异常尾部, 正常区间不变
        from app.pipeline1.quantile_models import QUANTILE_MAX_TREES

        rng = np.random.default_rng(5)
        X = rng.normal(size=(500, 3))
        y = X[:, 0] * 0.05 + rng.normal(0, 0.02, 500)
        X_es = rng.normal(size=(100, 3))
        y_es = X_es[:, 0] * 0.05 + rng.normal(0, 0.02, 100)
        params = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "random_state": 42,
            "verbosity": -1,
        }
        qset = QuantileModelSet(params).fit(X, y, eval_set=(X_es, y_es), es_patience=5)
        for q, model in qset.models.items():
            bi = getattr(model, "best_iteration_", None)
            n = bi if bi is not None else model.n_estimators
            assert n <= QUANTILE_MAX_TREES, (
                f"q={q} 树数 {n} 超上限 {QUANTILE_MAX_TREES}"
            )


class TestPainModel:
    def test_fit_predict(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(200, 2))
        y = (X[:, 0] + rng.normal(0, 0.5, 200) < -0.5).astype(float)
        pain = PainModel({"n_estimators": 10, "random_state": 42, "verbosity": -1}).fit(
            X, y
        )
        prob = pain.predict_proba(X[:10])
        assert ((prob >= 0) & (prob <= 1)).all()

    def test_pain_adjustment(self):
        assert PainModel.pain_adjustment(0.31) == 0.5  # >30% → 降仓50%
        assert PainModel.pain_adjustment(0.30) == 1.0


# ============================================================
# E3 GT-Score
# ============================================================
class TestGTScore:
    def test_all_positive(self):
        s = gt_score([0.02] * 20, [0.30] * 20)
        # 0.02 + 0.5×1×0.02 - 0 - 0 = 0.03
        assert s == pytest.approx(0.03)

    def test_downside_and_turnover_penalty(self):
        ics = [-0.10] * 3 + [0.02] * 17  # 线性插值后最差10%分位 = -0.10
        s = gt_score(ics, [0.50] * 20)
        # mean=0.002, pos=0.85, worst10=-0.10, turnover_pen=0.05
        expected = 0.002 + 0.5 * 0.85 * 0.002 - 0.3 * 0.10 - 0.05
        assert s == pytest.approx(expected)

    def test_beats_naive_mean_on_consistency(self):
        # 同 mean: 稳定正 IC 应优于大起大落的 IC
        stable = gt_score([0.02] * 20, [0.3] * 20)
        volatile = gt_score([0.10] * 10 + [-0.06] * 10, [0.3] * 20)
        assert stable > volatile


# ============================================================
# E4 IC 日度三色灯
# ============================================================
class TestTrafficLight:
    def _monitor(self):
        m = OOSMonitor()
        m.ic_history = [0.04, 0.06] * 10  # μ=0.05, σ=0.01
        return m

    def test_light_colors(self):
        m = self._monitor()
        assert m.ic_traffic_light(0.05) == "GREEN"
        assert m.ic_traffic_light(0.035) == "YELLOW"  # (μ-2σ, μ-1σ) = (0.03, 0.04)
        assert m.ic_traffic_light(0.02) == "RED"  # < μ-2σ

    def test_red_triggers_l1_immediately(self):
        m = self._monitor()
        r = m.daily_check(0.02)
        assert r["state"] == "RED_SIMULATE"
        assert r["light"] == "RED"

    def test_three_yellows_escalate_to_l1(self):
        m = self._monitor()
        m.daily_check(0.035)
        m.daily_check(0.035)
        r = m.daily_check(0.035)
        assert r["state"] == "RED_SIMULATE"  # 连续3日黄灯 → L1

    def test_insufficient_history_no_light(self):
        m = OOSMonitor()
        assert m.ic_traffic_light(-0.5) == "GREEN"  # 历史<5日不亮灯


# ============================================================
# E6 amihud 特征 + 清洗步骤5 + liquidity_cap
# ============================================================
class TestE6:
    def test_amihud_and_adv20_features(self):
        dates = pd.bdate_range("2025-01-01", periods=30)
        df = pd.DataFrame(
            {
                "symbol": "600519",
                "date": dates,
                "close_hfq": 100 * np.cumprod(1 + np.full(30, 0.001)),
                "amount": 1e9,
            }
        )
        out = FeatureEngineV35().dim19_amihud(df)
        assert "amihud_illiquidity" in out.columns
        assert out["adv20"].iloc[-1] == pytest.approx(1e9)
        assert out["amihud_illiquidity"].iloc[-1] > 0

    def test_step5_bottom_removed_when_pct_set(self):
        # E6 机制保留: 显式 bottom_amount_pct>0 时仍剔成交额后 pct% 尾部
        dates = pd.bdate_range("2025-01-01", periods=2)
        rows = []
        for d in dates:
            for i in range(10):
                rows.append(
                    {"symbol": f"6000{i:02d}", "date": d, "amount": 1e8 * (i + 1)}
                )
        df = pd.DataFrame(rows)
        out = CleaningPipeline(
            CleaningConfig(bottom_amount_pct=0.1)
        ).step5_amount_bottom(df)
        # 每日期剔除成交额最小 1 只 (rank_pct 0.1 不 > 0.1)
        assert len(out) == 18
        for _d, g in out.groupby("date"):
            assert g["amount"].min() > 1e8

    def test_step5_default_noop_both_boards(self):
        # 08-20 N=800 重扫定案: dual E6 0.1→0.0 (250d OOS top5 +2.04pp / top10 +0.57pp /
        # hit +1.2pp / wIC 0.2187 最高, 3/4 子窗) → 默认配置下 E6 两板块全保留
        dates = pd.bdate_range("2025-01-01", periods=1)
        rows = []
        for b, prefix in (("main", "600"), ("GEM", "300")):
            for i in range(10):
                rows.append(
                    {
                        "symbol": f"{prefix}{i:03d}",
                        "date": dates[0],
                        "board": b,
                        "amount": 1e8 * (i + 1),
                    }
                )
        df = pd.DataFrame(rows)
        out = CleaningPipeline().step5_amount_bottom(df)
        main_out = out[out["board"] == "main"]
        dual_out = out[out["board"] == "GEM"]
        assert len(main_out) == 10  # main 全保留
        assert len(dual_out) == 10  # dual E6=0 → 全保留

    def test_liquidity_cap(self):
        assert liquidity_cap(1e6, 2e8) == pytest.approx(1.0)  # 2亿×1%=200万 > 100万
        assert liquidity_cap(3e6, 2e8) == pytest.approx(2e8 * 0.01 / 3e6)
        assert liquidity_cap(3e6, 2e8, bear=True) == pytest.approx(2e8 * 0.005 / 3e6)
        assert liquidity_cap(1e6, np.nan) == 0.0  # ADV 未知 → 禁买


class TestPoolBlend:
    """08-20 定案: dual 入池键 = w*池分 + (1-w)*rank_pct(pred_ret_10d), per (date,board)
    top-N (pool_blend_cut). 250d 板级回放 top10 +27.7% vs 纯流动性 +26.1% (3/4 子窗).
    排名键保持纯 pred_ret_10d (blend 排名已证伪, 勿改).
    """

    @staticmethod
    def _blend_frame():
        dates = pd.bdate_range("2025-01-01", periods=2)
        rows = []
        for d in dates:
            for i in range(12):
                rows.append(
                    {
                        "symbol": f"3000{i:02d}",
                        "date": d,
                        "board": "GEM",
                        "liquidity_score": 0.9 - 0.02 * i,
                        "pred_ret_10d": -0.005 * i,
                    }
                )
            # 300911 型: 池分垫底 (纯流动性池被切), 预测涨幅最高 → blend 入池
            rows.append(
                {
                    "symbol": "309911",
                    "date": d,
                    "board": "GEM",
                    "liquidity_score": 0.01,
                    "pred_ret_10d": 0.20,
                }
            )
            for i in range(4):  # STAR 板独立切池
                rows.append(
                    {
                        "symbol": f"6880{i:02d}",
                        "date": d,
                        "board": "STAR",
                        "liquidity_score": 0.5 - 0.05 * i,
                        "pred_ret_10d": 0.0,
                    }
                )
            rows.append(  # main 不限池直通
                {
                    "symbol": "600001",
                    "date": d,
                    "board": "main",
                    "liquidity_score": 0.0,
                    "pred_ret_10d": 0.0,
                }
            )
        return pd.DataFrame(rows)

    def test_blend_cut_admits_high_pred_low_liq(self):
        """高预测股进入 blend 池; 纯流动性 top-N 会切掉它 (池分垫底)."""
        df = self._blend_frame()
        cfg = CleaningConfig(pool_blend_w=0.5, liquidity_top_n=10)
        out = CleaningPipeline(cfg).pool_blend_cut(df, pred_col="pred_ret_10d")
        assert (
            out["symbol"].eq("600001") & out["board"].eq("main")
        ).sum() == 2  # main 直通
        for _d, g in out.groupby("date"):
            gem = g[g["board"] == "GEM"]
            assert "309911" in gem["symbol"].values  # blend 入池
            assert len(gem) <= 10
            star = g[g["board"] == "STAR"]
            assert len(star) <= 10
            assert len(g[g["board"] == "main"]) == 1
        # 纯流动性 top-10 (池分) 对照: 309911 池分垫底 → 被切
        pure = df.groupby(["date", "board"])["liquidity_score"].rank(
            ascending=False, method="first"
        )
        liq_pool = df[pure <= 10]
        assert "309911" not in liq_pool[liq_pool["board"] == "GEM"]["symbol"].values

    def test_blend_cut_passthrough_missing_pred(self):
        """缺 pred 列 → fail-open 原样返回."""
        df = self._blend_frame().drop(columns="pred_ret_10d")
        out = CleaningPipeline().pool_blend_cut(df, pred_col="pred_ret_10d")
        assert len(out) == len(df)

    def test_blend_cut_raises_dual_nonempty_missing_liquidity(self):
        """[08-21 fail-open 修复] dual 非空但缺 liquidity_score → ValueError.

        旧守卫静默原样返回 → 生产 candidates ~3100 只从不切池, N=800+blend
        定案 (回放 top10 +27.7%) 从未生效. 缺列只可能是上游丢列 bug, 必须大声."""
        df = self._blend_frame().drop(columns="liquidity_score")
        with pytest.raises(ValueError, match="liquidity_score"):
            CleaningPipeline().pool_blend_cut(df, pred_col="pred_ret_10d")

    def test_blend_cut_dual_empty_silent_passthrough(self):
        """dual 当日无候选 (全空) → 合法场景原样返回, 不报错."""
        df = self._blend_frame()
        df = df[df["board"].eq("main")]
        out = CleaningPipeline().pool_blend_cut(df, pred_col="pred_ret_10d")
        assert len(out) == len(df)

    def test_run_inference_pool_blend_dual_uncut(self):
        """run_inference(pool_blend=True) → dual 全谱 (不切池); False → 前 N."""
        dates = pd.bdate_range("2025-01-01", periods=2)
        rows = []
        for d in dates:
            for sym, board in (
                ("600001", "main"),
                ("600002", "main"),
                ("300001", "GEM"),
                ("300002", "GEM"),
                ("300003", "GEM"),
                ("688001", "STAR"),
            ):
                rows.append(
                    {
                        "symbol": sym,
                        "date": d,
                        "board": board,
                        "open": 10.0,
                        "high": 10.5,
                        "low": 9.5,
                        "close": 10.0,
                        "close_hfq": 10.0,
                        "pre_close": 10.0,
                        "amount": 1e8,
                        "volume": 1e6,
                        "turnover_rate": 2.0,
                        "free_float_turnover_rate": 2.0,
                        "is_suspended": False,
                    }
                )
        panel = pd.DataFrame(rows)
        cfg = CleaningConfig(liquidity_top_n=1)
        _, dual_blend, _ = CleaningPipeline(cfg).run_inference(panel, pool_blend=True)
        _, dual_liq, _ = CleaningPipeline(cfg).run_inference(panel, pool_blend=False)
        per_day = dual_blend.groupby("date")["symbol"].count()
        assert (per_day == 4).all()  # 全谱双创 (2 GEM + 2 STAR)
        per_day2 = dual_liq.groupby("date")["symbol"].count()
        assert (per_day2 == 2).all()  # 纯流动性前 1/板块 (GEM 1 + STAR 1)


# ============================================================
# E7 动态准入 + E2 排序惩罚 + E1 分布权重 (清单)
# ============================================================
def _cands(rows: list[dict]) -> pd.DataFrame:
    base = {
        "board": "main",
        "industry": "白酒",
        "pred_ret_3d": 0.03,
        "pred_ret_5d": 0.05,
        "pred_ret_10d": 0.07,
        "prob_up": 0.70,
        "pred_ret_1d": 0.02,
        "pred_ret_2d": 0.03,
    }
    df = pd.DataFrame([{**base, **r} for r in rows])
    # 多视界概率列缺省 = prob_up (compound_prob 精确回退 == prob_up, 现有断言不变)
    for k in (2, 3, 5):
        if f"prob_up_{k}d" not in df.columns:
            df[f"prob_up_{k}d"] = df["prob_up"]
    return df


class TestDynamicEntry:
    def test_gate_filters_low_quality(self):
        """计算闸: prob 需超当日基准率 (均值), 净预期 compound 需 > 0."""
        cands = _cands(
            [
                {"symbol": "600001"},  # prob 0.70 > 基准 0.65, compound .028 > 0 → 过
                {"symbol": "600002", "prob_up": 0.55},  # prob < 基准 → 剔
                {  # 净预期为负 → 剔 (含 10d, 避免继承 base 的 +0.07 使 10d 主导翻转)
                    "symbol": "600003",
                    "pred_ret_1d": -0.02,
                    "pred_ret_3d": -0.03,
                    "pred_ret_5d": -0.05,
                    "pred_ret_10d": -0.07,
                },
            ]
        )
        out = ListGenerator().emit(cands)
        assert list(out["list"]["symbol"]) == ["600001"]

    def test_zero_passing_is_feature_not_bug(self):
        cands = _cands([{"symbol": "600001", "prob_up": 0.50}])
        out = ListGenerator().emit(cands)
        assert out["empty"] and len(out["list"]) == 0

    def test_bear_tightening(self):
        """bear: prob 门槛按 entry_prob_bear/entry_prob 比率收紧 [E11]."""
        cands = _cands(
            [
                {"symbol": "600001", "prob_up": 0.70},  # 基准=(.70+.60)/2=.65
                {"symbol": "600002", "prob_up": 0.60},  # 两种状态都过不了 prob 闸
            ]
        )
        # range: 0.70 > 0.65 → 过
        assert len(ListGenerator().emit(cands, market_state="range")["list"]) == 1
        # bear: 0.70 < 0.65×(0.65/0.60)≈0.704 → 不过
        assert ListGenerator().emit(cands, market_state="bear")["empty"]

    def test_bear_requires_positive_3d(self):
        """bear 下 compound>0 但 pred_3d<0 也被剔 (最近端净预期必须为正).

        1d 视界 2026-08-09 删除, bear 门改用最近端 3d.
        """
        cands = _cands(
            [
                {
                    "symbol": "600001",
                    "prob_up": 0.90,  # 远高于 bear 收紧后基准
                    "pred_ret_3d": -0.005,
                    "pred_ret_5d": 0.08,
                },
                {"symbol": "600002", "prob_up": 0.50},
            ]
        )
        # range: compound = pred_ret_10d = .07 > 0 → 过
        assert len(ListGenerator().emit(cands, market_state="range")["list"]) == 1
        # bear: pred_3d < 0 → 剔
        assert ListGenerator().emit(cands, market_state="bear")["empty"]

    def test_gate_q50_uses_3d5d_medians_not_1d(self):
        """闸3 (2026-08-09): 有 3d/5d 中位数列时用它们 (均须为正), 不再用 1d pred_q50.

        1d 中位数可为负 (T+1 不可执行, 旧闸误杀), 3d/5d 均为正即过闸;
        2d 视界 2026-08-09 删除, 2d 中位数列不再参与.
        """
        gen = ListGenerator(entry_prob=0.0)  # 跳过 prob 闸, 单独验闸3
        cands = _cands(
            [
                {
                    "symbol": "600001",
                    "pred_q50": -0.003,  # 1d 中位数负 (旧闸会误杀, 现在不影响)
                    "pred_q50_3d": 0.012,  # 3d 中位数正
                    "pred_q50_5d": 0.020,  # 5d 中位数正
                },
                {  # 3d 中位数为负 → 剔
                    "symbol": "600002",
                    "pred_q50_3d": -0.005,
                    "pred_q50_5d": 0.020,
                },
                {  # 5d 中位数为负 → 剔
                    "symbol": "600003",
                    "pred_q50_3d": 0.012,
                    "pred_q50_5d": -0.001,
                },
            ]
        )
        out = gen.emit(cands)
        assert list(out["list"]["symbol"]) == ["600001"]

    def test_gate_q50_falls_back_to_1d_when_no_3d5d(self):
        """旧 bundle (无 3d/5d 中位数列) 回退 1d pred_q50 闸."""
        gen = ListGenerator(entry_prob=0.0)
        cands = _cands(
            [
                {"symbol": "600001", "pred_q50": -0.003},  # 1d 中位数负 → 剔
                {"symbol": "600002"},  # 无 pred_q50 → fillna(compound) 正 → 过
            ]
        )
        out = gen.emit(cands)
        assert list(out["list"]["symbol"]) == ["600002"]

    def test_emit_ranks_by_magnitude(self):
        """排序 (2026-08-07 定案): 纯 pred_ret_10d 幅度降序 (close-to-close 实得口径赢 3d/组合,
        diag_10d_point_ret_20260807_100807; 旧 3d 混合降级影子).

        10d 幅度最高 → 第 1, 即使 prob/score 都低; 且 3d 幅度刻意反序 —
        600001 3d 最低却 10d 最高, 若误按 3d 排应得 [600002,600003,600001], 断言即失效.
        """
        gen = ListGenerator(entry_prob=0.0)
        cands = _cands(
            [
                {  # 10d 幅度最高 (但 3d 幅度最低, prob/score 都低) → 纯 10d 幅度仍第 1
                    "symbol": "600001",
                    "pred_ret_10d": 0.08,
                    "pred_ret_3d": 0.01,
                    "prob_up_3d": 0.55,
                    "score": 0.20,
                },
                {  # 10d 居中 (但 3d 最高, prob/score 最高) → 第 2
                    "symbol": "600002",
                    "pred_ret_10d": 0.05,
                    "pred_ret_3d": 0.08,
                    "prob_up_3d": 0.80,
                    "score": 0.95,
                },
                {  # 10d 幅度最低 → 最后
                    "symbol": "600003",
                    "pred_ret_10d": 0.02,
                    "pred_ret_3d": 0.05,
                    "prob_up_3d": 0.65,
                    "score": 0.50,
                },
            ]
        )
        out = gen.emit(cands)
        assert list(out["list"]["symbol"]) == ["600001", "600002", "600003"]

    def test_base_rate_is_per_board(self):
        """E7 概率闸 base_rate 按板块独立 (2026-08-09 修复).

        全市场单一 base_rate 会被双创退化模型的常数高概率 (1 树 LGBM → 常数 0.58/0.60)
        抬高混合基线, 主板 prob_up_10d 系统性偏低 → 整块被 E7 误杀 (08-07 起全 dual 清单根因).
        各板块对自家基线比较: main 0.50 vs 自家 0.495, 不被 dual 0.60 拖到混合 0.5475.
        2026-08-14: dual(GEM/STAR) 概率闸另加 +0.08 边际 (LEGACY_ENTRY_GATE.prob_margin).
        """
        cands = _cands(
            [
                {"symbol": "600001", "board": "main", "prob_up_10d": 0.50},
                {"symbol": "600002", "board": "main", "prob_up_10d": 0.49},
                {"symbol": "300001", "board": "GEM", "prob_up_10d": 0.78},
                {"symbol": "300002", "board": "GEM", "prob_up_10d": 0.60},
            ]
        )
        gen = ListGenerator(entry_prob=0.60)
        scored = gen.compute_scores(cands)
        # 板块内 base_rate = 自家 prob 均值 (main 0.495 / GEM 0.69), 非全市场混合 0.5925
        assert scored.loc[scored["symbol"] == "600001", "base_rate"].iloc[
            0
        ] == pytest.approx(0.495)
        assert scored.loc[scored["symbol"] == "300001", "base_rate"].iloc[
            0
        ] == pytest.approx(0.69)
        passed = gen.entry_filter(scored, market_state="range")
        # 每板块各保留"高于自家基线(+边际)"者: main 600001 / GEM 300001 (0.78 > 0.69+0.08);
        # main 600002 与 GEM 300002 低于门槛被剔
        assert sorted(passed["symbol"]) == ["300001", "600001"]

    def test_prob_margin_dual_only(self):
        """分板块概率边际 (2026-08-14 全量 250d OOS 定案): dual(GEM/STAR) 概率闸再收紧
        +0.08 → 命中 62.2→66.3% / 实得 +5.84→+6.66%; main 坍缩期无法评估保持 0.

        dual 股须 prob > base_rate + 0.08 (基准率=20日滚动板块日均), main 只需 > base_rate.
        GEM base = (0.70×4 + 0.78 + 0.82)/6 = 0.7333: 门槛 0.8133.
        0.78 在无边际时过 (0.78 > 0.7333)、有边际时剔 — 判别性用例.
        """
        cands = _cands(
            [
                # main: base=0.555, 0.56 > 0.555 → 过; 0.55 → 剔 (无边际)
                {"symbol": "600001", "board": "main", "prob_up": 0.56},
                {"symbol": "600002", "board": "main", "prob_up": 0.55},
                # GEM: 0.82 > 0.8133 → 过; 0.78 无边际才过; 0.70 剔
                {"symbol": "300001", "board": "GEM", "prob_up": 0.82},
                {"symbol": "300002", "board": "GEM", "prob_up": 0.78},
                {"symbol": "300003", "board": "GEM", "prob_up": 0.70},
                {"symbol": "300004", "board": "GEM", "prob_up": 0.70},
                {"symbol": "300005", "board": "GEM", "prob_up": 0.70},
                {"symbol": "300006", "board": "GEM", "prob_up": 0.70},
            ]
        )
        gen = ListGenerator(entry_prob=0.60)
        scored = gen.compute_scores(cands)
        passed = gen.entry_filter(scored, market_state="range")
        assert sorted(passed["symbol"]) == ["300001", "600001"]

    def test_pain_max_dual_only(self):
        """分板块疼痛闸 (2026-08-14 全量 250d OOS 定案, LEGACY_ENTRY_GATE.pain_max):
        dual(GEM/STAR) pain≤0.4 / main pain≤0.5. 叠加边际后 命中 66.3→75.3% /
        实得 +6.66→+7.39% (出票日 177→76 宁缺毋滥); 0.3 过严崩勿再收.

        判别用例: dual pain 0.45 (0.4 下剔) vs main pain 0.45 (0.5 下过).
        """
        cands = _cands(
            [
                # main: base=(0.61+0.65+0.50)/3=0.5867; 0.61/0.65 过 prob, 0.50 剔 (填充)
                {
                    "symbol": "600001",
                    "board": "main",
                    "prob_up": 0.61,
                    "pain_prob": 0.45,
                },
                {
                    "symbol": "600002",
                    "board": "main",
                    "prob_up": 0.65,
                    "pain_prob": 0.60,
                },
                {
                    "symbol": "600003",
                    "board": "main",
                    "prob_up": 0.50,
                    "pain_prob": 0.10,
                },
                # GEM: base=(0.90+0.90+0.60)/3=0.80, 门槛 0.88; 0.60 剔 (填充)
                {
                    "symbol": "300001",
                    "board": "GEM",
                    "prob_up": 0.90,
                    "pain_prob": 0.35,
                },
                {
                    "symbol": "300002",
                    "board": "GEM",
                    "prob_up": 0.90,
                    "pain_prob": 0.45,
                },
                {
                    "symbol": "300003",
                    "board": "GEM",
                    "prob_up": 0.60,
                    "pain_prob": 0.10,
                },
            ]
        )
        gen = ListGenerator(entry_prob=0.60)
        scored = gen.compute_scores(cands)
        passed = gen.entry_filter(scored, market_state="range")
        # 600001: pain 0.45 ≤ main 0.5 → 过; 600002: 0.6 > 0.5 → 剔
        # 300001: pain 0.35 ≤ dual 0.4 → 过; 300002: 0.45 > 0.4 → 剔
        assert sorted(passed["symbol"]) == ["300001", "600001"]

    def test_announce_blacklist_hard_excludes(self):
        """announce_score == -1.0 (公告事件窗禁买标记) → 硬剔除, 非仅 ×0.7 惩罚.

        安全网 #17: attach_scores 对事件窗口标的置 -1.0 并标记"禁买"; 旧实现只在
        compute_scores 里乘 (1+0.3×-1.0)=0.7, 标记票仍可进 top-N → 禁买不生效.
        """
        cands = _cands(
            [
                {"symbol": "600001", "announce_score": 0.0},  # 无公告 → 过
                {"symbol": "600002", "announce_score": -1.0},  # 禁买标记 → 硬剔
            ]
        )
        gen = ListGenerator(entry_prob=0.0)  # 跳过 prob 闸, 单独验公告禁买
        out = gen.emit(cands)
        assert list(out["list"]["symbol"]) == ["600001"]


class TestScorePainPenalty:
    def test_pain_penalty_lowers_score(self):
        cands = _cands(
            [
                {"symbol": "600001"},
                {"symbol": "600002", "pain_prob": 0.6},
            ]
        )
        gen = ListGenerator()
        scored = gen.compute_scores(cands).set_index("symbol")
        # pain_prob=0.6 → ×(1-0.5×0.6)=×0.7 → 排序分低于无惩罚票
        assert scored.loc["600002", "score"] < scored.loc["600001", "score"]


class TestDistributionWeights:
    @staticmethod
    def _dist_cands(n_normal=11) -> pd.DataFrame:
        inds = ["白酒", "电池", "保险", "半导体"]
        rows = [
            {
                "symbol": f"600{i:03d}",
                "industry": inds[i % 4],
                "pred_q50": 0.03,
                "ATR_pct": 0.02,
                "uncertainty_width": 0.04,
            }
            for i in range(n_normal)
        ]
        rows.append(
            {
                "symbol": "609999",
                "industry": "白酒",
                "pred_q50": 0.03,
                "ATR_pct": 0.02,
                "uncertainty_width": 0.20,
            }
        )  # 不确定性大 → 权重低
        return _cands(rows)

    def test_uncertainty_damps_weight(self):
        # escape hatch: 权重测试与准入闸无关, 跳过闸门 (GATE_OFF)
        gen = ListGenerator(entry_prob=0.0, entry_ret_mult=0.0)
        out = gen.emit(self._dist_cands())
        w = out["list"].set_index("symbol")["weight"]
        assert w["609999"] < w.drop("609999").min()

    def test_single_cap_and_no_renorm(self):
        # 5 只等原始权重: 每只 raw 权重 0.2 → clip 至 0.10, 不再归一 (余下留现金)
        cands = _cands([{"symbol": f"6000{i:02d}"} for i in range(5)])
        cands["industry"] = ["白酒", "电池", "保险", "半导体", "白酒"]
        gen = ListGenerator(entry_prob=0.0, entry_ret_mult=0.0)  # 跳过闸门
        out = gen.emit(cands)
        assert (out["list"]["weight"] <= 0.10 + 1e-4).all()  # 单票上限
        assert out["list"]["weight"].sum() == pytest.approx(0.5, abs=0.01)


# ============================================================
# E8 相关性簇阻断
# ============================================================
class TestClusterBlock:
    def _ret_matrix(self, days=25):
        rng = np.random.default_rng(11)
        a = rng.normal(0, 0.02, days)
        return pd.DataFrame(
            {
                "A": a,
                "B": a + rng.normal(0, 0.001, days),
                "C": rng.normal(0, 0.02, days),
            }
        )

    def test_cluster_grouping(self):
        clusters = cluster_block(["A", "B", "C"], self._ret_matrix())
        merged = [c for c in clusters if "A" in c]
        assert "B" in merged[0]  # A/B 高相关归同簇
        assert all(not ("C" in c and "A" in c) for c in clusters)

    def test_cluster_cap_enforced(self):
        ret = self._ret_matrix()
        clusters = cluster_block(["A", "B", "C"], ret)
        w = pd.Series({"A": 0.10, "B": 0.10, "C": 0.10})
        out = apply_cluster_caps(w, clusters, cap=0.15)
        ab = [c for c in clusters if "A" in c][0]
        assert out.loc[list(ab)].sum() == pytest.approx(0.15)
        assert out["C"] == pytest.approx(0.10)  # 独立簇不受影响


# ============================================================
# E9 波动率熔断
# ============================================================
class TestVolBreaker:
    def test_amplitude_fuse(self):
        assert vol_breaker_multiplier(0.04, 0.01, 0.01) == 0.10  # 振幅>3%

    def test_vol_surge_damp(self):
        assert vol_breaker_multiplier(0.02, 0.02, 0.01) == pytest.approx(0.6)

    def test_fuse_and_surge_stack(self):
        assert vol_breaker_multiplier(0.04, 0.02, 0.01) == pytest.approx(0.06)

    def test_normal(self):
        assert vol_breaker_multiplier(0.02, 0.01, 0.01) == 1.0


# ============================================================
# E11 熊市作战协议
# ============================================================
class TestBearProtocol:
    def _bear_market(self):
        close = pd.Series([100.0] * 19 + [95.0] * 6)  # 收盘 < MA20
        macd_hist = pd.Series([0.5, -0.1])  # 死叉
        return close, macd_hist

    def test_normal_to_defense(self):
        bp = BearProtocol()
        close, hist = self._bear_market()
        r = bp.update(close, hist)
        assert r["action"] == "FULL_EXIT" and r["cash_to"] == "REVERSE_REPO"
        assert bp.state == "DEFENSE"

    def test_recovery_requires_ic_confirmation(self):
        bp = BearProtocol()
        close, hist = self._bear_market()
        bp.update(close, hist)
        # IC 不足 → 维持 DEFENSE
        r = bp.update(close, hist, daily_ics_5d=[0.01] * 5)
        assert r["action"] == "HOLD" and bp.state == "DEFENSE"
        # IC > 0.02 → RECOVERY, 首周收紧参数
        r = bp.update(close, hist, daily_ics_5d=[0.03] * 5)
        assert r["action"] == "RESUME" and r["pos_cap"] == 0.30
        assert r["single_cap"] == 0.07 and r["cluster_cap"] == 0.12
        # 第二周 IC 维持 → 回 NORMAL
        bp.update(close, hist, daily_ics_5d=[0.03] * 5)  # week 2
        r = bp.update(close, hist, daily_ics_5d=[0.03] * 5)
        assert r["action"] == "NORMAL" and bp.state == "NORMAL"

    def test_recovery_fails_back_to_defense(self):
        bp = BearProtocol()
        close, hist = self._bear_market()
        bp.update(close, hist)
        bp.update(close, hist, daily_ics_5d=[0.03] * 5)  # → RECOVERY
        r = bp.update(close, hist, daily_ics_5d=[0.005] * 5)  # IC 失守
        assert r["action"] == "FULL_EXIT" and bp.state == "DEFENSE"

    def test_hard_rules(self):
        assert BearProtocol.is_downtrend(90, 100)  # 50MA<200MA → 禁抄底
        assert not BearProtocol.is_downtrend(110, 100)
        assert BearProtocol.knife_catching_ban(-0.08)  # 跌>7% 禁买
        assert not BearProtocol.knife_catching_ban(-0.05)
        assert BearProtocol.sector_fuse({"白酒": 2, "电池": 1}) == {"白酒"}

    def test_tightened_params(self):
        bp = BearProtocol()
        assert bp.tightened_params() == {}  # NORMAL 无收紧
        close, hist = self._bear_market()
        bp.update(close, hist)
        p = bp.tightened_params()
        assert p["prob_entry"] == 0.65 and p["liquidity_ratio"] == 0.005


# ============================================================
# E5 回测分层滑点 + Sortino + E9/E11 乘数钩子
# ============================================================
class TestBacktestV38:
    def _panel(self, days=30):
        dates = pd.bdate_range("2025-01-01", periods=days)
        frames = []
        for sym in ("600000", "600001"):
            close = 100 * np.cumprod(1 + np.full(days, 0.001))
            frames.append(
                pd.DataFrame(
                    {
                        "symbol": sym,
                        "date": dates,
                        "open": close,
                        "high": close * 1.01,
                        "low": close * 0.99,
                        "close": close,
                        "pre_close": pd.Series(close).shift(1).fillna(close[0]),
                        "industry": "白酒",
                        "adv20": 2e8,  # 1~5亿档 → 滑点 0.10%
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def test_tiered_slippage_in_exec_price(self):
        from app.pipeline1.backtest_v35 import BacktestEngineV35, BacktestProtocol

        panel = self._panel()
        eng = BacktestEngineV35(panel, BacktestProtocol())
        d = sorted(panel["date"].unique())[5]
        px_tiered = eng._exec_buy_price(d, "600000")
        eng_fixed = BacktestEngineV35(panel, BacktestProtocol(tiered_slippage=False))
        px_fixed = eng_fixed._exec_buy_price(d, "600000")
        # 分层 0.10% > 固定 0.05% → 买价更高
        assert px_tiered > px_fixed
        base = panel[(panel["date"] == d) & (panel["symbol"] == "600000")]["open"].iloc[
            0
        ]
        assert px_tiered == pytest.approx(base * 1.001)

    def test_slippage_sensitivity_multiplier(self):
        from app.pipeline1.backtest_v35 import BacktestEngineV35, BacktestProtocol

        panel = self._panel()
        eng = BacktestEngineV35(panel, BacktestProtocol(slippage_multiplier=2.0))
        d = sorted(panel["date"].unique())[5]
        bar = eng._bar(d, "600000")
        assert eng._slippage_for(bar) == pytest.approx(0.002)  # 0.10% × 2

    def test_sortino_in_metrics_and_multiplier_blocks_buys(self):
        from app.pipeline1.backtest_v35 import BacktestEngineV35

        panel = self._panel()
        lists = {
            d: pd.DataFrame(
                {
                    "symbol": ["600000"],
                    "score": [1.0],
                    "prob_up": [0.7],
                    "industry": ["白酒"],
                }
            )
            for d in sorted(panel["date"].unique())[:-1]
        }
        r = BacktestEngineV35(panel).run(lists)
        assert "sortino" in r["metrics"]
        # E9/E11: 乘数 0 → 当日不开新仓
        zero_mult = {d: 0.0 for d in sorted(panel["date"].unique())}
        r0 = BacktestEngineV35(panel).run(lists, daily_multiplier=zero_mult)
        assert len(r0["trades"][r0["trades"]["side"] == "buy"]) == 0


# ============================================================
# E4-L2 因子剔除 + Isotonic 校准默认
# ============================================================
class TestScreenerL2:
    def test_l2_evicts_after_ten_negative_periods(self, tmp_path):
        from app.pipeline1.ic_screener import ICScreener

        dates = pd.bdate_range("2024-01-01", periods=70)
        rng = np.random.default_rng(9)
        rows = []
        for d in dates:
            f = rng.normal(size=30)
            for i in range(30):
                # 因子与未来收益强负相关 → 窗口 IC 持续为负
                rows.append(
                    {
                        "date": d,
                        "factor": f[i],
                        "label_1d": -f[i] + rng.normal(0, 0.01),
                        "label_2d": -f[i] + rng.normal(0, 0.01),
                        "label_3d": -f[i] + rng.normal(0, 0.01),
                        "label_5d": -f[i] + rng.normal(0, 0.01),
                    }
                )
        df = pd.DataFrame(rows)
        df["label_cls"] = (df["label_1d"] > 0).astype(float)  # auc_score 需要
        sc = ICScreener(registry_path=str(tmp_path))
        r1 = sc.screen(df, ["factor"], "w1")
        assert r1["detail"]["factor"]["grade"] == "strong"  # 强度(绝对值)仍强
        # L2_NEG_PERIODS=10: 连续 10 期窗口 IC 为负才剔除
        r = None
        for w in range(2, 12):
            r = sc.screen(df, ["factor"], f"w{w}")
        # L2 驱逐: strong → weak (仍参与训练, 非 dead)
        assert r["detail"]["factor"]["grade"] == "weak"
        assert r["detail"]["factor"]["l2_evicted"] is True
        assert "factor" in r["factors"]


class TestIsotonicDefault:
    def test_default_method_isotonic(self):
        from app.pipeline1.prob_calibrator import ProbCalibrator

        rng = np.random.default_rng(5)
        raw = rng.uniform(0.3, 0.7, 500)
        y = (raw + rng.normal(0, 0.15, 500) > 0.5).astype(float)
        cal = ProbCalibrator().fit(raw, y)  # 默认 isotonic [V3.8]
        assert cal.method == "isotonic"
        prob = cal.predict_proba(raw)
        assert not np.allclose(prob, raw)
        # Isotonic 单调性: 校准曲线单调不减
        order = np.argsort(raw)
        assert (np.diff(prob[order]) >= -1e-12).all()


# ============================================================
# 10d 视界推理 (Gap 2: bundle 含/缺 10d 模型 → 输出/NaN 回退)
# ============================================================
class _FakeConstModel:
    """恒定输出模型: predict → 常量, predict_proba → 二分类列向量."""

    def __init__(self, const: float):
        self.const = float(const)

    def predict(self, X):
        return np.full(len(X), self.const)

    def predict_proba(self, X):
        p = float(np.clip(self.const, 0.05, 0.95))
        return np.column_stack([np.full(len(X), 1 - p), np.full(len(X), p)])


class _FakeConstCal:
    """恒定输出校准器 (predict_proba 单调映射, 不改变量纲)."""

    def predict_proba(self, raw):
        return np.clip(np.asarray(raw, dtype=float), 0.05, 0.95)


def _save_10d_bundle(tmp_path, with_10d: bool) -> str:
    """落盘 fake bundle (feature_cols=['f1'], 全视界 reg/cls + 校准器).

    with_10d=False → 模拟旧 bundle (无 10d_reg/10d_cls), 触发级联 NaN 回退.
    """
    import app.pipeline1.dual_track_trainer as dtt

    models = {
        "1d_reg": (_FakeConstModel(0.01), "label_1d"),
        "2d_reg": (_FakeConstModel(0.02), "label_2d"),
        "3d_reg": (_FakeConstModel(0.03), "label_3d"),
        "5d_reg": (_FakeConstModel(0.05), "label_5d"),
        "1d_cls": (_FakeConstModel(0.70), "label_1d_cls"),
        "2d_cls": (_FakeConstModel(0.72), "label_2d_cls"),
        "3d_cls": (_FakeConstModel(0.74), "label_3d_cls"),
        "5d_cls": (_FakeConstModel(0.76), "label_5d_cls"),
    }
    calibrators = {k: _FakeConstCal() for k in (2, 3, 5)}
    if with_10d:
        models["10d_reg"] = (_FakeConstModel(0.08), "label_10d")
        models["10d_cls"] = (_FakeConstModel(0.78), "label_10d_cls")
        calibrators[10] = _FakeConstCal()
    trained = {
        "board": "main",
        "feature_cols": ["f1"],
        "models": models,
        "calibrator": _FakeConstCal(),
        "calibrators": calibrators,
    }
    trainer = dtt.DualTrackTrainer(model_dir=str(tmp_path))
    return trainer.save(trained, "t")


def _10d_feats() -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=3)
    rows = []
    for d in dates:
        for sym in ("600001", "600002"):
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    "f1": 1.0,
                    "board": "main",
                    "industry": "白酒",
                }
            )
    return pd.DataFrame(rows)


class TestPredictor10d:
    def test_bundle_with_10d_populates_columns(self, tmp_path):
        from app.pipeline1.predictor import V35Predictor

        path = _save_10d_bundle(tmp_path, with_10d=True)
        out = V35Predictor({"main": path}).predict(_10d_feats(), "main")
        assert "pred_ret_10d" in out.columns and "prob_up_10d" in out.columns
        assert out["pred_ret_10d"].notna().all()
        assert np.isfinite(out["pred_ret_10d"]).all()
        assert np.allclose(out["pred_ret_10d"], 0.08)
        assert out["prob_up_10d"].between(0, 1).all()
        assert np.allclose(out["prob_up_10d"], 0.78)

    def test_bundle_without_10d_nan_fallback(self, tmp_path):
        from app.pipeline1.predictor import V35Predictor

        path = _save_10d_bundle(tmp_path, with_10d=False)
        out = V35Predictor({"main": path}).predict(_10d_feats(), "main")
        assert "pred_ret_10d" in out.columns and "prob_up_10d" in out.columns
        assert out["pred_ret_10d"].isna().all()
        assert out["prob_up_10d"].isna().all()

    def test_predict_keeps_pool_blend_passthrough_columns(self, tmp_path):
        """[08-21 fail-open 修复] liquidity_score/date 必须透传到输出.

        pool_blend_cut 按 (date, board) 分组、用 liquidity_score 池分切池;
        keep 白名单曾丢这两列 → 生产切池静默失效."""
        from app.pipeline1.predictor import V35Predictor

        feats = _10d_feats()
        feats["liquidity_score"] = 0.5
        path = _save_10d_bundle(tmp_path, with_10d=True)
        out = V35Predictor({"main": path}).predict(feats, "main")
        assert "liquidity_score" in out.columns
        assert np.allclose(out["liquidity_score"], 0.5)
        assert "date" in out.columns
