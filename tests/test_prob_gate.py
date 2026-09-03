"""Tests for app/pipeline_parallel/prob_head 真模型概率闸 (2026-08-15 定案).

交易规则: 短名单在 t3 门后、pred_mag_10d TOP-5 排名前过边际闸 —
  保留 ⇔ pred_prob > base_rate + margin (margin=0.08, base_rate=近20个可观测日
  mfe_3d 达标率均值). 铁律覆盖:
  - 无前瞻: base_rate 只用 mfe_3d 已可观测的日 (窗口需 +4 交易日未来价, 尾 4 行 NaN)
  - fail-open: bundle 缺失/过旧/当日截面不可用 → 大声告警 + 不杀清单
  - feature_cols 只进数值特征 (字符串列曾致 to_numpy 崩溃), raw 价格量额/meta/label 全剔
  - WORM bundle 训练→加载→预测 roundtrip; 特征缺列 (schema 漂移) → predict raise
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from app.pipeline_parallel import prob_head

# 生产口径成本 (label_engine): COST + 2×滑点, adv20=6e8 → 最高档 0.05% 单边
COST_TOTAL = 0.0013 + 2 * 0.0005


def test_feature_cols_numeric_filter_and_exclusions():
    """只有数值特征进模型; raw/meta/label/pred/score/派生列与字符串列全剔."""
    t = pd.DataFrame(
        {
            "symbol": ["A"],
            "date": ["2026-01-01"],
            "industry": ["银行"],  # 字符串 → 剔除 (曾致 to_numpy 崩溃)
            "open": [1.0],
            "close": [1.0],
            "adv20": [1.0],
            "mfe_3d": [0.1],
            "label_pain": [False],
            "label_pm_3d_net": [0.1],
            "pred_foo": [0.1],
            "score": [0.5],
            "f_num": [1.0],
            "f_num2": [2.0],
        }
    )
    assert prob_head.feature_cols(t) == ["f_num", "f_num2"]


def test_add_mfe_3d_formula_and_group_isolation():
    """mfe_3d = max(high[T+2..T+4]) / close[T+1] - 1 - cost; 尾 4 行 NaN; 不跨股泄漏."""
    n = 10
    df = pd.DataFrame(
        {
            "symbol": ["A"] * n + ["B"] * n,
            "date": pd.to_datetime([f"2026-08-{d:02d}" for d in range(1, n + 1)] * 2),
            "close_hfq": [10.0] * (2 * n),
            "high_hfq": [12.0] * n + [11.0] * n,  # B 低价验证分组隔离
            "adv20": [6e8] * (2 * n),
        }
    )
    out = prob_head._add_mfe_3d(df)
    exp_a = 12.0 / 10.0 - 1 - COST_TOTAL
    exp_b = 11.0 / 10.0 - 1 - COST_TOTAL
    assert np.isclose(out.loc[0, "mfe_3d"], exp_a)
    assert np.isclose(out.loc[n, "mfe_3d"], exp_b)  # B 首行用 B 自身未来价
    # 每 symbol 前 6 行定义, 尾 4 行 NaN (窗口需 +4 交易日未来价)
    for block in (slice(0, n), slice(n, 2 * n)):
        assert out["mfe_3d"].iloc[block].iloc[:6].notna().all()
        assert out["mfe_3d"].iloc[block].iloc[6:].isna().all()


def test_base_rate_no_lookahead_and_mixed_pattern():
    """base_rate = 最近20个可观测日达标率均值; 尾4行 NaN 必须 dropna 后取尾, 不能 rolling."""
    n = 30
    # close 交替 12/10 → mfe 交替 -0.0023(漏)/+0.1977(中), 只依赖 close[i+1] 可精确预期
    df = pd.DataFrame(
        {
            "symbol": ["A"] * n,
            "date": pd.date_range("2026-06-01", periods=n, freq="B"),
            "close_hfq": [12.0 if i % 2 == 0 else 10.0 for i in range(n)],
            "high_hfq": [12.0] * n,
            "adv20": [6e8] * n,
        }
    )
    base = prob_head._base_rate(df)
    # 可观测 26 日 (尾 4 行 NaN), 近 20 日 = 10 中 10 漏 → 0.5
    assert base == pytest.approx(0.5)


def test_base_rate_insufficient_observable_days():
    """可观测日 < base_rate_days → None (闸失效 fail-open)."""
    df = pd.DataFrame(
        {
            "symbol": ["A"] * 10,
            "date": pd.date_range("2026-06-01", periods=10, freq="B"),
            "close_hfq": [12.0 if i % 2 == 0 else 10.0 for i in range(10)],
            "high_hfq": [12.0] * 10,
            "adv20": [6e8] * 10,
        }
    )
    assert prob_head._base_rate(df) is None  # 可观测仅 6 日 < 20


def test_bundle_age_trading_days():
    dates = pd.to_datetime(
        ["2026-08-01", "2026-08-04", "2026-08-05", "2026-08-06"]
    ).to_numpy("datetime64[ns]")
    assert prob_head.bundle_age_trading_days(dates, "2026-08-06") == 0
    assert prob_head.bundle_age_trading_days(dates, "2026-08-01") == 3
    assert prob_head.bundle_age_trading_days(dates, "2026-08-03") is None  # 非交易日


def test_train_load_predict_roundtrip(monkeypatch, tmp_path):
    """WORM bundle 训练→加载→预测; 特征缺列 (schema 漂移) → raise."""
    monkeypatch.setitem(prob_head.PROB_GATE, "model_dir", str(tmp_path))
    monkeypatch.setitem(prob_head.LGB_PARAMS, "n_estimators", 20)
    monkeypatch.setitem(prob_head.LGB_PARAMS, "num_leaves", 7)
    rng = np.random.default_rng(42)
    n = 6000
    t = pd.DataFrame(rng.uniform(-1, 1, (n, 10)), columns=[f"f{i}" for i in range(10)])
    t["mfe_3d"] = rng.uniform(-0.02, 0.08, n)
    t["label_pain"] = False
    t["symbol"] = "SZ000001"
    t["date"] = pd.date_range("2024-01-01", periods=n, freq="B")

    path = prob_head.train_bundle("main", t, "2024-12-31", half_life=60)
    assert path.name.startswith("main_prob_hl60_") and path.suffix == ".joblib"

    b = prob_head.load_latest_tier("main", 60)
    assert b["board"] == "main"
    assert b["feat_cols"] == [f"f{i}" for i in range(10)]
    pred = prob_head.predict(b, t.head(50))
    assert len(pred) == 50
    assert ((pred >= 0) & (pred <= 1)).all()
    with pytest.raises(ValueError):
        prob_head.predict(b, t.drop(columns=["f3"]))

    # 集成装载: 配置档位全在 → 逐档返回; 缺任一档 → None (闸失效判据)
    monkeypatch.setitem(prob_head.PROB_GATE, "half_lives", [60, 7])
    assert prob_head.load_all_tiers("main") is None  # hl7 缺
    path7 = prob_head.train_bundle("main", t, "2024-12-31", half_life=7)
    assert path7.name.startswith("main_prob_hl7_")
    got = prob_head.load_all_tiers("main")
    assert sorted(got) == [7, 60]


def test_apply_prob_gate_drop_and_failopen(monkeypatch):
    """t3 门后: pred_prob ≤ base+margin 剔除; 闸不可用 (None) → fail-open 全留."""
    res = pd.DataFrame({"board": ["main", "main", "dual"], "symbol": ["A", "B", "C"]})
    prob = pd.Series({"A": 0.60, "B": 0.10})

    def fake(board):
        return (prob, 0.20) if board == "main" else None

    monkeypatch.setattr(prob_head, "gate_probabilities", fake)
    out = prob_head.apply_prob_gate(res)
    thr = 0.20 + prob_head.PROB_GATE["margin"]  # 0.28
    assert thr == pytest.approx(0.28)  # 定案边际 0.08 被测试钉住
    # main: A(0.60) 留, B(0.10) 剔; dual: 闸不可用 fail-open 全留
    assert out["symbol"].tolist() == ["A", "C"]
    # pred_prob 列附给排名键 blend (2026-08-15 A/B 定案): 可用板有值, fail-open 板 NaN
    assert out["pred_prob"].iloc[0] == pytest.approx(0.60)
    assert pd.isna(out["pred_prob"].iloc[1])


def test_apply_prob_gate_missing_prob_keeps_symbol(monkeypatch):
    """个股 pred_prob 缺失 (bundle 特征该股 NaN) → fail-open 保留, 不杀清单."""
    res = pd.DataFrame({"board": ["main", "main"], "symbol": ["A", "B"]})
    monkeypatch.setattr(
        prob_head, "gate_probabilities", lambda board: (pd.Series({"A": 0.60}), 0.20)
    )
    out = prob_head.apply_prob_gate(res)
    assert out["symbol"].tolist() == ["A", "B"]


def test_apply_prob_gate_disabled_returns_unchanged(monkeypatch):
    """enable=False → 原样返回, 不调用闸."""
    monkeypatch.setitem(prob_head.PROB_GATE, "enable", False)
    res = pd.DataFrame({"board": ["main"], "symbol": ["A"]})
    assert prob_head.apply_prob_gate(res)["symbol"].tolist() == ["A"]


# ---- 时间衰减样本加权 (2026-09-03 过闸接入) ----


def test_decay_sample_weights_monotone_and_half_life():
    """w = 0.5**(age/half_life), age 相对最新日: 严格单调递减; age=30 天 → 0.5**0.5."""
    dates = pd.Series(pd.date_range("2024-01-01", periods=121, freq="D"))
    w = prob_head.decay_sample_weights(dates, 60)
    age = (dates.max() - dates).dt.days.to_numpy()
    assert (np.diff(w[np.argsort(age)]) < 0).all()  # 随 age 严格单调递减
    assert w[-1] == pytest.approx(1.0)  # 最新日 age=0
    assert w[-31] == pytest.approx(0.5**0.5)  # age=30 天
    assert w[-61] == pytest.approx(0.5)  # age=60 天 = 半衰期


class _RecordingLGBM:
    """fit 参数记录桩 (模块级定义保证 joblib bundle 可 pickle); 不做真训练."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.fit_kwargs: dict = {}

    def fit(self, x, y, **kwargs):
        self.fit_kwargs = kwargs


def test_train_bundle_sample_weight_wiring(monkeypatch, tmp_path):
    """half_life 非 None → fit 传 sample_weight (len=行数, 日期升序→递减); None → 不传 (回退)."""
    monkeypatch.setitem(prob_head.PROB_GATE, "model_dir", str(tmp_path))
    monkeypatch.setattr(prob_head, "LGBMClassifier", _RecordingLGBM)
    rng = np.random.default_rng(42)
    n = 5100
    t = pd.DataFrame(rng.uniform(-1, 1, (n, 3)), columns=["f0", "f1", "f2"])
    t["mfe_3d"] = rng.uniform(-0.02, 0.08, n)
    t["label_pain"] = False
    t["symbol"] = "SZ000001"
    t["date"] = pd.date_range("2024-01-01", periods=n, freq="B")

    path = prob_head.train_bundle("main", t, "2024-12-31", half_life=30)
    w = joblib.load(path)["model"].fit_kwargs["sample_weight"]
    assert len(w) == n
    assert (np.diff(w) > 0).all()  # 日期升序 → age 降序 → 权重随行升序

    path2 = prob_head.train_bundle("main", t, "2024-12-31", half_life=None)
    assert path2.name.startswith("main_prob_nodecay_")
    assert "sample_weight" not in joblib.load(path2)["model"].fit_kwargs


class _ConstProba:
    """恒定概率预测桩 (predict_proba 契约: 列 0 = 负类, 列 1 = 正类)."""

    def __init__(self, p):
        self.p = p

    def predict_proba(self, x):
        n = len(x)
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])


def test_ensemble_predict_averages_tiers():
    """多半衰期集成 = 逐档 predict_proba[:, 1] 算术均值, index 对齐截面."""
    cs = pd.DataFrame({"f0": [1.0, 2.0]}, index=[10, 20])
    b1 = {"feat_cols": ["f0"], "model": _ConstProba(0.2)}
    b2 = {"feat_cols": ["f0"], "model": _ConstProba(0.6)}
    out = prob_head.ensemble_predict([b1, b2], cs)
    assert out.index.tolist() == [10, 20]
    assert out.tolist() == [pytest.approx(0.4), pytest.approx(0.4)]
