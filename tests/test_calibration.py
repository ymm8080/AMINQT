"""Tests for app/pipeline_parallel/calibration (mag_10d close-to-close 校准, 2026-08-07).

覆盖 (TDD):
  (a) 有足够横截面样本的日期 → mag 列存在且有限
  (b) 无前瞻守卫: 决策日 D 的拟合不得使用卖价 > D 的行 (损坏标签不可见)
  (c) 回退: 每股样本 < per_stock_min_n → 用横截面斜率
  (d) 收缩: shrink_kappa 增大 → 每股 mag 向横截面靠拢
  (e) 集成: build_merged_shortlist 输出按 mag_10d 降序 (在 test_pipeline_parallel 测)

合成面板: 每股 score → label_pm_10d_net 真实线性关系 + 噪声, 保证 OLS 有真实斜率.
"""

import numpy as np
import pandas as pd
import pytest

from app.pipeline_parallel.calibration import calibrate_mag10d

CROSS_MIN_N = 50


def _panel(n_dates=120, n_stocks=12, seed=42, cal_n=42):
    """合成面板: 每股 score 跨日略漂移, label = slope_i*score + intercept_i + 噪声.

    确保每股回归 (per-stock window) 有真实斜率, 且每股斜率不同 (便于测收缩).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-06", periods=n_dates)
    rows = []
    for i in range(n_stocks):
        slope = 0.06 + 0.02 * (i % 3)  # 每股斜率不同
        intercept = -0.01 * (i % 2)
        base = 0.5 + 0.4 * (i % n_stocks) / n_stocks
        for d in dates:
            sc = base + 0.03 * rng.normal()
            y = slope * sc + intercept + 0.002 * rng.normal()
            rows.append(
                {
                    "symbol": f"{'00' if i % 2 == 0 else '30'}{i:04d}",
                    "date": d,
                    "board": "main" if i % 2 == 0 else "dual",
                    "score": sc,
                    "label_pm_10d_net": y,
                }
            )
    return pd.DataFrame(rows)


def test_a_mag_finite_with_enough_cross_section():
    df = _panel()
    out = calibrate_mag10d(df, cal_n=42)
    assert {"symbol", "date", "board", "mag"}.issubset(set(out.columns))
    assert out["mag"].notna().all()
    # 校准窗 cal_n=42, 已实现边界 11 → 决策日最早需 cal_n+11 日后
    assert out["date"].min() > df["date"].min()


def _single_symbol_panel(n_dates=120, seed=1, slope=0.1, intercept=0.0):
    """单股每日一行 → 行 t 的 label 卖价在 t+11 (close_hfq[T+11])."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-06", periods=n_dates)
    rows = []
    for d in dates:
        sc = 0.5 + 0.05 * rng.normal()
        rows.append(
            {
                "symbol": "000001",
                "date": d,
                "board": "main",
                "score": sc,
                "label_pm_10d_net": slope * sc + intercept,
            }
        )
    return pd.DataFrame(rows)


def test_b_lookahead_guard_no_use_of_future_labels():
    """铁律守卫: 决策日 D 的拟合只能用卖价 ≤ close[D] 的行.

    单股面板, cal_n=62 → 横截面 n = 62-11 = 51 ≥ 50. 行 t 可用 ⇔ t+11 ≤ k
    (k = D 的日期索引). 污染 D 前最近 11 个交易日 (卖价尚未打印) 的 label
    → 必须对 D 的 mag 不可见; 而更早决策日 D1 (该区卖价已打印) → 可见.
    """
    clean = _single_symbol_panel()
    dates = clean["date"].to_numpy()

    # 污染索引 75..85 (11 行): 对 D=dates[85], 最后可用行 = k-11 = 74 → 75..85 全不可用
    corrupt = clean.copy()
    corrupt.loc[75:85, "label_pm_10d_net"] = 5.0

    D = dates[85]
    clean_mag = calibrate_mag10d(clean, cal_n=62)
    corrupt_mag = calibrate_mag10d(corrupt, cal_n=62)

    def _mag_at(df_mag, d):
        row = df_mag[df_mag["date"] == d]
        assert len(row) == 1
        return row["mag"].iloc[0]

    # 决策日 D: 污染区 (75..85) 全在不可用区 (卖价 > close[D]) → mag 不受污染影响
    assert _mag_at(clean_mag, D) == pytest.approx(_mag_at(corrupt_mag, D), abs=1e-9)

    # 更早决策日 D1 = dates[92]: 最后可用行 = 92-11 = 81 → 污染 75..81 可见
    D1 = dates[92]
    assert _mag_at(clean_mag, D1) != pytest.approx(_mag_at(corrupt_mag, D1), abs=1e-9)


def test_c_fallback_to_cross_section_when_sparse():
    """每股样本 < per_stock_min_n → 用横截面斜率 (回退)."""
    df = _panel()
    # 把其中一只股票的大部分行 label 置 NaN (只剩 < per_stock_min_n 有效样本)
    sym = df["symbol"].unique()[0]
    mask = df["symbol"] == sym
    df.loc[mask, "label_pm_10d_net"] = np.nan
    # 保留极少有效行 (如 5 行) → 每股样本不足 → 回退横截面
    keep = mask & (df["label_pm_10d_net"].isna())
    df.loc[keep, "label_pm_10d_net"] = 0.5  # 给 5 行有效样本

    # 用较大 per_stock_min_n → 该股必然回退; 横截面仍可用
    out = calibrate_mag10d(df, cal_n=42, per_stock_min_n=200)
    # 回退路径: mag = cs_slope*score + cs_intercept → 应与其它同分股同斜率
    # 简断言: 输出非空, 且每股有 mag (回退不崩)
    assert not out.empty
    assert out["mag"].notna().all()


def test_d_shrinkage_pulls_toward_cross_section():
    """收缩: kappa 增大 → λ=take/(take+κ) 减小 → 每股斜率向横截面靠拢."""
    df = _panel(n_dates=140, n_stocks=20)
    sym = df["symbol"].unique()[0]
    # 选一个每股斜率显著偏离横截面的股票: 该股 label 用极端斜率生成
    df.loc[df["symbol"] == sym, "label_pm_10d_net"] = (
        0.30 * df.loc[df["symbol"] == sym, "score"] + 0.02
    )

    # 显式低 per_stock_min_n: 生产默认 minn=50 (=纯横截面) 会让 cal_n=42 下每股样本
    # (42-11=31) 不足而全部回退 → 收缩路径不被触发; 本测试目标就是收缩逻辑本身.
    out_weak = calibrate_mag10d(
        df, cal_n=42, per_stock_min_n=15, shrink_kappa=1.0
    )  # 弱收缩 → 接近每股
    out_strong = calibrate_mag10d(
        df, cal_n=42, per_stock_min_n=15, shrink_kappa=100.0
    )  # 强收缩 → 接近横截面

    # 取尾日 (每股窗口已满, 避免早期稀疏回退横截面); 该日全股 mag 均值近似横截面参考
    d0 = out_weak["date"].max()
    mag_w = out_weak.loc[
        (out_weak["symbol"] == sym) & (out_weak["date"] == d0), "mag"
    ].iloc[0]
    mag_s = out_strong.loc[
        (out_strong["symbol"] == sym) & (out_strong["date"] == d0), "mag"
    ].iloc[0]
    cross_ref = out_weak.loc[out_weak["date"] == d0, "mag"].mean()
    # 强收缩应比弱收缩更接近横截面
    assert abs(mag_s - cross_ref) < abs(mag_w - cross_ref)


def test_e_lookahead_boundary_exact():
    """边界精确性: 决策日 D, 行 t 可用 ⇔ t+11 的卖价 ≤ close[D].

    用单股面板直接验证: D 前第 11 个交易日 (t = D-11) 的行应可用,
    t = D-10 及更近的不可用. 用污染法验证边界两侧.
    """
    n_dates = 100
    dates = pd.bdate_range("2025-01-06", periods=n_dates)
    rows = []
    slope, intercept = 0.1, 0.0
    for t, d in enumerate(dates):
        rows.append(
            {
                "symbol": "000001",
                "date": d,
                "board": "main",
                "score": 0.5 + 0.01 * t,
                "label_pm_10d_net": slope * (0.5 + 0.01 * t) + intercept,
            }
        )
    clean = pd.DataFrame(rows)
    D = dates[80]

    # 只污染 t = D-11 (边界可用行) → D 的 mag 必须变
    c1 = clean.copy()
    c1.loc[80 - 11, "label_pm_10d_net"] = 5.0
    # 只污染 t = D-10 (边界不可用行) → D 的 mag 必须不变
    c2 = clean.copy()
    c2.loc[80 - 10, "label_pm_10d_net"] = 5.0

    def _mag(df, d):
        return (
            calibrate_mag10d(df, cal_n=60).loc[lambda s: s["date"] == d, "mag"].iloc[0]
        )

    assert _mag(c1, D) != pytest.approx(_mag(clean, D), abs=1e-9)
    assert _mag(c2, D) == pytest.approx(_mag(clean, D), abs=1e-9)


def _single_symbol_panel_any_target(n_dates=100, seed=3, target="label_pm_5d_net"):
    """单股每日一行, 任意目标列 (label_horizon 参数化测试用)."""
    dates = pd.bdate_range("2025-01-06", periods=n_dates)
    rows = []
    for t, d in enumerate(dates):
        rows.append(
            {
                "symbol": "000001",
                "date": d,
                "board": "main",
                "score": 0.5 + 0.01 * t,
                target: 0.1 * (0.5 + 0.01 * t),
            }
        )
    return pd.DataFrame(rows)


def test_f_label_horizon_shifts_realized_boundary():
    """label_horizon=5 → realized_drop = buy_lag(1)+5 = 6.

    决策日 D: 行 t 可用 ⇔ t+6 ≤ D. 污染 D-6 行 (可用) → D 的 mag 必须变;
    污染 D-5 行 (不可用) → D 的 mag 必须不变 (与 10d 的 D-11/D-10 边界同构).
    """
    clean = _single_symbol_panel_any_target()
    D = clean["date"].iloc[80]

    def _mag(df):
        return (
            calibrate_mag10d(
                df, cal_n=60, target_col="label_pm_5d_net", label_horizon=5
            )
            .loc[lambda s: s["date"] == D, "mag"]
            .iloc[0]
        )

    c1 = clean.copy()
    c1.loc[80 - 6, "label_pm_5d_net"] = 5.0  # D-6 可用 → mag 变
    c2 = clean.copy()
    c2.loc[80 - 5, "label_pm_5d_net"] = 5.0  # D-5 不可用 → mag 不变

    assert _mag(c1) != pytest.approx(_mag(clean), abs=1e-9)
    assert _mag(c2) == pytest.approx(_mag(clean), abs=1e-9)


def test_g_label_horizon_default_equals_10d():
    """默认 label_horizon 与显式 10 行为一致 (回归)."""
    df = _panel()
    a = calibrate_mag10d(df, cal_n=42, target_col="label_pm_10d_net")
    b = calibrate_mag10d(df, cal_n=42, target_col="label_pm_10d_net", label_horizon=10)
    assert a["mag"].equals(b["mag"])


def test_h_negative_cross_slope_does_not_invert_ranking():
    """08-14 调查: 21 日窗 cs<0 时旧代码 mag=cs·score+ci 反转排名 → mag-top5=最低分股.

    250d 实测 (main 43% / dual 33% 日子 cs<0): 反序日 mag-top5 实得 +0.59%/+0.83%,
    同日 score-top5 实得 +2.59%/+4.13% → 负斜率是噪声假信号, 未延续. 铁律方向
    (高分=高预期): 负 cs 只取幅度 → mag 恒与 score 单调同序.
    """
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2025-01-06", periods=120)
    rows = []
    for i in range(12):
        base = 0.5 + 0.04 * i
        for d in dates:
            sc = base + 0.02 * rng.normal()
            y = -0.08 * sc + 0.02 + 0.001 * rng.normal()  # 全股负斜率 → 横截面 cs<0
            rows.append(
                {
                    "symbol": f"S{i:02d}",
                    "date": d,
                    "board": "main",
                    "score": sc,
                    "label_pm_10d_net": y,
                }
            )
    df = pd.DataFrame(rows)
    out = calibrate_mag10d(df, cal_n=42)
    d0 = out["date"].max()
    day = out[out["date"] == d0].merge(
        df[["symbol", "date", "score"]], on=["symbol", "date"]
    )
    day = day.sort_values("score")
    # 同序: 分数越高 mag 越高 (永不反转); 旧代码 cs<0 → 严格反序 → 此处 fail
    assert day["mag"].is_monotonic_increasing
