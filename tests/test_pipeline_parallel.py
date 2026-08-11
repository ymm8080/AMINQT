"""Tests for app/pipeline_parallel (并行多系统, 2026-08-04).

覆盖: MFE 净标签 / 池合成打分 / TOP-N 选股 / 双头量测 / 配置完整性 /
run_system 全流程 / OOS 窗口 (合成数据, 不触检查点). 不改动 app/pipeline1.
"""

import numpy as np
import pandas as pd
import pytest

from app.pipeline_parallel.config import (
    C2C_LABELS,
    FUSION,
    HORIZONS,
    MIN_MAG,
    MIN_WINRATE,
    PANEL,
    SLOW_BULL,
    SNIPER,
    SYSTEMS,
)
from app.pipeline_parallel.scoring import (
    cross_rank,
    dual_head_ok,
    measure_dual_head,
    pool_score,
    select_topn,
)


def _panel(n_dates=60, n_stocks=20, extra_cols=()):
    """合成面板: 每日期截面 n_stocks 只, 偶数→主板(00xxxxx), 奇数→双创(30xxxxx).

    f1 = 持久横截面因子 + 噪声 → 高分股今天高明天也高, 选股有真实 edge
    (i.i.d. 特征会让 TOP 选的未来收益趋零, 双头信号测不出来).
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-01-06", periods=n_dates)
    sym_base = rng.normal(0, 1, n_stocks)  # 持久因子
    rows = []
    for d in dates:
        for i in range(n_stocks):
            prefix = "00" if i % 2 == 0 else "30"
            r = {
                "date": d,
                "symbol": f"{prefix}{i:04d}",
                "f1": sym_base[i] + rng.normal(0, 0.1),
                "f2": rng.normal(0, 1),
            }
            for c in extra_cols:
                r[c] = rng.normal(0, 1) + sym_base[i] * 0.5
            rows.append(r)
    return pd.DataFrame(rows)


def _with_labels(df):
    """合成价格路径 + add_mfe_labels (MFE) + add_c2c_labels (close-to-close) + board 列."""
    from app.pipeline_parallel.backtest import add_c2c_labels, add_mfe_labels
    from app.pipeline_parallel.config import board_of

    df = df.copy()
    rng = np.random.default_rng(7)
    day_idx = df.groupby("symbol").cumcount()
    # 漂移用 symbol 平滑 f1 (组均值, 无日噪声): 归一化到 [0,1] → 指数漂移恒正单调,
    # close 永不为负/零. 2026-08-07 并行验收改 close-to-close: MFE 曾靠 high_hfq
    # 放大幅度, c2c 需价格路径本身上行才能过双头门 (threshold main 3%/dual 4%).
    f1 = df["f1"]
    f1_sym = df.groupby("symbol")["f1"].transform("mean")
    f1_pos = (f1_sym - f1_sym.min()) / (f1_sym.max() - f1_sym.min())
    df["close_hfq"] = (
        10.0
        * (1 + 0.12 * f1)
        * np.exp(0.08 * f1_pos * day_idx)
        * np.exp(rng.normal(0, 0.01, len(df)))
    )
    # 日内高点 >= 收盘 (OHLCV 不变量), 放大倍数随 f1 → 高 f1 的 MFE 更大
    df["high_hfq"] = np.maximum(
        df["close_hfq"] * (1 + 0.12 * df["f1"] + rng.normal(0, 0.02, len(df))),
        df["close_hfq"] * 1.0001,
    )
    df["adv20"] = rng.uniform(1e7, 1e9, len(df))
    df = add_mfe_labels(df, horizons=(3, 5, 10))
    # close-to-close 净标签 (生产口径, add_c2c_labels): label_pm_{k}d_net 全视界齐
    # (含 slow_bull 长视界 20/40d; mag_10d 校准目标列 label_pm_10d_net 同款)
    df = add_c2c_labels(df, horizons=(3, 5, 10, 20, 40))
    df["board"] = df["symbol"].map(board_of)
    return df


# ── 配置完整性 ──
def test_config_systems_present():
    assert set(SYSTEMS) == {"sniper", "fusion", "slow_bull"}
    assert SNIPER.enabled and FUSION.enabled and SLOW_BULL.enabled


def test_config_pools_nonempty_and_distinct():
    assert len(SNIPER.pool) >= 5
    assert len(FUSION.pool) >= 5
    assert len(SLOW_BULL.pool) >= 5
    assert set(FUSION.pool) != set(SNIPER.pool)
    assert set(SLOW_BULL.pool) != set(SNIPER.pool)
    assert set(SLOW_BULL.pool) != set(FUSION.pool)


def test_pv_corr_5_in_both_short_horizon_pools():
    # 2026-08-05 池内相关 OOS 边际: dual 全视界 Δwr +1.7~3.6% (生产公式已验证等价)
    assert "pv_corr_5" in SNIPER.pool
    assert "pv_corr_5" in FUSION.pool


def test_down_gap_pct_pruned_from_short_horizon_pools():
    # 2026-08-08 c2c LOO 审计 (250/300/200d OOS): 剔除 down_gap_pct 双板 STABLE_WIN
    # +0.45~+0.72pp — MFE 目标选入, 但 c2c 排名下不兑现 (down-gap→反弹→max-high 不传收盘)
    assert "down_gap_pct" not in SNIPER.pool
    assert "down_gap_pct" not in FUSION.pool


def test_both_systems_full_horizon_matrix():
    # 2026-08-04 用户: TOP-10 也要看 T+2/T+3; 两套统一测 T+2/3/5/10
    for spec in (SNIPER, FUSION):
        assert set(spec.horizons) == set(HORIZONS)
        assert len(spec.horizons) == len(spec.labels)
        for h, lab in zip(spec.horizons, spec.labels, strict=False):
            # 2026-08-07 用户: 并行验收改 close-to-close (可兑现), 非 MFE 触摸天花板
            assert lab == f"label_pm_{h}_net"
    assert set(HORIZONS) == {"3d", "5d", "10d"}
    assert len(C2C_LABELS) == 3


def test_config_panel_checkpoints_exist():
    import os

    if not os.path.exists(PANEL.main_checkpoint):
        pytest.skip("面板检查点文件不存在 (CI 无数据文件)")
    assert os.path.exists(PANEL.main_checkpoint)
    assert os.path.exists(PANEL.dual_checkpoint)


def test_board_of_prefix_mapping():
    from app.pipeline_parallel.config import board_of

    assert board_of("600000") == "main"  # 沪主板
    assert board_of("000001") == "main"  # 深主板
    assert board_of("300001") == "dual"  # 创业板
    assert board_of("688001") == "dual"  # 科创板
    assert board_of("900901") == "main"  # 未知前缀归 main


def test_board_thresholds_differ():
    # 2026-08-04 用户: MAIN/DUAL 上涨幅度阈值必须不同 (dual 20% 上限 > main 10%).
    # 2026-08-10 重锚 (c2c 基准): 旧 0.55/3-4% 是 MFE 时代绝对值 (MFE 全池基准~90%/+8%,
    # c2c 基准~45%/~0 → 结构上不可达), 降到"≥随机 50% + 盈利 1%"诚实基准; dual 仍高于 main.
    from app.pipeline_parallel.config import BOARD_THRESHOLDS

    assert (
        BOARD_THRESHOLDS["main"]["min_winrate"]
        == BOARD_THRESHOLDS["dual"]["min_winrate"]
        == 0.50
    )
    assert (
        BOARD_THRESHOLDS["dual"]["min_mag"] > BOARD_THRESHOLDS["main"]["min_mag"] > 0.0
    )


# ── MFE 净标签 ──
def test_add_mfe_labels_manual():
    from app.pipeline1.label_engine import COST, slippage_tier
    from app.pipeline_parallel.backtest import add_mfe_labels

    dates = pd.bdate_range("2025-01-06", periods=6)
    close = [10.0, 10.1, 10.5, 11.0, 10.8, 10.9]
    high = [10.2, 10.3, 10.8, 11.2, 11.0, 11.1]
    df = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["000001"] * 6,
            "close_hfq": close,
            "high_hfq": high,
            "adv20": [1e8] * 6,
        }
    )
    df = add_mfe_labels(df, horizons=(2,))
    # T 行 (index 0): exec=close[1]=10.1; MFE-2d=max(high[2],high[3])=max(10.8,11.2)=11.2
    cost = COST + 2 * slippage_tier(1e8)
    assert np.allclose(df.loc[0, "label_mfe_2d_net"], 11.2 / 10.1 - 1 - cost)
    # 尾部无未来价 → 标签 NaN (保守, 不可用路径)
    assert pd.isna(df.loc[5, "label_mfe_2d_net"])


def test_add_mfe_labels_never_below_target_date_return():
    # MFE 是窗口最高价, 必 >= 同视界目标日收盘净收益 (成本同口径)
    from app.pipeline1.label_engine import slippage_tier

    df = _with_labels(_panel(n_dates=8, n_stocks=5))
    g = df.groupby("symbol")
    for k, lab in ((3, "label_mfe_3d_net"), (5, "label_mfe_5d_net")):
        close_f = g["close_hfq"].shift(-1)
        close_t = g["close_hfq"].shift(-(k + 1))
        gross = close_t / close_f - 1
        # 生产净标签 (仅作比较, 不写入面板)
        slip = df["adv20"].map(slippage_tier)
        net = gross - (0.0013 + 2 * slip)
        diff = df[lab] - net
        valid = diff.dropna()
        assert len(valid) > 0
        assert (valid >= -1e-9).all()


# ── 可交易性门 (剔除慢性停牌股) ──
def test_tradability_gate_removes_sparse_stock():
    from app.pipeline_parallel.backtest import tradability_gate

    df = _panel(n_dates=10, n_stocks=4)
    sym = "300003"  # i=3 → 双创前缀
    dates = np.sort(df["date"].unique())
    keep_days = dates[-2:]  # 该股只在末 2 天有行 → 前期可交易性差
    df2 = df[(df["symbol"] != sym) | (df["date"].isin(keep_days))]
    kept, stats = tradability_gate(df2, lookback=5, min_presence=0.8)
    assert sym not in set(kept["symbol"])
    assert stats["removed_stocks"] == 1
    assert stats["removed_rows"] == 2  # 该股仅剩的 2 行也被剔 (前 5 日有行比例不足)
    # 正常股不受影响 (整窗在交易)
    assert stats["kept_stocks"] == 3


def test_tradability_gate_keeps_normal_stock():
    from app.pipeline_parallel.backtest import tradability_gate

    df = _panel(n_dates=10, n_stocks=4)
    kept, stats = tradability_gate(df, lookback=5, min_presence=0.8)
    assert stats["removed_stocks"] == 0
    assert len(kept) == len(df)


# ── 打分 ──
def test_cross_rank_bounds():
    df = _panel()
    r = cross_rank(df, "f1")
    assert r.min() >= 0.0 and r.max() <= 1.0
    assert (r.dropna() > 0).all()


def test_pool_score_equal_weight():
    df = _panel()
    s = pool_score(df, ("f1", "f2"))
    s1 = cross_rank(df, "f1")
    s2 = cross_rank(df, "f2")
    assert np.allclose(s, (s1 + s2) / 2)


def test_pool_score_missing_col_skips():
    df = _panel()
    s = pool_score(df, ("f1", "does_not_exist"))
    assert np.allclose(s, cross_rank(df, "f1"))


def test_pool_score_all_missing_raises():
    df = _panel()
    with pytest.raises(ValueError):
        pool_score(df, ("nope_a", "nope_b"))


def test_select_topn_per_date():
    df = _panel()
    s = pool_score(df, ("f1", "f2"))
    top = select_topn(df, s, 5)
    assert len(top) == df["date"].nunique() * 5
    assert top.groupby("date").size().eq(5).all()
    assert top["score"].notna().all()


# ── 双头量测 ──
def test_measure_dual_head_ok():
    df = _with_labels(_panel(extra_cols=("f1",)))
    top = select_topn(df, pool_score(df, ("f1",)), 5)
    sel = top.merge(
        df[["symbol", "date", "label_mfe_3d_net"]], on=["symbol", "date"], how="left"
    )
    m = measure_dual_head(sel, "label_mfe_3d_net")
    assert m["n"] > 0
    assert m["mag"] > 0
    assert m["winrate"] > 0.5
    assert dual_head_ok(m, MIN_WINRATE, MIN_MAG)


def test_dual_head_ok_rejects_low_winrate():
    m = {"mag": 0.01, "winrate": 0.40, "n": 500}
    assert not dual_head_ok(m, 0.55, 0.0)


def test_dual_head_ok_rejects_few():
    m = {"mag": 0.01, "winrate": 0.9, "n": 3}
    assert not dual_head_ok(m, 0.55, 0.0)


# ── run_system 全流程 (合成) ──
def test_run_system_end_to_end():
    from app.pipeline_parallel.backtest import run_system
    from app.pipeline_parallel.config import SNIPER

    df = _with_labels(_panel(extra_cols=SNIPER.pool))
    res = run_system(df, SNIPER, top_n=SNIPER.top_n)
    assert "per_horizon" in res
    assert "passed" in res
    # 四视界都在报告矩阵里
    assert set(res["per_horizon"]) == set(HORIZONS)
    assert res["per_horizon"]["3d"]["n"] > 0
    assert res["n_picks"] == df["date"].nunique() * SNIPER.top_n


def test_run_system_oos_mask_only_subset():
    from app.pipeline_parallel.backtest import run_system
    from app.pipeline_parallel.config import SNIPER

    df = _with_labels(_panel(extra_cols=SNIPER.pool))
    dates = np.sort(df["date"].unique())
    mask = df["date"].values >= dates[-4]
    full = run_system(df, SNIPER, 5)
    oos = run_system(df, SNIPER, 5, mask)
    assert oos["n_picks"] == 4 * 5
    assert oos["n_picks"] < full["n_picks"]


# ── run_all 全流程 (合成) ──
def test_run_all_reports_per_board_oos():
    from app.pipeline_parallel.backtest import run_all
    from app.pipeline_parallel.config import BOARD_PREFIXES, FUSION, SNIPER

    pool = tuple(dict.fromkeys(SNIPER.pool + FUSION.pool))
    df = _with_labels(_panel(extra_cols=pool))
    # oos_days=12: c2c 标签需 k+1 个未来日 (3d 需 4), OOS 窗须足够长让短视界可测
    out = run_all(df, "20260804_000000", oos_days=12)
    assert set(out["boards"]) == set(BOARD_PREFIXES)
    for b in BOARD_PREFIXES:
        systems = out["boards"][b]["systems"]
        assert set(systems) == {"sniper", "fusion", "slow_bull"}
        assert systems["sniper"]["enabled"] is True
        assert systems["fusion"]["enabled"] is True
        assert systems["slow_bull"]["enabled"] is True
        assert "full" in systems["sniper"] and "oos" in systems["sniper"]
        for name in ("sniper", "fusion"):
            s = systems[name]
            # 2026-08-04 用户: 验收只看 OOS, full 不参与保留判定
            assert s["full"]["kept"] is None
            assert set(s["oos"]) == {"oos"}  # oos_days=N → 单窗
            for _w, ow in s["oos"].items():
                assert ow["kept"] is True  # 合成持久因子下 OOS 双头必过
                assert ow["primary"]["passed"]
            assert set(s["full"]["primary"]["per_horizon"]) == set(HORIZONS)
    # OOS 窗口边界正确 (末 12 个交易日)
    dates = np.sort(df["date"].unique())
    assert out["window"]["oos"]["oos"]["start"] == str(dates[-12])
    assert out["window"]["oos"]["oos"]["trading_days"] == 12


def test_run_all_default_windows_6m_3m_10d():
    from app.pipeline_parallel.backtest import run_all
    from app.pipeline_parallel.config import OOS_WINDOWS

    assert set(OOS_WINDOWS) == {"6m", "3m", "10d"}
    assert OOS_WINDOWS["6m"] == 126
    assert OOS_WINDOWS["3m"] == 63
    # 2026-08-04 用户: "THAT IS LAST 15 TRADING DATES" — 10D 窗=末 15 交易日,
    # 使 T+5 (~10 日) 与 T+10 (~5 日) 在同一测试日均可测.
    assert OOS_WINDOWS["10d"] == 15
    # 合成面板只有 14 交易日 → 6m/3m 窗口天数超界, 应抛错 (真实面板才够长)
    df = _with_labels(_panel())
    with pytest.raises(ValueError):
        run_all(df, "20260804_000000")


def test_run_all_rejects_full_oos():
    from app.pipeline_parallel.backtest import run_all

    df = _with_labels(_panel())
    with pytest.raises(ValueError):
        run_all(df, "20260804_000000", oos_days=len(df["date"].unique()))


# ── TOP-5 vs TOP-10 分档对比 ──
def test_rank_bands_splits_top10_by_rank():
    from app.pipeline_parallel.backtest import rank_bands
    from app.pipeline_parallel.config import FUSION

    df = _with_labels(_panel(extra_cols=FUSION.pool))
    res = rank_bands(
        df, FUSION, 10, (("all10", 1, 10), ("first5", 1, 5), ("last5", 6, 10))
    )
    nd = df["date"].nunique()
    # 尾段无未来价 → 标签 NaN, 故用结构性等式而非固定计数
    assert 0 < res["all10"]["3d"]["n"] <= nd * 10
    assert res["first5"]["3d"]["n"] == res["last5"]["3d"]["n"]
    assert res["first5"]["3d"]["n"] + res["last5"]["3d"]["n"] == res["all10"]["3d"]["n"]
    # first5 是高分档, 应 >= last5 的幅度
    assert res["first5"]["3d"]["mag"] >= res["last5"]["3d"]["mag"]


def test_run_all_compare_per_board_oos():
    from app.pipeline_parallel.backtest import run_all
    from app.pipeline_parallel.config import BOARD_PREFIXES, FUSION, SNIPER

    pool = tuple(dict.fromkeys(SNIPER.pool + FUSION.pool))
    df = _with_labels(_panel(extra_cols=pool))
    out = run_all(df, "20260804_000000", oos_days=12)
    for b in BOARD_PREFIXES:
        c = out["boards"][b]["compare"]
        assert "full" in c and "oos" in c
        assert set(c["oos"]) == {"oos"}  # oos_days=N → 单窗
        for comp in [c["full"]] + [c["oos"][w] for w in c["oos"]]:
            assert set(comp["fusion"]) == {"all10", "first5", "last5"}
            assert comp["sniper_top5"]["3d"]["n"] > 0
            assert comp["fusion"]["all10"]["3d"]["n"] > 0


# ── 报告落盘: 每次回测一个日期命名目录 + 选股清单 ──
def test_write_worm_creates_date_subfolder(monkeypatch, tmp_path):
    import app.pipeline_parallel.backtest as bt
    from app.pipeline_parallel.backtest import run_all, write_worm
    from app.pipeline_parallel.config import FUSION, SNIPER

    pool = tuple(dict.fromkeys(SNIPER.pool + FUSION.pool))
    df = _with_labels(_panel(extra_cols=pool))
    out = run_all(df, "20260804_000000", oos_days=4)
    monkeypatch.setattr(bt, "BACKTEST_RESULT_DIR", tmp_path)
    p, log, run_dir = write_worm(out, "20260804_123000")
    assert run_dir == tmp_path / "20260804_123000"
    assert p.exists() and log.exists()
    assert p.name == "backtest.json" and log.name == "backtest.log"


def test_export_stock_lists_csvs(monkeypatch, tmp_path):
    import app.pipeline_parallel.backtest as bt
    from app.pipeline_parallel.backtest import export_stock_lists, run_all, write_worm
    from app.pipeline_parallel.config import FUSION, SNIPER

    pool = tuple(dict.fromkeys(SNIPER.pool + FUSION.pool))
    df = _with_labels(_panel(extra_cols=pool))
    out = run_all(df, "20260804_000000", oos_days=4)
    monkeypatch.setattr(bt, "BACKTEST_RESULT_DIR", tmp_path)
    _, _, run_dir = write_worm(out, "20260804_123000")
    dates = np.sort(df["date"].unique())
    # 传字符串 oos_start (runner 传 out["window"]["oos"]["start"], 真实数据回归点 2026-08-04)
    files = export_stock_lists(df, str(pd.Timestamp(dates[-4])), run_dir)
    for b in ("main", "dual"):
        for fname in (
            f"stocks_{b}_sniper_full.csv",
            f"stocks_{b}_sniper_oos.csv",
            f"stocks_{b}_fusion_full.csv",
            f"stocks_{b}_fusion_oos.csv",
            f"stocks_merged_oos_{b}.csv",
        ):
            assert fname in files, fname
            assert (run_dir / fname).exists(), fname
    stocks = pd.read_csv(run_dir / "stocks_main_sniper_oos.csv")
    assert {"date", "symbol", "score", "rk", "label_mfe_3d_net"}.issubset(
        set(stocks.columns)
    )
    assert len(stocks) > 0
    merged = pd.read_csv(run_dir / "stocks_merged_oos_main.csv")
    assert {"date", "symbol", "systems"}.issubset(set(merged.columns))
    assert len(merged) > 0


# ── 末 10 个交易日逐日报告 ──
def test_last_days_report_per_day_picks_and_figures():
    from app.pipeline_parallel.backtest import last_days_report
    from app.pipeline_parallel.config import FUSION, SNIPER

    pool = tuple(dict.fromkeys(SNIPER.pool + FUSION.pool))
    df = _with_labels(_panel(extra_cols=pool))
    ld = last_days_report(df, n_days=10)
    assert ld["n_days"] == 10
    dates = np.sort(df["date"].unique())
    assert ld["days"][-1]["date"] == str(pd.Timestamp(dates[-1]).date())
    for day in ld["days"]:
        s = day["sniper_top5"]
        f = day["fusion_top10"]
        assert len(s["picks"]) == SNIPER.top_n == 5
        assert len(f["picks"]) == FUSION.top_n == 10
        assert s["picks"][0]["rk"] == 1  # 按分数降序
        assert f["picks"][-1]["rk"] == 10
        for p in s["picks"]:
            assert set(p) == {
                "symbol",
                "rk",
                "score",
                "mfe_3d",
                "mfe_5d",
                "mfe_10d",
            }
        assert set(s["figure"]) == {"3d", "5d", "10d"}
        assert set(f["figure"]) == {"3d", "5d", "10d"}


def test_last_days_report_tail_horizons_nan():
    # 末交易日: T+2 已有未来价 (能测), T+10 无未来价 → n=0 如实标注
    from app.pipeline_parallel.backtest import last_days_report
    from app.pipeline_parallel.config import FUSION, SNIPER

    pool = tuple(dict.fromkeys(SNIPER.pool + FUSION.pool))
    df = _with_labels(_panel(extra_cols=pool))
    ld = last_days_report(df, n_days=10)
    # 末日: T+1 买价无未来价 → 所有视界 n=0 如实标注
    last_day = ld["days"][-1]["sniper_top5"]
    assert all(last_day["figure"][h]["n"] == 0 for h in ("3d", "5d", "10d"))
    assert all(p["mfe_3d"] is None for p in last_day["picks"])
    # 倒数第 5 日: T+3 可测 (窗口 T+3..T+5 有价, 需 4 个未来日), T+10 无未来价 → n=0.
    # 注: 3d MFE 标签需 t+4 未来价 → 末日倒数第 4 日起即不可测, 故用 -5
    d4 = ld["days"][-5]["sniper_top5"]
    assert d4["figure"]["3d"]["n"] == 5
    assert d4["figure"]["3d"]["mag"] is not None
    assert d4["figure"]["5d"]["n"] == 0  # 5d 需 t+6 未来价, 倒数第 5 日仍不够
    assert d4["figure"]["10d"]["n"] == 0
    assert d4["figure"]["10d"]["mag"] is None
    assert all(p["mfe_10d"] is None for p in d4["picks"])


def test_last_days_report_last_testable_dates():
    # 用户需求 (2026-08-04): "LAST TESTABLE DATES, T-5 HV 10 DATES WHILE T-10 HV 5
    # DATES..THAT IS LAST 15 TRADING DATES" — 长视界需更远未来价 → 可测末日期更早、
    # 可测天数更少, 但测试日(选股日)同一. 默认窗口 = 末 15 交易日.
    from app.pipeline_parallel.backtest import last_days_report
    from app.pipeline_parallel.config import FUSION, SNIPER

    pool = tuple(dict.fromkeys(SNIPER.pool + FUSION.pool))
    df = _with_labels(_panel(n_dates=20, extra_cols=pool))
    ld = last_days_report(df)  # 默认 n_days=15
    assert ld["n_days"] == 15
    lt = ld["last_testable"]
    assert set(lt) == {"3d", "5d", "10d"}
    assert all(lt[h]["n"] > 0 and lt[h]["last_date"] is not None for h in lt)
    # T+3 需未来 3 日价, T+10 需未来 10 日价 → 后者的可测末日期更早或相等
    lt_ts = {h: pd.Timestamp(lt[h]["last_date"]) for h in lt}
    assert lt_ts["10d"] <= lt_ts["5d"] <= lt_ts["3d"]
    # 视界越长可测天数越少 (T+10 需 11 日未来价 → 15 日内只有 ~5 日可测)
    ns = {h: lt[h]["n"] for h in lt}
    assert ns["3d"] > ns["5d"] > ns["10d"]
    assert ns["10d"] >= 1


def test_write_last_days_csv(tmp_path):
    from app.pipeline_parallel.backtest import last_days_report, write_last_days_csv
    from app.pipeline_parallel.config import FUSION, SNIPER

    pool = tuple(dict.fromkeys(SNIPER.pool + FUSION.pool))
    df = _with_labels(_panel(extra_cols=pool))
    ld = last_days_report(df, n_days=5)
    fname = write_last_days_csv(ld, tmp_path)
    assert fname == "last_5_days_picks.csv"
    csv = pd.read_csv(tmp_path / fname)
    assert {
        "date",
        "system",
        "rk",
        "symbol",
        "score",
        "mfe_3d",
        "mfe_5d",
        "mfe_10d",
    }.issubset(set(csv.columns))
    assert len(csv) == 5 * 5 + 5 * 10  # 5 天 × (狙击 5 + 融合 10)
    assert set(csv["system"]) == {"sniper", "fusion"}


# ── 合并模块: 最终短名单 (2026-08-04 用户: 一般管道设计, 验收/买入都基于最终短名单) ──
def _pool():
    from app.pipeline_parallel.config import FUSION, SNIPER

    return tuple(dict.fromkeys(SNIPER.pool + FUSION.pool))


def test_build_merged_shortlist_ranked():
    """2026-08-07 定案: 短名单按 mag_10d (score→label_pm_10d_net 校准幅度) 日截面降序,
    不再是共现优先; 共现仅作平局裁决. 排名键 mag 不在输出列 → 独立重算校准幅度验证次序."""
    from app.pipeline_parallel.backtest import build_merged_shortlist
    from app.pipeline_parallel.calibration import calibrate_mag10d
    from app.pipeline_parallel.scoring import pool_score

    df = _with_labels(_panel(extra_cols=_pool()))
    sl = build_merged_shortlist(df, top_n=10)
    assert {"date", "symbol", "systems", "co_occur", "score", "rk"}.issubset(
        set(sl.columns)
    )
    # 去重: 每 (date, symbol) 至多一行
    assert sl.duplicated(["date", "symbol"]).sum() == 0
    # rk 按日期截断到 top_n
    assert sl["rk"].max() <= 10
    # 独立重算 mag_10d (全池 score = max(sniper, fusion) 池分, 同 build_merged_shortlist)
    # → join 回输出, 验证每 (board,date) 组内 rk 按 mag 降序
    scored = df[["symbol", "date", "board", "label_pm_10d_net"]].copy()
    scored["score"] = np.maximum(
        pool_score(df, SNIPER.pool).values, pool_score(df, FUSION.pool).values
    )
    mag = calibrate_mag10d(
        scored.dropna(subset=["score"]),
        score_col="score",
        target_col="label_pm_10d_net",
    )
    if mag.empty:
        pytest.skip("合成面板无足够已实现横截面 (校准不出票)")
    check = sl.merge(mag, on=["symbol", "date", "board"], how="left")
    assert check["mag"].notna().any(), "校准幅度缺失, 无法验证 mag 排序"
    for (_b, _d), g in check.groupby(["board", "date"]):
        assert set(g["rk"]) == set(range(1, len(g) + 1))
        assert g["mag"].is_monotonic_decreasing
    # 共现标注: co_occur=True 的 systems 必含双系统; co_occur=False 的 systems 可为 "" (旧 top-N 之外)
    co = sl.loc[sl["co_occur"]]
    if len(co):
        assert (co["systems"].str.contains("+", regex=False)).all()
    assert set(sl["systems"]).issubset({"", "sniper", "fusion", "fusion+sniper"})


def test_merged_shortlist_systems_blank_outside_old_topn():
    """systems="" 契约: 旧 top-N (狙击 top5 / 融合 top10, 按各自池分) 之外被 mag_10d
    选中的股 → systems="", co_occur=False; build_daily_shortlists 对 systems="" 行
    的 prob/exp 走 NaN 兜底 (defensive map, 不崩溃).

    合成持久因子面板下 mag_10d 次序与特征分次序略有差异 → 旧 top-N 之外的股自然入选,
    systems="" 行天然出现 (不需要人造场景)."""
    from app.pipeline_parallel.backtest import (
        build_daily_shortlists,
        build_merged_shortlist,
        run_all,
    )
    from app.pipeline_parallel.config import BOARD_PREFIXES

    df = _with_labels(_panel(extra_cols=_pool()))
    # co_occur ⟺ systems 含双系统; systems 取值域含 "" (旧 top-N 之外)
    sl = build_merged_shortlist(df, top_n=10)
    if sl.empty:
        pytest.skip("合成面板校准不出票")
    assert (sl["co_occur"] == sl["systems"].str.contains("+", regex=False)).all()
    assert set(sl["systems"]).issubset({"", "sniper", "fusion", "fusion+sniper"})
    assert (sl["systems"] == "").any(), "期望 systems='' 行自然出现"

    # build_daily_shortlists 必须能在 systems="" 存在时不崩溃 (defensive map)
    out = run_all(df, "20260804_000000", oos_days=4)
    date = df["date"].max()
    for board in BOARD_PREFIXES:
        res = build_daily_shortlists(df, out, board, date)
        if res.empty:
            continue
        assert "systems" in res.columns
        blank_rows = res[res["systems"] == ""]
        # systems="" 行: prob/exp 走 NaN 兜底 (p.get("", ...) → (nan,nan)), 不崩溃
        if len(blank_rows):
            assert blank_rows["prob_3d"].isna().all()
            assert blank_rows["exp_3d"].isna().all()


def test_evaluate_merged_reports_per_horizon():
    from app.pipeline_parallel.backtest import evaluate_merged
    from app.pipeline_parallel.config import HORIZONS, MIN_MAG, MIN_WINRATE

    df = _with_labels(_panel(extra_cols=_pool()))
    per = evaluate_merged(df, top_n=10, crit=(MIN_WINRATE, MIN_MAG))
    assert set(per) == set(HORIZONS)
    # TOP-10 短名单在合成持久因子下, 各视界应有足量可测样本
    for _h, r in per.items():
        assert r["n"] > 0
        assert "mag" in r and "winrate" in r and "ok" in r
    # 合成 f1 高分 = 高 MFE → 短名单双头通过
    assert any(r["ok"] for r in per.values())


def test_build_daily_shortlists_single_date_col():
    """2026-08-05 bug: build_merged_shortlist 输出已含 date 列 (groupby key),
    再 insert(0,'date') 报 ValueError cannot insert date → 短名单文件写不出.
    回归: date 列唯一且置首, board 紧随其后, 且每板块短名单只含本板块 symbol."""
    import pandas as pd

    from app.pipeline_parallel.backtest import build_daily_shortlists, run_all
    from app.pipeline_parallel.config import BOARD_PREFIXES, board_of

    df = _with_labels(_panel(extra_cols=_pool()))
    out = run_all(df, "20260804_000000", oos_days=4)
    date = df["date"].max()
    for board in BOARD_PREFIXES:
        res = build_daily_shortlists(df, out, board, date)
        assert list(res.columns[:2]) == ["date", "board"]
        assert (res["date"] == str(pd.Timestamp(date).date())).all()
        assert "date" not in res.columns[2:]
        # 2026-08-05 二连 bug: day 未按 board 过滤 → main 短名单混入 30/68 双创股
        assert res["symbol"].map(board_of).eq(board).all()
        assert (res["board"] == board).all()


def test_daily_shortlist_per_horizon_exp_prob():
    """2026-08-05 用户: 每只股票要有各视界期望涨幅 + 概率(置信度).
    短名单须携带逐视界 exp_{h} (OOS 期望 MFE) 与 prob_{h} (OOS 胜率);
    共现股逐视界取两系统中胜率较高者; 无样本(n<5) 视界 → NaN."""
    from app.pipeline_parallel.backtest import build_daily_shortlists, run_all
    from app.pipeline_parallel.config import BOARD_PREFIXES, HORIZONS

    df = _with_labels(_panel(extra_cols=_pool()))
    # oos_days=12: c2c 标签需 k+1 个未来日 (3d 需 4) → OOS 窗须足够长让短视界可测
    out = run_all(df, "20260804_000000", oos_days=12)
    date = df["date"].max()
    for board in BOARD_PREFIXES:
        res = build_daily_shortlists(df, out, board, date)
        if res.empty:
            continue
        for h in HORIZONS:
            assert f"exp_{h}" in res.columns, f"缺 exp_{h}"
            assert f"prob_{h}" in res.columns, f"缺 prob_{h}"
        # 短视界应有可测样本 (OOS 窗 ≥ 最长视界+1 交易日); 长视界可能因缺未来价全 NaN, 允许
        assert res["prob_3d"].notna().sum() > 0
        assert res["exp_3d"].notna().sum() > 0
        # 概率在 (0,1] 区间 (合成面板无 0 胜率)
        assert res["prob_3d"].dropna().between(0, 1).all()


def test_build_conclusion_has_verdicts():
    from app.pipeline_parallel.backtest import build_conclusion, run_all
    from app.pipeline_parallel.config import BOARD_PREFIXES

    df = _with_labels(_panel(extra_cols=_pool()))
    out = run_all(df, "20260804_000000", oos_days=4)
    concl = build_conclusion(out)
    assert concl["oos_label"] == "oos"
    assert set(concl["boards"]) == set(BOARD_PREFIXES)
    for b in BOARD_PREFIXES:
        cuts = concl["boards"][b]["cuts"]
        assert set(cuts) == {"top5", "top10"}
        for cut in ("top5", "top10"):
            c = cuts[cut]
            assert "kept" in c and "best_horizon" in c
            if c["best_horizon"]:
                assert c["winrate"] is not None and c["n"] > 0
        assert isinstance(concl["boards"][b]["improvements"], list)
        assert (
            len(concl["boards"][b]["improvements"]) >= 1
        )  # 合成面板尾部 T+5/10 NaN → 有改进点
        assert "text" in concl["recommendation"]
        assert "sizing" in concl["recommendation"]


def test_write_worm_writes_conclusion(monkeypatch, tmp_path):
    import app.pipeline_parallel.backtest as bt
    from app.pipeline_parallel.backtest import run_all, write_worm

    df = _with_labels(_panel(extra_cols=_pool()))
    out = run_all(df, "20260804_000000", oos_days=4)
    monkeypatch.setattr(bt, "BACKTEST_RESULT_DIR", tmp_path)
    p, log, run_dir = write_worm(out, "20260804_123000")
    assert "conclusion" in out
    assert (run_dir / "conclusion.txt").exists()
    log_text = log.read_text(encoding="utf-8")
    assert "结论" in log_text  # 结论置顶
    assert log_text.index("结论") < log_text.index("PIPELINE")
