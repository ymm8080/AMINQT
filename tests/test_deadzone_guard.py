"""死区停推闸 (scripts/_deadzone_guard.py) 单测: V4 状态机 + 前视防护 + fail-open
+ 交付史装载 + 双线接线。参数锁值断言跟 _deadzone_guard 模块头走。
"""
import inspect

import numpy as np
import pandas as pd

from scripts import _deadzone_guard as dz


# ---------------------------------------------------------------- 参数锁值
def test_constants_locked_v4():
    assert dz.DZ_ENABLED is True
    assert dz.DZ_ENTER == 0.25
    assert dz.DZ_EXIT == 0.40
    assert dz.DZ_EXIT_DAYS == 2
    assert dz.DZ_WINDOW == 10
    assert dz.DZ_MIN_SAMPLES == 5
    assert dz.DZ_SETTLE == 4
    assert dz.DZ_COST == 0.0020
    assert dz.DZ_WIN == 0.05


# ---------------------------------------------------------------- 状态机
def _machine(win_by_day):
    di = np.arange(len(win_by_day))
    win = np.asarray(win_by_day, dtype=bool)
    return dz.alarm_indices(di, win, len(win_by_day))


def test_enter_hold_release_full_cycle():
    # 0-19 全赢; 20-40 全输; 41 起全赢 → 29 报警, 连续 2 采样日 ≥40% 于 48 解除
    alarm = _machine([True] * 20 + [False] * 21 + [True] * 19)
    assert min(alarm) == 29
    assert max(alarm) == 46
    assert len(alarm) == 18
    assert 47 not in alarm  # 第 1 个 ≥40% 采样日 (解除进度 1/2, 当日按原语义推)
    assert 48 not in alarm  # 第 2 个 → 解除


def test_reenter_after_relapse():
    win_by_day = [True] * 20 + [False] * 21 + [True] * 19 + [False] * 11
    alarm = _machine(win_by_day)
    assert 69 in alarm  # 50-60 转输后赢率再破 25% → 重进报警
    assert 68 not in alarm  # 28.6% 死区内不重进 (滞回)


def test_single_good_day_does_not_release():
    # 仅 41/42 两天赢 (2/7=28.6% < 出口 40%) → 报警不解除
    alarm = _machine([True] * 20 + [False] * 21 + [True] * 2 + [False] * 7)
    assert min(alarm) == 29
    assert 47 in alarm
    assert 49 in alarm


def test_exit_streak_resets_on_sampleless_day(monkeypatch):
    # 采样断日重置解除计数: day6 无样本 → day5 的 good=1 作废, day8 (30% 死区)
    # 仍属报警; 若不重置, day7 即解除且 30% 死区不重进 → 集合差异可判
    wr = {4: 0.10, 5: 0.50, 7: 0.50, 8: 0.30}
    monkeypatch.setattr(dz, "rolling_win_rates",
                        lambda di, win, n: wr)
    out = dz.alarm_indices(np.array([4]), np.array([True]), 9)
    assert out == {4, 8}


def test_no_lookahead_unsettled_disaster_invisible():
    # 20-24 天全是灾难票, 但 24 天夜里它们的结局未结算 (di+4 > 24) → 不可见
    alarm = _machine([True] * 20 + [False] * 5)
    assert alarm == set()


def test_insufficient_samples_never_alarms():
    # 每 3 天 1 票全输: 窗口内完结票 <5 → 永不报警 (fail-open 语义)
    di = np.arange(0, 30, 3)
    win = np.zeros(len(di), dtype=bool)
    assert dz.alarm_indices(di, win, 30) == set()


# ---------------------------------------------------------------- 结局计算
def _panel_fp(tmp_path, close: pd.DataFrame) -> str:
    fp = tmp_path / "panel.parquet"
    close.to_parquet(fp, index=False)
    return str(fp)


def test_settled_outcomes_grid_and_lookahead(tmp_path, monkeypatch):
    dates = pd.bdate_range("2026-01-05", periods=30)
    close = pd.DataFrame({
        "symbol": "600000.SH",  # 带后缀 → 裸 6 位归一
        "date": dates,
        "close_hfq": [10.0 * (1.02 ** k) for k in range(30)],
    })
    monkeypatch.setattr("config.settings.PANEL_V3_PATH",
                        _panel_fp(tmp_path, close))
    picks = pd.DataFrame({"date": dates[::2], "symbol": "600000"})
    out, grid = dz._settled_outcomes(picks, dates[-1])
    assert grid == list(dates[::2])
    assert list(out.columns) == ["date", "symbol", "di", "win"]
    # i≥1 且 i+4 ≤ 29: 面板序 0 (i=0) 与 28/29 之后无 → 留 2..24 偶序 12 行
    assert len(out) == 12
    assert out["win"].all()  # 1.02^4 − 0.2% ≈ +8.2% ≥ 5%
    assert out["di"].tolist() == list(range(1, 13))
    # 前视: 末段未完结票 (i=28, i+4 > 29) 只进格点不进结局
    assert len(grid) == 15
    assert (out["date"] < dates[28]).all()


def test_settled_outcomes_loss_and_nan_drop(tmp_path, monkeypatch):
    dates = pd.bdate_range("2026-01-05", periods=20)
    close = pd.DataFrame({
        "symbol": ["600000"] * 18 + [None, None],  # 末两日缺价
        "date": dates,
        "close_hfq": [10.0] * 18 + [None, None],
    })
    close["symbol"] = close["symbol"].fillna("600000")
    monkeypatch.setattr("config.settings.PANEL_V3_PATH",
                        _panel_fp(tmp_path, close))
    picks = pd.DataFrame({"date": dates, "symbol": "600000"})
    out, grid = dz._settled_outcomes(picks, dates[-1])
    assert len(grid) == 20
    assert (~out["win"]).all()  # 平价 → net4 = −0.2% < 5%
    # i≥1 且 i+4 ≤ 19 → i ∈ 1..15; i=15 卖价 (19 号) 缺 → NaN 剔 → 13 行
    assert len(out) == 13


# ---------------------------------------------------------------- 装载器
def test_load_density_history(tmp_path):
    fp = tmp_path / "prob10dens_20260105__prob10dens.csv"
    pd.DataFrame({"symbol": ["000001", "600000", "ABC123", None]}).to_csv(
        fp, index=False, encoding="utf-8-sig")
    (tmp_path / "prob10dens_bad.csv").write_text("x", encoding="utf-8")
    h = dz.load_density_history(list_dir=tmp_path)
    assert sorted(map(tuple, h.values)) == [
        ("20260105", "000001"), ("20260105", "600000")]


def test_load_top10_history_legacy_glob_and_board_files(tmp_path):
    legacy = tmp_path / "legacy_stocklist_20260105__M1.csv"
    pd.DataFrame({"symbol": [f"60000{k}" for k in range(8)]}).to_csv(
        legacy, index=False)
    # 板级旧命名 (legacy_stocklist_main_日期) 不该被日期正则收进
    board = tmp_path / "legacy_stocklist_main_20260101__X.csv"
    pd.DataFrame({"symbol": ["300001"]}).to_csv(board, index=False)
    h = dz.load_top10_history(list_dir=tmp_path)
    assert set(h["date"]) == {"20260105"}
    assert len(h) == 8
    assert h["symbol"].str.fullmatch(r"\d{6}").all()


# ---------------------------------------------------------------- is_alarm
def _loader_picks(dates):
    return pd.DataFrame({"date": dates, "symbol": "600000"})


def _flat_or_rising_panel(tmp_path, growth):
    dates = pd.bdate_range("2026-01-05", periods=30)
    close = pd.DataFrame({
        "symbol": "600000", "date": dates,
        "close_hfq": [10.0 * (growth ** k) for k in range(30)],
    })
    return dates, _panel_fp(tmp_path, close)


def test_is_alarm_true_in_dead_stretch(tmp_path, monkeypatch):
    dates, fp = _flat_or_rising_panel(tmp_path, 1.0)  # 平价 → 全输
    monkeypatch.setattr("config.settings.PANEL_V3_PATH", fp)
    monkeypatch.setitem(dz._LOADERS, "prob10dens",
                        lambda: _loader_picks(dates))
    ok, why = dz.is_alarm("prob10dens", dates[-1].strftime("%Y%m%d"))
    assert ok is True
    assert "报警线" in why


def test_is_alarm_false_in_healthy_stretch(tmp_path, monkeypatch):
    dates, fp = _flat_or_rising_panel(tmp_path, 1.02)  # 全赢
    monkeypatch.setattr("config.settings.PANEL_V3_PATH", fp)
    monkeypatch.setitem(dz._LOADERS, "prob10dens",
                        lambda: _loader_picks(dates))
    ok, why = dz.is_alarm("prob10dens", dates[-1].strftime("%Y%m%d"))
    assert ok is False
    assert "未达报警线" in why


def test_is_alarm_failopen_paths(tmp_path, monkeypatch):
    dates, fp = _flat_or_rising_panel(tmp_path, 1.0)
    monkeypatch.setattr("config.settings.PANEL_V3_PATH", fp)
    monkeypatch.setitem(dz._LOADERS, "prob10dens",
                        lambda: pd.DataFrame(columns=["date", "symbol"]))
    ok, why = dz.is_alarm("prob10dens", dates[-1].strftime("%Y%m%d"))
    assert ok is False and "完结样本不足" in why

    ok, why = dz.is_alarm("unknown_line", dates[-1].strftime("%Y%m%d"))
    assert ok is False and "fail-open" in why

    ok, why = dz.is_alarm("prob10dens", "19990101")  # 不在格点
    assert ok is False and "fail-open" in why


def test_is_alarm_disabled(monkeypatch):
    monkeypatch.setattr(dz, "DZ_ENABLED", False)
    ok, why = dz.is_alarm("top10", "20260904")
    assert ok is False and "关闭" in why


# ---------------------------------------------------------------- 停推标注
def test_annotate_stop_marker_and_md_banner(tmp_path):
    md = tmp_path / "legacy_stocklist_20260105__M1.md"
    md.write_text("# 清单 20260105\n600000\n", encoding="utf-8")
    mk = dz.annotate_stop("top10", "20260105", "滚动赢率 18.9% < 25%",
                          list_dir=tmp_path)
    assert mk.name == "STOPPED_DEADZONE_20260105__top10.txt"
    assert "18.9%" in mk.read_text(encoding="utf-8")
    assert "非推送故障" in mk.read_text(encoding="utf-8")
    text = md.read_text(encoding="utf-8")
    assert text.count("死区停推") == 1
    assert "18.9%" in text
    # 重跑幂等: 标记不重建不报错, 横幅不重复追加
    dz.annotate_stop("top10", "20260105", "滚动赢率 18.9% < 25%",
                     list_dir=tmp_path)
    assert md.read_text(encoding="utf-8").count("死区停推") == 1


def test_annotate_stop_density_line_no_md(tmp_path):
    mk = dz.annotate_stop("prob10dens", "20260105", "滚动赢率 20.0% < 25%",
                          list_dir=tmp_path)
    assert mk.exists()
    # 密度线无 legacy md 可挂横幅 → 不产生任何 md
    assert not list(tmp_path.glob("*.md"))


def test_write_push_result_note_overrides_all_rows(tmp_path):
    from scripts._ths_watchlist_push import write_push_result

    fp = write_push_result(tmp_path / "ths_watchlist_20260105__deadzone.txt",
                           ["600000", "000001"], ["600000"], note="deadzone")
    assert fp.name == "ths_push_result_20260105__deadzone.csv"
    import pandas as pd

    df = pd.read_csv(fp, dtype=str)
    assert df["status"].tolist() == ["deadzone", "deadzone"]


# ---------------------------------------------------------------- 接线
def test_wiring_call_sites():
    from scripts import _prob10_density_shadow as dens
    from scripts import _ths_watchlist_push as push

    src_push = inspect.getsource(push.main)
    assert '_deadzone_guard.is_alarm("top10"' in src_push
    assert "今晚停推" in src_push
    # 停推夜标注三件套 (09-05 用户: 分不清 "闸停推" 和 "没推成功")
    assert 'annotate_stop("top10"' in src_push
    assert 'note="deadzone"' in src_push

    src_dens = inspect.getsource(dens.main)
    assert '_deadzone_guard.is_alarm("prob10dens"' in src_dens
    assert 'annotate_stop("prob10dens"' in src_dens
    assert 'note="deadzone"' in src_dens
    # 闸在 txt 写出之前 (报警夜不落 ths_watchlist txt, flush guard 不会误动)
    assert (src_dens.index('is_alarm("prob10dens"')
            < src_dens.index("ths_watchlist_{date}__{MODULE}.txt"))
