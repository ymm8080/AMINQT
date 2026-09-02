"""Tests for app/pipeline1/freshness_guard — 特征族新鲜度守卫 (2026-09-02).

背景: 4 起"特征静默停更"全是事后数周才发现 (cyq@07-17 / sw_daily_history@07-31 /
announce_date@08-14 / fina 列冻结 + sw_ret_1d@08-27 整列 NaN). 全部判定函数纯注入
(伪 IO / 构造 cal), 不碰真数据; 唯 panel_column_daily_nonnull 用 tmp_path 造迷你
parquet 验证计数逻辑.
"""

import datetime

import pandas as pd
import pytest

from app.pipeline1.freshness_guard import (
    FreshnessIO,
    check_columns_entry,
    check_file_entry,
    check_watermark_entry,
    expected_trading_date,
    lag_trading_days,
    load_registry,
    panel_column_daily_nonnull,
    panel_stale_gate,
    run_checks,
)

# cal: 2026-08-24(Mon)..09-04(Fri) 工作日 (2026-09-02 真实为周三, 与生产一致形态)
CAL = pd.bdate_range("2026-08-24", "2026-09-04")
MON, FRI = CAL[0], pd.Timestamp("2026-08-28")  # 首周一 / 该周周五
WED = pd.Timestamp("2026-08-26")
NEXT_MON = pd.Timestamp("2026-08-31")

_FILE_ENTRY = {
    "name": "t_file",
    "kind": "file",
    "path": "x.parquet",
    "date_col": "date",
    "max_lag_days": 1,
}
_COLS_ENTRY = {
    "name": "t_cols",
    "kind": "panel_columns",
    "path": "panel.parquet",
    "columns": ["a", "b"],
    "lookback_days": 3,
    "min_nonnull": 1000,
}
_WM_ENTRY = {
    "name": "t_wm",
    "kind": "dir_watermark",
    "path": "cache_dir",
    "pattern": "_(\\d{8})\\.parquet$",
    "max_lag_days": 5,
}


# ── load_registry: 注册表校验 ───────────────────────────────────────────────


def _write_yaml(tmp_path, text):
    p = tmp_path / "registry.yaml"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_load_registry_valid(tmp_path):
    p = _write_yaml(
        tmp_path,
        "- name: a\n  kind: file\n  path: x.parquet\n  date_col: date\n"
        "  max_lag_days: 1\n",
    )
    reg = load_registry(p)
    assert len(reg) == 1 and reg[0]["name"] == "a"


def test_load_registry_entries_wrapper(tmp_path):
    """顶层 {entries: [...]} 形式 (yaml 头部放说明注释块) 同样可加载."""
    p = _write_yaml(
        tmp_path,
        "# 头注释\nentries:\n  - name: a\n    kind: file\n    path: x.parquet\n"
        "    date_col: date\n    max_lag_days: 1\n",
    )
    assert load_registry(p)[0]["name"] == "a"


def test_load_registry_missing_key_raises_with_name(tmp_path):
    p = _write_yaml(
        tmp_path,
        "- name: bad_entry\n  kind: file\n  path: x.parquet\n",  # 缺 date_col/max_lag_days
    )
    with pytest.raises(ValueError, match="bad_entry"):
        load_registry(p)


def test_load_registry_unknown_kind_raises_with_name(tmp_path):
    p = _write_yaml(tmp_path, "- name: odd\n  kind: nope\n  path: x\n")
    with pytest.raises(ValueError, match="odd"):
        load_registry(p)


def test_load_registry_empty_columns_raises(tmp_path):
    p = _write_yaml(
        tmp_path,
        "- name: empty_cols\n  kind: panel_columns\n  path: p.parquet\n"
        "  columns: []\n  lookback_days: 3\n  min_nonnull: 10\n",
    )
    with pytest.raises(ValueError, match="empty_cols"):
        load_registry(p)


# ── expected_trading_date: 有 cal / 无 cal 两路 ─────────────────────────────


def test_expected_trading_date_with_cal():
    d, src = expected_trading_date(pd.Timestamp("2026-09-02"), CAL)
    assert (d, src) == (datetime.date(2026, 9, 2), "trade_cal")


def test_expected_trading_date_with_cal_weekend_rolls_back():
    """周六问 → 期望日=周五 (cal 内 <=today 的最大日)."""
    d, src = expected_trading_date(pd.Timestamp("2026-08-29"), CAL)  # 周六
    assert (d, src) == (datetime.date(2026, 8, 28), "trade_cal")


def test_expected_trading_date_without_cal_falls_back_to_weekday():
    """cal None → 最近工作日 (周一~五) 回退; 周日回退到周五."""
    d, src = expected_trading_date(pd.Timestamp("2026-09-02"), None)
    assert (d, src) == (datetime.date(2026, 9, 2), "natural_fallback")
    d, src = expected_trading_date(pd.Timestamp("2026-08-30"), None)  # 周日
    assert (d, src) == (datetime.date(2026, 8, 28), "natural_fallback")


def test_expected_trading_date_empty_cal_falls_back():
    d, src = expected_trading_date(pd.Timestamp("2026-09-02"), pd.DatetimeIndex([]))
    assert src == "natural_fallback"


# ── lag_trading_days: 边界 ──────────────────────────────────────────────────


def test_lag_trading_days_same_day_zero():
    assert lag_trading_days(WED, WED, CAL) == 0


def test_lag_trading_days_one_trading_day():
    """周五 → 下周一 = 交易日 1 (跨周末不累计)."""
    assert lag_trading_days(FRI, NEXT_MON, CAL) == 1


def test_lag_trading_days_outside_cal_returns_none():
    assert lag_trading_days(pd.Timestamp("2026-08-20"), WED, CAL) is None
    assert lag_trading_days(WED, WED, None) is None


# ── check_file_entry: 违规 / 健康 / 读失败 三路 ─────────────────────────────


def test_check_file_entry_violation_when_lag_exceeds():
    """面板停周五, 下周四才跑 = 交易日 lag 4 > 1 → 违规."""
    v = check_file_entry(_FILE_ENTRY, FRI, pd.Timestamp("2026-09-03"), CAL)
    assert v is not None
    assert v["lag"] == 4 and v["threshold"] == 1
    assert v["critical"] is False and v["observed"] == "2026-08-28"


def test_check_file_entry_healthy_at_threshold():
    v = check_file_entry(_FILE_ENTRY, WED, WED, CAL)
    assert v is None
    # 恰好 lag=1 (周一面板停上周五, 周一晚跑链) → 健康
    assert check_file_entry(_FILE_ENTRY, FRI, NEXT_MON, CAL) is None


def test_check_file_entry_read_failed_never_passes():
    """读失败 (observed=None) 也判违规 — 绝不静默放行, 旧 except-pass 闸的漏洞."""
    v = check_file_entry(_FILE_ENTRY, None, WED, CAL)
    assert v is not None and v["observed"] is None
    assert "read_failed" in v["detail"]


def test_check_file_entry_critical_flag_propagates():
    entry = {**_FILE_ENTRY, "critical": True}
    v = check_file_entry(entry, None, WED, CAL)
    assert v["critical"] is True


def test_check_file_entry_natural_fallback_weekend_buffer():
    """cal 不可用回退自然日: 周一面板停周五=自然日 3, 阈值 1+2 周末缓冲 → 不误报;
    周二=自然日 4 > 3 → 违规."""
    assert check_file_entry(_FILE_ENTRY, FRI, NEXT_MON, None) is None
    assert (
        check_file_entry(_FILE_ENTRY, FRI, pd.Timestamp("2026-09-01"), None) is not None
    )


# ── check_columns_entry: 窗口内达标 / 全不达标 / 空列表 ─────────────────────


def test_check_columns_entry_any_day_in_window_passes():
    """回看窗口内任一日达标即健康 — 事件型列隔日有数也算活."""
    counts = [
        (pd.Timestamp("2026-08-28").date(), 5),  # 窗口外旧日, 低
        (pd.Timestamp("2026-08-31").date(), 20),  # 窗口内, 低
        (pd.Timestamp("2026-09-01").date(), 1200),  # 窗口内, 达标
        (pd.Timestamp("2026-09-02").date(), 0),  # 最新日全 NaN (事故形态)
    ]
    assert (
        check_columns_entry(_COLS_ENTRY, counts, pd.Timestamp("2026-09-02"), CAL)
        is None
    )


def test_check_columns_entry_all_below_threshold_violates():
    """最新日全 NaN + 窗口内其余日都低于阈值 → 违规 (08-27 sw_ret_1d 整列 NaN 形态)."""
    counts = [
        (pd.Timestamp("2026-08-31").date(), 999),
        (pd.Timestamp("2026-09-01").date(), 500),
        (pd.Timestamp("2026-09-02").date(), 0),
    ]
    v = check_columns_entry(_COLS_ENTRY, counts, pd.Timestamp("2026-09-02"), CAL)
    assert v is not None and v["threshold"] == 1000


def test_check_columns_entry_empty_list_violates():
    """面板在窗口内无行/读失败 → 空列表也判违规, 不静默放行."""
    v = check_columns_entry(_COLS_ENTRY, [], pd.Timestamp("2026-09-02"), CAL)
    assert v is not None


def test_check_columns_entry_ignores_days_outside_window():
    """窗口达标但最新 3 个交易日全 NaN (面板整块冻结在窗口前) → 仍违规."""
    counts = [(pd.Timestamp("2026-08-20").date(), 5000)]  # 窗口外达标日不得救
    v = check_columns_entry(_COLS_ENTRY, counts, pd.Timestamp("2026-09-02"), CAL)
    assert v is not None


# ── check_watermark_entry ───────────────────────────────────────────────────


def test_check_watermark_entry_three_ways():
    assert check_watermark_entry(_WM_ENTRY, WED.date(), WED, CAL) is None
    v = check_watermark_entry(
        _WM_ENTRY, pd.Timestamp("2026-08-10").date(), WED, CAL
    )  # lag 远超 5
    assert v is not None and v["kind"] == "dir_watermark"
    v2 = check_watermark_entry(_WM_ENTRY, None, WED, CAL)  # 目录缺失/无匹配
    assert v2 is not None and "read_failed" in v2["detail"]


# ── panel_stale_gate: 周一 vs 周五 回归 ─────────────────────────────────────


def test_panel_stale_gate_monday_fresh_friday_data_passes_with_cal():
    """回归: 周一跑链面板停周五 = 交易日 lag 1 → 放行 (旧自然日 >3 口径会误拦)."""
    cal_with_mon = pd.bdate_range("2026-08-24", "2026-08-31")
    allow, _ = panel_stale_gate(FRI.date(), pd.Timestamp("2026-08-31"), cal_with_mon)
    assert allow is True


def test_panel_stale_gate_monday_fresh_friday_data_passes_natural():
    """cal 不可用时自然日回退: 自然日 3 <= 4 → 同样放行, 两口径边界一致."""
    allow, _ = panel_stale_gate(FRI.date(), pd.Timestamp("2026-08-31"), None)
    assert allow is True


def test_panel_stale_gate_week_old_data_blocks():
    """停更一周: 交易日 lag 5 > 3 / 自然日 7 > 4, 两口径都拦."""
    cal_with_fri = pd.bdate_range("2026-08-24", "2026-09-04")
    allow, reason = panel_stale_gate(
        FRI.date(), pd.Timestamp("2026-09-04"), cal_with_fri
    )
    assert allow is False and "交易日落后" in reason
    allow2, reason2 = panel_stale_gate(FRI.date(), pd.Timestamp("2026-09-04"), None)
    assert allow2 is False and "自然日落后" in reason2


def test_panel_stale_gate_unreadable_never_passes():
    """读失败 (pmax=None) 不放行 — 读失败≠数据新鲜, 旧 except-pass 漏洞."""
    allow, reason = panel_stale_gate(None, pd.Timestamp("2026-09-02"), CAL)
    assert allow is False and "不可读" in reason


def test_panel_stale_gate_threshold_constants():
    """阈值常量锁死 (3 交易日 / 4 自然日), 防止有人调松而不留痕."""
    from app.pipeline1 import freshness_guard as fg

    assert fg._PANEL_MAX_LAG_TRADING == 3
    assert fg._PANEL_MAX_LAG_NATURAL == 4


# ── run_checks: 编排 + enabled=false 跳过 ──────────────────────────────────


class _FakeIO:
    """伪 IO: 按路径返回预置观测值, 不碰真数据.

    run_checks 传给 IO 的是解析到仓库根的绝对路径, 这里按尾部子串匹配注册表里的
    相对路径键 (真实 IO 契约: 收到的就是解析后路径).
    """

    def __init__(self, file_dates=None, watermarks=None, column_counts=None):
        self.file_dates = file_dates or {}
        self.watermarks = watermarks or {}
        self.column_counts = column_counts or {}

    @staticmethod
    def _lookup(table, path):
        path = str(path)
        for key, val in table.items():
            if path.endswith(key):
                return val
        return None

    def file_max_date(self, path, date_col):
        return self._lookup(self.file_dates, path)

    def dir_watermark(self, path, pattern):
        return self._lookup(self.watermarks, path)

    def panel_column_daily_nonnull(
        self, path, columns, cal, tail_days, date_col="date"
    ):
        return self._lookup(self.column_counts, path) or []


def test_run_checks_skips_disabled_and_collects_violations():
    registry = [
        {**_FILE_ENTRY, "name": "dead", "enabled": False},
        {**_FILE_ENTRY, "name": "stale_one", "path": "stale.parquet"},
        {**_FILE_ENTRY, "name": "fresh_one", "path": "fresh.parquet"},
    ]
    io = _FakeIO(
        file_dates={
            "stale.parquet": pd.Timestamp("2026-08-10").date(),  # lag 17 → 违规
            "fresh.parquet": WED.date(),  # lag 0 → 健康
        }
    )
    result = run_checks(registry, WED, CAL, io_impl=io)
    assert [s["name"] for s in result.skipped] == ["dead"]
    assert [v["name"] for v in result.violations] == ["stale_one"]
    assert [o["name"] for o in result.observations] == ["fresh_one"]
    assert result.has_critical_violation is False


def test_run_checks_critical_violation_detected():
    registry = [{**_FILE_ENTRY, "name": "crit", "critical": True}]
    io = _FakeIO(file_dates={})
    result = run_checks(registry, WED, CAL, io_impl=io)
    assert result.has_critical_violation is True


def test_run_checks_watermark_and_columns_kinds():
    registry = [_WM_ENTRY, _COLS_ENTRY]
    io = _FakeIO(
        watermarks={"cache_dir": pd.Timestamp("2026-08-25").date()},  # lag 1 ≤ 5
        column_counts={
            "panel.parquet": [
                (pd.Timestamp("2026-08-24").date(), 1500),  # 窗口内 (08-24..08-26)
                (pd.Timestamp("2026-08-25").date(), 1800),
                (pd.Timestamp("2026-08-26").date(), 2000),
                (pd.Timestamp("2026-08-27").date(), 9999),  # 窗口外 (>expected) 不计
            ]
        },
    )
    result = run_checks(registry, WED, CAL, io_impl=io)
    assert result.violations == []
    assert len(result.observations) == 2
    cols_obs = next(o for o in result.observations if o["name"] == "t_cols")
    assert cols_obs["observed"] == 2000


# ── panel_column_daily_nonnull: tmp_path 迷你 parquet ──────────────────────


def test_panel_column_daily_nonnull_counts_min_across_columns(tmp_path):
    """逐日非空计数取族内各列最小值: b 列缺数的日子按 b 计 (族健康须各列都有数)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-09-01", "2026-09-01", "2026-09-01", "2026-09-02", "2026-09-02"]
            ),
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],  # 09-02 有 2 个非空
            "b": [1.0, None, 3.0, None, None],  # 09-01 仅 2 个非空, 09-02 全空
        }
    )
    p = tmp_path / "mini.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(p))

    cal = pd.bdate_range("2026-09-01", "2026-09-02")
    out = panel_column_daily_nonnull(str(p), ["a", "b"], cal, 7)
    assert out == [
        (datetime.date(2026, 9, 1), 2),
        (datetime.date(2026, 9, 2), 0),
    ]
    # 尾部窗口收窄: tail_days 只含 09-02 时 09-01 不出现
    out2 = panel_column_daily_nonnull(
        str(p), ["a", "b"], pd.bdate_range("2026-09-02", "2026-09-02"), 1
    )
    assert out2 == [(datetime.date(2026, 9, 2), 0)]


def test_panel_column_daily_nonnull_missing_file_returns_empty(tmp_path):
    out = panel_column_daily_nonnull(str(tmp_path / "nope.parquet"), ["a"], None, 7)
    assert out == []


# ── dir_watermark: tmp_path 造目录 ─────────────────────────────────────────


def test_dir_watermark_takes_max_date(tmp_path):
    (tmp_path / "all__20260816_20260817.parquet").write_text("x")
    (tmp_path / "all__20260817_20260820.parquet").write_text("x")
    (tmp_path / "new_symbols_20260816_001140.parquet").write_text("x")  # 不匹配尾部日期
    from app.pipeline1.freshness_guard import dir_watermark

    assert dir_watermark(str(tmp_path), r"_(\d{8})\.parquet$") == datetime.date(
        2026, 8, 20
    )


def test_dir_watermark_missing_dir_returns_none(tmp_path):
    from app.pipeline1.freshness_guard import dir_watermark

    assert dir_watermark(str(tmp_path / "nope"), r"_(\d{8})\.parquet$") is None


# ── IO 接口束: FreshnessIO 三方法齐备 (防接口漂移) ─────────────────────────


def test_freshness_io_interface_shape():
    io = FreshnessIO(
        file_max_date=lambda p, c: None,
        panel_column_daily_nonnull=lambda p, c, cal, t, date_col="date": [],
        dir_watermark=lambda p, pat: None,
    )
    result = run_checks([_FILE_ENTRY], WED, CAL, io_impl=io)
    # 全部读失败 → 全部违规, 绝不静默放行
    assert len(result.violations) == 1 and result.observations == []
