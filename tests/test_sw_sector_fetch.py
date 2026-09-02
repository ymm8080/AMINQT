"""申万行业指数拉取 fail-fast 回归测试 (2026-09-02 事故).

背景: _daily_fetch sw 段旧逻辑拉取失败只 print FAILED + continue, 全空只打一行
WARN — 2026-08-27 与 09-01 两天全市场 sw_ret_1d/sw_index_close/sw_index_vol
整列 NaN/0, 无任何拦截. 纯注入测试: fetch_one 用伪拉取函数, 不触网.
"""

import pandas as pd

from app.pipeline1.sw_sector_fetch import (
    fetch_sw_sector_map,
    missing_industries,
    sw_daily_adapter,
)

IND2CODE = {"石油石化": "801010", "电气设备": "801730"}


def _ok_frame(pct, close, vol):
    return pd.DataFrame({"pct_chg": [pct], "close": [close], "vol": [vol]})


def test_parse_units_and_calls_every_industry():
    """成功路径: 每个在面板出现的行业都被拉取一次, 单位换算与面板口径一致
    (ret=pct_chg/100 小数, close 原值, vol 手→百万手)."""

    def fetch_one(code, s, e):
        assert code.endswith(".SI") and s == e == "20260902"
        return _ok_frame(1.23, 4567.89, 2_345_678.0)

    m = fetch_sw_sector_map(fetch_one, IND2CODE, set(IND2CODE), "20260902", delay=0)
    assert set(m) == set(IND2CODE)
    assert m["石油石化"] == {
        "sw_ret_1d": 0.0123,
        "sw_index_close": 4567.89,
        "sw_index_vol": 2.35,
    }


def test_retry_until_success():
    """瞬时失败 (限流/超时) 必须重试到成功, 而不是像旧逻辑一样 continue 丢弃."""
    state = {"n": 0}

    def flaky(code, s, e):
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("rate limited")
        return _ok_frame(0.5, 100.0, 1_000_000.0)

    m = fetch_sw_sector_map(
        flaky, {"A": "801010"}, {"A"}, "20260902", attempts=3, delay=0
    )
    assert state["n"] == 3
    assert "A" in m


def test_all_attempts_exhausted_code_absent_and_missing_detected():
    """3 次尝试全失败 → 该行业不在 map 里, missing_industries 精确报出缺失行业
    (这是 _daily_fetch [FATAL] sys.exit(1) fail-fast 的判定依据)."""

    def dead(code, s, e):
        raise RuntimeError("tushare down")

    m = fetch_sw_sector_map(dead, IND2CODE, set(IND2CODE), "20260902", delay=0)
    assert m == {}
    assert missing_industries(set(IND2CODE), m) == sorted(IND2CODE)


def test_empty_result_counts_as_failure():
    """拉到空 DataFrame (非异常) 同样算失败, 不得写入半空值."""

    def empty(code, s, e):
        return pd.DataFrame()

    m = fetch_sw_sector_map(
        empty, {"A": "801010"}, {"A"}, "20260902", attempts=2, delay=0
    )
    assert m == {}


def test_partial_failure_reports_only_missing():
    """部分行业成功部分失败 → missing 只含失败项 (成功项仍可用)."""

    def half(code, s, e):
        if code.startswith("801010"):
            raise RuntimeError("boom")
        return _ok_frame(0.0, 1.0, 0.0)

    m = fetch_sw_sector_map(half, IND2CODE, set(IND2CODE), "20260902", delay=0)
    assert set(m) == {"电气设备"}
    assert missing_industries(set(IND2CODE), m) == ["石油石化"]


def test_missing_industries_empty_when_complete():
    assert missing_industries({"电气设备"}, {"电气设备": {}}) == []


class _FakePro:
    """伪 Tushare 客户端: sw_daily 返回 sw_daily 真实 schema (pct_change 列名)."""

    def __init__(self, frame):
        self._frame = frame
        self.calls = []

    def sw_daily(self, ts_code, start_date, end_date):
        self.calls.append((ts_code, start_date, end_date))
        return self._frame


def test_sw_daily_adapter_renames_pct_change():
    """sw_daily 路线接线 (2026-09-02): index_daily 对申万指数 08-27 起死路线;
    adapter 必须把 sw_daily 的 pct_change 重命名为 pct_chg 保持下游口径不变,
    透传 ts_code/start/end, 空结果原样返回 (供重试逻辑按失败处理)."""
    frame = pd.DataFrame({"pct_change": [-2.01], "close": [2605.47], "vol": [467920.0]})
    fake = _FakePro(frame)
    fetch_one = sw_daily_adapter(fake)
    got = fetch_one("801010.SI", "20260902", "20260902")
    assert fake.calls == [("801010.SI", "20260902", "20260902")]
    assert got["pct_chg"].iloc[0] == -2.01
    assert "pct_change" not in got.columns

    empty = sw_daily_adapter(_FakePro(pd.DataFrame()))("801010.SI", "20260902", "20260902")
    assert len(empty) == 0


def test_sw_daily_adapter_end_to_end_with_fetch_sw_sector_map():
    """adapter 输出可直接喂 fetch_sw_sector_map: 单位换算链路无回归."""
    frame = pd.DataFrame({"pct_change": [1.23], "close": [4567.89], "vol": [2_345_678.0]})
    m = fetch_sw_sector_map(
        sw_daily_adapter(_FakePro(frame)), {"A": "801010"}, {"A"}, "20260902", delay=0
    )
    assert m["A"] == {"sw_ret_1d": 0.0123, "sw_index_close": 4567.89, "sw_index_vol": 2.35}
