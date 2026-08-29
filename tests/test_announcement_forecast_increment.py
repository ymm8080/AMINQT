"""公告管线业绩预告增量源 (2026-08-27 接线).

水位线 = forecast 缓存目录文件名 (all_/inc_) 中的最大结束日;
每次只拉 [水位+1, 目标日] 缺口, WORM 写 inc_<s>_<e>.parquet (0 行也写=水位标记),
使 fetch_earnings_forecast 的年窗缓存保持日常增量 (业绩预告已终审 FAIL 不入生产,
纯缓存备查 — 见 memory forecast-unlock-evt-data-state).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from scripts import run_announcement_pipeline as rap


class _FakeSupply:
    def _tushare_pro(self):
        return None


def _seed(out_dir, name, ann_dates=("20260101",)):
    path = out_dir / name
    pd.DataFrame({"ann_date": list(ann_dates), "ts_code": ["000001.SZ"]}).to_parquet(
        path, index=False
    )
    return path


def test_gap_from_year_file_fetches_and_writes_inc(tmp_path, monkeypatch):
    _seed(tmp_path, "all_20250826_20260825.parquet", ("20260825",))
    calls = []

    def fake_fetch_window(pro, s, e):
        calls.append((s, e))
        return pd.DataFrame(
            {"ts_code": ["000002.SZ"], "ann_date": ["20260826"], "symbol": ["000002"]}
        )

    monkeypatch.setattr(rap, "fetch_window", fake_fetch_window)
    df = rap._fetch_forecast_increment(_FakeSupply(), "20260827", out_dir=str(tmp_path))
    assert calls == [(dt.date(2026, 8, 26), dt.date(2026, 8, 27))]
    assert len(df) == 1
    inc = tmp_path / "inc_20260826_20260827.parquet"
    assert inc.exists()
    assert len(pd.read_parquet(inc)) == 1


def test_inc_file_counts_as_watermark(tmp_path, monkeypatch):
    _seed(tmp_path, "all_20250826_20260825.parquet")
    _seed(tmp_path, "inc_20260826_20260826.parquet")
    calls = []

    monkeypatch.setattr(
        rap, "fetch_window", lambda pro, s, e: calls.append((s, e)) or pd.DataFrame()
    )
    rap._fetch_forecast_increment(_FakeSupply(), "20260827", out_dir=str(tmp_path))
    assert calls == [(dt.date(2026, 8, 27), dt.date(2026, 8, 27))]


def test_zero_rows_still_writes_watermark(tmp_path, monkeypatch):
    _seed(tmp_path, "all_20250826_20260825.parquet")
    monkeypatch.setattr(rap, "fetch_window", lambda pro, s, e: pd.DataFrame())
    df = rap._fetch_forecast_increment(_FakeSupply(), "20260826", out_dir=str(tmp_path))
    assert df.empty
    assert (tmp_path / "inc_20260826_20260826.parquet").exists()


def test_no_baseline_cache_raises(tmp_path):
    with pytest.raises(RuntimeError, match="基线"):
        rap._fetch_forecast_increment(_FakeSupply(), "20260827", out_dir=str(tmp_path))


def test_up_to_date_skips_fetch(tmp_path, monkeypatch):
    _seed(tmp_path, "all_20250826_20260827.parquet", ("20260827",))

    def boom(pro, s, e):  # 不应被调用
        raise AssertionError("up-to-date 不应触发网络拉取")

    monkeypatch.setattr(rap, "fetch_window", boom)
    df = rap._fetch_forecast_increment(_FakeSupply(), "20260827", out_dir=str(tmp_path))
    assert df.empty
    assert not (tmp_path / "inc_20260827_20260827.parquet").exists()
