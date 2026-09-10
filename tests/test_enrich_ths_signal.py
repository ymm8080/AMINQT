"""merge_ths_signal 单测 — THS问财看涨信号池并入面板.

覆盖: 信号列改名(日期后缀)/symbol对齐/dup防护/看跌池按日广播/空文件跳过.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import enrich_one_source


def _make_bull_file(dir_path, dstr, codes, extra_col=None):
    df = pd.DataFrame(
        {
            "股票代码": codes,
            "股票简称": [f"股{i}" for i in range(len(codes))],
            "最新价": [10.0] * len(codes),
            f"准备拉升(条件说明)[{dstr}]": ["量价齐升"] * len(codes),
            f"买入信号inter[{dstr}]": ["月线bias买入||周线cci买入"] * len(codes),
            f"技术形态[{dstr}]": ["价升量涨||阳线"] * len(codes),
            "market_code": [17] * len(codes),
            "code": codes,
            "_query_date": [dstr] * len(codes),
        }
    )
    if extra_col:
        df[extra_col] = "x"
    df.to_parquet(dir_path / f"bull_{dstr}.parquet")


@pytest.fixture
def ths_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(enrich_one_source, "THS_SIGNAL_DIR", tmp_path)
    return tmp_path


def _panel():
    return pd.DataFrame(
        {
            "symbol": ["600104", "000001", "600104"],
            "date": pd.to_datetime(["2026-09-08", "2026-09-08", "2026-09-09"]),
            "close": [10.0, 20.0, 11.0],
        }
    )


def test_signal_columns_merged_on_symbol_date(ths_dir):
    _make_bull_file(ths_dir, "20260908", ["600104"])
    out = enrich_one_source.merge_ths_signal(_panel())
    assert "ths_bull" in out.columns and "ths_buy_signals" in out.columns
    row = out[(out["symbol"] == "600104") & (out["date"] == "2026-09-08")]
    assert row["ths_bull"].iloc[0] == 1
    # 未命中行保持NaN, 不伪造0
    assert out["ths_bull"].isna().sum() == 2


def test_signal_date_suffix_columns_renamed(ths_dir):
    _make_bull_file(ths_dir, "20260908", ["600104"])
    out = enrich_one_source.merge_ths_signal(_panel())
    for c in ["ths_ready_rise", "ths_buy_signals", "ths_tech_pattern"]:
        assert c in out.columns
    assert not any("[20260908]" in c for c in out.columns)


def test_duplicate_rows_do_not_explode_panel(ths_dir):
    # 同(symbol,date)重复行必须去重, 否则4M行面板会被merge炸行
    _make_bull_file(ths_dir, "20260908", ["600104", "600104"])
    out = enrich_one_source.merge_ths_signal(_panel())
    assert len(out) == len(_panel())


def test_bear_pool_broadcast_by_date(ths_dir):
    _make_bull_file(ths_dir, "20260908", ["600104"])
    pd.DataFrame([{"date": "20260908", "bear_count": 2390}]).to_csv(
        ths_dir / "bear_counts.csv", index=False
    )
    out = enrich_one_source.merge_ths_signal(_panel())
    day = out[out["date"] == "2026-09-08"]
    assert (day["ths_bear_pool"] == 2390).all()
    other = out[out["date"] != "2026-09-08"]
    assert other["ths_bear_pool"].isna().all()


def test_bear_duplicate_dates_do_not_explode_panel(ths_dir):
    # 断点重启会让bear_counts.csv积累重复日期行; dup曾把当日全部面板行翻倍
    _make_bull_file(ths_dir, "20260908", ["600104"])
    pd.DataFrame(
        [{"date": "20260908", "bear_count": 2390},
         {"date": "20260908", "bear_count": 2390}]
    ).to_csv(ths_dir / "bear_counts.csv", index=False)
    out = enrich_one_source.merge_ths_signal(_panel())
    assert len(out) == 3


def test_empty_bull_file_skipped(ths_dir):
    pd.DataFrame().to_parquet(ths_dir / "bull_20260908.parquet")
    out = enrich_one_source.merge_ths_signal(_panel())
    assert "ths_bull" not in out.columns
    assert len(out) == 3


def test_no_files_returns_panel_unchanged(ths_dir):
    out = enrich_one_source.merge_ths_signal(_panel())
    assert list(out.columns) == ["symbol", "date", "close"]
    assert len(out) == 3
