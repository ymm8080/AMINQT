"""merge_ths_signal 单测 — THS问财看涨/看跌信号并入面板.

覆盖: 信号列改名(日期后缀)/symbol对齐/du p防护/看跌旗标/条件0填充(特征资格门)/
计数解析("||"字面切分)/市场日计数广播/空文件跳过.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import enrich_one_source


def _make_bull_file(dir_path, dstr, codes, buy_sigs=None, tech=None):
    n = len(codes)
    df = pd.DataFrame(
        {
            "股票代码": codes,
            "股票简称": [f"股{i}" for i in range(n)],
            "最新价": [10.0] * n,
            f"准备拉升(条件说明)[{dstr}]": ["量价齐升"] * n,
            f"买入信号inter[{dstr}]": buy_sigs or ["月线bias买入||周线cci买入"] * n,
            f"技术形态[{dstr}]": tech or ["价升量涨||阳线"] * n,
            "market_code": [17] * n,
            "code": codes,
            "_query_date": [dstr] * n,
        }
    )
    df.to_parquet(dir_path / f"bull_{dstr}.parquet")


def _make_bear_file(dir_path, dstr, codes):
    n = len(codes)
    pd.DataFrame(
        {
            "股票代码": codes,
            "股票简称": [f"股{i}" for i in range(n)],
            "最新价": [10.0] * n,
            "最新涨跌幅": [-5.0] * n,
            "market_code": [17] * n,
            "code": codes,
            "_query_date": [dstr] * n,
        }
    ).to_parquet(dir_path / f"bear_{dstr}.parquet")


@pytest.fixture
def ths_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(enrich_one_source, "THS_SIGNAL_DIR", tmp_path)
    return tmp_path


def _panel():
    # 2026-09-05 < 信号起点(09-08): 验证区间前留NaN
    return pd.DataFrame(
        {
            "symbol": ["600104", "000001", "600104", "600104"],
            "date": pd.to_datetime(
                ["2026-09-08", "2026-09-08", "2026-09-09", "2026-09-05"]
            ),
            "close": [10.0, 20.0, 11.0, 9.0],
        }
    )


def test_bull_flag_and_zero_fill_semantics(ths_dir):
    _make_bull_file(ths_dir, "20260908", ["600104"])
    out = enrich_one_source.merge_ths_signal(_panel())
    assert out["ths_bull"].tolist()[:3] == [1.0, 0.0, 0.0]
    assert pd.isna(out["ths_bull"].iloc[3])  # 区间前留NaN
    assert out["ths_bull_buy_sig_n"].tolist()[:3] == [2.0, 0.0, 0.0]
    assert pd.isna(out["ths_bull_buy_sig_n"].iloc[3])
    assert out["ths_bull_tech_n"].tolist()[:3] == [2.0, 0.0, 0.0]
    assert pd.isna(out["ths_bull_tech_n"].iloc[3])


def test_signal_date_suffix_columns_renamed(ths_dir):
    _make_bull_file(ths_dir, "20260908", ["600104"])
    out = enrich_one_source.merge_ths_signal(_panel())
    for c in ["ths_bull_ready_rise", "ths_bull_buy_signals", "ths_bull_tech_pattern"]:
        assert c in out.columns
    assert not any("[20260908]" in c for c in out.columns)


def test_bear_flag_merged(ths_dir):
    _make_bull_file(ths_dir, "20260908", ["600104"])
    _make_bear_file(ths_dir, "20260908", ["000001"])
    out = enrich_one_source.merge_ths_signal(_panel())
    by = out.set_index(["symbol", "date"])
    assert by.loc[("600104", "2026-09-08"), "ths_bull"] == 1.0
    assert by.loc[("000001", "2026-09-08"), "ths_bear"] == 1.0
    assert by.loc[("000001", "2026-09-08"), "ths_bull"] == 0.0
    assert by.loc[("600104", "2026-09-08"), "ths_bear"] == 0.0


def test_duplicate_rows_do_not_explode_panel(ths_dir):
    _make_bull_file(ths_dir, "20260908", ["600104", "600104"])
    out = enrich_one_source.merge_ths_signal(_panel())
    assert len(out) == len(_panel())


def test_bear_duplicate_rows_do_not_explode_panel(ths_dir):
    _make_bull_file(ths_dir, "20260908", ["600104"])
    _make_bear_file(ths_dir, "20260908", ["000001", "000001"])
    out = enrich_one_source.merge_ths_signal(_panel())
    assert len(out) == 4


def test_bear_pool_broadcast_by_date(ths_dir):
    _make_bull_file(ths_dir, "20260908", ["600104"])
    pd.DataFrame(
        [{"date": "20260908", "bear_count": 2390},
         {"date": "20260908", "bear_count": 2390}]  # 断点重启dup防线
    ).to_csv(ths_dir / "bear_counts.csv", index=False)
    out = enrich_one_source.merge_ths_signal(_panel())
    day = out[out["date"] == "2026-09-08"]
    assert (day["ths_bear_pool"] == 2390).all()
    assert len(out) == 4
    other = out[out["date"] != "2026-09-08"]
    assert other["ths_bear_pool"].isna().all()


def test_single_signal_no_pipe_counts_as_one(ths_dir):
    _make_bull_file(ths_dir, "20260908", ["600104"],
                    buy_sigs=["月线bias买入"], tech=["阳线"])
    out = enrich_one_source.merge_ths_signal(_panel())
    assert out["ths_bull_buy_sig_n"].iloc[0] == 1.0
    assert out["ths_bull_tech_n"].iloc[0] == 1.0


def test_empty_bull_file_skipped(ths_dir):
    pd.DataFrame().to_parquet(ths_dir / "bull_20260908.parquet")
    out = enrich_one_source.merge_ths_signal(_panel())
    assert "ths_bull" not in out.columns
    assert len(out) == 4


def test_no_files_returns_panel_unchanged(ths_dir):
    out = enrich_one_source.merge_ths_signal(_panel())
    assert list(out.columns) == ["symbol", "date", "close"]
    assert len(out) == 4
