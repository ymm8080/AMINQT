"""合并清单单文件 (scripts/_stocklist_combined.py) 单测: 四线源页序 +
多模块重叠页 + 缺源跳页 + WORM 写出。
"""

import pandas as pd
import pytest

from scripts import _stocklist_combined as sc

DATE = "20260105"


def _legacy(tmp_path):
    pd.DataFrame(
        {
            "symbol": ["600000", "600001"],
            "pred_ret_10d": ["10.0%", "8.0%"],
            "prob_up_10d": ["70%", "60%"],
        }
    ).to_csv(tmp_path / f"legacy_stocklist_{DATE}__M1.csv", index=False)


def _parallel(tmp_path):
    pd.DataFrame(
        {
            "symbol": ["600000", "601000"],
            "pred_mag_10d": ["7.0%", "5.0%"],
            "pred_prob_10d": ["51%", "52%"],
        }
    ).to_csv(tmp_path / f"parallel_shortlist_{DATE}__M1.csv", index=False)


def test_build_overlap_first_and_missing_sources_skipped(tmp_path):
    _legacy(tmp_path)
    _parallel(tmp_path)
    sheets = sc.build(DATE, list_dir=tmp_path, shadow_dir=tmp_path / "nope")
    assert [n for n, _ in sheets] == ["多模块重叠", "LEGACY", "PARALLEL"]
    overlap = sheets[0][1]
    assert overlap["symbol"].tolist() == ["600000"]  # 唯一双模块票
    assert overlap["模块"].tolist() == ["LEGACY+PARALLEL"]
    row = overlap.iloc[0]
    assert row["LEGACY_预测10d"] == "10.0%"
    assert row["LEGACY_概率10d"] == "70%"
    assert row["PARALLEL_预测10d"] == "7.0%"
    assert row["PARALLEL_概率10d"] == "51%"
    # 单模块票不上重叠页
    assert set(overlap["symbol"]) == {"600000"}


def test_build_slowbull_from_shadow_dashdate_counts_overlap(tmp_path):
    _legacy(tmp_path)
    _parallel(tmp_path)
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    pd.DataFrame({"symbol": ["600000", "300001"], "wr5": ["0.1", "0.2"]}).to_csv(
        shadow / "slowbull_list_2026-01-05__v4.csv", index=False
    )
    sheets = sc.build(DATE, list_dir=tmp_path, shadow_dir=shadow)
    assert [n for n, _ in sheets] == ["多模块重叠", "LEGACY", "PARALLEL", "SLOW_BULL"]
    assert sheets[0][1].iloc[0]["模块数"] == 3  # 600000 三模块居首


def test_build_no_core_sources_raises(tmp_path):
    with pytest.raises(SystemExit):
        sc.build(DATE, list_dir=tmp_path, shadow_dir=tmp_path)


def test_market_fade_sheet_formats():
    fc = {
        "next_date": "2026-09-10",
        "state_date": "20260909",
        "close": 3951.51,
        "ret": 0.0028,
        "regime": "线上(0~+2%)",
        "above_ma20": 0.0038,
        "r5": 0.0026,
        "vratio": 1.01,
        "p_surge": 0.539,
        "p_fade_given_surge": 0.337,
        "p_fade_day": 0.182,
        "base_fade": 0.219,
        "verdict": "≈常态",
        "n_regime": 1409,
    }
    name, df = sc.market_fade_sheet(fc_fn=lambda: fc)
    assert name == "市场FADE预测"
    assert len(df) == 11
    got = dict(zip(df["指标"], df["值"]))
    assert got["P(回落日) 预测"] == "18%"
    assert got["预测交易日"] == "2026-09-10"
    assert "线上(0~+2%)" in got["行情带 (距MA20)"]


def test_market_fade_sheet_failopen():
    def boom():
        raise RuntimeError("refetch fail")

    assert sc.market_fade_sheet(fc_fn=boom) is None


def test_write_xlsx_roundtrip(tmp_path):
    _legacy(tmp_path)
    _parallel(tmp_path)
    sheets = sc.build(DATE, list_dir=tmp_path, shadow_dir=tmp_path)
    fp = sc.write(sheets, DATE, list_dir=tmp_path)
    assert fp.name == f"stocklist_combined_{DATE}.xlsx"
    xl = pd.ExcelFile(fp)
    assert xl.sheet_names == ["多模块重叠", "LEGACY", "PARALLEL"]
    leg = xl.parse("LEGACY", dtype=str)
    assert leg["symbol"].tolist() == ["600000", "600001"]
    assert leg["pred_ret_10d"].tolist() == ["10.0%", "8.0%"]  # 显示层原样
