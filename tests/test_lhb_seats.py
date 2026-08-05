# -*- coding: utf-8 -*-
"""Unit tests for lhb_seats — 席位静态分类 + top_inst → 8 列席位聚合 (LHB v2 上游)."""
import numpy as np
import pandas as pd

from app.pipeline1.lhb_seats import SEAT_COLS, classify_seat, seat_wide_from_top_inst


def _raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": [
                "000001.SZ", "000001.SZ", "000001.SZ", "000001.SZ",
                "000002.SZ", "000002.SZ",
                "600000.SH", "600000.SH",
            ],
            "exalter": [
                "机构专用",  # inst
                "拉萨团结路", "拉萨团结路",  # retail (同席位重复, 取 max)
                "国泰君安证券股份有限公司上海江苏路营业部",  # top (江苏路)
                "机构专用",  # inst
                "华鑫证券上海分公司",  # quant
                "机构专用",  # inst
                "国元证券杭州庆春路营业部",  # other → 并入 retail
            ],
            "buy": [100.0, 50.0, 20.0, 30.0, 40.0, 10.0, 60.0, 20.0],
            "sell": [10.0, 5.0, 0.0, 3.0, 4.0, 1.0, 6.0, 2.0],
        }
    )


def test_classify_seat_buckets():
    assert classify_seat("机构专用") == "inst"
    assert classify_seat("国泰君安证券股份有限公司上海江苏路营业部") == "top"
    assert classify_seat("华鑫证券上海分公司") == "quant"
    assert classify_seat("华宝证券上海东大名路营业部") == "quant"
    assert classify_seat("拉萨团结路") == "retail"
    assert classify_seat("国元证券杭州庆春路营业部") == "other"
    assert classify_seat(None) == "other"


def test_seat_wide_from_top_inst_aggregation():
    out = seat_wide_from_top_inst(_raw()).set_index("symbol")

    # 每股独立聚合 (spec: 席位金额必须落在具体股票上)
    assert out.shape[0] == 3
    assert set(out.columns) == set(SEAT_COLS)

    r = out.loc["000001"]
    assert r["lhb_inst_buy"] == 100.0 and r["lhb_inst_sell"] == 10.0
    assert r["lhb_top_buy"] == 30.0 and r["lhb_top_sell"] == 3.0
    # 同席位重复 → 取 max (50, 非 50+20), 缺席类别 → NaN
    assert r["lhb_retail_buy"] == 50.0 and r["lhb_retail_sell"] == 5.0
    assert np.isnan(r["lhb_quant_buy"]) and np.isnan(r["lhb_quant_sell"])

    r = out.loc["000002"]
    assert r["lhb_inst_buy"] == 40.0 and r["lhb_inst_sell"] == 4.0
    assert r["lhb_quant_buy"] == 10.0 and r["lhb_quant_sell"] == 1.0
    assert np.isnan(r["lhb_top_buy"]) and np.isnan(r["lhb_retail_buy"])

    # other 席位并入 retail (spec §2.4 "非聪明钱"整体)
    r = out.loc["600000"]
    assert r["lhb_inst_buy"] == 60.0 and r["lhb_inst_sell"] == 6.0
    assert r["lhb_retail_buy"] == 20.0 and r["lhb_retail_sell"] == 2.0
    assert np.isnan(r["lhb_top_buy"]) and np.isnan(r["lhb_quant_buy"])


def test_seat_wide_from_top_inst_empty_input():
    out = seat_wide_from_top_inst(pd.DataFrame())
    assert list(out.columns) == ["symbol"] + SEAT_COLS
    assert len(out) == 0
    assert len(seat_wide_from_top_inst(None)) == 0
