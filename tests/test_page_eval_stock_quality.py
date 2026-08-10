"""个股预测质量: _actual_forward_returns + _load_panel 基准日截断 (TDD)."""

from __future__ import annotations

import pandas as pd
import pytest

from app.streamlit import page_eval


def _panel(
    close_hfq: dict[str, list[float]],
    dates: list[str] | None = None,
) -> pd.DataFrame:
    """构造合成面板: 每 symbol 连续交易日, close_hfq 递增."""
    if dates is None:
        dates = [
            "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
            "2024-01-08", "2024-01-09", "2024-01-10",
        ]
    dts = pd.to_datetime(dates)
    rows = []
    for sym, closes in close_hfq.items():
        for d, c in zip(dts, closes, strict=True):
            rows.append({"symbol": sym, "date": d, "close_hfq": c})
    return pd.DataFrame(rows)


def test_actual_forward_returns_ok():
    """基准日为交易日时, 3/5/10 个交易日后的真实涨幅正确回算."""
    p = _panel({"600519": [100, 101, 102, 103, 104, 105, 106]})
    as_of = pd.DataFrame(
        {"symbol": ["600519"], "date": pd.to_datetime(["2024-01-03"])}
    )  # dseq=1, close=101
    out = page_eval._actual_forward_returns(p, as_of)
    row = out.iloc[0]
    assert row["close_asof"] == pytest.approx(101.0)
    # dseq 4 (2024-01-08) = 104 → 3d 实际 104/101-1
    assert row["actual_ret_3d"] == pytest.approx(104 / 101 - 1)
    # dseq 6 (2024-01-10) = 106 → 5d 实际 106/101-1
    assert row["actual_ret_5d"] == pytest.approx(106 / 101 - 1)
    # 10d 无数据 → NaN
    assert pd.isna(row["actual_ret_10d"])


def test_actual_forward_returns_not_matured():
    """基准日太接近面板末尾 → 实际涨幅为 NaN (未成熟)."""
    p = _panel({"600519": [100, 101, 102, 103, 104, 105, 106]})
    as_of = pd.DataFrame(
        {"symbol": ["600519"], "date": pd.to_datetime(["2024-01-10"])}
    )  # 最后一个交易日, 无未来
    out = page_eval._actual_forward_returns(p, as_of)
    for k in ("3", "5", "10"):
        assert pd.isna(out.iloc[0][f"actual_ret_{k}d"])


def test_actual_forward_returns_multi_symbol_independent():
    """多只股票各自按自己的交易日序列回算, 互不串扰."""
    p = _panel(
        {
            "600519": [100, 101, 102, 103, 104, 105, 106],
            "000001": [50, 51, 52, 53, 54, 55, 56],
        }
    )
    as_of = pd.DataFrame(
        {
            "symbol": ["600519", "000001"],
            "date": pd.to_datetime(["2024-01-03", "2024-01-03"]),
        }
    )
    out = page_eval._actual_forward_returns(p, as_of).set_index("symbol")
    assert out.loc["600519", "actual_ret_3d"] == pytest.approx(104 / 101 - 1)
    assert out.loc["000001", "actual_ret_3d"] == pytest.approx(54 / 51 - 1)


def test_load_panel_caps_as_of_date(monkeypatch, tmp_path):
    """基准日截断: 只保留 date <= as_of 的行; symbol 过滤同时生效."""
    path = tmp_path / "panel.parquet"
    _panel(
        {"600519": [1, 2, 3, 4], "000001": [1, 2, 3, 4]},
        dates=["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
    ).to_parquet(path)
    monkeypatch.setattr(page_eval, "PANEL_V3_PATH", str(path))

    # 只截日期
    full = page_eval._load_panel()
    assert full["date"].nunique() == 4
    capped = page_eval._load_panel(as_of_date=pd.Timestamp("2024-01-03"))
    assert capped["date"].nunique() == 2
    assert capped["date"].max().strftime("%Y-%m-%d") == "2024-01-03"

    # symbol + 日期双过滤
    one = page_eval._load_panel(symbols=["600519"], as_of_date=pd.Timestamp("2024-01-04"))
    assert set(one["symbol"]) == {"600519"}
    assert one["date"].nunique() == 3


def test_load_panel_missing_file(monkeypatch, tmp_path):
    """面板缺失 → 返回 None (预测端安全处理)."""
    monkeypatch.setattr(page_eval, "PANEL_V3_PATH", str(tmp_path / "nope.parquet"))
    assert page_eval._load_panel() is None
