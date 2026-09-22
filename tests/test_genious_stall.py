"""Tests for genious 交付的横盘提示记录 (0922 用户令: 最终 STOCKLIST 必须带记录,
光日志无用).

覆盖两点:
- write_stocklist_csv: 推送边车带 横盘提示/涨停提示 列 (有则带, 无则保持最小两列);
- _history_counts 对 genious_stocklist_{date}__{HHMMSS} 时间戳戳文件名的正则匹配
  (genious 边车每天带 __HHMMSS 戳, 与 legacy 平名文件混存互不污染).
"""

import pandas as pd

from scripts._genious_excel import write_stocklist_csv
from scripts._stall_marker import _history_counts


def test_csv_includes_stall_columns(tmp_path):
    s1 = pd.DataFrame(
        [
            {
                "排名": 1,
                "symbol": "300911",
                "横盘提示": "近10日未涨·冷静市",
                "涨停提示": "",
            },
            {"排名": 2, "symbol": "600001", "横盘提示": "", "涨停提示": "涨停次日不追"},
        ]
    )
    fp = write_stocklist_csv(s1, "20260205", list_dir=tmp_path)
    assert fp is not None and fp.exists()
    assert fp.name.startswith("genious_stocklist_20260205__")
    d = pd.read_csv(fp, dtype={"symbol": str})
    assert list(d.columns) == ["排名", "symbol", "横盘提示", "涨停提示"]
    assert d.loc[0, "横盘提示"] == "近10日未涨·冷静市"
    assert d.loc[1, "涨停提示"] == "涨停次日不追"


def test_csv_without_stall_columns_keeps_minimal(tmp_path):
    s1 = pd.DataFrame([{"排名": 1, "symbol": "300911"}])
    fp = write_stocklist_csv(s1, "20260205", list_dir=tmp_path)
    d = pd.read_csv(fp, dtype={"symbol": str})
    assert list(d.columns) == ["排名", "symbol"]


def test_csv_empty_list_no_file(tmp_path):
    s1 = pd.DataFrame(columns=["排名", "symbol"])
    fp = write_stocklist_csv(s1, "20260205", list_dir=tmp_path)
    assert fp is None
    assert not list(tmp_path.glob("genious_stocklist_*.csv"))


def test_history_counts_stamp_filenames(tmp_path):
    # genious 文件名带 __HHMMSS 戳 (WORM 同日重跑), 正则须匹配且不混入其他前缀
    pd.DataFrame({"symbol": ["300911"]}).to_csv(
        tmp_path / "genious_stocklist_20260203__203041.csv", index=False
    )
    pd.DataFrame({"symbol": ["300911", "600001"]}).to_csv(
        tmp_path / "genious_stocklist_20260204__203102.csv", index=False
    )
    pd.DataFrame({"symbol": ["300999"]}).to_csv(
        tmp_path / "legacy_stocklist_20260204__m1.csv", index=False
    )
    counts = _history_counts(str(tmp_path), "20260205", "genious_stocklist_", 20)
    assert counts.get("300911") == 2
    assert counts.get("600001") == 1
    assert "300999" not in counts
