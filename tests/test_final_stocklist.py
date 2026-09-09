"""终版清单 (scripts/_final_stocklist.py) 单测: 三源合并 + module 列 +
status/reason 优先级 + win_rate 展示 + xlsx 写出。
"""

import pandas as pd
import pytest

from scripts import _final_stocklist as fs

DATE = "20260105"


def _fixtures(tmp_path):
    pd.DataFrame(
        {"symbol": ["600000", "600001", "600002"], "pred_ret_10d": [0.1, 0.2, 0.3]}
    ).to_csv(tmp_path / f"legacy_stocklist_{DATE}__M1.csv", index=False)
    pd.DataFrame(
        {"symbol": ["601000", "601001"], "rank": [0, 1], "systems": ["sniper", None]}
    ).to_csv(tmp_path / f"parallel_shortlist_{DATE}__M1.csv", index=False)
    pd.DataFrame({"symbol": ["300001"]}).to_csv(
        tmp_path / f"prob10dens_{DATE}__prob10dens.csv", index=False
    )


def _deadzone_result(tmp_path):
    # 三源全部停推 → 结果单 status=deadzone (推送端三件套通道)
    pd.DataFrame(
        {
            "symbol": ["600000", "600001", "600002", "601000", "601001", "300001"],
            "status": ["deadzone"] * 6,
        }
    ).to_csv(tmp_path / f"ths_push_result_{DATE}__deadzone.csv", index=False)


def _patch_gate(monkeypatch, alarm=True, wr=0.2):
    monkeypatch.setattr(fs, "is_alarm", lambda line, date: (alarm, "why"))
    monkeypatch.setattr(fs, "win_rate", lambda line, date: wr)


def test_build_merges_three_sources_with_module(tmp_path, monkeypatch):
    _fixtures(tmp_path)
    _patch_gate(monkeypatch, alarm=False, wr=0.3)
    df = fs.build(DATE, list_dir=tmp_path)
    by_sym = df.set_index("symbol")
    assert by_sym.loc["600000", "module"] == "legacy"
    assert by_sym.loc["601000", "module"] == "sniper"
    assert by_sym.loc["601001", "module"] == ""  # 空 = parallel 无系统标记
    assert by_sym.loc["300001", "module"] == "prob10dens"
    assert (df["win_rate"] == "30.0%").all()
    # 行序: legacy → parallel → prob10dens (_SOURCES 顺序, 生产在前)
    assert df["symbol"].tolist() == [
        "600000",
        "600001",
        "600002",
        "601000",
        "601001",
        "300001",
    ]


def test_status_priority_flush_landed_deadzone_result(tmp_path, monkeypatch):
    _fixtures(tmp_path)
    _deadzone_result(tmp_path)
    # 600000 早上实推已落袋 (legacy 源结果单 landed) → 覆盖 deadzone
    pd.DataFrame(
        {"symbol": ["600000", "600001"], "status": ["landed", "manual"]}
    ).to_csv(tmp_path / f"ths_push_result_{DATE}__legacy__M1.csv", index=False)
    # 600002 被放量守卫删过 → flush 压过 landed/deadzone
    pd.DataFrame({"asof": [DATE], "symbol": ["600002"], "action": ["removed"]}).to_csv(
        tmp_path / f"ths_flush_removed_{DATE}__flushguard.csv", index=False
    )
    _patch_gate(monkeypatch, alarm=True, wr=0.2)
    df = fs.build(DATE, list_dir=tmp_path)
    by_sym = df.set_index("symbol")
    assert (by_sym.loc["600000", "status"], by_sym.loc["600000", "reason"]) == (
        "landed",
        "",
    )
    assert (by_sym.loc["600002", "status"], by_sym.loc["600002", "reason"]) == (
        "blocked",
        "flush",
    )
    for sym in ["600001", "601000", "300001"]:
        assert (by_sym.loc[sym, "status"], by_sym.loc[sym, "reason"]) == (
            "blocked",
            "deadzone",
        )


def test_no_alarm_uses_result_status_or_not_pushed(tmp_path, monkeypatch):
    _fixtures(tmp_path)
    pd.DataFrame(
        {"symbol": ["600000", "600001"], "status": ["manual", "ui_fail"]}
    ).to_csv(tmp_path / f"ths_push_result_{DATE}__legacy__M1.csv", index=False)
    _patch_gate(monkeypatch, alarm=False, wr=0.5)
    df = fs.build(DATE, list_dir=tmp_path)
    by_sym = df.set_index("symbol")
    assert by_sym.loc["600000", "reason"] == "manual"
    assert by_sym.loc["600001", "reason"] == "ui_fail"
    # 无结果单的票 → not_pushed
    assert by_sym.loc["601000", "reason"] == "not_pushed"
    assert (df["status"] == "blocked").all()


def test_win_rate_blank_when_none(tmp_path, monkeypatch):
    _fixtures(tmp_path)
    _patch_gate(monkeypatch, alarm=False, wr=None)
    df = fs.build(DATE, list_dir=tmp_path)
    assert (df["win_rate"] == "").all()


def test_build_no_sources_raises(tmp_path, monkeypatch):
    _patch_gate(monkeypatch)
    with pytest.raises(SystemExit):
        fs.build(DATE, list_dir=tmp_path)


def test_write_xlsx_roundtrip(tmp_path):
    df = pd.DataFrame(
        {"module": ["legacy"], "symbol": ["600000"], "status": ["landed"]}
    )
    fp = fs.write(df, DATE, list_dir=tmp_path)
    assert fp.name == f"stocklist_final_{DATE}__" + fp.name.split("__")[1]
    back = pd.read_excel(fp, dtype={"symbol": str})
    assert back["symbol"].tolist() == ["600000"]
    assert list(back.columns) == ["module", "symbol", "status"]


def test_archive_push_artifacts_moves_only_push_files(tmp_path):
    """????: ?? ths txt/????? ths_push_archive/, ?? CSV ??;
    ??? read_push_results ??? (???/???????)."""
    from scripts._ths_watchlist_push import read_push_results

    (tmp_path / f"ths_watchlist_{DATE}__18__parallel__M1.txt").write_text("600001\n")
    pd.DataFrame({"symbol": ["600001"], "status": ["landed"]}).to_csv(
        tmp_path / f"ths_push_result_{DATE}__18__parallel__M1.csv", index=False
    )
    (tmp_path / f"legacy_stocklist_{DATE}__M1.csv").write_text("symbol\n600001\n")

    n = fs.archive_push_artifacts(DATE, list_dir=tmp_path)
    assert n == 2
    assert not list(tmp_path.glob(f"ths_watchlist_{DATE}__*.txt"))
    assert not list(tmp_path.glob(f"ths_push_result_{DATE}__*.csv"))
    assert (
        tmp_path / fs.THS_PUSH_ARCHIVE / f"ths_push_result_{DATE}__18__parallel__M1.csv"
    ).exists()
    assert (tmp_path / f"legacy_stocklist_{DATE}__M1.csv").exists()  # ??????

    res = read_push_results(DATE, list_dir=tmp_path)
    assert res["source"].tolist() == ["parallel"]
    assert res["status"].tolist() == ["landed"]

    # ??: ????????
    assert fs.archive_push_artifacts(DATE, list_dir=tmp_path) == 0
