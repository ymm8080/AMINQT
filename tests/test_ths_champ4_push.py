"""冠军四段每日推送单测 (scripts/_ths_champ4_push.py, 2026-11-11 用户令).

- champ4_codes: 交付层 xlsx「冠军四段」页 symbol 列 (banner/图例/空行不混入,
  数值码 zfill 6 位, 保序去重, 非法码滤除)
- already_landed: 全 landed 结果单 → True (幂等跳过); blocked/manual → False (重推)
- _push_allowed: 总闸 False 时仅 genious__champ4 由 push_champ4_to_ths 放行,
  其余源仍停 (2026-11-03 停推口径不被 champ4 令扩大)
- main: dry-run 不写不推; 常规路径写 txt + 调 push_via_ths
"""

import pandas as pd
import pytest

from scripts import _ths_champ4_push as c4
from scripts import _ths_watchlist_push as w

DATE = "20261111"


def _write_genious_xlsx(path, champ_rows):
    """mimic 交付层: 冠军四段页 banner(row1) + 表头(row3, startrow=2) + 数据 + 页尾图例."""
    cols = ["明日开盘", "横盘提示", "排名", "symbol", "涨闸", "层", "执行档"]
    df = pd.DataFrame(champ_rows, columns=cols)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="冠军四段", index=False, startrow=2)
        ws = xw.sheets["冠军四段"]
        ws.cell(row=1, column=1, value="GENIOUS 冠军四段 — banner")
        ws.cell(row=4 + len(df) + 1, column=1, value="列说明: 图例行 (只 A 列)")


def test_champ4_codes_normalizes_dedups_and_skips_legend(tmp_path):
    _write_genious_xlsx(
        tmp_path / f"GENIOUS_{DATE}.xlsx",
        [
            ["", "", 1, "600001", "过闸", "T2", "T+1开盘进"],
            ["", "", 2, 1331, "没过闸", "T2", ""],  # 数值码 → zfill 001331
            ["", "", 3, "600001", "过闸", "T2", ""],  # 重复 → 去重保序
            ["", "", 4, "ABC", "过闸", "T2", ""],  # 非法码 → 滤除
        ],
    )
    codes = c4.champ4_codes(tmp_path / f"GENIOUS_{DATE}.xlsx")
    assert codes == ["600001", "001331"]


def test_champ4_codes_empty_sheet_returns_empty(tmp_path):
    _write_genious_xlsx(tmp_path / f"GENIOUS_{DATE}.xlsx", [])
    assert c4.champ4_codes(tmp_path / f"GENIOUS_{DATE}.xlsx") == []


def test_already_landed_true_when_all_landed(tmp_path):
    codes = ["600001", "001331"]
    txt = tmp_path / f"ths_watchlist_{DATE}__07__genious__champ4.txt"
    w.write_push_result(txt, codes, landed=codes)
    assert c4.already_landed(DATE, tmp_path) is True


def test_already_landed_false_when_blocked_or_manual(tmp_path):
    codes = ["600001", "001331"]
    txt = tmp_path / f"ths_watchlist_{DATE}__07__genious__champ4.txt"
    w.write_push_result(txt, codes, landed=[], blocked=True)
    assert c4.already_landed(DATE, tmp_path) is False
    w.write_push_result(
        tmp_path / f"ths_watchlist_{DATE}__08__genious__champ4.txt", codes, landed=["600001"]
    )  # 001331=manual
    assert c4.already_landed(DATE, tmp_path) is False


def test_already_landed_false_when_no_result(tmp_path):
    assert c4.already_landed(DATE, tmp_path) is False


@pytest.fixture
def gates(monkeypatch):
    """总闸 False + champ4 旗标 True — 双模块各自的名字绑定都要打."""
    monkeypatch.setattr(
        w, "GENIOUS", {**w.GENIOUS, "push_to_ths": False, "push_champ4_to_ths": True}
    )
    monkeypatch.setattr(
        c4, "GENIOUS", {**c4.GENIOUS, "push_to_ths": False, "push_champ4_to_ths": True}
    )


def test_push_allowed_champ4_exempt_preboard_still_blocked(gates):
    assert w._push_allowed("ths_watchlist_20261111__07__genious__champ4.txt") is True
    assert w._push_allowed("ths_watchlist_20261111__07__genious__preboard.txt") is False


def test_push_allowed_flag_off_restores_full_stop(monkeypatch):
    monkeypatch.setattr(
        w, "GENIOUS", {**w.GENIOUS, "push_to_ths": False, "push_champ4_to_ths": False}
    )
    assert w._push_allowed("ths_watchlist_20261111__07__genious__champ4.txt") is False


def test_push_allowed_master_on_pushes_everything(monkeypatch):
    monkeypatch.setattr(w, "GENIOUS", {**w.GENIOUS, "push_to_ths": True})
    assert w._push_allowed("ths_watchlist_20261111__07__genious__preboard.txt") is True


@pytest.fixture
def stub_log(monkeypatch):
    """静音心跳: main 的日志调用进内存列表, 断言不依赖 stderr/真实日志文件."""
    lines: list[str] = []

    class _Stub:
        def info(self, msg, *a):
            lines.append((msg % a) if a else msg)

        def warning(self, msg, *a):
            lines.append((msg % a) if a else msg)

        def error(self, msg, *a):
            lines.append((msg % a) if a else msg)

    monkeypatch.setattr(c4, "_setup_logging", lambda: None)
    monkeypatch.setattr(c4, "log", _Stub())
    return lines


def test_main_dry_run_lists_codes_without_writing(tmp_path, monkeypatch, gates, stub_log):
    _write_genious_xlsx(tmp_path / f"GENIOUS_{DATE}.xlsx", [["", "", 1, "600001", "过闸", "T2", ""]])
    monkeypatch.setattr(c4, "LIST_DIR", tmp_path)
    called = []
    monkeypatch.setattr(c4, "push_via_ths", lambda txt: called.append(txt) or True)
    assert c4.main([DATE, "--dry-run"]) == 0
    assert called == []
    assert not list(tmp_path.glob("ths_watchlist_*__genious__champ4.txt"))
    assert any("600001" in msg for msg in stub_log)


def test_main_writes_txt_and_pushes(tmp_path, monkeypatch, gates, stub_log):
    _write_genious_xlsx(tmp_path / f"GENIOUS_{DATE}.xlsx", [["", "", 1, "600001", "过闸", "T2", ""]])
    monkeypatch.setattr(c4, "LIST_DIR", tmp_path)
    called = []
    monkeypatch.setattr(c4, "push_via_ths", lambda txt: called.append(txt) or True)
    assert c4.main([DATE]) == 0
    assert len(called) == 1
    assert called[0].name.startswith(f"ths_watchlist_{DATE}__")
    assert called[0].name.endswith("__genious__champ4.txt")
    assert called[0].read_text(encoding="utf-8").split() == ["600001"]


def test_main_idempotent_skip_when_all_landed(tmp_path, monkeypatch, gates, stub_log):
    _write_genious_xlsx(tmp_path / f"GENIOUS_{DATE}.xlsx", [["", "", 1, "600001", "过闸", "T2", ""]])
    monkeypatch.setattr(c4, "LIST_DIR", tmp_path)
    w.write_push_result(
        tmp_path / f"ths_watchlist_{DATE}__07__genious__champ4.txt", ["600001"], landed=["600001"]
    )
    called = []
    monkeypatch.setattr(c4, "push_via_ths", lambda txt: called.append(txt) or True)
    assert c4.main([DATE]) == 0
    assert called == []


def test_main_force_repairs_despite_landed(tmp_path, monkeypatch, gates, stub_log):
    _write_genious_xlsx(tmp_path / f"GENIOUS_{DATE}.xlsx", [["", "", 1, "600001", "过闸", "T2", ""]])
    monkeypatch.setattr(c4, "LIST_DIR", tmp_path)
    w.write_push_result(
        tmp_path / f"ths_watchlist_{DATE}__07__genious__champ4.txt", ["600001"], landed=["600001"]
    )
    called = []
    monkeypatch.setattr(c4, "push_via_ths", lambda txt: called.append(txt) or True)
    assert c4.main([DATE, "--force"]) == 0
    assert len(called) == 1


def test_main_no_push_when_flag_off(tmp_path, monkeypatch, stub_log):
    _write_genious_xlsx(tmp_path / f"GENIOUS_{DATE}.xlsx", [["", "", 1, "600001", "过闸", "T2", ""]])
    monkeypatch.setattr(c4, "GENIOUS", {**c4.GENIOUS, "push_champ4_to_ths": False})
    monkeypatch.setattr(c4, "LIST_DIR", tmp_path)
    called = []
    monkeypatch.setattr(c4, "push_via_ths", lambda txt: called.append(txt) or True)
    assert c4.main([DATE]) == 0
    assert called == []


def test_main_empty_champ_sheet_is_ok(tmp_path, monkeypatch, gates, stub_log):
    _write_genious_xlsx(tmp_path / f"GENIOUS_{DATE}.xlsx", [])
    monkeypatch.setattr(c4, "LIST_DIR", tmp_path)
    called = []
    monkeypatch.setattr(c4, "push_via_ths", lambda txt: called.append(txt) or True)
    assert c4.main([DATE]) == 0
    assert called == []
