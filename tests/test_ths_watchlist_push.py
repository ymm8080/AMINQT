"""同花顺自选股推送单测: 当日 TOP10 → 导入 txt (2026-09-05 用户口径: 双源分开成单).

- collect_lists: parallel 短名单 rank 前 10 与 legacy 清单序前 10 各自独立成单,
  标签 = 源名__模型批次串 (09-05 用户: LEGACY 文件就叫 LEGACY + TXT 需要模型名
  和日期), 并行在前, 两单互不掺和 (不并集); 各单内部逐码去重, 跨源重复股保留
  (09-05 用户); 单侧缺失只返回另一侧
- REJECTED / gateinfo / preds_raw 等旁支文件不混入
- ths_txt_path / write_ths_txt: 每行一个 6 位代码, WORM 命名含日期+module
"""

import os

import pandas as pd
import pytest

from scripts import _ths_watchlist_push as mod


def _write_shortlist(path, rank_symbols):
    pd.DataFrame(
        [{"symbol": s, "rank": r} for r, s in enumerate(rank_symbols, start=1)]
    ).to_csv(path, index=False)


def _write_legacy(path, symbols):
    pd.DataFrame([{"symbol": s} for s in symbols]).to_csv(path, index=False)


def test_collect_lists_shortlist_top10_by_rank(tmp_path):
    rank_symbols = [f"6000{i:02d}" for i in range(1, 21)]  # rank 1..20
    _write_shortlist(tmp_path / "parallel_shortlist_20260901__M1.csv", rank_symbols)
    lists = mod.collect_lists("20260901", tmp_path)
    assert lists == [("parallel__M1", rank_symbols[:10])]


def test_collect_lists_two_sources_independent_no_cross_dedup(tmp_path):
    """双源分开成单 (2026-09-05): 各取各的 top10, 重叠码两单都保留 (推送是加自选,
    同码二次导入无害); 先 parallel 后 legacy."""
    _write_shortlist(
        tmp_path / "parallel_shortlist_20260901__M1.csv", ["600001", "600002", "600003"]
    )
    _write_legacy(
        tmp_path / "legacy_stocklist_20260901__M2.csv", ["600003", "603829", "002968"]
    )
    lists = mod.collect_lists("20260901", tmp_path)
    assert lists == [
        ("parallel__M1", ["600001", "600002", "600003"]),
        ("legacy__M2", ["600003", "603829", "002968"]),
    ]


def test_collect_lists_within_list_dedup_cross_source_kept(tmp_path):
    """各单内部去重; 跨源重复股两单都保留 (2026-09-05 用户拍板)."""
    _write_shortlist(
        tmp_path / "parallel_shortlist_20260901__M1.csv",
        ["600001", "600001", "600002"],
    )
    _write_legacy(
        tmp_path / "legacy_stocklist_20260901__M2.csv",
        ["600002", "603829", "603829"],
    )
    lists = mod.collect_lists("20260901", tmp_path)
    assert lists == [
        ("parallel__M1", ["600001", "600002"]),
        ("legacy__M2", ["600002", "603829"]),
    ]


def test_collect_lists_fallback_legacy_only(tmp_path):
    _write_legacy(
        tmp_path / "legacy_stocklist_20260901__M1.csv",
        ["002968", "603829", "603326", "600706", "001217"],
    )
    lists = mod.collect_lists("20260901", tmp_path)
    assert lists == [("legacy__M1", ["002968", "603829", "603326", "600706", "001217"])]


def test_collect_lists_filters_bad_codes(tmp_path):
    # parallel 侧去重; legacy 侧不去重只过滤
    _write_shortlist(
        tmp_path / "parallel_shortlist_20260901__M1.csv",
        ["002968", "nan", "2968", "600abc", "603829.0", "002968", "603829"],
    )
    _write_legacy(
        tmp_path / "legacy_stocklist_20260901__M2.csv",
        ["002968", "nan", "2968", "600abc", "603829.0", "002968", "603829"],
    )
    lists = mod.collect_lists("20260901", tmp_path)
    # "603829.0" 不满足 ^\d{6}$ → 过滤; 两单各自单内去重
    assert lists == [
        ("parallel__M1", ["002968", "603829"]),
        ("legacy__M2", ["002968", "603829"]),
    ]


def test_collect_lists_ignores_sidecar_files(tmp_path):
    _write_shortlist(tmp_path / "parallel_shortlist_20260901__M1.csv", ["002968"])
    _write_legacy(tmp_path / "legacy_stocklist_20260901__M1.csv", ["603829"])
    # 旁支文件: REJECTED / gateinfo / preds_raw / 其他日期 — 均不得混入
    _write_legacy(tmp_path / "legacy_gateinfo_20260901__M1.csv", ["999999"])
    _write_legacy(tmp_path / "parallel_preds_raw_20260901__M1.csv", ["888888"])
    _write_shortlist(tmp_path / "parallel_shortlist_20260831__M0.csv", ["666666"])
    lists = mod.collect_lists("20260901", tmp_path)
    # 同批次串两源也各带各的源名 → 文件名永不撞
    assert lists == [
        ("parallel__M1", ["002968"]),
        ("legacy__M1", ["603829"]),
    ]


def test_collect_lists_keeps_index_colliding_000xxx(tmp_path):
    """撞指数码 000xxx 保留在清单 (2026-09-05 用户澄清 "不是删除股票号");
    推送端隔离指数行 (见 _build_chunks), 清单侧不剔不补位."""
    rank_symbols = ["600001", "000985", "600002", "600003", "600004", "600005"]
    _write_shortlist(tmp_path / "parallel_shortlist_20260901__M1.csv", rank_symbols)
    _write_legacy(
        tmp_path / "legacy_stocklist_20260901__M2.csv", ["000001", "603829", "000123"]
    )
    lists = mod.collect_lists("20260901", tmp_path)
    assert lists == [
        ("parallel__M1", rank_symbols),
        ("legacy__M2", ["000001", "603829", "000123"]),
    ]


def test_collect_lists_no_delivered_raises(tmp_path):
    with pytest.raises(SystemExit):
        mod.collect_lists("20260901", tmp_path)


def test_txt_format_and_worm_naming(tmp_path):
    out = mod.ths_txt_path("20260901", "M20260828__D20260830excessfix", tmp_path)
    assert out.name == "ths_watchlist_20260901__M20260828__D20260830excessfix.txt"
    mod.write_ths_txt(["002968", "603829"], out)
    content = out.read_text(encoding="utf-8")
    assert content == "002968\n603829\n"


def test_flush_str_cannot_form_code_rows():
    """冲刷串必须无法被识别器解析成代码行 (旧 "ths-push" 被识别成两只美股, 09-03)."""
    assert not any(c.isalnum() for c in mod.FLUSH_STR)


def _dlg_img(rows):
    """合成对话框列表区截图: rows = [(y0, checked[, index[, up_red]])]; 白底,
    文字带 x40-80, 勾框 x17-28, index=True 时市场列 x295-320 加红字 ("中证"
    tag), up_red=True 时价格列 x380-420 加红字 (上涨股红价, 09-05 实证)."""
    import numpy as np

    img = np.full((100, 500, 3), 240, dtype=np.uint8)
    for row in rows:
        y0, checked = row[0], row[1]
        is_index = row[2] if len(row) > 2 else False
        up_red = row[3] if len(row) > 3 else False
        img[y0 : y0 + 12, 40:80] = 60  # 文字带 (深色)
        if checked:
            img[y0 : y0 + 12, 17:28] = 50  # 勾墨迹 (min<80)
        else:
            img[y0 : y0 + 12, 17:28] = 200  # 空框 (min>=80)
        if is_index:
            img[y0 : y0 + 12, 295:320] = (220, 40, 40)  # 市场列红 tag (中证)
        if up_red:
            img[y0 : y0 + 12, 380:420] = (220, 40, 40)  # 价格列红字 (涨)
    return img


def test_dlg_rows_detects_checked_and_unchecked():
    rows = mod._dlg_rows_from_img(_dlg_img([(20, True), (60, False)]))
    assert len(rows) == 2
    assert rows[0][1] is True
    assert rows[1][1] is False
    assert rows[0][2] is False
    assert abs(rows[0][0] - 26) <= 1
    assert abs(rows[1][0] - 66) <= 1


def test_dlg_rows_classifies_index_row_by_red_text():
    """指数行 = 市场列 x∈[285,331) 红字 ("中证" tag); 股票行灰"深A"无红 → False."""
    rows = mod._dlg_rows_from_img(
        _dlg_img([(20, True, False), (44, False, True), (68, True, False)])
    )
    assert [r[2] for r in rows] == [False, True, False]
    assert rows[1][1] is False  # 指数行未勾


def test_up_stock_red_price_not_index():
    """上涨股票行价格/涨幅红字 (x≥370) 不得判成指数行 — 09-05 实弹 002098
    +2.44% 红价触发旧整行红字判据, fail-closed 整批拒加 (误杀正常批)."""
    rows = mod._dlg_rows_from_img(_dlg_img([(20, True, False, True)]))
    assert len(rows) == 1
    assert rows[0][2] is False


def _patch_verify_ui(monkeypatch, tmp_path):
    """屏蔽核验循环的 UI 交互 (前台断言/点击/键盘) — 单测只验纯判定逻辑.

    失败现场 dump 重定向到 tmp_path — 曾把 10x10 零图 dump 进真实 tmp_t,
    覆盖丢失实弹 forensic 存档 idxchk1.png (09-05 事故)."""
    import uiautomation

    from scripts import _ths_ui as ui
    from scripts import _ths_watchlist_push as wmod

    calls = {"click": [], "keys": []}
    monkeypatch.setattr(ui, "assert_foreground_hexin", lambda *a, **k: None)
    monkeypatch.setattr(
        uiautomation, "Click", lambda x, y: calls["click"].append((x, y))
    )
    monkeypatch.setattr(uiautomation, "SendKeys", lambda s: calls["keys"].append(s))
    monkeypatch.setattr(wmod, "FAIL_DUMP_DIR", str(tmp_path))
    return calls


def _fake_dlg():
    import types

    return types.SimpleNamespace(
        BoundingRectangle=types.SimpleNamespace(left=0, top=0, right=500, bottom=200)
    )


def _patch_grab(monkeypatch):
    import numpy as np

    monkeypatch.setattr(
        "PIL.ImageGrab.grab", lambda bbox=None: np.zeros((10, 10, 3), dtype=np.uint8)
    )


def test_verify_full_checked_returns_map(monkeypatch, tmp_path):
    """全勾 → {code: True}; 无需补勾点击."""
    from scripts import _ths_watchlist_push as wmod

    _patch_grab(monkeypatch)
    monkeypatch.setattr(
        wmod, "_dlg_rows_from_img", lambda img: [(26, True, False), (58, True, False)]
    )
    calls = _patch_verify_ui(monkeypatch, tmp_path)
    res = wmod._ensure_dialog_rows_checked(_fake_dlg(), ["600001", "600002"])
    assert res == {"600001": True, "600002": True}
    assert calls["click"] == []
    assert calls["keys"] == []


def test_verify_keyboard_route_on_unchecked_partial_return(monkeypatch, tmp_path):
    """未勾行走键盘路线 (点行文字 + Space); 像素恒不翻 → 重查耗尽返回部分
    勾选现状 (调用方点加入先落袋, 09-05: 全勾等不到时部分勾也是净进展)."""
    from scripts import _ths_watchlist_push as wmod

    _patch_grab(monkeypatch)
    monkeypatch.setattr(
        wmod, "_dlg_rows_from_img", lambda img: [(26, True, False), (58, False, False)]
    )
    calls = _patch_verify_ui(monkeypatch, tmp_path)
    res = wmod._ensure_dialog_rows_checked(
        _fake_dlg(), ["600001", "600002"], max_rounds=2
    )
    assert res == {"600001": True, "600002": False}
    assert len(calls["click"]) >= 1
    assert "{Space}" in calls["keys"]


def test_verify_padded_batch_dead_position_success(monkeypatch, tmp_path):
    """垫批 [X, 000, None, Y] 死位路线: 指数行(第3位)未勾 + 股票行全勾 → 全码
    可入. 09-05 retry9 实弹: 该形状真实出现但被红字判据误判槽位错位拒加 —
    "中证" tag 实为深灰非红字, 位序即识别."""
    from scripts import _ths_watchlist_push as wmod

    _patch_grab(monkeypatch)
    monkeypatch.setattr(
        wmod,
        "_dlg_rows_from_img",
        lambda img: [
            (26, True, False),
            (58, True, False),
            (90, False, False),
            (122, True, False),
        ],
    )
    _patch_verify_ui(monkeypatch, tmp_path)
    res = wmod._ensure_dialog_rows_checked(
        _fake_dlg(), ["002098", "000985", None, "688693"]
    )
    assert res == {"002098": True, "000985": True, "688693": True}


def test_verify_padded_batch_index_slot_checked_fail_closed(monkeypatch, tmp_path):
    """垫批指数行 (None 槽位) 已勾 (无法取消) → None 不点加入."""
    from scripts import _ths_watchlist_push as wmod

    _patch_grab(monkeypatch)
    monkeypatch.setattr(
        wmod,
        "_dlg_rows_from_img",
        lambda img: [
            (26, True, False),
            (58, True, False),
            (90, True, False),
            (122, True, False),
        ],
    )
    _patch_verify_ui(monkeypatch, tmp_path)
    res = wmod._ensure_dialog_rows_checked(
        _fake_dlg(), ["600001", "000985", None, "600002"]
    )
    assert res is None


def test_verify_red_tag_on_stock_slot_checked_fail_closed(monkeypatch, tmp_path):
    """污染保险: 红 tag 诊断行落在股票槽位且已勾 (疑似指数行将随批入自选)
    → None. 红 tag 未勾只记诊断不拒 (加入只入勾选行, 无污染)."""
    from scripts import _ths_watchlist_push as wmod

    _patch_grab(monkeypatch)
    monkeypatch.setattr(wmod, "_dlg_rows_from_img", lambda img: [(26, True, True)])
    _patch_verify_ui(monkeypatch, tmp_path)
    assert wmod._ensure_dialog_rows_checked(_fake_dlg(), ["600001"]) is None


def test_verify_rowcount_mismatch_returns_none(monkeypatch, tmp_path):
    """行数核验不过 (识别未齐/杂行混入) → None."""
    from scripts import _ths_watchlist_push as wmod

    _patch_grab(monkeypatch)
    monkeypatch.setattr(wmod, "_dlg_rows_from_img", lambda img: [])
    _patch_verify_ui(monkeypatch, tmp_path)
    assert wmod._ensure_dialog_rows_checked(_fake_dlg(), ["600001"]) is None


def test_row_codes_pure_and_padded():
    """纯批行序=代码本身; 垫批撞码股后跟指数槽位 None."""
    assert mod._row_codes(["600001", "600002"]) == ["600001", "600002"]
    assert mod._row_codes(["600001", "000985", "600002"]) == [
        "600001",
        "000985",
        None,
        "600002",
    ]


def test_build_chunks_plain_no_collide():
    codes = [f"60{i:04d}" for i in range(25)]
    chunks = mod._build_chunks(codes)
    n_full, rem = divmod(len(codes), mod.CHUNK_SIZE)
    expect = [mod.CHUNK_SIZE] * n_full + ([rem] if rem else [])
    assert [len(c) for c in chunks] == expect
    assert sum(chunks, []) == codes


def test_build_chunks_colliding_padded_second_position():
    """撞码股单独垫批且在第 2 位: [X, 000, Y] → 行序 X/股/指/Y, 指数落第3位死位."""
    codes = ["600001", "000985", "600002"]
    chunks = mod._build_chunks(codes)
    assert chunks == [["600001", "600002"], ["600001", "000985", "600002"]]


def test_build_chunks_multiple_collide_each_own_batch():
    chunks = mod._build_chunks(["000985", "600001", "000001", "600002"])
    # 普通码一批; 每只撞码股各一垫批 (垫码复用普通码, 加自选幂等)
    assert chunks[0] == ["600001", "600002"]
    assert ["000985"] == [c for c in chunks[1] if mod.INDEX_COLLIDE_RE.match(c)]
    assert chunks[1][1] == "000985"
    assert ["000001"] == [c for c in chunks[2] if mod.INDEX_COLLIDE_RE.match(c)]
    assert chunks[2][1] == "000001"


def test_build_chunks_collide_without_fillers_solo_batch():
    """普通码不足 2 只 → 单码批 (指数行会勾上, 核验 fail-closed 跳过, 不入指数)."""
    assert mod._build_chunks(["000985"]) == [["000985"]]
    assert mod._build_chunks(["000985", "600001"]) == [["600001"], ["000985"]]


def test_dlg_rows_drops_sliver_bands():
    import numpy as np

    img = np.full((100, 500, 3), 240, dtype=np.uint8)
    img[30:37, 40:80] = 60  # 7px 碎带 <10px 下限 → 丢弃
    assert mod._dlg_rows_from_img(img) == []


def test_chunk_size_fits_visible_list():
    """批量上限 = 2 (粘贴自动勾选位置伪影: 每 4 行组第 3 位不自动勾, 09-03
    v8 实证 CHUNK=2 全落组内安全位 1/2; 批一大死位行勾不上, Space 也点不动)."""
    assert mod.CHUNK_SIZE <= 2


# ---------------- 缺码补推循环 + 推送结果单 (2026-09-05 用户拍板) ----------------


def _patch_push_verify(monkeypatch, verify, idle=True, dialog=True):
    """verify 注入 (_patch_push_ui 骨架 + 可调用核验结果)."""
    import ctypes
    import time as _time
    import types

    import uiautomation

    from scripts import _ths_ui as ui
    from scripts import _ths_watchlist_push as wmod

    calls = {"click": [], "verify": 0, "grid": set()}
    fake_dlg = types.SimpleNamespace(
        NativeWindowHandle=0,
        BoundingRectangle=types.SimpleNamespace(left=0, top=0, right=500, bottom=200),
    )
    monkeypatch.setattr(ui, "ensure_idle", lambda what="": idle)
    monkeypatch.setattr(ui, "ensure_watchlist_window", lambda: types.SimpleNamespace())
    monkeypatch.setattr(ui, "close_stray_windows", lambda: None)
    monkeypatch.setattr(ui, "find_window", lambda title: fake_dlg if dialog else None)
    monkeypatch.setattr(ui, "open_copy_recognition_dialog", lambda win: dialog)
    monkeypatch.setattr(ui, "close_x", lambda dlg: None)
    monkeypatch.setattr(ui, "assert_foreground_hexin", lambda *a, **k: None)
    monkeypatch.setattr(uiautomation, "SetClipboardText", lambda s: None)
    monkeypatch.setattr(
        uiautomation, "Click", lambda x, y: calls["click"].append((x, y))
    )
    monkeypatch.setattr(uiautomation, "SendKeys", lambda s: None)
    monkeypatch.setattr(ctypes.windll.user32, "SetWindowPos", lambda *a, **k: 1)
    monkeypatch.setattr(_time, "sleep", lambda s: None)

    def _verify(dlg, row_codes, log=print, max_rounds=4):
        calls["verify"] += 1
        res = verify(dlg, row_codes)
        # 模拟活会话: 核验勾选 + 点加入 → 网格出现该码
        calls["grid"].update(c for c, g in res.items() if c and g)
        return res

    monkeypatch.setattr(wmod, "_ensure_dialog_rows_checked", _verify)
    monkeypatch.setattr(
        ui, "read_all_codes", lambda win, log=print: sorted(calls["grid"])
    )
    return calls


def _run_push(tmp_path, codes):
    from scripts import _ths_watchlist_push as wmod

    txt = tmp_path / "ths_watchlist_20260905__09__parallel__M1.txt"
    txt.write_text("\n".join(codes) + "\n", encoding="utf-8")
    ok = wmod.push_via_ths(txt)
    return txt, ok


def _ledger(txt):
    import pandas as pd

    from scripts import _ths_watchlist_push as wmod

    fp = wmod._result_path(txt)
    assert fp.name == "ths_push_result_20260905__09__parallel__M1.csv"
    res = pd.read_csv(fp, dtype={"symbol": str})
    return dict(zip(res["symbol"], res["status"]))


def test_push_sweeps_until_complete(monkeypatch, tmp_path):
    """缺码循环补推 (2026-09-05 用户 "LOOP PUSHING TILL COMPLETE"): 第 1 轮
    600002 三次重贴都不勾, 第 2 轮只贴缺的并落袋 → 全落袋 ok=True."""
    state = {"n": 0}

    def verify(dlg, row_codes):
        state["n"] += 1
        codes = [c for c in row_codes if c is not None]
        if state["n"] <= 3:  # 第 1 轮: 3 次重贴 600002 都不勾
            return {c: c == "600001" for c in codes}
        return {c: True for c in codes}

    _patch_push_verify(monkeypatch, verify)
    txt, ok = _run_push(tmp_path, ["600001", "600002"])
    assert ok is True
    assert state["n"] == 4  # 轮1 ×3 + 轮2 ×1
    assert _ledger(txt) == {"600001": "landed", "600002": "landed"}


def test_push_sweep_cap_ten(monkeypatch, tmp_path):
    """10 轮上限 (2026-09-05 用户 "最多10轮"): 永不勾的码循环满 10 轮后停,
    不死循环, 结果单标 manual."""
    from scripts import _ths_watchlist_push as wmod

    def verify(dlg, row_codes):
        return {c: False for c in row_codes if c is not None}

    calls = _patch_push_verify(monkeypatch, verify)
    txt, ok = _run_push(tmp_path, ["600009"])
    assert ok is False
    assert calls["verify"] == wmod.PUSH_SWEEPS * wmod.RETRY_PASTE  # 10 轮×3 次
    assert _ledger(txt) == {"600009": "manual"}


def test_push_blocked_idle_gate_writes_blocked_ledger(monkeypatch, tmp_path):
    """空闲闸挡下 (用户在场) = 推送没开跑 → 全码 blocked, 不误报 manual."""

    def verify(dlg, row_codes):
        raise AssertionError("空闲闸挡下后不应有任何核验")

    _patch_push_verify(monkeypatch, verify, idle=False)
    txt, ok = _run_push(tmp_path, ["600001"])
    assert ok is False
    assert _ledger(txt) == {"600001": "blocked"}


def test_push_blocked_dialog_fail_writes_blocked_ledger(monkeypatch, tmp_path):
    """复制识别对话框从未出现 = 没开跑 → 全码 blocked."""

    def verify(dlg, row_codes):
        raise AssertionError("对话框未出现不应有核验")

    _patch_push_verify(monkeypatch, verify, dialog=False)
    txt, ok = _run_push(tmp_path, ["600001"])
    assert ok is False
    assert _ledger(txt) == {"600001": "blocked"}


def test_push_grid_truth_overrules_dialog_false_success(monkeypatch, tmp_path, capsys):
    """网格真值判词 (09-05 午前二次破案): 对话框全勾+加入已点, 但网格只见
    600001 → 600002 判 manual; 部分落袋=会话活, 不许报掉登录也不许假成功."""
    from scripts import _ths_ui as ui

    def verify(dlg, row_codes):
        return {c: True for c in row_codes if c is not None}

    _patch_push_verify(monkeypatch, verify)
    monkeypatch.setattr(ui, "read_all_codes", lambda win, log=print: ["600001"])
    txt, ok = _run_push(tmp_path, ["600001", "600002"])
    assert ok is False
    out = capsys.readouterr().out
    assert "已核验加入自选股" not in out
    assert "疑似掉登录" not in out
    assert "勾选判读假阳性" in out
    assert _ledger(txt) == {"600001": "landed", "600002": "manual"}


def test_push_grid_empty_with_dialog_landed_hints_relogin(
    monkeypatch, tmp_path, capsys
):
    """对话框全勾+加入已点但网格全空 = 加入整体无效果 → 疑似掉登录提示."""
    from scripts import _ths_ui as ui

    def verify(dlg, row_codes):
        return {c: True for c in row_codes if c is not None}

    _patch_push_verify(monkeypatch, verify)
    monkeypatch.setattr(ui, "read_all_codes", lambda win, log=print: [])
    txt, ok = _run_push(tmp_path, ["600001"])
    assert ok is False
    assert "疑似掉登录" in capsys.readouterr().out
    assert _ledger(txt) == {"600001": "manual"}


def test_push_grid_read_failure_fail_closed(monkeypatch, tmp_path, capsys):
    """网格读失败 (如用户回座打断) = 核验未完成 → fail-closed, 全码 manual,
    绝不输出成功判词."""
    from scripts import _ths_ui as ui

    def verify(dlg, row_codes):
        return {c: True for c in row_codes if c is not None}

    _patch_push_verify(monkeypatch, verify)

    def _boom(win, log=print):
        raise RuntimeError("用户回座")

    monkeypatch.setattr(ui, "read_all_codes", _boom)
    txt, ok = _run_push(tmp_path, ["600001"])
    assert ok is False
    assert "已核验加入自选股" not in capsys.readouterr().out
    assert _ledger(txt) == {"600001": "manual"}


def test_read_push_results_newest_per_source(tmp_path):
    """同源同日多份 (小时戳 WORM) 只取 mtime 最新; txt 名两代 (带/不带小时)
    的源名都能解析; 不同源互不掺和."""
    t1 = tmp_path / "ths_watchlist_20260905__08__parallel__M1.txt"
    t2 = tmp_path / "ths_watchlist_20260905__09__parallel__M1.txt"
    t3 = tmp_path / "ths_watchlist_20260905__09__legacy__M2.txt"
    t4 = tmp_path / "ths_watchlist_20260905__prob10dens.txt"
    mod.write_push_result(t1, ["600001"], ["600001"])  # 旧份: 全落袋
    mod.write_push_result(t2, ["600001", "600002"], ["600001"])  # 新份: 缺 1
    mod.write_push_result(t3, ["603829"], [])
    mod.write_push_result(t4, ["002098"], ["002098"])
    for fp, mt in ((mod._result_path(t1), 1000), (mod._result_path(t2), 2000)):
        os.utime(fp, (mt, mt))

    res = mod.read_push_results("20260905", tmp_path)
    par = dict(zip(*[res[res.source == "parallel"][c] for c in ("symbol", "status")]))
    assert par == {"600001": "landed", "600002": "manual"}  # 新份赢, 旧份不掺
    leg = dict(zip(*[res[res.source == "legacy"][c] for c in ("symbol", "status")]))
    assert leg == {"603829": "manual"}
    dens = dict(
        zip(*[res[res.source == "prob10dens"][c] for c in ("symbol", "status")])
    )
    assert dens == {"002098": "landed"}


def test_read_push_results_latest_date_default(tmp_path):
    """date 缺省 = 有结果单的最新日期; 无结果单返回空帧不炸."""
    empty = mod.read_push_results(list_dir=tmp_path)
    assert empty.empty
    assert list(empty.columns) == ["source", "symbol", "status"]

    t = tmp_path / "ths_watchlist_20260904__legacy__M0.txt"
    mod.write_push_result(t, ["600001"], ["600001"])
    res = mod.read_push_results(list_dir=tmp_path)
    assert res["source"].tolist() == ["legacy"]
