"""同花顺自选股推送单测: 当日 TOP10 → 导入 txt (2026-09-02 用户口径: 双模块都推).

- collect_codes: parallel 短名单 rank 前 10 + legacy 清单序前 10 并集, parallel 序
  在前逐码去重; 单侧缺失退化为另一侧前 10
- 代码去重保序, 非法代码过滤 (非 6 位数字)
- REJECTED / gateinfo / preds_raw 等旁支文件不混入
- ths_txt_path / write_ths_txt: 每行一个 6 位代码, WORM 命名含日期+module
"""

import pandas as pd
import pytest

from scripts import _ths_watchlist_push as mod


def _write_shortlist(path, rank_symbols):
    pd.DataFrame(
        [{"symbol": s, "rank": r} for r, s in enumerate(rank_symbols, start=1)]
    ).to_csv(path, index=False)


def _write_legacy(path, symbols):
    pd.DataFrame([{"symbol": s} for s in symbols]).to_csv(path, index=False)


def test_collect_codes_shortlist_top10_by_rank(tmp_path):
    rank_symbols = [f"6000{i:02d}" for i in range(1, 21)]  # rank 1..20
    _write_shortlist(tmp_path / "parallel_shortlist_20260901__M1.csv", rank_symbols)
    module, codes = mod.collect_codes("20260901", tmp_path)
    assert module == "M1"
    assert codes == rank_symbols[:10]


def test_collect_codes_union_both_modules_parallel_first(tmp_path):
    """双源并集 (2026-09-02): parallel 前 legacy 后, 重复码只留先出现的 parallel 侧."""
    _write_shortlist(
        tmp_path / "parallel_shortlist_20260901__M1.csv", ["600001", "600002", "600003"]
    )
    _write_legacy(
        tmp_path / "legacy_stocklist_20260901__M1.csv", ["600003", "603829", "002968"]
    )
    module, codes = mod.collect_codes("20260901", tmp_path)
    assert module == "M1"
    assert codes == ["600001", "600002", "600003", "603829", "002968"]


def test_collect_codes_fallback_legacy_top10(tmp_path):
    _write_legacy(
        tmp_path / "legacy_stocklist_20260901__M1.csv",
        ["002968", "603829", "603326", "600706", "001217"],
    )
    module, codes = mod.collect_codes("20260901", tmp_path)
    assert module == "M1"
    assert codes == ["002968", "603829", "603326", "600706", "001217"]


def test_collect_codes_filters_bad_codes_and_dedups(tmp_path):
    _write_shortlist(
        tmp_path / "parallel_shortlist_20260901__M1.csv",
        ["002968", "nan", "2968", "600abc", "603829.0", "002968", "603829"],
    )
    _, codes = mod.collect_codes("20260901", tmp_path)
    assert codes == ["002968", "603829"]


def test_collect_codes_ignores_sidecar_files(tmp_path):
    _write_shortlist(tmp_path / "parallel_shortlist_20260901__M1.csv", ["002968"])
    _write_legacy(tmp_path / "legacy_stocklist_20260901__M1.csv", ["603829"])
    # 旁支文件: REJECTED / gateinfo / preds_raw / 其他日期 — 均不得混入
    _write_legacy(tmp_path / "legacy_gateinfo_20260901__M1.csv", ["999999"])
    _write_legacy(tmp_path / "parallel_preds_raw_20260901__M1.csv", ["888888"])
    _write_shortlist(tmp_path / "parallel_shortlist_20260831__M0.csv", ["666666"])
    _, codes = mod.collect_codes("20260901", tmp_path)
    assert codes == ["002968", "603829"]


def test_collect_codes_no_delivered_raises(tmp_path):
    with pytest.raises(SystemExit):
        mod.collect_codes("20260901", tmp_path)


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
    """合成对话框列表区截图: rows = [(y0, checked)]; 白底, 文字带 x40-80, 勾框 x17-28."""
    import numpy as np

    img = np.full((100, 500, 3), 240, dtype=np.uint8)
    for y0, checked in rows:
        img[y0 : y0 + 12, 40:80] = 60  # 文字带 (深色)
        if checked:
            img[y0 : y0 + 12, 17:28] = 50  # 勾墨迹 (min<80)
        else:
            img[y0 : y0 + 12, 17:28] = 200  # 空框 (min>=80)
    return img


def test_dlg_rows_detects_checked_and_unchecked():
    rows = mod._dlg_rows_from_img(_dlg_img([(20, True), (60, False)]))
    assert len(rows) == 2
    assert rows[0][1] is True
    assert rows[1][1] is False
    assert abs(rows[0][0] - 26) <= 1
    assert abs(rows[1][0] - 66) <= 1


def test_dlg_rows_drops_sliver_bands():
    import numpy as np

    img = np.full((100, 500, 3), 240, dtype=np.uint8)
    img[30:37, 40:80] = 60  # 7px 碎带 <10px 下限 → 丢弃
    assert mod._dlg_rows_from_img(img) == []


def test_chunk_size_fits_visible_list():
    """对话框列表可视 ~16 行, 批量上限必须留裕量防滚动截断核验."""
    assert mod.CHUNK_SIZE <= 14
