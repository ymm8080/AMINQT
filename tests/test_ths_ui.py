"""_ths_ui 纯函数测试: 高亮判色 / 数字字形切分与匹配 / 空闲闸边界.

不触碰真实窗口 (find_window 等带 UIA 依赖的函数不在此测).
读码器真值验证 (19 行已知码复现) 在活体验证脚本完成, 这里测纯函数.
"""

from unittest import mock

import numpy as np

from scripts import _ths_ui


def _img_with_bands(height=400, width=100, bands=(), base=(45, 45, 45)):
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:, :] = base
    for y0, y1 in bands:
        img[y0:y1, :] = (64, 0, 128)
    return img


class TestHighlightColor:
    def test_calibration_color_hit(self):
        # 校准色 RGB(64,0,128) 必须命中; 未选中灰底 RGB(45,45,45) 必须不命中
        img = np.full((20, 10, 3), 45, dtype=np.uint8)
        img[5:15] = (64, 0, 128)
        hit = _ths_ui.is_highlight_color(img)
        assert hit[5:15].all()
        assert not hit[:5].any()


class TestSplitDigitCells:
    def test_six_separated_digits(self):
        seg = np.zeros((12, 60), dtype=bool)
        for d in range(6):
            seg[2:10, d * 10 + 1 : d * 10 + 8] = True
        cells = _ths_ui._split_digit_cells(seg)
        assert len(cells) == 6

    def test_touching_wide_group_split(self):
        # 两字触连成宽组 (真实场景: 6 组里 1 组宽, 中位宽不被污染) → 均分成 2 格
        seg = np.zeros((12, 100), dtype=bool)
        seg[2:10, 1:19] = True  # 宽 18 = 两个 9px 字触连
        for d in range(5):
            seg[2:10, 30 + d * 12 : 38 + d * 12] = True  # 5 个孤字宽 8
        cells = _ths_ui._split_digit_cells(seg)
        assert len(cells) == 7  # 宽组拆 2 + 孤字 5

    def test_sliver_group_dropped(self):
        # <3px 碎片组 (裁剪边缘残留) 必须丢弃, 不占数字格
        seg = np.zeros((12, 60), dtype=bool)
        seg[2:10, 0:1] = True  # 1px 碎片
        for d in range(6):
            seg[2:10, 5 + d * 9 : 12 + d * 9] = True
        cells = _ths_ui._split_digit_cells(seg)
        assert len(cells) == 6

    def test_empty_returns_empty(self):
        seg = np.zeros((12, 60), dtype=bool)
        assert _ths_ui._split_digit_cells(seg) == []


class TestDigitMatch:
    def test_template_self_match(self):
        feats, labels = _ths_ui._digit_templates()
        i = 7
        lab, dist = _ths_ui._match_digit(feats[i])
        assert lab == int(labels[i])
        assert dist < 0.01

    def test_all_ten_classes_present(self):
        _feats, labels = _ths_ui._digit_templates()
        assert set(np.unique(labels).tolist()) == set(range(10))

    def test_norm_glyph_shape(self):
        g = np.zeros((13, 8), dtype=bool)
        g[2:11, 1:7] = True
        f = _ths_ui._norm_glyph(g)
        assert f.shape == (16, 12)
        assert f.dtype == np.float32


class TestEnsureIdle:
    def test_user_active_returns_false(self):
        with mock.patch.object(_ths_ui, "user_idle_seconds", return_value=30.0):
            assert _ths_ui.ensure_idle(what="测试") is False

    def test_idle_enough_returns_true(self):
        with mock.patch.object(_ths_ui, "user_idle_seconds", return_value=120.0):
            assert _ths_ui.ensure_idle(what="测试") is True

    def test_boundary_exactly_min_passes(self):
        with mock.patch.object(_ths_ui, "user_idle_seconds", return_value=90.0):
            assert _ths_ui.ensure_idle(what="测试") is True

    def test_custom_min_used(self):
        with mock.patch.object(_ths_ui, "user_idle_seconds", return_value=45.0) as m:
            _ths_ui.ensure_idle(min_idle_s=30.0, what="测试")
            assert m.call_count == 1


def test_user_idle_seconds_smoke():
    idle = _ths_ui.user_idle_seconds()
    assert idle >= 0.0


def test_foreground_pid_smoke():
    pid = _ths_ui.foreground_pid()
    assert isinstance(pid, int)
    assert pid >= 0


def test_ths_hexin_path_default():
    # 环境变量缺省时落到默认安装路径 (单一来源: push 模块从这再导出)
    from scripts import _ths_watchlist_push as push_mod

    assert push_mod.THS_HEXIN_PATH == _ths_ui.THS_HEXIN_PATH
    assert _ths_ui.THS_HEXIN_PATH.name == "hexin.exe"


class TestCandidateConstrainedMatch:
    """约束匹配判词 (09-05 接产线): 判词只在候选码里挑, 治自由 OCR 8/0 形歧义.
    校准 = tmp_t/_ths_cand_calib_0905.py 19 行真值 19/19 分离."""

    def test_constants_locked(self):
        assert _ths_ui._CAND_BIN_THR == 110
        assert _ths_ui._CAND_SUM_MAX == 0.9
        assert _ths_ui._CAND_CELL_MAX == 0.17

    def _tpl_feats(self, code):
        feats, labels = _ths_ui._digit_templates()
        out = []
        for ch in code:
            m = np.where(labels == int(ch))[0][0]
            out.append(feats[m])
        return out

    def test_exact_template_match_wins(self):
        row = self._tpl_feats("123456")
        code, mx = _ths_ui._match_candidate(row, ["654321", "123456"])
        assert code == "123456"
        assert mx < 0.05

    def test_sum_pass_but_max_cell_rejects(self, monkeypatch):
        # 形近假码维度: 行真值 600000, 候选 600008 → 5 位贴 0.1 + 1 位 0.2
        # → 和 0.7 过 0.9 但单位 0.2 超 0.17 → 必须拒 (校准: 002868→002098
        # sum 0.34 即此形态, 靠 max_cell 杀)
        table = {("0", "8"): 0.2}

        def fake_dist(feat, d):
            return table.get((feat["ch"], str(d)), 0.1)

        monkeypatch.setattr(_ths_ui, "_digit_dist", fake_dist)
        row = [{"ch": ch} for ch in "600000"]
        code, mx = _ths_ui._match_candidate(row, ["600008"])
        assert code is None and mx == 1.0
        # 对照: 真码 0.1×6=0.6 全过
        code2, mx2 = _ths_ui._match_candidate(row, ["600000"])
        assert code2 == "600000" and mx2 == 0.1

    def test_read_rows_integration_template_glyph(self):
        # 合成整行: 把 6 个数字模板字形画进灰底画布 → 约束路径应读回候选码;
        # 候选不含真码 → None (宁缺勿错)
        feats, labels = _ths_ui._digit_templates()
        glyph_h, glyph_w = feats[0].shape
        canvas = np.zeros((40, 6 * glyph_w + 5 * 4 + 8), dtype=np.uint8)
        x = 4
        for ch in "123456":
            g = feats[np.where(labels == int(ch))[0][0]].astype(np.uint8) * 255
            canvas[12 : 12 + glyph_h, x : x + glyph_w] = g
            x += glyph_w + 4
        rows = _ths_ui._read_rows_from_gray(canvas, 0, candidates=["123456"])
        assert len(rows) == 1
        y0, y1, code, conf = rows[0]
        assert code == "123456"
        assert conf <= _ths_ui._CAND_CELL_MAX
        # 候选不含真码 → 拒
        rows2 = _ths_ui._read_rows_from_gray(canvas, 0, candidates=["654321"])
        assert rows2[0][2] is None
