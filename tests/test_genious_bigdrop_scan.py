# -*- coding: utf-8 -*-
"""GENIOUS 交付表 BIGDROP SCAN 标注列 (0915 用户令)。

守的是**语义**不是覆盖率: 压成一个布尔 / 顺序写反, 都会把零方向的规则支标成
大跌风险, 而这个列会被当"次日要跌"读 —— 正是 bigdrop 0915 拆分支要修掉的误读。
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from scripts import bigdrop_check as bc  # noqa: E402
from scripts._genious_excel import (  # noqa: E402
    BIGDROP_HIGH,
    BIGDROP_NONE,
    BIGDROP_VOL,
    _bigdrop_cells,
    _bigdrop_labels,
    _bigdrop_module_run,
    _norm_sym,
)

TH = 0.10
BASE = 0.05  # OOS 市场基准大跌率


def test_model_branch_is_high_risk():
    lab = _bigdrop_labels(np.array([0]), np.array([0.25]), TH)
    assert lab[0] == BIGDROP_HIGH


def test_rule_branch_is_volatility_not_high_risk():
    """规则举手但模型未达线 = 只有波动/跌停风险, 不带方向。"""
    lab = _bigdrop_labels(np.array([3]), np.array([0.02]), TH)
    assert lab[0] == BIGDROP_VOL


def test_silent_is_no_risk_not_blank():
    """都没举手 → **无风险**, 不是空 (用户 0916 令: 所有 STOCK 都应该有标志)。

    空值看着像"没数据"; 模块对每只票都给了读数, 空会把"算过且安全"与"根本没算"
    混成同一个样子。三态全出后, 列里再出现空就只剩"没评上分"一个意思。
    """
    lab = _bigdrop_labels(np.array([0]), np.array([0.001]), TH)
    assert lab[0] == BIGDROP_NONE
    assert lab[0] != ""


def test_no_state_is_ever_blank():
    """任何输入组合都不产生空 —— 这一列是完整判读, 不允许留白。"""
    sc = np.array([0, 1, 5, 0, 3, 0])
    p = np.array([0.0, 0.0, 0.9, 0.05, 0.01, TH])
    lab = _bigdrop_labels(sc, p, TH)
    assert all(x != "" for x in lab)
    assert set(lab) <= {BIGDROP_HIGH, BIGDROP_VOL, BIGDROP_NONE}


def test_model_wins_over_rule_when_both_fire():
    """两支同时命中 → 模型支优先, 不能退化成波动标。"""
    lab = _bigdrop_labels(np.array([5]), np.array([0.30]), TH)
    assert lab[0] == BIGDROP_HIGH


def test_threshold_boundary_is_inclusive():
    assert _bigdrop_labels(np.array([0]), np.array([TH]), TH)[0] == BIGDROP_HIGH
    assert _bigdrop_labels(np.array([0]), np.array([TH - 1e-9]), TH)[0] == BIGDROP_NONE


def test_rule_boundary_one_point_fires():
    """规则分 1 即入波动档 (服务口径 score>=1), 0 分不入。"""
    assert _bigdrop_labels(np.array([1]), np.array([0.0]), TH)[0] == BIGDROP_VOL
    assert _bigdrop_labels(np.array([0]), np.array([0.0]), TH)[0] == BIGDROP_NONE


def test_labels_align_positionally():
    """向量化标注按位对齐, 不能串行。"""
    sc = np.array([0, 3, 0, 2])
    p = np.array([0.99, 0.01, 0.001, 0.50])
    lab = _bigdrop_labels(sc, p, TH)
    assert list(lab) == [BIGDROP_HIGH, BIGDROP_VOL, BIGDROP_NONE, BIGDROP_HIGH]


def test_cells_carry_per_stock_multiple():
    """倍数 = 模型概率 ÷ 基准, 逐股不同 —— 同档内要能分出轻重, 不是分支常数。"""
    lab = np.array([BIGDROP_HIGH] * 2, dtype=object)
    cells = _bigdrop_cells(lab, np.array([0.25, 0.125]), BASE)
    assert cells == [f"{BIGDROP_HIGH} 5.0x", f"{BIGDROP_HIGH} 2.5x"]


def test_only_directional_branch_carries_multiple():
    """倍数只挂方向支。波动风险 的模型概率**按定义**低于报警线, 倍数恒 <1.95x,

    挂上去就等于在"风险"标签旁边写"比市场安全" —— 标签与数字自相矛盾 (用户 0916
    报的 "0.2/0.3x 不正常" 正是这个)。
    """
    lab = np.array([BIGDROP_HIGH, BIGDROP_VOL, BIGDROP_NONE], dtype=object)
    cells = _bigdrop_cells(lab, np.array([0.25, 0.10, 0.90]), BASE)
    assert cells == [f"{BIGDROP_HIGH} 5.0x", BIGDROP_VOL, BIGDROP_NONE]


def test_vol_branch_multiple_would_be_below_alarm_ratio():
    """构造性上界: 波动风险 的 p < th ⇒ 倍数 < th/base。数字被拿掉的理由是数学, 不是口味。"""
    th, base = 0.10, 0.05
    assert th / base == 2.0
    lab = _bigdrop_labels(np.array([1]), np.array([th - 1e-9]), th)
    assert lab[0] == BIGDROP_VOL
    assert _bigdrop_cells(lab, np.array([th - 1e-9]), base) == [BIGDROP_VOL]


def test_norm_sym_strips_exchange_suffix_and_zfills():
    """清单侧北交所票带 `.BJ`, 面板键是裸 6 位 —— 不归一则查表静默落空。

    落空的票显示成空, 与真判读长得一样, 但它的分**是算过的**, 被 key 格式丢掉了
    (实测 920367.BJ → 空 / 920367 → 波动风险)。三态落地后, 列里留空只剩
    "没评上分"一个意思 —— 更不能让 key 格式制造假空。
    """
    assert _norm_sym("920075.BJ") == "920075"
    assert _norm_sym("920075") == "920075"
    assert _norm_sym(" 1 ") == "000001"
    assert _norm_sym("600000.SH") == "600000"
    assert _norm_sym(1) == "000001"


def test_cells_align_positionally():
    """标注与倍数必须取自同一下标, 错位会把 A 的倍数挂到 B 上。"""
    sc = np.array([0, 3, 0, 2])
    p = np.array([0.99, 0.01, 0.001, 0.50])
    cells = _bigdrop_cells(_bigdrop_labels(sc, p, TH), p, BASE)
    assert cells == [
        f"{BIGDROP_HIGH} 19.8x",
        BIGDROP_VOL,
        BIGDROP_NONE,
        f"{BIGDROP_HIGH} 10.0x",
    ]


# ── 包新鲜度 (0916: 交付链 = genious 运行 + bigdrop 运行, 顺序执行) ────────────────


def test_bundle_fresh_when_data_date_matches():
    assert not bc.bundle_is_stale(
        {"tag": "20260916", "data_date": "20260916"}, "20260916"
    )


def test_bundle_stale_when_data_date_differs():
    assert bc.bundle_is_stale({"tag": "20260915", "data_date": "20260915"}, "20260916")


def test_bundle_stale_when_data_date_missing():
    """旧包没有 data_date —— 证不了同源就不假定同源 (否则永远不重建)。"""
    assert bc.bundle_is_stale({"tag": "20260916"}, "20260916")


def test_staleness_keys_on_data_date_not_wallclock_tag():
    """tag=今天但数据只到昨天 = 早于当日抓取建的包, 必须判陈旧 (密度页同类事故)。"""
    assert bc.bundle_is_stale({"tag": "20260916", "data_date": "20260915"}, "20260916")


# ── 模块运行: 非致命 + 同源零成本 ────────────────────────────────────────────────


NEW_BUNDLE = {"tag": "20260916", "data_date": "20260916", "oos_base": 0.0513}


class _Ctx:
    """模块运行的替身环境: 假包文件 + 假面板日期 + 可观测的 build 调用。"""

    def __init__(self, monkeypatch, tmp_path, bundle: dict, pmax: bool = True):
        self.calls = []
        monkeypatch.setattr(bc, "BUNDLE_DIR", tmp_path)
        (tmp_path / "bundle_latest.joblib").write_bytes(b"x")
        monkeypatch.setattr(bc, "load_bundle", lambda: bundle)
        monkeypatch.setattr(
            bc, "build", lambda *a, **k: (self.calls.append(1), NEW_BUNDLE)[1]
        )
        monkeypatch.setattr(
            "scripts._genious_excel.kt.panel_max_date",
            lambda path: datetime.date(2026, 9, 16) if pmax else None,
        )


def test_module_run_rebuilds_when_stale(monkeypatch, tmp_path):
    ctx = _Ctx(monkeypatch, tmp_path, {"tag": "20260915", "data_date": "20260915"})
    assert _bigdrop_module_run() is True
    assert ctx.calls == [1], "陈旧包必须触发建包"


def test_module_run_skips_when_fresh(monkeypatch, tmp_path):
    ctx = _Ctx(monkeypatch, tmp_path, {"tag": "20260916", "data_date": "20260916"})
    assert _bigdrop_module_run() is True
    assert ctx.calls == [], "同源包不该付建包代价"


def test_module_run_is_non_fatal_on_build_failure(monkeypatch, tmp_path):
    """建包炸了也不许掀翻交付链 —— 回退盘上旧包, 照常出清单。"""
    _Ctx(monkeypatch, tmp_path, {"tag": "20260915", "data_date": "20260915"})

    def boom(*a, **k):
        raise RuntimeError("训练炸了")

    monkeypatch.setattr(bc, "build", boom)
    assert _bigdrop_module_run() is False


def test_module_run_is_non_fatal_when_bundle_missing(monkeypatch, tmp_path):
    """缺包时 load_bundle() 会 sys.exit (BaseException) —— 必须先判存在性。"""
    monkeypatch.setattr(bc, "BUNDLE_DIR", tmp_path)
    monkeypatch.setattr(bc, "build", lambda *a, **k: pytest.fail("缺包时不该尝试建包"))
    assert _bigdrop_module_run() is False


def test_module_run_gives_up_when_panel_date_unreadable(monkeypatch, tmp_path):
    """面板 date 列读不出来 → 判不了新鲜度, 不猜、不盲建。"""
    ctx = _Ctx(monkeypatch, tmp_path, {"tag": "20260915"}, pmax=None)
    assert _bigdrop_module_run() is False
    assert ctx.calls == []
