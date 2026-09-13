# -*- coding: utf-8 -*-
"""PARALLEL 概率头 schema 漂移自愈 — 单测.

[0912 夜] 0910 批 bundle 引用 ths_* 5 列, 0912 面板重建已删列 → serving
predict() 直接 raise, 而新鲜度判据 (age < 21 skip) 看不见 → 训练入口按
feat_cols ⊄ 面板列 强制重训 (无视年龄)。
"""

from __future__ import annotations

import pandas as pd

from scripts._train_parallel_prob_head import _missing_feat_cols


def test_none_bundle_no_missing():
    assert _missing_feat_cols(None, ["a"]) == []


def test_missing_cols_detected():
    frame = pd.DataFrame(columns=["a", "c"])
    assert _missing_feat_cols({"feat_cols": ["a", "b"]}, frame.columns) == ["b"]


def test_all_present_empty():
    frame = pd.DataFrame(columns=["a", "b", "c"])
    assert _missing_feat_cols({"feat_cols": ["a", "c"]}, frame.columns) == []


def test_bundle_without_feat_cols_key():
    assert _missing_feat_cols({}, ["a"]) == []
