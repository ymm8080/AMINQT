# -*- coding: utf-8 -*-
"""Tests for PARALLEL per-head 特征集 (PARALLEL_HEAD_EXTRA_COLS, 2026-09-12).

8格矩阵 PARALLEL 侧: mag 头 extras = spec.pool 追加列 (pool_score 截面分位),
prob 头 extras = 面板外合成列 (feature_cols 宽集自动收录)。铁律: train/serve
同源合成 (synthesize_prob_extras), serving 缺列 predict raise (schema 漂移
fail-loud, 不静默 0 填充)。

覆盖: effective_pool (空配置不变/追加去重/未知板/None) / synthesize_prob_extras
(quality_factor 合成/已存在跳过/不可合成告警/空配置 no-op) / 端到端 (train_bundle
以合成列入 feat_cols → 原始截面 predict raise → 合成后 predict 通过)。
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from app.pipeline_parallel import prob_head
from app.pipeline_parallel.config import FUSION, SNIPER, effective_pool


# ── effective_pool ────────────────────────────────────────────
def test_effective_pool_default_empty_unchanged():
    # 提交默认全空 → 有效池 = 原 pool 逐位一致 (零行为变化)
    assert effective_pool(SNIPER, "main") == SNIPER.pool
    assert effective_pool(FUSION, "dual") == FUSION.pool


def test_effective_pool_none_or_unknown_board_unchanged():
    assert effective_pool(SNIPER, None) == SNIPER.pool
    assert effective_pool(SNIPER, "no_such_board") == SNIPER.pool


def test_effective_pool_appends_extras_dedup(monkeypatch):
    monkeypatch.setattr(
        "config.settings.PARALLEL_HEAD_EXTRA_COLS",
        {"main": {"mag": ["VAR51", "new_col"], "prob": []}},
    )
    got = effective_pool(SNIPER, "main")
    # VAR51 已在 pool → 不重复; new_col 追加在尾部
    assert got == tuple(SNIPER.pool) + ("new_col",)
    # 其它板不受影响
    assert effective_pool(SNIPER, "dual") == SNIPER.pool


# ── synthesize_prob_extras ────────────────────────────────────
def test_synthesize_prob_extras_default_noop():
    df = pd.DataFrame({"date": ["d1"], "roe": [1.0]})
    out = prob_head.synthesize_prob_extras(df, "main")
    assert "quality_factor" not in out.columns  # 默认空配置 → 不合成


def test_synthesize_quality_factor(monkeypatch):
    monkeypatch.setattr(
        "config.settings.PARALLEL_HEAD_EXTRA_COLS",
        {"main": {"mag": [], "prob": ["quality_factor"]}},
    )
    df = pd.DataFrame(
        {
            "date": ["d1", "d1", "d1"],
            "roe": [1.0, 2.0, 3.0],
            "eps": [3.0, 2.0, 1.0],
        }
    )
    out = prob_head.synthesize_prob_extras(df, "main")
    assert "quality_factor" in out.columns
    # roe/eps 逆序 → 每股两基列分位互补, 均值恒 2/3
    assert out["quality_factor"].to_numpy(dtype=float) == pytest.approx(
        [2 / 3, 2 / 3, 2 / 3]
    )


def test_synthesize_skips_existing_column(monkeypatch):
    monkeypatch.setattr(
        "config.settings.PARALLEL_HEAD_EXTRA_COLS",
        {"main": {"mag": [], "prob": ["quality_factor"]}},
    )
    df = pd.DataFrame(
        {"date": ["d1"], "roe": [1.0], "eps": [1.0], "quality_factor": [0.42]}
    )
    out = prob_head.synthesize_prob_extras(df, "main")
    assert out["quality_factor"].tolist() == [0.42]  # 已有 → 原值不动


def test_synthesize_unsynthesizable_warns(monkeypatch, capsys):
    monkeypatch.setattr(
        "config.settings.PARALLEL_HEAD_EXTRA_COLS",
        {"main": {"mag": [], "prob": ["not_a_synthesizable_col"]}},
    )
    df = pd.DataFrame({"date": ["d1"], "f1": [1.0]})
    out = prob_head.synthesize_prob_extras(df, "main")
    assert "not_a_synthesizable_col" not in out.columns  # 保持缺失
    assert "无法合成" in capsys.readouterr().out  # 大声告警


# ── 端到端: train_bundle 合成列入 feat_cols → predict schema 契约 ──
def _mk_panel(n_dates: int = 900, n_syms: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-02", periods=n_dates)
    rows = []
    for si in range(n_syms):
        close = 100.0 * np.cumprod(1.0 + 0.01 * (rng.random(n_dates) - 0.5))
        for di in range(n_dates):
            c = close[di]
            rows.append(
                {
                    "symbol": f"60000{si}",
                    "date": dates[di],
                    "close_hfq": c,
                    "high_hfq": c * 1.02,
                    "adv20": 1e7 + si,
                    "f1": float((si * 13 + di * 7) % 97) / 97.0,
                    "f2": float((si * 29 + di * 3) % 89) / 89.0,
                    "roe": float(si % 3) + 0.1 * (di % 5),
                    "eps": float((si + di) % 4) + 0.5,
                    "label_pain": float((si + di) % 2),
                }
            )
    df = pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)
    df["symbol"] = df["symbol"].astype(str)
    return df


def test_end_to_end_train_bundle_predict_schema(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "config.settings.PARALLEL_HEAD_EXTRA_COLS",
        {"main": {"mag": [], "prob": ["quality_factor"]}},
    )
    monkeypatch.setitem(prob_head.PROB_GATE, "model_dir", str(tmp_path))
    t = _mk_panel()
    # 训练侧与 _train_parallel_prob_head._load_board 同序: 先合成再算 mfe_3d
    t = prob_head.synthesize_prob_extras(t, "main")
    assert "quality_factor" in t.columns
    t = prob_head._add_mfe_3d(t)

    path = prob_head.train_bundle("main", t, str(t["date"].max().date()), None)
    bundle = joblib.load(path)
    assert "quality_factor" in bundle["feat_cols"]  # 合成列已入宽集

    # serving 原始截面 (未合成) → predict raise (schema 漂移 fail-loud)
    last = t["date"].max()
    cs = t[t["date"] == last].drop(columns=["mfe_3d", "label_pain", "quality_factor"])
    with pytest.raises(ValueError, match="schema 漂移"):
        prob_head.predict(bundle, cs)

    # 合成后 (gate_probabilities 同路径) → 正常出概率
    cs = prob_head.synthesize_prob_extras(cs, "main")
    pred = prob_head.predict(bundle, cs)
    assert len(pred) == len(cs)
    assert np.isfinite(pred.to_numpy(dtype=float)).all()
