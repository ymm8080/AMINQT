# -*- coding: utf-8 -*-
"""PARALLEL 排名键自适应 (app.pipeline_parallel.rank_source) — 纯函数单测.

[0912 夜用户令] parallel 也要自动选更好的一头出预测: 每次概率头重训后评估
mag(幅度头)/prob(概率头)/blend 三键 (3d/5d/10d 逐视界, 用户令非 10d 单列),
argmax 加权 IC 自选 (平局→blend); serving 按最新 WORM json 换板级排序键;
json 缺失/过旧/异常 → blend (2026-08-15 A/B 定案现状, fail-open).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.pipeline_parallel import rank_source


def _eval_frame(n_dates=3, n_syms=6) -> pd.DataFrame:
    """prob 与标签完全同序, mag 与标签完全反序 → 三键可分胜负."""
    rows = []
    for d in range(n_dates):
        for i in range(n_syms):
            rows.append(
                {
                    "symbol": f"s{i:03d}",
                    "date": pd.Timestamp("2026-09-01") + pd.Timedelta(days=d),
                    "mag": 0.10 - 0.01 * i,  # 反序
                    "prob": 0.30 + 0.05 * i,  # 正序
                    **{f"label_pm_{h}_net": 0.01 * i for h in ("3d", "5d", "10d")},
                }
            )
    return pd.DataFrame(rows)


def test_daily_rank_metrics_perfect_and_inverse():
    df = _eval_frame()
    good = rank_source.daily_rank_metrics(df, "prob", "label_pm_10d_net", top_n=2)
    assert good["ic"] == pytest.approx(1.0)
    assert good["n_days"] == 3
    bad = rank_source.daily_rank_metrics(df, "mag", "label_pm_10d_net", top_n=2)
    assert bad["ic"] == pytest.approx(-1.0)
    # top-2 by prob = s05/s04 → 标签均值 (0.05+0.04)/2
    assert good["top10_net"] == pytest.approx(0.045)


def test_daily_rank_metrics_all_nan_key_returns_empty():
    df = _eval_frame()
    df["prob"] = float("nan")
    m = rank_source.daily_rank_metrics(df, "prob", "label_pm_10d_net")
    assert m == {"ic": None, "top10_net": None, "n_days": 0}


def test_weighted_ic_skips_missing_horizons():
    v = rank_source.weighted_ic_from_horizons({"3d": None, "5d": 0.10, "10d": 0.30})
    assert v == pytest.approx((0.35 * 0.10 + 0.40 * 0.30) / 0.75)
    assert rank_source.weighted_ic_from_horizons({"3d": None}) is None


def test_evaluate_and_choose_picks_prob():
    ev = rank_source.evaluate_keys(_eval_frame(), eval_days=10, top_n=2)
    assert rank_source.choose_rank_key(ev) == "prob"
    assert ev["prob"]["per_horizon"]["3d"]["ic"] == pytest.approx(1.0)
    assert ev["mag"]["weighted_ic"] == pytest.approx(-1.0)
    # blend = mag×prob 在本造数下非单调 (驼峰) → 弱于 prob, 胜负仍可分
    assert ev["blend"]["weighted_ic"] < ev["prob"]["weighted_ic"]


def test_choose_tie_and_empty_default_blend():
    assert rank_source.choose_rank_key({}) == "blend"
    ev = {
        "mag": {"weighted_ic": 0.02},
        "prob": {"weighted_ic": 0.02},
        "blend": {"weighted_ic": 0.01},
    }
    assert rank_source.choose_rank_key(ev) == "blend"


def _payload(chosen="mag", trained_through="2026-09-10"):
    return {"board": "main", "chosen": chosen, "trained_through": trained_through}


def test_save_load_latest_and_worm_suffix(tmp_path):
    p1 = rank_source.save_rank_source(
        "main", _payload("mag"), directory=tmp_path, ts="20260910_2000"
    )
    # 同 ts 再存 → 不覆盖 (WORM), 追加序号
    p2 = rank_source.save_rank_source(
        "main", _payload("blend"), directory=tmp_path, ts="20260910_2000"
    )
    assert p2 != p1
    assert json.loads(p1.read_text(encoding="utf-8"))["chosen"] == "mag"
    rank_source.save_rank_source(
        "main", _payload("prob", "2026-09-12"), directory=tmp_path, ts="20260912_0800"
    )
    rec = rank_source.load_latest_rank_source("main", directory=tmp_path)
    assert rec["chosen"] == "prob"
    assert rank_source.load_latest_rank_source("dual", directory=tmp_path) is None


def test_load_latest_bad_json_returns_none(tmp_path):
    bad = tmp_path / "main_rank_source_20260912_0800.json"
    bad.write_text("not-json", encoding="utf-8")
    assert rank_source.load_latest_rank_source("main", directory=tmp_path) is None


def test_resolve_explicit_knob_passthrough(tmp_path):
    assert rank_source.resolve_rank_key("main", knob="mag", directory=tmp_path) == "mag"
    assert (
        rank_source.resolve_rank_key("main", knob="prob", directory=tmp_path) == "prob"
    )


def test_resolve_auto_uses_fresh_json(tmp_path):
    rank_source.save_rank_source(
        "main", _payload("mag"), directory=tmp_path, ts="20260910_2000"
    )
    got = rank_source.resolve_rank_key(
        "main", as_of="2026-09-12", knob="auto", directory=tmp_path, max_stale_days=45
    )
    assert got == "mag"


def test_resolve_auto_stale_json_falls_back_blend(tmp_path):
    rank_source.save_rank_source(
        "main", _payload("mag", "2026-06-01"), directory=tmp_path, ts="20260601_2000"
    )
    got = rank_source.resolve_rank_key(
        "main", as_of="2026-09-12", knob="auto", directory=tmp_path, max_stale_days=45
    )
    assert got == "blend"


def test_resolve_missing_or_bad_chosen_defaults_blend(tmp_path):
    assert (
        rank_source.resolve_rank_key(
            "main", as_of="2026-09-12", knob="auto", directory=tmp_path
        )
        == "blend"
    )
    rank_source.save_rank_source(
        "main", _payload("banana"), directory=tmp_path, ts="20260910_2000"
    )
    got = rank_source.resolve_rank_key(
        "main", as_of="2026-09-12", knob="auto", directory=tmp_path, max_stale_days=45
    )
    assert got == "blend"
