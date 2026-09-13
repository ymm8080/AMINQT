# -*- coding: utf-8 -*-
"""rank_source purged_v1 影子重评 (0913) — _shadow_prob 集成单测.

动机: 原口径用在役全史 bundle 评 eval 窗 → 尾部 eval 日 100% in-sample →
prob 系统性虚高 (0912 首评双板选 prob, walk-forward #11 实测 mag 碾压).
purged_v1 = eval 窗 cutoff 前推 RANK_SOURCE_PURGE_DAYS=11 交易日重拟合影子
集预测。本文件验证:
① 影子训练集确实在 eval 窗前 11 个交易日截断 (隔离性), 全半衰期档都拟合;
② 诚实影子下 mag 胜; 泄漏反例 (影子背全史) 下 prob 反超 → 口径可分胜负;
③ 影子 fit raise → fail-open (无 json 无台账, 不抛);
④ 旧 payload (无 eval_mode 字段) 兼容解析;
⑤ payload 带 eval_mode/eval_days, 台账行照写.
"""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd

from app.pipeline_parallel import prob_head, rank_source
from app.pipeline_parallel import calibration as calibration_mod
from app.pipeline_parallel.config import RANK_SOURCE_EVAL_DAYS, RANK_SOURCE_PURGE_DAYS
from config.settings import PROB_GATE
from scripts import _head_choice_ledger as ledger_mod
from scripts import _shortlist_t5_t10 as sl_mod
from scripts import _train_parallel_prob_head as tph

N_DATES, N_SYMS = 100, 8
FEATS = ["uid"]


def _prob_of(sym: str) -> float:
    """symbol → 单调于真实质量的 prob (in-sample 模型学到的信号形状)."""
    return 0.55 + 0.40 * int(sym[1:]) / (N_SYMS - 1)


def _synth(seed: int = 7) -> pd.DataFrame:
    """每股固定质量 q, mag = q + 较大噪声, 各视界标签 = q + 小噪声."""
    rng = np.random.default_rng(seed)
    q = np.linspace(-0.03, 0.03, N_SYMS)
    rows = []
    for d in range(N_DATES):
        date = pd.Timestamp("2026-01-01") + pd.offsets.BDay(d)
        for i in range(N_SYMS):
            rows.append(
                {
                    "symbol": f"s{i:03d}",
                    "date": date,
                    "uid": float(d * N_SYMS + i),
                    "score": 0.0,
                    "mag": q[i] + rng.normal(0, 0.02),
                    **{
                        f"label_pm_{h}_net": q[i] + rng.normal(0, 0.004)
                        for h in ("3d", "5d", "10d")
                    },
                }
            )
    return pd.DataFrame(rows)


class _MemModel:
    """uid 记忆器: 映射内 uid → 质量单调 prob; 未见 uid → 0.5±噪声 (无信号)."""

    def __init__(self, qmap: dict, seed: int):
        self.qmap = qmap
        self.rng = np.random.default_rng(seed)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.array([self.qmap.get(u, np.nan) for u in x[:, 0]])
        unseen = np.isnan(p)
        p[unseen] = 0.5 + self.rng.normal(0, 0.02, int(unseen.sum()))
        return np.column_stack([1 - p, p])


def _fit_stub_factory(full: pd.DataFrame, leaky: bool, captures: list):
    """leaky=False → 只背 train 行 (诚实影子); True → 闭包背全史 (模拟 in-sample)."""

    def stub(board, train, hl):
        src = full if leaky else train
        qmap = dict(zip(src["uid"].tolist(), (_prob_of(s) for s in src["symbol"])))
        captures.append(
            {
                "hl": hl,
                "train_max": pd.Timestamp(train["date"].max()),
                "n_train": int(len(train)),
            }
        )
        return _MemModel(qmap, seed=123), list(FEATS)

    return stub


def _patch_happy(monkeypatch, tmp_dir, t: pd.DataFrame, leaky: bool, captures: list):
    monkeypatch.setattr(
        prob_head, "load_all_tiers", lambda board: {7: {"feat_cols": list(FEATS)}}
    )
    monkeypatch.setattr(sl_mod, "_panel_per_stock", lambda: {("main", "both"): t})
    mag_lookup = {(r.symbol, r.date): r.mag for r in t.itertuples()}
    monkeypatch.setattr(
        calibration_mod,
        "calibrate_mag10d",
        lambda w: w.assign(
            mag=[mag_lookup[(s, d)] for s, d in zip(w["symbol"], w["date"])]
        ),
    )
    monkeypatch.setattr(prob_head, "bundle_dir", lambda: tmp_dir)
    monkeypatch.setattr(ledger_mod, "LEDGER_PATH", tmp_dir / "head_choice_ledger.csv")
    monkeypatch.setattr(
        prob_head, "_fit_cls_model", _fit_stub_factory(t, leaky, captures)
    )


def test_purge_isolation_all_tiers(tmp_path, monkeypatch):
    t = _synth()
    cap: list = []
    _patch_happy(monkeypatch, tmp_path, t, leaky=False, captures=cap)
    tph._eval_rank_source("main", t, "2026-05-31")
    assert len(cap) == len(PROB_GATE["half_lives"])
    assert {c["hl"] for c in cap} == set(PROB_GATE["half_lives"])
    # eval_lo = 评估窗首日 (use_dates[0]); 影子 cutoff 应恰在前推 11 交易日
    use_dates = sorted(t["date"].unique())[-(int(RANK_SOURCE_EVAL_DAYS) + 15):]
    eval_lo = pd.Timestamp(use_dates[0])
    uniq = np.unique(t["date"].values)
    pos = int(np.searchsorted(uniq, eval_lo.to_datetime64()))
    assert pos >= int(RANK_SOURCE_PURGE_DAYS)
    cut_expected = uniq[pos - int(RANK_SOURCE_PURGE_DAYS)]
    for c in cap:
        assert c["train_max"] == pd.Timestamp(cut_expected)
        assert c["train_max"] < eval_lo
        assert c["n_train"] == (pos - int(RANK_SOURCE_PURGE_DAYS) + 1) * N_SYMS  # <= cut 含当日
    gap = int(np.searchsorted(uniq, np.datetime64(eval_lo))) - int(
        np.searchsorted(uniq, cut_expected)
    )
    assert gap == int(RANK_SOURCE_PURGE_DAYS)


def test_purged_mag_wins_and_leak_flips_to_prob(tmp_path, monkeypatch):
    t = _synth()
    # ① 诚实影子: eval 行 uid 全部未见 → prob 无信号, mag (质量+噪声) 胜
    cap: list = []
    _patch_happy(monkeypatch, tmp_path, t, leaky=False, captures=cap)
    tph._eval_rank_source("main", t, "2026-05-31")
    rec = rank_source.load_latest_rank_source("main", directory=tmp_path)
    assert rec is not None
    assert rec["metrics"]["mag"]["weighted_ic"] > rec["metrics"]["prob"]["weighted_ic"]
    assert rec["chosen"] == "mag"
    # ② 泄漏反例: 影子背全史 (eval 行 uid 全在映射内) → prob 虚高反超
    cap2: list = []
    leak_dir = tmp_path / "leak"
    leak_dir.mkdir()
    _patch_happy(monkeypatch, leak_dir, t, leaky=True, captures=cap2)
    tph._eval_rank_source("main", t, "2026-05-31")
    rec2 = rank_source.load_latest_rank_source("main", directory=leak_dir)
    assert rec2 is not None
    assert rec2["metrics"]["prob"]["weighted_ic"] > rec2["metrics"]["mag"]["weighted_ic"]
    assert rec2["chosen"] == "prob"


def test_shadow_fit_raise_fails_open(tmp_path, monkeypatch, capsys):
    t = _synth()
    _patch_happy(monkeypatch, tmp_path, t, leaky=False, captures=[])

    def boom(board, train, hl):
        raise RuntimeError("boom")

    monkeypatch.setattr(prob_head, "_fit_cls_model", boom)
    tph._eval_rank_source("main", t, "2026-05-31")  # 不抛 (fail-open)
    assert "fail-open" in capsys.readouterr().out
    assert list(tmp_path.glob("*_rank_source_*.json")) == []
    assert not (tmp_path / "head_choice_ledger.csv").exists()


def test_old_payload_without_eval_mode_resolves(tmp_path):
    rank_source.save_rank_source(
        "main",
        {"board": "main", "chosen": "mag", "trained_through": "2026-09-10"},
        directory=tmp_path,
        ts="20260910_2000",
    )
    got = rank_source.resolve_rank_key(
        "main", as_of="2026-09-13", knob="auto", directory=tmp_path, max_stale_days=45
    )
    assert got == "mag"


def test_payload_eval_mode_and_ledger_row(tmp_path, monkeypatch):
    t = _synth()
    _patch_happy(monkeypatch, tmp_path, t, leaky=False, captures=[])
    tph._eval_rank_source("main", t, "2026-05-31")
    rec = rank_source.load_latest_rank_source("main", directory=tmp_path)
    assert rec["eval_mode"] == "purged_v1"
    assert rec["eval_days"] == int(RANK_SOURCE_EVAL_DAYS)
    assert rec["board"] == "main"
    ledger = tmp_path / "head_choice_ledger.csv"
    assert ledger.exists()
    rows = list(csv.DictReader(ledger.open(encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert row["pipeline"] == "parallel"
    assert row["board"] == "main"
    assert row["chosen"] == rec["chosen"]
    assert row["switched"] == "False"
    assert row["n_features"] == str(len(FEATS))
