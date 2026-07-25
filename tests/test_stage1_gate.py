# -*- coding: utf-8 -*-
"""P19.0 阶段一点火验收脚本 e2e 测试 (合成面板 + small 模式 + 伪特征 stub)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def _load_gate_module():
    """按路径加载 scripts/run_stage1_gate.py (scripts 非包)."""
    spec = importlib.util.spec_from_file_location(
        "run_stage1_gate", ROOT / "scripts" / "run_stage1_gate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_panel(days: int = 780, seed: int = 21) -> pd.DataFrame:
    """780 交易日 (>750 窗口) 双板块面板: 主板/双创各 6 只 (日 IC 需 >5 只/日)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=days)
    symbols = [("60051%d" % i, "main", "白酒") for i in range(6)] + [
        ("30001%d" % i, "GEM", "电子") for i in range(6)
    ]
    frames = []
    for sym, board, industry in symbols:
        close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, days))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates,
                    "board": board,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "close_hfq": close,
                    "open_hfq": close,
                    "high_hfq": close * 1.01,
                    "low_hfq": close * 0.99,
                    "volume": 1e7,
                    "amount": 1e9,
                    "turnover_rate": 5.0,
                    "free_float_turnover_rate": 5.0,
                    "pre_close": pd.Series(close).shift(1).fillna(close[0]),
                    "is_suspended": False,
                    "is_st": False,
                    "industry": industry,
                    "list_days": 1000,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class _StubFeatures:
    """跳过真实特征工程 (慢), 直接加 f1/f2 伪特征 + ATR_pct (波动桶分桶用)."""

    def build(self, df, float_shares_map=None):
        rng = np.random.default_rng(7)
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        df["f1"] = rng.normal(size=len(df))
        df["f2"] = rng.normal(size=len(df))
        ret = df.groupby("symbol")["close_hfq"].pct_change()
        df["ATR_pct"] = (
            ret.groupby(df["symbol"])
            .rolling(14, min_periods=5)
            .std()
            .reset_index(level=0, drop=True)
        )
        return df

    @staticmethod
    def feature_columns(df):
        return ["f1", "f2"]


def test_stage1_gate_small_mode(tmp_path, monkeypatch):
    """--small 跑通主函数: 返回码 0/1 + JSON 报告含四项门禁 checks."""
    panel_path = tmp_path / "panel.parquet"
    make_panel().to_parquet(panel_path)

    gate = _load_gate_module()
    monkeypatch.setattr(gate, "FeatureEngineV35", _StubFeatures)

    rc = gate.main(
        [
            "--panel",
            str(panel_path),
            "--tag",
            "pytest",
            "--small",
            "--out-dir",
            str(tmp_path),
            "--journal-dir",
            str(tmp_path / "journal"),
            "--model-dir",
            str(tmp_path / "models"),
        ]
    )
    assert rc in (0, 1)

    report_path = tmp_path / "stage1_gate_pytest.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["pass"] == (rc == 0)
    assert set(report["checks"]) == {
        "rank_ic",
        "icir",
        "high_vol_ic",
        "train_ic_no_leak",
    }
    # 双板块均有各自门禁结果
    assert set(report["boards"]) == {"main", "dual"}
    for b in ("main", "dual"):
        assert set(report["boards"][b]["gate"]["checks"]) == set(report["checks"])
    # WORM 账本已追加
    journal = tmp_path / "journal" / "backtest_runs.jsonl"
    assert journal.exists()
    rec = json.loads(journal.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["tag"] == "stage1_gate_pytest"
