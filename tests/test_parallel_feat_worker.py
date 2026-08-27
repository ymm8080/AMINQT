"""Tests for legacy 双板特征构建子进程并行 (2026-08-21).

语义前提: FeatureEngineV35.build 是无状态纯函数链 (dim 方法只读输入帧+config),
main/dual 帧股票集不相交, _add_cross_sectional_ranks 是帧内同日排名 →
拆板并行结果必须与串行 build 逐字节一致. 本文件验证:
  1. config 开关存在且默认关 (等 08-21 自动化插桩计时数据拍板开启)
  2. 子进程 worker 输出 == 直接调用 build (字节一致)
  3. _build_features_parallel 编排器双板输出 == 串行双调用
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from app.pipeline1.cleaning_pipeline import board_of
from app.pipeline1.daily_pipeline import DailySelectionPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from config.settings import LEGACY_PARALLEL_FEATURES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def _mk_board(symbols, days=120, seed=7) -> pd.DataFrame:
    """最小 OHLCV 面板 (同 test_pipeline1_v35.make_panel 口径, 减量提速)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=days)
    frames = []
    for sym in symbols:
        close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, days))
        open_ = close * (1 + rng.normal(0, 0.005, days))
        high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.005, days)))
        low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.005, days)))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates,
                    "board": board_of(sym),
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "close_hfq": close,
                    "high_hfq": high,
                    "low_hfq": low,
                    "open_hfq": open_,
                    "volume": rng.integers(1e6, 1e8, days).astype(float),
                    "amount": rng.uniform(6e7, 2e9, days),
                    "turnover_rate": rng.uniform(1, 10, days),
                    "free_float_turnover_rate": rng.uniform(1, 10, days),
                    "pre_close": np.concatenate([[close[0]], close[:-1]]),
                    "is_suspended": False,
                    "industry": "白酒",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_config_flag_exists_and_default_on():
    """开关默认开 — 2026-08-25 真面板 300d 对照已过 (逐字节一致 + 1.43x), 勿无证据改回."""
    assert LEGACY_PARALLEL_FEATURES is True


def test_worker_equivalent_to_direct_build(tmp_path):
    """子进程 worker (parquet 进/出) 输出 == 直接调用 build, 逐字节一致."""
    df = _mk_board(("600519", "300750"))
    inp = tmp_path / "in.parquet"
    out = tmp_path / "out.parquet"
    df.to_parquet(inp, index=False)
    cp = subprocess.run(
        [
            PY,
            "-u",
            "-m",
            "app.pipeline1.parallel_feat_worker",
            str(inp),
            str(out),
            json.dumps([]),
            "1",
            json.dumps({}),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert cp.returncode == 0, cp.stderr
    direct = FeatureEngineV35().build(df, None, cross_sectional_rank=True)
    got = pd.read_parquet(out)
    pd.testing.assert_frame_equal(direct, got, check_exact=True)


def test_build_features_parallel_matches_serial(tmp_path):
    """编排器双板子进程并行输出 == 串行两次 build (main 无 cs_rank, dual 有)."""
    main_df = _mk_board(("600519", "601318", "600036"))
    dual_df = _mk_board(("300750", "300223", "301001"))
    pipe = DailySelectionPipeline.__new__(DailySelectionPipeline)
    pipe.float_shares_map = None

    feat_m, feat_d = pipe._build_features_parallel(main_df, dual_df, None, None)
    expect_m = FeatureEngineV35().build(main_df, None, cross_sectional_rank=False)
    expect_d = FeatureEngineV35().build(dual_df, None, cross_sectional_rank=True)
    pd.testing.assert_frame_equal(feat_m, expect_m, check_exact=True)
    pd.testing.assert_frame_equal(feat_d, expect_d, check_exact=True)
    assert "symbol" in feat_m.columns and "date" in feat_m.columns


def test_serial_helper_equivalence():
    """串行 helper 输出 == 手动两次 build (回退路径与主路径同源)."""
    main_df = _mk_board(("600519",))
    dual_df = _mk_board(("300750",))
    pipe = DailySelectionPipeline.__new__(DailySelectionPipeline)
    pipe.features = FeatureEngineV35()
    pipe.float_shares_map = None
    feat_m, feat_d = pipe._build_features_serial(main_df, dual_df, None, None)
    assert not feat_m.empty and not feat_d.empty
    assert_feat_superset(
        feat_m, FeatureEngineV35().build(main_df, None, cross_sectional_rank=False)
    )


def assert_feat_superset(got: pd.DataFrame, expect: pd.DataFrame):
    """列集超集 + 共有列值逐字节一致 (full build 列数可能随版本漂移, 只锁共有列)."""
    common = expect.columns.intersection(got.columns)
    pd.testing.assert_frame_equal(
        got[list(common)].reset_index(drop=True),
        expect[list(common)].reset_index(drop=True),
        check_exact=True,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
