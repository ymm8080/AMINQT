"""cleaning_pipeline.load_panel_v3 — V3 面板读取预过滤契约测试.

2026-08-10 优化: V3 面板读取时用 pyarrow 行级预过滤 (amount>=5000万 且 非停牌),
下推到 parquet 扫描阶段 → 少读 ~20% 行, 降低重建检查点/训练的内存峰值.
该预过滤与 CleaningPipeline.run_train 内部 step2(amount>=min_amount) + step3(剔停牌)
同口径 — 实测输出与整表读取完全一致 (main/dual 行集差 0).
被 _refresh_parallel_checkpoints 与 _retrain_legacy_full 共用 (单一口径).

测试只验证契约 (参数/列/是否调用 pyarrow), 不跑真实 0.84GB 面板.
"""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

from app.pipeline1.cleaning_pipeline import (
    CleaningConfig,
    CleaningPipeline,
    load_panel_v3,
)
from config.settings import PANEL_V3_PATH


class _FakeTab:
    def __init__(self, captured):
        self._captured = captured

    def to_pandas(self):
        return self._captured["df"]


def _patch_read_table(monkeypatch, captured):
    def fake_read_table(path, columns=None, filters=None):
        captured["path"] = path
        captured["columns"] = columns
        captured["filters"] = filters
        captured["df"] = "panel"
        return _FakeTab(captured)

    monkeypatch.setattr(pq, "read_table", fake_read_table)


def test_load_panel_reads_full_columns(monkeypatch):
    """load_panel_v3 必须返回全部列 (特征/标签构建需要), 不得只读清洗列."""
    captured = {}
    _patch_read_table(monkeypatch, captured)

    df = load_panel_v3()

    assert df is captured["df"]
    assert captured["columns"] is None


def test_load_panel_prefilters_by_cleaning_rules(monkeypatch):
    """预过滤条件 = amount>=CleaningConfig.min_amount 且 非停牌 (与 run_train 同口径)."""
    captured = {}
    _patch_read_table(monkeypatch, captured)

    load_panel_v3()

    assert captured["filters"] is not None
    filts = {f[0]: f for f in captured["filters"]}
    assert "amount" in filts
    assert filts["amount"][1] == ">="
    assert filts["amount"][2] == CleaningConfig().min_amount
    assert "is_suspended" in filts
    assert filts["is_suspended"][1] == "="
    assert filts["is_suspended"][2] is False


def test_load_panel_defaults_to_v3_path(monkeypatch):
    """未传 path → 读 PANEL_V3_PATH."""
    captured = {}
    _patch_read_table(monkeypatch, captured)

    load_panel_v3()

    assert str(captured["path"]) == str(PANEL_V3_PATH)


def test_load_panel_accepts_custom_path(monkeypatch):
    """显式传 path → 用该路径."""
    captured = {}
    _patch_read_table(monkeypatch, captured)

    load_panel_v3("D:/tmp/custom.parquet")

    assert str(captured["path"]) == "D:/tmp/custom.parquet"


def test_step0_board_split_board_scoped():
    """board 限定下 step0 只拷贝目标板块切片, 另一板块返回空 df.

    2026-08-13 OOM 修复: dual-only 重训时 main 1.2M×109 的 ~1GB 块不再被
    分配 (空帧省内存). board=None 保持双板原行为.
    """
    df = pd.DataFrame(
        {
            "symbol": ["600000", "300750", "688001"],
            "date": ["2026-08-07"] * 3,
            "close": [1.0, 2.0, 3.0],
        }
    )

    main, dual = CleaningPipeline.step0_board_split(df, board="dual")
    assert main.empty
    assert not dual.empty
    assert set(dual["board"]) == {"GEM", "STAR"}

    main, dual = CleaningPipeline.step0_board_split(df, board="main")
    assert not main.empty
    assert dual.empty
    assert set(main["board"]) == {"main"}

    main, dual = CleaningPipeline.step0_board_split(df)
    assert not main.empty and not dual.empty
    assert len(main) + len(dual) == len(df)
