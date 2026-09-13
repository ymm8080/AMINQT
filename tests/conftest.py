"""Pytest shared fixtures + project root on sys.path.

Ensures `from config import settings`, `from app...`, `from data...`,
`from services...` resolve when running `pytest` from the project root.
"""

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(autouse=True)
def _hermetic_rank_source(monkeypatch):
    """[0912 夜] 排名键自适应测试隔离: 默认路径不读真实 data/prob_head/*.json.

    json 一旦真实落盘, blend 默认语义的既有测试会被真实 chosen 翻转 (结果依赖
    本机数据状态)。默认 directory=None → 视为无 json (回退 blend); 显式传
    directory=tmp_path 的模块测试仍走真实实现。
    """
    from app.pipeline_parallel import rank_source

    real = rank_source.load_latest_rank_source
    monkeypatch.setattr(
        rank_source,
        "load_latest_rank_source",
        lambda board, directory=None: None if directory is None else real(board, directory=directory),
    )
