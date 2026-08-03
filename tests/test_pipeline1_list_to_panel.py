# -*- coding: utf-8 -*-
"""PIPELINE1 → 选股看板 连通性单元测试.

验证从 PIPELINE1 生成的清单 (list_YYYYMMDD.parquet) 到选股看板
(data_service.load_latest_list / load_priority_symbols) 的完整数据链路,
确保 schema V1.4 字段对齐、priority 同步、清单可被面板正确加载.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

# ── 项目路径 ──
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.pipeline1.list_generator import SCHEMA_FIELDS  # noqa: E402
from app.streamlit.data_service import (  # noqa: E402
    apply_priority_tags,
    demo_list,
    list_available_dates,
    load_latest_list,
    load_list,
    load_priority_symbols,
    pipeline_buy_candidates,
    save_priority_symbols,
    toggle_priority,
)


# ── 合成 V1.4 schema 清单 ──


def _make_schema_list(symbols=("600519", "300750", "601318")) -> pd.DataFrame:
    """生成与 PIPELINE1 ListGenerator.emit 输出同构的 V1.4 清单."""
    rng = np.random.default_rng(42)
    n = len(symbols)
    return pd.DataFrame(
        {
            "symbol": list(symbols),
            "board": ["main"] * n,
            "day_change": rng.uniform(-0.03, 0.06, n),
            "pred_ret_1d": rng.uniform(-0.02, 0.05, n),
            "pred_ret_2d": rng.uniform(-0.03, 0.08, n),
            "pred_ret_3d": rng.uniform(-0.03, 0.09, n),
            "pred_ret_5d": rng.uniform(-0.04, 0.12, n),
            "prob_up": np.round(rng.uniform(0.42, 0.62, n), 3),
            "prob_up_2d": np.round(rng.uniform(0.40, 0.66, n), 3),
            "prob_up_3d": np.round(rng.uniform(0.38, 0.68, n), 3),
            "prob_up_5d": np.round(rng.uniform(0.36, 0.70, n), 3),
            "momentum": rng.choice(["high", "medium", "low"], n, p=[0.3, 0.5, 0.2]),
            "consensus_score": rng.uniform(1, n, n),
            "signal_conflict": rng.choice([0, 1], n, p=[0.8, 0.2]),
            "is_limit_up_close": 0,
            "is_one_word_limit": 0,
            "market_state": "range",
            "score": rng.uniform(0, 0.05, n),
            "compound_ret": rng.uniform(0.0, 0.06, n),
            "compound_prob": np.round(rng.uniform(0.42, 0.62, n), 6),
            "pred_q10": rng.uniform(-0.04, 0.01, n),
            "pred_q50": rng.uniform(-0.01, 0.04, n),
            "pred_q90": rng.uniform(0.01, 0.10, n),
            "uncertainty_width": rng.uniform(0.02, 0.12, n),
            "pain_prob": np.round(rng.uniform(0.0, 0.5, n), 3),
            "announce_score": rng.uniform(-1.0, 1.0, n),
            "weight": np.round(rng.uniform(0.02, 0.10, n), 4),
            "schema_version": "1.4",
        }
    )


@pytest.fixture()
def tmp_list_dir(tmp_path):
    """临时清单目录."""
    d = tmp_path / "lists"
    d.mkdir()
    return str(d)


@pytest.fixture()
def tmp_priority_path(tmp_path):
    """临时 priority.json 路径."""
    return str(tmp_path / "priority.json")


# ══════════════════════════════════════════════════════════
# 1. Schema 对齐: PIPELINE1 输出 → data_service 加载
# ══════════════════════════════════════════════════════════


class TestSchemaAlignment:
    """验证清单字段与 data_service 期望的列一致."""

    def test_schema_fields_match(self):
        """SCHEMA_FIELDS 常量与合成的 V1.4 清单列一致."""
        lst = _make_schema_list()
        assert list(lst.columns) == SCHEMA_FIELDS, (
            "清单 schema 与 SCHEMA_FIELDS 不匹配:\n"
            f"  清单: {list(lst.columns)}\n"
            f"  SCHEMA_FIELDS: {SCHEMA_FIELDS}"
        )

    def test_schema_version_constant(self):
        """所有行的 schema_version = '1.4'."""
        lst = _make_schema_list()
        assert (lst["schema_version"] == "1.4").all()


# ══════════════════════════════════════════════════════════
# 2. 清单持久化 → data_service 加载链路
# ══════════════════════════════════════════════════════════


class TestListPersistence:
    """模拟 PIPELINE1 to_parquet → data_service 加载."""

    def test_write_and_load_list(self, tmp_list_dir):
        """写入 list_YYYYMMDD.parquet → load_list 正确读取."""
        lst = _make_schema_list()
        path = os.path.join(tmp_list_dir, "list_20260731.parquet")
        lst.to_parquet(path, index=False)

        loaded = load_list("20260731", list_dir=tmp_list_dir)
        assert loaded is not None
        assert list(loaded.columns) == SCHEMA_FIELDS
        assert len(loaded) == len(lst)
        assert set(loaded["symbol"]) == set(lst["symbol"])

    def test_load_missing_returns_none(self, tmp_list_dir):
        """不存在的日期返回 None."""
        assert load_list("99990101", list_dir=tmp_list_dir) is None

    def test_list_available_dates_descending(self, tmp_list_dir):
        """日期列表按降序 (最新在前)."""
        for d in ("20260728", "20260729", "20260730", "20260731"):
            pd.DataFrame({"symbol": ["600519"], "score": [0.5]}).to_parquet(
                os.path.join(tmp_list_dir, f"list_{d}.parquet"), index=False
            )
        dates = list_available_dates(tmp_list_dir)
        assert dates == ["20260731", "20260730", "20260729", "20260728"]

    def test_load_latest_list_returns_newest(self, tmp_list_dir):
        """load_latest_list 返回最新日期的清单 + 日期字符串."""
        for d in ("20260729", "20260731"):
            lst = _make_schema_list(symbols=("600519",))
            lst.to_parquet(os.path.join(tmp_list_dir, f"list_{d}.parquet"), index=False)
        df, date = load_latest_list(list_dir=tmp_list_dir)
        assert date == "20260731"
        assert df is not None
        assert "priority" in df.columns  # apply_priority_tags 注入

    def test_load_latest_empty_dir(self, tmp_list_dir):
        """空目录 → (None, None)."""
        df, date = load_latest_list(list_dir=tmp_list_dir)
        assert df is None and date is None


# ══════════════════════════════════════════════════════════
# 3. Priority 同步: PIPELINE1 → priority.json → data_service
# ══════════════════════════════════════════════════════════


class TestPrioritySync:
    """验证 priority.json 的读写与清单标记联动."""

    def test_save_and_load_priority(self, tmp_priority_path):
        """save → load 往返一致."""
        syms = {"600519", "300750"}
        save_priority_symbols(syms, path=tmp_priority_path)
        loaded = load_priority_symbols(path=tmp_priority_path)
        assert loaded == syms

    def test_load_missing_returns_empty(self, tmp_path):
        """不存在的文件返回空集合."""
        assert load_priority_symbols(path=str(tmp_path / "nope.json")) == set()

    def test_toggle_adds_then_removes(self, tmp_priority_path):
        """toggle: 添加 → True, 再 toggle → False."""
        assert toggle_priority("600519", path=tmp_priority_path) is True
        assert "600519" in load_priority_symbols(path=tmp_priority_path)
        assert toggle_priority("600519", path=tmp_priority_path) is False
        assert "600519" not in load_priority_symbols(path=tmp_priority_path)

    def test_apply_priority_tags(self, tmp_priority_path):
        """apply_priority_tags: priority 标记股的 priority=True."""
        save_priority_symbols({"600519", "300750"}, path=tmp_priority_path)
        lst = _make_schema_list(("600519", "300750", "601318"))
        tagged = apply_priority_tags(lst, priority_path=tmp_priority_path)
        assert "priority" in tagged.columns
        assert tagged.loc[tagged["symbol"] == "600519", "priority"].iloc[0]
        assert tagged.loc[tagged["symbol"] == "300750", "priority"].iloc[0]
        assert not tagged.loc[tagged["symbol"] == "601318", "priority"].iloc[0]

    def test_pipeline_buy_candidates_with_pred_ret(self, tmp_list_dir):
        """pipeline_buy_candidates: pred_ret_1d > 0 的股票被推荐."""
        lst = _make_schema_list(("600519", "300750", "601318"))
        lst.loc[lst["symbol"] == "600519", "pred_ret_1d"] = 0.02
        lst.loc[lst["symbol"] == "300750", "pred_ret_1d"] = -0.01
        lst.loc[lst["symbol"] == "601318", "pred_ret_1d"] = 0.03
        candidates = pipeline_buy_candidates(lst)
        assert "600519" in candidates
        assert "601318" in candidates
        assert "300750" not in candidates

    def test_pipeline_buy_candidates_fallback_score(self):
        """无 pred_ret_1d 列时 fallback 到 score 前 30%."""
        lst = pd.DataFrame(
            {
                "symbol": ["A", "B", "C", "D", "E"],
                "score": [0.1, 0.08, 0.06, 0.04, 0.02],
            }
        )
        candidates = pipeline_buy_candidates(lst)
        assert "A" in candidates
        assert "B" in candidates  # 前 30% (≈1.5 → 至少 A, B)

    def test_pipeline_buy_candidates_empty(self):
        """空 DataFrame 返回空集合."""
        assert pipeline_buy_candidates(pd.DataFrame()) == set()
        assert pipeline_buy_candidates(pd.DataFrame({"symbol": []})) == set()


# ══════════════════════════════════════════════════════════
# 4. 端到端: PIPELINE1 写清单 → 看板加载 → priority 标记
# ══════════════════════════════════════════════════════════


class TestEndToEndListToPanel:
    """模拟 DailySelectionPipeline.run 的持久化 → 看板消费."""

    def test_e2e_pipeline_to_panel(self, tmp_list_dir, tmp_priority_path):
        # 1. 模拟 PIPELINE1 生成清单并 to_parquet
        lst = _make_schema_list(("600519", "300750", "601318", "000001"))
        trade_date = "20260731"
        lst.to_parquet(
            os.path.join(tmp_list_dir, f"list_{trade_date}.parquet"), index=False
        )

        # 2. 模拟 daily_pipeline.py 中的 priority.json 同步逻辑
        new_symbols = set(lst["symbol"].tolist())
        save_priority_symbols(new_symbols, path=tmp_priority_path)

        # 3. 看板加载最新清单
        df, date = load_latest_list(
            list_dir=tmp_list_dir, priority_path=tmp_priority_path
        )
        assert date == trade_date
        assert df is not None
        assert set(df["symbol"]) == new_symbols

        # 4. priority 标记应全部为 True (刚同步)
        assert df["priority"].all()

        # 5. pipeline_buy_candidates 应返回 pred_ret_1d > 0 的子集
        candidates = pipeline_buy_candidates(df)
        expected = set(df.loc[df["pred_ret_1d"] > 0, "symbol"])
        assert candidates == expected

    def test_e2e_multi_day_yesterday_carryover(self, tmp_list_dir):
        """两日清单: 第二天加载时应独立加载 (不混淆)."""
        # Day 1
        lst1 = _make_schema_list(("600519", "300750"))
        lst1.to_parquet(
            os.path.join(tmp_list_dir, "list_20260730.parquet"), index=False
        )
        # Day 2 (不同股票)
        lst2 = _make_schema_list(("601318", "000001"))
        lst2.to_parquet(
            os.path.join(tmp_list_dir, "list_20260731.parquet"), index=False
        )

        # 看板加载最新
        df, date = load_latest_list(list_dir=tmp_list_dir)
        assert date == "20260731"
        assert set(df["symbol"]) == {"601318", "000001"}
        assert "600519" not in set(df["symbol"])

    def test_demo_list_fallback_schema(self):
        """无清单时 demo_list 返回与 V1.4 schema 同构的 DataFrame."""
        demo = demo_list()
        assert "symbol" in demo.columns
        assert "score" in demo.columns
        assert "pred_ret_1d" in demo.columns
        assert "schema_version" in demo.columns
        assert (demo["schema_version"] == "1.4").all()


# ══════════════════════════════════════════════════════════
# 5. priority.json 格式兼容 (list / dict)
# ══════════════════════════════════════════════════════════


class TestPriorityFormat:
    """priority.json 支持两种格式: list 或 {symbols: [...]}."""

    def test_list_format(self, tmp_path):
        p = tmp_path / "priority.json"
        p.write_text(json.dumps(["600519", "300750"]), encoding="utf-8")
        syms = load_priority_symbols(path=str(p))
        assert syms == {"600519", "300750"}

    def test_dict_format(self, tmp_path):
        p = tmp_path / "priority.json"
        p.write_text(json.dumps({"symbols": ["601318", "000001"]}), encoding="utf-8")
        syms = load_priority_symbols(path=str(p))
        assert syms == {"601318", "000001"}

    def test_invalid_format_returns_empty(self, tmp_path):
        p = tmp_path / "priority.json"
        p.write_text("invalid", encoding="utf-8")
        syms = load_priority_symbols(path=str(p))
        assert syms == set()
