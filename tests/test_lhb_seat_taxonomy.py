# -*- coding: utf-8 -*-
"""LHB 席位分类表测试 (app/pipeline1/lhb_seat_taxonomy.py, 2026-09-23).

覆盖: 四规则类别判别 + 优先级 / exalter_norm 归一合并 / labeled 列名与顺序 /
空输入不抛异常 / _refresh_lhb_seat_cache 幂等覆盖 (不读真实 parquet).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app.pipeline1.lhb_seat_taxonomy import (  # noqa: E402
    CAT_ORDER,
    SEAT_LABELED_COLS,
    SEAT_RAW_COLS,
    label_seat_frame,
)


def _seat(df_rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(df_rows)


def _base_row(**kw) -> dict:
    row = dict(
        trade_date="20260923",
        ts_code="000001.SZ",
        exalter="营业部",
        buy=100.0,
        buy_rate=1.0,
        sell=50.0,
        sell_rate=0.5,
        net_buy=50.0,
        side="0",
        reason="测试",
    )
    row.update(kw)
    return row


class TestRuleCategories:
    def test_four_rule_categories(self):
        df = _seat(
            [
                _base_row(exalter="机构专用"),
                _base_row(exalter="沪股通专用"),
                _base_row(exalter="深股通专用"),
                _base_row(exalter="拉萨团结路第二证券营业部"),
                _base_row(exalter="国泰君安证券股份有限公司上海江苏路营业部"),
            ]
        )
        labeled, _ = label_seat_frame(df)
        got = dict(zip(labeled["exalter"], labeled["category"].astype(str)))
        assert got["机构专用"] == "机构专用"
        assert got["沪股通专用"] == "沪深股通"
        assert got["深股通专用"] == "沪深股通"
        assert got["拉萨团结路第二证券营业部"] == "拉萨散户团"
        assert got["国泰君安证券股份有限公司上海江苏路营业部"] == "普通营业部"

    def test_priority_inst_over_lasa(self):
        # 名字同时含 拉萨 和 机构专用 → 后者优先级更高
        df = _seat([_base_row(exalter="拉萨机构专用")])
        labeled, master = label_seat_frame(df)
        assert labeled.iloc[0]["category"] == "机构专用"
        assert master.iloc[0]["category"] == "机构专用"

    def test_category_is_ordered_categorical(self):
        df = _seat([_base_row(exalter="机构专用")])
        _, master = label_seat_frame(df)
        assert list(master["category"].cat.categories) == CAT_ORDER
        assert master["category"].cat.ordered


class TestExalterNorm:
    def test_norm_merges_two_writings(self):
        # 空白 + 全角括号 两种写法归一后合并到同一 master 行
        df = _seat(
            [
                _base_row(exalter="华鑫证券（上海）分公司"),
                _base_row(exalter="华鑫证券(上海)分公司"),
                _base_row(exalter="华鑫证券 (上海) 分公司"),
            ]
        )
        labeled, master = label_seat_frame(df)
        assert len(master) == 1
        assert master.iloc[0]["n_rows"] == 3
        assert labeled["exalter_norm"].nunique() == 1

    def test_strip_applied(self):
        df = _seat([_base_row(exalter="  机构专用  ")])
        labeled, _ = label_seat_frame(df)
        assert labeled.iloc[0]["exalter"] == "机构专用"


class TestOutputSchema:
    def test_labeled_columns_and_order(self):
        df = _seat([_base_row(exalter="机构专用")])
        labeled, _ = label_seat_frame(df)
        assert list(labeled.columns) == SEAT_LABELED_COLS
        assert list(labeled.columns) == [
            "trade_date",
            "ts_code",
            "exalter",
            "buy",
            "buy_rate",
            "sell",
            "sell_rate",
            "net_buy",
            "side",
            "reason",
            "dt",
            "exalter_norm",
            "category",
        ]

    def test_no_intermediate_cols_leak(self):
        df = _seat([_base_row(exalter="机构专用")])
        labeled, master = label_seat_frame(df)
        assert not {"_buy_f", "_sell_f", "seat_category", "rule_cat"} & set(
            labeled.columns
        )
        assert "rule_cat" not in master.columns


class TestEmptyInput:
    def test_empty_returns_empty_frames(self):
        labeled, master = label_seat_frame(pd.DataFrame(columns=SEAT_RAW_COLS))
        assert len(labeled) == 0 and len(master) == 0
        assert list(labeled.columns) == SEAT_LABELED_COLS

    def test_none_returns_empty_frames(self):
        labeled, master = label_seat_frame(None)
        assert len(labeled) == 0 and len(master) == 0


# ── _refresh_lhb_seat_cache: 从 _daily_fetch.py AST 提取函数体独立执行 ──────────


def _load_refresh_fn(cache_path, labeled_path):
    """_daily_fetch.py 是 import 即跑全链的顶层脚本, 不能裸 import; 走 AST 提取
    (与 tests/test_daily_fetch_moneyflow_cache.py 同思路)."""
    src = (REPO / "_daily_fetch.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(src)
    keep = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_refresh_lhb_seat_cache"
    ]
    assert keep, "_refresh_lhb_seat_cache must exist in _daily_fetch.py"
    new_tree = ast.Module(body=keep, type_ignores=[])
    namespace: dict = {
        "os": __import__("os"),
        "pd": pd,
        "np": __import__("numpy"),
        "SEAT_RAW_COLS": SEAT_RAW_COLS,
        "label_seat_frame": label_seat_frame,
        "LHB_SEAT_CACHE_PATH": str(cache_path),
        "LHB_SEAT_LABELED_PATH": str(labeled_path),
    }
    exec(compile(new_tree, "<lhb_seat_extract>", "exec"), namespace)
    return namespace


def _top_inst_day(trade_date="20260923") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [trade_date, trade_date],
            "ts_code": ["000001.SZ", "600000.SH"],
            "exalter": ["机构专用", "国泰君安证券股份有限公司上海江苏路营业部"],
            "buy": [100.0, 30.0],
            "buy_rate": [1.0, 0.3],
            "sell": [50.0, 10.0],
            "sell_rate": [0.5, 0.1],
            "net_buy": [50.0, 20.0],
            "side": ["0", "0"],
            "reason": ["a", "b"],
        }
    )


class TestRefreshLhbSeatCache:
    def _run(self, tmp_path, day_df, preexisting=None):
        cache = tmp_path / "lhb_seat_detail.parquet"
        labeled = tmp_path / "lhb_seat_detail_labeled.parquet"
        if preexisting is not None:
            preexisting.to_parquet(cache, index=False)
        ns = _load_refresh_fn(cache, labeled)
        ns["_refresh_lhb_seat_cache"]("20260923", day_df)
        return pd.read_parquet(cache), (pd.read_parquet(labeled) if labeled.exists() else None)

    def test_idempotent_same_day(self, tmp_path):
        got, lab = self._run(tmp_path, _top_inst_day())
        assert len(got) == 2
        assert len(lab) == 2
        # 同一天再跑一次 → 行数不变, 无重复
        cache = tmp_path / "lhb_seat_detail.parquet"
        labeled = tmp_path / "lhb_seat_detail_labeled.parquet"
        ns = _load_refresh_fn(cache, labeled)
        ns["_refresh_lhb_seat_cache"]("20260923", _top_inst_day())
        got2 = pd.read_parquet(cache)
        assert len(got2) == 2
        assert not got2.duplicated(subset=["trade_date", "ts_code", "exalter", "side"]).any()

    def test_appends_new_day_keeps_old(self, tmp_path):
        got, _ = self._run(tmp_path, _top_inst_day("20260923"), preexisting=_top_inst_day("20260922"))
        assert len(got) == 4
        assert set(got["trade_date"]) == {"20260922", "20260923"}

    def test_empty_response_refused(self, tmp_path):
        cache = tmp_path / "lhb_seat_detail.parquet"
        labeled = tmp_path / "lhb_seat_detail_labeled.parquet"
        ns = _load_refresh_fn(cache, labeled)
        ns["_refresh_lhb_seat_cache"]("20260923", pd.DataFrame())
        assert not cache.exists()

    def test_skeleton_response_refused(self, tmp_path):
        cache = tmp_path / "lhb_seat_detail.parquet"
        labeled = tmp_path / "lhb_seat_detail_labeled.parquet"
        _top_inst_day("20260922").to_parquet(cache, index=False)
        skel = _top_inst_day("20260923")
        skel["exalter"] = None  # 关键列全 NaN (骨架响应)
        ns = _load_refresh_fn(cache, labeled)
        ns["_refresh_lhb_seat_cache"]("20260923", skel)
        assert len(pd.read_parquet(cache)) == 2  # 拒写, 保持原状

    def test_missing_raw_col_refused(self, tmp_path):
        """缺任一 SEAT_RAW_COLS 列必须整日拒写.

        只查 5 列时, side 缺失会让「同股同席位不同 side」的两行 (买榜/卖榜) 带着 NaN side
        进合并, 被去重键折叠成一行 —— 丢数据且不报错.
        """
        cache = tmp_path / "lhb_seat_detail.parquet"
        labeled = tmp_path / "lhb_seat_detail_labeled.parquet"
        _top_inst_day("20260922").to_parquet(cache, index=False)
        day = _top_inst_day("20260923").drop(columns=["side"])
        ns = _load_refresh_fn(cache, labeled)
        ns["_refresh_lhb_seat_cache"]("20260923", day)
        assert len(pd.read_parquet(cache)) == 2  # 拒写, 保持原状

    def test_label_failure_does_not_advance_raw(self, tmp_path):
        """labeled 重建失败时 raw 不得先行落盘 (否则两表静默分裂, 页面 6 列照空).

        真触发: trade_date 非 %Y%m%d → label_seat_frame 内 pd.to_datetime 抛错.
        (不用 monkeypatch —— 函数体顶部的 `from ... import label_seat_frame` 是局部名,
        注入命名空间不起作用.)
        """
        cache = tmp_path / "lhb_seat_detail.parquet"
        labeled = tmp_path / "lhb_seat_detail_labeled.parquet"
        _top_inst_day("20260922").to_parquet(cache, index=False)
        ns = _load_refresh_fn(cache, labeled)
        bad = _top_inst_day("20260923")
        bad["trade_date"] = "NOTADATE"
        try:
            ns["_refresh_lhb_seat_cache"]("20260923", bad)
        except ValueError:
            pass
        old = pd.read_parquet(cache)
        assert len(old) == 2  # 仍是 20260922 那两行, 未被推进
        assert set(old["trade_date"]) == {"20260922"}
        assert not labeled.exists()  # labeled 也没被写坏

    def test_whitespace_duplicate_collapsed(self, tmp_path):
        """同席位同日同 side 仅空白不同 → 合并为一行 (否则该席位买卖额被重复计入)."""
        day = pd.concat(
            [
                _top_inst_day("20260923"),
                _top_inst_day("20260923").assign(
                    exalter=[
                        "机构专用",
                        " 国泰君安证券股份有限公司上海江苏路营业部 ",
                    ]
                ),
            ],
            ignore_index=True,
        )
        got, lab = self._run(tmp_path, day)
        assert len(got) == 2  # 4 行 → 席位空白去重后 2 行
        assert len(lab) == 2
