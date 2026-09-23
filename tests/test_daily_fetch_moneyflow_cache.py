"""_daily_fetch moneyflow 缓存测试 (2026-09-23 正式化).

资产 = PARQUET/moneyflow_daily.parquet (3 年回填 4.6M 行).
测试点: 派生列算式与回填脚本一致 / 幂等覆盖 / skeleton guard / 空数据拒写.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def _load_moneyflow_fn(cache_path):
    """从 _daily_fetch.py 源码提取 _refresh_moneyflow_cache 函数体独立执行.

    _daily_fetch.py 是 import 即跑全链的顶层脚本, 不能裸 import; 走 AST 提取
    (与 tests/test_daily_fetch_default_date.py 同思路). safe_fetch/pro 用 stub
    注入, 保证目标函数可独立执行.
    """
    src = (REPO / "_daily_fetch.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(src)
    keep = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_refresh_moneyflow_cache"
    ]
    assert keep, "_refresh_moneyflow_cache must exist in _daily_fetch.py"
    new_tree = ast.Module(body=keep, type_ignores=[])
    namespace: dict = {
        "os": __import__("os"),
        "pd": pd,
        "np": np,
        "time": __import__("time"),
        "MF_CACHE_PATH": str(cache_path),
        "safe_fetch": lambda fn, name, **kw: fn(**kw),
    }
    exec(compile(new_tree, "<moneyflow_extract>", "exec"), namespace)
    return namespace


class _FakePro:
    def __init__(self, day_df):
        self._day_df = day_df
        self.calls: list = []

    def moneyflow(self, trade_date=None):
        self.calls.append(trade_date)
        return self._day_df.copy()


def _day_frame(trade_date="20260923"):
    return pd.DataFrame(
        {
            "trade_date": [trade_date, trade_date],
            "ts_code": ["000001.SZ", "600000.SH"],
            "buy_lg_amount": [100.0, 10.0],
            "sell_lg_amount": [30.0, 0.0],
            "buy_elg_amount": [50.0, 5.0],
            "sell_elg_amount": [20.0, 0.0],
            "net_mf_amount": [100.0, 15.0],
            "buy_sm_amount": [1.0, 1.0],
            "sell_sm_amount": [2.0, 2.0],
        }
    )


class TestRefreshMoneyflowCache:
    def _run(self, tmp_path, day_df, preexisting=None):
        cache = tmp_path / "moneyflow_daily.parquet"
        if preexisting is not None:
            preexisting.to_parquet(cache, index=False)
        ns = _load_moneyflow_fn(cache)
        ns["pro"] = _FakePro(day_df)
        ns["_refresh_moneyflow_cache"]("20260923")
        # 函数体读模块级 MF_CACHE_PATH → exec 命名空间里它是全局变量
        return pd.read_parquet(cache), ns["pro"]

    def test_derived_cols_match_backfill_formula(self, tmp_path):
        got, _ = self._run(tmp_path, _day_frame())
        r0 = got[got.ts_code == "000001.SZ"].iloc[0]
        assert r0["mf_main_net"] == (100 - 30) + (50 - 20)
        assert r0["mf_inst_net"] == 50 - 20
        r1 = got[got.ts_code == "600000.SH"].iloc[0]
        assert r1["mf_main_net"] == 15.0

    def test_idempotent_same_day_overwrite(self, tmp_path):
        old = _day_frame("20260923")
        old.loc[old.ts_code == "000001.SZ", "net_mf_amount"] = 999.0  # 陈旧值
        got, _ = self._run(tmp_path, _day_frame("20260923"), preexisting=old)
        assert len(got) == 2  # 覆盖不追加
        v = got[got.ts_code == "000001.SZ"].iloc[0]["net_mf_amount"]
        assert v == 100.0  # 新值胜出, 999 消失

    def test_appends_new_day_keeps_old(self, tmp_path):
        prev = _day_frame("20260922")
        got, _ = self._run(tmp_path, _day_frame("20260923"), preexisting=prev)
        assert len(got) == 4
        assert set(got.trade_date) == {"20260922", "20260923"}

    def test_empty_day_skips_write(self, tmp_path):
        cache = tmp_path / "moneyflow_daily.parquet"
        ns = _load_moneyflow_fn(cache)
        ns["pro"] = _FakePro(pd.DataFrame())
        ns["_refresh_moneyflow_cache"]("20260923")
        assert not cache.exists()  # 空数据: 缓存文件根本不写
        assert ns["pro"].calls == ["20260923"]

    def test_skeleton_response_refused(self, tmp_path):
        cache = tmp_path / "moneyflow_daily.parquet"
        prev = _day_frame("20260922")
        prev.to_parquet(cache, index=False)
        skel = pd.DataFrame(
            {
                "trade_date": ["20260923"],
                "ts_code": ["000001.SZ"],
                "buy_lg_amount": [np.nan],
                "sell_lg_amount": [np.nan],
                "buy_elg_amount": [np.nan],
                "sell_elg_amount": [np.nan],
            }
        )
        ns = _load_moneyflow_fn(cache)
        ns["pro"] = _FakePro(skel)
        ns["_refresh_moneyflow_cache"]("20260923")
        got = pd.read_parquet(cache)
        assert len(got) == 2  # 骨架响应拒写, 缓存保持昨日原状

    def test_no_legacy_cols_leak(self, tmp_path):
        got, _ = self._run(tmp_path, _day_frame())
        assert not {"buy_sm_amount", "sell_sm_amount"} & set(got.columns)
        assert list(got.columns) == [
            "trade_date", "ts_code",
            "buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount",
            "net_mf_amount", "mf_main_net", "mf_inst_net",
        ]
