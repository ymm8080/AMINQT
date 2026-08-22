"""legacy LEGACY_PROB_GATE 接线测试 (list_generator.emit + daily_pipeline, 2026-08-16).

仿 parallel 侧接线口径 (tests/test_prob_gate.py 模块闸 + _shortlist_t5_t10 链路):
- 闸在 t3 门 (entry_filter) 后、pred_ret_10d 排名前 — 排名键保持纯 mag (prob 只作闸)
- fail-open: 未传输入 / 闸不可用 / 概率头异常 (schema 漂移) → 不杀清单
- 生产 board 命名 (main/GEM/STAR) → 闸按 main/dual 分组, 输出 board 原样保留
- daily_pipeline 输入组装: feats=当日截面 / tail=清洗帧尾 (adv20 由 amount 现算,
  无前瞻) / panel_dates=面板全局交易日 (bundle staleness 判定)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.pipeline1 import prob_head
from app.pipeline1.list_generator import ListGenerator

GATE_OFF = {"entry_prob": 0.0, "entry_ret_mult": 0.0}


@pytest.fixture(autouse=True)
def _hermetic_risk_scans(tmp_path, monkeypatch):
    """FINAL STOCK SCAN 隔离: 空缓存 → 永不剔除 (不依赖外部缓存文件状态)."""
    from app.pipeline1 import risk_overlays

    empty = tmp_path / "empty_block_trade.parquet"
    pd.DataFrame(columns=["symbol", "date"]).to_parquet(empty, index=False)
    real = risk_overlays.block_trade_recent_scan

    def _scan(symbols, ref_date, **kwargs):
        kwargs["cache_path"] = str(empty)
        return real(symbols, ref_date, **kwargs)

    monkeypatch.setattr(risk_overlays, "block_trade_recent_scan", _scan)

    empty_sf = tmp_path / "empty_share_float.parquet"
    pd.DataFrame(
        columns=["symbol", "ann_date", "float_date", "float_ratio"]
    ).to_parquet(empty_sf, index=False)
    real_sf = risk_overlays.share_float_upcoming_scan

    def _scan_sf(symbols, ref_date, **kwargs):
        kwargs["cache_path"] = str(empty_sf)
        return real_sf(symbols, ref_date, **kwargs)

    monkeypatch.setattr(risk_overlays, "share_float_upcoming_scan", _scan_sf)


def make_candidates() -> pd.DataFrame:
    """5 只候选: main 3 (pred_ret_10d 降序) + GEM/STAR 各 1; GATE_OFF 下全过 t3."""
    return pd.DataFrame(
        {
            "symbol": ["600001", "600002", "600003", "300001", "688001"],
            "board": ["main", "main", "main", "GEM", "STAR"],
            "pred_ret_3d": [0.01] * 5,
            "pred_ret_5d": [0.01] * 5,
            "pred_ret_10d": [0.030, 0.020, 0.010, 0.005, 0.004],
            "prob_up": [0.6] * 5,
            "prob_up_3d": [0.6] * 5,
            "prob_up_5d": [0.6] * 5,
            "prob_up_10d": [0.6] * 5,
        }
    )


def _panel_dates(n=80) -> np.ndarray:
    return pd.to_datetime(pd.date_range("2026-06-01", periods=n, freq="B")).to_numpy(
        "datetime64[ns]"
    )


def _gate_inputs() -> dict:
    return {
        "feats": {
            "main": pd.DataFrame({"symbol": ["600001", "600002", "600003"]}),
            "dual": pd.DataFrame({"symbol": ["300001", "688001"]}),
        },
        "tail": pd.DataFrame(),
        "panel_dates": _panel_dates(),
    }


def test_emit_prob_gate_drops_and_keeps_pure_mag_ranking(monkeypatch):
    """闸剔低概率股; 排名保持纯 pred_ret_10d (高 prob 不得插队); board 原样保留.

    thr = base_rate + margin = 0.20 + 0.22 = 0.42 (定案边际被钉住):
    main 600002 (prob 0.10) 剔; 600003 prob 0.90 > 600001 prob 0.60, 但排名仍按
    mag (0.030 > 0.010) — prob 只作闸不改排名 (blend 已证伪).
    dual (GEM/STAR) 不在 gated_boards → 不过闸.
    """
    seen: dict[str, list[str]] = {}

    def fake_gate(board, feat_day, tail, panel_dates):
        seen[board] = sorted(feat_day["symbol"].tolist())
        if board == "main":
            return (
                pd.Series({"600001": 0.60, "600002": 0.10, "600003": 0.90}),
                0.20,
            )
        return (pd.Series({"300001": 0.90, "688001": 0.90}), 0.20)

    monkeypatch.setattr(prob_head, "gate_probabilities", fake_gate)
    out = ListGenerator(**GATE_OFF).emit(make_candidates(), prob_gate=_gate_inputs())
    lst = out["list"]
    assert not out["empty"]
    # 600002 被闸剔; 余下按纯 mag 排名 (高 prob 的 600003 不插队)
    assert lst["symbol"].tolist() == ["600001", "600003", "300001", "688001"]
    # 生产 board 命名原样保留 (GEM/STAR 不被改写为 dual)
    assert lst["board"].tolist() == ["main", "main", "GEM", "STAR"]
    # 08-22 定案: 仅 main 过闸 (dual 撤闸, GEM/STAR 不过闸)
    assert set(seen) == {"main"}


def test_emit_gate_unavailable_failopen(monkeypatch):
    """闸不可用 (bundle 缺失/过旧) → fail-open 全留, 排名不变."""
    monkeypatch.setattr(prob_head, "gate_probabilities", lambda *a: None)
    out = ListGenerator(**GATE_OFF).emit(make_candidates(), prob_gate=_gate_inputs())
    assert out["list"]["symbol"].tolist() == [
        "600001",
        "600002",
        "600003",
        "300001",
        "688001",
    ]


def test_emit_gate_drops_all_returns_empty(monkeypatch):
    """闸可用且全部候选低于门槛 → 今日空清单 (宁缺毋滥)."""

    def fake_gate(board, feat_day, tail, panel_dates):
        return (pd.Series({"600001": 0.0, "600002": 0.0, "600003": 0.0}), 0.20)

    monkeypatch.setattr(prob_head, "gate_probabilities", fake_gate)
    cands = make_candidates()[make_candidates()["board"] == "main"]
    out = ListGenerator(**GATE_OFF).emit(cands, prob_gate=_gate_inputs())
    assert out["empty"] and len(out["list"]) == 0


def test_emit_gate_raise_failopen(monkeypatch):
    """概率头异常 (特征缺列 = schema 漂移) → 大声告警后 fail-open, 不杀清单."""

    def boom(res, feats, tail, panel_dates):
        raise ValueError("schema 漂移: 缺概率头特征列")

    monkeypatch.setattr(prob_head, "apply_prob_gate", boom)
    out = ListGenerator(**GATE_OFF).emit(make_candidates(), prob_gate=_gate_inputs())
    assert out["list"]["symbol"].tolist() == [
        "600001",
        "600002",
        "600003",
        "300001",
        "688001",
    ]


def test_emit_without_gate_inputs_skips_gate(monkeypatch):
    """未传 prob_gate (非生产调用方/手动脚本) → 闸完全跳过, 行为与接线前一致."""

    def boom(*a, **k):
        raise AssertionError("未传输入时不得调用概率闸")

    monkeypatch.setattr(prob_head, "apply_prob_gate", boom)
    out = ListGenerator(**GATE_OFF).emit(make_candidates())
    assert len(out["list"]) == 5


def test_emit_gate_runs_before_topn_truncation(monkeypatch):
    """闸在 TOP-N 截断前 (parallel gate-then-rank 同口径): 被闸剔的票让下一只顺延补位.

    17 只候选 > TOP_N=15: 闸剔最高 mag 票 → 若闸在截断后, 清单只剩 14 只;
    补位至 15 只证明闸作用于 t3 门后、排名截断前的候选池.
    """
    n = 17
    cands = pd.DataFrame(
        {
            "symbol": [f"6000{i:02d}" for i in range(n)],
            "board": ["main"] * n,
            "pred_ret_3d": [0.01] * n,
            "pred_ret_5d": [0.01] * n,
            "pred_ret_10d": [0.001 * (n - i) for i in range(n)],  # 600000 最高 mag
            "prob_up": [0.6] * n,
            "prob_up_3d": [0.6] * n,
            "prob_up_5d": [0.6] * n,
            "prob_up_10d": [0.6] * n,
        }
    )
    top = "600000"

    def fake_gate(board, feat_day, tail, panel_dates):
        prob = pd.Series({s: 0.60 for s in cands["symbol"]})
        prob[top] = 0.10  # 最高 mag 被闸剔 (0.10 ≤ 0.42)
        return (prob, 0.20)

    monkeypatch.setattr(prob_head, "gate_probabilities", fake_gate)
    out = ListGenerator(**GATE_OFF).emit(cands, prob_gate=_gate_inputs())
    lst = out["list"]
    assert len(lst) == 15  # TOP_N=15 照常满员 → 补位证明闸在截断前
    assert top not in lst["symbol"].tolist()
    # 排名纯 mag (prob 只作闸): 余下按 mag 降序 = 600001..600015
    assert lst["symbol"].tolist() == [f"6000{i:02d}" for i in range(1, 16)]


def test_apply_prob_gate_groups_gem_star_as_dual(monkeypatch):
    """显式放开 dual (gated_boards 含 dual): GEM/STAR 并入 dual 组过闸, 输出 board 原样."""
    monkeypatch.setitem(
        prob_head.LEGACY_PROB_GATE, "gated_boards", ["main", "dual"]
    )
    res = pd.DataFrame(
        {"board": ["main", "GEM", "STAR"], "symbol": ["600001", "300001", "688001"]}
    )
    calls: list[str] = []

    def fake_gate(board, feat_day, tail, panel_dates):
        calls.append(board)
        if board == "main":
            return (pd.Series({"600001": 0.60}), 0.20)
        return (pd.Series({"300001": 0.10, "688001": 0.90}), 0.20)

    monkeypatch.setattr(prob_head, "gate_probabilities", fake_gate)
    feats = {"main": pd.DataFrame(), "dual": pd.DataFrame()}
    out = prob_head.apply_prob_gate(res, feats, pd.DataFrame(), _panel_dates())
    assert calls == ["main", "dual"]
    # 300001 (0.10 ≤ 0.42) 剔; 688001 留; board 值不被改写
    assert out[["board", "symbol"]].values.tolist() == [
        ["main", "600001"],
        ["STAR", "688001"],
    ]


def test_apply_prob_gate_default_gates_main_only(monkeypatch):
    """08-22 定案默认: gated_boards=['main'] → dual/GEM/STAR 不过闸 (撤闸)."""
    res = pd.DataFrame(
        {"board": ["main", "GEM", "STAR"], "symbol": ["600001", "300001", "688001"]}
    )
    calls: list[str] = []

    def fake_gate(board, feat_day, tail, panel_dates):
        calls.append(board)
        return (pd.Series({"600001": 0.10}), 0.20)  # main 最低 prob → 剔

    monkeypatch.setattr(prob_head, "gate_probabilities", fake_gate)
    feats = {"main": pd.DataFrame(), "dual": pd.DataFrame()}
    out = prob_head.apply_prob_gate(res, feats, pd.DataFrame(), _panel_dates())
    assert calls == ["main"]  # 仅 main 过闸
    # main 600001 被剔; GEM/STAR 不过闸 → 原样保留
    assert out[["board", "symbol"]].values.tolist() == [
        ["GEM", "300001"],
        ["STAR", "688001"],
    ]


def test_prob_gate_inputs_builds_feats_tail_panel_dates():
    """daily_pipeline 输入组装: tail=近 base_rate_days+14 交易日清洗帧尾,
    adv20 由 amount 20 日均值现算 (清洗帧无 adv20); panel_dates=面板全局交易日."""
    from app.pipeline1.daily_pipeline import DailySelectionPipeline

    n = 60
    dates = pd.bdate_range("2026-05-01", periods=n)

    def frame(sym):
        return pd.DataFrame(
            {
                "symbol": [sym] * n,
                "date": dates,
                "board": "main" if sym.startswith("6") else "STAR",
                "close_hfq": [10.0] * n,
                "high_hfq": [12.0] * n,
                "amount": [1e9] * n,
            }
        )

    main_df = pd.concat([frame("600001"), frame("600002")], ignore_index=True)
    dual_df = frame("300001")
    feats = {
        "main": pd.DataFrame({"symbol": ["600001"], "f1": [1.0]}),
        "dual": pd.DataFrame({"symbol": ["300001"], "f1": [1.0]}),
    }
    panel = pd.concat([main_df, dual_df], ignore_index=True)
    out = DailySelectionPipeline._prob_gate_inputs(panel, main_df, dual_df, feats)

    assert set(out) == {"feats", "tail", "panel_dates"}
    assert out["feats"]["main"] is feats["main"]
    assert out["feats"]["dual"] is feats["dual"]
    tail = out["tail"]
    assert list(tail.columns) == ["symbol", "date", "close_hfq", "high_hfq", "adv20"]
    n_tail = prob_head.LEGACY_PROB_GATE["base_rate_days"] + 14
    uniq = sorted(pd.to_datetime(tail["date"]).unique())
    assert uniq == list(dates[-n_tail:])  # 只保留近 base_rate_days+14 交易日
    assert np.isclose(tail["adv20"].iloc[0], 1e9)  # amount 20 日均值现算
    assert np.array_equal(
        out["panel_dates"], pd.to_datetime(dates).to_numpy("datetime64[ns]")
    )


@pytest.fixture()
def stubbed_pipe(tmp_path, monkeypatch):
    """run() 接线测试: 清洗/特征/预测/清单全 stub, 只验证概率闸输入组装与传递."""
    from app.pipeline1 import pred_smoothing
    from app.pipeline1.daily_pipeline import DailySelectionPipeline
    from app.pipeline1.data_supply import DataSupplyChain

    n = 60
    dates = pd.bdate_range("2026-05-01", periods=n)
    main_df = pd.DataFrame(
        {
            "symbol": ["600001"] * n,
            "date": dates,
            "board": "main",
            "close_hfq": [10.0] * n,
            "high_hfq": [12.0] * n,
            "amount": [1e9] * n,
        }
    )
    dual_df = pd.DataFrame(
        {
            "symbol": ["300001"] * n,
            "date": dates,
            "board": "STAR",
            "close_hfq": [10.0] * n,
            "high_hfq": [12.0] * n,
            "amount": [1e9] * n,
        }
    )
    panel = pd.concat([main_df, dual_df], ignore_index=True)

    class StubCleaner:
        def run_inference(self, panel, pool_blend=None):
            return main_df, dual_df, "open"

        def pool_blend_cut(self, df, pred_col="pred_ret_10d", w=None):
            return df  # 桩: 不切池 (池内成员在测试数据中全保留)

    class StubFeatures:
        def build(self, df, float_shares_map=None, **kwargs):
            return df

    class StubPredictor:
        def __init__(self):
            self.bundles = {
                "main": {"feature_cols": ["f1"]},
                "dual": {"feature_cols": ["f1"]},
            }

        def predict(self, features, board):
            syms = sorted(features["symbol"].unique())
            return pd.DataFrame({"symbol": syms, "board": [board] * len(syms)})

    captured: dict = {}

    class StubLister:
        def emit(self, candidates, env=None, market_state="range", prob_gate=None):
            captured["prob_gate"] = prob_gate
            return {
                "mode": "normal",
                "list": candidates,
                "empty": True,
                "cap_position": 1.0,
                "schema_version": "1.4",
            }

    pipe = DailySelectionPipeline(
        supply=DataSupplyChain(cache_dir=str(tmp_path / "cache")),
        bundle_paths={},
        list_dir=str(tmp_path / "lists"),
    )
    pipe.cleaner = StubCleaner()
    pipe.features = StubFeatures()
    pipe.predictor = StubPredictor()
    pipe.lister = StubLister()
    # 预测平滑/历史底稿 → 不写真实目录 (lazy import 时从模块取, monkeypatch 生效)
    monkeypatch.setattr(pred_smoothing, "persist_raw_preds", lambda *a, **k: None)
    monkeypatch.setattr(pred_smoothing, "smooth_preds", lambda df, d, m: df)
    return pipe, captured, panel


def test_run_passes_prob_gate_inputs_to_emit(stubbed_pipe):
    """生产链路: run() 组装 feats/tail/panel_dates 并传给 emit (list_generator 接线)."""
    pipe, captured, panel = stubbed_pipe
    pipe.run("20260814", panel=panel)
    pg = captured["prob_gate"]
    assert pg is not None
    assert set(pg) == {"feats", "tail", "panel_dates"}
    # feats = 各板当日截面 (只有最新交易日一行, 非全历史)
    assert set(pg["feats"]) == {"main", "dual"}
    assert pg["feats"]["main"]["symbol"].tolist() == ["600001"]
    assert len(pg["feats"]["main"]) == 1
    assert pg["feats"]["dual"]["symbol"].tolist() == ["300001"]
    assert set(pg["tail"].columns) == {
        "symbol",
        "date",
        "close_hfq",
        "high_hfq",
        "adv20",
    }
    assert len(np.unique(pg["panel_dates"])) == 60


def test_run_prob_gate_disabled_passes_none(stubbed_pipe, monkeypatch):
    """LEGACY_PROB_GATE.enable=False → 不组装输入 (省计算), emit 收到 None."""
    from config import settings

    monkeypatch.setitem(settings.LEGACY_PROB_GATE, "enable", False)
    pipe, captured, panel = stubbed_pipe
    pipe.run("20260814", panel=panel)
    assert captured["prob_gate"] is None
