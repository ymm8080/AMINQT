"""TOP10 第二票 (2026-08-29 用户批准): 周五重训切换闸 = IC 闸 + TOP10 非劣闸.

新包 vs current 生产包同窗 (split_window test 段) 每日板内 top10 头对头,
已实现 label_pm_10d_net 日均净差: 全窗 ≥ 0 且前后半 ≥ tol_half (非劣容差) 才放行。
oos["pass"] 保持纯 IC 语义 (recalibrate 链独读)。

[09-01] 多 seed 集成判词: LGBM run-to-run 方差 ±0.04/日 ≈ 闸信号量级
(08-30 PASS vs 08-31 FAIL 近同配置翻面) → 新包 10d_reg 头按 seeds 重训多次,
各 seed 对 current 求差后按 median 聚合再判; 旧包恒单模型。
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

import config.settings as st
from app.pipeline1.dual_track_trainer import (
    DualTrackTrainer,
    _persist_second_vote_diag,
    top10_verdict,
)


@pytest.fixture(autouse=True)
def _isolated_diag_dir(tmp_path, monkeypatch):
    """判词明细落盘隔离到 tmp — 测试绝不写真实 DATA OTHERS."""
    monkeypatch.setattr(st, "DATA_OTHERS_DIR", str(tmp_path / "DATA OTHERS"))


class TestVerdictRule:
    def _delta(self, h1: float, h2: float) -> pd.Series:
        return pd.Series([h1] * 3 + [h2] * 3)

    def test_both_halves_positive_pass(self):
        v = top10_verdict(self._delta(0.004, 0.002), tol_half=-0.002)
        assert v["pass"] is True
        assert v["delta_full"] == pytest.approx(0.003)

    def test_half_below_tolerance_fail(self):
        assert (
            top10_verdict(self._delta(-0.003, 0.005), tol_half=-0.002)["pass"] is False
        )

    def test_within_tolerance_pass(self):
        assert (
            top10_verdict(self._delta(-0.001, 0.001), tol_half=-0.002)["pass"] is True
        )

    def test_full_negative_fail(self):
        assert (
            top10_verdict(self._delta(-0.001, -0.001), tol_half=-0.002)["pass"] is False
        )


class _RankModel:
    """pred = 方向 × f1 → 排名可控."""

    def __init__(self, sign: float = 1.0):
        self.sign = sign

    def predict(self, X):  # noqa: N802
        return self.sign * X[:, 0]


def _test_frame() -> pd.DataFrame:
    """12 日 × 12 票; f1 决定排名, 标签高位票集中在大 f1 端."""
    rows = []
    d0 = pd.Timestamp("2026-08-27")
    for k in range(12):
        for i in range(12):
            rows.append(
                {
                    "symbol": f"6000{i:02d}",
                    "date": d0 + pd.Timedelta(days=k),
                    "f1": float(i),
                    # 大 f1 票标签高 → 升序模型 (top10=高位票) 赢
                    "label_pm_10d_net": 0.001 * k + 0.001 * i,
                }
            )
    return pd.DataFrame(rows)


def _trained(sign: float, feature_cols=("f1",)) -> dict:
    return {
        "board": "main",
        "feature_cols": list(feature_cols),
        "models": {"10d_reg": (_RankModel(sign), "label_pm_10d_net")},
        "segs": {"test": _test_frame()},
    }


def _cur_bundle(sign: float, feature_cols=("f1",)) -> dict:
    """生产包不含 segs (save 白名单不落盘), 假包同构."""
    b = _trained(sign, feature_cols)
    del b["segs"]
    return b


def _make_trainer(tmp_path) -> DualTrackTrainer:
    tr = DualTrackTrainer.__new__(DualTrackTrainer)
    tr.model_dir = str(tmp_path)
    return tr


def _stub_train_one(tr, sign: float) -> list:
    """顶替 _train_one: 任意 seed 都返回固定方向模型 (免真实 LGBM), 记录 seed 调用."""
    calls: list = []

    def _fake(kind, segs, cols, board, seed=None):
        calls.append(seed)
        return _RankModel(sign), "label_pm_10d_net"

    tr._train_one = _fake
    return calls


def _pin_raw_head(monkeypatch) -> None:
    """raw_head 机制测试钉口径: 生产 config 已切 final_list_tool (委托 retrain 工具),
    这些测试验证的是回退口机制本身, 不随生产口径翻转."""

    cfg = dict(st.LEGACY_TOP10_SECOND_VOTE)
    cfg["caliber"] = "raw_head"
    monkeypatch.setattr(st, "LEGACY_TOP10_SECOND_VOTE", cfg)


class TestSecondVote:
    @pytest.fixture(autouse=True)
    def _caliber_raw_head(self, monkeypatch):
        _pin_raw_head(monkeypatch)

    def test_no_current_vacuous_pass(self, tmp_path):
        v = _make_trainer(tmp_path).top10_second_vote(_trained(sign=1.0))
        assert v["pass"] is True
        assert v.get("skipped") is True

    def test_new_better_passes(self, tmp_path):
        tr = _make_trainer(tmp_path)
        _stub_train_one(tr, 1.0)
        with open(tmp_path / "main_current.pkl", "wb") as fh:
            pickle.dump(_cur_bundle(sign=-1.0), fh)  # 旧包选低位票 → 输
        v = tr.top10_second_vote(_trained(sign=1.0))
        assert v["pass"] is True
        assert v["days"] == 12
        assert v["new_net"] > v["cur_net"]

    def test_new_worse_fails(self, tmp_path):
        tr = _make_trainer(tmp_path)
        _stub_train_one(tr, -1.0)
        with open(tmp_path / "main_current.pkl", "wb") as fh:
            pickle.dump(_cur_bundle(sign=1.0), fh)  # 旧包选高位票 → 赢
        v = tr.top10_second_vote(_trained(sign=-1.0))
        assert v["pass"] is False

    def test_disabled_config_skips(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            st, "LEGACY_TOP10_SECOND_VOTE", {"enable": False, "tol_half": -0.002}
        )
        v = _make_trainer(tmp_path).top10_second_vote(_trained(sign=1.0))
        assert v == {"skipped": True, "pass": True}

    def test_missing_cols_zero_filled(self, tmp_path):
        tr = _make_trainer(tmp_path)
        _stub_train_one(tr, 1.0)
        with open(tmp_path / "main_current.pkl", "wb") as fh:
            pickle.dump(_cur_bundle(sign=-1.0, feature_cols=("f1", "f_nonbrute")), fh)
        v = tr.top10_second_vote(_trained(sign=1.0))
        # f_nonbrute 补 0 → 旧包 pred=0 全并列, 排名退化为稳定序 → 新包仍赢
        assert v["pass"] is True

    def test_top5_split_reported(self, tmp_path):
        tr = _make_trainer(tmp_path)
        _stub_train_one(tr, 1.0)
        with open(tmp_path / "main_current.pkl", "wb") as fh:
            pickle.dump(_cur_bundle(sign=-1.0), fh)
        v = tr.top10_second_vote(_trained(sign=1.0))
        for arm in ("new", "cur"):
            assert set(v["top5_split"][arm]) == {"top5", "b6_10"}


class TestMultiSeedVerdict:
    """多 seed 集成判词 (09-01): 各 seed 判词按 median 聚合, 明细 WORM 落盘."""

    @pytest.fixture(autouse=True)
    def _caliber_raw_head(self, monkeypatch):
        _pin_raw_head(monkeypatch)

    def _setup(self, tmp_path, signs: dict, cur_sign: float = -1.0):
        tr = _make_trainer(tmp_path)
        calls: list = []

        def _fake(kind, segs, cols, board, seed=None):
            calls.append(seed)
            return _RankModel(signs[seed]), "label_pm_10d_net"

        tr._train_one = _fake
        with open(tmp_path / "main_current.pkl", "wb") as fh:
            pickle.dump(_cur_bundle(sign=cur_sign), fh)  # 赢家 vs cur → delta ±0.002/日
        return tr, calls

    def test_median_selects_middle_seed(self, tmp_path):
        tr, calls = self._setup(tmp_path, {42: 1.0, 43: -1.0, 44: 1.0})
        v = tr.top10_second_vote(_trained(sign=1.0))
        assert calls == [42, 43, 44]
        assert v["pass"] is True  # 中位 = 赢家 seed (平局 seed 拉不动中位)
        assert v["delta_full"] == pytest.approx(0.002)  # 非 mean (mean=0.00133)
        assert v["per_seed"]["43"]["delta_full"] == pytest.approx(
            0.0
        )  # 与旧包同款 → 平
        assert v["seeds"] == [42, 43, 44]

    def test_median_two_losers_fail(self, tmp_path):
        tr, _ = self._setup(
            tmp_path, {42: 1.0, 43: -1.0, 44: -1.0}, cur_sign=1.0
        )  # 旧包=赢家; seed42 平 (0), seed43/44 真输 (-0.002)
        v = tr.top10_second_vote(_trained(sign=1.0))
        assert v["pass"] is False  # 中位 = 输家 seed → 单 lucky draw 不再单独放行
        assert v["delta_full"] == pytest.approx(-0.002)

    def test_enabled_run_persists_worm_diag(self, tmp_path):
        tr, _ = self._setup(tmp_path, {42: 1.0, 43: -1.0, 44: 1.0})
        tr.top10_second_vote(_trained(sign=1.0))
        files = list(
            Path(st.DATA_OTHERS_DIR, "diag").glob("top10_second_vote_main_*.json")
        )
        assert len(files) == 1
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        assert payload["median_verdict"]["pass"] is True
        assert payload["agg"] == "median"
        assert set(payload["per_seed"]) == {"42", "43", "44"}
        for s in payload["per_seed"].values():
            assert len(s["delta_daily"]) == 12  # per-seed per-day deltas

    def test_disabled_multi_seed_behaves_legacy(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            st,
            "LEGACY_TOP10_SECOND_VOTE",
            {
                "enable": True,
                "tol_half": -0.002,
                "top_n": 10,
                "multi_seed_enable": False,
            },
        )
        tr, calls = self._setup(tmp_path, {None: 1.0})
        v = tr.top10_second_vote(_trained(sign=1.0))
        assert calls == []  # 不重训
        assert v["pass"] is True
        assert "seeds" not in v and "per_seed" not in v
        assert not Path(st.DATA_OTHERS_DIR, "diag").exists()  # 不落盘

    def test_config_defaults_multi_seed_on(self):
        cfg = st.LEGACY_TOP10_SECOND_VOTE
        assert cfg["multi_seed_enable"] is True
        assert cfg["multi_seed_seeds"] == [42, 43, 44]
        assert cfg["multi_seed_agg"] == "median"


class TestFinalListToolDelegation:
    """[09-02] caliber=final_list_tool: trainer 只留 IC 闸, 判决权移交
    retrain 脚本内终榜回放工具 — 不加载 current, 不做多 seed 重训."""

    def test_delegates_before_loading_current(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            st,
            "LEGACY_TOP10_SECOND_VOTE",
            {"enable": True, "caliber": "final_list_tool", "tol_half": -0.005},
        )
        tr = _make_trainer(tmp_path)
        calls = _stub_train_one(tr, 1.0)
        with open(tmp_path / "main_current.pkl", "wb") as fh:
            pickle.dump(_cur_bundle(sign=-1.0), fh)
        v = tr.top10_second_vote(_trained(sign=1.0))
        assert v == {
            "skipped": True,
            "pass": True,
            "reason": "delegated_finaltop_tool",
        }
        assert calls == []  # 多 seed 重训不再发生
        assert not Path(st.DATA_OTHERS_DIR, "diag").exists()  # 不落盘

    def test_config_default_caliber_is_final_list_tool(self):
        cfg = st.LEGACY_TOP10_SECOND_VOTE
        assert cfg["caliber"] == "final_list_tool"
        assert cfg["enable"] is True
        assert cfg["tol_half"] == -0.005
        assert cfg["win_rate_min"] == 0.5
        assert cfg["eval_days"] == 48
        assert cfg["min_days"] == 10


class TestWormDiagFile:
    def test_never_overwrites_existing(self, tmp_path):
        p1 = _persist_second_vote_diag("main", {"a": 1}, ts="20260901_120000")
        first = Path(p1).read_text(encoding="utf-8")
        p2 = _persist_second_vote_diag("main", {"a": 2}, ts="20260901_120000")
        assert p2 != p1
        assert "20260901_120000" in Path(p2).name
        assert Path(p1).read_text(encoding="utf-8") == first  # 首文件字节不动
        assert json.loads(Path(p2).read_text(encoding="utf-8"))["a"] == 2
