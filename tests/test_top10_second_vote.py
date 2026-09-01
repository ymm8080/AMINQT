"""TOP10 第二票 (2026-08-29 用户批准): 周五重训切换闸 = IC 闸 + TOP10 非劣闸.

新包 vs current 生产包同窗 (split_window test 段) 每日板内 top10 头对头,
已实现 label_pm_10d_net 日均净差: 全窗 ≥ 0 且前后半 ≥ tol_half (非劣容差) 才放行。
oos["pass"] 保持纯 IC 语义 (recalibrate 链独读)。
"""

from __future__ import annotations

import pickle

import pandas as pd
import pytest

from app.pipeline1.dual_track_trainer import DualTrackTrainer, top10_verdict


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


class TestSecondVote:
    def test_no_current_vacuous_pass(self, tmp_path):
        v = _make_trainer(tmp_path).top10_second_vote(_trained(sign=1.0))
        assert v["pass"] is True
        assert v.get("skipped") is True

    def test_new_better_passes(self, tmp_path):
        tr = _make_trainer(tmp_path)
        with open(tmp_path / "main_current.pkl", "wb") as fh:
            pickle.dump(_cur_bundle(sign=-1.0), fh)  # 旧包选低位票 → 输
        v = tr.top10_second_vote(_trained(sign=1.0))
        assert v["pass"] is True
        assert v["days"] == 12
        assert v["new_net"] > v["cur_net"]

    def test_new_worse_fails(self, tmp_path):
        tr = _make_trainer(tmp_path)
        with open(tmp_path / "main_current.pkl", "wb") as fh:
            pickle.dump(_cur_bundle(sign=1.0), fh)  # 旧包选高位票 → 赢
        v = tr.top10_second_vote(_trained(sign=-1.0))
        assert v["pass"] is False

    def test_disabled_config_skips(self, tmp_path, monkeypatch):
        import config.settings as st

        monkeypatch.setattr(
            st, "LEGACY_TOP10_SECOND_VOTE", {"enable": False, "tol_half": -0.002}
        )
        v = _make_trainer(tmp_path).top10_second_vote(_trained(sign=1.0))
        assert v == {"skipped": True, "pass": True}

    def test_missing_cols_zero_filled(self, tmp_path):
        tr = _make_trainer(tmp_path)
        with open(tmp_path / "main_current.pkl", "wb") as fh:
            pickle.dump(_cur_bundle(sign=-1.0, feature_cols=("f1", "f_nonbrute")), fh)
        v = tr.top10_second_vote(_trained(sign=1.0))
        # f_nonbrute 补 0 → 旧包 pred=0 全并列, 排名退化为稳定序 → 新包仍赢
        assert v["pass"] is True

    def test_top5_split_reported(self, tmp_path):
        tr = _make_trainer(tmp_path)
        with open(tmp_path / "main_current.pkl", "wb") as fh:
            pickle.dump(_cur_bundle(sign=-1.0), fh)
        v = tr.top10_second_vote(_trained(sign=1.0))
        for arm in ("new", "cur"):
            assert set(v["top5_split"][arm]) == {"top5", "b6_10"}
