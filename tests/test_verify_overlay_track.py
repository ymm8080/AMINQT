"""前向跟踪验证器单元测试 (scripts/_verify_overlay_track.py).

覆盖: per_date_summary 上下半区 MFE/上涨率对比 + Spearman 一致性 / 未来价不足时
该视界 n=0 与 None / load_snapshots 收集 concat (临时 CSV).
"""

from __future__ import annotations

import pandas as pd
import pytest

import scripts._verify_overlay_track as vt

H = ["label_mfe_2d_net", "label_mfe_3d_net", "label_mfe_5d_net", "label_mfe_10d_net"]


def _merged(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in H:
        df[c] = df[c].astype("float64")
    return df


class TestPerDateSummary:
    def test_top_half_beats_bottom_when_score_aligned(self):
        """final_score 与已实现 3d MFE 同向 → top 半区 MFE/上涨率更高, Spearman≈1."""
        m = _merged(
            [
                {
                    "board": "main",
                    "date": "2026-08-04",
                    "symbol": "000001",
                    "final_score": 0.90,
                    "prob_up": 0.60,
                    "w_pool": 0.2,
                    "w_prob": 0.8,
                    "prob_col": "prob_up_3d",
                    "label_mfe_2d_net": 0.03,
                    "label_mfe_3d_net": 0.05,
                    "label_mfe_5d_net": 0.08,
                    "label_mfe_10d_net": 0.10,
                },
                {
                    "board": "main",
                    "date": "2026-08-04",
                    "symbol": "000002",
                    "final_score": 0.80,
                    "prob_up": 0.55,
                    "w_pool": 0.2,
                    "w_prob": 0.8,
                    "prob_col": "prob_up_3d",
                    "label_mfe_2d_net": 0.02,
                    "label_mfe_3d_net": 0.03,
                    "label_mfe_5d_net": 0.06,
                    "label_mfe_10d_net": 0.09,
                },
                {
                    "board": "main",
                    "date": "2026-08-04",
                    "symbol": "000003",
                    "final_score": 0.70,
                    "prob_up": 0.50,
                    "w_pool": 0.2,
                    "w_prob": 0.8,
                    "prob_col": "prob_up_3d",
                    "label_mfe_2d_net": 0.01,
                    "label_mfe_3d_net": 0.01,
                    "label_mfe_5d_net": 0.03,
                    "label_mfe_10d_net": 0.04,
                },
                {
                    "board": "main",
                    "date": "2026-08-04",
                    "symbol": "000004",
                    "final_score": 0.60,
                    "prob_up": 0.45,
                    "w_pool": 0.2,
                    "w_prob": 0.8,
                    "prob_col": "prob_up_3d",
                    "label_mfe_2d_net": -0.01,
                    "label_mfe_3d_net": -0.02,
                    "label_mfe_5d_net": 0.01,
                    "label_mfe_10d_net": 0.02,
                },
                {
                    "board": "main",
                    "date": "2026-08-04",
                    "symbol": "000005",
                    "final_score": 0.50,
                    "prob_up": 0.40,
                    "w_pool": 0.2,
                    "w_prob": 0.8,
                    "prob_col": "prob_up_3d",
                    "label_mfe_2d_net": -0.03,
                    "label_mfe_3d_net": -0.04,
                    "label_mfe_5d_net": -0.01,
                    "label_mfe_10d_net": 0.00,
                },
            ]
        )
        r = vt.per_date_summary(m)["main|2026-08-04"]
        assert r["n"] == 5 and r["half"] == 2
        p3 = r["per_horizon"]["3d"]
        assert p3["n"] == 5
        assert p3["top_mfe"] == pytest.approx(0.04)  # (0.05+0.03)/2
        assert p3["bot_mfe"] == pytest.approx(-0.016667)  # (0.01-0.02-0.04)/3, _m 6位
        assert p3["top_wr"] == 1.0 and p3["bot_wr"] == pytest.approx(1 / 3)
        assert p3["top_minus_bot"] == pytest.approx(0.056667)
        assert r["spearman_3d"]["final_score"] == pytest.approx(1.0)
        assert r["w_pool"] == 0.2 and r["w_prob"] == 0.8
        assert r["prob_col"] == "prob_up_3d"

    def test_no_future_data_yields_nan_n0(self):
        """距选股日不足 h+1 交易日 → 该视界全部 n=0, mfe/spearman=None (不崩)."""
        m = _merged(
            [
                {
                    "board": "dual",
                    "date": "2026-08-04",
                    "symbol": "300001",
                    "final_score": 0.80,
                    "prob_up": 0.60,
                    "w_pool": 0.5,
                    "w_prob": 0.5,
                    "prob_col": "prob_up_3d",
                    "label_mfe_2d_net": None,
                    "label_mfe_3d_net": None,
                    "label_mfe_5d_net": None,
                    "label_mfe_10d_net": None,
                },
                {
                    "board": "dual",
                    "date": "2026-08-04",
                    "symbol": "300002",
                    "final_score": 0.70,
                    "prob_up": 0.55,
                    "w_pool": 0.5,
                    "w_prob": 0.5,
                    "prob_col": "prob_up_3d",
                    "label_mfe_2d_net": None,
                    "label_mfe_3d_net": None,
                    "label_mfe_5d_net": None,
                    "label_mfe_10d_net": None,
                },
            ]
        )
        r = vt.per_date_summary(m)["dual|2026-08-04"]
        for h in ("2d", "3d", "5d", "10d"):
            p = r["per_horizon"][h]
            assert p["n"] == 0
            assert p["top_mfe"] is None and p["bot_mfe"] is None
        assert r["spearman_3d"]["final_score"] is None

    def test_spearman_requires_min_n(self):
        """样本 <10 → Spearman 返回 None (小样本不硬出结论)."""
        m = _merged(
            [
                {
                    "board": "main",
                    "date": "2026-08-04",
                    "symbol": f"{i:06d}",
                    "final_score": i / 10,
                    "prob_up": 0.5,
                    "w_pool": 0.2,
                    "w_prob": 0.8,
                    "prob_col": "prob_up_3d",
                    **{c: (i - 5) / 100 for c in H},
                }
                for i in range(1, 5)
            ]
        )
        r = vt.per_date_summary(m)["main|2026-08-04"]
        assert r["spearman_3d"]["final_score"] is None


class TestLoadSnapshots:
    def test_glob_concat(self, tmp_path):
        a = tmp_path / "overlay_track_20260804__main.pkl.csv"
        b = tmp_path / "overlay_track_20260805__main.pkl.csv"
        a.write_text(
            "board,date,symbol,final_score\nmain,2026-08-04,000001,0.9\n",
            encoding="utf-8",
        )
        b.write_text(
            "board,date,symbol,final_score\nmain,2026-08-05,000002,0.8\n",
            encoding="utf-8",
        )
        snap = vt.load_snapshots(tmp_path)
        assert len(snap) == 2
        # pandas 3.0: dtype=str 返回 StringDtype 而非 object; 两者均可保前导零
        assert pd.api.types.is_string_dtype(snap["symbol"])
        assert list(snap["date"]) == ["2026-08-04", "2026-08-05"]

    def test_empty_dir(self, tmp_path):
        assert vt.load_snapshots(tmp_path).empty
