"""LEGACY 推理叠加模块单元测试 (app/pipeline_parallel/legacy_overlay.py).

覆盖: rerank 合成再排 (共现优先/组内 final_score 降序/rk 连续性) /
prob 列回退 / 空 legacy 降级 / merge 无 _x/_y 后缀冲突 /
legacy_predict 空输入 / cross_section 按日期过滤 (临时 parquet).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app.pipeline_parallel.legacy_overlay as ov


def _shortlist() -> pd.DataFrame:
    """构造 build_merged_shortlist 等价输出 (date/symbol/systems/score/co_occur/rk)."""
    return pd.DataFrame(
        [
            # 共现优先
            {"date": pd.Timestamp("2026-08-04"), "symbol": "000001", "systems": "fusion+sniper", "score": 0.90, "co_occur": True},
            {"date": pd.Timestamp("2026-08-04"), "symbol": "000002", "systems": "fusion+sniper", "score": 0.80, "co_occur": True},
            # 单系统, score 高者在前
            {"date": pd.Timestamp("2026-08-04"), "symbol": "000003", "systems": "sniper", "score": 0.95, "co_occur": False},
            {"date": pd.Timestamp("2026-08-04"), "symbol": "000004", "systems": "fusion", "score": 0.85, "co_occur": False},
            {"date": pd.Timestamp("2026-08-04"), "symbol": "000005", "systems": "fusion", "score": 0.70, "co_occur": False},
        ],
    ).assign(rk=[1, 2, 3, 4, 5])


def _legacy(probs=(0.6, 0.5, 0.9, 0.3, 0.4)) -> pd.DataFrame:
    """构造 V35Predictor.predict 输出切片 (symbol + prob/pred 列)."""
    return pd.DataFrame(
        {
            "symbol": ["000001", "000002", "000003", "000004", "000005"],
            "prob_up_3d": list(probs),
            "pred_ret_3d": [0.02, 0.03, 0.04, 0.01, -0.01],
            "composite_score": [0.016, 0.024, 0.031, 0.013, 0.009],
        }
    )


class TestRerank:
    def test_merge_no_suffix_collision(self):
        """merge 不得产生 _x/_y 后缀列 (回归: 2026-08-05 NaN 占位与 legacy 列冲突)."""
        res = ov.rerank(_shortlist(), _legacy())
        assert "prob_up_3d" in res.columns
        assert not any("_x" in c or "_y" in c for c in res.columns)

    def test_legacy_prob_attached(self):
        res = ov.rerank(_shortlist(), _legacy())
        got = res.set_index("symbol").loc["000003", "prob_up_3d"]
        assert got == pytest.approx(0.9)
        assert res.set_index("symbol").loc["000004", "composite_score"] == pytest.approx(0.013)

    def test_co_occur_first_then_final_score_desc(self):
        res = ov.rerank(_shortlist(), _legacy(), w_pool=0.5, w_prob=0.5)
        # 期望 final_score = 0.5*score + 0.5*prob_up_3d
        exp = _shortlist().merge(_legacy()[["symbol", "prob_up_3d"]], on="symbol")
        exp["final_score"] = 0.5 * exp["score"] + 0.5 * exp["prob_up_3d"]
        exp = exp.sort_values(["co_occur", "final_score"], ascending=[False, False])
        assert list(res["symbol"]) == list(exp["symbol"])
        # 共现两组在前
        assert list(res["symbol"][:2]) == ["000001", "000002"]
        # 组内 final_score 降序
        for grp, sub in res.groupby("co_occur"):
            assert list(sub["final_score"]) == sorted(sub["final_score"], reverse=True)

    def test_rk_final_continuous_and_rk_pool_kept(self):
        res = ov.rerank(_shortlist(), _legacy())
        assert list(res["rk_final"]) == list(range(1, len(res) + 1))
        # rk_pool 保持纯池分次序不变
        assert res["rk_pool"].tolist() == _shortlist()["rk"].tolist()

    def test_final_score_formula(self):
        res = ov.rerank(_shortlist(), _legacy(), w_pool=0.7, w_prob=0.3)
        # 000004: score 0.85, prob 0.3
        idx = res.set_index("symbol").loc["000004"]
        assert idx["final_score"] == pytest.approx(0.7 * 0.85 + 0.3 * 0.3)

    def test_prob_col_fallback_priority(self):
        """prob_up_3d 缺失 → 回退 prob_up_5d → 2d → prob_up."""
        # 手动构造: 只含 prob_up_5d
        leg5 = pd.DataFrame(
            {"symbol": ["000001", "000002", "000003", "000004", "000005"],
             "prob_up_5d": [0.6, 0.5, 0.9, 0.3, 0.4]}
        )
        res = ov.rerank(_shortlist(), leg5)
        assert res.attrs.get("prob_col") == "prob_up_5d"
        assert res["prob_up_5d"].notna().all()
        # 只含 prob_up (基础 1d 概率)
        leg1 = pd.DataFrame({"symbol": ["000001", "000002", "000003", "000004", "000005"],
                             "prob_up": [0.6, 0.5, 0.9, 0.3, 0.4]})
        res1 = ov.rerank(_shortlist(), leg1)
        assert res1.attrs.get("prob_col") == "prob_up"

    def test_empty_legacy_ranks_by_pool_only(self):
        """无 legacy 输出 → prob 列 NaN, 顺序退化为纯池分 (不崩)."""
        res = ov.rerank(_shortlist(), pd.DataFrame())
        assert res.attrs.get("prob_col") == ""
        assert res["prob_up_3d"].isna().all()
        # 纯池分: 共现优先, 组内 score 降序
        exp = _shortlist().sort_values(["co_occur", "score"], ascending=[False, False])
        assert list(res["symbol"]) == list(exp["symbol"])
        # final_score = w_pool*score (prob 填充 0); 默认 0.5/0.5 → 0.5*score, 排序单调不变
        assert res["final_score"].equals(0.5 * res["score"])

    def test_w_prob_zero_prob_ignored(self):
        """w_prob=0 → prob 不影响排序, final_score 纯池分."""
        res = ov.rerank(_shortlist(), _legacy(), w_pool=1.0, w_prob=0.0)
        assert res["final_score"].equals(res["score"].astype(float))


class TestOverlayWeights:
    def test_per_board_defaults(self):
        """main 偏 prob (0.2/0.8), dual 对半 (0.5/0.5) — 2026-08-05 正交性实证."""
        assert ov.overlay_weights("main") == (0.2, 0.8)
        assert ov.overlay_weights("dual") == (0.5, 0.5)

    def test_unknown_board_falls_back_balanced(self):
        assert ov.overlay_weights("nonexistent") == (0.5, 0.5)

    def test_main_defaults_prob_dominant(self):
        """main 权重下 prob 主导: 000003 (prob 0.9) 排到非共现组最前, 虽 score 非最高."""
        res = ov.rerank(_shortlist(), _legacy(), *ov.overlay_weights("main"))
        nonco = res[~res["co_occur"]]
        assert nonco.iloc[0]["symbol"] == "000003"
        # 000005 (score 0.70, prob 0.4) 胜过 000004 (score 0.85, prob 0.3) → 纯 prob 信号翻盘
        assert list(nonco["symbol"]) == ["000003", "000005", "000004"]

    def test_dual_defaults_balanced(self):
        """dual 权重下保持对半: 与显式 0.5/0.5 结果一致."""
        a = ov.rerank(_shortlist(), _legacy(), *ov.overlay_weights("dual"))
        b = ov.rerank(_shortlist(), _legacy(), w_pool=0.5, w_prob=0.5)
        assert a["final_score"].equals(b["final_score"])


class TestOverlaySnapshot:
    """overlay_snapshot_frame: 每交付日把再排结果截成前向跟踪快照 (稳定列集).

    目的: 用户 2026-08-05 "改动正不正两周后给实盘答案" — 快照记录当日交付名单的
    prob/score/final_score 组成与实际叠加权重, 供日后 join 已实现 MFE 验证.
    """

    def _snap(self, board="main", w_pool=0.2, w_prob=0.8, legacy=None,
              date="2026-08-04"):
        res = ov.rerank(_shortlist(), _legacy() if legacy is None else legacy)
        return ov.overlay_snapshot_frame(res, board, date, w_pool, w_prob)

    def test_canonical_columns_and_order(self):
        snap = self._snap()
        assert list(snap.columns) == list(ov.SNAPSHOT_COLS)

    def test_prob_values_normalized_to_prob_up(self):
        snap = self._snap()
        assert list(snap["prob_col"]) == ["prob_up_3d"] * len(snap)
        got = snap.set_index("symbol")["prob_up"]
        assert got.loc["000003"] == pytest.approx(0.9)
        assert got.loc["000004"] == pytest.approx(0.3)
        assert got.notna().all()

    def test_weights_and_ids_stamped(self):
        snap = self._snap()
        assert (snap["board"] == "main").all()
        assert (snap["date"] == "2026-08-04").all()
        assert (snap["w_pool"] == 0.2).all()
        assert (snap["w_prob"] == 0.8).all()

    def test_final_score_and_rank_kept(self):
        snap = self._snap()
        assert "final_score" in snap.columns
        assert "rk_final" in snap.columns
        assert "rk_pool" in snap.columns
        # rk_final 连续 (与 rerank 一致)
        assert list(snap["rk_final"]) == list(range(1, len(snap) + 1))

    def test_no_prob_degrade(self):
        """无 legacy → prob_col='', prob_up 全 NaN, final_score 纯池分, 列集仍规范."""
        res = ov.rerank(_shortlist(), pd.DataFrame())
        snap = ov.overlay_snapshot_frame(res, "dual", "2026-08-04", 0.5, 0.5)
        assert list(snap.columns) == list(ov.SNAPSHOT_COLS)
        assert list(snap["prob_col"]) == [""] * len(snap)
        assert snap["prob_up"].isna().all()
        assert snap["final_score"].equals(0.5 * snap["score"])

    def test_prob_fallback_col_normalized(self):
        """legacy 只含 prob_up_5d → 快照 prob_up 取该列, prob_col 标注来源."""
        leg5 = pd.DataFrame({"symbol": ["000001", "000002", "000003", "000004", "000005"],
                             "prob_up_5d": [0.6, 0.5, 0.9, 0.3, 0.4]})
        res = ov.rerank(_shortlist(), leg5)
        assert res.attrs.get("prob_col") == "prob_up_5d"
        snap = ov.overlay_snapshot_frame(res, "main", "2026-08-04", 0.5, 0.5)
        assert list(snap["prob_col"]) == ["prob_up_5d"] * len(snap)
        assert snap.set_index("symbol").loc["000003", "prob_up"] == pytest.approx(0.9)


class TestWriteSnapshot:
    """_write_snapshot: WORM 落盘 (日期分文件, 同名跳过不覆盖)."""

    def _snap(self):
        return ov.overlay_snapshot_frame(
            ov.rerank(_shortlist(), _legacy()), "main", "2026-08-04", 0.2, 0.8)

    def test_writes_dated_file_with_module_suffix(self, tmp_path):
        ov._write_snapshot(self._snap(), tmp_path, module="20260805_q2345")
        f = tmp_path / "overlay_track_20260804__20260805_q2345.csv"
        assert f.exists()
        df = pd.read_csv(f, dtype={"symbol": str})
        assert list(df.columns) == list(ov.SNAPSHOT_COLS)
        assert len(df) == 5

    def test_no_module_suffix_when_na(self, tmp_path):
        ov._write_snapshot(self._snap(), tmp_path, module="na")
        assert (tmp_path / "overlay_track_20260804.csv").exists()

    def test_worm_skip_on_existing(self, tmp_path, capsys):
        """同名文件已存在 → 不覆盖不追加, 内容保持不变."""
        ov._write_snapshot(self._snap(), tmp_path, module="v1")
        before = (tmp_path / "overlay_track_20260804__v1.csv").read_text(encoding="utf-8")
        ov._write_snapshot(self._snap(), tmp_path, module="v1")
        after = (tmp_path / "overlay_track_20260804__v1.csv").read_text(encoding="utf-8")
        assert before == after
        assert len(list(tmp_path.glob("overlay_track_*.csv"))) == 1
        assert "跳过覆盖" in capsys.readouterr().out


class TestLegacyPredict:
    def test_empty_symbols_returns_empty(self):
        df = pd.DataFrame({"symbol": ["000001"], "date": [pd.Timestamp("2026-08-04")]})
        out = ov.legacy_predict(df, "main", predictor=None, symbols=set())
        assert out.empty

    def test_filters_to_symbols(self):
        """只对给定 symbols 推理 (predictor 用 fake 验证 isin 过滤)."""
        class Fake:
            def predict(self, df, board):
                return df[["symbol"]].copy()
        df = pd.DataFrame({"symbol": ["000001", "000002", "000003"],
                           "date": [pd.Timestamp("2026-08-04")] * 3})
        out = ov.legacy_predict(df, "main", Fake(), symbols={"000001", "000003"})
        assert set(out["symbol"]) == {"000001", "000003"}


class TestPickProbCol:
    def test_priority_with_data(self):
        """按优先级取有真实数据的列."""
        df = pd.DataFrame({"prob_up_3d": [0.5], "prob_up": [0.6]})
        assert ov._pick_prob_col(df) == "prob_up_3d"
        df2 = pd.DataFrame({"prob_up_3d": [np.nan], "prob_up": [0.6]})
        assert ov._pick_prob_col(df2) == "prob_up"

    def test_all_nan_placeholder_skipped(self):
        """占位全 NaN 列不算可用 (回归: 2026-08-05 空 legacy 误选占位列)."""
        df = pd.DataFrame({"prob_up_3d": [np.nan], "prob_up_5d": [np.nan]})
        assert ov._pick_prob_col(df) == ""
        assert ov._pick_prob_col(pd.DataFrame()) == ""


class TestCrossSection:
    def test_filters_to_requested_date(self, tmp_path):
        """临时 parquet: 只取指定日期全截面 (含 symbol/date/industry/池特征)."""
        df = pd.DataFrame(
            {
                "symbol": ["000001", "000002"] * 2,
                "date": ([pd.Timestamp("2026-08-01")] * 2
                         + [pd.Timestamp("2026-08-04")] * 2),
                "industry": ["A", "B"] * 2,
                "amihud_illiq": [0.1, 0.2, 0.3, 0.4],
                "small_mv_premium": [0.5, 0.6, 0.7, 0.8],
            }
        )
        p = tmp_path / "checkpoint.parquet"
        df.to_parquet(p)
        ov.CHECKPOINTS["main"] = str(p)
        got, day = ov.cross_section("main", date="2026-08-04")
        assert day == pd.Timestamp("2026-08-04")
        assert list(got["symbol"]) == ["000001", "000002"]
        assert got["date"].nunique() == 1

    def test_latest_date_default(self, tmp_path):
        df = pd.DataFrame(
            {
                "symbol": ["000001"] * 2,
                "date": [pd.Timestamp("2026-08-01"), pd.Timestamp("2026-08-04")],
                "industry": ["A", "A"],
                "amihud_illiq": [0.1, 0.2],
            }
        )
        p = tmp_path / "checkpoint.parquet"
        df.to_parquet(p)
        ov.CHECKPOINTS["main"] = str(p)
        got, day = ov.cross_section("main")
        assert day == pd.Timestamp("2026-08-04")
        assert list(got["symbol"]) == ["000001"]
