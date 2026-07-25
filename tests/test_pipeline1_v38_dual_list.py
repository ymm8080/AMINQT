"""PIPELINE1 V3.8 第四批: LambdaRank / V3.7排序分 / D.8双清单 / 失效#5."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.pipeline1.announcement import AnnouncementFactor
from app.pipeline1.dual_list_runner import DualListRunner
from app.pipeline1.list_generator import ListGenerator


# ============================================================
# 阶段四 LambdaRank (trainer 集成)
# ============================================================
class TestLambdaRank:
    def test_rank_model_trained_and_saved(self, tmp_path):
        import app.pipeline1.dual_track_trainer as dtt

        dtt.LGB_PARAMS_REG["n_estimators"] = 10
        dtt.LGB_PARAMS_CLS["n_estimators"] = 10
        dtt.ES_PATIENCE = 3
        rng = np.random.default_rng(4)
        # 750 日 × 8 股面板 (排序需要横截面)
        dates = pd.bdate_range("2023-01-02", periods=750)
        frames = []
        for s in range(8):
            f = rng.normal(size=750)
            frames.append(pd.DataFrame({
                "symbol": f"60000{s}", "date": dates, "f1": f,
                "label_1d": f * 0.01 + rng.normal(0, 0.01, 750),
                "label_3d": rng.normal(0, 0.02, 750),
                "label_5d": rng.normal(0, 0.03, 750),
            }))
        df = pd.concat(frames, ignore_index=True)
        df["label_cls"] = (df["label_1d"] > 0.005).astype(float)
        trainer = dtt.DualTrackTrainer(model_dir=str(tmp_path))
        trained = trainer.train_window(df, "main", ["f1"])
        assert "rank_model" in trained
        model, label = trained["rank_model"]
        assert label == "label_1d"
        path = trainer.save(trained, "t")
        bundle = dtt.DualTrackTrainer.load(path)
        assert "rank_model" in bundle
        preds = bundle["rank_model"][0].predict(df[["f1"]].tail(8))
        assert len(preds) == 8

    def test_predictor_outputs_rank_score(self, tmp_path):
        from app.pipeline1.predictor import V35Predictor

        import app.pipeline1.dual_track_trainer as dtt
        dtt.LGB_PARAMS_REG["n_estimators"] = 5
        dtt.LGB_PARAMS_CLS["n_estimators"] = 5
        dtt.ES_PATIENCE = 2
        rng = np.random.default_rng(8)
        dates = pd.bdate_range("2023-01-02", periods=750)
        frames = []
        for s in range(6):
            f = rng.normal(size=750)
            frames.append(pd.DataFrame({
                "symbol": f"60000{s}", "date": dates, "f1": f,
                "label_1d": f * 0.01 + rng.normal(0, 0.01, 750),
                "label_3d": rng.normal(0, 0.02, 750),
                "label_5d": rng.normal(0, 0.03, 750),
            }))
        df = pd.concat(frames, ignore_index=True)
        df["label_cls"] = (df["label_1d"] > 0.005).astype(float)
        trainer = dtt.DualTrackTrainer(model_dir=str(tmp_path))
        path = trainer.save(trainer.train_window(df, "main", ["f1"]), "t")
        feats = df.copy()
        feats["board"] = "main"
        feats["industry"] = "白酒"
        out = V35Predictor({"main": path}).predict(feats, "main")
        assert "rank_score" in out.columns
        assert out["rank_score"].notna().all()


# ============================================================
# V3.7 排序分公式 (rank_score 存在时)
# ============================================================
class TestV37ScoreFormula:
    def test_rank_score_formula(self):
        gen = ListGenerator()
        df = pd.DataFrame({
            "symbol": ["A", "B"],
            "pred_ret_1d": [0.02, 0.02],
            "pred_ret_3d": [0.02, 0.02],
            "pred_ret_5d": [0.02, 0.02],
            "prob_up": [0.6, 0.6],
            "rank_score": [2.0, 1.0],
        })
        out = gen.compute_scores(df).set_index("symbol")
        # score = rank × (1+0.3·tanh(compound×100)) × prob_adjust → A 恰为 B 的 2 倍
        assert out.loc["A", "score"] == pytest.approx(2 * out.loc["B", "score"])

    def test_fallback_without_rank_score(self):
        gen = ListGenerator()
        df = pd.DataFrame({
            "symbol": ["A"],
            "pred_ret_1d": [0.02], "pred_ret_3d": [0.02], "pred_ret_5d": [0.02],
            "prob_up": [0.6],
        })
        out = gen.compute_scores(df)
        compound = 0.02
        assert out["score"].iloc[0] == pytest.approx(compound * 1.0)  # prob/base=1


# ============================================================
# D.8 双清单编排
# ============================================================
def _dual_cands() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["600001", "600002", "600003", "300001"],
        "board": ["main", "main", "main", "GEM"],
        "industry": ["白酒", "电池", "保险", "医药"],
        "pred_ret_1d": [0.03, 0.025, 0.02, 0.04],
        "pred_ret_3d": [0.05, 0.04, 0.03, 0.06],
        "pred_ret_5d": [0.07, 0.06, 0.05, 0.08],
        "prob_up": [0.72, 0.69, 0.60, 0.75],  # 600003 prob<0.68 出局
        "pain_prob": [0.10, 0.20, 0.05, 0.05],  # 600002 pain≥0.15 出局
        "score": [0.05, 0.04, 0.03, 0.06],
        "uncertainty_width": [0.05, 0.05, 0.05, 0.05],
        "pred_q50": [0.03, 0.025, 0.02, 0.04],
        "ATR_pct": [0.02, 0.02, 0.02, 0.02],
    })


class TestDualListRunner:
    def test_aggressive_grade_A_chain(self):
        out = DualListRunner.aggressive_entry(_dual_cands())
        # 只剩 600001: 600002 pain 出局, 600003 prob 出局, 300001 非主板出局
        assert list(out["symbol"]) == ["600001"]

    def test_emit_dual_profiles_and_worm(self, tmp_path):
        runner = DualListRunner(
            str(tmp_path),
            stable_lister=ListGenerator(entry_prob=0.55, entry_ret_mult=0.0),
        )
        out = runner.emit(_dual_cands(), "2026-07-25")
        assert (out["execution"]["profile"] == "aggressive").all()
        assert (out["shadow"]["profile"] == "stable").all()
        assert len(out["shadow"]) > len(out["execution"])
        # WORM: 两份清单均落盘且含 profile
        import json
        import os
        files = os.listdir(tmp_path)
        assert files and files[0].startswith("dual_2026-07-25")
        with open(os.path.join(tmp_path, files[0]), encoding="utf-8") as fh:
            profiles = {json.loads(line)["profile"] for line in fh}
        assert profiles == {"aggressive", "stable"}

    def test_bear_takeover_blocks_aggressive(self):
        runner = DualListRunner(
            stable_lister=ListGenerator(entry_prob=0.55, entry_ret_mult=0.0))
        out = runner.emit(_dual_cands(), "2026-07-25", market_state="bear")
        assert out["bear_takeover"]
        assert len(out["execution"]) == 0  # D.7: DEFENSE 攻击档只卖不买

    def test_monthly_adjudication_streak(self):
        runner = DualListRunner()
        good, bad = [0.03] * 20, [-0.02] * 20
        for _ in range(2):
            r = runner.monthly_adjudication(bad, [0.3] * 20, good, [0.3] * 20)
            assert not r["force_switch_to_stable"]
        r = runner.monthly_adjudication(bad, [0.3] * 20, good, [0.3] * 20)
        assert r["force_switch_to_stable"]  # 连续 3 月落后 → 强制切回


# ============================================================
# 失效条件 #5 (公告剔除)
# ============================================================
class TestAnnouncementInvalidation:
    def test_event_window_and_negative(self):
        af = AnnouncementFactor()
        af.add_manual_entry("600519", "2026-07-25", "财报", score=0.5)  # 事件窗口
        af.add_manual_entry("600000", "2026-07-25", "风险提示", score=-0.8)  # 强利空
        af.add_manual_entry("600001", "2026-07-25", "其他", score=0.5)  # 不受影响
        inv = af.list_invalidation("2026-07-25")
        assert set(inv) == {"600519", "600000"}

    def test_apply_invalidation(self):
        cands = pd.DataFrame({"symbol": ["A", "B", "C"], "score": [1, 2, 3]})
        out = AnnouncementFactor.apply_invalidation(cands, {"B": "失效#5"})
        assert list(out["symbol"]) == ["A", "C"]
