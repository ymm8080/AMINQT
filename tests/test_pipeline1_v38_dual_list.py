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
            frames.append(
                pd.DataFrame(
                    {
                        "symbol": f"60000{s}",
                        "date": dates,
                        "f1": f,
                        "label_1d": f * 0.01 + rng.normal(0, 0.01, 750),
                        "label_2d": rng.normal(0, 0.015, 750),
                        "label_3d": rng.normal(0, 0.02, 750),
                        "label_5d": rng.normal(0, 0.03, 750),
                    }
                )
            )
        df = pd.concat(frames, ignore_index=True)
        df["label_cls"] = (df["label_1d"] > 0.005).astype(float)
        for k in (2, 3, 5):
            df[f"label_{k}d_cls"] = (df[f"label_{k}d"] > 0.005).astype(float)
        trainer = dtt.DualTrackTrainer(model_dir=str(tmp_path))
        trained = trainer.train_window(df, "main", ["f1"])
        assert "rank_model" in trained
        model, label = trained["rank_model"]
        # ranker 使用循环最后视界的 label (5d), 非 1d
        assert label == "label_5d"
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
            frames.append(
                pd.DataFrame(
                    {
                        "symbol": f"60000{s}",
                        "date": dates,
                        "f1": f,
                        "label_1d": f * 0.01 + rng.normal(0, 0.01, 750),
                        "label_2d": rng.normal(0, 0.015, 750),
                        "label_3d": rng.normal(0, 0.02, 750),
                        "label_5d": rng.normal(0, 0.03, 750),
                    }
                )
            )
        df = pd.concat(frames, ignore_index=True)
        df["label_cls"] = (df["label_1d"] > 0.005).astype(float)
        for k in (2, 3, 5):
            df[f"label_{k}d_cls"] = (df[f"label_{k}d"] > 0.005).astype(float)
        trainer = dtt.DualTrackTrainer(model_dir=str(tmp_path))
        path = trainer.save(trainer.train_window(df, "main", ["f1"]), "t")
        feats = df.copy()
        feats["board"] = "main"
        feats["industry"] = "白酒"
        out = V35Predictor({"main": path}).predict(feats, "main")
        assert "rank_score" in out.columns
        assert out["rank_score"].notna().all()
        # [多视界] 每个视界概率列存在且落在 [0,1]
        for col in ("prob_up_2d", "prob_up_3d", "prob_up_5d"):
            assert col in out.columns
            assert out[col].between(0, 1).all()


# ============================================================
# V3.7 排序分公式 (rank_score 存在时)
# ============================================================
class TestV37ScoreFormula:
    def test_rank_score_formula(self):
        gen = ListGenerator()
        df = pd.DataFrame(
            {
                "symbol": ["A", "B"],
                "pred_ret_1d": [0.02, 0.02],
                "pred_ret_2d": [0.02, 0.02],
                "pred_ret_3d": [0.02, 0.02],
                "pred_ret_5d": [0.02, 0.02],
                "prob_up": [0.6, 0.6],
                "rank_score": [2.0, 1.0],
            }
        )
        out = gen.compute_scores(df).set_index("symbol")
        # score = rank × (1+0.3·tanh(compound×100)) × prob_adjust → A 恰为 B 的 2 倍
        assert out.loc["A", "score"] == pytest.approx(2 * out.loc["B", "score"])

    def test_fallback_without_rank_score(self):
        gen = ListGenerator()
        df = pd.DataFrame(
            {
                "symbol": ["A"],
                "board": ["main"],
                "pred_ret_1d": [0.02],
                "pred_ret_2d": [0.02],
                "pred_ret_3d": [0.02],
                "pred_ret_5d": [0.02],
                "prob_up": [0.6],
            }
        )
        out = gen.compute_scores(df)
        # 无 rank_score → 回退 pred_ret_1d 横截面 rank(pct=True)=1.0, prob/base=1
        assert out["score"].iloc[0] == pytest.approx(1.0 * 1.0)


# ============================================================
# D.8 双清单编排
# ============================================================
def _dual_cands() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["600001", "600002", "600003", "300001"],
            "board": ["main", "main", "main", "GEM"],
            "industry": ["白酒", "电池", "保险", "医药"],
            "pred_ret_1d": [0.03, 0.025, 0.02, 0.04],
            "pred_ret_2d": [0.04, 0.03, 0.025, 0.05],
            "pred_ret_3d": [0.05, 0.04, 0.03, 0.06],
            "pred_ret_5d": [0.07, 0.06, 0.05, 0.08],
            "prob_up": [0.72, 0.69, 0.60, 0.75],  # 600003 prob<0.68 出局
            "pain_prob": [0.10, 0.20, 0.05, 0.05],  # 600002 pain≥0.15 出局
            "score": [0.05, 0.04, 0.03, 0.06],
            "uncertainty_width": [0.05, 0.05, 0.05, 0.05],
            "pred_q50": [0.03, 0.025, 0.02, 0.04],
            "ATR_pct": [0.02, 0.02, 0.02, 0.02],
        }
    )


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
            stable_lister=ListGenerator(entry_prob=0.55, entry_ret_mult=0.0)
        )
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

    def test_monthly_adjudication_delegates_single_source(self):
        """裁决逻辑单点: runner 与 gt_score.dual_profile_verdict 结论一致."""
        from app.pipeline1.gt_score import dual_profile_verdict

        runner = DualListRunner()
        good, bad = [0.03] * 20, [-0.02] * 20
        for m in ("2026-05", "2026-06", "2026-07"):
            r = runner.monthly_adjudication(bad, [0.3] * 20, good, [0.3] * 20, month=m)
        v = dual_profile_verdict(
            runner._hist_shadow,
            runner._hist_exec,  # noqa: SLF001
        )
        assert r["force_switch_to_stable"] == v["force_switch_to_stable"]
        assert r["lose_streak"] == v["trailing_below"]


# ============================================================
# P21.4 资金不分仓校验 (D.8: 任何时刻只一份清单进入真实执行)
# ============================================================
class TestSingleLiveList:
    def test_normal_day_live_is_aggressive(self):
        from app.pipeline1.dual_list_runner import resolve_live_list

        runner = DualListRunner(
            stable_lister=ListGenerator(entry_prob=0.55, entry_ret_mult=0.0)
        )
        out = runner.emit(_dual_cands(), "2026-07-25")
        assert out["live_profile"] == "aggressive"
        live = resolve_live_list(out)
        assert (live["profile"] == "aggressive").all()
        assert list(live["symbol"]) == list(out["execution"]["symbol"])

    def test_bear_day_no_live_list(self):
        """熊市接管: live_profile=None → 无可执行清单 (只卖不买)."""
        from app.pipeline1.dual_list_runner import resolve_live_list

        runner = DualListRunner(
            stable_lister=ListGenerator(entry_prob=0.55, entry_ret_mult=0.0)
        )
        out = runner.emit(_dual_cands(), "2026-07-25", market_state="bear")
        assert out["live_profile"] is None
        assert len(resolve_live_list(out)) == 0

    def test_shadow_can_never_be_live(self):
        """shadow 与 live 通道不一致 → 断言拒绝 (严禁两份都买)."""
        from app.pipeline1.dual_list_runner import resolve_live_list

        runner = DualListRunner(
            stable_lister=ListGenerator(entry_prob=0.55, entry_ret_mult=0.0)
        )
        out = runner.emit(_dual_cands(), "2026-07-25")
        tampered = {**out, "shadow": out["shadow"].assign(profile="aggressive")}
        with pytest.raises(AssertionError, match="两份都买"):
            resolve_live_list(tampered)


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


# ============================================================
# D.9/E.2 执行清单动态输出 + 回测失效模拟
# ============================================================
class TestDynamicOutputsWiring:
    def test_execution_carries_stop_and_position(self):
        cands = _dual_cands()
        cands["close"] = [10.0, 10.0, 10.0, 10.0]
        cands.loc[0, "pred_q50"] = 0.08  # 让 E.2 RR 闸门通过 (RR=2.0)
        runner = DualListRunner(
            stable_lister=ListGenerator(entry_prob=0.55, entry_ret_mult=0.0)
        )
        out = runner.emit(cands, "2026-07-25")
        exe = out["execution"]
        assert len(exe) == 1
        # D.9/E.2: stop_price / position / rr 随清单输出 (影子留痕)
        assert {"dyn_stop_pct", "dyn_position", "dyn_rr", "stop_price"} <= set(
            exe.columns
        )
        row = exe.iloc[0]
        assert 0 < row["dyn_position"] <= 1.0
        assert row["stop_price"] == pytest.approx(
            10.0 * (1 - row["dyn_stop_pct"]), abs=0.01
        )

    def test_dynamic_gate_shadow_zero_position(self):
        """E.2 影子模式: A级入选但 RR 不达标 → position=0 (留痕不驱动, F.5)."""
        cands = _dual_cands()  # pred_q50=0.03 → RR=1.25 < 1.8
        runner = DualListRunner(
            stable_lister=ListGenerator(entry_prob=0.55, entry_ret_mult=0.0)
        )
        exe = runner.emit(cands, "2026-07-25")["execution"]
        assert len(exe) == 1
        assert exe.iloc[0]["dyn_position"] == 0.0

    def test_simulate_invalidations_for_backtest(self):
        from app.pipeline1.backtest_adjudication import simulate_invalidations

        d1, d2 = "2026-07-24", "2026-07-25"
        lists = {
            d1: pd.DataFrame({"symbol": ["A", "B"], "score": [1.0, 0.9]}),
            d2: pd.DataFrame({"symbol": ["A", "C"], "score": [1.0, 0.8]}),
        }
        out = simulate_invalidations(lists, {d1: {"B": "失效#5: 公告利空"}})
        assert list(out[d1]["symbol"]) == ["A"]
        assert list(out[d2]["symbol"]) == ["A", "C"]  # 无失效日不受影响
