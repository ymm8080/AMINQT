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
                        "label_10d": rng.normal(0, 0.04, 750),
                    }
                )
            )
        df = pd.concat(frames, ignore_index=True)
        df["label_cls"] = (df["label_1d"] > 0.005).astype(float)
        for k in (2, 3, 5, 10):
            df[f"label_{k}d_cls"] = (df[f"label_{k}d"] > 0.005).astype(float)
        trainer = dtt.DualTrackTrainer(model_dir=str(tmp_path))
        calls: list[str] = []
        real_train_ranker = dtt.DualTrackTrainer._train_ranker

        def counting_ranker(self, out, label):
            calls.append(label)
            return real_train_ranker(self, out, label)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(dtt.DualTrackTrainer, "_train_ranker", counting_ranker)
        try:
            trained = trainer.train_window(df, "main", ["f1"])
        finally:
            monkeypatch.undo()
        assert "rank_model" in trained
        model, label = trained["rank_model"]
        # [2026-08-20] ranker 只训一次 (3d 版曾白训并被 5d 版覆盖), 用 5d 标签
        assert calls == ["label_5d"]
        assert label == "label_5d"
        path = trainer.save(trained, "t")
        bundle = dtt.DualTrackTrainer.load(path)
        assert "rank_model" in bundle
        preds = bundle["rank_model"][0].predict(df[["f1"]].tail(8))
        assert len(preds) == 8

    def test_quantile_fit_uses_reduced_es_patience(self, tmp_path, monkeypatch):
        """[2026-08-20] 分位头早停 patience 减半 (100→50): LightGBM 返回 best_iteration
        树, patience 只控搜索停止点 — fit 必须收到 QUANTILE_ES_PATIENCE."""
        import app.pipeline1.dual_track_trainer as dtt
        from app.pipeline1 import quantile_models as qm

        dtt.LGB_PARAMS_REG["n_estimators"] = 10
        dtt.LGB_PARAMS_CLS["n_estimators"] = 10
        seen: list[int] = []

        class FakeQuantileSet:
            def __init__(self, *a, **k):
                pass

            def fit(self, *a, **k):
                seen.append(k["es_patience"])
                return self

        monkeypatch.setattr(qm, "QuantileModelSet", FakeQuantileSet)
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2023-01-02", periods=300)
        frames = []
        for s in range(6):
            f = rng.normal(size=300)
            frames.append(
                pd.DataFrame(
                    {
                        "symbol": f"60000{s}",
                        "date": dates,
                        "f1": f,
                        "label_3d": rng.normal(0, 0.02, 300),
                        "label_5d": rng.normal(0, 0.03, 300),
                    }
                )
            )
        df = pd.concat(frames, ignore_index=True)
        trainer = dtt.DualTrackTrainer(model_dir=str(tmp_path))
        trainer.train_window(df, "main", ["f1"])
        assert len(seen) == 2  # 3d + 5d 两视界
        assert all(v == dtt.QUANTILE_ES_PATIENCE for v in seen)
        # 生产 ES_PATIENCE=100 → 减半 (本文件他处测试会全局改 ES_PATIENCE, 用字面量断言)
        assert dtt.QUANTILE_ES_PATIENCE == 50

    def test_predictor_outputs_rank_score(self, tmp_path):
        import app.pipeline1.dual_track_trainer as dtt
        from app.pipeline1.predictor import V35Predictor

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
                        "label_10d": rng.normal(0, 0.04, 750),
                    }
                )
            )
        df = pd.concat(frames, ignore_index=True)
        df["label_cls"] = (df["label_1d"] > 0.005).astype(float)
        for k in (2, 3, 5, 10):
            df[f"label_{k}d_cls"] = (df[f"label_{k}d"] > 0.005).astype(float)
        trainer = dtt.DualTrackTrainer(model_dir=str(tmp_path))
        path = trainer.save(trainer.train_window(df, "main", ["f1"]), "t")
        feats = df.copy()
        feats["board"] = "main"
        feats["industry"] = "白酒"
        out = V35Predictor({"main": path}).predict(feats, "main")
        assert "rank_score" in out.columns
        assert out["rank_score"].notna().all()
        # [多视界] 每个视界概率列存在且落在 [0,1] (2d 2026-08-09 已删)
        for col in ("prob_up_3d", "prob_up_5d", "prob_up_10d"):
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


# ============================================================
# 超参按 (board, kind) 覆盖 (2026-08-08 复验定案, 架构=按视界分开配)
#   cls main 3d/5d/10d = 15 (1d/2d 已删) | cls dual 全默认 31
#   reg dual 3d/5d/10d = 15 | reg main 全默认 31
#   pain 两板 15 (与 cls 解耦: cls dual 仍 31)
# ============================================================
class TestModelParams:
    def test_cls_main_leaves_15_all_horizons(self):
        import app.pipeline1.dual_track_trainer as dtt

        for h in (3, 5, 10):
            assert dtt.model_params("main", f"{h}d_cls")["num_leaves"] == 15

    def test_cls_dual_default_31(self):
        import app.pipeline1.dual_track_trainer as dtt

        assert dtt.model_params("dual", "10d_cls").get("num_leaves") is None  # 默认 31

    def test_reg_main_default_31(self):
        import app.pipeline1.dual_track_trainer as dtt

        assert dtt.model_params("main", "10d_reg").get("num_leaves") is None  # 默认 31

    def test_reg_dual_scoped_to_scanned_horizons(self):
        import app.pipeline1.dual_track_trainer as dtt

        # 3d/5d/10d 有扫描证据 → 15
        assert dtt.model_params("dual", "3d_reg")["num_leaves"] == 15
        assert dtt.model_params("dual", "5d_reg")["num_leaves"] == 15
        assert dtt.model_params("dual", "10d_reg")["num_leaves"] == 15
        # 1d/2d 已删 (2026-08-09) → 不命中覆盖表 → 家族默认 31
        assert dtt.model_params("dual", "1d_reg").get("num_leaves") is None
        assert dtt.model_params("dual", "2d_reg").get("num_leaves") is None

    def test_dual_10d_reg_generalization_params_20260817(self):
        """[2026-08-17 补扫定案] dual 10d_reg = ms50_λ1 (top10 超额 +2.48→+3.41%,
        IC 0.029→0.035, 3/3 子窗, 扰动不翻); 只落这一个头, 勿扩散."""
        import app.pipeline1.dual_track_trainer as dtt

        p = dtt.model_params("dual", "10d_reg")
        assert p["min_child_samples"] == 50
        assert p["reg_lambda"] == 1.0
        # 其他头不受泛化覆盖表影响 (仍家族默认)
        assert dtt.model_params("dual", "3d_reg").get("min_child_samples") is None
        assert dtt.model_params("dual", "5d_reg").get("reg_lambda") is None
        assert dtt.model_params("main", "10d_reg").get("min_child_samples") is None
        assert dtt.model_params("dual", "10d_cls").get("reg_lambda") is None

    def test_pain_leaves_15_both_boards_decoupled_from_cls(self):
        import app.pipeline1.dual_track_trainer as dtt

        assert dtt.model_params("main", "pain")["num_leaves"] == 15
        assert dtt.model_params("dual", "pain")["num_leaves"] == 15
        # 解耦: dual 板 cls 仍默认 31, pain 独立 15
        assert dtt.model_params("dual", "10d_cls").get("num_leaves") is None

    def test_dual_cls_auc_early_stop_only(self):
        """[2026-08-11] 双创概率头: logloss 早停塌缩 1-2 树 → prob_up 全股常数.
        AUC 早停仅作用于 dual cls 三头; main cls / pain / reg 保持默认 logloss."""
        import app.pipeline1.dual_track_trainer as dtt

        for h in (3, 5, 10):
            assert dtt.model_params("dual", f"{h}d_cls").get("metric") == "auc"
            assert dtt.model_params("main", f"{h}d_cls").get("metric") is None
        assert dtt.model_params("dual", "pain").get("metric") is None
        assert dtt.model_params("dual", "3d_reg").get("metric") is None

    def test_returns_copy_not_module_const(self):
        import app.pipeline1.dual_track_trainer as dtt

        assert dtt.model_params("main", "10d_cls") is not dtt.LGB_PARAMS_CLS
        assert dtt.model_params("dual", "10d_reg") is not dtt.LGB_PARAMS_REG

    def test_kind_without_horizon_uses_family_default(self):
        """_acc_backtest 等调用方传 'cls'/'reg' 无视界 → 家族默认."""
        import app.pipeline1.dual_track_trainer as dtt

        assert dtt.model_params("main", "cls").get("num_leaves") is None
        assert dtt.model_params("main", "reg").get("num_leaves") is None

    def test_module_mutation_propagates(self):
        """调用时复制: 测试/调用方对 LGB_PARAMS_CLS 的修改必须传导到返回值."""
        import app.pipeline1.dual_track_trainer as dtt

        orig = dtt.LGB_PARAMS_CLS["n_estimators"]
        dtt.LGB_PARAMS_CLS["n_estimators"] = 7
        try:
            assert dtt.model_params("main", "10d_cls")["n_estimators"] == 7
            assert dtt.model_params("dual", "10d_cls")["n_estimators"] == 7
        finally:
            dtt.LGB_PARAMS_CLS["n_estimators"] = orig


# ============================================================
# REG_MIN_TREES 保底 (任务 #14, 2026-08-17)
#   dual 3d/5d_reg es 早停 2 树 = 常数列 (candidates 实测 std=0.0000),
#   main 3d/5d 也退化 (8/16 树) → 地板重训镜像 CLS_MIN_TREES=30
# ============================================================
class TestRegMinTrees:
    def test_reg_flat_es_refit_min_trees(self):
        """es 窗过平 → 3d_reg 早停塌缩; REG_MIN_TREES 地板必须兜底重训,
        pred_ret 有真实截面区分度 (镜像 cls/quantile 地板测试)."""
        import app.pipeline1.dual_track_trainer as dtt

        rng = np.random.default_rng(13)
        n, nf = 400, 3
        dates = pd.date_range("2025-01-01", periods=n)
        X = rng.normal(size=(n, nf))
        y = X[:, 0] * 0.02 + rng.normal(0, 0.03, n)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(nf)])
        df["date"] = dates
        # _resolve_label 链路: label_pm_3d → label_pm_3d_net (两级都在才解析到净标签)
        df["label_pm_3d"] = y
        df["label_pm_3d_net"] = y
        # es 段 y 全常数 → es 损失无下降 → 早停 < REG_MIN_TREES → 地板重训.
        # 先切分拿真实 es 日期 (split_window(400) → train=320/es=20/calib=20/test=60,
        # es = dates[320:340]), 再扁平化, 后重新切分 (split 只看 date, 标签不影响).
        segs = dtt.DualTrackTrainer.split_window(df, window_total=n)
        es_dates = segs["es"]["date"]
        df.loc[df["date"].isin(es_dates), ["label_pm_3d", "label_pm_3d_net"]] = 0.0

        segs = dtt.DualTrackTrainer.split_window(df, window_total=n)
        model, label = dtt.DualTrackTrainer()._train_one(
            "3d_reg", segs, [f"f{i}" for i in range(nf)], "dual"
        )
        assert label == "label_pm_3d_net"
        assert model.n_estimators == dtt.REG_MIN_TREES, (
            f"es 早停塌缩必须重训到地板 {dtt.REG_MIN_TREES} 棵树"
        )
        preds = model.predict(rng.normal(size=(50, nf)))
        assert preds.std() > 0, "地板重训后 pred_ret_3d 不应为常数"
