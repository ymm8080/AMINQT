"""legacy 双板 cls 校准器重校 (Platt) — 概率坍缩修复 (2026-08-06)
================================================================
只重拟合 1/2/3/5d_cls 的 ProbCalibrator: isotonic 阶跃把生产原始概率带
(如 3d_cls raw∈[0.57,0.68]) 压成单平台 → prob_up_3d 全 0.642, 概率列失去
区分度. 改为平滑单调单射的 Platt (保留 raw 排序). 模型/特征/分位模型/痛苦
模型全部保留当前 current bundle 原样 (20260805r OOS 刚过, 无需重训).

预处理镜像 run_training 同路径: CleaningPipeline.run_train → prepare_board_frame
(特征/标签/掩码) → split_window (770d 隔离段). OOS 加权 IC 过闸才发布新
current + 更新 current_meta.json.

用法: python scripts/_recalibrate_legacy_cls.py [tag] [--force]
  --force: OOS 模型信号未过闸也发布 (校准-only 覆盖 — 仅换校准器不碰模型;
           模型信号弱由制度门/下次全量重训处理, 见 2026-08-06 main 案例)
输出: {board}_{tag}.pkl (新 bundle, 校准器为 Platt)
"""

from __future__ import annotations

import gc
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.dual_track_trainer import DualTrackTrainer
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.feature_selector import BRUTE_FAMILIES, BruteForceGenerator
from app.pipeline1.train_runner import prepare_board_frame
from config.settings import PANEL_V3_PATH, data_others_path

MODEL_DIR = "models/pipeline1"
BOARDS = ("main", "dual")
REGISTRY_PATH = str(data_others_path("data/factor_registry"))


def report_calib_spread(board: str, trained: dict) -> None:
    """test 段上 3d_cls 校准后概率分布 (验证坍缩已修复: 不应再是单平台)."""
    cal = trained["calibrators"].get(3)
    if cal is None or "3d_cls" not in trained["models"]:
        print(f"[{board}] 无 3d_cls 校准器, 跳过坍缩检查", flush=True)
        return
    model, label = trained["models"]["3d_cls"]
    test = trained["segs"]["test"].dropna(subset=[label])
    cols = trained["feature_cols"]
    raw = model.predict_proba(np.nan_to_num(test[cols].values, nan=0.0))[:, 1]
    cal_p = cal.predict_proba(raw)
    print(
        f"[{board}] test段 3d_cls: method={cal.method} "
        f"raw nunique={np.unique(raw).size} "
        f"cal nunique={np.unique(cal_p).size} "
        f"cal range=[{cal_p.min():.4f},{cal_p.max():.4f}] std={cal_p.std():.4f}",
        flush=True,
    )


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y%m%d")
    force = "--force" in sys.argv
    t0 = time.time()
    panel = pd.read_parquet(str(PANEL_V3_PATH))
    cut = panel["date"].max() - pd.DateOffset(years=3)
    panel = panel[panel["date"] >= cut]
    print(
        f"[panel] {len(panel):,}r max={panel['date'].max():%Y-%m-%d} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    cleaner = CleaningPipeline()
    features = FeatureEngineV35()
    registry = FeatureRegistry(
        path=os.path.join(REGISTRY_PATH, "feature_registry.json")
    )
    main_df, dual_df = cleaner.run_train(panel)
    del panel
    gc.collect()
    print(
        f"[clean] main={len(main_df):,} dual={len(dual_df):,} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    trainer = DualTrackTrainer(model_dir=MODEL_DIR)
    results = {}
    for board, board_df in (("main", main_df), ("dual", dual_df)):
        if len(board_df) == 0:
            print(f"[{board}] 空, 跳过", flush=True)
            continue
        cur = os.path.join(MODEL_DIR, f"{board}_current.pkl")
        bundle = DualTrackTrainer.load(cur)
        df = prepare_board_frame(
            board_df,
            features,
            cross_sectional_rank=(board != "main"),
            registry=registry,
        )
        del board_df
        gc.collect()
        # 训练时 FeatureSelector 精选后 train_runner 会注入 BruteForce 特征
        # (_brute_* 列, 见 train_runner.select_features). 重校脚本不重跑精选,
        # 需按 bundle feature_cols 复现这些列, 否则 schema 校验报缺失.
        missing = [c for c in bundle["feature_cols"] if c not in df.columns]
        brute_missing = [c for c in missing if "_brute_" in c]
        if brute_missing:
            gen = BruteForceGenerator()
            raw_cols = gen._eligible(df)
            need = set(brute_missing)
            picks = []
            for fam in BRUTE_FAMILIES:
                new = gen.generate_columns(
                    df, fam, need, raw_cols=raw_cols, dtype="float32"
                )
                if new is None or not len(new.columns):
                    continue
                picks.append(new)
            if picks:
                _brute = pd.concat(picks, axis=1)
                for _c in _brute.columns:
                    df[_c] = _brute[_c].to_numpy()
                print(
                    f"[{board}] BruteForce 复现注入 {len(_brute.columns)} 列 "
                    f"(缺 {len(brute_missing)})",
                    flush=True,
                )
        del missing, brute_missing
        gc.collect()
        segs = trainer.split_window(df)
        missing = [c for c in bundle["feature_cols"] if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"[{board}] 预处理缺 {len(missing)} 个训练特征列: {missing[:8]} "
                f"(特征引擎/注册表与训练时不一致, 中止重校)"
            )
        trained = {
            "board": board,
            "feature_cols": bundle["feature_cols"],
            "models": bundle["models"],
            "segs": segs,
        }
        for extra in (
            "quantile_models",
            "quantile_models_2d",
            "quantile_models_3d",
            "quantile_models_5d",
            "pain_model",
            "rank_model",
        ):
            if extra in bundle:
                trained[extra] = bundle[extra]
        del df
        gc.collect()
        trainer.fit_calibrator(trained)
        report_calib_spread(board, trained)
        oos = trainer.validate_oos(trained)
        print(
            f"[{board}] OOS weighted_IC={oos.get('weighted_ic'):.4f} "
            f"ics={ {k: round(v, 3) for k, v in oos.get('ics', {}).items()} }",
            flush=True,
        )
        if not oos["pass"]:
            if not force:
                print(
                    f"[{board}] OOS 未过闸, 保留旧模型 (不下发新校准器)",
                    flush=True,
                )
                results[board] = {"switched": False, "oos": oos}
                del segs, trained
                gc.collect()
                continue
            print(
                f"[{board}] OOS 未过闸但 --force: 校准-only 覆盖发布 "
                f"(只换校准器, 模型/特征原样; 模型信号由制度门/下次重训处理)",
                flush=True,
            )
        path = trainer.save(trained, tag)
        results[board] = {"path": path, "oos": oos, "switched": True}
        print(f"[{board}] 新校准器落盘 -> {path}", flush=True)
        del segs, trained
        gc.collect()

    del dual_df, main_df
    gc.collect()

    from app.pipeline1.model_meta import load_modules, save_modules

    mods = load_modules()
    changed = False
    for board, res in results.items():
        if not res.get("switched"):
            continue
        cur = os.path.join(MODEL_DIR, f"{board}_current.pkl")
        bak = os.path.join(MODEL_DIR, f"{board}_current_recalib_backup.pkl")
        if os.path.exists(cur) and not os.path.exists(bak):
            shutil.copy(cur, bak)
            print(f"[{board}] 旧 current 备份 -> {bak}", flush=True)
        shutil.copy(res["path"], cur)
        mods[board] = {
            "tag": tag,
            "file": os.path.basename(res["path"]),
            "updated": time.strftime("%Y-%m-%d %H:%M"),
        }
        changed = True
        print(f"[{board}] switched -> current = {res['path']}", flush=True)
    if changed:
        save_modules(mods)
        print(f"[meta] current_meta.json = {mods}", flush=True)
    print(f"[done] ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
