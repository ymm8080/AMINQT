"""E7 闸3 改造: 补训 3d/5d 分位数五模型并合入现有 bundle.

背景: E7 闸3 原用 pred_q50 (1d 中位数), T+1 不可执行且全市场为负 → 0 只过闸.
用户定案 (2026-08-05 2d/3d/5d, 2026-08-09 去 2d): 闸3 用 3d/5d 中位数
(pred_q50_3d/5d, 均须为正). 旧 bundle 缺 3d/5d 分位模型 → 回退 1d pred_q50,
而 main 1d q50 退化 (3 树常数) → main 整块被误杀. 本脚本补训 3d/5d 分位模型
(QuantileModelSet, label_pm_{k}d_net) 到 main+dual, 保留现有 8 个回归/分类模型
+ 校准器 + pain/rank 不动, 低内存 (300 交易日切片, 已验证不 OOM).

注意: 300 日切片下 split_window 的 es 段为空 (train 吸收余量 460>300), 故本脚本
手动切分 train=前 280 日 / es=末 20 日 (镜像 SEG_MIN_DAYS["es"]=20 下限),
让早停 eval_set 非空, 语义与全量重训一致.

训练语义镜像 dual_track_trainer._train_extras:
  label 偏好 label_pm_{k}d_net → label_{k}d_net → label_{k}d
  risk_filter + float32 下转 + time_weights (B10 半衰期) + es 早停.

输出 (WORM):
  models/pipeline1/{board}_{tag}.pkl                    (新 bundle 快照)
  models/pipeline1/{board}_current.pkl                   (指针更新, 先备份)
  models/pipeline1/{board}_current_prequantile_backup.pkl (旧指针备份)

用法: python scripts/_retrain_quantile_extras.py [YYMMDDtag]
"""

import gc
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.dual_track_trainer import (
    ES_PATIENCE,
    LGB_PARAMS_REG,
    DualTrackTrainer,
    risk_filter,
)
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import LabelEngine
from app.pipeline1.quantile_models import QuantileModelSet
from config.settings import PANEL_V3_PATH

MODEL_DIR = "models/pipeline1"
SLICE_DAYS = 300  # 与 diag 同切片, 已验证内存安全
MASK_RECENT_DAYS = 6
HORIZONS = (3, 5)  # E7 闸3 中位数门视界 (2026-08-09 去 2d)
BOARDS = (
    "main",
    "dual",
)  # 两板块旧 bundle 均缺 3d/5d 分位模型 → 回退 1d pred_q50, main 被常数模型误杀
# 板级特征帧临时缓存: 特征构建 ~69min/板, 崩溃/会话中断后按 tag+板 复用, 避免整轮重跑.
# 属临时缓存 (非 WORM 输出), 可覆盖.
CHECKPOINT_DIR = "data/_splice_feats"


def _xy(seg: pd.DataFrame, cols: list[str], label: str):
    """镜像 _train_extras._xy: risk_filter + float32 + nan_to_num."""
    sub = risk_filter(seg.dropna(subset=[label]))
    cols_present = [c for c in cols if c in sub.columns]
    if cols_present:
        sub[cols_present] = sub[cols_present].astype("float32", copy=False)
    X = np.nan_to_num(sub[cols_present].values, nan=0.0)
    return X, sub[label].values


def train_quantile(trainer, segs, cols, horizon, board):
    q_label = next(
        (
            c
            for c in (
                f"label_pm_{horizon}d_net",
                f"label_{horizon}d_net",
                f"label_{horizon}d",
            )
            if c in segs["train"].columns
        ),
        None,
    )
    if q_label is None:
        print(f"[{board}] {horizon}d 无标签, 跳过", flush=True)
        return None
    X, y = _xy(segs["train"], cols, q_label)
    X_es, y_es = _xy(segs["es"], cols, q_label)
    params = {k: v for k, v in LGB_PARAMS_REG.items() if k != "objective"}
    qset = QuantileModelSet(params).fit(
        X,
        y,
        sample_weight=trainer.time_weights(
            segs["train"].dropna(subset=[q_label]).pipe(risk_filter)
        ),
        eval_set=(X_es, y_es) if len(y_es) else None,
        es_patience=ES_PATIENCE,
    )
    qset.label_ = q_label
    del X, y, X_es, y_es
    gc.collect()
    print(
        f"[{board}] {horizon}d 分位数五模型训练完成 (label={q_label}, 样本 {len(segs['train']):,})",
        flush=True,
    )
    return qset


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%Y%m%d")
    t0 = time.time()
    trainer = DualTrackTrainer(MODEL_DIR)
    cleaner = CleaningPipeline()
    features = FeatureEngineV35()

    def cp_path(board):
        return os.path.join(CHECKPOINT_DIR, f"{board}_{tag}.parquet")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    cached = {b: os.path.exists(cp_path(b)) for b in BOARDS}
    frames = {b: None for b in BOARDS}
    if not all(cached.values()):
        panel = pd.read_parquet(str(PANEL_V3_PATH))
        print(
            f"[panel] {len(panel):,}r max={panel['date'].max():%Y-%m-%d} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        dates = sorted(panel["date"].unique())
        panel = panel[panel["date"] >= dates[-SLICE_DAYS]]
        print(
            f"[slice] {SLICE_DAYS}d {panel['date'].min():%Y-%m-%d}.. {len(panel):,}r",
            flush=True,
        )
        for b, df in zip(("main", "dual"), cleaner.run_train(panel)):
            frames[b] = df
        del panel
        gc.collect()
        print(
            f"[clean] main={len(frames['main']):,} dual={len(frames['dual']):,} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    module_tag = f"{tag}_q35"
    modules = {}
    for board in BOARDS:
        cur = os.path.join(MODEL_DIR, f"{board}_current.pkl")
        with open(cur, "rb") as fh:
            bundle = pickle.load(fh)
        cols = bundle["feature_cols"]
        print(
            f"[{board}] bundle feature_cols={len(cols)} ({time.time() - t0:.0f}s)",
            flush=True,
        )

        if cached[board]:
            df = pd.read_parquet(cp_path(board))
            print(
                f"[{board}] resume checkpoint {cp_path(board)} -> {len(df):,}r "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
        else:
            df = frames[board]
            # 特征 → 路径标签 → 主标签 → 停牌/近端掩码 (与 prepare_board_frame 同序列)
            df = features.build(
                df, None, inference_cols=cols, cross_sectional_rank=(board == "dual")
            )
            print(f"[{board}] features build ({time.time() - t0:.0f}s)", flush=True)
            df = LabelEngine.build_path_labels(df)
            df = LabelEngine.build_labels(df)
            df = LabelEngine.mask_suspension(df)
            df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)
            df.to_parquet(cp_path(board), index=False)
            print(
                f"[{board}] checkpoint saved -> {cp_path(board)} ({time.time() - t0:.0f}s)",
                flush=True,
            )
        if len(df) == 0:
            print(f"[{board}] 空, 跳过", flush=True)
            continue
        # 手动 train/es 切分: 300 日切片下 split_window 的 es 段为空 (train 吸收余量
        # 460>300), 早停 eval_set 无法生效. 镜像 SEG_MIN_DAYS["es"]=20 下限:
        # train=前 280 日, es=末 20 日 (时序, 不重叠).
        _dates = sorted(df["date"].unique())
        _es_days = 20
        segs = {
            "train": df[df["date"] <= _dates[-_es_days - 1]],
            "es": df[df["date"] >= _dates[-_es_days]],
        }
        del df, _dates, _es_days
        gc.collect()

        for h in HORIZONS:
            qset = train_quantile(trainer, segs, cols, h, board)
            if qset is not None:
                bundle[f"quantile_models_{h}d"] = qset
            del qset
            gc.collect()
        del segs
        gc.collect()

        # WORM: 快照 + 更新指针 (先备份旧指针)
        snap = os.path.join(MODEL_DIR, f"{board}_{module_tag}.pkl")
        with open(snap, "wb") as fh:
            pickle.dump(bundle, fh)
        backup = os.path.join(MODEL_DIR, f"{board}_current_prequantile_backup.pkl")
        if not os.path.exists(backup):
            import shutil

            shutil.copy(cur, backup)
        with open(cur, "wb") as fh:
            pickle.dump(bundle, fh)
        modules[board] = {
            "tag": module_tag,
            "file": os.path.basename(snap),
            "updated": time.strftime("%Y-%m-%d %H:%M"),
        }
        del bundle
        gc.collect()
        print(
            f"[{board}] spliced → {snap} + {cur} ({time.time() - t0:.0f}s)", flush=True
        )

    from app.pipeline1.model_meta import save_modules

    save_modules(modules)
    print(f"[meta] models/pipeline1/current_meta.json = {modules}", flush=True)
    print(f"[done] 全部完成 ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
