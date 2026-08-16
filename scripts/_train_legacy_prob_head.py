"""_train_legacy_prob_head.py — legacy 并行式概率头训练 (2026-08-15 代码先行, 勿跑).

用户指示: 面板扩建 (V3 宇宙扩建四脚本) 完成前不要训练. 待序列 ⑦ 全过切 config 后,
再在重训窗口内运行本脚本 (重训内存独占闸覆盖: 启动闸 + 运行期采样).

定案 (memory legacy-blend-rank-verdict → 用户定案): legacy cls 概率头太粗, 建并行式
全局 LGBM 概率头 (mfe_3d >= abs_target 二分类), WORM bundle 落盘
data/prob_head_legacy/. 与并行脚本 (_train_parallel_prob_head.py) 唯一差异 = 数据流:
legacy 无 stage 特征检查点, 特征现场构建 (CleaningPipeline.run_train 训练端清洗 +
FeatureEngineV35.build, 与 _diag_legacy_hitrate_topn 同构; main csr=False / dual
csr=True). 全史扩窗 = 面板全部行 (mfe_3d 尾段 NaN 行自动排除), 预测目标是次日及
以后的决策日 → 无前瞻.

用法: python scripts/_train_legacy_prob_head.py [--force]
自判断: 最新 bundle 距今 < refit_every_days 交易日 → skip (--force 强制重训).
WORM: <board>_prob_<ts>.joblib, 旧 bundle 不覆盖不删除.
"""

from __future__ import annotations

import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from app.pipeline1 import prob_head
from app.pipeline1.cleaning_pipeline import CleaningPipeline, load_panel_v3
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import LabelEngine, _ensure_sorted
from app.pipeline1.predictor import V35Predictor
from config.settings import LEGACY_PROB_GATE, PANEL_V3_PATH

BUNDLES = {
    "main": "models/pipeline1/main_current.pkl",
    "dual": "models/pipeline1/dual_current.pkl",
}
# mfe_3d (close/high/adv20) + label_pain (low) 所需 raw 列, 从清洗帧取 (生产口径)
NEED_RAW = ["symbol", "date", "close_hfq", "high_hfq", "low_hfq", "adv20"]


def _attach_labels(feat: pd.DataFrame, dfb: pd.DataFrame) -> pd.DataFrame:
    """特征帧 ← mfe_3d + label_pain (merge symbol×date, 不依赖行序).

    label_pain = LabelEngine.build_path_labels 生产路径 (3d 浮亏>5%), 停牌污染遮蔽
    镜像 label_engine.mask_suspension 的 mdd_3d 分支 ([T+1,T+5] 含停牌 → NaN).
    """
    missing = [c for c in NEED_RAW if c not in dfb.columns]
    if missing == ["adv20"] and "amount" in dfb.columns:
        # adv20 是特征引擎内部中间量 (E5 滑点分层输入), 清洗帧无此列 —
        # 按 label_engine.add_net_labels 同口径从 amount 现算 (rolling 20, min_periods=20)
        dfb = _ensure_sorted(dfb)
        dfb["adv20"] = (
            dfb.groupby("symbol")["amount"]
            .rolling(20, min_periods=20)
            .mean()
            .reset_index(level=0, drop=True)
        )
    elif missing:
        raise ValueError(f"清洗帧缺 raw 列 (无法打 mfe/pain 标签): {missing}")
    raw = dfb[NEED_RAW].copy()
    raw["symbol"] = raw["symbol"].astype(str)
    raw = prob_head._add_mfe_3d(raw)
    pain = LabelEngine.build_path_labels(raw)["label_pain"]
    if "is_suspended" in dfb.columns:
        rs = (
            dfb.groupby("symbol")["is_suspended"]
            .rolling(5)
            .sum()
            .reset_index(level=0, drop=True)
        )
        vals = rs.values
        susp = np.zeros(len(vals), dtype=bool)
        if len(vals) > 4:
            susp[: len(vals) - 4] = vals[4:] > 0
        pain = pain.where(~pd.Series(susp, index=rs.index), np.nan)
    raw["label_pain"] = pain
    feat = feat.copy()
    feat["symbol"] = feat["symbol"].astype(str)
    feat["date"] = pd.to_datetime(feat["date"])
    feat = feat.merge(
        raw[["symbol", "date", "mfe_3d", "label_pain"]],
        on=["symbol", "date"],
        how="left",
    )
    return feat


def main() -> int:
    force = "--force" in sys.argv[1:]
    # 2026-08-16: 直读全量面板在重训窗口并发时 OOM (step2 sort 深拷贝 1.5GB 分配失败);
    # 统一走 load_panel_v3 预过滤口径 (amount>=min_amount + 非停牌, 与重训/parallel 检查点同标准)
    panel = load_panel_v3(path=PANEL_V3_PATH)
    print(f"[load] panel {len(panel):,}r max={panel['date'].max()}", flush=True)
    main_df, dual_df = CleaningPipeline().run_train(panel)
    del panel
    gc.collect()
    predictor = V35Predictor(BUNDLES)
    features = FeatureEngineV35()
    ok = True
    for board, dfb, csr in (("main", main_df, False), ("dual", dual_df, True)):
        if dfb.empty:
            print(f"[{board}] 清洗帧为空 -> skip", flush=True)
            ok = False
            continue
        cols = predictor.bundles[board]["feature_cols"]
        print(
            f"[{board}] 清洗 {len(dfb):,}r -> 构建特征 (inference_cols={len(cols)})",
            flush=True,
        )
        feat = features.build(dfb, None, inference_cols=cols, cross_sectional_rank=csr)
        t = _attach_labels(feat, dfb)
        del feat, dfb
        gc.collect()
        dates = np.unique(pd.to_datetime(t["date"]).values)
        latest = pd.Timestamp(dates[-1])
        b = prob_head.load_latest(board)
        age = (
            None
            if b is None
            else prob_head.bundle_age_trading_days(dates, str(b["trained_through"]))
        )
        if (
            not force
            and b is not None
            and age is not None
            and age < LEGACY_PROB_GATE["refit_every_days"]
        ):
            print(
                f"[{board}] skip: bundle 距今 {age} 交易日 < "
                f"{LEGACY_PROB_GATE['refit_every_days']} (面板最新 {latest:%Y-%m-%d})",
                flush=True,
            )
            del t
            gc.collect()
            continue
        path = prob_head.train_bundle(board, t, str(latest.date()))
        n_pos = int((t["mfe_3d"] >= LEGACY_PROB_GATE["abs_target"]).sum())
        print(
            f"[{board}] 训练 {len(t):,} 行 (正样本 {n_pos:,}, "
            f"特征 {len(prob_head.feature_cols(t.drop(columns=['mfe_3d'])))} 列) "
            f"-> {path.name}",
            flush=True,
        )
        del t
        gc.collect()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
