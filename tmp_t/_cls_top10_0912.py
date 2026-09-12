# -*- coding: utf-8 -*-
"""0912 逐特征 A/B (双头): 非新特征 base vs base+某个特征, 全局 IC + top10.

背景: NEW_7+DENSITY3 已由 force_include 两板注入 PIPELINE1 (d73e9692)。用户要
逐特征分开测作用 — 非新特征结果 vs 含某个特征的结果, 决定每个特征留/摘。
与尾部加权解耦 (LEGACY_TAIL_WEIGHT 已判 FAIL 关闭, reg 头干净口径)。

臂 = base (pin, 无新特征) / base+{f} 单特征
  main: pin 273 列已含出货密度 → 单特征臂只有 NEW_7 的 7 列
  dual: pin 无密度 → 单特征臂 = NEW_7 + DENSITY3 全 10 列
头 = 双头都训 (用户 0912: "幅度头也可以训"):
  10d_cls = prob_up_10d (prob10_pull 清单排名键)
  10d_reg = 幅度头 pred_10d (dual 入池 rank 键; 生产已关尾部加权)
指标 = 每头各出: 日截面全局 IC (mean/ICIR/胜率) + 该头排名 top10 实得/胜率
       + 赢家预测抬升
判词 = 某特征两板 Δtop10 均 ≥0 → 留 force_include; 任一板降 >1pp → 摘
前视警示: 获利盘斜率20 / 出货密度×3 (dual 臂) 继承 dim09 ChipDistribution 全帧网格
前视 (chip_distribution.py:52, 0912 夜合成验证历史行 max 漂移 58pp) — 这 4 列
正向结果视为可疑 (前视夸大), 负向结果加倍判死; 最终去留等网格修复后复测.

用法: python tmp_t/_cls_top10_0912.py  (串行两板, 自写日志 tmp_t/_cls_top10_0912.log)
重活: 哨兵 _cls_top10_0912.py 已入 scripts/_run_guard.HEAVY_SENTINELS.
"""

import gc
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from app.pipeline1.dual_track_trainer import DualTrackTrainer
from app.pipeline1.ram_guard import check_startup_gate, start_monitor
from config.settings import (
    PANEL_V3_PATH,
    RETRAIN_RAM_GUARD_MIN_FREE_GB,
    RETRAIN_RAM_GUARD_POLL_S,
    data_others_path,
)

TAG = "cls_top10_0912"
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cls_top10_0912.log")
REPORT_DIR = data_others_path("diag")

NEW_7 = [
    "SL差值", "SL标准化", "SL斜率20", "SL多头持续天数",
    "带20", "带20斜率20", "获利盘斜率20",
]
DENSITY3 = ["出货_density_5d", "出货_density_10d", "出货_density_20d"]
ALL10 = NEW_7 + DENSITY3

log = logging.getLogger(TAG)


def _setup_logging():
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)


def _worm(name: str, payload: dict) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    log.info("[WORM] %s", path)
    return str(path)


def _load_pin(board: str) -> set:
    registry_dir = str(data_others_path("data/factor_registry"))
    fn = "selected_main_pinned.json" if board == "main" else "selected_dual_pinned.json"
    with open(os.path.join(registry_dir, fn), encoding="utf-8") as fh:
        return set(json.load(fh).get("features", []))


def build_frame(board: str) -> pd.DataFrame:
    """与 night0912 编排同构: 3y 面板 → 清洗 → 全特征帧 (label_excess=False)."""
    from app.pipeline1.cleaning_pipeline import CleaningPipeline, load_panel_v3
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.train_runner import prepare_board_frame

    t0 = time.time()
    panel = load_panel_v3(path=PANEL_V3_PATH)
    cut = panel["date"].max() - pd.DateOffset(years=3)
    panel = panel[panel["date"] >= cut]
    cleaner = CleaningPipeline()
    main_df, dual_df = cleaner.run_train(panel, board=board)
    board_df = main_df if board == "main" else dual_df
    del main_df, dual_df, panel
    gc.collect()
    df = prepare_board_frame(
        board_df,
        FeatureEngineV35(),
        None,
        cross_sectional_rank=(board != "main"),
        registry=None,
        label_excess=False,
    )
    del board_df
    gc.collect()
    log.info("[feat] %s 帧构建完成 %d 行 %d 列 (%.0fs)", board, len(df), df.shape[1], time.time() - t0)
    return df


def _daily_spearman(pred: np.ndarray, y: np.ndarray, dates: np.ndarray) -> list[float]:
    from scipy.stats import spearmanr

    # 先剔非有限行: test 窗每日都有零星 NaN 标签, scipy 整日传染 (night0912 已踩)
    ok = np.isfinite(pred) & np.isfinite(y)
    pred, y, dates = pred[ok], y[ok], dates[ok]
    ics = []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < 5:
            continue
        r = spearmanr(pred[m], y[m])[0]
        if np.isfinite(r):
            ics.append(float(r))
    return ics


def pred_metrics(test: pd.DataFrame, pred: np.ndarray) -> dict:
    """单头口径: 全局日截面 IC + 该头排名 top10 / top10% 实得 + 赢家预测抬升."""
    y = test["label_pm_10d_net"].to_numpy(dtype=float)
    d = pd.DataFrame({"date": test["date"].to_numpy(), "y": y, "p": pred})
    d = d.dropna(subset=["y"])
    d["rk_p"] = d.groupby("date")["p"].rank(ascending=False, method="first")
    d["rk_y"] = d.groupby("date")["y"].rank(ascending=False, method="first")
    d["n"] = d.groupby("date")["y"].transform("size")
    top10 = d[d["rk_p"] <= 10]
    realized_hi = d[d["rk_y"] <= (d["n"] * 0.1).clip(lower=1)]
    # top10% 分位 (按本头预测排名取): 头部组合噪声的稳健中间层
    picked_hi = d[d["rk_p"] <= (d["n"] * 0.1).clip(lower=1)]
    ics = _daily_spearman(pred, y, test["date"].to_numpy())
    ics_arr = np.array(ics) if ics else np.array([np.nan])
    return {
        "ic_mean": float(np.nanmean(ics_arr)),
        "icir": float(np.nanmean(ics_arr) / np.nanstd(ics_arr)) if np.nanstd(ics_arr) > 0 else None,
        "ic_win": float(np.nanmean(ics_arr > 0)),
        "n_days": len(ics),
        "top10_real_net": float(top10["y"].mean()) if len(top10) else None,
        "top10_win": float((top10["y"] > 0).mean()) if len(top10) else None,
        "top10_n": int(len(top10)),
        "topdecile_real_net": float(picked_hi["y"].mean()) if len(picked_hi) else None,
        "winner_pred_lift": (
            float(realized_hi["p"].mean() / d["p"].mean()) if len(realized_hi) and d["p"].mean() else None
        ),
    }


def run_board(board: str) -> dict:
    df = build_frame(board)
    pins = _load_pin(board)
    t0 = time.time()
    base_cols = [f for f in sorted(pins) if f in df.columns and float(df[f].isna().mean()) < 0.95]
    avail = [f for f in ALL10 if f in df.columns and float(df[f].isna().mean()) < 0.95
             and f not in set(base_cols)]
    log.info("[ab:%s] base %d 列, 单特征臂 %d 列", board, len(base_cols), len(avail))

    keep = {"symbol", "date", "is_suspended", "close_hfq"}
    for c in ("label_10d", "label_pm_10d", "label_pm_10d_net", "label_10d_cls",
              "label_pm_10d_cls", "label_pm_10d_cls_net"):
        if c in df.columns:
            keep.add(c)
    keep |= set(base_cols) | set(avail)
    keep &= set(df.columns)
    df = df[list(keep)]  # 先瘦身再切分 (15.8GB 双驻留陷阱)
    gc.collect()
    segs = DualTrackTrainer.split_window(df, 770)
    del df
    gc.collect()
    test = segs["test"]
    log.info("[ab:%s] train=%d日 test=%d日", board,
             segs["train"]["date"].nunique(), test["date"].nunique())

    # 臂: base (无新特征) + 每个新特征单独加进 base
    arms = [("base", list(base_cols))] + [(f, base_cols + [f]) for f in avail]

    trainer = DualTrackTrainer()
    rep = {"board": board, "n_base": len(base_cols), "features": avail,
           "test_days": int(test["date"].nunique()),
           # 产物自带解释上下文: 4 列含前视, 正向结果不可直接采信
           "lookahead_caveat": "获利盘斜率20/出货_density_* 继承 dim09 ChipDistribution 全帧网格前视 "
                               "(chip_distribution.py:52, 0912 夜验证 58pp); 正向结果可疑, 网格修复后复测",
           "arms": {}}
    try:
        for arm, cols in arms:
            ta = time.time()
            m_reg, lab_reg = trainer._train_one("10d_reg", segs, cols, board)
            m_cls, lab_cls = trainer._train_one("10d_cls", segs, cols, board)
            X = np.nan_to_num(test[cols].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            pred_reg = m_reg.predict(X) if m_reg is not None else np.full(len(X), np.nan)
            proba = m_cls.predict_proba(X)[:, 1] if m_cls is not None else np.full(len(X), np.nan)
            rep["arms"][arm] = {
                "n_cols": len(cols), "labels": [lab_reg, lab_cls],
                "reg": pred_metrics(test, pred_reg),
                "cls": pred_metrics(test, proba),
            }
            m = rep["arms"][arm]
            log.info("[ab:%s] 臂 %-10s reg(IC=%.4f top10=%.4f D10%%=%.4f) cls(IC=%.4f top10=%.4f D10%%=%.4f) (%.0fs)",
                     board, arm,
                     m["reg"]["ic_mean"], m["reg"]["top10_real_net"] or np.nan,
                     m["reg"]["topdecile_real_net"] or np.nan,
                     m["cls"]["ic_mean"], m["cls"]["top10_real_net"] or np.nan,
                     m["cls"]["topdecile_real_net"] or np.nan, time.time() - ta)
            del m_reg, m_cls, X, pred_reg, proba
            gc.collect()
    finally:
        del segs
    gc.collect()
    _worm(f"{TAG}_{board}", rep)
    return rep


def main() -> int:
    _setup_logging()
    from scripts._run_guard import find_conflicts

    conflicts = find_conflicts()
    if conflicts:
        for c in conflicts:
            log.error("[guard] 冲突: %s (PID %s)", c["sentinel"], c["pid"])
        return 3
    check_startup_gate(RETRAIN_RAM_GUARD_MIN_FREE_GB * 1024**3)
    start_monitor(RETRAIN_RAM_GUARD_MIN_FREE_GB * 1024**3, RETRAIN_RAM_GUARD_POLL_S)

    for board in ("main", "dual"):
        run_board(board)
    log.info("[DONE] 逐特征 cls 头 A/B 两板完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
