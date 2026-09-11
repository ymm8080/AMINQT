# -*- coding: utf-8 -*-
"""影子重训: 10d 秩目标头 (LEGACY_10D_RANK_TARGET) OOS 达线验证 (688228 案).

2026-09-10. 用户指令修预测精度不加闸 → 10d_reg 改训 per-date 截面百分位 +
训练段桶中位映射回收益语义 (09378ee8, 生产 enable=False)。本脚本在进程内打开
开关 (不动 settings.py), 镜像 run_training 的清洗→特征→选择→训练全链路训练
影子包 (WORM tag=0910rankshadow, 绝不触 current), 在 OOS test 段 (末60交易日)
对拍生产 current 包:

  ① 校准缺口: promise 分桶 (>6% / 3-6% / 0-3%) 承诺 vs 实得 (gross label_10d,
     净承诺 vs 毛实得 ≈ 嵌 ~1.3pp 成本, 与 _pred10d_downtrend_audit_0910 同口径)
     达线 = pred10>6% 桶缺口 < 3pp (旧头 −9.7~−11.8pp)
  ② 排名头部质量: rank≤10 vs 11-30 实得差 (旧头头部≈0 输给 11-30 下跌股 +2.3pp)
  ③ RankIC 不降 (每桶 spearman 均值, 影子 vs 生产)
  ④ 下跌趋势切片 (r5<0, test 段内 pct_change(5)): 影子头部下跌股占比+实得
  ⑤ 688228 个案: 影子 vs 生产承诺水平 (末个 test 日)

用法: 夜链退出后 python tmp_t/_rankshadow_10d_0910.py [--board main|dual]
重活: 已入 HEAVY_SENTINELS (_run_guard), 启动即查冲突 + RAM 闸。
"""

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

# 进程内打开秩目标 (生产 settings.py 默认 False; 影子验证专用, 不落盘)
import config.settings as rank_cfg

rank_cfg.LEGACY_10D_RANK_TARGET["enable"] = True

from app.pipeline1.dual_track_trainer import (  # noqa: E402
    DualTrackTrainer,
    rank_map_apply,
)
from app.pipeline1.ram_guard import check_startup_gate, start_monitor  # noqa: E402
from app.pipeline1.train_runner import (  # noqa: E402
    prepare_board_frame,
    select_features,
)
from config.settings import (  # noqa: E402
    LEGACY_EXCESS_LABEL_BOARDS,
    PANEL_V3_PATH,
    RETRAIN_RAM_GUARD_MIN_FREE_GB,
    RETRAIN_RAM_GUARD_POLL_S,
    data_others_path,
)

MODEL_DIR = "models/pipeline1"
TAG = "0910rankshadow"


def _rank_ic(pred: np.ndarray, real: np.ndarray, dates: np.ndarray) -> tuple[float, int]:
    """每日 spearman(pred, real) 均值 (日样本 <5 跳过)."""
    from scipy.stats import spearmanr

    ics = []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < 5:
            continue
        r = spearmanr(pred[m], real[m])[0]
        if np.isfinite(r):
            ics.append(r)
    return (float(np.mean(ics)) if ics else float("nan")), len(ics)


def evaluate(board: str, trained: dict, test: pd.DataFrame) -> dict:
    model, _ = trained["models"]["10d_reg"]
    cols = trained["feature_cols"]
    X = np.nan_to_num(test[cols].to_numpy(dtype=float), nan=0.0)
    shadow = rank_map_apply(trained.get("10d_rank_map"), model.predict(X))
    mkt = trained.get("mkt_expected_10d")
    if mkt is not None:
        shadow = shadow + float(mkt)
    prod = DualTrackTrainer.load(os.path.join(MODEL_DIR, f"{board}_current.pkl"))
    Xp = np.nan_to_num(
        test[prod["feature_cols"]].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    prod_ret = prod["models"]["10d_reg"][0].predict(Xp)
    if prod.get("label_excess") and prod.get("mkt_expected_10d") is not None:
        prod_ret = prod_ret + float(prod["mkt_expected_10d"])

    d = pd.DataFrame(
        {
            "date": test["date"].values,
            "symbol": test["symbol"].astype(str).values,
            "shadow": shadow,
            "prod": prod_ret,
            "real": test["label_10d"].values,  # 毛口径实得 (净承诺 vs 毛实得, 同审计口径)
        }
    )
    # 下跌趋势: close_hfq test 段内 pct_change(5) (≤当日可知)
    hfq = test[["symbol", "date", "close_hfq"]].copy()
    hfq["r5"] = (
        hfq.sort_values(["symbol", "date"]).groupby("symbol")["close_hfq"].pct_change(5)
    )
    d = d.merge(hfq[["symbol", "date", "r5"]], on=["symbol", "date"], how="left")

    rep: dict = {"board": board, "n": int(len(d))}
    # ① 校准缺口 (promise 分桶)
    for name, (lo, hi) in (
        ("gt6", (0.06, 10.0)),
        ("3_6", (0.03, 0.06)),
        ("0_3", (0.0, 0.03)),
    ):
        blk = {}
        for head in ("shadow", "prod"):
            g = d[(d[head] > lo) & (d[head] <= hi)].dropna(subset=["real"])
            blk[head] = (
                {
                    "n": int(len(g)),
                    "promise": float(g[head].mean()) if len(g) else None,
                    "real": float(g["real"].mean()) if len(g) else None,
                    "gap_pp": float((g["real"] - g[head]).mean() * 100) if len(g) else None,
                }
            )
        rep[f"calib_{name}"] = blk
    # ② 排名头部质量 (各自 rank key)
    for head in ("shadow", "prod"):
        d[f"rk_{head}"] = d.groupby("date")[head].rank(ascending=False, method="first")
        top = d[d[f"rk_{head}"] <= 10].dropna(subset=["real"])
        mid = d[(d[f"rk_{head}"] > 10) & (d[f"rk_{head}"] <= 30)].dropna(subset=["real"])
        rep[f"rank_{head}"] = {
            "top10_real": float(top["real"].mean()),
            "r11_30_real": float(mid["real"].mean()),
            "spread_pp": float((top["real"].mean() - mid["real"].mean()) * 100),
            "n_top": int(len(top)),
        }
    # ③ RankIC
    real_v = d["real"].to_numpy(dtype=float)
    for head in ("shadow", "prod"):
        ic, n = _rank_ic(d[head].to_numpy(dtype=float), real_v, d["date"].to_numpy())
        rep[f"rankic_{head}"] = {"ic": ic, "n_days": n}
    # ④ 下跌趋势切片 (影子 rank≤10 中的下跌股)
    dn = d[d["r5"] < 0].dropna(subset=["real"])
    rep["down_slice"] = {
        "n_down": int(len(dn)),
        "shadow_top10_down": int(((dn["rk_shadow"] <= 10)).sum()),
        "prod_top10_down": int(((dn["rk_prod"] <= 10)).sum()),
        "shadow_top10_down_real": float(
            dn.loc[dn["rk_shadow"] <= 10, "real"].mean()
        ) if (dn["rk_shadow"] <= 10).any() else None,
        "prod_top10_down_real": float(
            dn.loc[dn["rk_prod"] <= 10, "real"].mean()
        ) if (dn["rk_prod"] <= 10).any() else None,
    }
    # ⑤ 688228 个案 (末个 test 日)
    sym = "688228"
    g = d[d["symbol"].str.startswith(sym)].sort_values("date")
    if len(g):
        last = g.iloc[-1]
        rep["case_688228"] = {
            "date": str(pd.Timestamp(last["date"]).date()),
            "shadow_promise": float(last["shadow"]),
            "prod_promise": float(last["prod"]),
            "real_gross_fwd10": None if pd.isna(last["real"]) else float(last["real"]),
            "r5_at_day": None if pd.isna(last["r5"]) else float(last["r5"]),
        }
    return rep


def shap_case(board: str, case_row: pd.DataFrame) -> dict:
    """688228 生产 10d 头 TreeSHAP 归因 (为何下跌股被预测 +5%+ 入 TOP10).

    用生产 current 包自己的 feature_cols (全特征帧里都有) + LightGBM 原生
    pred_contrib (无需 shap 包)。dual 超额口径先加回 mkt_expected。
    """
    prod = DualTrackTrainer.load(os.path.join(MODEL_DIR, f"{board}_current.pkl"))
    cols = prod["feature_cols"]
    X = np.nan_to_num(
        case_row[cols].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    m = prod["models"]["10d_reg"][0]
    raw = float(m.predict(X)[0])
    pred = raw + float(prod["mkt_expected_10d"]) if prod.get("label_excess") and prod.get("mkt_expected_10d") is not None else raw
    contrib = m.predict(X, pred_contrib=True)[0]  # (n_cols+1), 末位=bias
    vals = contrib[:-1]
    order = np.argsort(-np.abs(vals))[:12]
    return {
        "date": str(pd.Timestamp(case_row["date"].iloc[0]).date()),
        "pred10_raw": raw,
        "pred10_abs": pred,
        "bias": float(contrib[-1]),
        "top_contrib": [
            {"feature": cols[i], "contrib": float(vals[i])} for i in order
        ],
    }


def train_board(board: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    from app.pipeline1.cleaning_pipeline import CleaningPipeline, load_panel_v3
    from app.pipeline1.feature_engine_v35 import FeatureEngineV35
    from app.pipeline1.feature_selector import FeatureSelector
    from config.settings import data_others_path as _dop

    t0 = time.time()
    panel = load_panel_v3(path=PANEL_V3_PATH)
    cut = panel["date"].max() - pd.DateOffset(years=3)
    panel = panel[panel["date"] >= cut]
    print(f"[panel] {len(panel)} rows 3y cut ({time.time() - t0:.0f}s)", flush=True)
    cleaner = CleaningPipeline()
    main_df, dual_df = cleaner.run_train(panel, board=board)
    board_df = main_df if board == "main" else dual_df
    del main_df, dual_df
    gc.collect()
    del panel
    gc.collect()
    use_xrank = board != "main"
    label_excess = board in LEGACY_EXCESS_LABEL_BOARDS
    features = FeatureEngineV35()
    df = prepare_board_frame(
        board_df,
        features,
        None,
        cross_sectional_rank=use_xrank,
        registry=None,
        label_excess=label_excess,
    )
    del board_df
    gc.collect()
    # 688228 个案行 (全特征帧, 最新日) — 供生产头 SHAP 归因 (del df 前取)
    case_row = (
        df[df["symbol"].astype(str).str.startswith("688228")]
        .sort_values("date")
        .tail(1)
    )
    # 与 run_training 同序: extras (label_excess + mkt_expected_*) 在 select_features
    # 释放 df 前从 attrs 取出 (select 后 attrs 帧已 del)
    extras = None
    if label_excess:
        extras = {"label_excess": True, **df.attrs.get("mkt_expected", {})}
    print(f"[feat] build done ({time.time() - t0:.0f}s)", flush=True)
    registry_path = str(_dop("data/factor_registry"))
    selector = FeatureSelector(config=None, registry_dir=registry_path)
    cols, augmented_df = select_features(df, board, TAG, selector, registry=None)
    del df
    gc.collect()
    print(f"[sel] {len(cols)} features ({time.time() - t0:.0f}s)", flush=True)

    trainer = DualTrackTrainer(model_dir=MODEL_DIR)
    from app.pipeline1.checkpoint import TrainingCheckpoint

    ck = TrainingCheckpoint(MODEL_DIR, board, TAG)
    trained = trainer.train_window(augmented_df, board, cols, checkpoint=ck)
    del augmented_df
    gc.collect()
    trained.update(extras or {})
    oos = trainer.validate_oos(trained)
    path = trainer.save(trained, TAG)
    if ck is not None:
        ck.clear()
    print(f"[done] {board} OOS weighted_IC={oos.get('weighted_ic'):.4f} "
          f"({time.time() - t0:.0f}s) → {path}", flush=True)
    return trained, trained["segs"]["test"], case_row


def main() -> int:
    from scripts._run_guard import find_conflicts

    others = find_conflicts()
    if others:
        for c in others:
            print(f"[guard] 冲突: {c['sentinel']} (PID {c['pid']})", flush=True)
        print("[guard] 有重训/预测进程在跑, 退出 (rc=3)", flush=True)
        return 3
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", choices=("main", "dual"), default=None)
    args = ap.parse_args()
    boards = (args.board,) if args.board else ("main", "dual")
    check_startup_gate(RETRAIN_RAM_GUARD_MIN_FREE_GB * 1024**3)
    start_monitor(RETRAIN_RAM_GUARD_MIN_FREE_GB * 1024**3, RETRAIN_RAM_GUARD_POLL_S)

    all_reps = {}
    for board in boards:
        trained, test, case_row = train_board(board)
        all_reps[board] = evaluate(board, trained, test)
        if len(case_row):
            all_reps[board]["shap_688228"] = shap_case(board, case_row)
        del trained, test
        gc.collect()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = data_others_path("diag") / f"rankshadow_10d_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(all_reps, fh, ensure_ascii=False, indent=2, default=str)
    for board, rep in all_reps.items():
        print(f"\n===== {board} =====", flush=True)
        for key, v in rep.items():
            print(f"{key}: {json.dumps(v, ensure_ascii=False)}", flush=True)
        c6 = rep.get("calib_gt6") or {}
        sh = (c6.get("shadow") or {}).get("gap_pp")
        ok = sh is not None and sh < 3.0
        print(f"[verdict] pred10>6% 校准缺口 影子 {sh}pp → 达线 {'PASS' if ok else 'FAIL'} "
              f"(阈值 <3pp; 生产对照 {(c6.get('prod') or {}).get('gap_pp')}pp)", flush=True)
    print(f"[worm] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
