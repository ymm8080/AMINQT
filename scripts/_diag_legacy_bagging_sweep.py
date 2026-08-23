"""_diag_legacy_bagging_sweep.py — legacy 主训练 bagging_fraction / bagging_freq 扫描 (2026-08-22).

背景: legacy 主训练 LGB_PARAMS_REG/CLS 无任何 bagging (单 seed=42, 无 subsample/
feature_fraction). 08-22 预测质量盘点发现短视界头退化 (3d/5d 塌 30 树地板,
prob_up 仅 3 唯一值), 结构缺口 = 主训练零方差削减 (而新概率闸 prob_head 反而有
subsample=0.8+colsample=0.8). 本脚本扫 bagging_fraction/freq, 目标是找到能
提升预测质量的取值 — 核心看 10d_reg (生产排名键), 辅助看 3d_cls (退化头是否
因 bagging 方差削减而让早停晚触发/概率更分散).

训练同构生产 (dual_track_trainer, 只训 main 两个头):
  - 数据: PANEL_V3_PATH → load_panel_v3 预过滤 → 3y 窗口 → CleaningPipeline.run_train
    (board='main') → prepare_board_frame (FeatureEngineV35 全量 build + LabelEngine 标签)
  - 特征列 = models/pipeline1/main_current.pkl 的 feature_cols (353)
  - split_window (WINDOW_TOTAL=770 反锚四段) + risk_filter + time_weights
    (HALF_LIFE=250) + huber/binary + n_estimators=1000 + lr=0.05 + early_stopping
    (ES_PATIENCE=100) + random_state=42
  - 标签 (生产 _resolve_label): 10d_reg → label_pm_10d_net; 3d_cls → label_pm_3d_cls_net

网格: bagging_fraction {0.6,0.7,0.8,0.9} × bagging_freq {1,2,5} 全交叉 (12 组合)
      + ref (无 bagging, 生产现状). 其余 = 生产参数 (num_leaves: 10d_reg 默认 31,
      3d_cls=15 按 NUM_LEAVES_OVERRIDE).

评估 (验收只看 OOS 铁律; --test-window 可切 60=生产 / 125=干净段确认, 仅本脚本):
  - 10d_reg: OOS test 段: (a) 逐日 Rank IC (pred vs label),
    (b) TOP-10 实得 (buy T+1 close, sell T+11 close, -0.002, 停牌 ffill),
    (c) 子窗稳定性 (3 等分), (d) seed=43 扰动
  - 3d_cls: test 段 AUC + prob 唯一值数 + 子窗 AUC (退化头是否因 bagging 复活)

判词 [2026-08-22 用户裁决: 判断规则主要看 TOP10]:
  10d_reg = TOP-10 实得超额为主 (rec.excess > ref) + 子窗 >=2/3 赢 + 扰动不翻转
            → 有档 (选稳定>最高); Rank IC 降为次键 (排序 tiebreak + 报告), 不做硬门.
  3d_cls 辅助: AUC 不降 + prob 唯一值不缩 (退化不更糟).

内存铁律: 每 fit 前 free RAM < 5GB → sleep 60s 重试; 串行; fit 完立即 del + gc.
WORM: DATA_OTHERS/diag/legacy_bagging_sweep_<ts>.json/.csv
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from app.pipeline1.cleaning_pipeline import CleaningPipeline, load_panel_v3
from app.pipeline1.drift_monitor import compute_realized
from app.pipeline1.dual_track_trainer import (
    ES_PATIENCE,
    HALF_LIFE,
    LGB_PARAMS_CLS,
    LGB_PARAMS_REG,
    MIN_ES_DATES,
    MIN_TRAIN_DAYS,
    NUM_LEAVES_OVERRIDE,
    WINDOW_TOTAL,
    DualTrackTrainer,
    risk_filter,
)
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.feature_selector import BRUTE_FAMILIES, BruteForceGenerator
from app.pipeline1.train_runner import prepare_board_frame
from config.settings import PANEL_V3_PATH, data_others_path

# ── 网格 / 协议 ─────────────────────────────────────────────
GRID_FRAC = (0.6, 0.7, 0.8, 0.9)
GRID_FREQ = (1, 2, 5)
RANDOM_STATE = 42  # 生产固定种子
PERTURB_SEED = 43  # 扰动种子
HEADS = ("10d_reg", "3d_cls")  # 10d_reg=排名键, 3d_cls=退化头
COST = 0.0020  # compute_realized 往返成本 (与 drift_monitor 一致)
BUY_LAG, SELL_LAG = 1, 11  # 10d 排名键实得口径
TOP_N = 10
RAM_MIN_FREE_GB = 5.0
RAM_MAX_WAIT_S = 1800
MAIN_BUNDLE = "models/pipeline1/main_current.pkl"
REGISTRY_PATH = str(data_others_path("data/factor_registry/feature_registry.json"))
# 特征构建检查点 (feather): 构建 ~55min, 崩溃/125d 确认重跑可 --use-built-frame 跳过.
# 生产 train_runner 在 select_features 里后注入 brute 族列, prepare_board_frame 不含 →
# 检查点必须存注入后的完整帧, 否则重跑仍缺列.
BUILT_FRAME_CP = data_others_path("diag/legacy_bagging_built_frame_main.feather")
CP_COLS = [
    "symbol",
    "date",
    "is_suspended",
    "close_hfq",
    "label_pm_10d_net",
    "label_pm_3d_cls_net",
]


def wait_for_ram(tag: str) -> None:
    """每 fit 前查 free RAM, < 5GB 则 sleep 60s 重试 (最多 30min)."""
    import psutil

    t0 = time.time()
    while True:
        free = psutil.virtual_memory().available / 1e9
        if free >= RAM_MIN_FREE_GB:
            print(f"[ram:{tag}] free={free:.1f}GB OK", flush=True)
            return
        waited = time.time() - t0
        if waited >= RAM_MAX_WAIT_S:
            print(
                f"[ram:{tag}] 等 {RAM_MAX_WAIT_S:.0f}s 后 free 仍 {free:.1f}GB"
                f" < {RAM_MIN_FREE_GB}GB, 继续 (不放弃)",
                flush=True,
            )
            return
        print(
            f"[ram:{tag}] free={free:.1f}GB < {RAM_MIN_FREE_GB}GB, "
            f"sleep 60s (已等 {waited:.0f}s)",
            flush=True,
        )
        time.sleep(60)


# ── 评估 (10d_reg: Rank IC + top10 实得, 同 _diag_legacy_dual_reg_sweep) ──
def per_day_rank_ic(df: pd.DataFrame, pred_col: str, label_col: str) -> dict:
    """逐日截面 Rank IC (Spearman) → {mean_ic, icir, pos_day_ratio, n_days}."""
    g = df.groupby("date")
    pred_r = g[pred_col].rank(pct=True)
    lab_r = g[label_col].rank(pct=True)
    tmp = pd.DataFrame(
        {"date": df["date"].values, "pred": pred_r.values, "lab": lab_r.values}
    )
    vals = []
    for _d, sub in tmp.groupby("date"):
        if len(sub) < 5:
            continue
        p = sub["pred"].values
        labels = sub["lab"].values
        m = np.isfinite(p) & np.isfinite(labels)
        if m.sum() < 5:
            continue
        cc = np.corrcoef(p[m], labels[m])[0, 1]
        if np.isfinite(cc):
            vals.append(cc)
    daily = np.asarray(vals, dtype=float)
    if len(daily) == 0:
        return {"mean_ic": np.nan, "icir": np.nan, "pos_day_ratio": np.nan, "n_days": 0}
    icir = float(daily.mean() / (daily.std(ddof=1) + 1e-12) * np.sqrt(len(daily)))
    return {
        "mean_ic": float(daily.mean()),
        "icir": float(icir),
        "pos_day_ratio": float((daily > 0).mean()),
        "n_days": int(len(daily)),
        "mean_abs": float(np.abs(daily).mean()),
    }


def per_day_topn_realized(
    df: pd.DataFrame, pred_col: str, real_col: str, n: int = TOP_N
) -> dict:
    """每日预测前 N 只实得均值 vs 当日全池实得均值 (超额)."""
    top_ret, pool_ret, excess, pos_day = [], [], [], []
    for _d, sub in df.groupby("date"):
        if len(sub) < 5:
            continue
        lab = sub[real_col].to_numpy(dtype=float)
        pool = float(np.nanmean(lab))
        order = np.argsort(sub[pred_col].to_numpy(dtype=float))[::-1]
        k = min(n, len(sub))
        top = float(np.nanmean(lab[order[:k]]))
        top_ret.append(top)
        pool_ret.append(pool)
        excess.append(top - pool)
        pos_day.append(1.0 if top > pool else 0.0)
    e = np.asarray(excess, dtype=float)
    if len(e) == 0:
        return {
            "excess": np.nan,
            "top_ret": np.nan,
            "pool_ret": np.nan,
            "pos_day": np.nan,
            "n_days": 0,
        }
    return {
        "excess": float(e.mean()),
        "top_ret": float(np.asarray(top_ret).mean()),
        "pool_ret": float(np.asarray(pool_ret).mean()),
        "pos_day": float((e > 0).mean()),
        "n_days": int(len(e)),
    }


def window_topn_excess(
    df: pd.DataFrame, pred_col: str, real_col: str, n: int = TOP_N, k: int = 3
) -> list[dict]:
    """OOS 按时间切 k 等长子窗, 每子窗 top-N 超额 — 参数稳定性检验."""
    uniq = sorted(df["date"].unique())
    if len(uniq) < k:
        return []
    total = len(uniq)
    date_to_win = {d: min(i * k // total, k - 1) for i, d in enumerate(uniq)}
    wk = df.copy()
    wk["_win"] = wk["date"].map(date_to_win)
    out: list[dict] = []
    for w in range(k):
        sub = wk[wk["_win"] == w]
        tn = per_day_topn_realized(sub, pred_col, real_col, n)
        out.append(
            {
                "win": f"{w + 1}/{k}",
                "n_days": tn.get("n_days", 0),
                "topn_excess": tn.get("excess", np.nan),
                "top_ret": tn.get("top_ret", np.nan),
                "pool_ret": tn.get("pool_ret", np.nan),
            }
        )
    return out


# ── 评估 (3d_cls: AUC + prob 唯一值 + 子窗 AUC) ──
def cls_eval(df: pd.DataFrame, prob_col: str, label_col: str, k: int = 3) -> dict:
    """OOS test 段: AUC + prob 唯一值数 + 子窗 AUC."""
    sub = df[df[label_col].notna()].copy()
    if len(sub) < 20 or sub[label_col].nunique() < 2:
        return {"auc": np.nan, "prob_nunique": 0, "subwindow_auc": [], "n": len(sub)}
    p = sub[prob_col].to_numpy(dtype=float)
    y = sub[label_col].to_numpy(dtype=float)
    auc = float(roc_auc_score(y, p))
    nuniq = int(pd.Series(p).nunique())
    # 子窗 AUC
    uniq = sorted(sub["date"].unique())
    total = len(uniq)
    date_to_win = {d: min(i * k // total, k - 1) for i, d in enumerate(uniq)}
    wk = sub.copy()
    wk["_win"] = wk["date"].map(date_to_win)
    subw = []
    for w in range(k):
        ws = wk[wk["_win"] == w]
        if len(ws) < 20 or ws[label_col].nunique() < 2:
            subw.append(np.nan)
        else:
            subw.append(
                float(roc_auc_score(ws[label_col].to_numpy(), ws[prob_col].to_numpy()))
            )
    return {"auc": auc, "prob_nunique": nuniq, "subwindow_auc": subw, "n": len(sub)}


def combo_key(frac: float | None, freq: int | None) -> str:
    if frac is None:
        return "ref"
    return f"f{frac:g}_q{freq}"


def is_ref(frac: float | None, freq: int | None) -> bool:
    return frac is None


# ── 单组合: 生产同构训练 + OOS 评估 ──────────────────────────
def fit_and_eval(
    head: str,
    train: pd.DataFrame,
    es: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    w: np.ndarray,
    realized: pd.DataFrame | None,
    label: str,
    frac: float | None,
    freq: int | None,
    seed: int,
) -> tuple[dict, int]:
    """生产 _train_one 同构训练 (huber/binary, bagging 注入), test 段评估."""
    wait_for_ram(combo_key(frac, freq))
    is_cls = head.endswith("cls")
    base = LGB_PARAMS_CLS if is_cls else LGB_PARAMS_REG
    params = dict(base)
    nl = NUM_LEAVES_OVERRIDE.get(("main", head))
    if nl is not None:
        params["num_leaves"] = nl
    if frac is not None:
        params["bagging_fraction"] = frac
        params["bagging_freq"] = freq
    params["random_state"] = seed

    X = np.nan_to_num(train[feature_cols].values, nan=0.0)
    y = train[label].values
    X_es = np.nan_to_num(es[feature_cols].values, nan=0.0)
    y_es = es[label].values
    model = lgb.LGBMClassifier(**params) if is_cls else lgb.LGBMRegressor(**params)
    use_es = es["date"].nunique() >= MIN_ES_DATES
    model.fit(
        X,
        y,
        sample_weight=w,
        eval_set=[(X_es, y_es)] if use_es else None,
        callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)] if use_es else None,
    )
    n_trees = int(model.best_iteration_ or params["n_estimators"])
    if n_trees <= 5:
        ev = getattr(model, "evals_result_", None)
        curve = ""
        if ev and "valid_0" in ev and ev["valid_0"]:
            mname = list(ev["valid_0"])[0]
            scores = ev["valid_0"][mname]
            curve = (
                "["
                + ",".join(f"{s:.4f}" for s in scores[:5])
                + (",..." if len(scores) > 5 else "")
                + "]"
            )
        print(
            f"    [!] {head}/{combo_key(frac, freq)} seed={seed} 早停仅 {n_trees} 树 "
            f"eval_curve{curve}",
            flush=True,
        )

    if is_cls:
        prob = model.predict_proba(np.nan_to_num(test[feature_cols].values, nan=0.0))[
            :, 1
        ]
    else:
        prob = model.predict(np.nan_to_num(test[feature_cols].values, nan=0.0))
    del model, X, X_es
    gc.collect()

    oos = test[["date", "symbol", label]].copy()
    oos["_pred"] = prob
    rec: dict = {
        "params": {
            "bagging_fraction": frac,
            "bagging_freq": freq,
        },
        "head": head,
        "seed": seed,
        "n_trees": n_trees,
        "n_train_rows": int(len(train)),
        "n_train_days": int(train["date"].nunique()),
        "n_es_days": int(es["date"].nunique()),
    }
    if is_cls:
        ce = cls_eval(oos, "_pred", label)
        rec["auc"] = ce["auc"]
        rec["prob_nunique"] = ce["prob_nunique"]
        rec["cls_n"] = ce["n"]
        rec["subwindow_auc"] = ce["subwindow_auc"]
    else:
        oos = oos.merge(realized, on=["date", "symbol"], how="inner")
        if oos.empty:
            raise RuntimeError(f"{head}/{combo_key(frac, freq)}: OOS 无 realized 行")
        ic = per_day_rank_ic(oos, "_pred", label)
        tn = per_day_topn_realized(oos, "_pred", "realized_net", TOP_N)
        wins = window_topn_excess(oos, "_pred", "realized_net", TOP_N, 3)
        rec["mean_ic"] = ic.get("mean_ic", np.nan)
        rec["icir"] = ic.get("icir", np.nan)
        rec["pos_day_ratio"] = ic.get("pos_day_ratio", np.nan)
        rec["n_ic_days"] = ic.get("n_days", 0)
        rec["top10"] = tn
        rec["subwindows"] = wins
    del oos
    gc.collect()
    return rec, n_trees


# ── 判词 ────────────────────────────────────────────────────
def _subwin_wins(rec: dict, ref: dict) -> int:
    return sum(
        1
        for i, s in enumerate(rec.get("subwindows", []))
        if i < len(ref.get("subwindows", []))
        and np.isfinite(s["topn_excess"])
        and np.isfinite(ref["subwindows"][i]["topn_excess"])
        and s["topn_excess"] > ref["subwindows"][i]["topn_excess"]
    )


def verdict_reg(
    results: dict[str, dict], perturbation: dict[str, dict], ref_key: str
) -> dict:
    """10d_reg: TOP-10 实得超额为主 + 子窗 >=2/3 + 扰动不翻转 → 有档.

    [2026-08-22 用户裁决] 判断规则主要看 TOP10: Rank IC 降为次键
    (排序 tiebreak + 报告口径), 不再作为硬门拦截.
    """
    ref = results[ref_key]
    ref_exc, ref_ic = ref["top10"]["excess"], ref["mean_ic"]

    cands = []
    for k, rec in results.items():
        if k == ref_key:
            continue
        w = _subwin_wins(rec, ref)
        if rec["top10"]["excess"] > ref_exc and w >= 2:
            cands.append((k, rec, w))
    cands.sort(key=lambda t: (-t[1]["top10"]["excess"], -t[1]["mean_ic"]))

    if not cands:
        return {
            "has_tier": False,
            "verdict": "无档赢, 保持生产默认 (无 bagging)",
            "text": (
                f"无组合比 ref (top10_exc={ref_exc:+.4f}, ic={ref_ic:.5f}) "
                f"在 top-10 实得超额同时赢且子窗多数赢 (TOP10 为主)"
            ),
            "ref": ref_key,
        }
    best_key, best, w = cands[0]
    detail = {
        "best": best_key,
        "best_top10_excess": best["top10"]["excess"],
        "best_mean_ic": best["mean_ic"],
        "best_subwin_wins": w,
        "ref_top10_excess": float(ref_exc),
        "ref_mean_ic": float(ref_ic),
    }
    if perturbation:
        b43 = perturbation.get(best_key)
        r43 = perturbation.get(ref_key)
        if b43 is not None and r43 is not None:
            flip = b43["top10"]["excess"] <= r43["top10"]["excess"]
            detail["perturb_flip"] = bool(flip)
            detail["best_seed43_top10_excess"] = b43["top10"]["excess"]
            detail["ref_seed43_top10_excess"] = r43["top10"]["excess"]
            if flip:
                return {
                    "has_tier": False,
                    "verdict": "无档赢 (扰动翻转)",
                    "text": (
                        f"{best_key} seed=42 TOP10+子窗赢, 但 seed=43 扰动 top-10 超额 "
                        f"{b43['top10']['excess']:+.4f} <= ref {r43['top10']['excess']:+.4f}"
                        f" — 不稳健, 不推荐"
                    ),
                    "ref": ref_key,
                    **detail,
                }
            detail["perturb_flip"] = False
    return {
        "has_tier": True,
        "verdict": f"有档: 推荐 {best_key}",
        "text": (
            f"有档: 推荐 bagging_fraction={best['params']['bagging_fraction']}, "
            f"bagging_freq={best['params']['bagging_freq']} — top-10 实得超额 "
            f"{best['top10']['excess']:+.4f} vs ref {ref_exc:+.4f} (TOP10 为主), "
            f"Rank IC {best['mean_ic']:.5f} vs {ref_ic:.5f} (次键), 子窗 {w}/3 赢, "
            f"扰动 seed=43 不翻转 (稳定 > 最高)"
        ),
        "ref": ref_key,
        **detail,
    }


def verdict_cls(
    results: dict[str, dict], perturbation: dict[str, dict], ref_key: str
) -> dict:
    """3d_cls 辅助: AUC 不降 (>= ref) + prob 唯一值不缩 (>= ref) → 退化不更糟."""
    ref = results[ref_key]
    ref_auc, ref_nuniq = ref["auc"], ref["prob_nunique"]
    ok = [
        k
        for k, rec in results.items()
        if k != ref_key
        and np.isfinite(rec["auc"])
        and np.isfinite(ref_auc)
        and rec["auc"] >= ref_auc
        and rec["prob_nunique"] >= ref_nuniq
    ]
    best_key = max(ok, key=lambda k: results[k]["auc"]) if ok else None
    if best_key is None:
        return {
            "has_tier": False,
            "verdict": "3d_cls: 无 bagging 组合同时保 AUC 且不缩 prob 唯一值",
            "text": (
                f"ref auc={ref_auc:.4f} prob_nunique={ref_nuniq}; 无组合两者同保 → "
                f"bagging 不能改善退化头, 该头维持现状 (已按 REG/CLS_MIN_TREES 地板保护)"
            ),
            "ref": ref_key,
        }
    best = results[best_key]
    return {
        "has_tier": True,
        "verdict": f"3d_cls: {best_key} 保 AUC 且 prob 更分散",
        "text": (
            f"{best_key} auc={best['auc']:.4f} (ref {ref_auc:.4f}), "
            f"prob_nunique={best['prob_nunique']} (ref {ref_nuniq}) — "
            f"bagging 未让退化头更糟, 辅助支持"
        ),
        "ref": ref_key,
        "best": best_key,
    }


def _row(rec: dict, key: str, is_ref: bool = False, perturb_of: str = "") -> dict:
    if rec.get("head", "").endswith("cls"):
        subw = {f"sub{i + 1}_auc": np.nan for i in range(3)}
        for i, v in enumerate(rec.get("subwindow_auc", [])[:3]):
            subw[f"sub{i + 1}_auc"] = v
        return {
            "combo": key,
            "head": rec.get("head", ""),
            "bagging_fraction": rec["params"]["bagging_fraction"],
            "bagging_freq": rec["params"]["bagging_freq"],
            "seed": rec.get("seed", ""),
            "n_trees": rec.get("n_trees", ""),
            "auc": rec.get("auc", np.nan),
            "prob_nunique": rec.get("prob_nunique", np.nan),
            **subw,
            "is_ref": is_ref,
            "perturb_of": perturb_of,
        }
    tn = rec.get("top10", {})
    subs = {f"sub{i + 1}_excess": np.nan for i in range(3)}
    for i, s in enumerate(rec.get("subwindows", [])[:3]):
        subs[f"sub{i + 1}_excess"] = s.get("topn_excess", np.nan)
        subs[f"sub{i + 1}_n_days"] = s.get("n_days", 0)
    return {
        "combo": key,
        "head": rec.get("head", ""),
        "bagging_fraction": rec["params"]["bagging_fraction"],
        "bagging_freq": rec["params"]["bagging_freq"],
        "seed": rec.get("seed", ""),
        "n_trees": rec.get("n_trees", ""),
        "mean_ic": rec.get("mean_ic", np.nan),
        "icir": rec.get("icir", np.nan),
        "pos_day_ratio": rec.get("pos_day_ratio", np.nan),
        "n_ic_days": rec.get("n_ic_days", 0),
        "top10_excess": tn.get("excess", np.nan),
        "top10_top_ret": tn.get("top_ret", np.nan),
        "top10_pool_ret": tn.get("pool_ret", np.nan),
        "top10_pos_day": tn.get("pos_day", np.nan),
        "top10_n_days": tn.get("n_days", 0),
        **subs,
        "is_ref": is_ref,
        "perturb_of": perturb_of,
    }


def _split_4way(
    frame: pd.DataFrame, window_total: int, test_days: int
) -> dict[str, pd.DataFrame]:
    """反锚四段切分, test_days 可覆盖 (默认 60 同生产; --test-window 125 干净段确认).

    镜像 dual_track_trainer.split_window 语义 (反锚靠最近端, train 吸收余量,
    calib=es 共享验证窗), 仅把 test 段长度参数化. 生产 split_window 一律不动 —
    125d 只用于本脚本评估窗口, 不改生产 test=60 验收.
    """
    dates = sorted(frame["date"].unique())[-window_total:]
    n = len(dates)
    es_d = 20  # 同生产 _ES_FLOOR
    if n <= test_days + es_d + MIN_TRAIN_DAYS:
        raise RuntimeError(
            f"[split] test_days={test_days} 需 n>{test_days + es_d + MIN_TRAIN_DAYS}, "
            f"实际 n={n}"
        )
    es_dates = dates[-test_days - es_d : -test_days]
    return {
        "train": frame[frame["date"].isin(dates[: -test_days - es_d])],
        "es": frame[frame["date"].isin(es_dates)],
        "calib": frame[frame["date"].isin(es_dates)],
        "test": frame[frame["date"].isin(dates[-test_days:])],
    }


def run_head(
    head: str,
    frame: pd.DataFrame,
    feature_cols: list[str],
    realized: pd.DataFrame,
    t0: float,
    max_n: int = 0,
    skip_perturb: bool = False,
    test_window: int = 60,
    combos_filter: list[str] | None = None,
) -> tuple[dict, dict, dict, list[dict], dict]:
    """单头: 切分 → 网格 → 扰动 → 判词.

    max_n > 0: 只跑前 max_n 个网格组合 + ref (冒烟模式).
    skip_perturb: 跳过 seed=43 扰动 (冒烟).
    test_window: OOS test 段日数 (60=生产; 125 干净段确认). 仅评估窗口, 不动生产.
    combos_filter: 只跑指定组合 (如 ['ref','f0.9_q2']), 125d 确认省时用.
    """
    is_cls = head.endswith("cls")
    # 标签 (生产 _resolve_label 语义)
    label = f"label_pm_{head.split('d')[0]}d" + ("_cls" if is_cls else "")
    label_net = f"{label}_net"
    if label_net in frame.columns:
        label = label_net
    if label not in frame.columns:
        raise RuntimeError(f"[{head}] 面板缺标签 {label}")

    keep = ["symbol", "date", "is_suspended", label] + feature_cols
    fmin = frame[keep].copy()
    for c in feature_cols:
        fmin[c] = fmin[c].astype("float32", copy=False)
    if test_window != 60:
        segs = _split_4way(fmin, WINDOW_TOTAL, test_window)
    else:
        segs = DualTrackTrainer.split_window(fmin, WINDOW_TOTAL)
    train = risk_filter(segs["train"].dropna(subset=[label]))
    es = risk_filter(segs["es"].dropna(subset=[label]))
    test = segs["test"]  # 不 dropna: 3d_cls 评估时自然掩 NaN; 10d realized merge 剔除
    del segs, fmin
    gc.collect()
    if len(test) < 500 or test["date"].nunique() < 30:
        raise RuntimeError(
            f"[{head}] OOS test 段不足: rows={len(test)} days={test['date'].nunique()}"
        )
    w = DualTrackTrainer.time_weights(train, HALF_LIFE)
    total_days = train["date"].nunique() + es["date"].nunique() + test["date"].nunique()
    print(
        f"[{head}] split train={train['date'].nunique()}d es={es['date'].nunique()}d "
        f"test={test['date'].nunique()}d total={total_days} label={label} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    if total_days < 715:
        raise RuntimeError(
            f"[{head}] 切分日数异常 {total_days}d (期望 ≈728d), 疑似窗口错位, 中止"
        )

    if combos_filter:
        combos = [(None, None) if k == "ref" else _parse_key(k) for k in combos_filter]
    else:
        grid_combos = [(f, q) for f in GRID_FRAC for q in GRID_FREQ]
        if max_n and max_n < len(grid_combos):
            grid_combos = grid_combos[:max_n]
        combos = grid_combos + [(None, None)]
    results: dict[str, dict] = {}
    for i, (frac, freq) in enumerate(combos):
        key = combo_key(frac, freq)
        tag = f"{head}/{key}"
        print(f"[{i + 1}/{len(combos)}] {tag} seed={RANDOM_STATE} ...", flush=True)
        rec, n_trees = fit_and_eval(
            head,
            train,
            es,
            test,
            feature_cols,
            w,
            realized,
            label,
            frac,
            freq,
            RANDOM_STATE,
        )
        results[key] = rec
        _print_rec(head, rec, key, n_trees, t0)

    # 扰动: ref + 最优 1-2, seed=43
    perturbation: dict[str, dict] = {}
    ref_key = "ref"
    if is_cls:
        ranked = sorted(
            (k for k in results if k != ref_key and np.isfinite(results[k]["auc"])),
            key=lambda k: (
                -results[k]["auc"],
                -results[k]["prob_nunique"],
            ),
        )
    else:
        ranked = sorted(
            (k for k in results if k != ref_key),
            key=lambda k: (
                -results[k]["top10"]["excess"],
                -results[k]["mean_ic"],
            ),
        )
    if not skip_perturb:
        perturb_keys = list(dict.fromkeys([ref_key] + ranked[:2]))
        print(
            f"\n[{head}] [perturb] seed={PERTURB_SEED} combos={perturb_keys}",
            flush=True,
        )
        for k in perturb_keys:
            if k == ref_key:
                frac, freq = None, None
            else:
                frac, freq = _parse_key(k)
            rec, _nt = fit_and_eval(
                head,
                train,
                es,
                test,
                feature_cols,
                w,
                realized,
                label,
                frac,
                freq,
                PERTURB_SEED,
            )
            perturbation[k] = rec
            _print_rec(head, rec, k, _nt, t0, suffix="seed=43")

    if is_cls:
        verdict = verdict_cls(results, perturbation, ref_key)
    else:
        verdict = verdict_reg(results, perturbation, ref_key)
    return results, perturbation, verdict, combos, {"label": label}


def _parse_key(key: str) -> tuple[float, int]:
    if key == "ref":
        return None, None
    fs, qs = key[1:].split("_")
    return float(fs), int(qs[1:])


def _print_rec(
    head: str, rec: dict, key: str, n_trees: int, t0: float, suffix: str = ""
) -> None:
    if head.endswith("cls"):
        subw = ",".join(
            f"{s:+.4f}" if np.isfinite(s) else "-" for s in rec.get("subwindow_auc", [])
        )
        print(
            f"    -> {key} auc={rec['auc']:.4f} nuniq={rec['prob_nunique']} "
            f"trees={n_trees} sub_auc=[{subw}] {suffix} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    else:
        tn = rec["top10"]
        ws = ",".join(
            f"{s['topn_excess']:+.4f}" if np.isfinite(s["topn_excess"]) else "-"
            for s in rec["subwindows"]
        )
        print(
            f"    -> {key} ic={rec['mean_ic']:.5f} icir={rec['icir']:.2f} "
            f"pos={rec['pos_day_ratio']:.2f} trees={n_trees} "
            f"top10_exc={tn['excess']:+.4f} (n={tn['n_days']}d) "
            f"sub_exc=[{ws}] {suffix} ({time.time() - t0:.0f}s)",
            flush=True,
        )


def _inject_brute_cols(frame: pd.DataFrame, missing: list[str]) -> pd.DataFrame:
    """生产 train_runner.select_features 的 BruteForce 后注入 (2026-08-11 OOM 修复版).

    FeatureEngineV35.build 不含 brute 族列; 生产在 select_features 里对选中缺失列
    按需注入. generate_columns 逐 symbol 流式, 只驻留 need 交集列, 位置赋值防 join
    宽帧 OOM. 镜像 train_runner.py:137-169.
    """
    gen = BruteForceGenerator()
    raw_cols = gen._eligible(frame)
    need = set(missing)
    picks = []
    for fam in BRUTE_FAMILIES:
        new = gen.generate_columns(frame, fam, need, raw_cols=raw_cols, dtype="float32")
        if new is None or not len(new.columns):
            continue
        picks.append(new)
    if picks:
        _brute = pd.concat(picks, axis=1)
        for _c in _brute.columns:
            frame[_c] = _brute[_c].to_numpy()
    return frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-n", type=int, default=0, help=">0 时每头只跑前 N 组合 (冒烟)"
    )
    ap.add_argument("--skip-perturb", action="store_true")
    ap.add_argument("--heads", default=",".join(HEADS), help="逗号分隔头列表")
    ap.add_argument(
        "--test-window",
        type=int,
        default=60,
        help="OOS test 段日数 (60=生产; 125 干净段确认; 仅本脚本评估, 不动生产 split_window)",
    )
    ap.add_argument(
        "--combos",
        default=None,
        help="只跑指定组合, 逗号分隔 (如 ref,f0.9_q2,f0.7_q5); 默认全网格",
    )
    ap.add_argument(
        "--use-built-frame",
        default=None,
        help="载入已构建特征帧 (feather, 含 brute 注入), 跳过 ~55min 面板构建",
    )
    args = ap.parse_args()
    heads = tuple(h.strip() for h in args.heads.split(",") if h.strip())

    t0 = time.time()
    if args.use_built_frame:
        # ── 0) 复用已构建特征帧 (含 brute 注入), 跳过面板/清洗/构建 ~55min ──
        wait_for_ram("cp-load")
        frame = pd.read_feather(args.use_built_frame)
        print(
            f"[cp-load] {len(frame):,}r {len(frame.columns)}c "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
    else:
        # ── 1) 面板: pyarrow 预过滤 + 3y 窗口 ──
        wait_for_ram("panel")
        panel = load_panel_v3(PANEL_V3_PATH)
        print(
            f"[load] {len(panel):,}r max={panel['date'].max()} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        cut = panel["date"].max() - pd.DateOffset(years=3)
        panel = panel[panel["date"] >= cut]
        print(
            f"[3y] {pd.Timestamp(cut).date()}.. {len(panel):,}r "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

        # ── 2) 清洗: 只拆 main 板 ──
        cleaner = CleaningPipeline()
        main_df, _dual_df = cleaner.run_train(panel, board="main")
        del panel, _dual_df
        gc.collect()
        print(
            f"[clean:main] {len(main_df):,}r dates={main_df['date'].nunique()} "
            f"symbols={main_df['symbol'].nunique()} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        if len(main_df) == 0:
            raise RuntimeError("main 清洗后无样本")

        # ── 3) 特征 + 标签 (生产同构, main 不加截面排名) ──
        wait_for_ram("build")
        registry = FeatureRegistry(path=REGISTRY_PATH)
        features = FeatureEngineV35()
        frame = prepare_board_frame(
            main_df, features, None, cross_sectional_rank=False, registry=registry
        )
        del main_df
        gc.collect()
        print(
            f"[feat] {len(frame):,}r {len(frame.columns)}c "
            f"label_pm_10d_net NaN={frame['label_pm_10d_net'].isna().mean():.2%} "
            f"label_pm_3d_cls_net NaN={frame['label_pm_3d_cls_net'].isna().mean():.2%} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    # ── 4) 特征列 = 生产 main_current bundle ──
    with open(MAIN_BUNDLE, "rb") as fh:
        bundle = pickle.load(fh)
    feature_cols = list(bundle["feature_cols"])
    miss = [c for c in feature_cols if c not in frame.columns]
    if miss:
        # brute 族列由生产 select_features 后注入; prepare_board_frame 不含 → 同构注入
        print(
            f"[inject] 特征 build 缺 {len(miss)} 列 (brute 族, 生产后注入), 注入...",
            flush=True,
        )
        frame = _inject_brute_cols(frame, miss)
        del miss
        gc.collect()
        miss = [c for c in feature_cols if c not in frame.columns]
        if miss:
            raise RuntimeError(f"注入后仍缺 {len(miss)} 列: {miss[:10]}")
    print(f"[feats] {len(feature_cols)} 列 = {MAIN_BUNDLE}", flush=True)

    # ── 5) 实得 (10d 排名键口径, 仅 10d_reg 用) ──
    realized = compute_realized(
        frame[["symbol", "date", "close_hfq"]],
        frame["date"],
        buy_lag=BUY_LAG,
        sell_lag=SELL_LAG,
        cost=COST,
    )
    print(
        f"[realized] {len(realized):,}r dates={realized['date'].nunique()}"
        f" ({time.time() - t0:.0f}s)",
        flush=True,
    )
    if realized.empty:
        raise RuntimeError("compute_realized 返回空")

    # 检查点 (仅完整构建时落盘): 后续崩溃重跑/125d 确认用 --use-built-frame 跳过构建
    if not args.use_built_frame:
        wait_for_ram("cp-save")
        BUILT_FRAME_CP.parent.mkdir(parents=True, exist_ok=True)
        frame[CP_COLS + feature_cols].to_feather(BUILT_FRAME_CP)
        print(
            f"[cp] 特征帧 ({len(CP_COLS) + len(feature_cols)} 列) → {BUILT_FRAME_CP} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    # ── 6) 逐头扫描 ──
    all_results: dict[str, dict] = {}
    all_perturb: dict[str, dict] = {}
    all_verdict: dict[str, dict] = {}
    all_labels: dict[str, str] = {}
    for head in heads:
        print(f"\n{'=' * 60}\n[HEAD] {head}\n{'=' * 60}", flush=True)
        combos_filter = (
            None
            if not args.combos
            else [c.strip() for c in args.combos.split(",") if c.strip()]
        )
        results, perturb, verdict, combos, meta = run_head(
            head,
            frame,
            feature_cols,
            realized,
            t0,
            max_n=args.max_n,
            skip_perturb=args.skip_perturb,
            test_window=args.test_window,
            combos_filter=combos_filter,
        )
        all_results[head] = results
        all_perturb[head] = perturb
        all_verdict[head] = verdict
        all_labels[head] = meta["label"]
        print(f"\n[{head}] === 判词 ===\n{verdict['text']}", flush=True)

    # ── 7) WORM 输出 ──
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    out = {
        "meta": {
            "board": "main",
            "heads": list(heads),
            "labels": all_labels,
            "grid": {
                "bagging_fraction": list(GRID_FRAC),
                "bagging_freq": list(GRID_FREQ),
                "seed": RANDOM_STATE,
                "perturb_seed": PERTURB_SEED,
                "num_leaves": {
                    h: NUM_LEAVES_OVERRIDE.get(("main", h), 31) for h in heads
                },
            },
            "ref": "无 bagging (生产现状)",
            "oos_days": None,
            "n_features": len(feature_cols),
            "feature_source": MAIN_BUNDLE,
            "panel": str(PANEL_V3_PATH),
            "cost": COST,
            "buy_lag": BUY_LAG,
            "sell_lag": SELL_LAG,
            "top_n": TOP_N,
            "realized": "drift_monitor.compute_realized: buy=T+1 close, sell=T+11 close, "
            "-0.002, 停牌 ffill (仅 10d_reg 用)",
            "run_at": ts,
        },
        "results": all_results,
        "perturbation": all_perturb,
        "verdict": all_verdict,
    }
    json_path = out_dir / f"legacy_bagging_sweep_{ts}.json"
    json_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    rows = []
    for head in heads:
        for k, rec in all_results[head].items():
            rows.append(_row(rec, k, is_ref=(k == "ref")))
        for k, rec in all_perturb[head].items():
            rows.append(_row(rec, k, perturb_of=k))
    df_csv = pd.DataFrame(rows)
    csv_path = out_dir / f"legacy_bagging_sweep_{ts}.csv"
    df_csv.to_csv(csv_path, index=False)
    print(f"\n[saved] {json_path}", flush=True)
    print(f"[saved] {csv_path} ({time.time() - t0:.0f}s)", flush=True)

    print("\n=== 判词汇总 ===", flush=True)
    for head in heads:
        print(f"[{head}] {all_verdict[head]['text']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
