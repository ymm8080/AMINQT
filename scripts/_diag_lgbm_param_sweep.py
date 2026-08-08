"""_diag_lgbm_param_sweep.py — legacy LGBM 超参扫描 (walk-forward, 无前瞻, 2026-08-08).

用户需求: legacy/parallel 模型里存在默认设置的参数, 遍历找更好的值 (类比 mag10d 扫描).
本脚本聚焦 legacy LGBM 超参: dual_track_trainer.LGB_PARAMS_REG/CLS 只设了
objective/n_estimators/learning_rate/random_state, 其余 (num_leaves/min_child_samples/
subsample/colsample_bytree/reg_lambda...) 全是 LightGBM 出厂默认, 从未遍历.

评估协议 (镜像生产口径 + 铁律):
  - 特征表 = 并行主检查点 (data/_diag_stage_main_3y.parquet, 已含特征+标签, 与生产同引擎
    FeatureEngineV35), 特征列 = 生产 current bundle 的 feature_cols (316 个, 全在检查点内).
  - OOS = 末 250 交易日 (只评估, 不训练; 验收只看 OOS 铁律). 逐日 Rank IC → mean/ICIR/正日率.
  - train = OOS 前全部日期 (主检查点 ~478 交易日), es = train 末 20 日 (早停, 镜像 _ES_FLOOR).
  - 拟合 = lgb.LGBMRegressor(huber, n_estimators=1000 上限, early_stopping patience=100,
    sample_weight=time_weights half-life 250), 与生产 _train_one 同构.
  - 输出 data/_diag_lgbm_{grid}_{ts}.json (WORM), 每组合含 逐日 IC 汇总.

用法: python scripts/_diag_lgbm_param_sweep.py [--board main|dual] [--kind 3d_reg|2d_reg,...]
      [--grid leaves_ms|sampling|reg] [--oos 250] [--es 20] [--max-n 0]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import lightgbm as lgb

from app.pipeline1.label_engine import LABEL_WEIGHTS

# 生产 LGB_PARAMS_REG 基线 (扫描时逐组合覆盖)
BASE_PARAMS = {
    "objective": "huber",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "random_state": 42,
    "verbosity": -1,
}
ES_PATIENCE = 100

CKPT = {
    "main": ("data/_diag_stage_main_3y.parquet", "models/pipeline1/main_current.pkl"),
    "dual": ("data/_diag_stage_dual_3y.parquet", "models/pipeline1/dual_current.pkl"),
}

# 预设网格: 每轮扫一组, 永远叠加"全默认"组合作基线
GRIDS = {
    # R1 树结构 (num_leaves 默认31, min_child_samples 默认20)
    "leaves_ms": {
        "num_leaves": [15, 31, 63],
        "min_child_samples": [5, 20, 50],
    },
    # R2 采样 (colsample_bytree 默认1.0, subsample 默认1.0)
    "sampling": {
        "colsample_bytree": [0.6, 0.8, 1.0],
        "subsample": [0.7, 0.9, 1.0],
    },
    # R3 正则 + 学习率 (reg_lambda 默认0, lr 默认0.05)
    "reg": {
        "reg_lambda": [0.0, 1.0, 10.0],
        "learning_rate": [0.03, 0.05, 0.08],
    },
}

# 复权标签列 (reg 族): kind → label 列
KIND_LABEL = {
    "1d_reg": "label_pm_1d_net",
    "2d_reg": "label_pm_2d_net",
    "3d_reg": "label_pm_3d_net",
    "5d_reg": "label_pm_5d_net",
    "10d_reg": "label_pm_10d_net",
    # cls 族 (binary 分类, 输出概率 → 对 cls 净标签做 Rank IC, 与 validate_oos 口径一致)
    "1d_cls": "label_pm_cls_net",
    "2d_cls": "label_pm_2d_cls_net",
    "3d_cls": "label_pm_3d_cls_net",
    "5d_cls": "label_pm_5d_cls_net",
    "10d_cls": "label_pm_10d_cls_net",
    # quantile 族 (E1): pred_q50 中位数 → 对净标签做 Rank IC (E7 闸3 用中位数)
    "1d_q": "label_pm_1d_net",
    "2d_q": "label_pm_2d_net",
    "3d_q": "label_pm_3d_net",
    "5d_q": "label_pm_5d_net",
    "10d_q": "label_pm_10d_net",
    # rank 族 (阶段四 LambdaRank): rank score → 对净标签做 Rank IC
    "3d_rank": "label_pm_3d_net",
    "5d_rank": "label_pm_5d_net",
    "10d_rank": "label_pm_10d_net",
    # pain (E2): label_pain 极不均衡, AUC 评估 (无 10d 语义, label 固定 3日浮亏)
    "pain": "label_pain",
}
KIND_HORIZON = {
    "1d_reg": 1, "2d_reg": 2, "3d_reg": 3, "5d_reg": 5, "10d_reg": 10,
    "1d_cls": 1, "2d_cls": 2, "3d_cls": 3, "5d_cls": 5, "10d_cls": 10,
    "1d_q": 1, "2d_q": 2, "3d_q": 3, "5d_q": 5, "10d_q": 10,
    "3d_rank": 3, "5d_rank": 5, "10d_rank": 10,
}


def per_day_rank_ic(df: pd.DataFrame, pred_col: str, label_col: str) -> dict:
    """逐日截面 Rank IC (Spearman): 返回 {mean_ic, icir, pos_day_ratio, n_days, mean_abs}."""
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
        cc = np.corrcoef(sub["pred"].values, sub["lab"].values)[0, 1]
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


def per_day_topn(
    df: pd.DataFrame, pred_col: str, label_col: str, ns: tuple[int, ...] = (10, 15, 20)
) -> dict:
    """金标准: 每日取预测前 N 只, 实得收益 vs 当日全池基线 (超额收益).

    excess = topN 平均实得收益 - 当日全池平均实得收益 (主评 10d 口径).
    返回 {str(N): {"excess", "top_ret", "pool_ret", "pos_day", "n_days"}}.
    """
    out: dict[int, dict] = {
        n: {"top_ret": [], "pool_ret": [], "excess": [], "pos_day": []} for n in ns
    }
    for _, sub in df.groupby("date"):
        if len(sub) < 5:
            continue
        lab = sub[label_col].to_numpy(dtype=float)
        pool = float(np.nanmean(lab))
        order = np.argsort(sub[pred_col].to_numpy(dtype=float))[::-1]
        for n in ns:
            k = min(n, len(sub))
            top_ret = float(np.nanmean(lab[order[:k]]))
            out[n]["top_ret"].append(top_ret)
            out[n]["pool_ret"].append(pool)
            out[n]["excess"].append(top_ret - pool)
            out[n]["pos_day"].append(1.0 if top_ret > pool else 0.0)
    res: dict[str, dict] = {}
    for n in ns:
        e = np.asarray(out[n]["excess"], dtype=float)
        if len(e) == 0:
            res[str(n)] = {
                "excess": np.nan, "top_ret": np.nan, "pool_ret": np.nan,
                "pos_day": np.nan, "n_days": 0,
            }
            continue
        res[str(n)] = {
            "excess": float(e.mean()),
            "top_ret": float(np.asarray(out[n]["top_ret"]).mean()),
            "pool_ret": float(np.asarray(out[n]["pool_ret"]).mean()),
            "pos_day": float((e > 0).mean()),
            "n_days": int(len(e)),
        }
    return res


def window_breakdown(
    df: pd.DataFrame,
    pred_col: str,
    label_col: str,
    n: int,
    k: int = 3,
    use_auc: bool = False,
) -> list[dict]:
    """OOS 按时间切 k 段, 每段 {primary, topn_excess} — 参数稳定性检验.

    铁律 (Kimi 扫参可靠性): 最优参数若在相邻时间窗口翻转 → 噪声, 弃用.
    """
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
        if len(sub) < 5:
            out.append({"primary": np.nan, "topn_excess": np.nan})
            continue
        prim = (
            per_day_auc(sub, pred_col, label_col).get("mean_auc", np.nan)
            if use_auc
            else per_day_rank_ic(sub, pred_col, label_col).get("mean_ic", np.nan)
        )
        tn = per_day_topn(sub, pred_col, label_col).get(str(n), {})
        out.append({"primary": prim, "topn_excess": tn.get("excess", np.nan)})
    return out


def time_weights(df: pd.DataFrame, half_life: int = 250) -> np.ndarray:
    dates = sorted(df["date"].unique())
    age = {d: len(dates) - 1 - i for i, d in enumerate(dates)}
    return np.array([0.5 ** (age[d] / half_life) for d in df["date"]])


def build_slices(
    panel: pd.DataFrame,
    feature_cols: list[str],
    kind: str,
    oos_n: int,
    es_n: int,
) -> dict:
    """从全面板切出 (train/es/oos) 三个矩阵 + 元数据, 无前瞻."""
    label = KIND_LABEL[kind]
    dates = sorted(panel["date"].unique())
    if len(dates) <= oos_n + es_n + 50:
        raise RuntimeError(f"日期不足: total={len(dates)} oos={oos_n} es={es_n}")
    oos_dates = set(dates[-oos_n:])
    train_dates = dates[: -oos_n]  # 全部 OOS 前日期
    es_dates = set(train_dates[-es_n:])
    fit_dates = set(train_dates[:-es_n])

    m = panel["date"].isin(fit_dates) & ~panel["is_suspended"].astype(bool)
    fit_df = panel.loc[m].dropna(subset=[label])
    m_es = panel["date"].isin(es_dates) & ~panel["is_suspended"].astype(bool)
    es_df = panel.loc[m_es].dropna(subset=[label])
    oos_df = panel.loc[
        panel["date"].isin(oos_dates) & ~panel["is_suspended"].astype(bool)
    ].dropna(subset=[label])
    # 金标准实得收益列: cls 族用连续净收益 (label_pm_{h}d_net), 其余族 label 即连续收益
    real_label = label
    if kind.endswith("_cls"):
        h = KIND_HORIZON.get(kind)
        cand = f"label_pm_{h}d_net" if h else None
        if cand and cand in panel.columns:
            real_label = cand
            oos_df = oos_df.dropna(subset=[real_label])

    # 铁律: 特征矩阵只含 feature_cols, label 只能作 y (绝不能混进 X, 否则模型拿 label 预测自己 → 假 IC=1)
    X = np.nan_to_num(fit_df[feature_cols].values.astype("float32", copy=False), nan=0.0)
    y = fit_df[label].values
    X_es = np.nan_to_num(es_df[feature_cols].values.astype("float32", copy=False), nan=0.0)
    y_es = es_df[label].values
    oos_X = np.nan_to_num(oos_df[feature_cols].values.astype("float32", copy=False), nan=0.0)
    return {
        "X": X, "y": y, "w": time_weights(fit_df),
        "fit_dates": fit_df["date"].values,  # 与 X 行对齐, rank 模型按 date 分组
        "X_es": X_es, "y_es": y_es,
        "es_dates": es_df["date"].values,
        "oos_frame": oos_df[["date", "symbol", label] + ([real_label] if real_label != label else [])].copy(),
        "oos_X": oos_X,
        "label": label,
        "real_label": real_label,
        "n_fit_days": len(fit_dates), "n_es_days": len(es_dates),
        "n_train": int(len(fit_df)), "n_oos": int(len(oos_df)),
    }


def _fit_regressor(s: dict, params: dict):
    use_es = len(s["y_es"]) >= 1000
    model = lgb.LGBMRegressor(**params)
    model.fit(
        s["X"], s["y"], sample_weight=s["w"],
        eval_set=[(s["X_es"], s["y_es"])] if use_es else None,
        callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)] if use_es else None,
    )
    return model


def _fit_classifier(s: dict, params: dict):
    """cls 族: binary, 输出正类概率 → 对 cls 净标签 Rank IC (validate_oos 口径)."""
    use_es = len(s["y_es"]) >= 1000
    cls_params = {k: v for k, v in params.items() if k != "objective"}
    cls_params["objective"] = "binary"
    model = lgb.LGBMClassifier(**cls_params)
    model.fit(
        s["X"], s["y"], sample_weight=s["w"],
        eval_set=[(s["X_es"], s["y_es"])] if use_es else None,
        callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)] if use_es else None,
    )
    return model


def _fit_quantile(s: dict, params: dict):
    """quantile 族 (E1 口径): 5 分位 LGBMRegressor, 返回 pred_q50 预测."""
    from app.pipeline1.quantile_models import QuantileModelSet

    use_es = len(s["y_es"]) >= 1000
    base = {k: v for k, v in params.items() if k != "objective"}
    qset = QuantileModelSet(base).fit(
        s["X"], s["y"], sample_weight=s["w"],
        eval_set=(s["X_es"], s["y_es"]) if use_es else None,
        es_patience=ES_PATIENCE,
    )
    pred = qset.predict(s["oos_X"])["pred_q50"].values
    # n_trees 取中位数分位模型 (代表)
    n_trees = int(getattr(qset.models[0.50], "best_iteration_", 0) or params.get("n_estimators", 0))
    return pred, n_trees


def _sort_by_date(X: np.ndarray, y: np.ndarray, dates: np.ndarray, w: np.ndarray):
    """按 date 稳定排序 (rank 模型要求同 date 行连续以构成 group)."""
    order = np.argsort(dates, kind="stable")
    return X[order], y[order], dates[order], w[order]


def _rank_gains(y: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """生产口径 (dual_track_trainer._train_ranker): date 截面分位 → gain 0-4."""
    df = pd.DataFrame({"date": dates, "y": y})
    g = df.groupby("date")["y"].rank(pct=True).pipe(lambda s: (s * 5).clip(0, 4.999).astype(int))
    return g.values


def _group_sizes(dates: np.ndarray) -> np.ndarray:
    """rank 模型 group = 每个 date 的样本数 (dates 需已排序)."""
    _, counts = np.unique(dates, return_counts=True)
    return counts


def _fit_ranker(s: dict, params: dict):
    """rank 族 (阶段四 LambdaRank 口径): 净标签截面分位 gain 0-4, group=date, 返回 rank score."""
    import lightgbm as lgb

    use_es = len(s["y_es"]) >= 1000
    X, y, dates, w = _sort_by_date(s["X"], s["y"], s["fit_dates"], s["w"])
    rp = {k: v for k, v in params.items() if k != "objective"}
    rp["objective"] = "lambdarank"
    rp["lambdarank_truncation_level"] = 25  # 生产定值 (V3.8 §2.2)
    model = lgb.LGBMRanker(**rp)
    gains = _rank_gains(y, dates)
    group = _group_sizes(dates)
    kwargs = {}
    if use_es:
        X_es, y_es, dates_es, w_es = _sort_by_date(s["X_es"], s["y_es"], s["es_dates"], np.ones(len(s["es_dates"])))
        gains_es = _rank_gains(y_es, dates_es)
        kwargs = {
            "eval_set": [(X_es, gains_es)],
            "eval_group": [_group_sizes(dates_es)],
            "callbacks": [lgb.early_stopping(ES_PATIENCE, verbose=False)],
        }
    model.fit(X, gains, group=group, sample_weight=w, **kwargs)
    pred = model.predict(s["oos_X"])
    n_trees = int(model.best_iteration_ or params.get("n_estimators", 0))
    return pred, n_trees


def _fit_pain(s: dict, params: dict):
    """pain 族 (E2): label_pain 分类 → pain_prob, AUC 评估."""
    from app.pipeline1.quantile_models import PainModel

    use_es = len(s["y_es"]) >= 1000
    base = {k: v for k, v in params.items() if k != "objective"}
    pain = PainModel(base).fit(
        s["X"], s["y"], sample_weight=s["w"],
        eval_set=(s["X_es"], s["y_es"]) if use_es else None,
        es_patience=ES_PATIENCE,
    )
    pred = pain.predict_proba(s["oos_X"])
    n_trees = int(getattr(pain.model, "best_iteration_", 0) or params.get("n_estimators", 0))
    return pred, n_trees


def per_day_auc(df: pd.DataFrame, pred_col: str, label_col: str) -> dict:
    """逐日 AUC (pain 极不均衡, 用 AUC 而非 Rank IC)."""
    from sklearn.metrics import roc_auc_score

    vals = []
    for _d, sub in df.groupby("date"):
        if len(sub) < 5:
            continue
        try:
            a = roc_auc_score(sub[label_col].values, sub[pred_col].values)
        except ValueError:
            continue  # 单类日内无 AUC
        if np.isfinite(a):
            vals.append(a)
    daily = np.asarray(vals, dtype=float)
    if len(daily) == 0:
        return {"mean_auc": np.nan, "n_days": 0, "pos_day_ratio": np.nan}
    return {
        "mean_auc": float(daily.mean()),
        "n_days": int(len(daily)),
        "pos_day_ratio": float((daily > 0.5).mean()),
    }


def fit_eval(s: dict, params: dict, pred_col: str, kind: str = "", n: int = 15) -> dict:
    use_cls = kind.endswith("_cls")
    use_q = kind.endswith("_q")
    use_rank = kind.endswith("_rank")
    use_pain = kind == "pain"
    if use_pain:
        pred, n_trees = _fit_pain(s, params)
    elif use_cls:
        model = _fit_classifier(s, params)
        pred = model.predict_proba(s["oos_X"])[:, 1]
        n_trees = int(model.best_iteration_ or params.get("n_estimators", 0))
    elif use_q:
        pred, n_trees = _fit_quantile(s, params)
    elif use_rank:
        pred, n_trees = _fit_ranker(s, params)
    else:
        model = _fit_regressor(s, params)
        pred = model.predict(s["oos_X"])
        n_trees = int(model.best_iteration_ or params.get("n_estimators", 0))
    oos = s["oos_frame"].copy()
    oos[pred_col] = pred
    topn = per_day_topn(oos, pred_col, s["real_label"])
    wins = window_breakdown(oos, pred_col, s["real_label"], n, use_auc=use_pain)
    if use_pain:
        ic = per_day_auc(oos, pred_col, s["label"])
        ic["n_trees"] = n_trees
        ic["topn"] = topn
        ic["windows"] = wins
        return ic
    ic = per_day_rank_ic(oos, pred_col, s["label"])
    ic["n_trees"] = n_trees
    ic["topn"] = topn
    ic["windows"] = wins
    return ic


def expand_grid(grid: dict) -> list[dict]:
    """网格 → 组合列表 (全组合 + 恒含全默认基线)."""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    import itertools

    combos = [dict(zip(keys, c)) for c in itertools.product(*values)]
    default = {}
    if "num_leaves" in keys:
        default["num_leaves"] = 31
    if "min_child_samples" in keys:
        default["min_child_samples"] = 20
    if "colsample_bytree" in keys:
        default["colsample_bytree"] = 1.0
    if "subsample" in keys:
        default["subsample"] = 1.0
    if "reg_lambda" in keys:
        default["reg_lambda"] = 0.0
    if "learning_rate" in keys:
        default["learning_rate"] = 0.05
    if default not in combos:
        combos.append(default)
    return combos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="main", choices=["main", "dual"])
    ap.add_argument("--kind", default="3d_reg",
                    help="逗号分隔 reg 族 (默认 3d_reg; 终选可用 2d,3d,5d,10d_reg)")
    ap.add_argument("--grid", default="leaves_ms", choices=list(GRIDS))
    ap.add_argument(
        "--fixed",
        default="",
        help="额外固定超参, 如 'num_leaves=63,min_child_samples=20' (覆盖 BASE_PARAMS, 供 R2/R3 承接 R1 最优)",
    )
    ap.add_argument(
        "--cand",
        default="",
        help="终选: 只跑指定候选组合 (分号分隔), 如 'num_leaves=31,min_child_samples=20;num_leaves=63,min_child_samples=20'",
    )
    ap.add_argument("--oos", type=int, default=250)
    ap.add_argument("--es", type=int, default=20)
    ap.add_argument("--n", type=int, default=15, help="TOP-N 金标准评估的名单长度 (默认 15)")
    ap.add_argument("--max-n", type=int, default=0, help=">0 时只跑前 N 个组合 (调试)")
    args = ap.parse_args()
    args.kind = args.kind.replace(" ", "")

    fixed = {}
    for kv in args.fixed.replace(" ", "").split(","):
        if not kv:
            continue
        k, v = kv.split("=")
        fixed[k] = float(v) if "." in v or "e" in v.lower() else int(v)

    ckpt_path, bundle_path = CKPT[args.board]
    t0 = time.time()
    bundle = pd.read_pickle(bundle_path)
    feature_cols = [c for c in bundle["feature_cols"]]
    panel = pd.read_parquet(ckpt_path)
    miss = [c for c in feature_cols if c not in panel.columns]
    if miss:
        raise RuntimeError(f"检查点缺特征 {len(miss)} 个: {miss[:10]}")
    need = ["date", "symbol", "is_suspended"] + feature_cols
    panel = panel[need + [c for c in panel.columns if c in KIND_LABEL.values()]]
    # 派生缺失的 cls_net 标签 (label_engine 口径: cls_net = (net > 0); 并行检查点只有 reg net 标签)
    for h in (1, 2, 3, 5, 10):
        col = f"label_pm_{h}d_cls_net"
        if col in KIND_LABEL.values() and col not in panel.columns:
            base = f"label_pm_{h}d_net"
            if base in panel.columns:
                panel[col] = (panel[base] > 0).astype("float")
                panel.loc[panel[base].isna(), col] = np.nan
                print(f"[derive] {col} <- {base}>0", flush=True)
    print(
        f"[data] board={args.board} rows={len(panel):,} dates={panel['date'].nunique()} "
        f"feats={len(feature_cols)} ({time.time() - t0:.0f}s)", flush=True
    )

    kinds = [k.strip() for k in args.kind.split(",") if k.strip() in KIND_LABEL]
    if args.cand:
        combos = []
        for spec in args.cand.split(";"):
            c = {}
            for kv in spec.split(","):
                if not kv:
                    continue
                k, v = kv.split("=")
                c[k] = float(v) if "." in v else int(v)
            combos.append(c)
        print(f"[cand] {len(combos)} 个候选组合", flush=True)
    else:
        combos = expand_grid(GRIDS[args.grid])
        if args.max_n > 0:
            combos = combos[: args.max_n]
        print(f"[grid] {args.grid}: {len(combos)} 组合 (含默认基线)", flush=True)

    results: dict[str, dict] = {}  # combo_key -> {kind: ic, weighted_ic, params}
    for kind in kinds:
        s = build_slices(panel, feature_cols, kind, args.oos, args.es)
        print(
            f"[{kind}] train={s['n_train']:,}r/{s['n_fit_days']}d es={s['n_es_days']}d "
            f"oos={s['n_oos']:,}r label={s['label']}", flush=True
        )
        for ci, combo in enumerate(combos):
            params = {**BASE_PARAMS, **fixed, **combo}
            pkey = ",".join(f"{k}={v}" for k, v in combo.items()) or "default"
            print(f"  [{ci+1}/{len(combos)}] {pkey} ...", flush=True)
            ic = fit_eval(s, params, "_pred", kind, args.n)
            results.setdefault(pkey, {"params": combo, "kinds": {}})
            results[pkey]["kinds"][kind] = ic
            met = f"auc={ic['mean_auc']:.4f}" if "mean_auc" in ic else f"ic={ic['mean_ic']:.5f}"
            tn = ic.get("topn", {}).get(str(args.n))
            tn_s = f" topn{args.n}exc={tn['excess']:.4f}" if tn else ""
            wins = ic.get("windows")
            win_s = ""
            if wins:
                prim = ",".join(
                    f"{w['primary']:.4f}" if np.isfinite(w["primary"]) else "-"
                    for w in wins
                )
                tne = ",".join(
                    f"{w['topn_excess']:.4f}" if np.isfinite(w["topn_excess"]) else "-"
                    for w in wins
                )
                win_s = f" win_prim=[{prim}] win_tn=[{tne}]"
            print(
                f"    -> {met} icir={ic.get('icir', 0):.2f} "
                f"pos={ic['pos_day_ratio']:.2f} trees={ic['n_trees']} "
                f"n_days={ic['n_days']} ({time.time() - t0:.0f}s){tn_s}{win_s}", flush=True
            )
        del s
        gc.collect()

    # 加权 IC (跨 kind, LABEL_WEIGHTS 口径; 1d 权重=0 自动忽略; pain 用 mean_auc 不进加权)
    def _ic_val(ic: dict) -> float:
        return ic.get("mean_ic", ic.get("mean_auc", float("nan")))

    for _pkey, rec in results.items():
        tot, wt = 0.0, 0.0
        for k, ic in rec["kinds"].items():
            v = ic.get("mean_ic", float("nan"))
            if not np.isfinite(v):
                continue
            w = LABEL_WEIGHTS.get(KIND_HORIZON[k], 0.0)
            tot += w * v
            wt += w
        rec["weighted_ic"] = tot / wt if wt else np.nan

    rows = sorted(
        results.items(), key=lambda kv: (kv[1]["weighted_ic"] is np.nan, kv[1]["weighted_ic"]),
        reverse=True,
    )
    print("\n=== leaderboard (weighted_ic) ===", flush=True)
    for _pkey, rec in rows:
        kk = ",".join(f"{k}={v}" for k, v in rec["params"].items())
        print(f"  {rec['weighted_ic']:.5f}  [{kk}]  " + "  ".join(
            f"{k}:{_ic_val(v):.5f}" for k, v in rec['kinds'].items()), flush=True
        )

    # TOP-N 金标准 leaderboard (按第一 kind 的 topN 超额收益排序)
    def _tn_exc(rec: dict):
        for v in rec["kinds"].values():
            if v.get("topn") and v["topn"].get(str(args.n)):
                return v["topn"][str(args.n)]["excess"]
        return np.nan

    trows = sorted(
        results.items(),
        key=lambda kv: (np.isnan(_tn_exc(kv[1])), _tn_exc(kv[1])),
        reverse=True,
    )
    print(f"\n=== leaderboard (topn{args.n} excess, 金标准) ===", flush=True)
    for _pkey, rec in trows:
        e = _tn_exc(rec)
        kk = ",".join(f"{k}={v}" for k, v in rec["params"].items())
        print(f"  topn{args.n}exc={e:.4f}  [{kk}]", flush=True)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "meta": {
            "board": args.board, "kinds": kinds, "grid": args.grid, "oos_days": args.oos,
            "es_days": args.es, "n_top": args.n, "ckpt": os.path.basename(ckpt_path),
            "n_feats": len(feature_cols), "run_at": ts,
        },
        "results": {k: {"params": v["params"], "kinds": v["kinds"],
                        "weighted_ic": v["weighted_ic"]} for k, v in results.items()},
    }
    path = f"data/_diag_lgbm_{args.grid}_{ts}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(f"[saved] {path} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
