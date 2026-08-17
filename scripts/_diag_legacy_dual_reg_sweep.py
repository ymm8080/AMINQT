"""_diag_legacy_dual_reg_sweep.py — legacy dual 10d_reg 超参补扫 (2026-08-17).

背景: legacy dual 10d_reg 排名信号极弱 (250d 诊断: 日截面 Pearson corr 0.015 vs
main 0.149, top-5 超额 +0.29pp vs +3.55pp). 08-08 只扫了 num_leaves (dual reg 定案
15, 已入 NUM_LEAVES_OVERRIDE, 勿动); min_child_samples / reg_lambda 从未扫过
(LightGBM 出厂默认 20 / 0). 本脚本补扫这两参, 目标提升 dual 10d_reg 泛化/排名
质量 (用户只关心 top-10 质量).

训练同构生产 (dual_track_trainer, 只训 10d_reg 一个头):
  - 数据: PANEL_V3_PATH → load_panel_v3 预过滤 (amount>=5e7 + 非停牌, pyarrow 下推)
    → 3y 窗口 (镜像 run_training) → CleaningPipeline.run_train(board='dual') →
    prepare_board_frame (FeatureEngineV35 全量 build + LabelEngine 标签, 生产同构)
  - 特征列 = models/pipeline1/dual_current.pkl 的 feature_cols (58)
  - split_window (WINDOW_TOTAL=770 反锚四段) + risk_filter + time_weights
    (HALF_LIFE=250) + huber + n_estimators=1000 + lr=0.05 + early_stopping
    (ES_PATIENCE=100) + label_pm_10d_net + random_state=42

网格: min_child_samples {20,50,80,100} × reg_lambda {0,1,5,10}, num_leaves=15 固定,
其余 = LGB_PARAMS_REG 基线. 基线 = 全默认组 (ms=20, λ=0) 作 ref (共 16 组合含 ref).

评估 (验收只看 OOS 铁律): split_window 的 test 段 (末 60 交易日) 为 OOS:
  a. 逐日 Rank IC (Spearman: pred vs 当日 test 段内 label_pm_10d_net) → mean/ICIR/正日率
  b. TOP-10 实得: 每 OOS 日按 pred 降序取 top-10, drift_monitor.compute_realized
     口径 (买 T+1 收盘, 卖 T+11 收盘, -0.002, 停牌 ffill) → 均值 vs 池子均值 (超额)
  c. 子窗稳定性: OOS 按日切 3 等长子窗, 每子窗 top-10 超额
  d. 扰动: 最优 1-2 组合 + ref 用 random_state=43 重训, 确认 top-10 超额不翻转

判词: 组合比 ref 在 top-10 超额 AND Rank IC 上同时赢 + 子窗多数 (>=2/3) 赢 →
有档 (选稳定 > 最高); 否则 "无档赢, 保持生产默认".

内存铁律: 每 fit 前 psutil free RAM < 5GB → sleep 60s 重试 (最多 30min, 防并发
OOM); 串行训练; 每组合 fit 完立即 del 大对象 + gc.collect(); 面板 pyarrow 预过滤.

WORM: DATA_OTHERS/diag/legacy_dual_reg_sweep_<ts>.json/.csv
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

import numpy as np
import pandas as pd
import lightgbm as lgb

from app.pipeline1.cleaning_pipeline import CleaningPipeline, load_panel_v3
from app.pipeline1.drift_monitor import compute_realized
from app.pipeline1.dual_track_trainer import (
    DualTrackTrainer,
    ES_PATIENCE,
    HALF_LIFE,
    LGB_PARAMS_REG,
    WINDOW_TOTAL,
    risk_filter,
)
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.feature_registry import FeatureRegistry
from app.pipeline1.train_runner import prepare_board_frame
from config.settings import PANEL_V3_PATH, data_others_path

# ── 网格 / 协议 ─────────────────────────────────────────────
GRID_MS = (20, 50, 80, 100)
GRID_LAMBDA = (0.0, 1.0, 5.0, 10.0)
NUM_LEAVES = 15  # 08-08 扫描定案, 勿动
RANDOM_STATE = 42  # 生产 LGB_PARAMS_REG 固定种子
PERTURB_SEED = 43  # 扰动种子
KIND = "10d_reg"
LABEL = "label_pm_10d_net"
COST = 0.0020  # compute_realized 往返成本 (与 drift_monitor 一致)
BUY_LAG, SELL_LAG = 1, 11  # 买 T+1 收盘, 卖 T+11 收盘
TOP_N = 10
RAM_MIN_FREE_GB = 5.0
RAM_MAX_WAIT_S = 1800  # 最多等 30 分钟
DUAL_BUNDLE = "models/pipeline1/dual_current.pkl"
REGISTRY_PATH = str(data_others_path("data/factor_registry/feature_registry.json"))


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


# ── 评估 (与 _diag_lgbm_param_sweep 同口径, 实得列换 realized_net) ──
def per_day_rank_ic(df: pd.DataFrame, pred_col: str, label_col: str) -> dict:
    """逐日截面 Rank IC (Spearman) → {mean_ic, icir, pos_day_ratio, n_days, mean_abs}."""
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
        # 逐日掩码 NaN (label 缺未来数据/停牌 → 每日截面都有 NaN 行, 08-17 教训:
        # 不掩码 np.corrcoef 恒 NaN → n_ic_days=0 假结果)
        p = sub["pred"].values
        l = sub["lab"].values
        m = np.isfinite(p) & np.isfinite(l)
        if m.sum() < 5:
            continue
        cc = np.corrcoef(p[m], l[m])[0, 1]
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
    """金标准: 每日预测前 N 只实得均值 vs 当日全池实得均值 (超额), n_days 等."""
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


def combo_key(ms: int, lam: float) -> str:
    return f"ms{ms}_lam{lam:g}"


def is_ref(ms: int, lam: float) -> bool:
    return ms == 20 and lam == 0.0


# ── 单组合: 生产同构训练 + OOS 评估 ──────────────────────────
def fit_and_eval(
    train: pd.DataFrame,
    es: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    w: np.ndarray,
    realized: pd.DataFrame,
    ms: int,
    lam: float,
    seed: int,
) -> tuple[dict, int]:
    """与生产 _train_one 同构: huber/1000 树/lr=0.05/num_leaves=15/es 早停/sample_weight."""
    wait_for_ram(combo_key(ms, lam))
    params = dict(LGB_PARAMS_REG)
    params.update(
        {
            "num_leaves": NUM_LEAVES,
            "min_child_samples": ms,
            "reg_lambda": lam,
            "random_state": seed,
        }
    )
    X = np.nan_to_num(train[feature_cols].values, nan=0.0)
    y = train[LABEL].values
    X_es = np.nan_to_num(es[feature_cols].values, nan=0.0)
    y_es = es[LABEL].values
    model = lgb.LGBMRegressor(**params)
    use_es = es["date"].nunique() >= 20  # 生产 MIN_ES_DATES 语义
    model.fit(
        X,
        y,
        sample_weight=w,
        eval_set=[(X_es, y_es)] if use_es else None,
        callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)] if use_es else None,
    )
    n_trees = int(model.best_iteration_ or params["n_estimators"])
    if n_trees <= 5:
        # 08-17 首跑教训: es 窗过平 → 早停 1 树 → 16 组合全雷同 (假结果).
        # ≤5 树时打印 eval 曲线前 5 轮供诊断 (默认 metric=l2).
        ev = getattr(model, "evals_result_", None)
        curve = ""
        if ev and "valid_0" in ev and ev["valid_0"]:
            mname = list(ev["valid_0"])[0]
            scores = ev["valid_0"][mname]
            curve = "[" + ",".join(f"{s:.4f}" for s in scores[:5]) + (",..." if len(scores) > 5 else "") + "]"
        print(
            f"    [!] {combo_key(ms, lam)} seed={seed} 早停仅 {n_trees} 树 "
            f"(es 窗可能过平) eval_curve{curve}",
            flush=True,
        )
    pred = model.predict(np.nan_to_num(test[feature_cols].values, nan=0.0))
    del model, X, X_es
    gc.collect()

    oos = test[["date", "symbol", LABEL]].copy()
    oos["_pred"] = pred
    oos = oos.merge(realized, on=["date", "symbol"], how="inner")
    if oos.empty:
        raise RuntimeError(f"{combo_key(ms, lam)}: OOS 无 realized 行, 数据不足")
    ic = per_day_rank_ic(oos, "_pred", LABEL)
    tn = per_day_topn_realized(oos, "_pred", "realized_net", TOP_N)
    wins = window_topn_excess(oos, "_pred", "realized_net", TOP_N, 3)
    del oos
    gc.collect()
    return {
        "params": {"min_child_samples": ms, "reg_lambda": lam, "num_leaves": NUM_LEAVES},
        "seed": seed,
        "n_trees": n_trees,
        "n_train_rows": int(len(train)),
        "n_train_days": int(train["date"].nunique()),
        "n_es_days": int(es["date"].nunique()),
        "mean_ic": ic.get("mean_ic", np.nan),
        "icir": ic.get("icir", np.nan),
        "pos_day_ratio": ic.get("pos_day_ratio", np.nan),
        "n_ic_days": ic.get("n_days", 0),
        "top10": tn,
        "subwindows": wins,
    }, n_trees


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=0, help=">0 时只跑前 N 个组合 (冒烟)")
    ap.add_argument("--skip-perturb", action="store_true", help="跳过扰动重训")
    args = ap.parse_args()

    t0 = time.time()
    # ── 1) 面板: pyarrow 预过滤 + 3y 窗口 (镜像 run_training) ──
    wait_for_ram("panel")
    panel = load_panel_v3(PANEL_V3_PATH)
    print(
        f"[load] {len(panel):,}r max={panel['date'].max()} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    cut = panel["date"].max() - pd.DateOffset(years=3)
    panel = panel[panel["date"] >= cut]
    print(
        f"[3y] {pd.Timestamp(cut).date()}.. {len(panel):,}r ({time.time() - t0:.0f}s)",
        flush=True,
    )

    # ── 2) 清洗: 只拆 dual 板 (生产 run_train 同构) ──
    cleaner = CleaningPipeline()
    _main_df, dual_df = cleaner.run_train(panel, board="dual")
    del panel, _main_df
    gc.collect()
    print(
        f"[clean:dual] {len(dual_df):,}r dates={dual_df['date'].nunique()} "
        f"symbols={dual_df['symbol'].nunique()} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    if len(dual_df) == 0:
        raise RuntimeError("dual 清洗后无样本 — 面板扩建未完成或数据缺失, 请检查 PANEL_V3_PATH")

    # ── 3) 特征 + 标签 (生产同构: FeatureEngineV35 全量 + LabelEngine) ──
    wait_for_ram("build")
    registry = FeatureRegistry(path=REGISTRY_PATH)
    features = FeatureEngineV35()
    df = prepare_board_frame(
        dual_df, features, None, cross_sectional_rank=True, registry=registry
    )
    del dual_df
    gc.collect()
    print(
        f"[feat] {len(df):,}r {len(df.columns)}c label_nan={df[LABEL].isna().mean():.2%} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    if LABEL not in df.columns:
        raise RuntimeError(f"面板缺 {LABEL} 标签列, 训练数据不可用")

    # ── 4) 实得 (compute_realized 口径: 买 T+1 收盘, 卖 T+11 收盘, -0.002, ffill) ──
    realized = compute_realized(
        df[["symbol", "date", "close_hfq"]],
        df["date"],
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
        raise RuntimeError("compute_realized 返回空 — 面板末尾无 T+11 未来数据")

    # ── 5) 特征列 = 生产 current bundle (gate_d 50 + 事件注入 8) ──
    with open(DUAL_BUNDLE, "rb") as fh:
        bundle = pickle.load(fh)
    feature_cols = list(bundle["feature_cols"])
    miss = [c for c in feature_cols if c not in df.columns]
    if miss:
        raise RuntimeError(
            f"特征 build 缺 {len(miss)} 列 (bundle 58 列应全可生成): {miss[:10]}"
        )
    print(f"[feats] {len(feature_cols)} 列 = {DUAL_BUNDLE}", flush=True)

    # ── 6) 最小帧 + 一次切分 (split_window 反锚四段, 全部组合共用) ──
    # 注意: 不做全局 dropna(label)! label 缺末 11 交易日 (需 T+11 未来) → 全局
    # dropna 会把窗口整体前移 11 天 (es/test 窗错位, 08-17 首跑教训: es 窗过平 →
    # 10d_reg 早停 1 树, 16 组合全雷同). 改逐段 dropna (生产 _train_one 同构):
    # train/es 段 dropna + risk_filter; test 段保留全部行 (末 60 交易日 = OOS,
    # 评估时 realized merge 自然剔除末 11 日无 realized 的行).
    keep = ["symbol", "date", "is_suspended", LABEL] + feature_cols
    frame = df[keep].copy()
    del df
    gc.collect()
    for c in feature_cols:  # 生产 _train_one: float32 下转
        frame[c] = frame[c].astype("float32", copy=False)
    segs = DualTrackTrainer.split_window(frame, WINDOW_TOTAL)
    del frame
    gc.collect()
    train = risk_filter(segs["train"].dropna(subset=[LABEL]))
    es = risk_filter(segs["es"].dropna(subset=[LABEL]))
    test = segs["test"]  # 不 dropna: realized merge 时自然剔除 (validate_oos 同构)
    del segs
    gc.collect()
    if len(test) < 500 or test["date"].nunique() < 30:
        raise RuntimeError(
            f"OOS test 段不足: rows={len(test)} days={test['date'].nunique()} "
            f"(需 >=500 行 / >=30 日)"
        )
    w = DualTrackTrainer.time_weights(train, HALF_LIFE)  # 半衰期 250, 只算一次
    print(
        f"[split] train={len(train):,}r/{train['date'].nunique()}d "
        f"es={es['date'].nunique()}d test={test['date'].nunique()}d "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    # 08-17 首跑教训守卫: 三段日数应 ≈ 728 (3y 窗口); 若 train 明显缺日说明
    # 有数据段被误删 (es/test 窗错位), 拒绝继续.
    total_days = train["date"].nunique() + es["date"].nunique() + test["date"].nunique()
    if total_days < 715:
        raise RuntimeError(
            f"切分日数异常: train+es+test={total_days}d (期望 ≈728d), "
            f"疑似标签预删导致窗口错位, 中止"
        )

    # ── 7) 网格串行训练 ──
    combos = [(ms, lam) for ms in GRID_MS for lam in GRID_LAMBDA]
    if args.max_n > 0:
        combos = combos[: args.max_n]
    results: dict[str, dict] = {}
    for i, (ms, lam) in enumerate(combos):
        key = combo_key(ms, lam)
        print(
            f"[{i + 1}/{len(combos)}] {key} seed={RANDOM_STATE} ...", flush=True
        )
        rec, n_trees = fit_and_eval(
            train, es, test, feature_cols, w, realized, ms, lam, RANDOM_STATE
        )
        results[key] = rec
        tn = rec["top10"]
        ws = ",".join(
            f"{s['topn_excess']:+.4f}" if np.isfinite(s["topn_excess"]) else "-"
            for s in rec["subwindows"]
        )
        print(
            f"    -> ic={rec['mean_ic']:.5f} icir={rec['icir']:.2f} "
            f"pos={rec['pos_day_ratio']:.2f} trees={n_trees} "
            f"top10_exc={tn['excess']:+.4f} (n={tn['n_days']}d) "
            f"sub_exc=[{ws}] ({time.time() - t0:.0f}s)",
            flush=True,
        )

    # ── 8) 扰动: 最优 1-2 组合 + ref, seed=43 重训 ──
    perturbation: dict[str, dict] = {}
    if not args.skip_perturb:
        ref_key = combo_key(20, 0.0)
        ranked = sorted(
            (k for k in results if not is_ref(*_parse_key(k))),
            key=lambda k: (
                -results[k]["top10"]["excess"],
                -results[k]["mean_ic"],
            ),
        )
        perturb_keys = list(dict.fromkeys([ref_key] + ranked[:2]))
        print(
            f"\n[perturb] seed={PERTURB_SEED} combos={perturb_keys}", flush=True
        )
        for k in perturb_keys:
            ms, lam = _parse_key(k)
            rec, _nt = fit_and_eval(
                train, es, test, feature_cols, w, realized, ms, lam, PERTURB_SEED
            )
            perturbation[k] = rec
            print(
                f"  [{k} seed={PERTURB_SEED}] ic={rec['mean_ic']:.5f} "
                f"top10_exc={rec['top10']['excess']:+.4f} ({time.time() - t0:.0f}s)",
                flush=True,
            )

    # ── 9) 判词: 稳定 > 最高; 双指标同时赢 + 子窗多数赢 + 扰动不翻转 → 有档 ──
    verdict = _verdict(results, perturbation, ref_key=combo_key(20, 0.0))

    # ── 10) WORM 输出 ──
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    out = {
        "meta": {
            "board": "dual",
            "kind": KIND,
            "label": LABEL,
            "grid": {
                "min_child_samples": list(GRID_MS),
                "reg_lambda": list(GRID_LAMBDA),
                "num_leaves": NUM_LEAVES,
                "seed": RANDOM_STATE,
                "perturb_seed": PERTURB_SEED,
            },
            "ref": {"min_child_samples": 20, "reg_lambda": 0.0},
            "oos_days": int(test["date"].nunique()),
            "n_train_rows": int(len(train)),
            "n_train_days": int(train["date"].nunique()),
            "n_features": len(feature_cols),
            "feature_source": DUAL_BUNDLE,
            "panel": str(PANEL_V3_PATH),
            "cost": COST,
            "buy_lag": BUY_LAG,
            "sell_lag": SELL_LAG,
            "top_n": TOP_N,
            "realized": "drift_monitor.compute_realized: buy=T+1 close, sell=T+11 close, "
            "-0.002, 停牌 ffill",
            "run_at": ts,
        },
        "results": results,
        "perturbation": perturbation,
        "verdict": verdict,
    }
    json_path = out_dir / f"legacy_dual_reg_sweep_{ts}.json"
    json_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    rows = []
    for k, rec in results.items():
        rows.append(_row(rec, k, is_ref=is_ref(*_parse_key(k))))
    for k, rec in perturbation.items():
        rows.append(_row(rec, k, perturb_of=k))
    df_csv = pd.DataFrame(rows)
    csv_path = out_dir / f"legacy_dual_reg_sweep_{ts}.csv"
    df_csv.to_csv(csv_path, index=False)
    print(f"\n[saved] {json_path}", flush=True)
    print(f"[saved] {csv_path} ({time.time() - t0:.0f}s)", flush=True)

    print("\n=== leaderboard (top10 excess) ===", flush=True)
    for k in sorted(results, key=lambda k: -results[k]["top10"]["excess"]):
        r = results[k]
        print(
            f"  top10exc={r['top10']['excess']:+.4f} ic={r['mean_ic']:.5f} "
            f"icir={r['icir']:.2f}  [{k}]",
            flush=True,
        )
    print(f"\n=== 判词 ===\n{verdict['text']}", flush=True)
    return 0


def _parse_key(key: str) -> tuple[int, float]:
    ms_s, lam_s = key.split("_")
    return int(ms_s[2:]), float(lam_s[3:])


def _row(rec: dict, key: str, is_ref: bool = False, perturb_of: str = "") -> dict:
    tn = rec["top10"]
    subs = {f"sub{i + 1}_excess": np.nan for i in range(3)}
    for i, s in enumerate(rec["subwindows"][:3]):
        subs[f"sub{i + 1}_excess"] = s.get("topn_excess", np.nan)
        subs[f"sub{i + 1}_n_days"] = s.get("n_days", 0)
    return {
        "combo": key,
        "min_child_samples": rec["params"]["min_child_samples"],
        "reg_lambda": rec["params"]["reg_lambda"],
        "seed": rec["seed"],
        "n_trees": rec["n_trees"],
        "mean_ic": rec["mean_ic"],
        "icir": rec["icir"],
        "pos_day_ratio": rec["pos_day_ratio"],
        "n_ic_days": rec["n_ic_days"],
        "top10_excess": tn["excess"],
        "top10_top_ret": tn["top_ret"],
        "top10_pool_ret": tn["pool_ret"],
        "top10_pos_day": tn["pos_day"],
        "top10_n_days": tn["n_days"],
        **subs,
        "is_ref": is_ref,
        "perturb_of": perturb_of,
    }


def _verdict(
    results: dict[str, dict],
    perturbation: dict[str, dict],
    ref_key: str,
) -> dict:
    """双指标 (top10 超额 AND Rank IC) 同时赢 + 子窗 >=2/3 赢 → 有档; 否则无档."""
    ref = results[ref_key]
    ref_exc, ref_ic = ref["top10"]["excess"], ref["mean_ic"]

    def _subwin_wins(rec: dict) -> int:
        return sum(
            1
            for i, s in enumerate(rec["subwindows"])
            if i < len(ref["subwindows"])
            and np.isfinite(s["topn_excess"])
            and np.isfinite(ref["subwindows"][i]["topn_excess"])
            and s["topn_excess"] > ref["subwindows"][i]["topn_excess"]
        )

    cands = []
    for k, rec in results.items():
        if k == ref_key:
            continue
        w = _subwin_wins(rec)
        if (
            rec["top10"]["excess"] > ref_exc
            and rec["mean_ic"] > ref_ic
            and w >= 2
        ):
            cands.append((k, rec, w))
    cands.sort(key=lambda t: (-t[1]["top10"]["excess"], -t[1]["mean_ic"]))

    if not cands:
        return {
            "has_tier": False,
            "verdict": "无档赢, 保持生产默认",
            "text": (
                f"无组合比基线 (ms=20, λ=0, top10_exc={ref_exc:+.4f}, "
                f"ic={ref_ic:.5f}) 在 top-10 超额 AND Rank IC 同时赢且子窗多数赢"
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
    # 扰动检查: seed 43 下 best 仍须赢 ref (top-10 超额不翻转)
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
                    "verdict": "无档赢, 保持生产默认 (扰动翻转)",
                    "text": (
                        f"{best_key} 在 seed=42 双指标+子窗赢, 但 seed=43 扰动下 "
                        f"top-10 超额 {b43['top10']['excess']:+.4f} <= ref "
                        f"{r43['top10']['excess']:+.4f} — 不稳健, 不推荐",
                    ),
                    "ref": ref_key,
                    **detail,
                }
            detail["perturb_flip"] = False
    return {
        "has_tier": True,
        "verdict": f"有档: 推荐 {best_key}",
        "text": (
            f"有档: 推荐 min_child_samples={best['params']['min_child_samples']}, "
            f"reg_lambda={best['params']['reg_lambda']} — top-10 超额 "
            f"{best['top10']['excess']:+.4f} vs ref {ref_exc:+.4f}, "
            f"Rank IC {best['mean_ic']:.5f} vs {ref_ic:.5f}, "
            f"子窗 {w}/3 赢, 扰动 seed=43 不翻转 (稳定 > 最高)"
        ),
        "ref": ref_key,
        **detail,
    }


if __name__ == "__main__":
    raise SystemExit(main())
