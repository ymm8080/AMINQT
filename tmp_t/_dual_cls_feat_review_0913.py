# -*- coding: utf-8 -*-
"""[0913 晨] LEGACY dual 特征集 cls 对齐复审 — main 版 (_main_cls_feat_review_0913)
的 dual 姊妹棒。人工晨间启动 (夜里不跑, 等 relay-8 main 判词后再手动起)。

背景 (dual 考古错位): dual pin (selected_dual_pinned.json, 203 列) 是 2026-08-17
gate_d **reg 判官**时代冻结 (ab61ac7b), 而 dual 服务头 0912 (2608c697) 才从 reg
残差切到 cls proba — 组成证据与服务头错位 3.5 周。feature_selector.py:1404 自注
"dual cls 头部增益部分是低基数效应, 重新上岗时须复核" → 本复审兑现该自注: 只回答
一个问题: **reg 时代选的组成在 cls 头下是不是拖累**。

设计 (同尺寸隔离组成效应, TARGET_N=200 同 main):
  A0_prod  三层组成 = pin 列 + gate_d force_include 运行时层 (出货×3 + SL族×7)
           + LEGACY_HEAD_EXTRA_COLS["dual"]["cls"] extras (= 交付头现状)
  T{n_t}   按 cls gain 重要度 (bundle 现役 3 cls 头归一化增益求均) 排 top-200
           + extras — "cls 自己想要的 200"
  D{n_drop} 现 pin 剔 cls gain 垫底 + extras (≈同尺寸) — "只扔 reg 遗产"
  判官 = OOS TOP10 实净 (3d/5d/10d cls 均值, relay-7 同帧同口径)。
低基数校验 (dual 特有): base top10 实净仅 ~0.54%, 判词噪声大 → 每臂每视界记
n_days 与逐日 top10 净值标准误 (std/sqrt(n_days)); Δ < 2×SE 视为平局维持 pin。

判词走向: T/D 显著赢 → 季度重扫协议 (11-14) 的判官从 reg 口径切 cls TOP10
并提前跑 dual; 平/输 → pin 对 cls 稳健, 维持到季度重扫。

用法: python tmp_t/_dual_cls_feat_review_0913.py (冲突即退 rc=3, 等待器重试;
帧缓存不存在即退 rc=4, 提示先跑 tmp_t/_dualreg_recon_0912.py)
帧缓存: _dualreg_recon_frame_0912.parquet (relay-7 同源)。
"""

import gc
import json
import logging
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.stdout.reconfigure(encoding="utf-8")
_err = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dual_cls_feat_review_0913.err"), "a", encoding="utf-8", buffering=1)
sys.stderr = _err
sys.excepthook = lambda t, v, tb: (print("".join(__import__("traceback").format_exception(t, v, tb)), file=_err, flush=True), os._exit(1))

import numpy as np
import pandas as pd

TAG = "dual_cls_feat_review_0913"
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dual_cls_feat_review_0913.log")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "pipeline1")
PIN_FILE = "selected_dual_pinned.json"  # 0817 gate_d reg 判官冻结 (203 列), pin 与 bundle 冲突以 pin 为准
WINDOW_TOTAL = 770
CLS_KINDS = ("3d_cls", "5d_cls", "10d_cls")
TARGET_N = 200  # feature-count-scan-verdict 峰值尺寸 (dual 峰值同 main)

log = logging.getLogger(TAG)


def _setup_logging():
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)


def _worm(name: str, payload: dict) -> str:
    from config.settings import data_others_path

    rd = data_others_path("diag")
    rd.mkdir(parents=True, exist_ok=True)
    path = rd / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    log.info("[WORM] %s", path)
    return str(path)


def _daily_spearman(pred: np.ndarray, y: np.ndarray, dates: np.ndarray) -> list[float]:
    from scipy.stats import spearmanr

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


def head_metrics(test: pd.DataFrame, pred: np.ndarray, label: str, top10_label: str) -> dict:
    y = test[label].to_numpy(dtype=float)
    ics = _daily_spearman(pred, y, test["date"].to_numpy())
    arr = np.array(ics) if ics else np.array([np.nan])
    d = pd.DataFrame({"date": test["date"].to_numpy(), "y": test[top10_label].to_numpy(dtype=float), "p": pred}).dropna(subset=["y"])
    d["rk_p"] = d.groupby("date")["p"].rank(ascending=False, method="first")
    top10 = d[d["rk_p"] <= 10]
    daily_net = top10.groupby("date")["y"].mean()  # 逐日 top10 净值 → 低基数 SE
    n_top_days = int(daily_net.shape[0])
    net_se = float(daily_net.std(ddof=1) / np.sqrt(n_top_days)) if n_top_days > 1 else None
    return {
        "ic_mean": float(np.nanmean(arr)),
        "ic_win": float(np.nanmean(arr > 0)),
        "n_days": len(ics),
        "top10_label": top10_label,
        "top10_real_net": float(top10["y"].mean()) if len(top10) else None,
        "top10_win": float((top10["y"] > 0).mean()) if len(top10) else None,
        "top10_n_days": n_top_days,
        "top10_net_se": net_se,
    }


def _align_X(test: pd.DataFrame, cols: list[str]) -> np.ndarray:
    parts = [test[c].to_numpy(dtype=np.float32) if c in test.columns else np.zeros(len(test), dtype=np.float32) for c in cols]
    return np.nan_to_num(np.column_stack(parts), nan=0.0, posinf=0.0, neginf=0.0)


def _cls_gain_rank(bundle: dict, pool: list[str]) -> pd.Series:
    """现役 3 cls 头 gain 重要度 → 归一化求均 → 排秩 (构造臂启发式)."""
    cols = list(bundle["feature_cols"])
    agg = pd.Series(0.0, index=cols)
    n_used = 0
    for name, (model, _label) in bundle["models"].items():
        if not str(name).endswith("cls"):
            continue
        imp = getattr(model, "feature_importances_", None)
        if imp is None or len(imp) != len(cols):
            continue
        s = pd.Series(imp, index=cols, dtype=float)
        tot = s.sum()
        if tot <= 0:
            continue
        agg += s / tot
        n_used += 1
    if n_used == 0:
        raise RuntimeError("bundle 内无可用 cls feature_importances_")
    agg = agg / n_used
    absent = [c for c in pool if c not in agg.index]
    if absent:
        # dual 特有风险: pin 来源 ≠ bundle feature_cols, 不在 bundle 的池列无法排秩
        log.warning("[gain] %d 池列不在 bundle feature_cols (T/D 臂剔除): %s", len(absent), absent)
    present = agg.reindex([c for c in pool if c in agg.index]).dropna()
    log.info("[gain] %d cls 头聚合, 排秩池 %d 列 (top5: %s)", n_used, len(present), list(present.sort_values(ascending=False).head(5).index))
    return present.sort_values(ascending=False)


def main() -> int:
    _setup_logging()
    from scripts._run_guard import find_conflicts

    conflicts = find_conflicts()
    if conflicts:
        for c in conflicts:
            log.error("[guard] 冲突: %s (PID %s)", c["sentinel"], c["pid"])
        return 3

    import config.settings as cfg
    from app.pipeline1.dual_track_trainer import DualTrackTrainer
    from app.pipeline1.feature_selector import FeatureSelector
    from config.settings import data_others_path

    _dir = os.path.dirname(os.path.abspath(__file__))
    frame_cache = os.path.join(_dir, "_dualreg_recon_frame_0912.parquet")
    if not os.path.exists(frame_cache):
        log.error("帧缓存不存在: %s — 先跑 tmp_t/_dualreg_recon_0912.py 生成后再来", frame_cache)
        return 4
    board = "dual"

    # pin 来源 = 注册中心 JSON (非 bundle["feature_cols"]); bundle/pin 冲突以 pin 为准
    pin_path = data_others_path("factor_registry") / PIN_FILE
    with open(pin_path, encoding="utf-8") as fh:
        pin_payload = json.load(fh)
    pin_cols = list(pin_payload["features"]) if isinstance(pin_payload, dict) else list(pin_payload)
    log.info("[pin] %s: %d 列 (registry 声明 selected_count=%s)", pin_path, len(pin_cols), pin_payload.get("selected_count") if isinstance(pin_payload, dict) else "?")

    # A0 三层组成 (列裁剪前算好 — force_include/extras 必须活在 keep 里, 否则被静默裁掉)
    fi_cols = list(FeatureSelector.DEFAULT_CONFIG["dual"]["gate_d"].get("force_include") or [])
    prod_extras = list((cfg.LEGACY_HEAD_EXTRA_COLS.get(board) or {}).get("cls") or [])

    df = pd.read_parquet(frame_cache)
    with open(os.path.join(MODEL_DIR, "dual_current.pkl"), "rb") as fh:
        bundle = pickle.load(fh)
    labels = {v[1] for v in bundle["models"].values()}
    keep = {"symbol", "date", "is_suspended", "close_hfq"} | labels | set(pin_cols)
    keep |= set(fi_cols) | set(prod_extras)
    keep |= {c for c in df.columns if c.startswith("label_")}
    keep &= set(df.columns)
    df = df[list(keep)]
    gc.collect()

    # A0_prod 三层组成: pin + gate_d force_include 运行时层 + per-head extras
    if "quality_factor" in prod_extras and "quality_factor" not in df.columns:
        from app.pipeline1.feature_selector import add_quality_factor

        add_quality_factor(df)
    for layer, lc in (("pin", pin_cols), ("force_include", fi_cols), ("extras", prod_extras)):
        missing = [c for c in lc if c not in df.columns]
        if missing:
            log.warning("[A0:%s] 帧缓存缺 %d 列: %s", layer, len(missing), missing)
    base = sorted(set(pin_cols) | set(fi_cols))
    a0_cols = [c for c in base if c in df.columns and float(df[c].isna().mean()) < 0.95]
    a0_cols += [c for c in prod_extras if c in df.columns and c not in a0_cols]

    # 臂构造: 排秩池 = a0 去掉 extras (extras 是 cls 判词资产恒留)
    pool = [c for c in a0_cols if c not in prod_extras]
    rank = _cls_gain_rank(bundle, pool)
    n_t = min(TARGET_N, len(rank))
    t200 = list(rank.head(n_t).index) + prod_extras
    d160 = list(rank.index) + prod_extras  # rank 已含全池 → 等价 a0; 剔垫底
    n_drop = min(len(rank) - n_t, 160) if len(rank) > n_t else 0
    if n_drop > 0:
        d160 = list(rank.head(len(rank) - n_drop).index) + prod_extras
    else:
        d160 = a0_cols  # 池不足 200 → D 退化 A0, 大声标注
        log.warning("[D] 排秩池 %d ≤ %d, 退化 A0", len(rank), TARGET_N)

    arms = {"A0_prod": a0_cols, f"T{n_t}": t200, f"D{n_drop}": d160}
    rep: dict = {"tag": TAG, "board": board, "pin_n": len(pin_cols), "force_include_n": len(fi_cols), "pool_n": len(rank),
                 "prod_extras": prod_extras, "arms_spec": {a: len(c) for a, c in arms.items()},
                 "note": "同尺寸组成对齐: T=cls gain top-N; D=剔垫底; 判官 cls TOP10 实净; base 低基数, Δ<2×SE=平局"}

    trainer = DualTrackTrainer()
    segs = DualTrackTrainer.split_window(df, WINDOW_TOTAL)
    del df
    gc.collect()
    test = segs["test"]
    net_label = {k: DualTrackTrainer._resolve_label(f"{k}d_reg", segs["train"].columns) for k in (3, 5, 10)}
    rep["net_labels"] = net_label
    rep["test_days"] = int(test["date"].nunique())

    rep["arms"] = {}
    for arm, cols in arms.items():
        armrep: dict = {}
        for kind in CLS_KINDS:
            t0 = time.time()
            model, label = trainer._train_one(kind, segs, cols, board)
            pred = model.predict_proba(_align_X(test, cols))[:, 1]
            armrep[kind] = head_metrics(test, pred, label, top10_label=net_label[int(kind.split("_")[0][:-1])])
            m = armrep[kind]
            log.info("[%s:%s] %-7s IC=%.4f top10(净)=%+.4f win=%.1f%% SE=%.4f n=%dd (%.0fs)",
                     board, arm, kind, m["ic_mean"],
                     m["top10_real_net"] if m["top10_real_net"] is not None else np.nan,
                     (m["top10_win"] or 0) * 100,
                     m["top10_net_se"] if m["top10_net_se"] is not None else np.nan,
                     m["top10_n_days"], time.time() - t0)
            del model, pred
            gc.collect()
        armrep["top10_net_mean"] = float(np.mean([armrep[k]["top10_real_net"] for k in CLS_KINDS]))
        log.info("[%s:%s] TOP10 净均 = %+.4f", board, arm, armrep["top10_net_mean"])
        rep["arms"][arm] = armrep

    a0 = rep["arms"]["A0_prod"]["top10_net_mean"]
    rep["verdict_inputs"] = {
        "a0_top10_net_mean": a0,
        "delta_vs_a0": {a: rep["arms"][a]["top10_net_mean"] - a0 for a in rep["arms"] if a != "A0_prod"},
        "low_base": {a: {k: {"n_days": rep["arms"][a][k]["top10_n_days"],
                             "net_se": rep["arms"][a][k]["top10_net_se"]} for k in CLS_KINDS}
                     for a in rep["arms"]},
        "wiring": "T/D 显著赢→季度重扫(11-14)判官切cls TOP10并提前跑dual; Δ < 2×SE 视为平局维持 pin; 平/输→pin稳健维持",
    }
    log.info("[verdict] Δvs A0: %s", {a: f"{v:+.4f}" for a, v in rep["verdict_inputs"]["delta_vs_a0"].items()})
    _worm(TAG, rep)
    log.info("[DONE] dual 特征集 cls 对齐复审完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
