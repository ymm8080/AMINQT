# -*- coding: utf-8 -*-
"""[0913 #8] LEGACY cls 超参周末自适应再扫 — 自进化链头插步 (param_resweep)。

relay-7 (0913 夜) 三杠杆双板全败给 A0_prod → 静态接线零改动; 判死=方法条件性,
本 harness 把同臂池转成周期性自动重挑: 每周自进化链先跑本步, 赢家过准入闸落
param_source json (WORM), 紧随其后的 retrain 步 (knob auto) 即生效。
回退三层: LEGACY_PARAM_SOURCE[board]="code" > 删最新 json > 代码表。

臂 (判官 = cls 3/5/10d TOP10 实净均, relay-7 同帧同口径; 帧用生产 Layer1 最新
features_{board}_*.parquet, 不用夜扫缓存帧):
  A0_prod  代码表基线 (旋钮强制 "code", 隔离 json 泄漏进对照臂)
  J_json   现役 json (旋钮 "auto"; 仅当生产旋钮 auto 且 json 新鲜非空时入池,
           其存在检验现役 json 是否还挣得席位)
  C_hl60   cls 半衰期 250→60 (双板; relay-7 W_hl60 复挑)
  C_auc    main cls 早停 metric→AUC (main 独有; dual 代码表已是 auc)
  C_nl31   main cls num_leaves 15→31 出厂 (main 独有; dual 默认已 31)

准入 (闸准入协议 0909 变体; 误杀富集 N/A — 参数臂无删票面, 回退=json/旋钮):
  A: 测试窗 ≥ 40 交易日; B: 双半窗稳 (挑战者−A0 日 TOP10 净差 全窗/前后半 ≥0)。

着陆: 挑战者赢+过闸 → save; A0 赢+json 现役 → 空 overrides 退役; J 赢+过闸 →
刷新 trained_through 续 45d 新鲜度; J 赢不过闸 → 退役; 其余 → 零写入。
生产旋钮 "code" 时只判不写 (json 已被旋钮压制, 写入无意义)。
fail-soft: 本步不在 _CRITICAL, 崩溃不阻链。

用法: python scripts/_param_resweep_weekly.py (重活哨兵已注册 _run_guard)
"""

from __future__ import annotations

import gc
import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

TAG = "param_resweep_weekly"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "pipeline1")
WINDOW_TOTAL = 770
CLS_KINDS = ("3d_cls", "5d_cls", "10d_cls")
MIN_TEST_DAYS = 40

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger(TAG)

# 每板挑战臂: (臂名, {"nl": int, "es_auc": bool, "hl60": bool}); A0/J 由 run_board 注入
ARMS: dict[str, list[tuple[str, dict]]] = {
    "main": [
        ("C_hl60", {"hl60": True}),
        ("C_auc", {"es_auc": True}),
        ("C_nl31", {"nl": 31}),
    ],
    "dual": [
        ("C_hl60", {"hl60": True}),
    ],
}


# ── 纯判定函数 (单测对象) ──────────────────────────────────────────
def half_window_delta(chal_daily: pd.Series, a0_daily: pd.Series) -> dict:
    """挑战者−A0 日净差序列 → 全窗/前半/后半均值 (对齐交集日期)."""
    both = pd.concat([chal_daily, a0_daily], axis=1, join="inner").dropna()
    if len(both) < 2:
        return {"delta_full": float("nan"), "delta_h1": float("nan"), "delta_h2": float("nan"), "n_days": int(len(both))}
    delta = both.iloc[:, 0] - both.iloc[:, 1]
    half = len(delta) // 2
    return {
        "delta_full": float(delta.mean()),
        "delta_h1": float(delta.iloc[:half].mean()),
        "delta_h2": float(delta.iloc[half:].mean()),
        "n_days": int(len(delta)),
    }


def admission_verdict(n_days: int, delta: dict, min_days: int = MIN_TEST_DAYS) -> dict:
    """准入闸: A 样本量 + B 双半窗稳 (全窗/前后半 ≥0, 严格)."""
    gates = {
        "A_days": n_days >= min_days,
        "B_full": bool(np.isfinite(delta.get("delta_full", np.nan)) and delta["delta_full"] >= 0),
        "B_h1": bool(np.isfinite(delta.get("delta_h1", np.nan)) and delta["delta_h1"] >= 0),
        "B_h2": bool(np.isfinite(delta.get("delta_h2", np.nan)) and delta["delta_h2"] >= 0),
    }
    return {"pass": all(gates.values()), "gates": gates, "delta": delta, "n_days": n_days}


def landing_action(arm_net: dict[str, float], admissions: dict[str, dict], json_active: bool) -> dict:
    """着陆判定 (纯函数). arm_net = {臂名: top10_net_mean}."""
    winner = max(arm_net, key=lambda a: arm_net[a])
    if winner == "A0_prod":
        return {"action": "retire" if json_active else "none", "arm": winner}
    if winner == "J_json":
        ok = bool(admissions.get("J_json", {}).get("pass"))
        return {"action": "refresh" if ok else "retire", "arm": winner}
    if admissions.get(winner, {}).get("pass"):
        return {"action": "save", "arm": winner}
    return {"action": "none", "arm": winner, "reason": "admission_fail"}


def build_payload(board: str, arm: str, arm_cfg: dict, trained_through: str, evidence: dict) -> dict:
    """挑战臂 → param_source json payload (半衰期独立顶层字段, 勿混 overrides)."""
    overrides: dict[str, dict] = {}
    if arm_cfg.get("es_auc"):
        overrides = {k: {"metric": "auc"} for k in CLS_KINDS}
    elif arm_cfg.get("nl"):
        overrides = {k: {"num_leaves": int(arm_cfg["nl"])} for k in CLS_KINDS}
    payload = {
        "board": board,
        "trained_through": trained_through,
        "overrides": overrides,
        "evidence": evidence,
    }
    if arm_cfg.get("hl60"):
        payload["cls_half_life_days"] = 60
    return payload


# ── 判官 (relay-7 同口径 + 日序列供半窗) ──────────────────────────
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
    return {
        "ic_mean": float(np.nanmean(arr)),
        "ic_win": float(np.nanmean(arr > 0)),
        "top10_real_net": float(top10["y"].mean()) if len(top10) else None,
        "top10_win": float((top10["y"] > 0).mean()) if len(top10) else None,
        "top10_daily": top10.groupby("date")["y"].mean(),
    }


def _align_X(test: pd.DataFrame, cols: list[str]) -> np.ndarray:
    parts = [test[c].to_numpy(dtype=np.float32) if c in test.columns else np.zeros(len(test), dtype=np.float32) for c in cols]
    return np.nan_to_num(np.column_stack(parts), nan=0.0, posinf=0.0, neginf=0.0)


# ── 单板扫描 ──────────────────────────────────────────────────────
def run_board(board: str) -> dict:
    import config.settings as cfg
    import app.pipeline1.dual_track_trainer as tr
    from app.pipeline1 import param_source
    from app.pipeline1.dual_track_trainer import DualTrackTrainer
    from scripts.train_predict_main import find_latest_features, load_features_for_training

    features_path = find_latest_features(board)
    with open(os.path.join(MODEL_DIR, f"{board}_current.pkl"), "rb") as fh:
        bundle = pickle.load(fh)
    pin_cols = list(bundle["feature_cols"])
    df = load_features_for_training(board, features_path, set(pin_cols))
    log.info("[%s] Layer1 帧=%s pin=%d 列 → %d 行", board, os.path.basename(features_path), len(pin_cols), len(df))

    prod_extras = list((cfg.LEGACY_HEAD_EXTRA_COLS.get(board) or {}).get("cls") or [])
    if "quality_factor" in prod_extras and "quality_factor" not in df.columns:
        from app.pipeline1.feature_selector import add_quality_factor

        add_quality_factor(df)
    # extras 帧内可用性过滤 (relay-8 02:43 事故同型防线: 合成不粘 → 训练/对齐列数错位)
    extras_avail = [c for c in prod_extras if c in df.columns]
    missing = [c for c in prod_extras if c not in df.columns]
    if missing:
        log.warning("[%s] extras %d 列帧内缺失剔除: %s", board, len(missing), missing)
    cols = [c for c in sorted(set(pin_cols)) if c in df.columns and float(df[c].isna().mean()) < 0.95]
    cols += [c for c in extras_avail if c not in cols]

    # J_json 入池判定: 生产旋钮 auto 且 json 对 cls 族有非空效果
    knob_now = dict(cfg.LEGACY_PARAM_SOURCE).get(board, "auto")
    json_active = knob_now == "auto" and (
        any(param_source.resolve_param_override(board, k, knob="auto") for k in CLS_KINDS)
        or param_source.resolve_cls_half_life(board, knob="auto") is not None
    )
    arms = ([("A0_prod", {})] + ([("J_json", {})] if json_active else []) + ARMS[board])

    trainer = DualTrackTrainer()
    segs = DualTrackTrainer.split_window(df, WINDOW_TOTAL)
    trained_through = str(df["date"].max())[:10]
    del df
    gc.collect()
    test = segs["test"]
    net_label = {k: DualTrackTrainer._resolve_label(f"{k}d_reg", segs["train"].columns) for k in (3, 5, 10)}
    rep: dict = {
        "board": board,
        "features": os.path.basename(features_path),
        "trained_through": trained_through,
        "n_cols": len(cols),
        "extras_avail": extras_avail,
        "knob_now": knob_now,
        "json_active": json_active,
        "arms_spec": [a for a, _ in arms],
        "net_labels": net_label,
        "test_days": int(test["date"].nunique()),
        "arms": {},
    }

    orig_mp = tr.model_params
    orig_override = dict(tr.NUM_LEAVES_OVERRIDE)
    orig_source = dict(cfg.LEGACY_PARAM_SOURCE)
    orig_tw = DualTrackTrainer.time_weights
    try:
        for arm, arm_cfg in arms:
            # 旋钮隔离: A0/C_* 强制 code; J 用 auto (现役 json)
            cfg.LEGACY_PARAM_SOURCE = {**orig_source, board: ("auto" if arm == "J_json" else "code")}
            if arm_cfg.get("es_auc"):
                def _mp(b, k, _o=orig_mp):
                    p = _o(b, k)
                    if k.endswith("cls"):
                        p["metric"] = "auc"
                    return p
                tr.model_params = _mp
            if arm_cfg.get("nl"):
                for kind in CLS_KINDS:
                    tr.NUM_LEAVES_OVERRIDE[(board, kind)] = arm_cfg["nl"]
            if arm_cfg.get("hl60"):
                trainer.time_weights = lambda df_, _f=orig_tw: _f(df_, half_life=60)
            try:
                t_arm = time.time()
                armrep: dict = {}
                for kind in CLS_KINDS:
                    t0 = time.time()
                    model, label = trainer._train_one(kind, segs, cols, board)
                    pred = model.predict_proba(_align_X(test, cols))[:, 1]
                    armrep[kind] = head_metrics(test, pred, label, top10_label=net_label[int(kind.split("_")[0][:-1])])
                    m = armrep[kind]
                    log.info("[%s:%s] %-7s IC=%.4f top10(净)=%+.4f win=%.1f%% (%.0fs)",
                             board, arm, kind, m["ic_mean"],
                             m["top10_real_net"] if m["top10_real_net"] is not None else np.nan,
                             (m["top10_win"] or 0) * 100, time.time() - t0)
                    del model, pred
                    gc.collect()
                armrep["top10_net_mean"] = float(np.mean([armrep[k]["top10_real_net"] for k in CLS_KINDS]))
                daily = pd.concat([armrep[k]["top10_daily"] for k in CLS_KINDS], axis=1).mean(axis=1)
                armrep["top10_daily_agg"] = {str(d.date()): float(v) for d, v in daily.items()}
                rep["arms"][arm] = armrep
                log.info("[%s:%s] TOP10 净均 = %+.4f (%.0fs)", board, arm, armrep["top10_net_mean"], time.time() - t_arm)
                rep["_daily_" + arm] = daily
            finally:
                tr.model_params = orig_mp
                tr.NUM_LEAVES_OVERRIDE = dict(orig_override)
                try:
                    del trainer.time_weights
                except AttributeError:
                    pass
    finally:
        tr.model_params = orig_mp
        tr.NUM_LEAVES_OVERRIDE = orig_override
        cfg.LEGACY_PARAM_SOURCE = orig_source

    # 准入 + 着陆
    a0_daily = rep.pop("_daily_A0_prod")
    arm_net = {a: rep["arms"][a]["top10_net_mean"] for a in rep["arms"]}
    admissions = {}
    for a in rep["arms"]:
        if a == "A0_prod":
            continue
        delta = half_window_delta(rep.pop("_daily_" + a), a0_daily)
        admissions[a] = admission_verdict(rep["test_days"], delta)
    landing = landing_action(arm_net, admissions, json_active)
    rep["verdict"] = {
        "arm_net": arm_net,
        "admissions": admissions,
        "landing": landing,
    }
    log.info("[%s] winner=%s action=%s | Δvs A0: %s", board, landing["arm"], landing["action"],
             {a: f"{v - arm_net['A0_prod']:+.4f}" for a, v in arm_net.items() if a != "A0_prod"})

    # 着陆写入 (旋钮 code 时只判不写)
    if landing["action"] == "none" or knob_now != "auto":
        if landing["action"] != "none":
            log.info("[%s] 旋钮=%s, 只判不写 json", board, knob_now)
        return rep
    evidence = {
        "sweep": TAG,
        "judge": "cls 3/5/10d TOP10 实净均 + 双半窗准入",
        "arm_net": arm_net,
        "admissions": {a: {"pass": v["pass"], "delta": v["delta"]} for a, v in admissions.items()},
        "features": rep["features"],
    }
    if landing["action"] == "save":
        arm_cfg = next(c for a, c in arms if a == landing["arm"])
        payload = build_payload(board, landing["arm"], arm_cfg, trained_through, evidence)
    elif landing["action"] == "refresh":
        rec = param_source.load_latest_param_source(board) or {}
        payload = {**rec, "board": board, "trained_through": trained_through, "evidence": evidence}
    else:  # retire
        payload = {"board": board, "trained_through": trained_through, "overrides": {},
                   "evidence": {**evidence, "action": "retire: json 输给 A0/J-不稳, 空 overrides 退役"}}
    path = param_source.save_param_source(board, payload)
    log.info("[%s] %s → %s", board, landing["action"], path)
    rep["verdict"]["written"] = str(path)
    return rep


def _worm(payload: dict) -> str:
    from config.settings import data_others_path

    rd = data_others_path("diag")
    rd.mkdir(parents=True, exist_ok=True)
    path = rd / f"{TAG}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    log.info("[WORM] %s", path)
    return str(path)


def main() -> int:
    from scripts._run_guard import find_conflicts

    conflicts = find_conflicts()
    if conflicts:
        for c in conflicts:
            log.error("[guard] 冲突: %s (PID %s)", c["sentinel"], c["pid"])
        return 3

    rep: dict = {"tag": TAG, "boards": {}}
    for board in ("main", "dual"):
        rep["boards"][board] = run_board(board)
        gc.collect()
    _worm(rep)
    log.info("[DONE] 周末超参再扫完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
