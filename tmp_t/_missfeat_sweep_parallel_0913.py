# -*- coding: utf-8 -*-
"""[0913 夜] PARALLEL 漏网/LEGACY已验特征双集清扫 (用户令: "CONTINUE EVALUATION
10 CLEAR FEATURES AND SWEEP MISSING FEATURES IN PARALLELS" + "10 FEATURES IS
WHAT WE CHECK IN LEGACY MODULE LIKE SL, 20 ETC" + "ANOTHER SET IS WHAT IS
MISSING YET MAYBE HV POSITIVE INFORMATION") — 夜链第6棒.

双集设计 (用户两条指令合并):
  集A = LEGACY 0912 已验 S-L 族 10 列 (dual force_include 全名单: SL差值/
        SL标准化/SL斜率20/SL多头持续天数/带20/带20斜率20/获利盘斜率20 +
        出货_density_5/10/20d)。已核实: **两板 prob feat_cols 均已含全部10列**
        (brute 自主收录), 池全无 → prob 侧=扣除臂 (现役贡献检验, #11 同构),
        mag 侧=加法臂。
  集B = 非pin OOS IC 审计漏网 8 列 (换手2/估值2/股息2/规模2; prob 仅缺
        turnover_rate/total_mv, 池全无) → mag 加法臂 + prob 补缺加法臂。

臂 (每板): mag 加法×8 (集A族2 + 集B族4 + 换手单列2) / prob 扣除×2 (集A) +
prob 补缺×2 (集B) = 12 臂 + base。
判词 = 整体服务面 (用户令 OVERALL 内建): 每臂 (mag 臂×prob_base / prob 臂×
mag_base) 三键 mag/prob/blend × 3 视界 rank_source.evaluate_keys +
choose_rank_key; 判官 = 所选键 TOP10 净均 Δvs base + 换键检测 + 各键 Δ 全记录.
main 服务键已切 prob (rank_source_20260913_002453) → main 上 prob 臂即交付面.

用法: python tmp_t/_missfeat_sweep_parallel_0913.py (自排队: 清场+RAM>=5.5GB)
沙箱 bundle: tmp_t/_missfeat_sandbox_0913; 哨兵已入 HEAVY_SENTINELS.
"""

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.stdout.reconfigure(encoding="utf-8")
_err = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_missfeat_sweep_parallel_0913.err"), "a", encoding="utf-8", buffering=1)
sys.stderr = _err
sys.excepthook = lambda t, v, tb: (print("".join(__import__("traceback").format_exception(t, v, tb)), file=_err, flush=True), os._exit(1))

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.pipeline1.ram_guard import check_startup_gate, start_monitor
from app.pipeline_parallel import prob_head
from app.pipeline_parallel.calibration import calibrate_mag10d
from app.pipeline_parallel.config import FUSION, SNIPER
from app.pipeline_parallel.rank_source import choose_rank_key, evaluate_keys
from app.pipeline_parallel.scoring import pool_score
from config.settings import (
    RETRAIN_RAM_GUARD_MIN_FREE_GB,
    RETRAIN_RAM_GUARD_POLL_S,
    data_others_path,
)

TAG = "missfeat_sweep_parallel_0913"
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_missfeat_sweep_parallel_0913.log")
SANDBOX = Path("tmp_t") / "_missfeat_sandbox_0913"
STAGE = {
    "main": Path("data") / "_diag_stage_main_3y.parquet",
    "dual": Path("data") / "_diag_stage_dual_3y.parquet",
}
TEST_DAYS = 60
HALF_LIVES = (15,)
NET_LABELS = ("label_pm_3d_net", "label_pm_5d_net", "label_pm_10d_net")
AUDIT_CSV = Path(r"D:\AMINQT\DATA OTHERS\diag\panel_nonpin_oosic_20260912.csv")

# 集A: LEGACY 已验 S-L 族 (prob 已含 → 扣除臂; 池全无 → mag 加法臂)
SL7 = ["SL差值", "SL标准化", "SL斜率20", "SL多头持续天数", "带20", "带20斜率20", "获利盘斜率20"]
DENS3 = ["出货_density_5d", "出货_density_10d", "出货_density_20d"]
# 集B: 审计漏网 (mag 加法臂; prob 仅补 turnover_rate/total_mv)
MAG_FAMILY_ARMS = {
    "+SL族7": SL7,
    "+出货密度3": DENS3,
    "+换手2": ["turnover_rate", "free_float_turnover_rate"],
    "+估值2": ["pb", "pe_ttm"],
    "+股息2": ["dv_ratio", "dv_ttm"],
    "+规模2": ["total_mv", "circ_mv"],
    "+换手turnover_rate": ["turnover_rate"],
    "+换手free_float": ["free_float_turnover_rate"],
}
PROB_ADD_ARMS = {
    "+turnover_rate": ["turnover_rate"],
    "+total_mv": ["total_mv"],
}
PROB_REMOVE_ARMS = {
    "−SL族7": SL7,
    "−出货密度3": DENS3,
}
ALL_CANDS = sorted({c for v in (MAG_FAMILY_ARMS, PROB_ADD_ARMS, PROB_REMOVE_ARMS) for c in sum(v.values(), [])})

log = logging.getLogger(TAG)


def _setup_logging():
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)


def _worm(name: str, payload: dict) -> str:
    rd = data_others_path("diag")
    rd.mkdir(parents=True, exist_ok=True)
    path = rd / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    log.info("[WORM] %s", path)
    return str(path)


def _audit_meta() -> dict:
    try:
        a = pd.read_csv(AUDIT_CSV).set_index("column")
        return {c: {"ic": float(a.loc[c, "ic_mean"]), "cov": float(a.loc[c, "coverage"])} for c in ALL_CANDS if c in a.index}
    except Exception:
        return {}


def load_stage_cands(board: str) -> tuple[pd.DataFrame, list[str]]:
    """load_stage 镜像 + 候选列; mfe_3d 读后算 (同产线口径)."""
    bundle = prob_head.load_latest_tier(board, 7)
    if bundle is None:
        raise RuntimeError(f"[{board}] 产线概率头 bundle 缺失 (hl7)")
    base_cols = list(bundle["feat_cols"])
    del bundle
    schema = set(pq.ParquetFile(STAGE[board]).schema_arrow.names)
    stale = [c for c in base_cols if c not in schema]
    if stale:
        log.warning("[load:%s] bundle feat_cols %d 列缺席 stage: %s", board, len(stale), stale)
        base_cols = [c for c in base_cols if c in schema]
    overhead = ["symbol", "date", "close_hfq", "high_hfq", "adv20", "label_pain", "label_pm_10d_net", *NET_LABELS]
    pool_cols = sorted(set(SNIPER.pool) | set(FUSION.pool))
    pool_missing = [c for c in pool_cols if c not in schema]
    if pool_missing:
        log.warning("[load:%s] 池列缺席 stage (read 剔除, pool_score 同口径跳过): %s", board, pool_missing)
        pool_cols = [c for c in pool_cols if c in schema]
    cands = [c for c in ALL_CANDS if c in schema and c not in base_cols]
    miss = [c for c in ALL_CANDS if c not in schema]
    if miss:
        log.warning("[load:%s] 候选列缺席 stage (相关臂自动降级): %s", board, miss)
    cols = sorted(set(base_cols) | set(overhead) | set(pool_cols) | set(cands))
    t0 = time.time()
    df = pd.read_parquet(STAGE[board], columns=cols)
    df["symbol"] = df["symbol"].astype(str)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df = prob_head._add_mfe_3d(df)
    log.info("[load:%s] %d 行, base %d 列, 候选补读 %d (%.0fs)", board, len(df), len(base_cols), len(cands), time.time() - t0)
    return df, base_cols


def _mag_scores(df: pd.DataFrame, board: str, extra: list[str]) -> pd.Series:
    sn = pool_score(df, tuple(SNIPER.pool) + tuple(extra))
    fu = pool_score(df, tuple(FUSION.pool) + tuple(extra))
    work = pd.DataFrame(
        {
            "symbol": df["symbol"],
            "date": df["date"],
            "board": board,
            "score": np.maximum(sn.to_numpy(), fu.to_numpy()),
            "label_pm_10d_net": df["label_pm_10d_net"],
        }
    ).dropna(subset=["score"])
    mag = calibrate_mag10d(work, target_col="label_pm_10d_net", label_horizon=10)
    m = mag.set_index(["symbol", "date"])["mag"]
    return pd.Series(
        m.reindex(pd.MultiIndex.from_frame(df[["symbol", "date"]])).to_numpy(),
        index=df.index,
    )


def _prob_scores(df: pd.DataFrame, board: str, cols: list[str], cut, trained_through: str, test_idx) -> np.ndarray:
    overhead = ["symbol", "date", "close_hfq", "high_hfq", "adv20", "mfe_3d", "label_pain", "label_pm_10d_net"]
    need = list(dict.fromkeys(cols + overhead))
    t_train = df.loc[df["date"] < cut, need].reset_index(drop=True)
    assert prob_head.feature_cols(t_train) == cols, f"列限制失效: {cols[-3:]}"
    bundles = [joblib.load(prob_head.train_bundle(board, t_train, trained_through, hl)) for hl in HALF_LIVES]
    del t_train
    gc.collect()
    t_test = df.loc[test_idx, need].reset_index(drop=True)
    prob = prob_head.ensemble_predict(bundles, t_test).to_numpy()
    del bundles, t_test
    gc.collect()
    return prob


def _eval_overall(test: pd.DataFrame, name: str) -> dict:
    ev = evaluate_keys(test[["symbol", "date", "mag", "prob"] + list(NET_LABELS)], eval_days=TEST_DAYS)
    chosen = choose_rank_key(ev)
    top10_mean = {}
    for key, per in ev.items():
        vals = [per["per_horizon"][h]["top10_net"] for h in ("3d", "5d", "10d")]
        vals = [v for v in vals if v is not None]
        top10_mean[key] = float(np.mean(vals)) if vals else None
    log.info("[overall:%s] chosen=%s | top10净均 mag=%s prob=%s blend=%s",
             name, chosen,
             *(f"{top10_mean.get(k):+.4f}" if top10_mean.get(k) is not None else "n/a" for k in ("mag", "prob", "blend")))
    return {"chosen": chosen, "weighted_ic": {k: ev[k]["weighted_ic"] for k in ev}, "top10_net_mean": top10_mean}


def _frame_from(df, test_idx, mag: pd.Series, prob: np.ndarray) -> pd.DataFrame:
    fr = df.loc[test_idx, ["symbol", "date"]].reset_index(drop=True)
    fr["mag"] = mag.loc[test_idx].to_numpy(dtype=float)
    fr["prob"] = prob
    for c in NET_LABELS:
        fr[c] = df.loc[test_idx, c].to_numpy()
    return fr


def run_board(board: str, audit_meta: dict) -> dict:
    df, base_cols = load_stage_cands(board)
    miss_labels = [c for c in NET_LABELS if c not in df.columns]
    if miss_labels:
        raise RuntimeError(f"[{board}] stage 缺净标签列: {miss_labels}")
    dates = np.sort(df["date"].unique())
    cut = dates[-TEST_DAYS]
    trained_through = str(pd.Timestamp(df.loc[df["date"] < cut, "date"].max()).date())
    test_idx = df["date"] >= cut
    rep: dict = {"board": board, "n_base": len(base_cols), "test_start": str(pd.Timestamp(cut).date()),
                 "cand_meta": {c: audit_meta.get(c) for c in ALL_CANDS if c in df.columns}, "arms": {}}

    t0 = time.time()
    mag_base = _mag_scores(df, board, [])
    log.info("[%s] mag_base 完成 (%.0fs)", board, time.time() - t0)
    SANDBOX.mkdir(parents=True, exist_ok=True)
    orig_bundle_dir = prob_head.bundle_dir
    prob_head.bundle_dir = lambda: SANDBOX
    try:
        t0 = time.time()
        prob_base = _prob_scores(df, board, base_cols, cut, trained_through, test_idx)
        log.info("[%s] prob_base 完成 (%.0fs)", board, time.time() - t0)
        rep["arms"]["base"] = _eval_overall(_frame_from(df, test_idx, mag_base, prob_base), "base")

        # ── prob 臂: 补缺加法 (集B缺席列) + 扣除 (集A现役列) — main 现役键=prob 即交付面 ──
        prob_arms: dict[str, list[str]] = {}
        for arm, extra in PROB_ADD_ARMS.items():
            if not any(c in df.columns for c in extra):
                log.warning("[%s] prob臂 %s 列缺席, 跳过", board, arm)
                continue
            if all(c in base_cols for c in extra):
                log.info("[%s] prob臂 %s 列已在 base, 跳过 (无行为差)", board, arm)
                continue
            prob_arms[arm] = base_cols + [c for c in extra if c in df.columns and c not in base_cols]
        for arm, drop in PROB_REMOVE_ARMS.items():
            dset = {c for c in drop if c in df.columns}
            if not dset & set(base_cols):
                log.info("[%s] prob臂 %s 列不在 base, 跳过", board, arm)
                continue
            prob_arms[arm] = [c for c in base_cols if c not in dset]
        for arm, cols in prob_arms.items():
            t0 = time.time()
            prob_arm = _prob_scores(df, board, cols, cut, trained_through, test_idx)
            log.info("[%s] prob臂 %-10s (n=%d) 训练+预测完成 (%.0fs)", board, arm, len(cols), time.time() - t0)
            rep["arms"][f"prob {arm}"] = _eval_overall(_frame_from(df, test_idx, mag_base, prob_arm), f"prob {arm}")
            del prob_arm
            gc.collect()

        # ── mag 臂: 池+族/单列 (集A+集B) ──
        for arm, extra in MAG_FAMILY_ARMS.items():
            ex = [c for c in extra if c in df.columns]
            if not ex:
                log.warning("[%s] mag臂 %s 列全缺席, 跳过", board, arm)
                continue
            t0 = time.time()
            mag_arm = _mag_scores(df, board, ex)
            log.info("[%s] mag臂 %s 完成 (%.0fs)", board, arm, time.time() - t0)
            rep["arms"][f"mag {arm}"] = _eval_overall(_frame_from(df, test_idx, mag_arm, prob_base), f"mag {arm}")
            del mag_arm
            gc.collect()
    finally:
        prob_head.bundle_dir = orig_bundle_dir

    base_chosen = rep["arms"]["base"]["chosen"]
    base_top = rep["arms"]["base"]["top10_net_mean"][base_chosen]
    rep["verdict_inputs"] = {
        "base_chosen": base_chosen,
        "base_chosen_top10": base_top,
        "per_arm": {},
    }
    for a, v in rep["arms"].items():
        if a == "base":
            continue
        ch, top = v["chosen"], v["top10_net_mean"][v["chosen"]]
        rep["verdict_inputs"]["per_arm"][a] = {
            "chosen": ch,
            "key_flipped": ch != base_chosen,
            "chosen_top10": top,
            "delta_vs_base_chosen_top10": (top - base_top) if (top is not None and base_top is not None) else None,
            "per_key_delta_top10": {
                k: (v["top10_net_mean"][k] - rep["arms"]["base"]["top10_net_mean"][k])
                for k in ("mag", "prob", "blend")
                if v["top10_net_mean"][k] is not None and rep["arms"]["base"]["top10_net_mean"][k] is not None
            },
        }
        log.info("[%s] %-18s chosen=%s%s Δtop10(chosen)=%+.4f | Δkeys mag=%+.4f prob=%+.4f blend=%+.4f",
                 board, a, ch, " (换键!)" if ch != base_chosen else "",
                 rep["verdict_inputs"]["per_arm"][a]["delta_vs_base_chosen_top10"] or 0.0,
                 *[(rep["verdict_inputs"]["per_arm"][a]["per_key_delta_top10"] or {}).get(k, float("nan")) for k in ("mag", "prob", "blend")])
    del df
    gc.collect()
    _worm(f"{TAG}_{board}", rep)
    return rep


def main() -> int:
    _setup_logging()
    import psutil

    from scripts._run_guard import find_conflicts

    deadline = time.time() + 6 * 3600
    while True:
        c = find_conflicts()
        free_gb = psutil.virtual_memory().available / 1024**3
        if not c and free_gb >= 5.5:
            break
        if time.time() >= deadline:
            log.error("[guard] 6h 等待超时, 放弃")
            return 3
        log.info("等待: 冲突=%s free=%.1fGB", [x["sentinel"] for x in c] or "无", free_gb)
        time.sleep(300)
    log.info("[guard] 清场确认 (free=%.1fGB)", free_gb)
    check_startup_gate(RETRAIN_RAM_GUARD_MIN_FREE_GB * 1024**3)
    start_monitor(RETRAIN_RAM_GUARD_MIN_FREE_GB * 1024**3, RETRAIN_RAM_GUARD_POLL_S)

    audit_meta = _audit_meta()
    for board in ("main", "dual"):
        run_board(board, audit_meta)
    log.info("[DONE] PARALLEL 双集特征清扫两板完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
