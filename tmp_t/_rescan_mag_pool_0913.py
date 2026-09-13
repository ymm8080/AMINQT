# -*- coding: utf-8 -*-
"""[0913] PARALLEL mag 池重扫 (SNIPER/FUSION) — 用户令 "MAKE PARALLEL RESWEEP
START AFTER 8 HOURS"。设计 = parallel-pool-rescan-design-0913 记忆 (agent 定案):

池 = 0804 WORM 手定 (config.py:157-203), 5 周陈旧 (c2c 切换/down_gap 剔除/宇宙
3000→4960/8格 extras 后从未重扫) → 最大 PARALLEL 杠杆。

两段贪心 (每板):
  L0 = 候选单列加法 (audit csv 筛 |ic|≥0.02 & cov≥0.9 & 非原始OHLCV, ≤15) +
       现役 7 池列 LOO 剔除
  L1 = L0 加法胜者 (Δ>0 且双半窗稳) top-3 两两/三组合 + 族包 (换手2/规模2/筹码4)
判官 = 整体服务面 (missfeat sweep 同款): (mag_臂×prob_base) 三键×3视界
  evaluate_keys + choose_rank_key; 所选键 TOP10 净均 Δvs base + 换键 +
  双半窗 (前/后 30d 各评, 两半 Δ>0 才算稳) + 误杀富集 (掉出TOP10 中
  5d净≥10% 占比 vs base TOP10 占比, >1.5x 红旗)。

调度: 常量 START_AT 前只睡不抢 (0913 09:15, 用户令 8h 后); 之后自排队
(清场+RAM>=5.5GB)。state: SCHEDULED→GUARD→RUNNING→DONE/FAIL_RC。
用法: python tmp_t/_rescan_mag_pool_0913.py; 哨兵已入 HEAVY_SENTINELS。
"""

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
# _ab_families_0912 import 时劫持 stderr → 先 import 再建自己的重定向覆盖 (replay 同序)
from _ab_families_0912 import FAMILIES

_err = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_rescan_mag_pool_0913.err"), "a", encoding="utf-8", buffering=1)
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

TAG = "rescan_mag_pool_0913"
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_rescan_mag_pool_0913.log")
STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_rescan_mag_pool_0913.state.json")
SANDBOX = Path(ROOT) / "tmp_t" / "_rescan_pool_sandbox_0913"  # ROOT 锚定: schtasks 发射 cwd 不定
STAGE = {
    "main": Path(ROOT) / "data" / "_diag_stage_main_3y.parquet",
    "dual": Path(ROOT) / "data" / "_diag_stage_dual_3y.parquet",
}
TEST_DAYS = 60
HALF_LIVES = (15,)
NET_LABELS = ("label_pm_3d_net", "label_pm_5d_net", "label_pm_10d_net")
AUDIT_CSV = Path(r"D:\AMINQT\DATA OTHERS\diag\panel_nonpin_oosic_20260912.csv")
START_AT = "2026-09-13 05:00:00"  # 0913 晨用户令提前手动发射 (09:15 schtasks 已停用防双跑)

MIN_ABS_IC = 0.02
MIN_COV = 0.90
MAX_CANDS = 15
RAW_DENY = {
    "amount", "volume", "close", "open", "high", "low", "pre_close",
    "close_hfq", "open_hfq", "high_hfq", "low_hfq",
    "up_limit_raw", "down_limit_raw", "pctChg", "is_suspended",
}
FAMILY_PACKS = {
    "+换手2": ["turnover_rate", "free_float_turnover_rate"],
    "+规模2": ["total_mv", "circ_mv"],
    "+筹码4": list(FAMILIES["筹码chip4"]),
}
ENRICH_HIT = 0.10  # 5d 净 ≥10% 记"误杀"
ENRICH_MAX = 1.5

log = logging.getLogger(TAG)


def _setup_logging():
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    for h in (logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)


def _state(phase: str, **kw) -> None:
    payload = {"phase": phase, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}
    with open(STATE_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


# 崩溃也落终态 (对抗性审查): 模块顶 excepthook 直接 os._exit 绕过 __main__ 的 state 写 → RUNNING 悬挂
sys.excepthook = lambda t, v, tb: (
    _state("FAIL_CRASH"),
    print("".join(__import__("traceback").format_exception(t, v, tb)), file=_err, flush=True),
    os._exit(1),
)


def _worm(name: str, payload: dict) -> str:
    rd = data_others_path("diag")
    rd.mkdir(parents=True, exist_ok=True)
    path = rd / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    log.info("[WORM] %s", path)
    return str(path)


def pick_candidates() -> tuple[list[str], dict]:
    """audit csv → 候选单列 (|ic|/cov 闸 + 剔原始 OHLCV), 按 |ic| 降序 ≤15."""
    a = pd.read_csv(AUDIT_CSV)
    a = a[a["column"].notna()]
    m = (a["abs_ic"] >= MIN_ABS_IC) & (a["coverage"] >= MIN_COV) & (~a["column"].isin(RAW_DENY))
    a = a[m].sort_values("abs_ic", ascending=False).head(MAX_CANDS)
    meta = {r["column"]: {"ic": float(r["ic_mean"]), "cov": float(r["coverage"])} for _, r in a.iterrows()}
    log.info("[cand] audit 筛出 %d 候选: %s", len(meta), list(meta))
    return list(meta), meta


def load_stage_cands(board: str, cands: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """load_stage 镜像 + 池列/候选列; mfe_3d 读后算 (missfeat 同款)."""
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
    pool_missing = [c for c in pool_cols if c not in schema]  # pv_corr_5 缺席 stage: 入池(2fbd4374)早于检查点转储
    if pool_missing:
        log.warning("[load:%s] 池列缺席 stage (read 剔除, pool_score 同口径跳过): %s", board, pool_missing)
        pool_cols = [c for c in pool_cols if c in schema]
    cands = [c for c in cands if c in schema]
    miss = [c for c in FAMILY_PACKS_COLS if c not in schema and c not in base_cols]
    if miss:
        log.warning("[load:%s] 族包列缺席 stage (相关臂降级): %s", board, miss)
    cols = sorted(set(base_cols) | set(overhead) | set(pool_cols) | set(cands) | {c for c in FAMILY_PACKS_COLS if c in schema})
    t0 = time.time()
    df = pd.read_parquet(STAGE[board], columns=cols)
    df["symbol"] = df["symbol"].astype(str)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df = prob_head._add_mfe_3d(df)
    log.info("[load:%s] %d 行, base %d 列, 候选 %d, 池列 %d (%.0fs)", board, len(df), len(base_cols), len(cands), len(pool_cols), time.time() - t0)
    return df, base_cols


FAMILY_PACKS_COLS = sorted({c for v in FAMILY_PACKS.values() for c in v})


def _pools(add: list[str], drop: set[str]) -> tuple[list[str], list[str]]:
    sn = [c for c in SNIPER.pool if c not in drop] + [c for c in add if c not in SNIPER.pool]
    fu = [c for c in FUSION.pool if c not in drop] + [c for c in add if c not in FUSION.pool]
    return sn, fu


def _mag_scores_pools(df: pd.DataFrame, board: str, sn: list[str], fu: list[str]) -> pd.Series:
    sn_s = pool_score(df, tuple(sn))
    fu_s = pool_score(df, tuple(fu))
    work = pd.DataFrame(
        {
            "symbol": df["symbol"],
            "date": df["date"],
            "board": board,
            "score": np.maximum(sn_s.to_numpy(), fu_s.to_numpy()),
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


def _top10_mean(ev: dict, key: str) -> float | None:
    vals = [ev[key]["per_horizon"][h]["top10_net"] for h in ("3d", "5d", "10d")]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _eval_overall(test: pd.DataFrame, name: str) -> dict:
    ev = evaluate_keys(test[["symbol", "date", "mag", "prob"] + list(NET_LABELS)], eval_days=TEST_DAYS)
    chosen = choose_rank_key(ev)
    top10_mean = {k: _top10_mean(ev, k) for k in ("mag", "prob", "blend")}
    log.info("[overall:%s] chosen=%s | top10净均 mag=%s prob=%s blend=%s",
             name, chosen,
             *(f"{top10_mean[k]:+.4f}" if top10_mean[k] is not None else "n/a" for k in ("mag", "prob", "blend")))
    return {"chosen": chosen, "weighted_ic": {k: ev[k]["weighted_ic"] for k in ev}, "top10_net_mean": top10_mean}


def _eval_halves(fr: pd.DataFrame, key: str) -> dict:
    """前/后半窗各评 (evaluate_keys trailing 全量=该半全部已实现日), 返回该键 top10 净均."""
    ds = np.sort(fr["date"].unique())
    mid = ds[len(ds) // 2]
    out = {}
    for half, sub in (("front", fr[fr["date"] < mid]), ("back", fr[fr["date"] >= mid])):
        ev = evaluate_keys(sub[["symbol", "date", "mag", "prob"] + list(NET_LABELS)], eval_days=TEST_DAYS)
        out[half] = _top10_mean(ev, key)
    return out


def _loser_enrich(fr_base: pd.DataFrame, fr_arm: pd.DataFrame, key: str) -> dict:
    """base TOP10 掉出者中 5d 净≥10% 占比 vs base TOP10 整体占比; >1.5x = 误杀红旗."""
    def _top(fr):
        s = fr[["date", "symbol", key]].dropna()
        q = s.sort_values(key, ascending=False)  # 与判官 daily_rank_metrics 同式 (弃 rank method="first" 的次序差)
        h = q.groupby("date", sort=False).head(10)
        return set(zip(h["date"], h["symbol"]))

    bi = _top(fr_base)
    dropped = bi - _top(fr_arm)
    b = fr_base.set_index(["date", "symbol"])["label_pm_5d_net"].dropna()  # 未实现 5d 标签不入分母 (尾部数日)

    def _rate(members: set) -> float | None:
        vals = b.reindex([x for x in members if x in b.index])
        return float((vals >= ENRICH_HIT).mean()) if len(vals) else None

    base_rate = _rate(bi)
    if not dropped:
        return {"n_dropped": 0, "base_hit_rate": base_rate, "drop_hit_rate": None, "enrich": None, "flag": False}
    drop_rate = _rate(dropped)
    enrich = drop_rate / base_rate if (base_rate is not None and base_rate > 0 and drop_rate is not None) else None
    return {
        "n_dropped": len(dropped),
        "base_hit_rate": base_rate,
        "drop_hit_rate": drop_rate,
        "enrich": enrich,
        "flag": bool(enrich is not None and enrich > ENRICH_MAX),
    }


def _frame_from(df, test_idx, mag: pd.Series, prob: np.ndarray) -> pd.DataFrame:
    fr = df.loc[test_idx, ["symbol", "date"]].reset_index(drop=True)
    fr["mag"] = mag.loc[test_idx].to_numpy(dtype=float)
    fr["prob"] = prob
    for c in NET_LABELS:
        fr[c] = df.loc[test_idx, c].to_numpy()
    return fr


def run_board(board: str, cands: list[str], cand_meta: dict) -> dict:
    df, base_cols = load_stage_cands(board, cands)
    pool_union = sorted(set(SNIPER.pool) | set(FUSION.pool))
    rep: dict = {"board": board, "n_base": len(base_cols), "pool": pool_union,
                 "cand_meta": {c: cand_meta.get(c) for c in cands}, "arms": {}}
    frs: dict[str, pd.DataFrame] = {}  # 臂名 → test 帧 (半窗/富集复用)

    dates = np.sort(df["date"].unique())
    cut = dates[-TEST_DAYS]
    trained_through = str(pd.Timestamp(df.loc[df["date"] < cut, "date"].max()).date())
    test_idx = df["date"] >= cut

    t0 = time.time()
    sn0, fu0 = _pools([], set())
    mag_base = _mag_scores_pools(df, board, sn0, fu0)
    log.info("[%s] mag_base 完成 (%.0fs)", board, time.time() - t0)
    SANDBOX.mkdir(parents=True, exist_ok=True)
    orig_bundle_dir = prob_head.bundle_dir
    prob_head.bundle_dir = lambda: SANDBOX
    try:
        t0 = time.time()
        prob_base = _prob_scores(df, board, base_cols, cut, trained_through, test_idx)
        log.info("[%s] prob_base 完成 (%.0fs)", board, time.time() - t0)
        fr_base = _frame_from(df, test_idx, mag_base, prob_base)
        rep["arms"]["base"] = _eval_overall(fr_base, "base")
        frs["base"] = fr_base

        def _run_mag(arm: str, add: list[str], drop: set[str]) -> None:
            ex = [c for c in add if c in df.columns]
            if add and not ex:
                log.warning("[%s] 臂 %s 列全缺席, 跳过", board, arm)
                return
            t0 = time.time()
            sn, fu = _pools(ex, drop)
            mag_arm = _mag_scores_pools(df, board, sn, fu)
            fr = _frame_from(df, test_idx, mag_arm, prob_base)
            rep["arms"][arm] = _eval_overall(fr, arm)
            frs[arm] = fr
            log.info("[%s] 臂 %s 完成 (%.0fs)", board, arm, time.time() - t0)
            del mag_arm, fr
            gc.collect()

        # ── L0: 候选单列加法 + 现役池列 LOO ──
        l0_add = {f"+{c}": [c] for c in cands if c not in pool_union}
        for c in cands:
            if c in pool_union:
                log.info("[%s] 候选 %s 已在池内, 加法臂由 LOO 覆盖", board, c)
        l0_loo = {f"−{c}": [] for c in pool_union}
        log.info("[%s] L0: %d 加法 + %d LOO", board, len(l0_add), len(l0_loo))
        for arm, add in l0_add.items():
            _run_mag(arm, add, set())
        for arm, col in l0_loo.items():
            _run_mag(arm, [], {col})

        # ── 护栏聚合 (L1 选胜者用) ──
        base_chosen = rep["arms"]["base"]["chosen"]
        base_top = rep["arms"]["base"]["top10_net_mean"][base_chosen]
        base_half = _eval_halves(frs["base"], base_chosen)

        def _guarded(arm: str) -> dict | None:
            if base_top is None:
                return None
            v = rep["arms"].get(arm)
            if not v or v["top10_net_mean"][v["chosen"]] is None:
                return None
            delta = v["top10_net_mean"][v["chosen"]] - base_top
            halves = _eval_halves(frs[arm], v["chosen"])
            base_h = _eval_halves(frs["base"], v["chosen"])
            half_ok = all(
                (h_v is not None and b_v is not None and h_v - b_v > 0)
                for h_v, b_v in ((halves["front"], base_h["front"]), (halves["back"], base_h["back"]))
            )
            enr = _loser_enrich(frs["base"], frs[arm], v["chosen"])
            return {"delta": delta, "halves": halves, "half_ok": half_ok, "enrich": enr}

        guard = {a: _guarded(a) for a in list(l0_add) + list(l0_loo)}

        # ── L1: 加法胜者 top-3 组合 + 族包 ──
        winners = sorted(
            ((a, g["delta"]) for a in list(l0_add) if (g := guard.get(a)) and g["delta"] > 0 and g["half_ok"]),
            key=lambda x: -x[1],
        )[:3]
        log.info("[%s] L1 胜者 (Δ>0 且双半窗稳): %s", board, winners)
        wcols = [l0_add[a][0] for a, _ in winners]
        combos: dict[str, list[str]] = {}
        for i in range(len(wcols)):
            for j in range(i + 1, len(wcols)):
                combos[f"+{wcols[i]}+{wcols[j]}"] = [wcols[i], wcols[j]]
        if len(wcols) == 3:
            combos["+" + "+".join(wcols)] = list(wcols)
        for arm, pack in FAMILY_PACKS.items():
            ex = [c for c in pack if c in df.columns]
            if len(ex) == len(pack):
                combos[arm] = pack
        seen_fs: set[frozenset] = {frozenset(v) for v in l0_add.values()}
        for arm, cols_ in list(combos.items()):
            fs = frozenset(cols_)
            if fs in seen_fs:
                continue
            seen_fs.add(fs)
            _run_mag(arm, cols_, set())
    finally:
        prob_head.bundle_dir = orig_bundle_dir

    # ── 判词输入 (全臂: Δ + 换键 + 双半窗 + 误杀富集) ──
    rep["verdict_inputs"] = {
        "base_chosen": base_chosen,
        "base_chosen_top10": base_top,
        "base_half": base_half,
        "per_arm": {},
    }
    for a in rep["arms"]:
        if a == "base":
            continue
        v = rep["arms"][a]
        ch, top = v["chosen"], v["top10_net_mean"][v["chosen"]]
        g = guard.get(a) if a in (l0_add or {}) or a in (l0_loo or {}) else _guarded(a)
        enr = (g or {}).get("enrich") if g else _loser_enrich(frs["base"], frs[a], ch)
        halves = (g or {}).get("halves") if g else _eval_halves(frs[a], ch)
        rep["verdict_inputs"]["per_arm"][a] = {
            "chosen": ch,
            "key_flipped": ch != base_chosen,
            "chosen_top10": top,
            "delta_vs_base_chosen_top10": (top - base_top) if (top is not None and base_top is not None) else None,
            "half_stable": (g or {}).get("half_ok"),
            "half_top10": halves,
            "loser_enrich": enr,
            "per_key_delta_top10": {
                k: (v["top10_net_mean"][k] - rep["arms"]["base"]["top10_net_mean"][k])
                for k in ("mag", "prob", "blend")
                if v["top10_net_mean"][k] is not None and rep["arms"]["base"]["top10_net_mean"][k] is not None
            },
        }
        vi = rep["verdict_inputs"]["per_arm"][a]
        log.info("[%s] %-24s chosen=%s%s Δ=%s 半稳=%s 富集=%s", board, a, ch,
                 " (换键!)" if vi["key_flipped"] else "",
                 f"{vi['delta_vs_base_chosen_top10']:+.4f}" if vi["delta_vs_base_chosen_top10"] is not None else "n/a",
                 vi["half_stable"],
                 f"{(vi['loser_enrich'] or {}).get('enrich'):.2f}" if (vi["loser_enrich"] or {}).get("enrich") is not None else "n/a")
    del df, frs
    gc.collect()
    _worm(f"{TAG}_{board}", rep)
    return rep


def main() -> int:
    _setup_logging()
    import psutil

    from scripts._run_guard import find_conflicts

    # ── 延时相: START_AT 前只睡 (用户令 8h 后) ──
    start_ts = time.mktime(time.strptime(START_AT, "%Y-%m-%d %H:%M:%S"))
    _state("SCHEDULED", start_at=START_AT)
    log.info("[sched] START_AT=%s (延时 %.1fh)", START_AT, max(0.0, start_ts - time.time()) / 3600)
    while time.time() < start_ts:
        time.sleep(300)
        if int(time.time() - start_ts) % 1800 < 300:
            log.info("[sched] 延时等待中, 剩 %.1fh", max(0.0, start_ts - time.time()) / 3600)

    # ── 自排队: 清场 + RAM>=5.5GB, 6h 上限 ──
    _state("GUARD")
    deadline = time.time() + 6 * 3600
    while True:
        c = find_conflicts()
        free_gb = psutil.virtual_memory().available / 1024**3
        if not c and free_gb >= 5.5:
            break
        if time.time() >= deadline:
            log.error("[guard] 6h 等待超时, 放弃")
            _state("FAIL_WAIT_TIMEOUT")
            return 3
        log.info("等待: 冲突=%s free=%.1fGB", [x["sentinel"] for x in c] or "无", free_gb)
        time.sleep(300)
    log.info("[guard] 清场确认 (free=%.1fGB)", free_gb)
    _state("RUNNING")
    check_startup_gate(RETRAIN_RAM_GUARD_MIN_FREE_GB * 1024**3)
    start_monitor(RETRAIN_RAM_GUARD_MIN_FREE_GB * 1024**3, RETRAIN_RAM_GUARD_POLL_S)

    cands, cand_meta = pick_candidates()
    for board in ("dual", "main"):  # 小板先行: 中途被杀也先落 dual WORM (对抗性审查)
        run_board(board, cands, cand_meta)
    log.info("[DONE] PARALLEL 池重扫两板完成")
    _state("DONE")
    return 0


if __name__ == "__main__":
    rc = main()
    if rc != 0:
        _state("FAIL_RC" if rc != 3 else "FAIL_WAIT_TIMEOUT", last_rc=rc)
    raise SystemExit(rc)
