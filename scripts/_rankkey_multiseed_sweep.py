"""_rankkey_multiseed_sweep.py — legacy 排名键多 seed × 多子窗 125d 终审 (2026-09-01).

背景: 排名键此前判死 (legacy-blend-rank-verdict / weight-gate-ab-verdict / 0830
排名键挑战者全判死) 全部是单 seed 单日期评审 — 10d_reg 头 run-to-run 方差 ±0.04/日
≈ 闸信号 (08-30 PASS / 08-31 FAIL 翻面, bcc93126). 本脚本用多 seed 取中位 + 子窗
连续性判据重审: mag / prob / blend 在 125 交易日回测下是否稳健胜过生产纯 mag.

设计:
  池固定: 生产包 ({board}_current.pkl) 逐日推理产出 E7 闸池 (基准闸通过 + 仅被
  pain 拦下的行), 所有键/seed 同池 — 只变排名变量.
  两个 walk-forward 头, 各 3 seed (multi_seed_seeds, 第二票闸同款), 每
  REFIT_EVERY 交易日重训, 评估日只看当天. seed = random_state + refit 相位错位
  (第 i 个 seed 错位 i×PHASE_STEP_DAYS 交易日): LGB_PARAMS_REG 无 bagging 子采样,
  单换 random_state 对 reg 头零方差 (冒烟实证 3 seed 输出全同); 第二票闸实测的
  ±0.04/日 run-to-run 方差真源是训练窗差 1 日 (08-30 vs 08-31 "窗仅差 1 日"),
  相位错位让 3 seed = 3 个不同训练窗相位, 直接复现该方差源 (multi seed windows).
    prob 头: 生产配方 (prob_head.LGB_PARAMS, mfe_3d>=abs_target 二分类),
             训练掩码止于 pos-4 (mfe_3d 需 +4 日, replay 08-16 审计修复同款)
    reg 头:  生产配方 (model_params("10d_reg") + split_window 反锚切分 +
             time_weights + ES 早停 + REG_MIN_TREES 地板, 即 _train_one 全路径),
             标签 = label_10d_net (pivot T+10 c2c 净, 扣 0.2% 往返成本;
             PM 分层滑点与固定 0.2% 差 ~0.05%, 对排名对比不敏感),
             训练掩码止于 pos-11 (10d 标签需 +11 日, 比 prob 头多 7 天 embargo)
  与生产唯一偏差: 10d_reg 头剥掉 deterministic/force_row_wise/num_threads=1 单线程
  块 — 36 次 reg 拟合单线程 ≈ 9h 无法当天完成; 直方图归约噪声本就是 ±0.04/日
  方差的组成部分, 中位聚合 + 子窗判据对其稳健 (闸的 deterministic 用途不受影响).

排名键 (逐 seed; 一套 seed 训练同时产出两头, 与生产链同构):
  mag       = 生产包 pred_ret_10d (基线, 现状, 无 seed 方差)
  mag_wf    = walk-forward reg 头 (seed s)
  prob_wf   = walk-forward prob 头 (seed s)
  blend_new = mag_wf × prob_wf
  blend_ex  = mag_wf × (prob_wf − base_prod)

判据 (预登记, per board, E7 池, depth=10, 125d 全窗):
  挑战者过闸 = seed 中位 delta(vs mag 基线) > 0 且 ≥2/3 seed delta > 0
             且 4 子窗中 ≥3 子窗 delta > 0.  depth 5/15 同号为扰动佐证 (不翻转).
  生产闸池 (E7 + prob_wf > base_prod + 0.08, fail-open) 作次口径同报.

检查点: data/_diag_rankkey_wf_{prob|reg}_{board}_s{seed}_e{eval}.parquet
        (崩溃重跑免重训; 特征不落盘, 与 prob_head_replay 同权衡)
        data/_diag_rankkey_scored_{board}_e{eval}.parquet
        (逐日 scored 池全量成分 + base_prod; 闸/池口径再变免推理)
池口径: E7 闸镜像生产 list_generator, q50 符号闸跟随 LEGACY_ENTRY_GATE.
        q50_sign_gate 配置旗 (2026-09-02 三臂回放判死默认撤).
WORM: DATA OTHERS/diag/rankkey_multiseed_{ts}.csv/.json + *_daily_{ts}.csv

用法:
  python scripts/_rankkey_multiseed_sweep.py                        # 125d 全量
  python scripts/_rankkey_multiseed_sweep.py --slice 120 --eval 20  # 冒烟
  python scripts/_rankkey_multiseed_sweep.py --pool-from-ckpt       # 改闸后免推理重分析
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from app.pipeline1 import dual_track_trainer as dtt
from app.pipeline1 import prob_head
from app.pipeline1.cleaning_pipeline import CleaningPipeline, load_panel_v3
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import LabelEngine, _ensure_sorted
from app.pipeline1.list_generator import ListGenerator
from app.pipeline1.predictor import V35Predictor
from config.settings import (
    DATA_DIR,
    LEGACY_ENTRY_GATE,
    LEGACY_PROB_GATE,
    LEGACY_TOP10_SECOND_VOTE,
    PANEL_V3_PATH,
    data_others_path,
)
from scripts._run_guard import find_conflicts

BUNDLES = {
    "main": "models/pipeline1/main_current.pkl",
    "dual": "models/pipeline1/dual_current.pkl",
}
SEEDS = tuple(int(s) for s in LEGACY_TOP10_SECOND_VOTE["multi_seed_seeds"])
REFIT_EVERY = int(LEGACY_PROB_GATE["refit_every_days"])
PHASE_STEP_DAYS = 7  # 各 seed 的 refit 相位错位步长 (见 docstring: multi seed windows)
ABS_TARGET = float(LEGACY_PROB_GATE["abs_target"])
COST = 0.0020  # 往返成本 (prob_head_replay 同款): 佣金+印花税+滑点 ≈ 0.2%
REALIZED_BUY_LAG = 1
REALIZED_SELL_LAG = 11
REG_HORIZON = REALIZED_SELL_LAG
REG_EMBARGO = REG_HORIZON  # 10d 标签需 cal[j+11] → 训练掩码止于 pos-11
PROB_EMBARGO = 4  # mfe_3d 需 +4 日 → pos-4 (replay 08-16 审计修复)
BASE_TAIL_DAYS = 35
PROB_GATE_MARGIN = 0.08  # 生产闸池 margin (41eff001 同款)
# 闸3 q50 符号闸镜像生产 (2026-09-02 三臂回放判死默认撤, list_generator 同款旗):
# False → E7 池不再按 pred_q50_3d/5d>0 过滤 (20260902 撤闸后口径)
Q50_SIGN_GATE = bool(LEGACY_ENTRY_GATE.get("q50_sign_gate", False))
DEPTHS = (5, 10, 15)
DEPTH_VERDICT = 10
N_SUB = 4
KEYS = (
    "key:mag",
    "key:mag_wf",
    "key:prob_wf",
    "key:blend_new",
    "key:blend_ex",
)
KEY_BASELINE = "key:mag"
KEY_LABELS = {
    "key:mag": "mag(生产包, 现状基线)",
    "key:mag_wf": "mag_wf(walk-forward reg 头)",
    "key:prob_wf": "prob_wf(纯概率, walk-forward)",
    "key:blend_new": "blend_new=mag_wf×prob_wf",
    "key:blend_ex": "blend_ex=mag_wf×(prob_wf−base_prod)",
}


# ── 纯函数 (单测见 tests/test_rankkey_multiseed.py) ─────────────────────────


def _daily_topn(df: pd.DataFrame, rank_col: str, depth: int) -> pd.DataFrame:
    """逐日 rank_col 降序取前 depth (NaN 排最后, 同 rank_ab 矩阵口径)."""
    return (
        df.sort_values(["date", rank_col], ascending=[True, False])
        .groupby("date", sort=False)
        .head(depth)
    )


def _daily_net(top: pd.DataFrame) -> pd.Series:
    """逐日 top-N realized_net 均值 (T+10 净, ±0.04/日 方差同口径)."""
    return top.groupby("date")["realized_net"].mean()


def _seed_median_series(per_seed: dict) -> pd.Series:
    """{seed: 逐日净序列} → 逐日跨 seed 中位 (某 seed 缺日则取其余 seed)."""
    return pd.DataFrame(per_seed).median(axis=1)


def _sub_means(series: pd.Series, n_sub: int = N_SUB) -> list[float]:
    """时序等分 n_sub 段的段均值 (空段 NaN), 子窗连续性判据用."""
    vals = series.to_numpy(dtype=float)
    step = max(1, len(vals) // n_sub)
    out = []
    for i in range(n_sub):
        s0 = i * step
        s1 = len(vals) if i == n_sub - 1 else (i + 1) * step
        seg = vals[s0:s1]
        out.append(float(seg.mean()) if len(seg) else float("nan"))
    return out


def _challenger_verdict(delta_by_seed: dict, sub_deltas: list[float]) -> dict:
    """预登记判据: 中位 delta>0 且 ≥2/3 seed 为正 且 ≥3/4 有效子窗为正."""
    deltas = [float(v) for v in delta_by_seed.values()]
    med = float(np.median(deltas)) if deltas else float("nan")
    seeds_pos = int(sum(1 for d in deltas if d > 0))
    valid = [d for d in sub_deltas if np.isfinite(d)]
    subs_pos = int(sum(1 for d in valid if d > 0))
    passed = bool(
        np.isfinite(med)
        and med > 0
        and len(deltas) > 0
        and seeds_pos * 3 >= 2 * len(deltas)
        and len(valid) > 0
        and subs_pos * 4 >= 3 * len(valid)
    )
    return {
        "delta_median": med,
        "seeds_pos": seeds_pos,
        "n_seeds": len(deltas),
        "subs_pos": subs_pos,
        "n_subs_valid": len(valid),
        "pass": passed,
    }


def _reg_labels_from_matrix(
    price: np.ndarray,
    sym_rows: np.ndarray,
    j_cols: np.ndarray,
    cost: float,
    horizon: int,
) -> np.ndarray:
    """T+horizon c2c 净标签: buy=cal[j+1] 收盘, sell=cal[j+horizon] 收盘.

    price: (n_symbol, n_cal) ffill 收盘矩阵 (与池 realized_net 同 pivot 同口径);
    越界 (j+horizon ≥ n_cal) / 非有限 / 买价 ≤0 → NaN (训练时被 _train_one dropna).
    """
    n = price.shape[1]
    buy_j = j_cols + REALIZED_BUY_LAG
    sell_j = j_cols + horizon
    out = np.full(len(j_cols), np.nan, dtype=float)
    ok = (buy_j < n) & (sell_j < n)
    if ok.any():
        r, b, s = sym_rows[ok], buy_j[ok], sell_j[ok]
        pb, ps = price[r, b], price[r, s]
        net = ps / pb - 1.0 - cost
        bad = ~(np.isfinite(net) & np.isfinite(pb) & (pb > 0))
        net[bad] = np.nan
        out[ok] = net
    return out


# ── 生产配方构件 (镜像 replay / _train_one) ─────────────────────────────────


def _build_realized_pivot(panel: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """symbol×date → close_hfq (每股 ffill 处理停牌), 返回 (宽表 pivot, 交易日历)."""
    cal = np.unique(pd.to_datetime(panel["date"].to_numpy()).normalize().to_numpy())
    cal = np.sort(cal)
    pivot = (
        panel.assign(dt=pd.to_datetime(panel["date"]).dt.normalize())
        .pivot_table(index="symbol", columns="dt", values="close_hfq", aggfunc="last")
        .sort_index()
    )
    pivot.index = pivot.index.astype(str)
    pivot = pivot.reindex(columns=pd.to_datetime(cal)).ffill(axis=1)
    return pivot, cal


def _realized_net(
    pivot: pd.DataFrame, cal: np.ndarray, i: int, symbol: str, cost: float = COST
) -> float:
    """决策日 cal[i] 的 T+10 净实得 (replay 同款, 池行评估口径)."""
    buy_dt = pd.Timestamp(cal[i + REALIZED_BUY_LAG])
    sell_dt = pd.Timestamp(cal[i + REALIZED_SELL_LAG])
    try:
        pb = float(pivot.at[symbol, buy_dt])
        ps = float(pivot.at[symbol, sell_dt])
    except KeyError:
        return float("nan")
    if not (np.isfinite(pb) and np.isfinite(ps)) or pb <= 0:
        return float("nan")
    return ps / pb - 1.0 - cost


def _gate_mask(
    scored: pd.DataFrame,
    prob_margin: float = 0.0,
    ret_thresh: float = 0.0,
    pain_thresh: float = 0.5,
) -> pd.Series:
    """生产 entry_filter 非 bear 口径 (replay 同款)."""
    ok = pd.Series(True, index=scored.index)
    cp = (
        scored["compound_prob"]
        if "compound_prob" in scored.columns
        else scored["prob_up"]
    )
    cr = (
        scored["compound_ret"]
        if "compound_ret" in scored.columns
        else scored["pred_ret_10d"]
    )
    ok &= cp > scored["base_rate"] + prob_margin
    ok &= cr > ret_thresh
    if Q50_SIGN_GATE and all(
        c in scored.columns for c in ("pred_q50_3d", "pred_q50_5d")
    ):
        ok &= (scored["pred_q50_3d"].fillna(cr) > 0) & (
            scored["pred_q50_5d"].fillna(cr) > 0
        )
    if "pain_prob" in scored.columns and pain_thresh is not None:
        ok &= scored["pain_prob"].fillna(0) <= pain_thresh
    return ok


def _build_raw_labels(dfb: pd.DataFrame) -> pd.DataFrame:
    """清洗帧 → 小 raw 帧 (symbol/date/close/high/low/adv20/mfe_3d/label_pain)."""
    if "adv20" not in dfb.columns:
        if "amount" not in dfb.columns:
            raise ValueError("清洗帧缺 amount (无法现算 adv20 打标签)")
        dfb = _ensure_sorted(dfb)
        dfb["adv20"] = (
            dfb.groupby("symbol")["amount"]
            .rolling(20, min_periods=20)
            .mean()
            .reset_index(level=0, drop=True)
        )
    raw = dfb[["symbol", "date", "close_hfq", "high_hfq", "low_hfq", "adv20"]].copy()
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
    return raw


def _unpin_deterministic() -> None:
    """10d_reg 头剥掉 deterministic 单线程块 (见模块 docstring 偏差说明)."""
    orig = dtt.model_params

    def fast(board: str, kind: str) -> dict:
        p = orig(board, kind)
        if kind == "10d_reg":
            for k in ("deterministic", "force_row_wise", "num_threads"):
                p.pop(k, None)
        return p

    dtt.model_params = fast


# ── walk-forward 训练 ────────────────────────────────────────────────────────


def _wf_prob_board(
    feat: pd.DataFrame,
    eval_days: list,
    board_dates_arr: np.ndarray,
    idx: np.ndarray,
    board: str,
    eval_n: int,
) -> None:
    """prob 头 walk-forward × SEEDS (生产配方, 掩码止于 pos-4), 检查点落盘."""
    feat_cols = prob_head.feature_cols(feat)
    y = (feat["mfe_3d"] >= ABS_TARGET).astype(float)
    ok_arr = (y.notna() & feat["label_pain"].notna()).to_numpy()
    x_all = feat[feat_cols].to_numpy(dtype="float32")
    for seed in SEEDS:
        ckpt = DATA_DIR / f"_diag_rankkey_wf_prob_{board}_s{seed}_e{eval_n}.parquet"
        if ckpt.exists():
            print(f"[wf-prob:{board}] seed={seed} 从检查点恢复 {ckpt.name}", flush=True)
            continue
        phase = PHASE_STEP_DAYS * SEEDS.index(seed)
        t0 = time.time()
        model = None
        rows: list[pd.DataFrame] = []
        n_refits = 0
        for k, d in enumerate(eval_days):
            pos = int(np.searchsorted(board_dates_arr, np.datetime64(d)))
            if model is None or (k + phase) % REFIT_EVERY == 0:
                tr = (idx < pos - PROB_EMBARGO) & ok_arr
                model = LGBMClassifier(
                    **{**prob_head.LGB_PARAMS, "random_state": int(seed)}
                )
                model.fit(x_all[tr], y.loc[tr].to_numpy())
                n_refits += 1
            te = idx == pos
            if not te.any():
                continue
            p = model.predict_proba(x_all[te])[:, 1]
            rows.append(
                feat.loc[te, ["symbol", "date"]].assign(pred=p).reset_index(drop=True)
            )
        pd.concat(rows, ignore_index=True).to_parquet(str(ckpt))
        print(
            f"[wf-prob:{board}] seed={seed} 完成: {n_refits} 次重训 "
            f"→ {ckpt.name} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    del x_all
    gc.collect()


def _wf_reg_board(
    feat: pd.DataFrame,
    feat_cols_reg: list[str],
    eval_days: list,
    board_dates_arr: np.ndarray,
    idx: np.ndarray,
    board: str,
    eval_n: int,
) -> None:
    """reg 头 walk-forward × SEEDS (生产 _train_one 全路径, 掩码止于 pos-11)."""
    trainer = dtt.DualTrackTrainer()
    for seed in SEEDS:
        ckpt = DATA_DIR / f"_diag_rankkey_wf_reg_{board}_s{seed}_e{eval_n}.parquet"
        if ckpt.exists():
            print(f"[wf-reg:{board}] seed={seed} 从检查点恢复 {ckpt.name}", flush=True)
            continue
        phase = PHASE_STEP_DAYS * SEEDS.index(seed)
        t0 = time.time()
        model = None
        rows: list[pd.DataFrame] = []
        n_refits = 0
        for k, d in enumerate(eval_days):
            pos = int(np.searchsorted(board_dates_arr, np.datetime64(d)))
            if model is None or (k + phase) % REFIT_EVERY == 0:
                tr_mask = idx < (pos - REG_EMBARGO)
                segs = trainer.split_window(feat.loc[tr_mask])
                model, _label = trainer._train_one(
                    "10d_reg", segs, feat_cols_reg, board, seed=seed
                )
                del segs
                gc.collect()
                n_refits += 1
            te = idx == pos
            if not te.any():
                continue
            X_te = np.nan_to_num(feat.loc[te, feat_cols_reg].to_numpy(), nan=0.0)
            pred = model.predict(X_te)
            rows.append(
                feat.loc[te, ["symbol", "date"]]
                .assign(pred=pred)
                .reset_index(drop=True)
            )
            if (k + 1) % 25 == 0 or k == len(eval_days) - 1:
                print(
                    f"[wf-reg:{board}] seed={seed} {k + 1}/{len(eval_days)} "
                    f"(refits={n_refits}, {time.time() - t0:.0f}s)",
                    flush=True,
                )
        pd.concat(rows, ignore_index=True).to_parquet(str(ckpt))
        print(
            f"[wf-reg:{board}] seed={seed} 完成: {n_refits} 次重训 → {ckpt.name} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )


# ── 主流程 ───────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=420, help="面板切片交易日数")
    ap.add_argument("--eval", type=int, default=125, help="评估的已实现决策日数")
    ap.add_argument(
        "--pool-from-ckpt",
        action="store_true",
        help="跳过推理: 从 scored 检查点重建池直接分析 (闸口径按当前配置重推)",
    )
    args = ap.parse_args()

    hits = find_conflicts()
    if hits:
        print(f"[guard] 存活重活进程冲突, 退出: {hits}", flush=True)
        return 2

    _unpin_deterministic()

    t0 = time.time()
    print(
        f"[cfg] slice={args.slice} eval={args.eval} seeds={SEEDS} "
        f"(refit 相位错位 {PHASE_STEP_DAYS}d/seed) refit_every={REFIT_EVERY} "
        f"embargo(prob/reg)={PROB_EMBARGO}/{REG_EMBARGO} "
        f"depths={DEPTHS} n_sub={N_SUB} q50_sign_gate={Q50_SIGN_GATE}",
        flush=True,
    )

    if args.pool_from_ckpt:
        ckpt_pool, ckpt_base = _load_pool_from_ckpt(args.eval)
        print(
            f"[ckpt] 池重建 {len(ckpt_pool):,} 行 "
            f"(q50_sign_gate={Q50_SIGN_GATE}, {time.time() - t0:.0f}s)",
            flush=True,
        )
        return _analyze(ckpt_pool, ckpt_base, args.eval, t0)

    predictor = V35Predictor(BUNDLES)
    cleaner = CleaningPipeline()
    features = FeatureEngineV35()
    lister = ListGenerator()

    print(f"[load] panel {PANEL_V3_PATH}", flush=True)
    panel = load_panel_v3(path=PANEL_V3_PATH)
    print(
        f"[load] {len(panel):,}r max={panel['date'].max()} ({time.time() - t0:.0f}s)",
        flush=True,
    )
    dates_all = sorted(pd.unique(pd.to_datetime(panel["date"])))
    cut = dates_all[-args.slice]
    panel = panel[pd.to_datetime(panel["date"]) >= cut].reset_index(drop=True)
    print(
        f"[slice] {pd.Timestamp(cut).date()}.. {len(panel):,}r ({time.time() - t0:.0f}s)",
        flush=True,
    )

    pivot, cal = _build_realized_pivot(panel)
    print(
        f"[pivot] symbols={len(pivot)} days={len(cal)} ({time.time() - t0:.0f}s)",
        flush=True,
    )

    main_df, dual_df, state = cleaner.run_inference(panel)
    print(
        f"[clean] valve={state} main={len(main_df):,} dual={len(dual_df):,} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    del panel
    gc.collect()

    all_cal = pd.to_datetime(cal)
    i_of = {d: i for i, d in enumerate(all_cal)}

    detail: list[dict] = []
    scored_frames: dict[str, list] = {}
    base_maps: dict[str, dict] = {}
    board_meta: dict[str, dict] = {}

    for board, dfb, csr in (("main", main_df, False), ("dual", dual_df, True)):
        cols = predictor.bundles[board]["feature_cols"]
        feat = features.build(dfb, None, inference_cols=cols, cross_sectional_rank=csr)
        feat = feat.reset_index(drop=True)
        feat["symbol"] = feat["symbol"].astype(str)
        feat["date"] = pd.to_datetime(feat["date"])
        print(
            f"[feat:{board}] {len(feat):,}r {len(feat.columns)}c "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

        day_dates = sorted(pd.unique(feat["date"]))
        eval_days = [
            d
            for d in day_dates
            if d in i_of and i_of[d] + REALIZED_SELL_LAG < len(all_cal)
        ][-args.eval :]
        print(
            f"[{board}] eval days {len(eval_days)} "
            f"({pd.Timestamp(eval_days[0]).date()}..{pd.Timestamp(eval_days[-1]).date()})",
            flush=True,
        )

        # -- 1) 候选池: 基准闸行 + 仅被 pain 拦下的行 (replay 同款) --
        warm_days = [d for d in day_dates if d < eval_days[0]]
        for d in warm_days:
            day_feat = feat[feat["date"] == d]
            if day_feat.empty:
                continue
            try:
                pred = predictor.predict(day_feat, board)
                if not pred.empty:
                    lister.compute_scores(pred)
            except Exception:
                pass
        print(f"[{board}] base_rate 预热 {len(warm_days)} 天", flush=True)

        for k, d in enumerate(eval_days):
            di = i_of[d]
            day_feat = feat[feat["date"] == d]
            if day_feat.empty:
                continue
            try:
                pred = predictor.predict(day_feat, board)
            except Exception as exc:
                print(
                    f"[{board}] {pd.Timestamp(d).date()} predict err: {exc}",
                    flush=True,
                )
                continue
            if pred.empty:
                continue
            scored = lister.compute_scores(pred)
            scored["date"] = d
            scored["board"] = board
            cp = (
                scored["compound_prob"]
                if "compound_prob" in scored.columns
                else scored["prob_up"]
            )
            cr = (
                scored["compound_ret"]
                if "compound_ret" in scored.columns
                else scored["pred_ret_10d"]
            )
            ing = pd.DataFrame(
                {
                    "date": d,
                    "board": board,
                    "symbol": scored["symbol"].astype(str).to_numpy(),
                    "pred_ret_10d": cr.to_numpy(),
                    "prob": cp.to_numpy(),
                    "base_rate": (
                        scored["base_rate"].to_numpy()
                        if "base_rate" in scored.columns
                        else np.nan
                    ),
                    "pain_prob": (
                        scored["pain_prob"].to_numpy()
                        if "pain_prob" in scored.columns
                        else np.nan
                    ),
                    "pred_q50_3d": (
                        scored["pred_q50_3d"].to_numpy()
                        if "pred_q50_3d" in scored.columns
                        else np.nan
                    ),
                    "pred_q50_5d": (
                        scored["pred_q50_5d"].to_numpy()
                        if "pred_q50_5d" in scored.columns
                        else np.nan
                    ),
                    "realized_net": [
                        _realized_net(pivot, cal, di, str(s)) for s in scored["symbol"]
                    ],
                }
            )
            scored_frames.setdefault(board, []).append(ing)
            base_mask = _gate_mask(scored)
            pain_mask = _gate_mask(scored, pain_thresh=None)
            keep = (base_mask | pain_mask).to_numpy()
            pe = (~base_mask).to_numpy()
            for tup, pex in zip(ing.loc[keep].itertuples(index=False), pe[keep]):
                detail.append(
                    {
                        "date": str(pd.Timestamp(d).date()),
                        "board": board,
                        "symbol": tup.symbol,
                        "pred_ret_10d": float(tup.pred_ret_10d),
                        "prob": float(tup.prob),
                        "base_rate": float(tup.base_rate),
                        "pain_excluded": bool(pex),
                        "realized_net": float(tup.realized_net),
                    }
                )
            if (k + 1) % 25 == 0 or k == len(eval_days) - 1:
                print(
                    f"[{board}] detail {k + 1}/{len(eval_days)} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )

        # -- 2) 标签 + base_prod 逐日序列 (生产 _base_rate, 尾切片止于 pos-4) --
        raw = _build_raw_labels(dfb)
        del dfb
        gc.collect()
        board_dates_arr = np.array(pd.to_datetime(day_dates))
        base_map: dict[pd.Timestamp, float] = {}
        for d in eval_days:
            pos = int(np.searchsorted(board_dates_arr, np.datetime64(d)))
            lo = max(0, pos - BASE_TAIL_DAYS + 1)
            tail = raw[
                (raw["date"] >= pd.Timestamp(board_dates_arr[lo]))
                & (raw["date"] <= pd.Timestamp(board_dates_arr[pos - 4]))
            ]
            b = prob_head._base_rate(tail)
            base_map[pd.Timestamp(d)] = b if b is not None else np.nan
        base_maps[board] = base_map
        n_ok = sum(1 for v in base_map.values() if np.isfinite(v))
        print(
            f"[{board}] base_prod {n_ok}/{len(eval_days)} 日可用 "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

        # -- 2.5) scored 池检查点: 未来闸/池口径再变 → --pool-from-ckpt 分钟级重分析 --
        if scored_frames.get(board):
            sf = pd.concat(scored_frames[board], ignore_index=True)
            sf["base_prod"] = sf["date"].map(base_map)
            ckpt = DATA_DIR / f"_diag_rankkey_scored_{board}_e{args.eval}.parquet"
            sf.to_parquet(str(ckpt))
            print(f"[{board}] scored 检查点 {len(sf):,} 行 → {ckpt.name}", flush=True)

        # -- 3) 标签并入特征帧 + reg 净标签 (pivot 向量化, 无逐股循环) --
        feat = feat.merge(
            raw[["symbol", "date", "mfe_3d", "label_pain"]],
            on=["symbol", "date"],
            how="left",
        )
        del raw
        gc.collect()
        price = pivot.to_numpy(dtype="float64")
        sym_rows = pivot.index.get_indexer(feat["symbol"].to_numpy())
        j_cols = np.searchsorted(cal, feat["date"].to_numpy())
        if not np.all(cal[j_cols] == feat["date"].to_numpy()):
            print(f"[{board}] FAIL 特征帧日期不在 pivot 日历中", flush=True)
            return 2
        feat["label_10d_net"] = _reg_labels_from_matrix(
            price, sym_rows, j_cols, COST, horizon=REG_HORIZON
        )
        print(
            f"[{board}] reg 净标签 {feat['label_10d_net'].notna().sum():,}/"
            f"{len(feat):,} 行可用 ({time.time() - t0:.0f}s)",
            flush=True,
        )

        idx = np.searchsorted(board_dates_arr, feat["date"].values)
        board_meta[board] = {
            "feat_cols": cols,
            "eval_days": eval_days,
            "board_dates_arr": board_dates_arr,
            "idx": idx,
        }

        # -- 4) walk-forward × seeds (prob 先行释放 x_all, reg 复用 feat 帧路径) --
        _wf_prob_board(feat, eval_days, board_dates_arr, idx, board, args.eval)
        _wf_reg_board(feat, cols, eval_days, board_dates_arr, idx, board, args.eval)
        del feat
        gc.collect()

    if not detail:
        print("无任何过闸候选", flush=True)
        return 1

    pool_df = pd.DataFrame(detail)
    pool_df["date"] = pd.to_datetime(pool_df["date"])
    return _analyze(pool_df, base_maps, args.eval, t0)


def _load_pool_from_ckpt(eval_n: int) -> tuple[pd.DataFrame, dict]:
    """scored 检查点 → (pool_df, base_maps), 闸掩码按当前 Q50_SIGN_GATE 重推.

    池 = 通过非 pain 条件 (prob>base_rate 且 ret>0 且 [q50 闸]) 的全部行;
    pain_excluded = 其中仅被 pain>0.5 拦下的行 (与 detail 循环 base|pain 同语义:
    base|pain = 非 pain 条件通过). 未来闸/池口径再变 → 分钟级重分析, 免 3h 推理.
    """
    frames: list[pd.DataFrame] = []
    base_maps: dict[str, dict] = {}
    for board in ("main", "dual"):
        ck = DATA_DIR / f"_diag_rankkey_scored_{board}_e{eval_n}.parquet"
        fr = pd.read_parquet(str(ck))
        fr["date"] = pd.to_datetime(fr["date"])
        ok = (fr["prob"] > fr["base_rate"]) & (fr["pred_ret_10d"] > 0)
        if Q50_SIGN_GATE and {"pred_q50_3d", "pred_q50_5d"}.issubset(fr.columns):
            ok &= (fr["pred_q50_3d"].fillna(fr["pred_ret_10d"]) > 0) & (
                fr["pred_q50_5d"].fillna(fr["pred_ret_10d"]) > 0
            )
        rec = fr[ok].copy()
        rec["pain_excluded"] = (rec["pain_prob"].fillna(0) > 0.5).to_numpy()
        frames.append(rec)
        base_maps[board] = (
            rec.drop_duplicates("date").set_index("date")["base_prod"].to_dict()
        )
    return pd.concat(frames, ignore_index=True), base_maps


def _analyze(pool_df: pd.DataFrame, base_maps: dict, eval_n: int, t0: float) -> int:
    """逐 seed 键排名 → 日净序列 → 跨 seed 中位 → 判据 → WORM 落盘."""
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)

    rows_out: list[dict] = []
    verdicts: dict = {}
    daily_out: list[pd.DataFrame] = []

    for board in ("main", "dual"):
        b = pool_df[pool_df["board"] == board].copy()
        e7 = b[~b["pain_excluded"].fillna(False)].copy()
        e7["base_prod"] = e7["date"].map(base_maps[board])
        eval_days = sorted(b["date"].unique())

        for pool_name, per_seed_pools in (("E7池", None), ("生产闸池", "gate")):
            per_seed_frames: dict = {}
            for seed in SEEDS:
                s = e7.copy()
                wfp = pd.read_parquet(
                    str(
                        DATA_DIR
                        / f"_diag_rankkey_wf_prob_{board}_s{seed}_e{eval_n}.parquet"
                    )
                )
                wfp["date"] = pd.to_datetime(wfp["date"])
                wfr = pd.read_parquet(
                    str(
                        DATA_DIR
                        / f"_diag_rankkey_wf_reg_{board}_s{seed}_e{eval_n}.parquet"
                    )
                )
                wfr["date"] = pd.to_datetime(wfr["date"])
                s = s.merge(wfp, on=["symbol", "date"], how="left").rename(
                    columns={"pred": "prob_wf"}
                )
                s = s.merge(wfr, on=["symbol", "date"], how="left").rename(
                    columns={"pred": "reg_wf"}
                )
                s["key:mag"] = s["pred_ret_10d"]
                s["key:mag_wf"] = s["reg_wf"]
                s["key:prob_wf"] = s["prob_wf"]
                s["key:blend_new"] = s["reg_wf"] * s["prob_wf"]
                s["key:blend_ex"] = s["reg_wf"] * (s["prob_wf"] - s["base_prod"])
                if per_seed_pools == "gate":
                    keep = (
                        (s["prob_wf"] > s["base_prod"] + PROB_GATE_MARGIN)
                        | s["prob_wf"].isna()
                        | s["base_prod"].isna()
                    )
                    s = s[keep]
                per_seed_frames[seed] = s

            depths = DEPTHS if pool_name == "E7池" else (DEPTH_VERDICT,)
            key_series: dict = {}
            for seed, s in per_seed_frames.items():
                for key in KEYS:
                    for depth in depths:
                        net = _daily_net(_daily_topn(s, key, depth))
                        key_series.setdefault((key, depth), {})[seed] = net
                        r = net.dropna()
                        rows_out.append(
                            {
                                "board": board,
                                "pool": pool_name,
                                "key": key,
                                "seed": seed,
                                "depth": depth,
                                "n_days": int(net.notna().sum()),
                                "mean": float(r.mean()) if len(r) else np.nan,
                                "hit": float((r > 0).mean()) if len(r) else np.nan,
                                "med": float(r.median()) if len(r) else np.nan,
                                "ge5": float((r >= 0.05).mean()) if len(r) else np.nan,
                            }
                        )

            print(
                f"\n===== {board} / {pool_name} (depth={DEPTH_VERDICT}, "
                f"{len(eval_days)} 评估日, {len(SEEDS)} seed 中位) =====",
                flush=True,
            )
            print(
                f"  {'键':<28}{'日均净':>9}{'逐seed':>26}{'delta中位':>10}"
                f"{'seed正':>7}{'子窗正':>7}  判定",
                flush=True,
            )
            verdicts.setdefault(board, {})[pool_name] = {}
            for key in KEYS:
                ks = key_series[(key, DEPTH_VERDICT)]
                bs = key_series[(KEY_BASELINE, DEPTH_VERDICT)]
                med_series = _seed_median_series(ks)
                full = float(med_series.mean())
                per_seed_means = [float(ks[s].mean()) for s in SEEDS]
                if key == KEY_BASELINE:
                    print(
                        f"  {KEY_LABELS[key]:<28}{full:>+9.2%}"
                        f"{' '.join(f'{m:>+8.2%}' for m in per_seed_means):>26}"
                        f"{'—':>10}{'—':>7}{'—':>7}  基线",
                        flush=True,
                    )
                    continue
                delta_by_seed = {
                    s: float((ks[s].dropna() - bs[s].dropna()).mean()) for s in SEEDS
                }
                dmed = (_seed_median_series(ks) - _seed_median_series(bs)).dropna()
                sub_d = _sub_means(dmed)
                v = _challenger_verdict(delta_by_seed, sub_d)
                verdicts[board][pool_name][key] = {
                    **v,
                    "full_median": full,
                    "per_seed_means": dict(zip(SEEDS, per_seed_means)),
                    "sub_deltas": sub_d,
                    "delta_by_seed": delta_by_seed,
                }
                sub_s = "/".join(f"{x:+.2%}" for x in sub_d)
                print(
                    f"  {KEY_LABELS[key]:<28}{full:>+9.2%}"
                    f"{' '.join(f'{m:>+8.2%}' for m in per_seed_means):>26}"
                    f"{v['delta_median']:>+10.2%}"
                    f"{v['seeds_pos']}/{v['n_seeds']:>3}"
                    f"{v['subs_pos']}/{v['n_subs_valid']:>3}  "
                    f"{'通过' if v['pass'] else '不通过'} [子窗delta {sub_s}]",
                    flush=True,
                )
                daily = pd.DataFrame(
                    {
                        "board": board,
                        "pool": pool_name,
                        "key": key,
                        "date": med_series.index,
                        "net10_med": med_series.to_numpy(),
                    }
                )
                for s in SEEDS:
                    daily[f"net10_s{s}"] = ks[s].reindex(med_series.index).to_numpy()
                daily_out.append(daily)

    # ── WORM 落盘 ──
    pd.DataFrame(rows_out).to_csv(out_dir / f"rankkey_multiseed_{ts}.csv", index=False)
    (out_dir / f"rankkey_multiseed_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "slice": None,
                "eval": eval_n,
                "q50_sign_gate": Q50_SIGN_GATE,
                "seeds": list(SEEDS),
                "refit_every": REFIT_EVERY,
                "embargo": {"prob": PROB_EMBARGO, "reg": REG_EMBARGO},
                "cost": COST,
                "abs_target": ABS_TARGET,
                "keys": KEY_LABELS,
                "verdict_rule": (
                    "seed中位delta>0 且 ≥2/3 seed delta>0 且 ≥3/4 子窗 delta>0 "
                    "(depth=10, E7池 为主口径)"
                ),
                "verdicts": verdicts,
                "n_detail": len(pool_df),
                "runtime_s": round(time.time() - t0, 0),
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    pd.concat(daily_out, ignore_index=True).to_csv(
        out_dir / f"rankkey_multiseed_daily_{ts}.csv", index=False
    )
    print(
        f"\n[saved] {out_dir}/rankkey_multiseed_{ts}.csv/.json + *_daily_{ts}.csv "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
