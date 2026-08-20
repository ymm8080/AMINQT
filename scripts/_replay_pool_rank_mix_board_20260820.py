"""_replay_pool_rank_mix_20260820.py — 入池排名混合回测: 池分×预期涨幅 混合选池 vs 纯池分.

用户需求 (2026-08-20): T-5/T-10/T-20 短名单名额目前由"池分"(流动性 rank_5050) 独占 —
池外股票连模型预测都拿不到, 永远进不了短名单. 改为 池分×预期涨幅 (pred_ret_10d,
bundle dual_20260819 的 10d_reg) 混合入池. 案例: 300911 (+13.5% 10d 预测,
8/18 池分 rank 307/1499 在 N=200 定案时被切, 8/19 涨停).

名词查证 (2026-08-20, 详见报告):
  - 池分 = cleaning_pipeline.step2_liquidity 的 liquidity_score (0.5*rank_amount +
    0.5*rank_ff_turnover, rank_5050; cleaning_pipeline.py:204-298), 推理端每 date+board
    取前 liquidity_top_n=800 (CleaningConfig, cleaning_pipeline.py:103-105).
    池分只决定**成员资格** (谁有特征/预测), 从不参与短名单最终排名.
  - 短名单排名键: legacy = pred_ret_10d (list_generator._rank_by_magnitude,
    list_generator.py:416-429, TOP_N=15 无 cut 列); parallel = pred_mag_10d
    (_shortlist_t5_t10.rank_and_truncate, _shortlist_t5_t10.py:976-1012, cut=T-5;
    backtest.build_daily_shortlists top_ns=(5,10), backtest.py:1367/1397, cut=T-5/T-10).
  - "T-5/T-10/T-20 名额" = 每板块 top-5/top-10/top-20 短名单槽位 (sweep 脚本 TOPN=(5,10,20)
    同口径). 池分作用 = 纯成员资格: 池被切 (如 300911 8/18) → 无预测 → 任何 T-N 槽位都进不去.

变体 (250d OOS, 仅 dual 板块):
  (a) 基线生产: 流动性前 800 (run_inference 全链: step2 截断→step3→step4→E6) →
      特征(池内截面排名) → 10d_reg 预测 → 按 _pred_10d 排名 top10/top5 10d 实得.
  (b) 混合选池 (w ∈ {0.5, 0.7}): 全谱宇宙 (step2 apply_top_n=False, 不截断) → 特征 →
      预测 → blend = w*rank_pct(liquidity_score) + (1-w)*rank_pct(pred_ret_10d) →
      每 date 前 800 → E6 → 再按 pred_ret_10d 排名 top10/top5.
  (c) 池内重排 (w ∈ {0.5, 0.7}): 成员 = (a) 的池 (流动性前 800, 同特征同预测),
      短名单改用 blend 排名 → 量化"排名用混合分"效应; 若最终排名回 pred_ret_10d
      则与 (a) 逐位相同 = no-op 确认 (JSON 内附 topn_if_ranked_by_pred_10d 证据).
  (d) 控制组: 全谱特征 (同 b 的特征帧) + 流动性前 800 → E6 → pred 排名.
      b−a = 特征截面变大 + blend 换成员; d−a = 仅特征截面变大; b−d = blend 成员效应.

300911 个案: 8/18 + 8/19, 每变体是否入池 + 池内 pred/blend 排名 (能否进 top10),
另附全谱 trace (流动性排名/E6 成交额分位/blend 排名) 解释被切原因.

输出: BACKTEST_RESULT_DIR/pool_rank_mix_board_20260820_<ts>/result.json (WORM)
生产口径对照版: (b) 混合选池按 per-(date, board) 切前 800 (GEM/STAR 各 800, 同生产
run_inference 池结构), 修正 v1 (日期级合并切 800) 的口径混扰。
用法: python scripts/_replay_pool_rank_mix_20260820.py [--eval-days=250] [--weights=0.5,0.7] [--blend=sum|prod]

运行注意: 重计算 (面板+特征) — 严禁与 E6 sweep / 重训并行 (OOM). 特征帧为全谱
(~1800 只/日 vs 生产 800), 内存峰值高, 各阶段已 gc.collect().
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.pipeline1.cleaning_pipeline import (
    CleaningConfig,
    CleaningPipeline,
)
from app.pipeline1.dual_track_trainer import DualTrackTrainer
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.label_engine import MASK_RECENT_DAYS, LabelEngine
from config.settings import BACKTEST_RESULT_DIR, PANEL_V3_PATH

MODEL_DIR = "models/pipeline1"
BUNDLE = "dual_20260819.pkl"  # 当前生产 bundle (10d_reg 排名键)
FEATURE_WARMUP_DAYS = 270  # 特征滚动窗口暖机天数
EVAL_DAYS = 250
N_SUB = 4  # 子窗稳定性分段数
TOPNS = (5, 10)
WEIGHTS = (0.5, 0.7)  # w = 池分权重; blend = w*rank(池分) + (1-w)*rank(预期涨幅)
CASE_SYMBOL = "300911"
CASE_DTS = (pd.Timestamp("2026-08-18"), pd.Timestamp("2026-08-19"))


def ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9
    except Exception:
        return -1.0


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """标签 (与 _sweep_e6_for_N800 同口径): 路径标签 + 视界净收益 + 停牌/近期掩码."""
    df = LabelEngine.build_path_labels(df)
    df = LabelEngine.build_labels(df)
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=MASK_RECENT_DAYS)
    return df


def project_eval(
    df: pd.DataFrame, cols: list[str], labels: list[str], eval_days: int
) -> tuple[pd.DataFrame, list[str]]:
    """取末 eval_days 交易日 + 投影到模型列/标签/date/symbol/池分/成交额 (省内存)."""
    ddates = sorted(df["date"].unique())
    eval_start = ddates[-eval_days]
    eval_df = df[df["date"] >= eval_start].copy()
    keep = (
        [c for c in (cols + labels) if c in eval_df.columns]
        + ["date", "symbol"]
        + [c for c in ("liquidity_score", "amount", "board") if c in eval_df.columns]
    )
    missing = [c for c in cols if c not in eval_df.columns]
    return eval_df[keep].sort_values(["date", "symbol"]).reset_index(drop=True), missing


def predict_10d(eval_df: pd.DataFrame, models: dict, cols: list[str]) -> pd.DataFrame:
    """10d_reg 推理 → _pred_10d (排名键 = 生产 pred_ret_10d 原始模型输出)."""
    reg10, _ = models["10d_reg"]
    X = np.nan_to_num(eval_df[cols].values, nan=0.0)
    out = eval_df.copy()
    out["_pred_10d"] = reg10.predict(X)
    return out


def blend_score(liq: pd.Series, pred: pd.Series, w: float, mode: str) -> pd.Series:
    """混合分: sum = w*rank(池分)+(1-w)*rank(预期涨幅); prod = 几何加权."""
    if mode == "prod":
        return liq ** w * pred ** (1 - w)
    return w * liq + (1 - w) * pred


def cut_top_n(
    df: pd.DataFrame, key: str, n: int, group_cols: list[str] | None = None
) -> pd.DataFrame:
    """每 (date[, board]) 按 key 降序取前 n (池成员资格) — 生产口径 per-board 切池."""
    cols = group_cols or ["date", "board"]
    if not all(c in df.columns for c in cols):
        cols = ["date"]
    rk = df.groupby(cols)[key].rank(ascending=False, method="first")
    return df[rk <= n].copy()


def e6_cut(df: pd.DataFrame, pct: float) -> pd.DataFrame:
    """[E6] 每 date 成交额后 pct% 剔除 (镜像 cleaning_pipeline._drop_bottom)."""
    if pct <= 0:
        return df
    rk = df.groupby("date")["amount"].rank(pct=True)
    return df[rk > pct].copy()


def variant_metrics(
    eval_df: pd.DataFrame,
    models: dict,
    cols: list[str],
    n_sub: int,
    rank_key: str = "_pred_10d",
) -> dict:
    """TOP-N 10d 实得 (label_pm_10d_net): 全窗 mean/hit + 子窗 + wIC + 池规模."""
    lab = "label_pm_10d_net"
    sub = eval_df.dropna(subset=[lab]).copy()
    out: dict = {}
    for n in TOPNS:
        top = sub.sort_values(rank_key, ascending=False).groupby("date").head(n)
        out[f"10d_n{n}"] = {
            "mean_ret": float(top[lab].mean()) if len(top) else None,
            "hit": float((top[lab] > 0).mean()) if len(top) else None,
            "rows": int(len(top)),
        }
    dates = sorted(sub["date"].unique())
    step = len(dates) // n_sub
    for n in TOPNS:
        segs = []
        for i in range(n_sub):
            s0, s1 = i * step, len(dates) if i == n_sub - 1 else (i + 1) * step
            dsub = sub[sub["date"].isin(dates[s0:s1])]
            top = dsub.sort_values(rank_key, ascending=False).groupby("date").head(n)
            segs.append(float(top[lab].mean()) if len(top) else None)
        out[f"10d_n{n}_sub"] = segs
    trained = {"segs": {"test": eval_df}, "feature_cols": cols, "models": models}
    try:
        oos = DualTrackTrainer(model_dir=MODEL_DIR).validate_oos(trained)
        out["weighted_ic"] = oos["weighted_ic"]
    except Exception as exc:
        out["weighted_ic"] = None
        print(f"[warn] validate_oos 失败: {exc}", flush=True)
    out["pool_days"] = int(eval_df["date"].nunique())
    out["pool_rows"] = int(len(eval_df))
    return out


def case_report(
    pool_df: pd.DataFrame, blend_col: str | None = None
) -> dict:
    """CASE_SYMBOL (300911) 在某变体最终池内 8/18+8/19 的状态: 入池? pred/blend 排名?"""
    out = {}
    for dt in CASE_DTS:
        d = pool_df[pool_df["date"] == dt]
        r = d[d["symbol"] == CASE_SYMBOL]
        if r.empty:
            out[str(dt.date())] = {"in_pool": False}
            continue
        r = r.iloc[0]
        pred_rank = int((d["_pred_10d"] > r["_pred_10d"]).sum() + 1)
        rec = {
            "in_pool": True,
            "pred_ret_10d": float(r["_pred_10d"]),
            "pred_rank_in_pool": pred_rank,
            "in_top10": pred_rank <= 10,
            "in_top5": pred_rank <= 5,
        }
        if blend_col is not None and blend_col in d.columns:
            rec["blend_rank_in_pool"] = int((d[blend_col] > r[blend_col]).sum() + 1)
        out[str(dt.date())] = rec
    return out


def trace_300911(
    full_df: pd.DataFrame, n_pool: int, e6: float, weights: tuple, blend_mode: str
) -> dict:
    """全谱帧 (预切池) 上 300911 的逐规则归因: 池分排名 / E6 成交额分位 / blend 排名.

    解释每变体"入池或未入池"的精确原因 (N 截断 vs E6 成交额尾切 vs 清洗剔除).
    """
    out = {}
    for dt in CASE_DTS:
        d = full_df[full_df["date"] == dt].copy()
        if d.empty or (d["symbol"] == CASE_SYMBOL).sum() == 0:
            out[str(dt.date())] = {"in_full_after_clean": False}
            continue
        # 派生列全部先算, 再取 300911 行 (避免视图/索引错位)
        d["_r_liq"] = d["liquidity_score"].rank(ascending=False, method="first")
        r_pred = d["_pred_10d"].rank(pct=True)
        for w in weights:
            d[f"_blend{w}"] = blend_score(d["liquidity_score"], r_pred, w, blend_mode)
        r911 = d.loc[d["symbol"] == CASE_SYMBOL].iloc[0]
        rec = {
            "in_full_after_clean": True,
            "n_universe": int(len(d)),
            "amount": float(r911["amount"]),
            "liquidity_score": float(r911["liquidity_score"]),
            "liquidity_rank_in_universe": int(r911["_r_liq"]),
            "in_liquidity_top800": bool(r911["_r_liq"] <= n_pool),
        }
        liq_set = d[d["_r_liq"] <= n_pool]
        if len(liq_set):
            pct = liq_set["amount"].rank(pct=True).get(r911.name)
            rec["e6_amount_pct_in_liq800"] = None if pct is None else float(pct)
            rec["e6_cut_liq800"] = bool(
                rec["in_liquidity_top800"]
                and pct is not None
                and float(pct) <= e6
            )
        for w in weights:
            bkey = f"_blend{w}"
            brank = int((d[bkey] > r911[bkey]).sum() + 1)
            rec[f"blend{w}_rank_in_universe"] = brank
            rec[f"in_blend{w}_top800"] = brank <= n_pool
            bset = d[d[bkey].rank(ascending=False, method="first") <= n_pool]
            if len(bset):
                pct = bset["amount"].rank(pct=True).get(r911.name)
                rec[f"e6_amount_pct_in_blend{w}800"] = None if pct is None else float(pct)
                rec[f"e6_cut_blend{w}800"] = bool(
                    brank <= n_pool and pct is not None and float(pct) <= e6
                )
            # 生产口径: per-board 排名/切池 (GEM/STAR 各取前 n)
            if "board" in d.columns:
                g = d.groupby(["date", "board"])[bkey]
                brank_b = int(g.rank(ascending=False, method="first").get(r911.name))
                rec[f"blend{w}_rank_in_board"] = brank_b
                rec[f"in_blend{w}_board_top800"] = brank_b <= n_pool
        out[str(dt.date())] = rec
    return out


def main() -> int:
    import sys as _sys

    _eval_days = EVAL_DAYS
    _weights = WEIGHTS
    _blend_mode = "sum"
    for _a in _sys.argv[1:]:
        if _a.startswith("--eval-days="):
            _eval_days = int(_a.split("=", 1)[1])
        elif _a.startswith("--weights="):
            _weights = tuple(float(x) for x in _a.split("=", 1)[1].split(","))
        elif _a.startswith("--blend="):
            _blend_mode = _a.split("=", 1)[1].strip().lower()
    if _blend_mode not in ("sum", "prod"):
        print(f"[FATAL] --blend 必须 sum|prod, got {_blend_mode}", flush=True)
        return 3
    warmup_days = FEATURE_WARMUP_DAYS + _eval_days
    n_sub = max(2, _eval_days // 60)
    t0 = time.time()
    print(f"[ram] available {ram_gb():.1f} GB", flush=True)

    # ---- bundle 特征列 ----
    bundle = DualTrackTrainer.load(os.path.join(MODEL_DIR, BUNDLE))
    cols = list(bundle["feature_cols"])
    models = bundle["models"]
    labels = sorted({lbl for _, (_, lbl) in models.items()})
    print(f"[bundle] {BUNDLE} n_feats={len(cols)} labels={len(labels)}", flush=True)

    # ---- 面板: pyarrow 下推 (amount>=5000万 + date>=cutoff), 只留 dual ----
    cfg = CleaningConfig()
    n_pool = cfg.liquidity_top_n
    e6 = cfg.bottom_amount_pct
    # 去重取唯一交易日! 若不去重, iloc[-warmup_days] 取的是倒数第 N 行 (同一交易日内的行) → 面板只剩 1 天
    all_dates = pd.Series(
        sorted(pd.to_datetime(pq.read_table(str(PANEL_V3_PATH), columns=["date"])["date"]).unique())
    )
    assert len(all_dates) > warmup_days, f"panel 只有 {len(all_dates)} 个交易日, 不够 warmup {warmup_days}"
    cutoff = all_dates.iloc[-warmup_days]
    print(
        f"[panel] cutoff {pd.Timestamp(cutoff).date()} (last {warmup_days} trading days, "
        f"total {len(all_dates)} unique days)",
        flush=True,
    )
    del all_dates
    gc.collect()
    panel = pq.read_table(
        str(PANEL_V3_PATH),
        filters=[("amount", ">=", cfg.min_amount), ("date", ">=", cutoff)],
    ).to_pandas()
    panel = panel[panel["is_suspended"] == 0].reset_index(drop=True)
    # 本实验只涉 dual (main 不限池无此问题) → 提前剔 main 省内存
    panel = panel[panel["symbol"].astype(str).str.startswith(("30", "68"))].reset_index(
        drop=True
    )
    print(f"[panel] dual-only {len(panel):,}r", flush=True)
    # 防呆: 回放必须覆盖 >= eval_days 个交易日, 否则结果无意义
    n_days = panel["date"].nunique()
    assert n_days >= _eval_days, (
        f"回放面板只有 {n_days} 个交易日 < eval {_eval_days} — cutoff 推导有误, 终止"
    )
    print(f"[panel] eval 窗口交易日 {n_days} (>= {_eval_days}) OK", flush=True)

    features = FeatureEngineV35()
    cleaner = CleaningPipeline(cfg)
    results: dict = {}

    # ================= (a) 生产池帧 + (c) 池内重排 =================
    ta = time.time()
    main_b, dual_a, state = cleaner.run_inference(panel)
    del main_b
    gc.collect()
    if state == "empty":
        print("[FATAL] (a) valve empty → abort", flush=True)
        return 3
    df_a = features.build(dual_a, None, inference_cols=cols, cross_sectional_rank=True)
    del dual_a
    gc.collect()
    df_a = build_labels(df_a)
    eval_a, missing = project_eval(df_a, cols, labels, _eval_days)
    del df_a
    gc.collect()
    if missing:
        print(f"[FATAL] (a) 缺失 {missing[:5]} → abort", flush=True)
        return 3
    eval_a = predict_10d(eval_a, models, cols)
    results["a_liq800"] = variant_metrics(eval_a, models, cols, n_sub)
    results["a_liq800"]["case_300911"] = case_report(eval_a)
    r_a = results["a_liq800"]
    print(
        f"[a] 池 {eval_a['date'].nunique()}d/{len(eval_a):,}r "
        f"top10={r_a['10d_n10']['mean_ret']:+.3f}(hit {r_a['10d_n10']['hit']:.3f}) "
        f"top5={r_a['10d_n5']['mean_ret']:+.3f} wIC={r_a['weighted_ic']:.4f} "
        f"({time.time() - ta:.0f}s)",
        flush=True,
    )

    # (c) 池内重排: 同 (a) 的池, 短名单改用 blend 排名 (成员不变 → 特征/预测逐位相同)
    r_pred_a = eval_a.groupby("date")["_pred_10d"].rank(pct=True)
    for w in _weights:
        ev = eval_a.assign(_blend=blend_score(eval_a["liquidity_score"], r_pred_a, w, _blend_mode))
        results[f"c_rerank_w{w}"] = variant_metrics(ev, models, cols, n_sub, "_blend")
        results[f"c_rerank_w{w}"]["case_300911"] = case_report(ev, "_blend")
        # no-op 证据: 最终排名回 pred_ret_10d (生产口径) → 与 (a) 逐位相同
        pkey = variant_metrics(ev, models, cols, n_sub, "_pred_10d")
        results[f"c_rerank_w{w}"]["pool_identical_to_a"] = True
        results[f"c_rerank_w{w}"]["topn_if_ranked_by_pred_10d"] = {
            k: pkey[k] for k in ("10d_n5", "10d_n10")
        }
        rc = results[f"c_rerank_w{w}"]
        print(
            f"[c w={w}] 池内重排 blend 排名: top10={rc['10d_n10']['mean_ret']:+.3f} "
            f"(若按 pred 排名 = {rc['topn_if_ranked_by_pred_10d']['10d_n10']['mean_ret']:+.3f} = a)",
            flush=True,
        )
        del ev
        gc.collect()
    del eval_a
    gc.collect()

    # ================= (b) 混合选池 + (d) 控制组: 全谱帧 =================
    t_full = time.time()
    _, dual_full = CleaningPipeline.step0_board_split(panel, board="dual")
    del panel
    gc.collect()
    dual_full = dual_full.sort_values(["symbol", "date"]).reset_index(drop=True)
    dual_full = cleaner.step2_liquidity(dual_full, apply_top_n=False)  # 不截断: 全谱宇宙
    dual_full = cleaner.step3_extreme(dual_full)
    dual_full, state = cleaner.step4_tradability(dual_full, inference_only=True)
    if state == "empty":
        print("[FATAL] (full) valve empty → abort", flush=True)
        return 3
    df_full = features.build(dual_full, None, inference_cols=cols, cross_sectional_rank=True)
    del dual_full
    gc.collect()
    df_full = build_labels(df_full)
    eval_full, missing = project_eval(df_full, cols, labels, _eval_days)
    del df_full
    gc.collect()
    if missing:
        print(f"[FATAL] (full) 缺失 {missing[:5]} → abort", flush=True)
        return 3
    eval_full = predict_10d(eval_full, models, cols)
    print(
        f"[full] 全谱特征帧 eval {eval_full['date'].nunique()}d/{len(eval_full):,}r "
        f"({time.time() - t_full:.0f}s)",
        flush=True,
    )

    # (d) 控制: 全谱特征 + 流动性前 800/板块 → E6 (隔离"特征截面变大"效应)
    pool_d = e6_cut(
        cut_top_n(eval_full, "liquidity_score", n_pool, ["date", "board"]), e6
    )
    results["d_fullfeat_liq800"] = variant_metrics(pool_d, models, cols, n_sub)
    results["d_fullfeat_liq800"]["case_300911"] = case_report(pool_d)
    r_d = results["d_fullfeat_liq800"]
    print(
        f"[d] 全谱特征+流动性800: top10={r_d['10d_n10']['mean_ret']:+.3f} "
        f"(hit {r_d['10d_n10']['hit']:.3f}) wIC={r_d['weighted_ic']:.4f}",
        flush=True,
    )
    del pool_d
    gc.collect()

    # (b) 混合选池 (生产口径 per-board): blend 前 800/板块 → E6 → pred 排名
    r_pred_full = eval_full.groupby(["date", "board"])["_pred_10d"].rank(pct=True)
    for w in _weights:
        ef = eval_full.assign(
            _blend=blend_score(eval_full["liquidity_score"], r_pred_full, w, _blend_mode)
        )
        pool_b = e6_cut(
            cut_top_n(ef, "_blend", n_pool, ["date", "board"]), e6
        )
        results[f"b_blend_w{w}"] = variant_metrics(pool_b, models, cols, n_sub)
        results[f"b_blend_w{w}"]["case_300911"] = case_report(pool_b, "_blend")
        r_b = results[f"b_blend_w{w}"]
        print(
            f"[b w={w}] 混合选池: 池 {pool_b['date'].nunique()}d/{len(pool_b):,}r "
            f"top10={r_b['10d_n10']['mean_ret']:+.3f}(hit {r_b['10d_n10']['hit']:.3f}) "
            f"top5={r_b['10d_n5']['mean_ret']:+.3f} wIC={r_b['weighted_ic']:.4f} "
            f"sub={[round(x, 4) for x in r_b['10d_n10_sub'] if x is not None]}",
            flush=True,
        )
        del ef, pool_b
        gc.collect()

    results["_case_trace_300911"] = trace_300911(
        eval_full, n_pool, e6, _weights, _blend_mode
    )
    print("[case] 300911 全谱 trace:")
    for dt, rec in results["_case_trace_300911"].items():
        print(f"  {dt}: {json.dumps(rec, ensure_ascii=False, default=str)}", flush=True)
    del eval_full
    gc.collect()

    # ================= 落盘 (WORM) =================
    results["_meta"] = {
        "bundle": BUNDLE,
        "eval_days": _eval_days,
        "warmup_days": FEATURE_WARMUP_DAYS,
        "weights": list(_weights),
        "blend_mode": _blend_mode,
        "n_pool_dual": n_pool,
        "e6_bottom_amount_pct_dual": e6,
        "n_sub": n_sub,
        "topns": list(TOPNS),
        "case_symbol": CASE_SYMBOL,
        "panel_days": int(warmup_days),
        "elapsed_sec": int(time.time() - t0),
    }
    out_dir = Path(BACKTEST_RESULT_DIR) / (
        f"pool_rank_mix_board_20260820_{_eval_days}d_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\n[done] 结果 WORM -> {out_dir} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
