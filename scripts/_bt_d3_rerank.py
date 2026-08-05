# -*- coding: utf-8 -*-
"""d3 目标重排序 vs 旧 score 排序 — OOS 回测验证 (2026-08-05 用户请求).

问题: legacy 清单 2026-08-05 起改按 d3 目标排序
    _rank_by_d3_target = 0.5×norm(pred_ret_3d) + 0.5×norm(prob_up_3d)
用户要求用最近 3m/1m 回测证明: 排前面的股票是否真的有更高涨幅 + 概率.

方法 (走查 OOS, 全部在 current model 的 held-out test 段内, 训练未见过标签):
  窗口 = 模型 test 段 = 最后 60 交易日 (split_window: train→es(20)→calib(20)→test(60)).
  252 交易日暖机 (引擎最长回看) → 每交易日:
    cleaner.run_inference → FeatureEngineV35.build(inference_cols) →
    V35Predictor.predict (current bundle) → ListGenerator.compute_scores +
    entry_filter (E7 闸) → 同批入选股分别按 score 与 d3 blend 排序 → top-N.
  实测: 从 close_hfq 计算 t→t+k (k=2/3/5) 收益、T+1 入场收益、MFE, 命中率.

明确不含 (对比两排序内部一致, 不影响结论方向):
  FINAL STOCK SCAN (大宗/解禁), D18 大盘空仓触发 (env=None), 交易成本未扣,
  holding bonus (昨日清单回填) 关闭.

输出: data/_bt_d3_rerank_{ts}.json + admitted parquet (WORM).

用法: python scripts/_bt_d3_rerank.py [--force]   (--force 重算特征缓存)
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.pipeline1.cleaning_pipeline import CleaningPipeline, board_of  # noqa: E402
from app.pipeline1.dual_track_trainer import DualTrackTrainer  # noqa: E402
from app.pipeline1.feature_engine_v35 import FeatureEngineV35  # noqa: E402
from app.pipeline1.list_generator import ListGenerator  # noqa: E402
from app.pipeline1.predictor import V35Predictor  # noqa: E402
from config.settings import PANEL_V3_PATH  # noqa: E402

PANEL = str(PANEL_V3_PATH)
MODEL_DIR = os.path.join(ROOT, "models", "pipeline1")
WARMUP_DAYS = 252  # 引擎最长回看窗口 (年线特征)
TEST_DAYS = 60  # 模型 held-out test 段长度
CACHE_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
# 特征缓存用稳定名 (面板/模型不变即可复用, 避免重复 20 分钟构建)
FEAT_MAIN_CACHE = os.path.join(ROOT, "data", "_bt_d3_rerank_feat_main.parquet")
FEAT_DUAL_CACHE = os.path.join(ROOT, "data", "_bt_d3_rerank_feat_dual.parquet")
ADMITTED_CACHE = os.path.join(ROOT, "data", f"_bt_d3_rerank_admitted_{CACHE_TS}.parquet")
OUT_JSON = os.path.join(ROOT, "data", f"_bt_d3_rerank_{CACHE_TS}.json")

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes("float64").columns:
        df[c] = df[c].astype("float32")
    return df


def _mem(label: str) -> None:
    try:
        import psutil

        print(
            f"  [mem] {label}: RSS={psutil.Process().memory_info().rss / 1e9:.2f}GB",
            flush=True,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 实测收益 (从 close_hfq, 清洗后连续非停牌 bar)
# ---------------------------------------------------------------------------
def realized_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    sym = df["symbol"]
    chfq = df["close_hfq"]
    out = pd.DataFrame({"symbol": sym, "date": df["date"]})
    shifted = {k: chfq.groupby(sym).shift(-k) for k in range(1, 7)}
    t1, t2, t3, t4, t5, t6 = (shifted[k] for k in range(1, 7))
    out["c2c_2"] = t2 / chfq.values - 1
    out["c2c_3"] = t3 / chfq.values - 1
    out["c2c_5"] = t5 / chfq.values - 1
    # T+1 入场 (close_{t+1} 代理 price_1455), label_pm_kd = close_{t+1+k}/entry - 1
    out["exec_2"] = t3 / t1 - 1
    out["exec_3"] = t4 / t1 - 1
    out["exec_5"] = t6 / t1 - 1
    out["mfe3"] = pd.concat([t1, t2, t3, t4], axis=1).max(axis=1).values / t1.values - 1
    out["mfe5"] = (
        pd.concat([t1, t2, t3, t4, t5, t6], axis=1).max(axis=1).values / t1.values - 1
    )
    return out


# ---------------------------------------------------------------------------
# 1. 面板 → 清洗 (warmup 起点起) → 实测收益 → 特征 (每板块一次, 缓存到磁盘)
# ---------------------------------------------------------------------------
def prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.DataFrame]:
    force = "--force" in sys.argv
    bundle_cols = {}
    for board in ("main", "dual"):
        b = DualTrackTrainer.load(os.path.join(MODEL_DIR, f"{board}_current.pkl"))
        bundle_cols[board] = list(b["feature_cols"])
    log(f"bundle feature_cols: main={len(bundle_cols['main'])} dual={len(bundle_cols['dual'])}")

    panel = pd.read_parquet(PANEL)
    panel["date"] = pd.to_datetime(panel["date"])
    dates = np.array(sorted(panel["date"].unique()))
    test_start = pd.Timestamp(dates[-TEST_DAYS])
    warmup_start = pd.Timestamp(dates[-TEST_DAYS - WARMUP_DAYS])
    log(
        f"panel {len(dates)} 日, test 段={test_start.date()}..{dates[-1].date()}, "
        f"warmup 起点={warmup_start.date()}"
    )

    base = panel[panel["date"] >= warmup_start].copy()
    del panel
    gc.collect()
    # 实测收益必须从 raw 密集面板算 (清洗后只剩 ~480 流动性股/日, shift 会跳过缺日错位+虚高)
    log("计算实测收益 (raw 面板, 避免清洗稀疏面板 shift 错位)...")
    realized = realized_frame(base)
    log(f"实测收益帧: {len(realized)} rows")
    _mem("realized")

    log("清洗 (run_inference, 每板块 top-N + 8000万安全阀 + 后20% 剔除)...")
    cleaner = CleaningPipeline()
    main_df, dual_df, _valve = cleaner.run_inference(base)
    del base
    gc.collect()
    log(f"清洗后: main rows={len(main_df)} dual rows={len(dual_df)}")
    _mem("cleaned")

    features = {}
    for board, df, cache, cross_rank in (
        ("main", main_df, FEAT_MAIN_CACHE, False),
        ("dual", dual_df, FEAT_DUAL_CACHE, True),
    ):
        if not force and os.path.exists(cache):
            feat = pd.read_parquet(cache)
            log(f"[{board}] 特征缓存命中: {len(feat)} rows")
            features[board] = feat
            continue
        t0 = time.time()
        feat = FeatureEngineV35().build(
            df,
            None,
            inference_cols=bundle_cols[board],
            cross_sectional_rank=cross_rank,
        )
        log(f"[{board}] 特征构建 {time.time() - t0:.1f}s -> {len(feat)} rows")
        gc.collect()
        missing = [c for c in bundle_cols[board] if c not in feat.columns]
        if missing:
            log(f"[{board}] 特征缺失 {len(missing)} 列补0: {missing[:6]}")
            for c in missing:
                feat[c] = 0.0
        for col in ("industry", "close_hfq", "close", "amount", "is_suspended"):
            if col not in feat.columns and col in df.columns:
                feat = feat.merge(
                    df[["symbol", "date", col]], on=["symbol", "date"], how="left"
                )
        keep = (
            ["symbol", "date", "board"]
            + [c for c in bundle_cols[board] if c in feat.columns]
            + ["industry", "close_hfq", "close", "amount", "is_suspended"]
        )
        keep = [c for c in keep if c in feat.columns]
        feat = feat[keep].copy()
        _downcast(feat)
        feat = feat[feat["date"] >= test_start].reset_index(drop=True)
        feat.to_parquet(cache, index=False)
        log(f"[{board}] 特征落盘({len(feat)} rows, {len(feat.columns)} cols) -> {os.path.basename(cache)}")
        _mem(board)
        features[board] = feat
    del main_df, dual_df
    gc.collect()
    return features["main"], features["dual"], test_start, realized


# ---------------------------------------------------------------------------
# 2. 逐日预测 + E7 准入 + 双排序
# ---------------------------------------------------------------------------
def run_days(feat_main, feat_dual, test_start, realized) -> pd.DataFrame:
    bundles = {}
    for board in ("main", "dual"):
        p = os.path.join(MODEL_DIR, f"{board}_current.pkl")
        if os.path.exists(p):
            bundles[board] = p
    predictor = V35Predictor(bundles)
    lister = ListGenerator()
    dates = np.array(sorted(realized["date"].unique()))
    scored = dates[dates >= test_start][:-3]  # t+3 需未来 3 个 bar
    log(f"逐日打分: {len(scored)} 日 = {pd.Timestamp(scored[0]).date()}..{pd.Timestamp(scored[-1]).date()}")

    rows = []
    days_with_list = 0
    for i, date in enumerate(scored):
        cand_frames = []
        for board, feat in (("main", feat_main), ("dual", feat_dual)):
            today = feat[feat["date"] == date].copy()
            if len(today) == 0 or board not in predictor.bundles:
                continue
            # predictor.predict 的 keep 需要 board 列 (特征缓存可能缺列)
            today["board"] = (
                "main" if board == "main" else today["symbol"].map(board_of)
            )
            try:
                cand_frames.append(predictor.predict(today, board))
            except Exception as exc:  # 单日单板块失败不致命
                log(f"[{pd.Timestamp(date).date()} {board}] 预测失败: {exc}")
        if not cand_frames:
            continue
        cands = pd.concat(cand_frames, ignore_index=True)
        cands = V35Predictor.mark_yesterday_list(cands, None)
        scored_ = lister.compute_scores(cands)
        passed = lister.entry_filter(scored_)
        if len(passed) == 0:
            continue
        days_with_list += 1
        passed = passed.copy()
        g, p = passed["pred_ret_3d"], passed["prob_up_3d"]
        blend = 0.5 * (g - g.min()) / (g.max() - g.min()) + 0.5 * (p - p.min()) / (
            p.max() - p.min()
        )
        passed["d3_blend"] = blend
        passed["rank_score"] = passed["score"].rank(ascending=False, method="first")
        passed["rank_d3"] = blend.rank(ascending=False, method="first")
        cols = [
            "symbol", "board", "score", "d3_blend",
            "pred_ret_2d", "prob_up_2d",
            "pred_ret_3d", "prob_up_3d",
            "pred_ret_5d", "prob_up_5d",
            "rank_score", "rank_d3",
        ]
        keep = [c for c in cols if c in passed.columns]
        rec = passed[keep].assign(date=date).copy()
        if "industry" in passed.columns:
            rec["industry"] = passed["industry"].values
        rows.append(rec)
    if not rows:
        raise SystemExit("无任何日过 E7 准入")
    admitted = pd.concat(rows, ignore_index=True)
    admitted = admitted.merge(realized, on=["symbol", "date"], how="left")
    log(
        f"E7 入选汇总: {len(admitted)} 行 / {admitted['date'].nunique()} 日 "
        f"(过闸日 {days_with_list}/{len(scored)})"
    )
    return admitted


# ---------------------------------------------------------------------------
# 3. 汇总对比
# ---------------------------------------------------------------------------
def _metrics(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n == 0:
        return {"n": 0}
    return {
        "n": int(n),
        "mean_c2c3": float(sub["c2c_3"].mean()),
        "mean_c2c5": float(sub["c2c_5"].mean()),
        "mean_exec3": float(sub["exec_3"].mean()),
        "mean_mfe3": float(sub["mfe3"].mean()),
        "hit3": float((sub["c2c_3"] > 0).mean()),
        "hit5": float((sub["c2c_5"] > 0).mean()),
        "hit_exec3": float((sub["exec_3"] > 0).mean()),
    }


def _per_ranking(admitted: pd.DataFrame) -> dict:
    out = {}
    for ranking in ("score", "d3"):
        rk = f"rank_{ranking}"
        r = {f"top{N}": _metrics(admitted[admitted[rk] <= N]) for N in (5, 10, 15)}
        r["all"] = _metrics(admitted)
        buckets = {}
        for lo, hi in ((1, 5), (6, 10), (11, 15)):
            buckets[f"{lo}-{hi}"] = _metrics(
                admitted[(admitted[rk] >= lo) & (admitted[rk] <= hi)]
            )
        r["buckets"] = buckets
        out[ranking] = r
    return out


def summarize(admitted: pd.DataFrame, test_start: pd.Timestamp) -> dict:
    res: dict = {
        "window_start": str(test_start.date()),
        "model": "20260805_q2345 (current)",
    }
    res.update(_per_ranking(admitted))

    ok = admitted.dropna(subset=["c2c_3"])
    from scipy.stats import spearmanr, pearsonr

    res["corr"] = {}
    for pred_col in ("rank_score", "rank_d3", "pred_ret_3d", "prob_up_3d", "d3_blend"):
        if pred_col not in ok.columns:
            continue
        rho, pv = spearmanr(ok[pred_col], ok["c2c_3"])
        pr, _ = pearsonr(ok[pred_col], ok["c2c_3"])
        res["corr"][pred_col] = {
            "spearman_rho": float(rho),
            "p": float(pv),
            "pearson_r": float(pr),
            "n": int(len(ok)),
        }

    # 1m 子窗口 (最后 ≤21 交易日, 同样 clean OOS)
    dates = np.array(sorted(admitted["date"].unique()))
    m1_n = min(21, len(dates))
    m1 = admitted[admitted["date"] >= dates[-m1_n]]
    res["m1"] = {
        "window_start": str(pd.Timestamp(dates[-m1_n]).date()),
        "n_days": int(m1["date"].nunique()),
        "n_picks": int(len(m1)),
    }
    res["m1"].update(_per_ranking(m1))

    res["n_days"] = int(admitted["date"].nunique())
    res["n_picks"] = int(len(admitted))
    res["caveats"] = (
        "不含 FINAL STOCK SCAN / D18 空仓 / 交易成本 / holding bonus; "
        "exec 用 close_{t+1} 代理 price_1455; 停牌日剔除"
    )
    return res


def _pp(v) -> str:
    return "     -" if v is None or pd.isna(v) else f"{v * 100:8.2f}%"


def print_table(res: dict, tag: str) -> None:
    print("\n" + "=" * 100)
    print(f"{tag.upper()} — d3 目标排序 vs 旧 score 排序 (模型 test 段 OOS, {res['n_days']} 日 {res['n_picks']} 入选)")
    print("=" * 100)
    print(
        f"{'':>10s} {'n':>5s} {'mean d3':>9s} {'d3 hit':>8s} {'mean d5':>9s} {'d5 hit':>8s} "
        f"{'mean exec3':>11s} {'mean MFE3':>11s}"
    )
    for N in (5, 10, 15):
        for ranking in ("score", "d3"):
            m = res[ranking][f"top{N}"]
            if not m.get("n"):
                print(f"{'top'+str(N):>7s} {ranking:>3s}    (无数据)")
                continue
            print(
                f"{'top'+str(N):>7s} {ranking:>3s} {m['n']:>5d} "
                f"{_pp(m['mean_c2c3'])} {_pp(m['hit3'])} {_pp(m['mean_c2c5'])} {_pp(m['hit5'])} "
                f"{_pp(m['mean_exec3'])} {_pp(m['mean_mfe3'])}"
            )
        print("-" * 100)
    print("\n分桶 (实测 d3, 按预测序):")
    for ranking in ("score", "d3"):
        parts = []
        for k, m in res[ranking]["buckets"].items():
            parts.append(
                f"{k}=[n{m['n']} m:{m['mean_c2c3'] * 100:+.2f}% h:{m['hit3'] * 100:.0f}%]"
                if m.get("n")
                else f"{k}=[]"
            )
        print(f"  {ranking:>5s}: " + "  ".join(parts))
    if "corr" in res:
        print(f"\n预测序 vs 实测 d3 收益 (Spearman, n={res['corr'].get('rank_score', {}).get('n', 0)}):")
        for pc, v in res["corr"].items():
            print(f"  {pc:>14s}: rho={v['spearman_rho']:+.4f} (p={v['p']:.4f})")


def main() -> None:
    feat_main, feat_dual, test_start, realized = prepare()
    _mem("features")
    admitted = run_days(feat_main, feat_dual, test_start, realized)
    admitted.to_parquet(ADMITTED_CACHE, index=False)
    log(f"admitted parquet -> {os.path.basename(ADMITTED_CACHE)}")

    res = summarize(admitted, test_start)
    res["generated_at"] = CACHE_TS
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2, default=float)
    log(f"JSON -> {os.path.basename(OUT_JSON)}")
    print_table(res, "3m")
    print_table(res["m1"], "1m")


if __name__ == "__main__":
    main()
