"""诊断: 输出级预测平滑 (EMA, 2026-08-06 四模块稳定化) 是否有正面效果的伪回测.

问题: 同一只股票相邻交易日预测(预期涨幅/达到概率)剧烈变化 → 加了 Layer2 输出级 EMA.
这里验证: 平滑是否真的让预测**更好** (而非只是更稳).

方法 (walk-forward 固定模型, 无 look-ahead):
  1. 用今日模型 (current_meta) + 真实 cleaner+feature_engine 在本地面板上重建过去 N 个
     交易日 (日期 D 只取 date==D 的特征行 → 因果, 不引用未来).
  2. raw = 逐日原始预测; smoothed = 用生产 pred_smoothing.smooth_preds (历史底稿逐日累加).
  3. realized = close_hfq 研究口径 (ret_{h}d = close[T+h]/close-1; up_{h}d = ret>0.5%).
  4. 对比 raw vs smoothed 的: 逐日 IC / 方向准确率 / MAE / top-20% 命中 / Brier /
     稳定度 (相邻日 |Δ|) / 清单 top-10 换手率.
  5. 两个范围: 全候选 vs "过去一周短名单" 股票.

说明: 绝对准确率与当时生产 (旧模型) 不同; 但 raw-vs-smoothed 用同一预测流对比, 公平.

结果 WORM → {BACKTEST_RESULT_DIR}/smoothing_backtest_<ts>/.
用法: python scripts/_diag_smoothing_backtest.py [window_days=10]
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.feature_engine_v35 import FeatureEngineV35
from app.pipeline1.predictor import V35Predictor
from config.settings import BACKTEST_RESULT_DIR, PANEL_V3_PATH

WINDOW = int(sys.argv[1]) if len(sys.argv) > 1 else 10
CLS_THRESHOLD = 0.005  # 与 label_engine.CLS_THRESHOLD 一致 (净收益覆盖成本的 +0.5%)
BUNDLES = {
    "main": "models/pipeline1/main_current.pkl",
    "dual": "models/pipeline1/dual_current.pkl",
}
HORIZONS = (1, 2, 3, 5)


def collect_shortlist_union() -> set[str]:
    """过去一周交付清单 (legacy + parallel + data/lists) 的 symbol 并集."""
    syms: set[str] = set()
    roots = {
        "stocklist": Path("D:/AMINQT/DAILY OPERATION/STOCK LIST"),
        "lists": Path("D:/AMINQT/AMINQT CODES/data/lists"),
    }
    pats = ["legacy_stocklist_2026080*.csv", "STOCK LIST 2026080*.xlsx",
            "parallel_shortlist_2026080*.csv", "list_2026080*.parquet"]
    for root in roots.values():
        for pat in pats:
            for fp in sorted(root.glob(pat)):
                try:
                    if fp.suffix == ".csv":
                        df = pd.read_csv(fp, dtype={"symbol": str})
                    elif fp.suffix == ".xlsx":
                        df = pd.read_excel(fp, dtype={"symbol": str})
                    else:
                        df = pd.read_parquet(fp)
                    if "symbol" in df.columns:
                        syms |= {str(s) for s in df["symbol"].dropna().tolist()}
                except Exception as e:  # noqa: BLE001
                    print(f"  [skip] {fp.name}: {e}")
    return syms


def realized_from_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """研究口径已实现收益/上涨 (close_hfq, groupby symbol)."""
    out = panel[["symbol", "date", "close_hfq"]].sort_values(["symbol", "date"]).copy()
    g = out.groupby("symbol")["close_hfq"]
    for h in HORIZONS:
        out[f"ret_{h}d"] = g.shift(-h) / out["close_hfq"] - 1
        out[f"up_{h}d"] = (out[f"ret_{h}d"] > CLS_THRESHOLD).astype(float)
    return out[["symbol", "date"] + [f"ret_{h}d" for h in HORIZONS] + [f"up_{h}d" for h in HORIZONS]]


def daily_ic(df: pd.DataFrame, pred: str, real: str, min_n: int = 5) -> float:
    vals = []
    for _d, g in df.dropna(subset=[pred, real]).groupby("date"):
        if len(g) >= min_n and g[real].nunique() > 1:
            r = spearmanr(g[pred], g[real])
            if r.statistic == r.statistic:
                vals.append(r.statistic)
    return float(np.mean(vals)) if vals else float("nan")


def direction_acc(df: pd.DataFrame, pred: str, real: str) -> float:
    d = df.dropna(subset=[pred, real])
    d = d[d[real].abs() > 1e-9]
    return float((np.sign(d[pred]) == np.sign(d[real])).mean()) if len(d) else float("nan")


def mae(df: pd.DataFrame, pred: str, real: str) -> float:
    d = df.dropna(subset=[pred, real])
    return float((d[pred] - d[real]).abs().mean()) if len(d) else float("nan")


def brier(df: pd.DataFrame, prob: str, up: str) -> float:
    d = df.dropna(subset=[prob, up])
    return float(((d[prob] - d[up]) ** 2).mean()) if len(d) else float("nan")


def top20_alpha(df: pd.DataFrame, pred: str, real: str, q: float = 0.2) -> dict:
    """每日按 pred 降序取 top q 分位的平均已实现收益 vs 当日全池平均 → 平均 alpha."""
    alphas, baselines, n_obs = [], [], 0
    for _d, g in df.dropna(subset=[pred, real]).groupby("date"):
        if len(g) < 10:
            continue
        thr = g[pred].quantile(1 - q)
        top = g[g[pred] >= thr]
        base = g[real].mean()
        alphas.append(top[real].mean() - base)
        baselines.append(base)
        n_obs += len(top)
    return {
        "alpha_top20": float(np.mean(alphas)) if alphas else float("nan"),
        "baseline": float(np.mean(baselines)) if baselines else float("nan"),
        "n_top_obs": n_obs,
    }


def topN_churn(scores: pd.DataFrame, n: int = 10) -> float:
    """相邻日 top-N (按 score 降序) 换手率: 1 - |prev∩cur|/N, 越小越稳."""
    days = sorted(scores["date"].unique())
    churns = []
    for a, b in zip(days, days[1:], strict=False):
        ta = set(scores[scores["date"] == a].sort_values("score", ascending=False).head(n)["symbol"])
        tb = set(scores[scores["date"] == b].sort_values("score", ascending=False).head(n)["symbol"])
        churns.append(1 - len(ta & tb) / n)
    return float(np.mean(churns)) if churns else float("nan")


def d3_rank_score(df: pd.DataFrame) -> pd.DataFrame:
    """legacy 排名口径: 0.5*norm(pred_ret_3d) + 0.5*norm(prob_up_3d) (当日截面 min-max)."""
    g = df.groupby("date")
    df["score"] = 0.5 * g["pred_ret_3d"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min() + 1e-12)
    ) + 0.5 * g["prob_up_3d"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min() + 1e-12)
    )
    return df


def main() -> None:
    t0 = time.time()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = BACKTEST_RESULT_DIR / f"smoothing_backtest_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[universe] 收集过去一周短名单股票...", flush=True)
    shortlist_union = collect_shortlist_union()
    print(f"[universe] 短名单并集 {len(shortlist_union)} 只: {sorted(shortlist_union)[:20]}...", flush=True)

    print("[panel] 加载面板...", flush=True)
    panel = pd.read_parquet(str(PANEL_V3_PATH))
    panel["date"] = pd.to_datetime(panel["date"])
    all_dates = sorted(panel["date"].unique())
    cut = all_dates[-300]
    panel = panel[panel["date"] >= cut].reset_index(drop=True)
    print(f"[panel] {len(panel):,}r  {cut.date()}..{all_dates[-1].date()} ({time.time()-t0:.0f}s)", flush=True)

    # 真实 cleaner + 特征 (一次构建, 逐日因果切片)
    cleaner = CleaningPipeline()
    main_df, dual_df, _valve = cleaner.run_inference(panel)
    predictor = V35Predictor(BUNDLES)
    feats: dict[str, pd.DataFrame] = {}
    for board, df in (("main", main_df), ("dual", dual_df)):
        if len(df) == 0 or board not in predictor.bundles:
            continue
        cols = predictor.bundles[board]["feature_cols"]
        feats[board] = FeatureEngineV35().build(
            df, None, inference_cols=cols, cross_sectional_rank=(board == "dual")
        )
        print(f"[feat] {board}: {len(feats[board]):,}r ({time.time()-t0:.0f}s)", flush=True)

    # 逐日推理 → raw 长表
    days = [d for d in all_dates if d >= all_dates[-WINDOW]]
    raw_frames = []
    for D in days:
        frames = []
        for board, feat in feats.items():
            fD = feat[feat["date"] == D]
            if len(fD) == 0:
                continue
            fD = fD.sort_values("date").groupby("symbol").tail(1)
            frames.append(predictor.predict(fD, board))
        if not frames:
            continue
        day_raw = pd.concat(frames, ignore_index=True)
        day_raw["date"] = D
        raw_frames.append(day_raw)
    raw = pd.concat(raw_frames, ignore_index=True)
    print(f"[raw] {len(raw):,} 预测行 ({len(raw['date'].unique())} 交易日) ({time.time()-t0:.0f}s)", flush=True)

    # 生产平滑: 临时底稿目录, 逐日 persist→smooth (与 daily_pipeline 同语义)
    from app.pipeline1 import pred_smoothing
    from app.pipeline1.model_meta import load_modules, module_id

    mod = module_id(load_modules())
    tmpdir = tempfile.mkdtemp(prefix="smooth_bt_")
    pred_smoothing.STOCK_LIST_DIR = Path(tmpdir)
    smooth_frames = []
    for D in days:
        day_raw_d = raw[raw["date"] == D]
        if len(day_raw_d) == 0:
            continue
        dstr = D.strftime("%Y%m%d")
        pred_smoothing.persist_raw_preds(day_raw_d, dstr, mod)
        sm = pred_smoothing.smooth_preds(day_raw_d, dstr, mod)
        sm["date"] = D
        smooth_frames.append(sm)
    smooth = pd.concat(smooth_frames, ignore_index=True)
    print(f"[smooth] 完成 ({time.time()-t0:.0f}s)", flush=True)

    # 合并已实现
    realized = realized_from_panel(panel)
    raw = raw.merge(realized, on=["symbol", "date"], how="left")
    smooth = smooth.merge(realized, on=["symbol", "date"], how="left")

    # 范围
    scopes = {
        "full": raw,
        "shortlist": raw[raw["symbol"].isin(shortlist_union)],
    }

    summary: dict = {"ts": ts, "window_days": WINDOW, "module": mod,
                     "n_days": len(days), "shortlist_union_n": len(shortlist_union)}
    rows = []
    for scope_name, sdf in scopes.items():
        smdf = smooth[smooth["symbol"].isin(set(sdf["symbol"]))] if scope_name == "shortlist" else smooth
        smdf = smdf[smdf["date"].isin(set(sdf["date"]))]
        for h in HORIZONS:
            pred = f"pred_ret_{h}d"
            prob = f"prob_up_{h}d" if h != 1 else "prob_up"
            up = f"up_{h}d"
            real = f"ret_{h}d"
            for tag, df in (("raw", sdf), ("smooth", smdf)):
                row = {
                    "scope": scope_name, "horizon": f"{h}d", "tag": tag,
                    "n": int(df.dropna(subset=[real]).shape[0]),
                    "ic": daily_ic(df, pred, real),
                    "dir_acc": direction_acc(df, pred, real),
                    "mae": mae(df, pred, real),
                    "brier": brier(df, prob, up),
                    "ic_prob": daily_ic(df, prob, up),
                    **top20_alpha(df, pred, real),
                }
                rows.append(row)
    summary["metrics"] = rows
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False)

    # 稳定度: 相邻日 |Δ| (同 symbol), raw vs smooth
    stab_cols = [f"pred_ret_{h}d" for h in HORIZONS] + ["prob_up", "prob_up_2d", "prob_up_3d", "prob_up_5d"]
    stab_rows = []
    for tag, df in (("raw", raw), ("smooth", smooth)):
        d0 = df[["symbol", "date"]].drop_duplicates()
        for col in stab_cols:
            d = d0.merge(df[["symbol", "date", col]], on=["symbol", "date"], how="left").dropna()
            d = d.sort_values(["symbol", "date"])
            d["prev"] = d.groupby("symbol")[col].shift(1)
            d = d.dropna(subset=["prev"])
            delta = (d[col] - d["prev"]).abs()
            stab_rows.append({
                "tag": tag, "col": col,
                "mean_abs_delta": float(delta.mean()) if len(delta) else float("nan"),
                "median_abs_delta": float(delta.median()) if len(delta) else float("nan"),
                "p90_abs_delta": float(delta.quantile(0.9)) if len(delta) else float("nan"),
                "n_pairs": int(len(delta)),
            })
    summary["stability"] = stab_rows
    pd.DataFrame(stab_rows).to_csv(out_dir / "stability.csv", index=False)

    # 清单 top-10 换手率 (d3 排名口径)
    churn = {}
    raw_sc = d3_rank_score(raw).dropna(subset=["pred_ret_3d", "prob_up_3d"])
    sm_sc = d3_rank_score(smooth).dropna(subset=["pred_ret_3d", "prob_up_3d"])
    for tag, sdf in (("raw", raw_sc), ("smooth", sm_sc)):
        churn[tag] = topN_churn(sdf, n=10)
    summary["top10_churn"] = churn

    # 交付股票的 before/after 明细 (过去一周短名单 ∪ 窗口内)
    deliv = raw[raw["symbol"].isin(shortlist_union)][
        ["date", "symbol", "board", "pred_ret_3d", "prob_up_3d", "ret_3d", "up_3d"]
    ].merge(
        smooth[["date", "symbol", "pred_ret_3d", "prob_up_3d"]],
        on=["date", "symbol"], suffixes=("_raw", "_sm"),
    ).sort_values(["symbol", "date"])
    deliv.to_csv(out_dir / "delivered_before_after.csv", index=False)

    summary["out_dir"] = str(out_dir)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # 打印
    print("\n" + "=" * 78)
    print(f"[结果] {out_dir}")
    print(f"窗口 {WINDOW} 交易日 | 模块 {mod} | 短名单并集 {len(shortlist_union)} 只")
    m = pd.DataFrame(rows)
    for scope in ("full", "shortlist"):
        sub = m[m["scope"] == scope]
        print(f"\n--- 范围: {scope} ---")
        for h in HORIZONS:
            hr = sub[sub["horizon"] == f"{h}d"]
            if hr.empty:
                continue
            r_, sm_ = hr[hr["tag"] == "raw"].iloc[0], hr[hr["tag"] == "smooth"].iloc[0]
            print(
                f"  {h}d:  IC raw={r_['ic']:+.4f} sm={sm_['ic']:+.4f} | "
                f"dirAcc raw={r_['dir_acc']:.4f} sm={sm_['dir_acc']:.4f} | "
                f"MAE raw={r_['mae']:.4f} sm={sm_['mae']:.4f} | "
                f"Brier(raw={r_['brier']:.4f}/sm={sm_['brier']:.4f}) | "
                f"top20α raw={r_['alpha_top20']:+.4f} sm={sm_['alpha_top20']:+.4f}"
            )
    print("\n--- 稳定度 (相邻日 |Δ| 均值, 越低越稳) ---")
    for s in stab_rows:
        print(f"  {s['tag']:6s} {s['col']:14s} meanΔ={s['mean_abs_delta']:.5f} "
              f"p90Δ={s['p90_abs_delta']:.5f} (n={s['n_pairs']})")
    print(f"\n--- 清单 top-10 换手率 (越低越稳): raw={churn.get('raw', float('nan')):.3f} "
          f"smooth={churn.get('smooth', float('nan')):.3f}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
