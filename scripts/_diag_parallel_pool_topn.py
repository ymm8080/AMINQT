"""诊断: 并行管线 全池 vs 每日 top-N 短名单 回测 (2026-08-07 用户).

用户: "paraller i need to backtest on all stocks vs topn stock list to check
      1. whether better stock will be picked up, secondly whether the prediction
      and probability will be better"
后续: "if it is effective i might apply to all stocks in legacy and paraller
      modules and impact the stock ranking"

并行管线 = 特征池全池打分 (pool_score) → 每系统每日 top-N (狙击TOP-5 ∪ 融合TOP-10)
→ 去重合并短名单 (共现优先, 分数降序) → 校准 score→pred_mag/pred_prob.
FULL RUN 只落 top-N 行, 全池分数不落盘 → 本脚本用 load_panel + pool_score 重算全池.

评估窗口 = 末 30 交易日; 校准窗口 = 其前 126 交易日 (walk-forward, 校准不偷看评估).
对比全池 vs 每日短名单:
  Q1 选股效果: 短名单已实现 MFE (label_mfe_*_net) vs 全池基线均值/上涨率
  Q2 预测&概率质量: pred_mag/pred_prob 对已实现的判别力 (rank IC) 在全池 vs 短名单内

结果 WORM → BACKTEST_RESULT_DIR/parallel_pool_topn_<ts>/
用法: python scripts/_diag_parallel_pool_topn.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from app.pipeline_parallel import indicators
from app.pipeline_parallel.backtest import add_mfe_labels, tradability_gate
from app.pipeline_parallel.config import FUSION, HORIZONS, PANEL, SNIPER, board_of
from app.pipeline_parallel.scoring import pool_score
from config.settings import BACKTEST_RESULT_DIR

CAL_N = 160  # 校准+评估合计交易日 (前 130 校准 + 末 30 评估)
EVAL_DAYS = 30
TAIL_DAYS = 220  # 尾部读取交易日 (CAL_N + prepare_adx 滚动窗口预热缓冲)
ABS_TARGET = {"2d": 0.02, "3d": 0.03, "5d": 0.04, "10d": 0.06}
CLS_THRESHOLD = 0.005
CAL_BINS = 6
CAL_MIN_N = 5


def _platt_fit(score: np.ndarray, hit: np.ndarray) -> callable:
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()
    lr.fit(score.reshape(-1, 1), hit.astype(int))
    return lambda s: float(lr.predict_proba(np.array([[s]]))[0, 1])


def build_calibration(
    rec: pd.DataFrame, h: str, target: float
) -> tuple[list[float], callable]:
    """score→(分位桶期望 MFE, Platt 达到概率). rec 须含 score + mfe_{h}."""
    edges = np.quantile(rec["score"], np.linspace(0.0, 1.0, CAL_BINS + 1))
    idx = np.searchsorted(edges, rec["score"], side="right") - 1
    idx = np.clip(idx, 0, CAL_BINS - 1)
    bin_mag = rec.groupby(idx)["mfe"].mean()
    plat = _platt_fit(rec["score"].to_numpy(), (rec["mfe"] >= target).to_numpy())

    def _apply(score: float) -> tuple[float, float, int]:
        b = int(
            np.clip(np.searchsorted(edges, score, side="right") - 1, 0, CAL_BINS - 1)
        )
        n = int((idx == b).sum())
        mag = float(bin_mag.get(b, np.nan))
        prob = float(plat(score))
        return mag, prob, n

    return _apply, plat


POOL_COLS = [
    "VAR51",
    "amihud_illiq",
    "amihud_illiquidity",
    "down_gap_pct",
    "limit_dist_pct",
    "ret_reversal_5d",
    "small_mv_premium",
]
NEEDED_COLS = sorted(
    set(
        [
            "symbol",
            "date",
            "close_hfq",
            "high_hfq",
            "low_hfq",
            "volume",
            "turnover_rate",
            "adv20",
        ]
        + POOL_COLS
    )
)


def load_panel_tail(tail_days: int = TAIL_DAYS) -> pd.DataFrame:
    """内存安全版 load_panel: pyarrow 列选择 + 行过滤, 只读尾部交易日.

    全 3y 检查点 979k×514 列深拷贝会 OOM (3.75 GiB) → 只读尾部 tail_days 个
    交易日的必需列, 再镜像 load_panel 最小处理链:
      add_mfe_labels(2,3,5,10) → tradability_gate → board 列 → prepare_adx(补 pv_corr_5).
    """
    slices = []
    for ckpt in (PANEL.main_checkpoint, PANEL.dual_checkpoint):
        t = pq.read_table(ckpt, columns=NEEDED_COLS)
        df = t.to_pandas()
        dates = sorted(df["date"].unique())
        cut = dates[-tail_days]
        df = df[df["date"] >= cut].reset_index(drop=True)
        slices.append(df)
        del t, df
        import gc

        gc.collect()
    work = pd.concat(slices, ignore_index=True).sort_values(
        ["symbol", "date"], ignore_index=True
    )
    work = add_mfe_labels(work, horizons=(2, 3, 5, 10))
    work, _gate = tradability_gate(work)
    work["board"] = work["symbol"].map(board_of)
    work = indicators.prepare_adx(work)  # 补 pv_corr_5 (checkpoint 无此列)
    return work


def realized_from_work(work: pd.DataFrame) -> pd.DataFrame:
    out = work[["symbol", "date"]].copy()
    for h in ("2d", "3d", "5d", "10d"):
        lab = f"label_mfe_{h}_net"
        out[f"mfe_{h}"] = work[lab].values if lab in work.columns else np.nan
    return out


def daily_ic(sub: pd.DataFrame, pred: str, real: str, min_n: int = 5) -> list[float]:
    vals = []
    for _d, g in sub.dropna(subset=[pred, real]).groupby("date"):
        if len(g) >= min_n and g[real].nunique() > 1:
            r = spearmanr(g[pred], g[real])
            if r.statistic == r.statistic:
                vals.append(r.statistic)
    return vals


def main() -> None:
    t0 = time.time()
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = BACKTEST_RESULT_DIR / f"parallel_pool_topn_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[panel] 内存安全尾部加载 (pyarrow 列选择)...", flush=True)
    work = load_panel_tail()
    all_dates = sorted(work["date"].unique())
    cut_date = all_dates[-CAL_N]
    work = work[work["date"] >= cut_date].reset_index(drop=True)
    all_dates = all_dates[-CAL_N:]
    print(
        f"[panel] {len(work):,}r / {work['symbol'].nunique():,}只 / "
        f"{len(all_dates)} 交易日 {cut_date.date()}..{all_dates[-1].date()} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    # ── 全池逐系统打分 (向量化) ──
    print("[score] 全池 pool_score (sniper/fusion)...", flush=True)
    work["score_sniper"] = np.nan
    work["score_fusion"] = np.nan
    for spec in (SNIPER, FUSION):
        col = f"score_{spec.name}"
        for board in ("main", "dual"):
            bm = work["board"] == board
            work.loc[bm, col] = pool_score(work[bm], spec.pool).values
    print(f"[score] done ({time.time() - t0:.0f}s)", flush=True)

    # 每日每系统 top-N (与生产 select_topn 同口径): 狙击TOP-5 ∪ 融合TOP-10, 去重, 共现优先+分数降序
    short_frames = []
    for D in all_dates[-EVAL_DAYS:]:
        day = work[work["date"] == D]
        frames = []
        for spec in (SNIPER, FUSION):
            sub = day
            sc = sub[f"score_{spec.name}"]
            sc = sc.fillna(-1.0)
            top_idx = sc.sort_values(ascending=False).head(spec.top_n).index
            topn = sub.loc[top_idx, ["symbol", "board"]].copy()
            topn["score"] = sub.loc[top_idx, f"score_{spec.name}"].values
            topn["system"] = spec.name
            frames.append(topn)
        if not frames:
            continue
        merged = pd.concat(frames, ignore_index=True)
        g = merged.groupby(["symbol", "board"], as_index=False).agg(
            systems=("system", lambda x: "+".join(sorted(set(x)))),
            score=("score", "max"),
        )
        g["co_occur"] = g["systems"].str.contains("+", regex=False)
        g = g.sort_values(["co_occur", "score"], ascending=[False, False])
        g["rk"] = np.arange(1, len(g) + 1)
        g["date"] = D
        short_frames.append(g)
    short = pd.concat(short_frames, ignore_index=True)
    print(
        f"[shortlist] 每日合并短名单: {len(short):,} 行 / {len(short['date'].unique())} 日",
        flush=True,
    )

    # ── walk-forward 校准: 前 126 日全池 fit → 末 30 日 apply ──
    cal_dates = all_dates[:-EVAL_DAYS]
    eval_dates = all_dates[-EVAL_DAYS:]
    print(
        f"[calib] 校准 {len(cal_dates)} 日 ({cal_dates[0].date()}..{cal_dates[-1].date()})",
        flush=True,
    )

    cal_pool = work[work["date"].isin(cal_dates)].copy()
    # 每 (board, h): 全池两系统 max score → mfe 校准
    cal_pool["score"] = cal_pool[["score_sniper", "score_fusion"]].max(axis=1)
    cals = {}
    for h in HORIZONS:
        lab = f"label_mfe_{h}_net"
        if lab not in cal_pool.columns:
            continue
        rec = cal_pool[["score", lab]].rename(columns={lab: "mfe"}).dropna()
        if len(rec) < 50:
            continue
        cals[h] = build_calibration(rec, h, ABS_TARGET[h])[0]

    # 给全池+短名单 apply pred_mag/pred_prob (每行用该行 score)
    def apply_cal(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for h in HORIZONS:
            out[f"pred_mag_{h}"] = np.nan
            out[f"pred_prob_{h}"] = np.nan
        for _, r in out.iterrows():
            if not np.isfinite(r.get("score", np.nan)):
                continue
            for h in HORIZONS:
                if h not in cals:
                    continue
                mag, prob, _n = cals[h](float(r["score"]))
                out.at[_, f"pred_mag_{h}"] = mag
                out.at[_, f"pred_prob_{h}"] = prob
        return out

    pool_eval = work[work["date"].isin(eval_dates)].copy()
    pool_eval["score"] = pool_eval[["score_sniper", "score_fusion"]].max(axis=1)
    print("[calib] 全池 apply 校准...", flush=True)
    pool_eval = apply_cal(pool_eval)
    short_eval = short[short["date"].isin(eval_dates)].copy()
    short_eval = apply_cal(short_eval)

    # 已实现 MFE (全池/短名单)
    real = realized_from_work(work)
    pool_eval = pool_eval.merge(real, on=["symbol", "date"], how="left")
    short_eval = short_eval.merge(real, on=["symbol", "date"], how="left")

    # ── 指标 ──
    print("\n========== Q1 选股效果: 短名单 vs 全池 已实现 MFE ==========", flush=True)
    summary: dict = {"ts": ts, "cal_days": len(cal_dates), "eval_days": len(eval_dates)}
    sel_rows = []
    for h in HORIZONS:
        realcol = f"mfe_{h}"
        for tag, df in (("pool", pool_eval), ("short", short_eval)):
            sub = df.dropna(subset=[realcol])
            if len(sub) < 5:
                continue
            sel_rows.append(
                {
                    "horizon": h,
                    "group": tag,
                    "n": len(sub),
                    "n_days": sub["date"].nunique(),
                    "mfe_mean": float(sub[realcol].mean()),
                    "win_pct": float((sub[realcol] > 0).mean()),
                    "win_cls_pct": float((sub[realcol] > CLS_THRESHOLD).mean()),
                    "hit_target_pct": float((sub[realcol] >= ABS_TARGET[h]).mean()),
                }
            )
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(out_dir / "q1_selection.csv", index=False)
    for h in HORIZONS:
        row = sel[sel["horizon"] == h]
        if len(row) < 2:
            continue
        p = row[row["group"] == "pool"].iloc[0]
        s = row[row["group"] == "short"].iloc[0]
        print(
            f"  T+{h[:-1]}: 全池 MFE {p['mfe_mean']:+.4f} (涨 {p['win_pct']:.0%}) | "
            f"短名单 MFE {s['mfe_mean']:+.4f} (涨 {s['win_pct']:.0%}) | "
            f"Δ {s['mfe_mean'] - p['mfe_mean']:+.4f}"
        )
        summary[f"q1_{h}"] = {
            "pool_mfe": float(p["mfe_mean"]),
            "short_mfe": float(s["mfe_mean"]),
            "delta": float(s["mfe_mean"] - p["mfe_mean"]),
            "pool_win": float(p["win_pct"]),
            "short_win": float(s["win_pct"]),
            "pool_hit": float(p["hit_target_pct"]),
            "short_hit": float(s["hit_target_pct"]),
        }

    print(
        "\n========== Q2 预测&概率质量: pred vs 已实现 rank IC (全池 vs 短名单) ==========",
        flush=True,
    )
    ic_rows = []
    for h in HORIZONS:
        realcol = f"mfe_{h}"
        for tag, df in (("pool", pool_eval), ("short", short_eval)):
            for pred, realname in (
                (f"pred_mag_{h}", realcol),
                (f"pred_prob_{h}", realcol),
            ):
                v = daily_ic(df, pred, realname)
                if not v:
                    continue
                ic_rows.append(
                    {
                        "horizon": h,
                        "group": tag,
                        "pred": "mag" if "mag" in pred else "prob",
                        "ic_mean": float(np.mean(v)),
                        "n_days": len(v),
                    }
                )
    icd = pd.DataFrame(ic_rows)
    icd.to_csv(out_dir / "q2_pred_ic.csv", index=False)
    for h in HORIZONS:
        for pred in ("mag", "prob"):
            row = icd[(icd["horizon"] == h) & (icd["pred"] == pred)]
            if len(row) < 2:
                continue
            p = row[row["group"] == "pool"].iloc[0]
            s = row[row["group"] == "short"].iloc[0]
            print(
                f"  T+{h[:-1]} {pred}: 全池 IC {p['ic_mean']:+.4f} ({p['n_days']}日) | "
                f"短名单 IC {s['ic_mean']:+.4f} ({s['n_days']}日) | "
                f"Δ {s['ic_mean'] - p['ic_mean']:+.4f}"
            )
            summary[f"q2_{h}_{pred}"] = {
                "pool_ic": float(p["ic_mean"]),
                "short_ic": float(s["ic_mean"]),
                "delta": float(s["ic_mean"] - p["ic_mean"]),
            }

    # 逐日明细 (Q1) 落盘供复查
    daily_rows = []
    for h in HORIZONS:
        realcol = f"mfe_{h}"
        for tag, df in (("pool", pool_eval), ("short", short_eval)):
            for D, g in df.dropna(subset=[realcol]).groupby("date"):
                daily_rows.append(
                    {
                        "date": str(D.date()),
                        "horizon": h,
                        "group": tag,
                        "n": len(g),
                        "mfe": float(g[realcol].mean()),
                        "win": float((g[realcol] > 0).mean()),
                    }
                )
    pd.DataFrame(daily_rows).to_csv(out_dir / "q1_daily.csv", index=False)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[done] {time.time() - t0:.0f}s → {out_dir}")


if __name__ == "__main__":
    main()
