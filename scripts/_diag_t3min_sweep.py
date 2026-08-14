"""_diag_t3min_sweep.py — 并行短名单入选门 t3_min 扫描 (2026-08-14).

问题: 并行交付入选门 SHORTLIST_SCORE.select_gate.t3_min = 0.00 (config/settings.py),
即"凡 T+3 预期涨幅为正即入选" → 每板块再按 pred_mag_10d 取 TOP-5. 末位候选
(接近零 T+3 预期) 恰恰是拖累命中率的股. 扫描 t3_min 能否用"砍末位"换更高命中率/幅度.

生产同款: calibrate_mag10d (cal_n=21 纯横截面, 日界无前瞻), score=max(sniper,fusion)
池分, 排名键 pred_mag_10d (label_horizon=10), 入选门 pred_ret_3d (label_horizon=3).
每个 t3_min 阈值: 候选池 = pred_ret_3d > t3_min (IRON RULE 只列预测上涨股), 再按
pred_mag_10d 每板块日降序取 TOP-5. 只评估已实现日 (label_pm_10d_net 非 NaN),
末 250 已实现交易日 (oos-only-acceptance 口径), 分 4 子窗看稳定性.

主指标: 10d 命中率/实得 (与验收口径一致), 附 3d/5d; 代价 = 出股数/日出股日占比
(砍太狠 → 空仓日变多, 单日不足 5 只). 选"稳定 + 命中率/幅度同时改善"的档.
WORM 输出 data/_diag_t3min_sweep_<ts>.csv/.json.

用法: python scripts/_diag_t3min_sweep.py [--eval-days=125|60|250]
默认 eval-days=250 (项目验收口径, 末 250 已实现交易日); 125/60 为补充短窗
(检查弱市是否改变最优 t3_min). 结果 WORM data/_diag_t3min_sweep_<eval>d_<ts>.csv/.json.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.pipeline_parallel.calibration import calibrate_mag10d
from app.pipeline_parallel.config import FUSION, SNIPER
from app.pipeline_parallel.scoring import pool_score
from config.settings import DATA_DIR

POOL_COLS = sorted({c for c in set(SNIPER.pool) | set(FUSION.pool) if c != "pv_corr_5"})
EVAL_DAYS = 250  # 评估: 末 N 个已实现交易日 (可用 --eval-days 覆盖)
N_TAIL_OFFSET = 80  # 决策日载入 = eval_days + 该余量 (校准窗 + 标签视界)
TOPN = 5  # 2026-08-14 定案: 每板块 TOP-5
# t3_min 扫描网格 (分数): 0=当前生产基线; 0.25%~2% 逐档收紧
T3MINS = (0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02)
HORIZONS = ("3d", "5d", "10d")
LABEL = {h: f"label_pm_{h}_net" for h in HORIZONS}


def _load_board(board: str, n_tail: int) -> pd.DataFrame | None:
    """同 _diag_mag_frontier: 3y 诊断面板截 n_tail 决策日, 算 score=max(sniper,fusion)."""
    fp = DATA_DIR / f"_diag_stage_{board}_3y.parquet"
    dates = pd.to_datetime(pq.read_table(str(fp), columns=["date"]).to_pandas()["date"])
    uniq = np.unique(dates.values)
    if len(uniq) < n_tail + 20:
        return None
    cutoff = uniq[-(n_tail + 20)]
    t = pq.read_table(
        str(fp),
        columns=["symbol", "date"] + POOL_COLS + list(LABEL.values()),
        filters=[("date", ">=", cutoff)],
    ).to_pandas()
    t["symbol"] = t["symbol"].astype(str)
    t["board"] = board
    sn = pool_score(t, SNIPER.pool)
    fu = pool_score(t, FUSION.pool)
    t["score"] = np.maximum(sn.values, fu.values)
    t = t.dropna(subset=["score"])
    return t[["symbol", "date", "board", "score"] + list(LABEL.values())].copy()


def _sub_window_metrics(top: pd.DataFrame, days: list, n_sub: int) -> list[dict]:
    """末 EVAL_DAYS 已实现日分 n_sub 段, 每段 10d 命中率/实得均值 (稳定性)."""
    out = []
    n = len(days)
    step = n // n_sub
    for i in range(n_sub):
        s0, s1 = i * step, n if i == n_sub - 1 else (i + 1) * step
        seg = top[top["date"].isin(days[s0:s1])]
        out.append(
            {
                "win": f"{i + 1}/{n_sub}",
                "rows": int(len(seg)),
                "hit10": float((seg[LABEL["10d"]] > 0).mean())
                if len(seg)
                else float("nan"),
                "mean10": float(seg[LABEL["10d"]].mean()) if len(seg) else float("nan"),
            }
        )
    return out


def _eval_threshold(rr: pd.DataFrame, days: list, t3: float, n_sub: int) -> dict:
    """单个 t3_min: 候选=pred_ret_3d>t3 → 按 pred_mag_10d 每板块日 TOP-5 → 命中/幅度/出股."""
    g = rr[rr["pred_ret_3d"] > t3]
    top = (
        g.sort_values(["date", "pred_mag_10d"], ascending=[True, False])
        .groupby("date", sort=True)
        .head(TOPN)
    )
    n = int(len(top))
    n_days = int(top["date"].nunique())
    row = {
        "t3_min": round(float(t3), 4),
        "rows": n,
        "days_with_picks": n_days,
        "days_total": len(days),
        "picks_per_day": n / len(days),  # 含空仓日, 砍太狠 → 该值塌缩
        "avg_picks_per_active_day": n / n_days if n_days else float("nan"),
    }
    for h in HORIZONS:
        col = LABEL[h]
        row[f"realized_{h}"] = float(top[col].mean()) if n else float("nan")
        row[f"hit_{h}"] = float((top[col] > 0).mean()) if n else float("nan")
    row["pct_ge5pct"] = float((top[LABEL["10d"]] >= 0.05).mean()) if n else float("nan")
    row["pct_ge10pct"] = (
        float((top[LABEL["10d"]] >= 0.10).mean()) if n else float("nan")
    )
    row["sub_windows"] = _sub_window_metrics(top, days, n_sub)
    return row


def main() -> int:
    _eval_days = EVAL_DAYS
    _args = [a for a in sys.argv[1:] if a.startswith("--eval-days=")]
    if _args:
        _eval_days = int(_args[-1].split("=", 1)[1])
    n_tail = _eval_days + N_TAIL_OFFSET
    n_sub = max(2, _eval_days // 60)
    all_rows: list[dict] = []
    for board in ("main", "dual"):
        t = _load_board(board, n_tail)
        if t is None:
            print(f"[{board}] 面板不足 -> skip", flush=True)
            continue
        work = t[["symbol", "date", "board", "score"] + list(LABEL.values())].copy()
        p3 = calibrate_mag10d(work, target_col=LABEL["3d"], label_horizon=3)
        p10 = calibrate_mag10d(work, target_col=LABEL["10d"], label_horizon=10)
        mm = work.merge(
            p3.rename(columns={"mag": "pred_ret_3d"}),
            on=["symbol", "date"],
            how="inner",
        )
        mm = mm.merge(
            p10.rename(columns={"mag": "pred_mag_10d"}),
            on=["symbol", "date"],
            how="inner",
        )
        mm["date"] = pd.to_datetime(mm["date"])
        rr = mm.dropna(subset=[LABEL["10d"]])
        days = sorted(rr["date"].unique())[-_eval_days:]
        rr = rr[rr["date"].isin(days)].reset_index(drop=True)
        print(
            f"\n===== {board}  末 {len(days)} 已实现交易日 "
            f"(T+3 门 / T+10 排名, TOP-{TOPN}, 净收益) =====",
            flush=True,
        )
        print(
            f"{'t3_min':>7} {'出股/日':>7} {'有票日%':>7} {'实得3d':>8} "
            f"{'实得5d':>8} {'实得10d':>8} {'命中3d':>7} {'命中5d':>7} {'命中10d':>7} "
            f"{'≥+5%':>6} {'≥+10%':>6}"
        )
        for t3 in T3MINS:
            r = _eval_threshold(rr, days, t3, n_sub)
            r["board"] = board
            all_rows.append(r)
            sub = r["sub_windows"]
            subs = (
                "  ".join(
                    f"{s['win']}:{s['hit10']:.0%}/{s['mean10']:+.2%}" for s in sub
                )
                if sub
                else "n/a"
            )
            print(
                f"{r['t3_min']:>7.2%} {r['picks_per_day']:>7.2f} "
                f"{r['days_with_picks'] / r['days_total']:>7.0%} "
                f"{r['realized_3d']:>+8.2%} {r['realized_5d']:>+8.2%} "
                f"{r['realized_10d']:>+8.2%} {r['hit_3d']:>7.0%} "
                f"{r['hit_5d']:>7.0%} {r['hit_10d']:>7.0%} "
                f"{r['pct_ge5pct']:>6.0%} {r['pct_ge10pct']:>6.0%}",
                flush=True,
            )
            print(f"    sub: {subs}", flush=True)

    if not all_rows:
        print("[error] 无任何板块可评估", flush=True)
        return 1
    df = pd.DataFrame(all_rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = DATA_DIR / f"_diag_t3min_sweep_{_eval_days}d_{ts}.csv"
    df.to_csv(out, index=False)
    (DATA_DIR / f"_diag_t3min_sweep_{_eval_days}d_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "eval_days": _eval_days,
                "topn": TOPN,
                "rows": df.to_dict("records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n[saved] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
