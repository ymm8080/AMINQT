"""_diag_parallel_pain_gate.py — 并行短名单疼痛闸扫描 (2026-08-14, legacy 经验移植).

背景: legacy dual 250d 定案 — 痛苦闸 pain_prob≤0.4 (GBM PainModel) 叠加概率边际后
命中 62→75% / 实得 +5.84→+7.39% (出票日 37% 宁缺毋滥). 并行侧入选门只有 t3_min
(08-14 定案 main=0 / dual=0.5%), 无疼痛闸. 本脚本验证: 用面板 label_pain (3日浮亏
>5%) 经 calibrate_mag10d 同款机制 (纯横截面 OLS, 日界无前瞻, 生产已有定案
mag10d-cal-window-sensitivity) 拟合每日 pain 代理 → 闸 "pain 代理 ≤ θ" 能否在
t3 门之上再提命中率/幅度.

生产同款: score=max(sniper,fusion), 排名键 pred_mag_10d (label_horizon=10),
入选门 pred_ret_3d (label_horizon=3, 已落地分板块阈值), pain 代理 label_horizon=3
(决策日 D 预测 D+1..D+3 持有窗口浮亏风险, 与 label_pm_3d_net 同对齐).
只评估已实现日 (label_pm_10d_net 非 NaN), 末 250 已实现交易日, 4 子窗稳定性.

主指标: 10d 命中率/实得 (验收口径), 附 3d/5d; 代价 = 出股数/出票日占比.
WORM 输出 data/_diag_parallel_pain_gate_<ts>.csv/.json.

用法: python scripts/_diag_parallel_pain_gate.py [--eval-days=250]
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
EVAL_DAYS = 250
N_TAIL_OFFSET = 80  # 决策日载入 = eval_days + 余量 (校准窗 + 标签视界)
TOPN = 5  # 2026-08-14 定案: 每板块 TOP-5
# 2026-08-14 t3_min 定案 (SHORTLIST_SCORE.select_gate): 分板块已落地阈值
T3_LANDED = {"main": 0.0, "dual": 0.005}
# pain 代理闸扫描网格: None=关闸基线 (生产现状); 0.2~0.6 逐档收紧
PAIN_THRESH = (None, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60)
HORIZONS = ("3d", "5d", "10d")
LABEL = {h: f"label_pm_{h}_net" for h in HORIZONS}


def _load_board(board: str, n_tail: int) -> pd.DataFrame | None:
    """同 _diag_t3min_sweep: 3y 诊断面板截 n_tail 决策日, score=max(sniper,fusion)."""
    fp = DATA_DIR / f"_diag_stage_{board}_3y.parquet"
    dates = pd.to_datetime(pq.read_table(str(fp), columns=["date"]).to_pandas()["date"])
    uniq = np.unique(dates.values)
    if len(uniq) < n_tail + 20:
        return None
    cutoff = uniq[-(n_tail + 20)]
    t = pq.read_table(
        str(fp),
        columns=["symbol", "date"] + POOL_COLS + list(LABEL.values()) + ["label_pain"],
        filters=[("date", ">=", cutoff)],
    ).to_pandas()
    t["symbol"] = t["symbol"].astype(str)
    t["board"] = board
    sn = pool_score(t, SNIPER.pool)
    fu = pool_score(t, FUSION.pool)
    t["score"] = np.maximum(sn.values, fu.values)
    t = t.dropna(subset=["score"])
    return t[
        ["symbol", "date", "board", "score"] + list(LABEL.values()) + ["label_pain"]
    ].copy()


def _sub_window_metrics(top: pd.DataFrame, days: list, n_sub: int) -> list[dict]:
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


def _eval_thresh(rr: pd.DataFrame, days: list, t3: float, theta, n_sub: int) -> dict:
    """t3 门 (已落地) + pain 代理闸 θ → pred_mag_10d 排名 → 每板块日 TOP-5."""
    g = rr[rr["pred_ret_3d"] > t3]
    if theta is not None:
        g = g[g["pain_proxy"] <= theta]
    top = (
        g.sort_values(["date", "pred_mag_10d"], ascending=[True, False])
        .groupby("date", sort=True)
        .head(TOPN)
    )
    n = int(len(top))
    n_days = int(top["date"].nunique())
    row = {
        "t3_min": round(float(t3), 4),
        "pain_thresh": None if theta is None else round(float(theta), 4),
        "rows": n,
        "days_with_picks": n_days,
        "days_total": len(days),
        "picks_per_day": n / len(days),
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
        work = t[
            ["symbol", "date", "board", "score"] + list(LABEL.values()) + ["label_pain"]
        ].copy()
        p3 = calibrate_mag10d(work, target_col=LABEL["3d"], label_horizon=3)
        p10 = calibrate_mag10d(work, target_col=LABEL["10d"], label_horizon=10)
        pp = calibrate_mag10d(work, target_col="label_pain", label_horizon=3)
        # calibrate 输出带 board 列, 多次 merge 会与左帧 board 冲突 → 先丢 (该帧本就单板块)
        mm = work.merge(
            p3.drop(columns=["board"]).rename(columns={"mag": "pred_ret_3d"}),
            on=["symbol", "date"],
            how="inner",
        )
        mm = mm.merge(
            p10.drop(columns=["board"]).rename(columns={"mag": "pred_mag_10d"}),
            on=["symbol", "date"],
            how="inner",
        )
        mm = mm.merge(
            pp.drop(columns=["board"]).rename(columns={"mag": "pain_proxy"}),
            on=["symbol", "date"],
            how="inner",
        )
        mm["pain_proxy"] = mm["pain_proxy"].clip(0.0, 1.0)
        mm["date"] = pd.to_datetime(mm["date"])
        rr = mm.dropna(subset=[LABEL["10d"]])
        days = sorted(rr["date"].unique())[-_eval_days:]
        rr = rr[rr["date"].isin(days)].reset_index(drop=True)
        print(
            f"\n===== {board}  末 {len(days)} 已实现交易日 "
            f"(pain 代理分布: mean {rr['pain_proxy'].mean():.3f} / "
            f"q25 {rr['pain_proxy'].quantile(0.25):.3f} / q50 {rr['pain_proxy'].quantile(0.5):.3f} / "
            f"q75 {rr['pain_proxy'].quantile(0.75):.3f}, t3 门 {T3_LANDED[board]:.2%}) =====",
            flush=True,
        )
        print(
            f"{'pain闸':>7} {'出股/日':>7} {'有票日%':>7} {'实得3d':>8} "
            f"{'实得5d':>8} {'实得10d':>8} {'命中3d':>7} {'命中5d':>7} {'命中10d':>7} "
            f"{'≥+5%':>6} {'≥+10%':>6}"
        )
        for theta in PAIN_THRESH:
            r = _eval_thresh(rr, days, T3_LANDED[board], theta, n_sub)
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
            th = "关" if theta is None else f"{theta:.2f}"
            print(
                f"{th:>7} {r['picks_per_day']:>7.2f} "
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
    out = DATA_DIR / f"_diag_parallel_pain_gate_{_eval_days}d_{ts}.csv"
    df.to_csv(out, index=False)
    (DATA_DIR / f"_diag_parallel_pain_gate_{_eval_days}d_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "eval_days": _eval_days,
                "topn": TOPN,
                "t3_landed": T3_LANDED,
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
