"""_diag_parallel_prob_gate.py — 并行短名单概率闸扫描 (2026-08-14, legacy 经验移植 #2).

背景: legacy dual 250d 定案 — 概率边际 prob > base_rate + 0.08 (逐股 GBM 概率) 与
疼痛闸合砍后命中 62→75%. 并行侧入选门只有 t3_min (08-14 定案 main=0 / dual=0.5%),
生产虽有 pred_prob_3d (横截面 Platt, select_confident 预留 prob_min 参数但默认 0
即关闭), 从未扫过"概率闸在 t3 门之上能否再提命中/幅度".

生产口径复现 (scripts/_shortlist_t5_t10.py):
- prob = 横截面 Platt: score → P(mfe_3d ≥ ABS_TARGET[3d]=0.03), mfe = _add_mfe
  生产口径 (窗口最高价/买入价-1-成本, 净 MFE), 训练窗 = 近 142 交易日
  (PER_STOCK_WINDOW+12, 生产每日运行重训) → 本脚本逐决策日 walk-forward 重训,
  日界无前瞻. score = max(sniper, fusion) (与 t3/pain 扫描同约定; 生产 "both"
  键 Platt 用 sniper∪fusion 双行拟合, 差异二阶, 不影响"闸是否有效"结论).
- 闸变体: 固定档 prob > p 与 legacy 同款相对档 prob > base_rate + m, 其中
  base_rate = 近 20 交易日板块达标率 P(mfe_3d ≥ 3%) 滚动均值 (决策日可观测).
  生产 EMA 平滑 (α=0.35,K=12) 不在扫描内 — 闸若有效, 平滑只增稳不减效.
- 叠加已落地 t3 门, 排名键 pred_mag_10d, 每板块日 TOP-5, 末 250 已实现交易日,
  4 子窗稳定性 (同 _diag_t3min_sweep/_diag_parallel_pain_gate).

主指标: 10d 命中率/实得, 附 3d/5d; 代价 = 出股数/出票日占比.
WORM 输出 data/_diag_parallel_prob_gate_<ts>.csv/.json.

用法: python scripts/_diag_parallel_prob_gate.py [--eval-days=250]
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression

from app.pipeline1.label_engine import COST, slippage_tier
from app.pipeline_parallel.calibration import calibrate_mag10d
from app.pipeline_parallel.config import FUSION, SNIPER
from app.pipeline_parallel.scoring import pool_score
from config.settings import DATA_DIR

POOL_COLS = sorted({c for c in set(SNIPER.pool) | set(FUSION.pool) if c != "pv_corr_5"})
EVAL_DAYS = 250
TRAIN_DAYS = 142  # 生产 Platt 训练窗 = PER_STOCK_WINDOW(130) + 12
N_TAIL_MARGIN = 20  # 尾部余量 (标签视界 + 校准窗)
TOPN = 5  # 2026-08-14 定案: 每板块 TOP-5
T3_LANDED = {"main": 0.0, "dual": 0.005}  # 08-14 t3_min 定案 (生产已落地)
ABS_TARGET = 0.03  # _shortlist_t5_t10.ABS_TARGET["3d"]
BASE_RATE_DAYS = 20  # legacy 同款滚动达标率窗
HORIZONS = ("3d", "5d", "10d")
LABEL = {h: f"label_pm_{h}_net" for h in HORIZONS}
# 概率闸扫描网格: 固定档 + legacy 同款相对档 (base_rate + margin)
PROB_FIXED = (0.0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50)
PROB_MARGIN = (0.0, 0.02, 0.05, 0.08, 0.10, 0.15)


def _add_mfe(df: pd.DataFrame) -> pd.DataFrame:
    """同 _shortlist_t5_t10._add_mfe 生产口径: mfe_kd = 窗口最高价/买入价-1-成本(净)."""
    g = df.groupby("symbol", sort=False)
    exec_px = g["close_hfq"].shift(-1)
    max_off = 4  # 只算 3d: shifts 2..4
    shifts = pd.concat(
        [g["high_hfq"].shift(-off) for off in range(2, max_off + 1)],
        axis=1,
        keys=range(2, max_off + 1),
    )
    slip = df["adv20"].map(slippage_tier)
    cost_total = COST + 2 * slip
    peak = shifts.loc[:, 2:4].max(axis=1, skipna=False)
    df["mfe_3d"] = peak / exec_px - 1 - cost_total
    return df


def _load_board(board: str, n_tail: int) -> pd.DataFrame | None:
    """同 _diag_t3min_sweep + mfe 生产口径标签: 3y 诊断面板截 n_tail 决策日."""
    fp = DATA_DIR / f"_diag_stage_{board}_3y.parquet"
    dates = pd.to_datetime(pq.read_table(str(fp), columns=["date"]).to_pandas()["date"])
    uniq = np.unique(dates.values)
    if len(uniq) < n_tail + 20:
        return None
    cutoff = uniq[-(n_tail + 20)]
    need = (
        ["symbol", "date", "close_hfq", "high_hfq", "adv20"]
        + POOL_COLS
        + list(LABEL.values())
    )
    t = pq.read_table(
        str(fp), columns=need, filters=[("date", ">=", cutoff)]
    ).to_pandas()
    t["symbol"] = t["symbol"].astype(str)
    t = t.sort_values(["date", "symbol"]).reset_index(drop=True)
    t["board"] = board
    sn = pool_score(t, SNIPER.pool)
    fu = pool_score(t, FUSION.pool)
    t["score"] = np.maximum(sn.values, fu.values)
    t = t.dropna(subset=["score"])
    t = _add_mfe(t)
    return t.copy()


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


def _eval_gate(rr: pd.DataFrame, days: list, t3: float, gate: str, n_sub: int) -> dict:
    """t3 门 + 概率闸 (gate 列已预算好) → pred_mag_10d 排名 → 每板块日 TOP-5."""
    g = rr[(rr["pred_ret_3d"] > t3) & rr["_gate_pass"]]
    top = (
        g.sort_values(["date", "pred_mag_10d"], ascending=[True, False])
        .groupby("date", sort=True)
        .head(TOPN)
    )
    n = int(len(top))
    n_days = int(top["date"].nunique())
    row = {
        "gate": gate,
        "t3_min": round(float(t3), 4),
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


def _walkforward_probs(t: pd.DataFrame, dates: np.ndarray, n_eval: int) -> pd.DataFrame:
    """逐决策日 walk-forward Platt: 训练=近 TRAIN_DAYS 交易日 (日界无前瞻).

    返回 DataFrame[symbol, date, pred_prob_3d] (仅末 n_eval 个决策日).
    """
    t = t.copy()
    t["date"] = pd.to_datetime(t["date"])
    d_idx = np.searchsorted(dates, t["date"].values)
    y = (t["mfe_3d"] >= ABS_TARGET).astype(int)
    ok = t["mfe_3d"].notna() & y.notna()
    out_rows = []
    for i in range(len(dates) - n_eval, len(dates)):
        mask = (d_idx < i) & (d_idx >= i - TRAIN_DAYS) & ok.values
        if mask.sum() < 50:
            continue
        clf = LogisticRegression(max_iter=1000)
        clf.fit(t.loc[mask, ["score"]].to_numpy(), y.loc[mask].to_numpy())
        day = t[d_idx == i]
        probs = clf.predict_proba(day[["score"]].to_numpy())[:, 1]
        out_rows.append(
            pd.DataFrame(
                {
                    "symbol": day["symbol"].values,
                    "date": day["date"].values,
                    "pred_prob_3d": probs,
                }
            )
        )
        if (i - (len(dates) - n_eval)) % 50 == 0:
            print(
                f"    [prob] {i - (len(dates) - n_eval)}/{n_eval} 决策日已拟合",
                flush=True,
            )
    if not out_rows:
        return pd.DataFrame(columns=["symbol", "date", "pred_prob_3d"])
    return pd.concat(out_rows, ignore_index=True)


def main() -> int:
    _eval_days = EVAL_DAYS
    _args = [a for a in sys.argv[1:] if a.startswith("--eval-days=")]
    if _args:
        _eval_days = int(_args[-1].split("=", 1)[1])
    n_tail = _eval_days + TRAIN_DAYS + N_TAIL_MARGIN
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
            p3.drop(columns=["board"]).rename(columns={"mag": "pred_ret_3d"}),
            on=["symbol", "date"],
            how="inner",
        )
        mm = mm.merge(
            p10.drop(columns=["board"]).rename(columns={"mag": "pred_mag_10d"}),
            on=["symbol", "date"],
            how="inner",
        )
        mm["date"] = pd.to_datetime(mm["date"])
        rr = mm.dropna(subset=[LABEL["10d"]])
        days = sorted(rr["date"].unique())[-_eval_days:]
        rr = rr[rr["date"].isin(days)].reset_index(drop=True)

        # walk-forward Platt 概率 (训练用 t 的 score+mfe, 预测落在末 _eval_days 决策日)
        all_dates = np.unique(np.sort(pd.to_datetime(t["date"]).values))
        probs = _walkforward_probs(t, all_dates, _eval_days)
        rr = rr.merge(probs, on=["symbol", "date"], how="left")
        # 滚动达标率 base_rate: 近 20 交易日 P(mfe_3d >= 3%) 板块均值 (决策日可观测)
        daily_rate = (
            t.assign(_hit=(t["mfe_3d"] >= ABS_TARGET).astype(float))
            .groupby("date")["_hit"]
            .mean()
        )
        base = (
            daily_rate.rolling(BASE_RATE_DAYS, min_periods=BASE_RATE_DAYS)
            .mean()
            .shift(1)
            .rename("base_rate")
        )
        rr = rr.merge(base, left_on="date", right_index=True, how="left")

        pcol = rr["pred_prob_3d"]
        bcol = rr["base_rate"]
        print(
            f"\n===== {board}  末 {len(days)} 已实现交易日 "
            f"(prob 分布: mean {pcol.mean():.3f} / q25 {pcol.quantile(0.25):.3f} / "
            f"q50 {pcol.quantile(0.5):.3f} / q75 {pcol.quantile(0.75):.3f}; "
            f"base_rate 均值 {bcol.mean():.3f}, t3 门 {T3_LANDED[board]:.2%}) =====",
            flush=True,
        )
        print(
            f"{'概率闸':>12} {'出股/日':>7} {'有票日%':>7} {'实得3d':>8} "
            f"{'实得5d':>8} {'实得10d':>8} {'命中3d':>7} {'命中5d':>7} {'命中10d':>7} "
            f"{'≥+5%':>6} {'≥+10%':>6}"
        )
        variants: list[tuple[str, pd.Series]] = [
            ("关(基线)", pd.Series(True, index=rr.index))
        ]
        for p in PROB_FIXED:
            variants.append((f"prob>{p:.2f}", pcol > p))
        for m in PROB_MARGIN:
            variants.append((f"base+{m:.2f}", pcol > bcol + m))
        for gate, pass_mask in variants:
            rr["_gate_pass"] = pass_mask.fillna(False).to_numpy()
            r = _eval_gate(rr, days, T3_LANDED[board], gate, n_sub)
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
                f"{gate:>12} {r['picks_per_day']:>7.2f} "
                f"{r['days_with_picks'] / r['days_total']:>7.0%} "
                f"{r['realized_3d']:>+8.2%} {r['realized_5d']:>+8.2%} "
                f"{r['realized_10d']:>+8.2%} {r['hit_3d']:>7.0%} "
                f"{r['hit_5d']:>7.0%} {r['hit_10d']:>7.0%} "
                f"{r['pct_ge5pct']:>6.0%} {r['pct_ge10pct']:>6.0%}",
                flush=True,
            )
            print(f"    sub: {subs}", flush=True)
        rr = rr.drop(columns=["_gate_pass"])

    if not all_rows:
        print("[error] 无任何板块可评估", flush=True)
        return 1
    df = pd.DataFrame(all_rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = DATA_DIR / f"_diag_parallel_prob_gate_{_eval_days}d_{ts}.csv"
    df.to_csv(out, index=False)
    (DATA_DIR / f"_diag_parallel_prob_gate_{_eval_days}d_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "eval_days": _eval_days,
                "topn": TOPN,
                "train_days": TRAIN_DAYS,
                "abs_target": ABS_TARGET,
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
