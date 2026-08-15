"""_diag_parallel_gbm_wf.py — 并行真模型第二阶段: walk-forward 概率头验证 (2026-08-15).

阶段1 (单次时间切分, _diag_parallel_gbm_signal.py, WORM _diag_parallel_gbm_signal_20260814_131720)
结论: 并行特征空间有真信号; dual prob>base+0.08 全窗全指标赢 + 4/4 子窗实得赢
(弱市窗反而改善 — 无 Platt 概率闸的空仓 regime 假象); main 微赢; pain 真模型双板全输,
疼痛路线关闭 (见 [[parallel-gbm-signal-stage1]])。

本阶段按生产口径验证概率闸 (重活):
- 每 REFIT_EVERY=21 交易日用 trailing TRAIN_DAYS=242 行重训概率头
  (legacy load_panel window_days=242 同款)
- 测试窗 = 末 250 已实现交易日 (与验收窗同口径); 无前瞻: refit 只用 < 当日 的行
- base_rate = 近 20 交易日 mfe 达标率滚动均值 shift(1) (legacy 配方)
- 边际扫描 margin ∈ {0.04, 0.06, 0.08, 0.10} → 按 [[param-sweep-reliability]] 选稳定平台
- 4 子窗稳定性 + 空窗检验 (Platt 假象的墓志铭)
WORM 输出 data/_diag_parallel_gbm_wf_<ts>.csv/.json。

用法: python scripts/_diag_parallel_gbm_wf.py
注意: 与 daily automation 错峰运行 (双任务并发必 OOM)。
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier

from app.pipeline1.label_engine import COST, slippage_tier
from app.pipeline_parallel.calibration import calibrate_mag10d
from app.pipeline_parallel.config import FUSION, SNIPER
from app.pipeline_parallel.scoring import pool_score
from config.settings import DATA_DIR

EVAL_DAYS = 250
TOPN = 5
T3_LANDED = {"main": 0.0, "dual": 0.005}
ABS_TARGET = 0.03  # 生产 Platt 目标: mfe_3d >= 3%
BASE_RATE_DAYS = 20
TRAIN_DAYS = 242  # legacy load_panel window_days 同款
REFIT_EVERY = 21  # 生产重训节奏代理 (~每月)
MARGINS = (0.04, 0.06, 0.08, 0.10)  # legacy 配方 0.08
META = {
    "symbol", "date", "board", "is_suspended", "name", "code", "exec_px",
}
RAW_COLS = {
    "open", "high", "low", "close", "open_hfq", "high_hfq", "low_hfq", "close_hfq",
    "volume", "amount", "pre_close", "turnover_rate", "total_mv", "adv20",
}
LGB_PARAMS = dict(
    objective="binary",
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=200,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)


def _add_mfe(df: pd.DataFrame) -> pd.DataFrame:
    """同 _shortlist_t5_t10._add_mfe 生产口径: mfe_3d = 窗口最高价/买入价-1-成本."""
    g = df.groupby("symbol", sort=False)
    exec_px = g["close_hfq"].shift(-1)
    shifts = pd.concat(
        [g["high_hfq"].shift(-off) for off in range(2, 5)], axis=1, keys=range(2, 5)
    )
    slip = df["adv20"].map(slippage_tier)
    cost_total = COST + 2 * slip
    peak = shifts.max(axis=1, skipna=False)
    df["mfe_3d"] = peak / exec_px - 1 - cost_total
    return df


def _load_board(board: str) -> pd.DataFrame | None:
    fp = DATA_DIR / f"_diag_stage_{board}_3y.parquet"
    schema = pq.read_schema(str(fp)).names
    need = [
        c
        for c in schema
        if not c.startswith("label_")
        and c not in META
        and not c.startswith("pred_")
    ]
    need += ["symbol", "date", "label_pain", "label_pm_3d_net", "label_pm_10d_net"]
    t = pq.read_table(str(fp), columns=list(dict.fromkeys(need))).to_pandas()
    t["symbol"] = t["symbol"].astype(str)
    t["date"] = pd.to_datetime(t["date"])
    t = t.sort_values(["date", "symbol"]).reset_index(drop=True)
    t["board"] = board
    sn = pool_score(t, SNIPER.pool)
    fu = pool_score(t, FUSION.pool)
    t["score"] = np.maximum(sn.values, fu.values)
    t = t.dropna(subset=["score"])
    t = _add_mfe(t)
    return t


def _feat_cols(t: pd.DataFrame) -> list[str]:
    cols = [
        c
        for c in t.columns
        if c
        not in {
            "symbol", "date", "board", "score", "mfe_3d", "label_pain",
            "label_pm_3d_net", "label_pm_10d_net",
        }
        and c not in RAW_COLS
        and not c.startswith("label_")
        and pd.api.types.is_numeric_dtype(t[c].dtype)
    ]
    return cols


def _eval_top5(
    mm: pd.DataFrame, gate_col: str | None, t3: float, days: list
) -> tuple[dict, pd.DataFrame]:
    """生产同款 TOP-5 (t3 门 + pred_mag_10d 排名) + 可选真模型闸."""
    g = mm[mm["pred_ret_3d"] > t3]
    if gate_col:
        g = g[g[gate_col]]
    top = (
        g.sort_values(["date", "pred_mag_10d"], ascending=[True, False])
        .groupby("date", sort=True)
        .head(TOPN)
    )
    n = len(top)
    row = {
        "rows": n,
        "days_with_picks": int(top["date"].nunique()),
        "picks_per_day": n / len(days),
        "realized_10d": float(top["label_pm_10d_net"].mean()) if n else float("nan"),
        "hit_10d": float((top["label_pm_10d_net"] > 0).mean()) if n else float("nan"),
        "pct_ge5pct": float((top["label_pm_10d_net"] >= 0.05).mean()) if n else float("nan"),
        "pct_ge10pct": (
            float((top["label_pm_10d_net"] >= 0.10).mean()) if n else float("nan")
        ),
    }
    return row, top


def _sub_windows(top: pd.DataFrame, days: list, n_sub: int) -> list[dict]:
    """4 子窗命中/实得 (稳定性检验)."""
    step = len(days) // n_sub
    subs = []
    for i in range(n_sub):
        s0, s1 = i * step, len(days) if i == n_sub - 1 else (i + 1) * step
        seg = top[top["date"].isin(days[s0:s1])]
        subs.append(
            {
                "win": f"{i + 1}/{n_sub}",
                "rows": int(len(seg)),
                "hit10": float((seg["label_pm_10d_net"] > 0).mean()) if len(seg) else float("nan"),
                "mean10": float(seg["label_pm_10d_net"].mean()) if len(seg) else float("nan"),
            }
        )
    return subs


def main() -> int:
    train_mode = "trailing242"
    _args = [a for a in sys.argv[1:] if a.startswith("--train-mode=")]
    if _args:
        train_mode = _args[-1].split("=", 1)[1]
    if train_mode not in ("trailing242", "expanding"):
        print(f"[error] 未知 --train-mode={train_mode}", flush=True)
        return 2
    rows_out: list[dict] = []
    for board in ("main", "dual"):
        t = _load_board(board)
        if t is None:
            print(f"[{board}] 面板不足 -> skip", flush=True)
            continue
        dates = np.unique(t["date"].values)
        cal_test = dates[-EVAL_DAYS:]
        feat_cols = _feat_cols(t)
        y_prob = (t["mfe_3d"] >= ABS_TARGET).astype(float)
        prob_ok = y_prob.notna() & t["label_pain"].notna()
        idx = np.searchsorted(dates, t["date"].values)

        # ---- walk-forward: 每 21 交易日重训, 预测当日行 ----
        # train_mode: trailing242 = 最近 242 行 (legacy load_panel 同款);
        #             expanding   = 全史 (阶段1 赢家的设计, 生产=每月扩窗重训)
        model = None
        pred = pd.Series(np.nan, index=t.index, dtype="float64")
        n_refits = 0
        for k, d in enumerate(cal_test):
            pos = len(dates) - EVAL_DAYS + k
            if model is None or pos % REFIT_EVERY == 0:
                lo = max(0, pos - TRAIN_DAYS) if train_mode == "trailing242" else 0
                tr = (idx >= lo) & (idx < pos) & prob_ok
                X = t.loc[tr, feat_cols].to_numpy(dtype="float32")
                y = y_prob.loc[tr].to_numpy()
                model = LGBMClassifier(**LGB_PARAMS)
                model.fit(X, y)
                n_refits += 1
            te = t["date"].values == d
            if not te.any():
                continue
            Xd = t.loc[te & prob_ok, feat_cols].to_numpy(dtype="float32")
            if len(Xd) == 0:
                continue
            pred.loc[te & prob_ok] = model.predict_proba(Xd)[:, 1]
        print(
            f"\n[{board}] walk-forward 完成 ({train_mode}): {n_refits} 次重训 / "
            f"测试 {len(cal_test)} 日, 特征 {len(feat_cols)}",
            flush=True,
        )

        # ---- 生产同款闸评估 ----
        work = t[["symbol", "date", "board", "score", "label_pm_3d_net",
                  "label_pm_10d_net"]].copy()
        p3 = calibrate_mag10d(work, target_col="label_pm_3d_net", label_horizon=3)
        p10 = calibrate_mag10d(work, target_col="label_pm_10d_net", label_horizon=10)
        mm = work.merge(
            p3.drop(columns=["board"]).rename(columns={"mag": "pred_ret_3d"}),
            on=["symbol", "date"], how="inner",
        ).merge(
            p10.drop(columns=["board"]).rename(columns={"mag": "pred_mag_10d"}),
            on=["symbol", "date"], how="inner",
        )
        mm["date"] = pd.to_datetime(mm["date"])
        rr = mm.dropna(subset=["label_pm_10d_net"])
        days = sorted(rr["date"].unique())[-EVAL_DAYS:]
        rr = rr[rr["date"].isin(days)].reset_index(drop=True)

        sub = t.loc[prob_ok & np.isin(t["date"].values, cal_test), ["symbol", "date"]].copy()
        sub["pred_prob"] = pred[sub.index].to_numpy()
        rr = rr.merge(sub, on=["symbol", "date"], how="left")

        daily_rate = (
            t.assign(_hit=(t["mfe_3d"] >= ABS_TARGET).astype(float))
            .groupby("date")["_hit"].mean()
        )
        base = (
            daily_rate.rolling(BASE_RATE_DAYS, min_periods=BASE_RATE_DAYS)
            .mean().shift(1).rename("base_rate")
        )
        rr = rr.merge(base, left_on="date", right_index=True, how="left")

        for m in MARGINS:
            rr[f"_pg_{m}"] = rr["pred_prob"] > rr["base_rate"] + m
        gates = [("基线(关)", None)] + [
            (f"prob>base+{m:.2f}", f"_pg_{m}") for m in MARGINS
        ]
        print(
            f"\n===== {board}  末 250 已实现交易日 (t3 门 {T3_LANDED[board]:.2%}) =====",
            flush=True,
        )
        print(
            f"{'闸':>18} {'票/日':>6} {'实得10d':>8} {'命中10d':>7} "
            f"{'≥+5%':>6} {'≥+10%':>6}",
            flush=True,
        )
        n_sub = max(2, EVAL_DAYS // 60)
        for gname, gcol in gates:
            r, top = _eval_top5(rr, gcol, T3_LANDED[board], days)
            r["board"] = board
            r["gate"] = gname
            r["sub_windows"] = _sub_windows(top, days, n_sub)
            rows_out.append(r)
            sub_s = "  ".join(
                f"{s['win']}:{s['hit10']:.0%}/{s['mean10']:+.2%}"
                for s in r["sub_windows"]
            )
            print(
                f"{gname:>18} {r['picks_per_day']:>6.2f} {r['realized_10d']:>+8.2%} "
                f"{r['hit_10d']:>7.0%} {r['pct_ge5pct']:>6.0%} "
                f"{r['pct_ge10pct']:>6.0%}",
                flush=True,
            )
            print(f"    sub: {sub_s}", flush=True)

    if not rows_out:
        print("[error] 无任何板块可评估", flush=True)
        return 1
    df = pd.DataFrame(rows_out)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = DATA_DIR / f"_diag_parallel_gbm_wf_{ts}.csv"
    df.to_csv(out, index=False)
    (DATA_DIR / f"_diag_parallel_gbm_wf_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "eval_days": EVAL_DAYS,
                "topn": TOPN,
                "train_mode": train_mode,
                "train_days": TRAIN_DAYS,
                "refit_every": REFIT_EVERY,
                "margins": list(MARGINS),
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
