"""_diag_parallel_gbm_signal.py — 并行"真模型"第一阶段: 信号检验 (2026-08-15).

背景: 并行闸空间已穷尽 ([[parallel-prob-gate-verdict]]/[[parallel-pain-gate-verdict]]),
根因 = 只有 21d 横截面 OLS + 单分数 Platt, 学不出个股概率/疼痛。legacy 的赢家配方
(dual 命中 62→75%) 是**全局 LGBM 分类头** (quantile_models.PainModel +
dual_track_trainer 的 cls 头, 面板级特征, 非逐股), 并行侧没有对应物。

第一阶段 (本脚本, 便宜): 时间切分单次拟合, 回答两个问题 —
1. 并行特征空间 (509/528 列) 能不能学出概率 (mfe_3d≥3%) / 疼痛 (label_pain)?
   → OOS AUC + 概率分布展宽 (Platt 的 IQR 只有 3pp 是死因)
2. 真模型输出的闸 (legacy 同款: prob > base_rate+0.08 & pain_prob ≤ 0.4)
   在 250d TOP-5 上能否赢基线? (基线 main 60.1%/+3.63%, dual 68.2%/+8.06%)

无前瞻: 训练窗 = 末 250 交易日之前全部行; 测试窗 = 末 250 交易日
(与验收窗同口径); base_rate = 近 20 交易日 mfe 达标率滚动均值 shift(1)。
若 AUC≈0.5/分布仍平坦 → 并行特征空间无信号, 路线关闭;
若有信号 → 第二阶段 walk-forward 逐日重训 + 全闸扫描 (重活, 另行)。

WORM 输出 data/_diag_parallel_gbm_signal_<ts>.csv/.json。
用法: python scripts/_diag_parallel_gbm_signal.py
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
    # 池分数只用于 TOP-5 评估 (与生产同: max(sniper, fusion))
    sn = pool_score(t, SNIPER.pool)
    fu = pool_score(t, FUSION.pool)
    t["score"] = np.maximum(sn.values, fu.values)
    t = t.dropna(subset=["score"])
    t = _add_mfe(t)
    return t


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
    """4 子窗命中/实得 (稳定性检验, 同 _diag_gap_pick_eval 口径)."""
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
    rows_out: list[dict] = []
    for board in ("main", "dual"):
        t = _load_board(board)
        if t is None:
            print(f"[{board}] 面板不足 -> skip", flush=True)
            continue
        dates = np.unique(t["date"].values)
        test_dates = set(dates[-EVAL_DAYS:])
        tr_mask = ~t["date"].isin(test_dates)
        te_mask = t["date"].isin(test_dates)

        feat_cols = [
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
        dropped_obj = [
            c for c in t.columns
            if c not in feat_cols
            and not c.startswith("label_")
            and not pd.api.types.is_numeric_dtype(t[c].dtype)
            and c not in {"symbol", "date", "board"}
        ]
        if dropped_obj:
            print(f"[{board}] 非数值列剔除 {len(dropped_obj)}: {dropped_obj[:10]}", flush=True)
        y_prob = (t["mfe_3d"] >= ABS_TARGET).astype(float)
        y_pain = t["label_pain"].astype(float)
        prob_ok = y_prob.notna() & y_pain.notna()
        X_tr = t.loc[tr_mask & prob_ok, feat_cols].to_numpy(dtype="float32")
        X_te = t.loc[te_mask & prob_ok, feat_cols].to_numpy(dtype="float32")
        if len(X_tr) < 5000 or len(X_te) < 500:
            print(f"[{board}] 样本不足 -> skip", flush=True)
            continue
        # mfe 尾段 NaN (未来价缺失) → prob 目标不可训练, 但疼痛目标仍可
        p_tr = y_prob.loc[tr_mask & prob_ok].to_numpy()
        p_te = y_prob.loc[te_mask & prob_ok].to_numpy()
        pain_tr = y_pain.loc[tr_mask & prob_ok].to_numpy()
        pain_te = y_pain.loc[te_mask & prob_ok].to_numpy()

        m_prob = LGBMClassifier(**LGB_PARAMS)
        m_prob.fit(X_tr, p_tr)
        prob_hat = m_prob.predict_proba(X_te)[:, 1]
        auc_prob = _auc(p_te, prob_hat)

        m_pain = LGBMClassifier(**LGB_PARAMS)
        m_pain.fit(X_tr, pain_tr)
        pain_hat = m_pain.predict_proba(X_te)[:, 1]
        auc_pain = _auc(pain_te, pain_hat)
        print(
            f"\n[{board}] 训练 {len(X_tr):,} / 测试 {len(X_te):,} 行, "
            f"特征 {len(feat_cols)}",
            flush=True,
        )
        print(
            f"[{board}] OOS AUC: 概率(mfe3d>=3%) {auc_prob:.4f} | "
            f"疼痛(label_pain) {auc_pain:.4f}",
            flush=True,
        )
        print(
            f"[{board}] 概率分布: q25 {np.quantile(prob_hat, .25):.3f} / "
            f"q50 {np.quantile(prob_hat, .5):.3f} / q75 {np.quantile(prob_hat, .75):.3f} "
            f"(Platt 同口径 IQR 仅 3pp)",
            flush=True,
        )

        # 闸级评估: 生产同款 TOP-5 基线 + prob 边际 + pain + 组合
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

        # 真模型预测回贴 (te_mask 行的顺序已丢失 → 按 (symbol,date) 索引重建)
        te_idx = t.loc[te_mask & prob_ok, ["symbol", "date"]].reset_index(drop=True)
        te_idx["pred_prob"] = prob_hat
        te_idx["pred_pain"] = pain_hat
        rr = rr.merge(te_idx, on=["symbol", "date"], how="left")

        daily_rate = (
            t.assign(_hit=(t["mfe_3d"] >= ABS_TARGET).astype(float))
            .groupby("date")["_hit"].mean()
        )
        base = (
            daily_rate.rolling(BASE_RATE_DAYS, min_periods=BASE_RATE_DAYS)
            .mean().shift(1).rename("base_rate")
        )
        rr = rr.merge(base, left_on="date", right_index=True, how="left")

        rr["_prob_gate"] = rr["pred_prob"] > rr["base_rate"] + 0.08
        rr["_pain_gate"] = rr["pred_pain"].fillna(0) <= 0.4
        rr["_combo"] = rr["_prob_gate"] & rr["_pain_gate"]
        gates = [
            ("基线(关)", None),
            ("prob>base+0.08", "_prob_gate"),
            ("pain<=0.4", "_pain_gate"),
            ("组合(prob&pain)", "_combo"),
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
            subs = _sub_windows(top, days, n_sub)
            r["sub_windows"] = subs
            rows_out.append(r)
            sub_s = "  ".join(
                f"{s['win']}:{s['hit10']:.0%}/{s['mean10']:+.2%}" for s in subs
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
    out = DATA_DIR / f"_diag_parallel_gbm_signal_{ts}.csv"
    df.to_csv(out, index=False)
    (DATA_DIR / f"_diag_parallel_gbm_signal_{ts}.json").write_text(
        json.dumps(
            {"ts": ts, "eval_days": EVAL_DAYS, "topn": TOPN, "rows": df.to_dict("records")},
            indent=2, ensure_ascii=False,
        ), encoding="utf-8",
    )
    print(f"\n[saved] {out}", flush=True)
    return 0


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


if __name__ == "__main__":
    raise SystemExit(main())
