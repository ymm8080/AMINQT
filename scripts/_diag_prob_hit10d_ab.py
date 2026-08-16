"""_diag_prob_hit10d_ab.py — 10d 命中概率头进排名键 A/B (2026-08-15).

问题: 生产排名键 blend = pred_mag_10d × pred_prob, 其中 pred_prob 的目标是
mfe_3d≥3% (3天视界, 生产 Platt 配方)。排名键是 10 天视界 — 用 3 天概率调制
10 天排名, 视界错配。本脚本训练一个目标 = (label_pm_10d_net > 0) (即回测报告
里"命中"的那个标签, 含成本 c2c) 的概率头, 在生产顺序 (t3 门 → 概率闸 0.08 →
排名) 下比较排名键。
方法论同 _diag_parallel_gbm_signal.py 阶段1: 单次时间切分 (训练 = 末 250 交易日
之前全部行, 预测 = 末 250 已实现交易日), 无前瞻。这是新头的信号检验, 不是生产
定案 — 若全窗/子窗不赢即关闭; 若赢 → 阶段2 walk-forward (另行)。
3 天概率沿用生产 walk-forward 检查点 (_diag_replay_wf_pred_<board>.parquet,
与 250d 生产口径复验同源), 概率闸 base_rate 逐日序列同复验脚本。

排名键: mag(关闸)=闸关闭基线; mag=纯 pred_mag_10d (生产 08-07 定案);
blend3=mag×prob_3d (生产 08-15 定案); prob10=纯 pred_prob_10d_hit;
blend10=mag×pred_prob_10d_hit。
指标: 250 已实现交易日 label_pm_10d_net 实得/命中/≥5%/≥10% + 4 子窗, TOP-5 与 TOP-10。
用法: python scripts/_diag_prob_hit10d_ab.py
WORM: data/_diag_prob_hit10d_ab_<ts>.csv/.json
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

from app.pipeline_parallel import prob_head
from app.pipeline_parallel.calibration import calibrate_mag10d
from app.pipeline_parallel.config import FUSION, SNIPER
from app.pipeline_parallel.scoring import pool_score
from config.settings import DATA_DIR, PROB_GATE

EVAL_DAYS = 250
T3_LANDED = {"main": 0.0, "dual": 0.005}
BASE_RATE_DAYS = 20
MARGIN = float(PROB_GATE["margin"])


def _load_board(board: str) -> pd.DataFrame:
    """同复验脚本: 全特征 + score + mfe_3d + 标签 (行按日期排序)."""
    fp = DATA_DIR / f"_diag_stage_{board}_3y.parquet"
    schema = pq.read_schema(str(fp)).names
    need = [
        c
        for c in schema
        if not c.startswith("label_")
        and c not in prob_head.META
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
    t = prob_head._add_mfe_3d(t)
    return t


def _train_hit10d_head(
    t: pd.DataFrame, cutoff: pd.Timestamp
) -> tuple[list[str], object]:
    """10d 命中概率头 (阶段1 单次切分): 训练 = date < cutoff 且标签可观测的行.

    目标 = (label_pm_10d_net > 0), 行掩码同 prob_head (label_pain 非 NaN = 可交易行).
    """
    cols = prob_head.feature_cols(t)
    y = (t["label_pm_10d_net"] > 0).astype(float)
    tr = (t["date"] < cutoff) & y.notna() & t["label_pain"].notna()
    x = t.loc[tr, cols].to_numpy(dtype="float32")
    if len(x) < 5000:
        raise ValueError(f"训练样本不足 ({len(x)})")
    model = LGBMClassifier(**prob_head.LGB_PARAMS)
    model.fit(x, y.loc[tr].to_numpy())
    return cols, model


def _prod_base_series(t: pd.DataFrame, dates: np.ndarray) -> pd.Series:
    """生产口径 base_rate 逐决策日序列 (同复验脚本, 无前瞻)."""
    base_map: dict[pd.Timestamp, float] = {}
    pos_all = np.searchsorted(dates, t["date"].values)
    for k, d in enumerate(dates):
        if k < BASE_RATE_DAYS + 14:
            continue
        rows = np.where((pos_all >= k - BASE_RATE_DAYS - 14) & (pos_all <= k))[0]
        tail = t.iloc[rows][["symbol", "date", "close_hfq", "high_hfq", "adv20"]].copy()
        b = prob_head._base_rate(tail)
        if b is not None:
            base_map[pd.Timestamp(d)] = b
    return pd.Series(base_map, name="base_prod")


def _eval_topn(
    g: pd.DataFrame, key: str, topn: int, days: list
) -> tuple[dict, pd.DataFrame]:
    top = (
        g.sort_values(["date", key], ascending=[True, False])
        .groupby("date", sort=True)
        .head(topn)
    )
    n = len(top)
    row = {
        "rows": n,
        "days_with_picks": int(top["date"].nunique()),
        "picks_per_day": n / len(days),
        "realized_10d": float(top["label_pm_10d_net"].mean()) if n else float("nan"),
        "hit_10d": float((top["label_pm_10d_net"] > 0).mean()) if n else float("nan"),
        "pct_ge5pct": float((top["label_pm_10d_net"] >= 0.05).mean())
        if n
        else float("nan"),
        "pct_ge10pct": (
            float((top["label_pm_10d_net"] >= 0.10).mean()) if n else float("nan")
        ),
    }
    return row, top


def _sub_windows(top: pd.DataFrame, days: list, n_sub: int) -> list[dict]:
    step = len(days) // n_sub
    subs = []
    for i in range(n_sub):
        s0, s1 = i * step, len(days) if i == n_sub - 1 else (i + 1) * step
        seg = top[top["date"].isin(days[s0:s1])]
        subs.append(
            {
                "win": f"{i + 1}/{n_sub}",
                "rows": int(len(seg)),
                "hit10": float((seg["label_pm_10d_net"] > 0).mean())
                if len(seg)
                else float("nan"),
                "mean10": float(seg["label_pm_10d_net"].mean())
                if len(seg)
                else float("nan"),
            }
        )
    return subs


def main() -> int:
    rows_out: list[dict] = []
    for board in ("main", "dual"):
        t = _load_board(board)
        dates = np.unique(t["date"].values)
        cal_test = dates[-EVAL_DAYS:]

        # ---- 10d 命中概率头 (阶段1 单次切分, 无前瞻) ----
        cutoff = pd.Timestamp(cal_test[0])
        cols10, model10 = _train_hit10d_head(t, cutoff)
        tt_mask = np.isin(t["date"].values, cal_test)
        p10 = model10.predict_proba(t.loc[tt_mask, cols10].to_numpy(dtype="float32"))[
            :, 1
        ]
        print(
            f"[{board}] 10d 命中头: 训练样本截至 {t.loc[t['date'] < cutoff, 'date'].max()}, "
            f"特征 {len(cols10)} 列, 测试 {len(p10)} 行, "
            f"概率 q25={np.percentile(p10, 25):.3f} q75={np.percentile(p10, 75):.3f}",
            flush=True,
        )

        # ---- 预测复用检查点 (250d 生产口径复验同源, 免重训) ----
        ckpt = DATA_DIR / f"_diag_replay_wf_pred_{board}.parquet"
        if not ckpt.exists():
            print(
                f"[{board}] 无预测检查点 {ckpt.name} -> skip (先跑复验脚本)", flush=True
            )
            continue
        cp = pq.read_table(str(ckpt)).to_pandas()
        pred = pd.Series(cp["pred"].to_numpy(), index=t.index, dtype="float64")

        # ---- 评估框架 (同复验: t3 门 + label_pm_10d_net 实得) ----
        work = t[
            ["symbol", "date", "board", "score", "label_pm_3d_net", "label_pm_10d_net"]
        ].copy()
        p3 = calibrate_mag10d(work, target_col="label_pm_3d_net", label_horizon=3)
        p10m = calibrate_mag10d(work, target_col="label_pm_10d_net", label_horizon=10)
        mm = work.merge(
            p3.drop(columns=["board"]).rename(columns={"mag": "pred_ret_3d"}),
            on=["symbol", "date"],
            how="inner",
        ).merge(
            p10m.drop(columns=["board"]).rename(columns={"mag": "pred_mag_10d"}),
            on=["symbol", "date"],
            how="inner",
        )
        mm["date"] = pd.to_datetime(mm["date"])
        rr = mm.dropna(subset=["label_pm_10d_net"])
        days = sorted(rr["date"].unique())[-EVAL_DAYS:]
        rr = rr[rr["date"].isin(days)].reset_index(drop=True)

        sub = t.loc[np.isin(t["date"].values, cal_test), ["symbol", "date"]].copy()
        sub["pred_prob"] = pred[sub.index].to_numpy()
        sub["pred_prob_10d"] = p10
        rr = rr.merge(sub, on=["symbol", "date"], how="left")
        rr = rr.merge(
            _prod_base_series(t, dates), left_on="date", right_index=True, how="left"
        )

        # ---- 生产顺序: t3 门 → 概率闸 → 排名 ----
        pool = rr[rr["pred_ret_3d"] > T3_LANDED[board]].copy()
        keep_gate = (pool["pred_prob"] > pool["base_prod"] + MARGIN) | pool[
            "pred_prob"
        ].isna()
        gated = pool[keep_gate].copy()
        for df in (pool, gated):
            df["rank_mag"] = df["pred_mag_10d"]
            df["rank_prob10"] = df["pred_prob_10d"]
            df["rank_blend3"] = df["pred_mag_10d"] * df["pred_prob"]
            df["rank_blend10"] = df["pred_mag_10d"] * df["pred_prob_10d"]

        combos = [
            ("mag(关闸)", pool, "rank_mag"),
            ("mag", gated, "rank_mag"),
            ("blend3=mag×prob3", gated, "rank_blend3"),
            ("prob10", gated, "rank_prob10"),
            ("blend10=mag×prob10", gated, "rank_blend10"),
        ]
        n_sub = max(2, EVAL_DAYS // 60)
        print(
            f"\n===== {board} 末 250 已实现交易日 (t3 {T3_LANDED[board]:.2%}, "
            f"闸 +{MARGIN:.2f}) =====",
            flush=True,
        )
        for topn in (5, 10):
            print(f"--- TOP-{topn} ---", flush=True)
            for gname, g, key in combos:
                r, top = _eval_topn(g, key, topn, days)
                r["board"] = board
                r["topn"] = topn
                r["rank_key"] = gname
                r["sub_windows"] = _sub_windows(top, days, n_sub)
                rows_out.append(r)
                sub_s = "  ".join(
                    f"{s['win']}:{s['hit10']:.0%}/{s['mean10']:+.2%}"
                    for s in r["sub_windows"]
                )
                print(
                    f"{gname:>18} 票/日{r['picks_per_day']:>5.2f} "
                    f"实得{r['realized_10d']:>+8.2%} 命中{r['hit_10d']:>6.0%} "
                    f"≥5%{r['pct_ge5pct']:>5.0%} ≥10%{r['pct_ge10pct']:>5.0%}",
                    flush=True,
                )
                print(f"    sub: {sub_s}", flush=True)

    if not rows_out:
        print("[error] 无任何板块可评估", flush=True)
        return 1
    df = pd.DataFrame(rows_out)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = DATA_DIR / f"_diag_prob_hit10d_ab_{ts}.csv"
    df.to_csv(out, index=False)
    (DATA_DIR / f"_diag_prob_hit10d_ab_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "eval_days": EVAL_DAYS,
                "t3_landed": T3_LANDED,
                "margin": MARGIN,
                "note": "10d 命中概率头 (label_pm_10d_net>0) 进排名键 A/B, 阶段1 单次"
                "切分 (训练=末250日前, 预测=末250日); blend3=生产现状 mag×prob_3d; "
                "生产顺序 t3 门 → 概率闸(margin) → 排名; 3d 概率复用 "
                "_diag_replay_wf_pred_<board>.parquet 检查点",
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
