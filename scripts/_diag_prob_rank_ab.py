"""_diag_prob_rank_ab.py — 概率头进排名键 A/B (2026-08-15).

问题: 概率头 (mfe_3d≥3% 真模型) 目前只当闸用; 排名键 = 纯 pred_mag_10d (08-07 定案).
本脚本在生产顺序 (t3 门 → 概率闸 0.08 → 排名) 下, 比较排名键掺入 pred_prob 是否提升
TOP-3/5/10 质量。预测复用 data/_diag_replay_wf_pred_<board>[_<tag>].parquet 检查点
(免重训, 与 250d 生产口径复验 _diag_parallel_gbm_prod_replay.py 同源; --tag 选杠杆检查点)。
排名键: mag10(关闸)=生产闸关闭基线; mag10=纯 pred_mag_10d (生产现状); mag5/mag3=纯校准;
prob=纯 pred_prob; 矩阵 cell=mag{3,5,10}×prob。
指标: 250 已实现交易日 label_pm_10d_net 实得/命中/≥5%/≥10% + 4 子窗,
TOP-3/5/10 三深度 (08-22 用户定案: rank A/B 看 blend vs mag 在 3/5/10 深度)。
2026-08-22 用户定案"NO NEED FRESH RAINING": 只乘已有值 — prob=生产 3d 头 l0 检查点,
mag3/5/10=calibrate_mag10d 三视界 (纯校准无训练)。新训练 (prob5/prob10 头) 已取消。
用法: python scripts/_diag_prob_rank_ab.py [--tag <replay检查点后缀>]
WORM: data/_diag_prob_rank_ab_<ts>.csv/.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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
    need += [
        "symbol",
        "date",
        "label_pain",
        "label_pm_3d_net",
        "label_pm_5d_net",
        "label_pm_10d_net",
    ]
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
    ap = argparse.ArgumentParser(
        description="并行排名键 A/B (blend vs mag, 3/5/10 深度)"
    )
    ap.add_argument("--prob3-tag", default="l0", help="prob3 检查点后缀 (生产 3d 头)")
    ap.add_argument("--prob5-tag", default="h5", help="prob5 检查点后缀 (5d 头)")
    ap.add_argument("--prob10-tag", default="h10", help="prob10 检查点后缀 (10d 头)")
    ap.add_argument(
        "--eval-days",
        type=int,
        default=EVAL_DAYS,
        help="评估的已实现决策日数 (60 初筛 / 125 确认)",
    )
    ap.add_argument(
        "--no-pure-prob",
        action="store_true",
        help="不评估纯 prob 排名键 (只留 mag 与 mag×prob, 08-23 用户定案)",
    )
    args = ap.parse_args()
    eval_days = args.eval_days
    rows_out: list[dict] = []
    for board in ("main", "dual"):
        t = _load_board(board)
        dates = np.unique(t["date"].values)
        cal_test = dates[-eval_days:]

        # ---- 预测复用检查点 (prob3=生产 3d 头; prob5/prob10=新训 5d/10d 头; 位置对齐 t.index) ----
        prob_cols: dict[str, pd.Series] = {}
        for ptag, colname in (
            (args.prob3_tag, "prob3"),
            (args.prob5_tag, "prob5"),
            (args.prob10_tag, "prob10"),
        ):
            suffix = f"_{ptag}" if ptag else ""
            ckpt = DATA_DIR / f"_diag_replay_wf_pred_{board}{suffix}.parquet"
            if not ckpt.exists():
                print(
                    f"[{board}] 无 {colname} 检查点 {ckpt.name} -> 跳过该列", flush=True
                )
                continue
            cp = pq.read_table(str(ckpt)).to_pandas()
            prob_cols[colname] = pd.Series(
                cp["pred"].to_numpy(), index=t.index, dtype="float64"
            )
        if "prob3" not in prob_cols:
            print(f"[{board}] 无 prob3 检查点 (生产头) -> skip", flush=True)
            continue

        # ---- 评估框架 (同复验: t3 门 + label_pm_10d_net 实得) ----
        # mag3/5/10 = calibrate_mag10d 三视界 (纯校准, 无训练); prob = 已有 prob3 值
        work = t[
            [
                "symbol",
                "date",
                "board",
                "score",
                "label_pm_3d_net",
                "label_pm_5d_net",
                "label_pm_10d_net",
            ]
        ].copy()
        p3 = calibrate_mag10d(work, target_col="label_pm_3d_net", label_horizon=3)
        p5 = calibrate_mag10d(work, target_col="label_pm_5d_net", label_horizon=5)
        p10 = calibrate_mag10d(work, target_col="label_pm_10d_net", label_horizon=10)
        mm = (
            work.merge(
                p3.drop(columns=["board"]).rename(columns={"mag": "pred_mag_3d"}),
                on=["symbol", "date"],
                how="inner",
            )
            .merge(
                p5.drop(columns=["board"]).rename(columns={"mag": "pred_mag_5d"}),
                on=["symbol", "date"],
                how="inner",
            )
            .merge(
                p10.drop(columns=["board"]).rename(columns={"mag": "pred_mag_10d"}),
                on=["symbol", "date"],
                how="inner",
            )
        )
        mm["date"] = pd.to_datetime(mm["date"])
        rr = mm.dropna(subset=["label_pm_10d_net"])
        days = sorted(rr["date"].unique())[-eval_days:]
        rr = rr[rr["date"].isin(days)].reset_index(drop=True)

        sub = t.loc[np.isin(t["date"].values, cal_test), ["symbol", "date"]].copy()
        for colname, s in prob_cols.items():
            sub[colname] = s[sub.index].to_numpy()
        rr = rr.merge(sub, on=["symbol", "date"], how="left")
        rr = rr.merge(
            _prod_base_series(t, dates), left_on="date", right_index=True, how="left"
        )

        # ---- 生产顺序: t3 门 → 概率闸 → 排名 (闸固定用生产 prob3) ----
        pool = rr[rr["pred_mag_3d"] > T3_LANDED[board]].copy()
        keep_gate = (pool["prob3"] > pool["base_prod"] + MARGIN) | pool["prob3"].isna()
        gated = pool[keep_gate].copy()
        mag_cols = [
            ("mag10", "pred_mag_10d"),
            ("mag5", "pred_mag_5d"),
            ("mag3", "pred_mag_3d"),
        ]
        for df in (pool, gated):
            for mname, mcol in mag_cols:
                df[f"rank_{mname}"] = df[mcol]
            for pname in prob_cols:
                df[f"rank_{pname}"] = df[pname]
                for mname, mcol in mag_cols:
                    df[f"rank_{mname}x_{pname}"] = df[mcol] * df[pname]

        combos = [
            ("mag10(关闸)", pool, "rank_mag10"),
            ("mag10", gated, "rank_mag10"),
            ("mag5", gated, "rank_mag5"),
            ("mag3", gated, "rank_mag3"),
        ]
        for pname in prob_cols:
            if not args.no_pure_prob:
                combos.append((pname, gated, f"rank_{pname}"))
            for mname, _ in mag_cols:
                combos.append((f"{mname}×{pname}", gated, f"rank_{mname}x_{pname}"))
        n_sub = max(2, eval_days // 60)
        print(
            f"\n===== {board} 末 {eval_days} 已实现交易日 (t3 {T3_LANDED[board]:.2%}, "
            f"闸 +{MARGIN:.2f}) =====",
            flush=True,
        )
        for topn in (
            3,
            5,
            10,
        ):  # 08-22 用户定案: rank A/B 看 blend vs mag 在 3/5/10 深度
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
                    f"{gname:>10} 票/日{r['picks_per_day']:>5.2f} "
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
    out = DATA_DIR / f"_diag_prob_rank_ab_{ts}.csv"
    df.to_csv(out, index=False)
    (DATA_DIR / f"_diag_prob_rank_ab_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "eval_days": eval_days,
                "t3_landed": T3_LANDED,
                "margin": MARGIN,
                "note": "排名键矩阵 A/B (2026-08-22 用户定案: 只乘已有值, 无新训练): "
                "mag10=纯 pred_mag_10d (生产现状); mag5/mag3=calibrate_mag10d 视界5/3 纯校准; "
                "prob=纯 pred_prob (生产 3d 头); 矩阵=mag{3,5,10}×prob. "
                "生产顺序 t3 门 → 概率闸(margin) → 排名; "
                "预测复用 _diag_replay_wf_pred_<board>[_<tag>].parquet 检查点 (l0=prob3)",
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
