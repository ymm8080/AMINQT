"""_diag_legacy_parprob_ab.py — 并行概率头迁移测试: legacy dual 250d 重放 (2026-08-15).

背景: legacy blend A/B 证伪 (dual cls 概率头 22 唯一值太粗). 用户方向: legacy 需要
并行式概率头. 先做零训练迁移测试 — 把并行 PR 1aaa380e 的 walk-forward 概率头预测
(data/_diag_replay_wf_pred_dual.parquet, 与 _diag_stage_dual_3y.parquet 行对齐,
同 _diag_prob_rank_ab 用法) 按 symbol×date 接到 legacy 重放 CSV 上, 测 4 组合:
  cur      = legacy 生产 (pob>+0.08 & pain≤0.4), rank mag          (基线)
  par_gate = 并行概率头闸 (par_prob > base_par+0.08) & pain≤0.4, rank mag
  par_blend= 并行概率头闸 同左, rank mag × par_prob
  cur_blend= legacy 生产闸, rank mag × par_prob (只换排名键)
指标: TOP-5/10/15 × full/126d/63d 命中/实得/中位/≥5%/≥10% + 4 等分子窗.
WORM: DATA_OTHERS/diag/legacy_parprob_ab_<ts>.csv/.json
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config.settings import DATA_DIR, PROB_GATE, data_others_path

PROB_MARGIN = 0.08  # legacy landed dual 边际
PAIN_MAX = 0.4
BASE_RATE_DAYS = int(PROB_GATE["base_rate_days"])  # 20
MARGIN = float(PROB_GATE["margin"])  # 0.08 (并行生产口径)
DEPTHS = (5, 10, 15)
WINDOWS = {"full": 10**9, "126d": 126, "63d": 63}
N_SUB = 4


def _stats(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {
            "n_days": 0,
            "picks": 0,
            "avg_picks": 0.0,
            "hit": float("nan"),
            "mean": float("nan"),
            "med": float("nan"),
            "ge5": float("nan"),
            "ge10": float("nan"),
        }
    r = sub["realized_net"].dropna()
    return {
        "n_days": int(sub["date"].nunique()),
        "picks": int(len(sub)),
        "avg_picks": float(len(sub) / max(1, sub["date"].nunique())),
        "hit": float((r > 0).mean()) if len(r) else float("nan"),
        "mean": float(r.mean()) if len(r) else float("nan"),
        "med": float(r.median()) if len(r) else float("nan"),
        "ge5": float((r >= 0.05).mean()) if len(r) else float("nan"),
        "ge10": float((r >= 0.10).mean()) if len(r) else float("nan"),
    }


def _sub_windows(top: pd.DataFrame) -> list[dict]:
    dates = np.sort(top["date"].unique())
    step = max(1, len(dates) // N_SUB)
    subs = []
    for i in range(N_SUB):
        s0, s1 = i * step, len(dates) if i == N_SUB - 1 else (i + 1) * step
        seg = top[top["date"].isin(dates[s0:s1])]
        subs.append(
            {
                "win": f"{i + 1}/{N_SUB}",
                "rows": int(len(seg)),
                "hit": float((seg["realized_net"] > 0).mean())
                if len(seg)
                else float("nan"),
                "mean": float(seg["realized_net"].mean()) if len(seg) else float("nan"),
            }
        )
    return subs


def _prod_base_series(t: pd.DataFrame, dates: np.ndarray) -> pd.Series:
    """并行生产口径 base_rate 逐日序列 (同 _diag_prob_rank_ab, 无前瞻)."""
    from app.pipeline_parallel import prob_head

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
    return pd.Series(base_map, name="base_par")


def main() -> int:
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else sorted(
            glob.glob(str(data_others_path("diag") / "legacy_hitrate_topn_*.csv"))
        )[-1]
    )
    df = pd.read_csv(path, dtype={"symbol": str})
    df["date"] = pd.to_datetime(df["date"])
    b = df[df["board"] == "dual"].copy()
    dates_all = np.sort(b["date"].unique())

    # ---- 并行 walk-forward 概率头 (行对齐 stage 检查点, 同 _diag_prob_rank_ab) ----
    # 对齐修复 (2026-08-15): stage parquet 在预测检查点生成后被追加 8 只 STAR 股
    # 全史行 (尾部无序块, 与头块零碰撞) → 按文件序截断到检查点行数再排序,
    # 即复现复验脚本当时的行序 (append-only row groups 不改变原行序).
    stage = DATA_DIR / "_diag_stage_dual_3y.parquet"
    ckpt = DATA_DIR / "_diag_replay_wf_pred_dual.parquet"
    if not (stage.exists() and ckpt.exists()):
        print("[error] 缺并行 stage/预测检查点 -> 先跑并行复验脚本", flush=True)
        return 1
    cp = pq.read_table(str(ckpt), columns=["pred"]).to_pandas()["pred"]
    n_ckpt = len(cp)
    t = pq.read_table(
        str(stage), columns=["symbol", "date", "close_hfq", "high_hfq", "adv20"]
    ).to_pandas()
    print(
        f"[align] stage {len(t)} 行 -> 截断 {n_ckpt} (检查点长度), 尾部追加 "
        f"{len(t) - n_ckpt} 行弃用",
        flush=True,
    )
    t = t.iloc[:n_ckpt]
    t["symbol"] = t["symbol"].astype(str)
    t["date"] = pd.to_datetime(t["date"])
    t = t.sort_values(["date", "symbol"]).reset_index(drop=True)
    pred = pd.Series(cp.to_numpy(), index=t.index, dtype="float64")
    t["pred_prob"] = pred.to_numpy()
    # 结构自检: pred 只应在复验 250 日评估窗内非 NaN, 且值域 [0,1]
    uniq = np.unique(t["date"].values)
    eval_dates = set(uniq[-250:])
    bad = t["pred_prob"].notna() & ~t["date"].isin(eval_dates)
    assert not bad.any(), f"对齐错误: {bad.sum()} 行 pred 落在评估窗外"
    assert (t["pred_prob"].dropna() >= 0).all() and (t["pred_prob"].dropna() <= 1).all()
    print(
        f"[align] OK: pred 非 NaN {t['pred_prob'].notna().sum()} 行全部落在末 250 日窗内",
        flush=True,
    )
    dates = np.unique(t["date"].values)
    base = _prod_base_series(t, dates)
    t = t.merge(base.rename("base_par"), left_on="date", right_index=True, how="left")
    join = t[["symbol", "date", "pred_prob", "base_par"]].drop_duplicates(
        ["symbol", "date"]
    )
    n0 = len(b)
    b = b.merge(join, on=["symbol", "date"], how="left")
    cov = int(b["pred_prob"].notna().sum())
    print(
        f"[join] legacy dual 候选 {n0} 票 × 并行概率头: 命中 {cov} "
        f"({cov / n0:.0%}), base_par 缺失 {b['base_par'].isna().sum()}",
        flush=True,
    )

    # ---- 组合 ----
    b["pob"] = b["prob"] - b["base_rate"]
    b["rank_mag"] = b["pred_ret_10d"]
    b["rank_blend"] = b["pred_ret_10d"] * b["pred_prob"]
    combos = [
        (
            "cur",
            b[(b["pob"] > PROB_MARGIN) & (b["pain_prob"].fillna(0) <= PAIN_MAX)],
            "rank_mag",
        ),
        (
            "par_gate",
            b[
                (b["pred_prob"] > b["base_par"] + MARGIN)
                & (b["pain_prob"].fillna(0) <= PAIN_MAX)
            ],
            "rank_mag",
        ),
        (
            "par_blend",
            b[
                (b["pred_prob"] > b["base_par"] + MARGIN)
                & (b["pain_prob"].fillna(0) <= PAIN_MAX)
            ],
            "rank_blend",
        ),
        (
            "cur_blend",
            b[(b["pob"] > PROB_MARGIN) & (b["pain_prob"].fillna(0) <= PAIN_MAX)],
            "rank_blend",
        ),
    ]
    rows: list[dict] = []
    for gname, pool, rkcol in combos:
        print(
            f"\n[{gname}] 池 {len(pool)} 票 / {pool['date'].nunique()} 出票日",
            flush=True,
        )
        for depth in DEPTHS:
            for wname, wdays in WINDOWS.items():
                cutoff = dates_all[0] if wdays >= len(dates_all) else dates_all[-wdays]
                w = pool[pool["date"].values >= cutoff]
                top = (
                    w.sort_values(["date", rkcol], ascending=[True, False])
                    .groupby("date", sort=False)
                    .head(depth)
                )
                s = _stats(top)
                rows.append({"combo": gname, "depth": depth, "window": wname, **s})
                sub_s = "  ".join(
                    f"{x['win']}:{x['hit']:.0%}/{x['mean']:+.2%}"
                    for x in _sub_windows(top)
                )
                print(
                    f"  top{depth:>2}/{wname:>4} 日{s['n_days']:>3} "
                    f"票/日{s['avg_picks']:>5.2f} 命中{s['hit']:>6.1%} "
                    f"实得{s['mean']:>+8.2%} ≥5%{s['ge5']:>6.1%} ≥10%{s['ge10']:>6.1%}",
                    flush=True,
                )
                print(f"    sub: {sub_s}", flush=True)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    pd.DataFrame(rows).to_csv(out_dir / f"legacy_parprob_ab_{ts}.csv", index=False)
    (out_dir / f"legacy_parprob_ab_{ts}.json").write_text(
        json.dumps(
            {
                "ts": ts,
                "source": os.path.basename(path),
                "joined": f"{cov}/{n0}",
                "parallel_margin": MARGIN,
                "legacy_margin": PROB_MARGIN,
                "pain_max": PAIN_MAX,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\n[saved] {out_dir}/legacy_parprob_ab_{ts}.csv/.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
