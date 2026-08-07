"""T+2/T+3 联合门 vs 纯 T+3 门 — 最近 2~3 个月干净回测 (2026-08-07).

用户问: legacy 清单用 "T+2/T+3 联合估值" 会不会更好 (并行 select_confident 联合门
`keep = (T+3>t3_min) | (T+2>t2_min 且 T+3>t3_floor)` 的逻辑能否移植).
legacy 无干净历史预测 → 用并行模块 250 天 OOS 全池 (rank_daily.parquet, 零偷看:
pred_mag 每股 130 日收缩回归校准只含当日及以前) 截取最近 N 交易日复刻生产流程:

  候选池   = 每板块按特征分 score 取 top-POOL
  旧门清单 = 池内 mag_3d > 0 (纯 T+3) → 按 mag_3d 降序 top-TOP
  联合清单 = 池内 (mag_3d>0) | (mag_2d>t2_min 且 mag_3d>t3_floor) → 按 mag_3d 降序 top-TOP
  救回带   = 池内 mag_3d∈(t3_floor,0] 且 mag_2d>t2_min (联合门多留的股)

对比已实现 MFE (real_3d/real_2d, 上涨率). WORM → BACKTEST_RESULT_DIR/joint_gate_recent_<ts>/

用法: python scripts/_diag_joint_gate_recent.py [trailing_days=60]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import BACKTEST_RESULT_DIR

SRC = (
    r"D:\AMINQT\DATA OTHERS\BACKTESTING RESULT"
    r"\parallel_rank_compare_20260807_033821\rank_daily.parquet"
)
CLS = 0.005
BOARDS = ("main", "dual")
T2_MIN, T3_FLOOR, T3_MIN = 0.01, -0.01, 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trailing_days", nargs="?", type=int, default=60)
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = BACKTEST_RESULT_DIR / f"joint_gate_recent_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(SRC)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    dates = sorted(df["date"].unique())
    keep = set(dates[-args.trailing_days:])
    df = df[df["date"].isin(keep)].reset_index(drop=True)
    # 只保留有 T+3 已实现的 (最后 ~5 日无前向窗口)
    df = df.dropna(subset=["real_3d", "real_2d", "mag_3d", "mag_2d"])
    print(f"[data] {len(df):,} 行 / {df['date'].nunique()} 交易日 "
          f"({df['date'].min().date()}..{df['date'].max().date()})", flush=True)

    rows: list[dict] = []
    for _d, g in df.groupby("date"):
        for board in BOARDS:
            bd = g[g["board"] == board]
            if len(bd) < 10:
                continue
            pool = bd.nlargest(30, "score")
            old = pool[pool["mag_3d"] > T3_MIN]
            joint = pool[
                (pool["mag_3d"] > T3_MIN)
                | ((pool["mag_2d"] > T2_MIN) & (pool["mag_3d"] > T3_FLOOR))
            ]
            rescue = pool[
                (pool["mag_3d"] > T3_FLOOR)
                & (pool["mag_3d"] <= T3_MIN)
                & (pool["mag_2d"] > T2_MIN)
            ]
            # 两种排序: mag(并行生产按 pred_mag_3d) / score(混合分=legacy d3 混合类比)
            for rank_key, label in (("mag_3d", "by_mag"), ("score", "by_score")):
                for tag, sub in (("old", old), ("joint", joint)):
                    if sub.empty:
                        continue
                    lst = sub.sort_values(rank_key, ascending=False).head(10)
                    rows.append(
                        {
                            "date": _d, "board": board, "rank": label,
                            "group": tag, "n": len(lst),
                            "ret3d": float(lst["real_3d"].mean()),
                            "ret2d": float(lst["real_2d"].mean()),
                            "win3d": float((lst["real_3d"] > 0).mean()),
                            "win3d5": float((lst["real_3d"] > CLS).mean()),
                        }
                    )
            if len(rescue) >= 2:
                rows.append(
                    {
                        "date": _d, "board": board, "rank": "rescue",
                        "group": "rescue", "n": len(rescue),
                        "ret3d": float(rescue["real_3d"].mean()),
                        "ret2d": float(rescue["real_2d"].mean()),
                        "win3d": float((rescue["real_3d"] > 0).mean()),
                        "win3d5": float((rescue["real_3d"] > CLS).mean()),
                    }
                )

    daily = pd.DataFrame(rows)
    daily.to_csv(out_dir / "daily.csv", index=False)

    summary: dict = {
        "ts": ts, "trailing_days": args.trailing_days,
        "t2_min": T2_MIN, "t3_floor": T3_FLOOR, "t3_min": T3_MIN,
        "note": "干净 OOS; real=MFE (触摸天花板口径, 非实得); "
                "legacy 无干净历史, 用并行全池复刻门逻辑",
        "boards": {},
    }
    print(f"\n===== T+2/T+3 联合门 vs 纯 T+3 门 (最近 {args.trailing_days} 交易日) =====", flush=True)
    for board in BOARDS:
        sub = daily[daily["board"] == board]
        if sub.empty:
            continue
        print(f"\n[{board}]", flush=True)
        for rank, rlabel in (("by_mag", "按 mag_3d 排序 (并行生产)"), ("by_score", "按混合分排序 (legacy 类比)"), ("rescue", "救回带 (联合门多留的股)")):
            rs = sub[sub["rank"] == rank]
            if rs.empty:
                continue
            print(f"  {rlabel}:", flush=True)
            print(f"    {'组':<8}{'日':>4}{'均n':>6}{'MFE3d':>9}{'MFE2d':>9}{'涨率3d':>9}{'涨率>0.5%':>10}", flush=True)
            for grp in ("old", "joint", "rescue"):
                g = rs[rs["group"] == grp]
                if g.empty:
                    continue
                r = g.mean(numeric_only=True)
                print(
                    f"    {grp:<8}{int(len(g)):>4}{r['n']:>6.1f}{r['ret3d']:>+9.4f}"
                    f"{r['ret2d']:>+9.4f}{r['win3d']:>9.1%}{r['win3d5']:>10.1%}",
                    flush=True,
                )
            o = rs[rs["group"] == "old"].set_index("date")["ret3d"]
            j = rs[rs["group"] == "joint"].set_index("date")["ret3d"]
            common = o.index.intersection(j.index)
            if len(common):
                win_days = int((j[common] > o[common]).sum())
                print(f"    联合门逐日 MFE3d 赢纯T+3门: {win_days}/{len(common)} 天",
                      flush=True)

    # 汇总落盘
    agg_rows = []
    for (board, rank), sub in daily.groupby(["board", "rank"]):
        for grp in ("old", "joint", "rescue"):
            g = sub[sub["group"] == grp]
            if g.empty:
                continue
            r = g.mean(numeric_only=True)
            agg_rows.append(
                {
                    "board": board, "rank": rank, "group": grp,
                    "n_days": int(len(g)), "avg_n": round(float(r["n"]), 2),
                    "mfe_3d": round(float(r["ret3d"]), 5),
                    "mfe_2d": round(float(r["ret2d"]), 5),
                    "win_3d": round(float(r["win3d"]), 4),
                    "win_3d_5p": round(float(r["win3d5"]), 4),
                }
            )
    pd.DataFrame(agg_rows).to_csv(out_dir / "agg.csv", index=False)
    summary["boards"] = agg_rows

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\nWORM: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
