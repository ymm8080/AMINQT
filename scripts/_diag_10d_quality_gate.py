"""10d 预测作为【质量门】是否有真实增益 — 并行 250d 干净 OOS (2026-08-07).

用户问: 做 10d 预测能否挑出更高质量股. 先不建 legacy 10d 模型 —
并行 rank_daily.parquet 已有 mag_10d/real_10d (同 MFE 口径), 免费测:
  池   = 每板块按特征分 score 取 top-30
  组A  = top-10 by mag_3d                     (纯 T+3 幅度排名, legacy 现状方向)
  组B  = top-10 by t35=0.5·mag_3d+0.5·mag_5d  (上一轮定案的 T+3/T+5 组合排名键)
  组C  = 池内 mag_10d>0 → top-10 by t35       (10d 质量门: 要求 10 日趋势仍向上)
  组D  = top-10 by mag_10d                    (纯 10d 幅度排名, 10d 是否本身最强判别)

结局 = real_comp (模块自身加权 MFE) + real_3d/5d/10d + 上涨率. WORM.

用法: python scripts/_diag_10d_quality_gate.py [trailing_days=250]
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
BOARDS = ("main", "dual")
POOL_N, TOP_N = 30, 10
W = {"2d": 0.25, "3d": 0.40, "5d": 0.25, "10d": 0.10}
T35 = (0.5, 0.5)  # t35 组合权重 (mag_3d, mag_5d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trailing_days", nargs="?", type=int, default=250)
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = BACKTEST_RESULT_DIR / f"diag_10d_gate_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(SRC)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    dates = sorted(df["date"].unique())
    keep = set(dates[-args.trailing_days :])
    df = df[df["date"].isin(keep)].reset_index(drop=True)
    need = ["real_3d", "real_5d", "real_10d", "mag_3d", "mag_5d", "mag_10d"]
    df = df.dropna(subset=need)
    df["real_comp"] = (
        W["2d"] * df["real_2d"]
        + W["3d"] * df["real_3d"]
        + W["5d"] * df["real_5d"]
        + W["10d"] * df["real_10d"]
    )
    df["t35"] = T35[0] * df["mag_3d"] + T35[1] * df["mag_5d"]
    print(
        f"[data] {len(df):,} 行 / {df['date'].nunique()} 交易日 "
        f"({df['date'].min().date()}..{df['date'].max().date()})",
        flush=True,
    )

    KEYS = (
        ("mag_3d", "纯 mag_3d 排名 (现状方向)"),
        ("t35", "T+3/T+5 组合排名 (已定案)"),
        ("gate_10d_t35", "10d门+组合排名 (10d>0 再按 t35)"),
        ("mag_10d", "纯 mag_10d 排名"),
    )
    rows: list[dict] = []
    for _d, g in df.groupby("date"):
        for board in BOARDS:
            bd = g[g["board"] == board]
            if len(bd) < POOL_N:
                continue
            pool = bd.nlargest(POOL_N, "score")
            groups = {
                "mag_3d": (pool, "mag_3d"),
                "t35": (pool, "t35"),
                "gate_10d_t35": (pool[pool["mag_10d"] > 0], "t35"),
                "mag_10d": (pool, "mag_10d"),
            }
            for key, label in KEYS:
                sub, sort_by = groups[key]
                if sub.empty:
                    continue
                lst = sub.sort_values(sort_by, ascending=False).head(TOP_N)
                if len(lst) < 5:
                    continue
                rows.append(
                    {
                        "date": _d,
                        "board": board,
                        "key": key,
                        "label": label,
                        "n": len(lst),
                        "comp": float(lst["real_comp"].mean()),
                        "ret3d": float(lst["real_3d"].mean()),
                        "ret5d": float(lst["real_5d"].mean()),
                        "ret10d": float(lst["real_10d"].mean()),
                        "win_comp": float((lst["real_comp"] > 0).mean()),
                        "win3d": float((lst["real_3d"] > 0).mean()),
                        "win5d": float((lst["real_5d"] > 0).mean()),
                    }
                )

    daily = pd.DataFrame(rows)
    daily.to_csv(out_dir / "daily.csv", index=False)

    agg_rows = []
    print(
        f"\n===== 10d 质量门 vs 纯 3d / T3+T5 组合 (最近 {args.trailing_days} 交易日) =====",
        flush=True,
    )
    for board in BOARDS:
        sub = daily[daily["board"] == board]
        if sub.empty:
            continue
        print(f"\n[{board}]  (结局=模块自身 real_comp 加权 MFE)", flush=True)
        print(
            f"  {'组':<24}{'日':>4}{'均n':>6}{'comp':>9}{'3d':>9}{'5d':>9}{'10d':>9}"
            f"{'comp涨率':>9}{'5d涨率':>9}",
            flush=True,
        )
        for key, label in KEYS:
            g = sub[sub["key"] == key]
            if g.empty:
                continue
            r = g.mean(numeric_only=True)
            print(
                f"  {label:<24}{int(len(g)):>4}{r['n']:>6.1f}{r['comp']:>+9.4f}"
                f"{r['ret3d']:>+9.4f}{r['ret5d']:>+9.4f}{r['ret10d']:>+9.4f}"
                f"{r['win_comp']:>9.1%}{r['win5d']:>9.1%}",
                flush=True,
            )
            agg_rows.append(
                {
                    "board": board,
                    "key": key,
                    "label": label,
                    "n_days": int(len(g)),
                    "avg_n": round(float(r["n"]), 2),
                    "comp": round(float(r["comp"]), 5),
                    "ret3d": round(float(r["ret3d"]), 5),
                    "ret5d": round(float(r["ret5d"]), 5),
                    "ret10d": round(float(r["ret10d"]), 5),
                    "win_comp": round(float(r["win_comp"]), 4),
                    "win5d": round(float(r["win5d"]), 4),
                }
            )
        # 逐日配对: 门 vs 组合基准
        base = sub[sub["key"] == "t35"].set_index("date")["comp"]
        for key, label in (("gate_10d_t35", "10d门"), ("mag_10d", "纯10d")):
            other = sub[sub["key"] == key].set_index("date")["comp"]
            common = base.index.intersection(other.index)
            if len(common):
                wins = int((other[common] > base[common]).sum())
                print(
                    f"  逐日 {label} comp 赢 T3+T5 组合: {wins}/{len(common)} 天",
                    flush=True,
                )

    pd.DataFrame(agg_rows).to_csv(out_dir / "agg.csv", index=False)
    summary = {
        "ts": ts,
        "trailing_days": args.trailing_days,
        "horizon_w": W,
        "t35": T35,
        "note": "干净 OOS; real=MFE (触摸天花板口径, 非实得); 门=mag_10d>0 准入再按组合排名",
        "boards": agg_rows,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\nWORM: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
