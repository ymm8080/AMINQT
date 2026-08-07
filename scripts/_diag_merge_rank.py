"""T+2/T+3 merge 估值作为【排名键】是否更好 — 并行 250d 干净 OOS 复测 (2026-08-07).

用户问: legacy 清单用 "T+2/T+3 联合估值" 会不会更好.
上一轮已证: 作为【准入门】(select_confident 联合门)在 top-10 短名单上是结构性 no-op
(并行 pred_mag 由 score 每股校准派生, top-30 池 mag_3d 全正, 救回带 0 只, 40/60/120 天一致).

本脚本测另一层解释: 把多视界幅度合并成一个"估值分"当排名键, 是否比纯 mag_3d 排名好:
  键A  mag_3d                      (并行生产现状)
  键B  mag_2d                      (单视界对照)
  键C  pred_comp = 0.25·mag_2d+0.40·mag_3d+0.25·mag_5d+0.10·mag_10d  (模块自身 horizon_w 合并)
  键D  pred_t23 = 0.5·mag_2d+0.5·mag_3d   (T+2/T+3 两视界合并 = 用户原话)
  键E  0.5·norm(score)+0.5·norm(mag_3d)  (把握度×幅度混合 = legacy d3 混合的类比; score/mag 近正交)

结局用模块自身目标口径 real_comp = 0.25·real_2d+0.40·real_3d+0.25·real_5d+0.10·real_10d
+ real_3d / real_2d / 上涨率. 干净 OOS 零偷看. WORM → BACKTEST_RESULT_DIR/merge_rank_<ts>/

用法: python scripts/_diag_merge_rank.py [trailing_days=60]
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
W = {"2d": 0.25, "3d": 0.40, "5d": 0.25, "10d": 0.10}  # 与 config SHORTLIST_SCORE horizon_w 一致


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["pred_comp"] = (
        W["2d"] * d["mag_2d"] + W["3d"] * d["mag_3d"]
        + W["5d"] * d["mag_5d"] + W["10d"] * d["mag_10d"]
    )
    d["real_comp"] = (
        W["2d"] * d["real_2d"] + W["3d"] * d["real_3d"]
        + W["5d"] * d["real_5d"] + W["10d"] * d["real_10d"]
    )
    d["pred_t23"] = 0.5 * d["mag_2d"] + 0.5 * d["mag_3d"]
    s = d["score"].rank(pct=True)
    m = d["mag_3d"].rank(pct=True)
    d["blend"] = 0.5 * s + 0.5 * m
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trailing_days", nargs="?", type=int, default=60)
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = BACKTEST_RESULT_DIR / f"merge_rank_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(SRC)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    dates = sorted(df["date"].unique())
    keep = set(dates[-args.trailing_days:])
    df = df[df["date"].isin(keep)].reset_index(drop=True)
    need = ["real_2d", "real_3d", "real_5d", "real_10d",
            "mag_2d", "mag_3d", "mag_5d", "mag_10d"]
    df = df.dropna(subset=need)
    df = _keys(df)
    print(f"[data] {len(df):,} 行 / {df['date'].nunique()} 交易日 "
          f"({df['date'].min().date()}..{df['date'].max().date()})", flush=True)

    KEYS = (
        ("mag_3d", "纯 mag_3d (并行现状)"),
        ("mag_2d", "纯 mag_2d"),
        ("pred_comp", "T+2/3/5/10 加权合并 (模块 horizon_w)"),
        ("pred_t23", "T+2/T+3 两视界合并"),
        ("blend", "把握度×幅度混合 (legacy d3 类比)"),
    )
    rows: list[dict] = []
    for _d, g in df.groupby("date"):
        for board in BOARDS:
            bd = g[g["board"] == board]
            if len(bd) < POOL_N:
                continue
            pool = bd.nlargest(POOL_N, "score")
            for key, label in KEYS:
                lst = pool.sort_values(key, ascending=False).head(TOP_N)
                rows.append(
                    {
                        "date": _d, "board": board, "key": key, "label": label,
                        "n": len(lst),
                        "comp": float(lst["real_comp"].mean()),
                        "ret3d": float(lst["real_3d"].mean()),
                        "ret2d": float(lst["real_2d"].mean()),
                        "win_comp": float((lst["real_comp"] > 0).mean()),
                        "win3d": float((lst["real_3d"] > 0).mean()),
                    }
                )

    daily = pd.DataFrame(rows)
    daily.to_csv(out_dir / "daily.csv", index=False)

    agg_rows = []
    print(f"\n===== T+2/T+3 merge 排名键 vs 纯 mag_3d (最近 {args.trailing_days} 交易日) =====", flush=True)
    for board in BOARDS:
        sub = daily[daily["board"] == board]
        if sub.empty:
            continue
        print(f"\n[{board}]  (结局=模块自身目标口径 real_comp 加权 MFE)", flush=True)
        print(f"  {'键':<28}{'日':>4}{'comp':>9}{'3d':>9}{'2d':>9}{'comp涨率':>9}{'3d涨率':>9}", flush=True)
        for key, label in KEYS:
            g = sub[sub["key"] == key]
            if g.empty:
                continue
            r = g.mean(numeric_only=True)
            print(
                f"  {label:<28}{int(len(g)):>4}{r['comp']:>+9.4f}{r['ret3d']:>+9.4f}"
                f"{r['ret2d']:>+9.4f}{r['win_comp']:>9.1%}{r['win3d']:>9.1%}",
                flush=True,
            )
            agg_rows.append(
                {
                    "board": board, "key": key, "label": label,
                    "n_days": int(len(g)),
                    "comp": round(float(r["comp"]), 5),
                    "ret3d": round(float(r["ret3d"]), 5),
                    "ret2d": round(float(r["ret2d"]), 5),
                    "win_comp": round(float(r["win_comp"]), 4),
                    "win3d": round(float(r["win3d"]), 4),
                }
            )
        # 逐日配对: 每个 merge 键 vs mag_3d
        base = sub[sub["key"] == "mag_3d"].set_index("date")["comp"]
        for key, label in (("pred_comp", "合并键"), ("blend", "混合键")):
            other = sub[sub["key"] == key].set_index("date")["comp"]
            common = base.index.intersection(other.index)
            if len(common):
                wins = int((other[common] > base[common]).sum())
                print(f"  逐日 {label} comp 赢 mag_3d: {wins}/{len(common)} 天", flush=True)

    pd.DataFrame(agg_rows).to_csv(out_dir / "agg.csv", index=False)
    summary = {
        "ts": ts, "trailing_days": args.trailing_days,
        "horizon_w": W,
        "note": "干净 OOS; real=MFE (触摸天花板口径); "
                "pred_comp/blend 均为每日池内预测值排名, 无偷看",
        "boards": agg_rows,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    print(f"\nWORM: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
