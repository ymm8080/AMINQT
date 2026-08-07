"""诊断: 高上涨率 + 高幅度 短名单机制对比 (2026-08-07 用户).

用户: "我要的是高上涨率的高上涨幅度的股票筛选机制..所以选PREDICTION还是特征 RANKING"

读 _diag_parallel_rank_compare.py 落盘的 rank_daily.parquet (逐日逐股
score/mag_h/prob_h/已实现MFE), 离线对比排名策略 (主视界 3d, TOP-N=10):

  A 特征排名        score 降序 TOP-10                (把握度基准)
  B 预测排名        mag_3d 降序 TOP-10               (幅度基准)
  D 两段-把握优先   score TOP-30 → 池内 mag_3d 降序 TOP-10
  E 两段-幅度优先   mag_3d TOP-30 → 池内 score 降序 TOP-10
  F 混合排名        0.5·z(score)+0.5·z(mag_3d) TOP-10
  G 两段-幅度门槛   score TOP-30 → 池内 mag_3d>0 过滤 → score 降序 TOP-10
                    (特征保把握, 预测只做负预期过滤)

目标 = 同时高上涨率(MFE>0 占比) 高幅度(平均 MFE). 只看 3d 已实现.
输出: 终端表格 + summary.json (WORM).
用法: python scripts/_diag_rank_strategies.py <run_dir>
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import BACKTEST_RESULT_DIR

TOP_N = 10
STAGE1_N = 30
H = "3d"
ABS_TARGET = {"2d": 0.02, "3d": 0.03, "5d": 0.04, "10d": 0.06}
CLS_THRESHOLD = 0.005


def pick_top(df: pd.DataFrame, key: str, n: int = TOP_N) -> pd.DataFrame:
    return df.sort_values(key, ascending=False).head(n)


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else None
    if run is None:
        dirs = sorted((BACKTEST_RESULT_DIR).glob("parallel_rank_compare_*"))
        if not dirs:
            print("[fatal] 无 parallel_rank_compare_* 目录")
            return 1
        run = str(dirs[-1])
    src = Path(run) / "rank_daily.parquet"
    if not src.exists():
        print(f"[fatal] 缺 {src} (需重跑 _diag_parallel_rank_compare.py 生成)")
        return 1
    df = pd.read_parquet(src)
    mag, prob, real = f"mag_{H}", f"prob_{H}", f"real_{H}"
    df = df.dropna(subset=[mag, prob, real]).copy()
    print(
        f"[load] {len(df):,}r / 评估 {df['date'].nunique()} 日 (视界 {H})", flush=True
    )

    strategies = {
        "A_feature10": lambda g: pick_top(g, "score"),
        "B_predmag10": lambda g: pick_top(g, mag),
        "D_win_first": lambda g: pick_top(pick_top(g, "score", STAGE1_N), mag),
        "E_mag_first": lambda g: pick_top(pick_top(g, mag, STAGE1_N), "score"),
        "F_blend": lambda g: (
            lambda gg: pick_top(
                gg.assign(
                    blend=0.5 * gg["score"].rank(pct=True)
                    + 0.5 * gg[mag].rank(pct=True)
                ),
                "blend",
            )
        )(g),
        "G_mag_gate": lambda g: pick_top(
            pick_top(g, "score", STAGE1_N).query(f"{mag} > 0"), "score"
        ),
    }

    print(
        f"\n{'板块':<5}{'策略':<16}{'日':>4}{'N':>5}{'均MFE':>10}{'上涨率':>9}{'达标率':>9}"
        f"{'单日>3%占比':>11}",
        flush=True,
    )
    rows = []
    for b in ("main", "dual"):
        sub = df[df["board"] == b]
        for name, fn in strategies.items():
            dres = []
            for D, g in sub.groupby("date"):
                pick = fn(g)
                y = pick[real].dropna()
                if y.empty:
                    continue
                dres.append(
                    {
                        "date": D,
                        "n": int(len(y)),
                        "mfe": float(y.mean()),
                        "win": float((y > 0).mean()),
                        "hit": float((y >= ABS_TARGET[H]).mean()),
                    }
                )
            d = pd.DataFrame(dres)
            if d.empty:
                continue
            rows.append(
                {
                    "board": b,
                    "strategy": name,
                    "n_days": int(d["date"].nunique()),
                    "n": int(d["n"].sum()),
                    "mfe_mean": float(d["mfe"].mean()),
                    "win_pct": float(d["win"].mean()),
                    "hit_pct": float(d["hit"].mean()),
                    "hit3_pct": float((d["hit"] > 0.3).mean()),
                }
            )
            print(
                f"{b:<5}{name:<16}{int(d['date'].nunique()):>4}{int(d['n'].sum()):>5}"
                f"{d['mfe'].mean():>+10.4f}{d['win'].mean():>9.1%}{d['hit'].mean():>9.1%}"
                f"{(d['hit'] > 0.3).mean():>11.1%}",
                flush=True,
            )

    out = pd.DataFrame(rows)
    out_path = Path(run) / "strategies_3d.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWORM: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
