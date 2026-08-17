"""margin 重扫同日期分解: 把"挑股效应"(同一天内高 margin 是否选得更好)与"丢日效应"分开.

读 WORM CSV (legacy_prob_gate_margin_sweep_<ts>.csv), 以 anchor margin 存活日集为
公共日, 比较各 margin 在这些天的 top-N 表现. 输出 WORM JSON.

用法: python scripts/_diag_legacy_prob_gate_margin_sameday.py <sweep_csv>
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import data_others_path

MARGINS = [0.00, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16]
MKEY = {m: f"{m:.2f}" for m in MARGINS}


def topn(sub: pd.DataFrame, n: int, col: str = "pred_ret_10d") -> pd.DataFrame:
    return (
        sub.sort_values(["date", col], ascending=[True, False])
        .groupby("date", sort=False)
        .head(n)
    )


def win_break(r: pd.Series, dates: pd.Series) -> list[dict]:
    dl = sorted(dates.dropna().unique())
    if not dl:
        return []
    step = max(1, len(dl) // 3)
    out = []
    for i in range(3):
        s0, s1 = i * step, len(dl) if i == 2 else (i + 1) * step
        seg = r[dates.isin(dl[s0:s1])]
        out.append(
            {
                "win": f"{i + 1}/3",
                "hit10": float((seg > 0).mean()) if len(seg) else float("nan"),
                "mean10": float(seg.mean()) if len(seg) else float("nan"),
            }
        )
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python _diag_legacy_prob_gate_margin_sameday.py <sweep_csv>")
        return 2
    csv_path = Path(sys.argv[1])
    df = pd.read_csv(csv_path, dtype={"symbol": str})
    df["date"] = pd.to_datetime(df["date"])
    ts = csv_path.stem.rsplit("_", 1)[-1]

    result: dict = {"source": csv_path.name, "ts": ts, "boards": {}}
    for board in ("main", "dual"):
        sub = df[(df["board"] == board) & (~df["pain_excluded"])].copy()
        kept = {
            m: sub[
                (sub["pred_prob"] > sub["base_prod"] + m)
                | sub["pred_prob"].isna()
                | sub["base_prod"].isna()
            ]
            for m in MARGINS
        }
        surv = pd.DataFrame(
            {MKEY[m]: kept[m].groupby("date").size().gt(0) for m in MARGINS}
        )
        board_out: dict = {"day_survival": {k: int(v) for k, v in surv.sum().items()}}
        for anchor in (0.08, 0.10):
            days = set(surv.index[surv[MKEY[anchor]].fillna(False)])
            rows = []
            for m in MARGINS:
                t = topn(kept[m][kept[m]["date"].isin(days)], 5)
                r = t["realized_net"].dropna()
                if not len(r):
                    continue
                rows.append(
                    {
                        "margin": m,
                        "days": int(t["date"].nunique()),
                        "picks": int(len(t)),
                        "hit": float((r > 0).mean()),
                        "mean": float(r.mean()),
                        "sub_windows": win_break(r, t["date"]),
                    }
                )
            board_out[f"anchor_{anchor:.2f}"] = rows
        result["boards"][board] = board_out

    out_dir = data_others_path("diag")
    out_path = out_dir / f"legacy_prob_gate_margin_sameday_{ts}.json"
    json.dump(
        result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2
    )
    for board, bo in result["boards"].items():
        print(f"\n===== {board} 日存活: {bo['day_survival']} =====")
        for anchor in ("anchor_0.08", "anchor_0.10"):
            print(f"[{anchor}]")
            for r in bo[anchor]:
                sw = "  ".join(
                    f"{w['win']}:{w['hit10']:.0%}/{w['mean10']:+.2%}"
                    for w in r["sub_windows"]
                )
                print(
                    f"  m={r['margin']:.2f} days={r['days']:>4} picks={r['picks']:>4} "
                    f"hit={r['hit']:.1%} mean={r['mean']:+.2%}  {sw}"
                )
    print(f"\n[WORM] {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
