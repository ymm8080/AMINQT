"""_bt_price_band_ab.py — 价格分层过滤 A/B 回测 (2026-08-14).

_diag_price_band_top10 发现: 现行合并 top-10 中
  - dual 低<10: 全窗/126d/63d 三窗全负 (命中 17-30%, 实得 -3.2%~-7.5%)
  - main 高>30: 近 126d/63d 显著差 (63d 命中 28.3%, 实得 -2.11%), 全窗不差
  - main 低<10: 弱市反而最好 (63d +5.11%)
本脚本做生产同口径 A/B (与 _diag 同 loader + build_merged_shortlist):
  变体 A 基准 = 现行 top-10
  变体 B     = 剔除 dual 低<10
  变体 C     = 剔除 main 高>30
  变体 D     = B+C
量测 = _merged_dual 生产验收口径 (双头: 胜率/幅度 vs 窗口无条件基线, ok=胜率≥0.50
且幅度≥板块阈值 main 1.0% / dual 1.5%), 窗口 = full(仅参考) / 126d / 63d OOS。
结论原则: 只看 OOS; 变体须在 OOS 子窗口赢基线 (命中+实得), 且不显著牺牲出票量。

WORM: DATA OTHERS/diag/price_band_ab_<ts>.json
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from app.pipeline_parallel import config as pc
from app.pipeline_parallel.backtest import _merged_dual, build_merged_shortlist
from config.settings import data_others_path
from scripts._diag_price_band_top10 import _load_work

WINDOWS = {"full": 10**9, "126d": 126, "63d": 63}
LABELS = [
    ("3d", "label_pm_3d_net"),
    ("5d", "label_pm_5d_net"),
    ("10d", "label_pm_10d_net"),
]


def _attach_band(sl: pd.DataFrame, work: pd.DataFrame) -> pd.DataFrame:
    px = work[["symbol", "date", "close_hfq"]].drop_duplicates(["symbol", "date"])
    m = sl.merge(px, on=["symbol", "date"], how="inner")
    m["band"] = pd.cut(
        m["close_hfq"],
        bins=[0.0, 10.0, 30.0, float("inf")],
        labels=["低<10", "中10-30", "高>30"],
        include_lowest=True,
        right=False,
    )
    return m


def _variant(m: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "A_基准":
        return m
    if name == "B_剔dual低价":
        return m[~((m["board"] == "dual") & (m["band"] == "低<10"))]
    if name == "C_剔main高价":
        return m[~((m["board"] == "main") & (m["band"] == "高>30"))]
    if name == "D_B加C":
        return m[
            ~((m["board"] == "dual") & (m["band"] == "低<10"))
            & ~((m["board"] == "main") & (m["band"] == "高>30"))
        ]
    raise ValueError(name)


def _hit_stats(sub: pd.DataFrame, sl: pd.DataFrame) -> dict:
    out: dict = {}
    for h, lab in LABELS:
        v = (
            sub[["symbol", "date", lab]]
            .merge(sl[["symbol", "date"]], on=["symbol", "date"], how="inner")[lab]
            .dropna()
        )
        out[h] = {
            "n": int(len(v)),
            "hit": float((v > 0).mean()) if len(v) else float("nan"),
            "mean": float(v.mean()) if len(v) else float("nan"),
        }
    return out


def main() -> int:
    t0 = time.time()
    work = _load_work()
    sl = build_merged_shortlist(work, 10)
    m = _attach_band(sl, work)
    dates = np.sort(work["date"].unique())
    print(
        f"[data] merged {len(m):,} 票 / {m['date'].nunique()} 决策日 ({time.time() - t0:.0f}s)",
        flush=True,
    )

    crit = {
        b: (pc.BOARD_THRESHOLDS[b]["min_winrate"], pc.BOARD_THRESHOLDS[b]["min_mag"])
        for b in ("main", "dual")
    }
    rows: list[dict] = []
    for vname in ("A_基准", "B_剔dual低价", "C_剔main高价", "D_B加C"):
        vm = _variant(m, vname)
        for board in ("main", "dual"):
            vmb = vm[vm["board"] == board]
            for wname, wdays in WINDOWS.items():
                cutoff = dates[0] if wdays >= len(dates) else dates[-wdays]
                sub_w = work[(work["board"] == board) & (work["date"].values >= cutoff)]
                vmb[vmb["date"].isin(set(sub_w["date"]))][["date", "symbol"]]
                for cut, n in (("T-5", 5), ("T-10", 10)):
                    sl_c = vmb[(vmb["rk"] <= n) & vmb["date"].isin(set(sub_w["date"]))][
                        ["date", "symbol"]
                    ]
                    ph = _merged_dual(sub_w, sl_c, crit[board])
                    hs = _hit_stats(sub_w, sl_c)
                    for h, r in ph.items():
                        rows.append(
                            {
                                "variant": vname,
                                "board": board,
                                "window": wname,
                                "cut": cut,
                                "horizon": h,
                                "picks": int(len(sl_c)),
                                "winrate": r.get("winrate"),
                                "mag": r.get("mag"),
                                "ok": bool(r.get("ok")),
                                "delta_wr": r.get("delta_wr"),
                                "hit": hs[h]["hit"],
                                "mean": hs[h]["mean"],
                                "n_hit": hs[h]["n"],
                            }
                        )
        print(f"[{vname}] done ({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    (out_dir / f"price_band_ab_{ts}.json").write_text(
        json.dumps({"ts": ts, "rows": rows}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(
        f"\n{'变体':<12}{'板':<4}{'窗':<6}{'切':<5}{'h':>3} | "
        f"{'票':>5}{'胜率':>7}{'幅度':>8}{'ok':>4}{'Δ胜率':>7} | {'命中':>7}{'实得':>8}",
        flush=True,
    )
    for _, r in df.iterrows():
        print(
            f"{r['variant']:<12}{r['board']:<4}{r['window']:<6}{r['cut']:<5}{r['horizon']:>3} | "
            f"{int(r['picks']):>5}{r['winrate']:>7.1%}{r['mag']:>+8.2%}{str(r['ok']):>4}"
            f"{r['delta_wr'] if r['delta_wr'] is None else r['delta_wr']:>7.2%} | "
            f"{r['hit']:>7.1%}{r['mean']:>+8.2%}",
            flush=True,
        )

    # 结论: 只看 OOS (126d/63d), 每变体 vs 基准
    print("\n===== OOS 结论 (每窗每切档 10d 命中/实得 vs 基准) =====", flush=True)
    base = df[df["variant"] == "A_基准"]
    for vname in ("B_剔dual低价", "C_剔main高价", "D_B加C"):
        vd = df[df["variant"] == vname]
        for board in ("main", "dual"):
            for wname in ("126d", "63d"):
                for cut, h in (("T-5", "10d"), ("T-10", "10d")):
                    b = base[
                        (base.board == board)
                        & (base.window == wname)
                        & (base.cut == cut)
                        & (base.horizon == h)
                    ].iloc[0]
                    v = vd[
                        (vd.board == board)
                        & (vd.window == wname)
                        & (vd.cut == cut)
                        & (vd.horizon == h)
                    ].iloc[0]
                    print(
                        f"{vname} {board} {wname} {cut}: "
                        f"命中 {b['hit']:.1%}→{v['hit']:.1%} ({v['hit'] - b['hit']:+.1%}) | "
                        f"实得 {b['mean']:+.2%}→{v['mean']:+.2%} ({v['mean'] - b['mean']:+.2%}) | "
                        f"票 {int(b['picks'])}→{int(v['picks'])}",
                        flush=True,
                    )
    print(
        f"\n[saved] {out_dir}/price_band_ab_{ts}.json ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
