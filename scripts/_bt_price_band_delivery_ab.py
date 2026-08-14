"""_bt_price_band_delivery_ab.py — 价格分层过滤的交付级 A/B 回放 (2026-08-14).

_bt_price_band_ab (池级) 已证: dual 低<10 剔除 6/6 窗全赢, main 高>30 剔除均值全赢
但 −40% 出票。本脚本补**交付级**验证 — 完整复刻 _shortlist_t5_t10 生产链:
  expand_candidates(全板块) → pred_mag_10d (共享 calibrate_mag10d) →
  select_confident (pred_ret_3d > t3_min, main=0 / dual=0.5%, 经 _c2c_latest 同源
  calibrate_mag10d target=label_pm_3d_net) → 价格带过滤变体 → rank T-5 →
  逐窗量 realized 10d/5d/3d c2c 净 (命中/实得)。
t3_min 门与价格带的交互: 若 t3 门已拦掉大部分低价毒票, B 的边际价值变小。

08-14 保真修正: 生产 _panel_per_stock 的 score = 板块内截面分位 (pool_score 于
单板块帧), 且不跑 tradability_gate; label 直接读检查点列。首版回放误用
_diag_price_band_top10 的 _load_work (双板合并 rank + tradability_gate) → 与生产
交付链不对齐, 现改 _load_work_delivery 直读检查点 (同 _panel_per_stock 口径)。

变体: A=基准(仅t3门) B=A+剔dual低<10 C=A+剔main高>30 D=B+C
窗口: full(参考) / 126d / 63d OOS。WORM: DATA OTHERS/diag/price_band_delivery_ab_<ts>.json
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from app.pipeline_parallel import config as pc
from app.pipeline_parallel.backtest import (
    _window_cutoff,
    board_of,
    calibrate_mag10d,
    pool_score,
)
from config.settings import SHORTLIST_SCORE, data_others_path

T3_MIN = SHORTLIST_SCORE["select_gate"]["t3_min"]  # {"main": 0.0, "dual": 0.005}
WINDOWS = {"full": 10**9, "126d": 126, "63d": 63}
LABELS = [
    ("3d", "label_pm_3d_net"),
    ("5d", "label_pm_5d_net"),
    ("10d", "label_pm_10d_net"),
]
# 交付链所需列 (同 _panel_per_stock: 池特征缺 pv_corr_5 自动跳过; label 直读检查点)
_POOL_COLS = sorted(
    {c for c in set(pc.SNIPER.pool) | set(pc.FUSION.pool) if c != "pv_corr_5"}
)


def _load_work_delivery() -> pd.DataFrame:
    """生产交付口径面板: 双检查点 242d 窗, 无 tradability_gate, label 直读.

    与 _shortlist_t5_t10._panel_per_stock 同口径 (板块内截面分位分), 只多读全窗
    历史供逐日重放 (calibrate_mag10d 每决策日只回看 21d, 口径一致).
    """
    slices = []
    for ckpt in (pc.PANEL.main_checkpoint, pc.PANEL.dual_checkpoint):
        cutoff = _window_cutoff(ckpt, pc.PANEL.window_days)
        kw = {}
        if cutoff is not None:
            kw["filters"] = [("date", ">=", cutoff)]
        df = pd.read_parquet(ckpt, columns=_READ_COLS, **kw)
        slices.append(df)
        del df
        gc.collect()
    work = pd.concat(slices, ignore_index=True).sort_values(
        ["symbol", "date"], ignore_index=True
    )
    del slices
    gc.collect()
    work["symbol"] = work["symbol"].astype(str)
    work["board"] = work["symbol"].map(board_of)
    print(
        f"[load] {len(work):,}r | {work['date'].nunique()} 决策日 "
        f"(生产交付口径: 无 tradability_gate, label 直读)",
        flush=True,
    )
    return work


_READ_COLS = (
    ["symbol", "date", "close_hfq"]
    + _POOL_COLS
    + [
        "label_pm_3d_net",
        "label_pm_5d_net",
        "label_pm_10d_net",
    ]
)


def _cal_frame(work: pd.DataFrame, board: str) -> pd.DataFrame:
    """全池 score=max(sniper,fusion) 板块内截面分 + 两目标校准 (mag10 排名 / c2c3 门)。"""
    sub = work[work["board"] == board]
    s1 = pool_score(sub, pc.SNIPER.pool)
    s2 = pool_score(sub, pc.FUSION.pool)
    scored = sub[["symbol", "date"]].copy()
    scored["board"] = board
    scored["score"] = np.maximum(s1.values, s2.values)
    scored["label_pm_10d_net"] = sub["label_pm_10d_net"].values
    scored["label_pm_5d_net"] = sub["label_pm_5d_net"].values
    scored["label_pm_3d_net"] = sub["label_pm_3d_net"].values
    scored["close_hfq"] = sub["close_hfq"].values
    scored = scored.dropna(subset=["score"])
    mag10 = calibrate_mag10d(scored, score_col="score", target_col="label_pm_10d_net")
    c2c3 = calibrate_mag10d(
        scored, score_col="score", target_col="label_pm_3d_net", label_horizon=3
    )
    m = scored.drop(columns=["board"]).merge(mag10, on=["symbol", "date"], how="inner")
    m = m.merge(
        c2c3[["symbol", "date", "mag"]].rename(columns={"mag": "mag3"}),
        on=["symbol", "date"],
        how="inner",
    )
    m["board"] = board
    return m


def main() -> int:
    t0 = time.time()
    work = _load_work_delivery()
    frames = {}
    for board in ("main", "dual"):
        f = _cal_frame(work, board)
        f["board"] = board
        frames[board] = f
        print(f"[cal:{board}] {len(f):,} 票×日 ({time.time() - t0:.0f}s)", flush=True)
    del work
    gc.collect()

    m = pd.concat(frames.values(), ignore_index=True)
    m["band"] = pd.cut(
        m["close_hfq"],
        bins=[0.0, 10.0, 30.0, float("inf")],
        labels=["低<10", "中10-30", "高>30"],
        include_lowest=True,
        right=False,
    )
    dates = np.sort(m["date"].unique())

    rows: list[dict] = []
    for vname, vmask in (
        ("A_基准", None),
        ("B_剔dual低价", (m["board"] == "dual") & (m["band"] == "低<10")),
        ("C_剔main高价", (m["board"] == "main") & (m["band"] == "高>30")),
        (
            "D_B加C",
            ((m["board"] == "dual") & (m["band"] == "低<10"))
            | ((m["board"] == "main") & (m["band"] == "高>30")),
        ),
    ):
        v = m if vmask is None else m[~vmask]
        # t3 门 (生产 select_confident): 每板阈值, 弱市硬拦属正常
        gated = v[v["mag3"] > v["board"].map(T3_MIN)]
        # rank T-5 (生产 rank_and_truncate): 每日期×板按 mag10 降序
        gated = gated.sort_values(
            ["board", "date", "mag"], ascending=[True, True, False]
        )
        gated["rk"] = gated.groupby(["board", "date"], sort=False).cumcount() + 1
        t5 = gated[gated["rk"] <= 5]
        print(
            f"[{vname}] t3门后 {len(gated):,} → T-5 {len(t5):,} ({time.time() - t0:.0f}s)",
            flush=True,
        )
        for board in ("main", "dual"):
            tb = t5[t5["board"] == board]
            for wname, wdays in WINDOWS.items():
                cutoff = dates[0] if wdays >= len(dates) else dates[-wdays]
                sub = tb[tb["date"].values >= cutoff]
                for h, lab in LABELS:
                    r = sub[lab].dropna()
                    rows.append(
                        {
                            "variant": vname,
                            "board": board,
                            "window": wname,
                            "horizon": h,
                            "picks": int(len(sub)),
                            "n_days": int(sub["date"].nunique()),
                            "hit": float((r > 0).mean()) if len(r) else float("nan"),
                            "mean": float(r.mean()) if len(r) else float("nan"),
                            "med": float(r.median()) if len(r) else float("nan"),
                            "max": float(r.max()) if len(r) else float("nan"),
                        }
                    )

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    (out_dir / f"price_band_delivery_ab_{ts}.json").write_text(
        json.dumps(
            {"ts": ts, "rows": rows, "t3_min": T3_MIN},
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    df = pd.DataFrame(rows)
    print(
        f"\n{'变体':<12}{'板':<4}{'窗':<6}{'h':>3} | {'票':>5}{'日':>4}{'命中':>7}{'实得':>8}{'中位':>8}{'最大':>8}",
        flush=True,
    )
    for _, r in df.iterrows():
        print(
            f"{r['variant']:<12}{r['board']:<4}{r['window']:<6}{r['horizon']:>3} | "
            f"{int(r['picks']):>5}{int(r['n_days']):>4}{r['hit']:>7.1%}{r['mean']:>+8.2%}"
            f"{r['med']:>+8.2%}{r['max']:>+8.2%}",
            flush=True,
        )
    print(
        f"\n[saved] {out_dir}/price_band_delivery_ab_{ts}.json ({time.time() - t0:.0f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
