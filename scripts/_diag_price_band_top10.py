# -*- coding: utf-8 -*-
"""_diag_price_band_top10.py — 价格分层在现行 top-10 流程上的验证 (2026-08-14).

背景: limit-pullback 研究唯一真超额 = 价格分层 (主板低价<10 Sharpe0.90 /
双创中价10-30 Sharpe0.93, 2026-08-10), 从未集成进任何生产路径。
本脚本用**生产同口径**重放并行合并 top-10 短名单 (load 检查点 242d →
_finalize_slice → c2c 净标签 → 可交易性门 → build_merged_shortlist 全池
mag_10d 校准 top-10), 把每次选股按决策日 close_hfq 分层为
低<10 / 中10-30 / 高>30, 对比命中率与实得 (3d/5d/10d c2c 净)。

只读诊断, 不进管线。若价格带分层真实 → 再走完整 250d OOS 子窗口 A/B 回测
才准集成 (参数扫描可靠性原则)。

WORM: DATA OTHERS/diag/price_band_top10_<ts>.json + <ts>.csv (逐票逐带).

用法:
  python scripts/_diag_price_band_top10.py
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
    add_c2c_labels,
    board_of,
    build_merged_shortlist,
    tradability_gate,
)
from scripts._reclassify_all_features import _finalize_slice
from config.settings import data_others_path

READ_COLS = [
    "symbol", "date", "close_hfq", "adv20", "is_suspended",
    "amihud_illiq", "small_mv_premium", "amihud_illiquidity",
    "VAR51", "ret_reversal_5d", "limit_dist_pct",
]
BANDS = [(0.0, 10.0, "低<10"), (10.0, 30.0, "中10-30"), (30.0, float("inf"), "高>30")]
SUB_WINDOWS = {"full": 10**9, "126d": 126, "63d": 63}
LABELS = [("3d", "label_pm_3d_net"), ("5d", "label_pm_5d_net"), ("10d", "label_pm_10d_net")]


def _load_work() -> pd.DataFrame:
    """生产同口径面板: 双检查点 242d 窗 + c2c 净标签 + 可交易性门 + board 列。

    只读合并 top-10 所需列 (12 列) — build_merged_shortlist 只碰 pool 特征与
    label_pm_10d_net, 与全列读取的排名结果逐字节一致 (pool_score 跳过缺列)。
    """
    slices = []
    for ckpt in (pc.PANEL.main_checkpoint, pc.PANEL.dual_checkpoint):
        cutoff = _window_cutoff(ckpt, pc.PANEL.window_days)
        kw = {}
        if cutoff is not None:
            kw["filters"] = [("date", ">=", cutoff)]
        df = pd.read_parquet(ckpt, columns=READ_COLS, **kw)
        df = _finalize_slice(df)
        df = add_c2c_labels(df, horizons=(3, 5, 10), already_sorted=True)
        slices.append(df)
        del df
        gc.collect()
    work = pd.concat(slices, ignore_index=True).sort_values(
        ["symbol", "date"], ignore_index=True
    )
    del slices
    gc.collect()
    work, gate = tradability_gate(work)
    work["board"] = work["symbol"].map(board_of)
    print(
        f"[load] {len(work):,}r | 可交易性门剔除 {gate['removed_rows']:,} 行/"
        f"{gate['removed_stocks']} 只", flush=True,
    )
    return work


def _band_stats(sub: pd.DataFrame) -> dict:
    out: dict = {}
    for h, lab in LABELS:
        r = sub[lab].dropna()
        if len(r) == 0:
            out[h] = {"n": 0, "hit": float("nan"), "mean": float("nan"),
                      "med": float("nan"), "max": float("nan"),
                      "p95": float("nan"), "ge5": float("nan"), "ge10": float("nan")}
            continue
        out[h] = {
            "n": int(len(r)),
            "hit": float((r > 0).mean()),
            "mean": float(r.mean()),
            "med": float(r.median()),
            "max": float(r.max()),
            "p95": float(r.quantile(0.95)),
            "ge5": float((r >= 0.05).mean()),
            "ge10": float((r >= 0.10).mean()),
        }
    return out


def main() -> int:
    t0 = time.time()
    work = _load_work()
    # 生产同口径合并 top-10 (rk = 每日期 mag_10d 排名 1..10)
    sl = build_merged_shortlist(work, 10)
    print(f"[merged] {len(sl):,} 票 | {sl['date'].nunique()} 决策日 ({time.time() - t0:.0f}s)", flush=True)
    if sl.empty:
        print("合并短名单为空", flush=True)
        return 1

    # 逐票: 决策日 close_hfq 分层 + 实得标签
    px = work[["symbol", "date", "close_hfq"]].drop_duplicates(["symbol", "date"])
    labs = work[["symbol", "date"] + [lab for _, lab in LABELS]].drop_duplicates(["symbol", "date"])
    m = sl.merge(px, on=["symbol", "date"], how="inner").merge(labs, on=["symbol", "date"], how="inner")
    m["band"] = pd.cut(
        m["close_hfq"], bins=[0.0, 10.0, 30.0, float("inf")],
        labels=["低<10", "中10-30", "高>30"], include_lowest=True, right=False,
    )
    m["top"] = np.where(m["rk"] <= 5, "T-5", "T-10后5")
    dates = np.sort(m["date"].unique())
    m["_di"] = np.searchsorted(dates, m["date"].values)
    max_di = m["_di"].max()
    del px, labs, work
    gc.collect()

    summary: list[dict] = []
    print(f"\n{'板块':<4}{'窗口':<6}{'分层':<10}{'切档':<10}{'票':>5} | "
          f"{'10d命中':>7}{'10d均':>8}{'10d中':>8}{'10d最大':>8}{'10d≥5%':>8} | "
          f"{'5d命中':>7}{'5d均':>8} | {'3d命中':>7}{'3d均':>8}", flush=True)
    for board in ("main", "dual"):
        mb = m[m["board"] == board]
        for wname, wdays in SUB_WINDOWS.items():
            sub_w = mb[mb["_di"] >= max_di - wdays + 1]
            for _lo, _hi, band in BANDS:
                sb = sub_w[sub_w["band"] == band]
                for top, sb2 in (("T-5", sb[sb["top"] == "T-5"]), ("T-10", sb)):
                    s = _band_stats(sb2)
                    print(
                        f"{board:<4}{wname:<6}{band:<10}{top:<10}{len(sb2):>5} | "
                        f"{s['10d']['hit']:>7.1%}{s['10d']['mean']:>+8.2%}{s['10d']['med']:>+8.2%}"
                        f"{s['10d']['max']:>+8.2%}{s['10d']['ge5']:>8.1%} | "
                        f"{s['5d']['hit']:>7.1%}{s['5d']['mean']:>+8.2%} | "
                        f"{s['3d']['hit']:>7.1%}{s['3d']['mean']:>+8.2%}", flush=True,
                    )
                    summary.append({
                        "board": board, "window": wname, "band": band, "cut": top,
                        **{f"{h}_{k}": v for h, d in s.items() for k, v in d.items()},
                    })

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = data_others_path("diag")
    os.makedirs(str(out_dir), exist_ok=True)
    m.drop(columns=["_di"]).to_csv(out_dir / f"price_band_top10_{ts}.csv", index=False)
    (out_dir / f"price_band_top10_{ts}.json").write_text(
        json.dumps({"ts": ts, "summary": summary, "n_picks": len(m)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n[saved] {out_dir}/price_band_top10_{ts}.csv/.json ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
