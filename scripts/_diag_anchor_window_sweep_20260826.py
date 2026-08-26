"""_diag_anchor_window_sweep_20260826.py — 报告幅度锚定窗长重扫 (用户 "锚定扫" 2026-08-26).

问题: ANCHOR_WINDOW=250 的大半是前期好行情, 弱市锚值虚高 (08-25 诊断: 清单
pred_ret_10d +2.11% vs 实得 -0.82%, 根因即锚顺周期).

方法 (无 look-ahead 重放): 逐板块逐视界, s(D) = 决策日 D 模型 top-10 的
label_pm_{h}_net 均值 (与 _trailing_realized 同公式, 每日恰好 top 行 → 池化均值
= 日均值均值). 决策日 D 可用的锚 (生产口径, 只见 ≤D-成熟滞后的已实现日):
    A_w(D) = s.shift(lag).rolling(w).mean()   [lag = h+1 交易日]
误差 err(D) = A_w(D) − s(D) (锚报数 vs 当日入选后来实得).

扫 w ∈ {60, 90, 120, 180, 250} (含 250=现状), 扰动 = w±10 与 top 9/11;
评估 = 尾 250 个已实现日 (主) / 全可用 (参), 3 段子窗稳定性 (扫参可靠性原则:
选稳定 > 单点最优). 输出 WORM CSV → DATA OTHERS/diag/anchor_window_sweep_<ts>.csv.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import data_others_path
from scripts import _shortlist_t5_t10 as sl

WINDOWS = [60, 90, 120, 180, 250]  # 250 = 现状
EVAL_DAYS = 250                    # 主评估 = 尾 250 已实现日 (~1 年)
N_CHUNKS = 3
TOPS = [9, 10, 11]                 # 10 = 现状, 9/11 = 扰动
FRAME_WINDOW = 700                 # 加宽面板 (~712 交易日, 3y 全量)


def daily_top_mean(fr: pd.DataFrame, h: str, top: int) -> pd.Series:
    """逐决策日 top-N 实得均值 (与 _trailing_realized 单日贡献同公式)."""
    lab = f"label_pm_{h}_net"
    df = fr.dropna(subset=[lab])
    top_rows = (
        df.sort_values(["date", "score"], ascending=[True, False])
        .groupby("date", sort=False)
        .head(top)
    )
    return top_rows.groupby("date")[lab].mean().sort_index()


def err_series(s: pd.Series, w: int, lag: int) -> pd.Series:
    """A_w(D) − s(D); 锚用 s.shift(lag) 只见已成熟历史 (生产同口径)."""
    return s.shift(lag).rolling(w, min_periods=w).mean() - s


def chunk_bias(e: pd.Series, n: int) -> list[float]:
    parts = np.array_split(e.index.to_numpy(), n)
    out = []
    for p in parts:
        seg = e[e.index.isin(p)]
        out.append(float(seg.mean()) if len(seg) else float("nan"))
    return out


def main() -> int:
    rows = []
    for board in ("main", "dual"):
        fr = sl._anchor_frame(board, window=FRAME_WINDOW)
        if fr.empty:
            print(f"[{board}] 加宽面板不足, 跳过", flush=True)
            continue
        for h in sl.HORIZONS:
            lag = int(h[:-1]) + 1
            s_by_top = {t: daily_top_mean(fr, h, t) for t in TOPS}
            s = s_by_top[10]
            ctx = {
                "近60日实得": s.tail(60).mean(),
                "近250日实得": s.tail(250).mean(),
                "全史实得": s.mean(),
            }
            print(
                f"\n===== [{board}] 视界 {h} (滞后 {lag} 日; 每日 top-10 实得均值) =====",
                flush=True,
            )
            print(
                "  实得水平: " + "  ".join(f"{k} {v:+.2%}" for k, v in ctx.items()),
                flush=True,
            )
            print(
                f"  {'w':>4} | {'bias(尾250)':>10} {'MAE':>7} | {'块1':>7} {'块2':>7} {'块3':>7} | "
                f"{'w-10':>7} {'w+10':>7} | {'top9':>7} {'top11':>7} | N",
                flush=True,
            )
            for w in WINDOWS:
                e = err_series(s, w, lag).dropna()
                e_eval = e.tail(EVAL_DAYS)
                if e_eval.empty:
                    continue
                chunks = chunk_bias(e_eval, N_CHUNKS)
                p10 = lambda v: (f"{v:+.2%}" if np.isfinite(v) else "—")  # noqa: E731
                print(
                    f"  {w:>4} | {p10(e_eval.mean())} {p10(e_eval.abs().mean()):>7} | "
                    f"{' '.join(p10(c) for c in chunks)} | "
                    f"{p10(err_series(s, max(w - 10, 20), lag).tail(EVAL_DAYS).mean())} "
                    f"{p10(err_series(s, w + 10, lag).tail(EVAL_DAYS).mean())} | "
                    f"{p10(err_series(s_by_top[9], w, lag).tail(EVAL_DAYS).mean())} "
                    f"{p10(err_series(s_by_top[11], w, lag).tail(EVAL_DAYS).mean())} | "
                    f"{len(e_eval)}",
                    flush=True,
                )
                rows.append(
                    {
                        "board": board,
                        "horizon": h,
                        "window": w,
                        "lag": lag,
                        "n_eval": len(e_eval),
                        "bias_tail250": e_eval.mean(),
                        "mae_tail250": e_eval.abs().mean(),
                        **{f"chunk{i + 1}": c for i, c in enumerate(chunks)},
                        "bias_full": e.mean() if len(e) else np.nan,
                        **ctx,
                    }
                )
    df = pd.DataFrame(rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = data_others_path("diag") / f"anchor_window_sweep_{ts}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[saved] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
