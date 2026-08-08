"""mag_10d close-to-close 校准排名模块 (2026-08-07 定案).

并行系统 (除慢牛) 短名单排名 = 每股收缩回归 `score → label_pm_10d_net`
(T+10 close-to-close 校准幅度) 的全板块日截面降序 → TOP5/TOP10.

校准 = 每股收缩回归 (empirical-Bayes partial pooling):
  mag = slope·score + intercept
  slope = λ·slope_per + (1-λ)·slope_cross,  λ = take/(take+shrink_kappa)
  intercept = ȳ − slope·x̄ (拟合线过该股 (score, label) 质心)
横截面 OLS (板块内 score→label) 兜底, 每股样本 < per_stock_min_n 时直接用.

无前瞻 (铁律): 拟合窗 [D-cal_n, D) 只用**已实现**标签. 行 t 的卖价
close_hfq[T+11] (label_pm_10d_net[t] = close_hfq[T+11]/close_hfq[T+1]−1−cost)
须在决策日 D 之前已打印 → 行 t 可用 ⇔ 该股 t+11 的日期 < D. 实现上:
  - 横截面: 板块**唯一日期**数组 bd_unique 上, iD_date = 首个 date ≥ D 的索引;
    已实现日期边界 = bd_unique[iD_date - realized_drop], 再映射回行索引
    `iEnd = searchsorted(bd, bd_unique[iD_date - realized_drop], right)` — 即丢
    D 前最近 realized_drop 个交易日 (日界, 非行界; 多股/日时行界会漏未来标签);
    不足 cross_min_n → 该板块当日不出票.
  - 每股: 有效样本计数 `vD = searchsorted(pos, searchsorted(gd, D, "right") - realized_drop, "left")`
    (gd = 该股日期数组, pos = 有效行位置数组), 恰好丢最后 realized_drop 行.
    realized_drop = buy_lag + label_horizon = 11.

纯向量化横截面 (前缀和) + 每股 numpy 窗口 OLS (与诊断脚本同法), 禁 pandas 行循环.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.pipeline_parallel.config import MAG10D_CAL

# 已实现边界 = 买日滞后 + 视界 (标签需未来 buy_lag+label_horizon 日的卖价)
_REALIZED_DROP = int(MAG10D_CAL["buy_lag"]) + int(MAG10D_CAL["label_horizon"])


def _ols_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """单变量最小二乘 (numpy 快速路径)."""
    xm = float(x.mean())
    ym = float(y.mean())
    var = float(((x - xm) ** 2).sum())
    if var <= 1e-12:
        return 0.0, ym
    slope = float(((x - xm) * (y - ym)).sum() / var)
    return slope, ym - slope * xm


def calibrate_mag10d(
    panel: pd.DataFrame,
    cal_n: int = MAG10D_CAL["cal_n"],
    per_stock_window: int = MAG10D_CAL["per_stock_window"],
    per_stock_min_n: int = MAG10D_CAL["per_stock_min_n"],
    shrink_kappa: float = MAG10D_CAL["shrink_kappa"],
    score_col: str = MAG10D_CAL["score_col"],
    target_col: str = MAG10D_CAL["target_col"],
) -> pd.DataFrame:
    """面板长表 (symbol/date/board/score_col/target_col) → DataFrame[symbol, date, board, mag].

    Walk-forward, per-board. 只用**已实现**标签 (行 t 可用 ⇔ 其 label 的卖价已打印).
    横截面样本 < cross_min_n → 该板块当日不出票. 输出 date = 决策日 D.

    效率: 前缀和横截面 OLS + 每股 numpy 窗口 OLS (与 diag_10d_param_sweep 同法).
    """
    if "board" not in panel.columns:
        raise ValueError("calibrate_mag10d 要求 panel 携带 'board' 列 (两调用方均保证)")
    need = ["symbol", "date", "board", score_col, target_col]
    work = panel[need].copy()
    # 关键: 调用方可能传入被 mask/过滤的子帧 (索引非连续) → 复位为 RangeIndex,
    # 否则下方 s64[g.index] 会把索引标签误当位置索引 → IndexError (生产 build_merged_shortlist 同样受影响)
    work = work.reset_index(drop=True)
    work["symbol"] = work["symbol"].astype(str)
    work["board"] = work["board"].astype(str)
    s64 = work["date"].to_numpy().astype("datetime64[ns]")

    # 每股预提取: (gd, gs, gy, pos) — 与诊断脚本 sym_data 同构
    sym_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for sym, g in work.groupby("symbol", sort=False):
        gd = s64[g.index.to_numpy()]
        gs = g[score_col].to_numpy(float)
        gy = g[target_col].to_numpy(float)
        valid = np.isfinite(gs) & np.isfinite(gy)
        pos = np.nonzero(valid)[0].astype(np.int64)
        sym_data[sym] = (gd, gs, gy, pos)

    boards = np.unique(work["board"].to_numpy())
    # 每板块 score→label 前缀和 (横截面 OLS 用)
    cross: dict[str, tuple[np.ndarray, ...]] = {}
    for board in boards:
        bm = work["board"].to_numpy() == board
        bx = work.loc[bm, score_col].to_numpy(float)
        by = work.loc[bm, target_col].to_numpy(float)
        bvalid = np.isfinite(bx) & np.isfinite(by)
        bd = s64[np.nonzero(bm)[0][bvalid]]
        bx = bx[bvalid]
        by = by[bvalid]
        bo = np.argsort(bd, kind="stable")
        bd, bx, by = bd[bo], bx[bo], by[bo]
        psx = np.concatenate([[0.0], np.cumsum(bx)])
        psy = np.concatenate([[0.0], np.cumsum(by)])
        psxx = np.concatenate([[0.0], np.cumsum(bx * bx)])
        psxy = np.concatenate([[0.0], np.cumsum(bx * by)])
        cross[board] = (bd, psx, psy, psxx, psxy)

    # 每板块日截面 (symbols, scores) 预提取
    board_dates: dict[str, list] = {}
    day_cross: dict[tuple[str, object], tuple[np.ndarray, np.ndarray]] = {}
    for board in boards:
        bm = work["board"].to_numpy() == board
        sub = work.loc[bm, ["date", "symbol", score_col]]
        dates = sorted(sub["date"].unique())
        board_dates[board] = dates
        for D, g in sub.groupby("date", sort=True):
            day_cross[(board, D)] = (
                g["symbol"].to_numpy(str),
                g[score_col].to_numpy(float),
            )

    rows: list[tuple] = []
    for board in boards:
        bdates = board_dates[board]
        date_idx = {d: i for i, d in enumerate(bdates)}
        bd, psx, psy, psxx, psxy = cross[board]
        for D in bdates:
            dc = day_cross.get((board, D))
            if dc is None:
                continue
            syms, scores = dc
            if len(syms) == 0:
                continue
            di = date_idx[D]
            if di < _REALIZED_DROP:
                continue  # 尚无已实现标签 → 该板块当日不出票
            cal_lo = bdates[max(0, di - cal_n)]
            # 横截面只用到已实现标签: 行 t 可用 ⇔ t+11 ≤ di (卖价 close[bdates[t+11]] 在决策时已打印).
            # iEnd = 首行 date > bdates[di-11] → 纳入日期 ≤ bdates[di-11] 的全部行.
            iLo = int(np.searchsorted(bd, np.datetime64(cal_lo), side="left"))
            iEnd = int(
                np.searchsorted(bd, np.datetime64(bdates[di - _REALIZED_DROP]), side="right")
            )
            n = iEnd - iLo
            if n < MAG10D_CAL["cross_min_n"]:
                continue  # 该板块当日横截面不足 → 不出票
            Sx = psx[iEnd] - psx[iLo]
            Sy = psy[iEnd] - psy[iLo]
            Sxx = psxx[iEnd] - psxx[iLo]
            Sxy = psxy[iEnd] - psxy[iLo]
            var = n * Sxx - Sx * Sx
            if var <= 1e-12:
                cs, ci = 0.0, (Sy / n if n else 0.0)
            else:
                cs = (n * Sxy - Sx * Sy) / var
                ci = (Sy - cs * Sx) / n
            cal_lo64 = np.datetime64(cal_lo)
            D64 = np.datetime64(D)
            # 每股窗口 OLS + 收缩 (per-symbol numpy 循环, 与诊断脚本一致)
            for i in range(len(syms)):
                sc = scores[i]
                if not np.isfinite(sc):
                    continue
                sd = sym_data.get(str(syms[i]))
                if sd is None:
                    continue
                gd, gs, gy, pos = sd
                vD = int(
                    np.searchsorted(
                        pos,
                        int(np.searchsorted(gd, D64, side="right")) - _REALIZED_DROP,
                        side="left",
                    )
                )
                sym_iLo = int(np.searchsorted(gd, cal_lo64, side="left"))
                vLo = int(np.searchsorted(pos, sym_iLo, side="left"))
                vc = vD - vLo
                if vc >= per_stock_min_n:
                    take = min(vc, per_stock_window)
                    r = pos[vD - take : vD]
                    x = gs[r]
                    y = gy[r]
                    raw_slope, _ = _ols_slope_intercept(x, y)
                    lam = take / (take + shrink_kappa)
                    slope = lam * raw_slope + (1 - lam) * cs
                    intercept = float(y.mean()) - slope * float(x.mean())
                else:
                    slope, intercept = cs, ci
                mag = slope * sc + intercept
                if np.isfinite(mag):
                    rows.append((syms[i], D, board, mag))

    if not rows:
        return pd.DataFrame(columns=["symbol", "date", "board", "mag"])
    return pd.DataFrame(rows, columns=["symbol", "date", "board", "mag"])
