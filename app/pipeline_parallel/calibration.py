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
    realized_drop = buy_lag + label_horizon (10d=11, 5d=6, 3d=4; 由 label_horizon 参数驱动).

纯向量化横截面 (前缀和) + 每股 numpy 窗口 OLS (与诊断脚本同法), 禁 pandas 行循环.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.pipeline_parallel.config import MAG10D_CAL

# 已实现边界 = 买日滞后 + 视界 (标签需未来 buy_lag+label_horizon 日的卖价);
# 按视界局部计算 (10d=11, 5d=6, 3d=4), 见 calibrate_mag10d 内 realized_drop.


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
    label_horizon: int = MAG10D_CAL["label_horizon"],
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
    # 已实现边界按视界算: buy_lag + label_horizon (10d=11, 5d=6, 3d=4 交易日)
    realized_drop = int(MAG10D_CAL["buy_lag"]) + int(label_horizon)
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
        for dt, g in sub.groupby("date", sort=True):
            day_cross[(board, dt)] = (
                g["symbol"].to_numpy(str),
                g[score_col].to_numpy(float),
            )

    rows: list[tuple] = []
    for board in boards:
        bdates = board_dates[board]
        date_idx = {d: i for i, d in enumerate(bdates)}
        bd, psx, psy, psxx, psxy = cross[board]
        for dt in bdates:
            dc = day_cross.get((board, dt))
            if dc is None:
                continue
            syms, scores = dc
            if len(syms) == 0:
                continue
            di = date_idx[dt]
            if di < realized_drop:
                continue  # 尚无已实现标签 → 该板块当日不出票
            cal_lo = bdates[max(0, di - cal_n)]
            # 横截面只用到已实现标签: 行 t 可用 ⇔ t+11 ≤ di (卖价 close[bdates[t+11]] 在决策时已打印).
            # i_end = 首行 date > bdates[di-11] → 纳入日期 ≤ bdates[di-11] 的全部行.
            i_lo = int(np.searchsorted(bd, np.datetime64(cal_lo), side="left"))
            i_end = int(
                np.searchsorted(
                    bd, np.datetime64(bdates[di - realized_drop]), side="right"
                )
            )
            n = i_end - i_lo
            if n < MAG10D_CAL["cross_min_n"]:
                continue  # 该板块当日横截面不足 → 不出票
            sx = psx[i_end] - psx[i_lo]
            sy = psy[i_end] - psy[i_lo]
            sxx = psxx[i_end] - psxx[i_lo]
            sxy = psxy[i_end] - psxy[i_lo]
            var = n * sxx - sx * sx
            if var <= 1e-12:
                cs, ci = 0.0, (sy / n if n else 0.0)
            else:
                cs = (n * sxy - sx * sy) / var
                # 08-14 调查: 21 日窗 cs<0 是噪声假信号 (main 43% / dual 33% 日), 旧逻辑
                # mag=cs·score+ci 反转排名去选最低分股, 250d 反序日实得大幅跑输 score-top5
                # (main +0.59% vs +2.59%, dual +0.83% vs +4.13%). 铁律方向: 高分=高预期
                # → 负 cs 只取幅度, mag 恒与 score 单调同序 (同序则 top-5 == 池分 top-5).
                cs = abs(cs)
                ci = (sy - cs * sx) / n
            cal_lo64 = np.datetime64(cal_lo)
            dt64 = np.datetime64(dt)
            # 每股窗口 OLS + 收缩 (per-symbol numpy 循环, 与诊断脚本一致)
            for i in range(len(syms)):
                sc = scores[i]
                if not np.isfinite(sc):
                    continue
                sd = sym_data.get(str(syms[i]))
                if sd is None:
                    continue
                gd, gs, gy, pos = sd
                v_d = int(
                    np.searchsorted(
                        pos,
                        int(np.searchsorted(gd, dt64, side="right")) - realized_drop,
                        side="left",
                    )
                )
                sym_i_lo = int(np.searchsorted(gd, cal_lo64, side="left"))
                v_lo = int(np.searchsorted(pos, sym_i_lo, side="left"))
                vc = v_d - v_lo
                if vc >= per_stock_min_n:
                    take = min(vc, per_stock_window)
                    r = pos[v_d - take : v_d]
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
                    rows.append((syms[i], dt, board, mag))

    if not rows:
        return pd.DataFrame(columns=["symbol", "date", "board", "mag"])
    return pd.DataFrame(rows, columns=["symbol", "date", "board", "mag"])
