# -*- coding: utf-8 -*-
"""增持/减持信号 → "5天内会涨吗" 分类器的命中率 (买入清单视角).

用户目标: 产出 5 天持有期的买入清单. 评估口径 = 准确率 (命中率), 不是全市场排序.
  - 预测: "该股 5 天净收益 > 0" (label_pm_5d_net, B9 PM 执行口径, 扣 COST+2×滑点)
  - 信号: 过去 N 个交易日内有增持公告 / 有减持公告 (trailing 因果窗口, 无前视)
  - 度量: 命中率 vs 基线, lift, 平均 5d 净收益, 每日信号数 (清单规模)

生产掩码: 主板 + 非 ST + 停牌窗口标签置 NaN + 近 6 交易日标签置 NaN.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.pipeline1.label_engine import COST, slippage_tier  # noqa: E402

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
RAW = os.path.join(ROOT, "data", "_holder_cmp_raw.parquet")

WINDOWS = (1, 3, 5, 10, 20, 30)


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 区间 (二项命中率的置信区间)."""
    if n == 0:
        return np.nan, np.nan
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def main() -> None:
    # 1. 面板 → 主板 + 非 ST
    df = pd.read_parquet(
        PANEL,
        columns=[
            "symbol",
            "date",
            "close_hfq",
            "amount",
            "board",
            "is_suspended",
            "is_st",
        ],
    )
    df = df[df["board"] == "main"].copy()
    df = df[df["is_st"].ne(True)].copy() if df["is_st"].dtype.kind == "b" else df.copy()
    if "is_st" in df.columns and df["is_st"].astype(int).sum() > 0:
        df = df[df["is_st"] != True].copy()  # noqa: E712
    df = df.dropna(subset=["close_hfq"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    print(f"panel main-board non-ST rows: {len(df)}  stocks={df['symbol'].nunique()}")

    # 2. 事件 → 每日净比率 (ann_date 为信息公开日, 无前视)
    raw = pd.read_parquet(RAW)
    raw["date"] = pd.to_datetime(raw["date"])
    net = (
        raw.groupby(["symbol", "date"], as_index=False)["signed_ratio"]
        .sum()
        .rename(columns={"signed_ratio": "net_ratio"})
    )
    df = df.merge(net, on=["symbol", "date"], how="left")

    # 3. 标签: label_pm_5d / 3d = close[T+1+k]/close[T+1]-1; net = 毛 - (COST + 2×滑点)
    g = df.groupby("symbol")["close_hfq"]
    c1 = g.shift(-1)
    for k in (3, 5):
        df[f"label_pm_{k}d"] = g.shift(-(k + 1)) / c1 - 1
    if "amount" in df.columns:
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        df["adv20"] = (
            df.groupby("symbol")["amount"]
            .rolling(20, min_periods=20)
            .mean()
            .reset_index(level=0, drop=True)
        )
        slip = df["adv20"].map(slippage_tier)
    else:
        slip = pd.Series(0.0015, index=df.index)
    cost_total = COST + 2 * slip
    for k in (3, 5):
        df[f"label_pm_{k}d_net"] = df[f"label_pm_{k}d"] - cost_total

    # 4. 掩码: 停牌窗口 [T, T+k+1] 内停牌 → 标签 NaN; 近 6 交易日 → NaN
    n = 5
    susp = (
        df.groupby("symbol")["is_suspended"]
        .rolling(n + 2)
        .sum()
        .shift(-(n + 2))
        .reset_index(level=0, drop=True)
    )
    df.loc[susp.fillna(0) > 0, [f"label_pm_{k}d" for k in (3, 5)]] = np.nan
    df.loc[susp.fillna(0) > 0, [f"label_pm_{k}d_net" for k in (3, 5)]] = np.nan
    recent = sorted(df["date"].unique())[-6:]
    df.loc[df["date"].isin(recent), [f"label_pm_{k}d" for k in (3, 5)]] = np.nan
    df.loc[df["date"].isin(recent), [f"label_pm_{k}d_net" for k in (3, 5)]] = np.nan

    # 5. 信号: 过去 N 日 增持额 / 减持额 (滚动和)
    netp = df["net_ratio"].fillna(0.0).clip(lower=0.0)
    netm = df["net_ratio"].fillna(0.0).clip(upper=0.0).abs()

    def _roll(s: pd.Series, N: int) -> pd.Series:
        return (
            s.groupby(df["symbol"])
            .rolling(N, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

    for N in WINDOWS:
        df[f"inc_{N}"] = _roll(netp, N)
        df[f"dec_{N}"] = _roll(netm, N)

    # 6. 基线 + 各信号命中率
    d5 = df.dropna(subset=["label_pm_5d_net"])
    base_rate = (d5["label_pm_5d_net"] > 0).mean()
    base_mean = d5["label_pm_5d_net"].mean()
    n_dates = d5["date"].nunique()
    print(f"\n基线 (全主板非ST, n={len(d5)}, 交易日={n_dates}):")
    print(
        f"  5d净命中率 P(5d净>0) = {base_rate * 100:.2f}%  平均5d净 = {base_mean * 100:+.3f}%"
    )

    def _report(signal: str, desc: str) -> None:
        sub = d5[d5[signal].fillna(0) > 0]
        n = len(sub)
        if n < 30:
            print(f"  [{desc:<28s}] n={n:<6d} 样本过少, 跳过")
            return
        hit = (sub["label_pm_5d_net"] > 0).mean()
        mean5 = sub["label_pm_5d_net"].mean()
        mean5g = sub["label_pm_5d"].mean()
        lo, hi = _wilson(hit, n)
        per_day = n / n_dates
        lift = (hit - base_rate) / base_rate * 100
        print(
            f"  [{desc:<28s}] n={n:<7d} {per_day:6.1f}/日  命中={hit * 100:6.2f}% "
            f"({lo * 100:.2f}-{hi * 100:.2f})  lift={lift:+6.1f}%  平均5d净={mean5 * 100:+6.3f}% "
            f"(毛={mean5g * 100:+.3f}%)"
        )

    print("\n===== 信号命中率 (label_pm_5d_net>0, 扣 COST+2×滑点) =====")
    _report("inc_1", "增持(公告当日)")
    for N in WINDOWS:
        _report(f"inc_{N}", f"增持 近{N}日")
        _report(f"dec_{N}", f"减持 近{N}日")
    # 强增持: 近5日累计净增持比例 > 阈值
    for thr in (0.5, 1.0, 2.0):
        _report("inc_5", f"强增持 近5日>={thr}%")
        strong = d5["inc_5"].fillna(0) > thr
        sub = d5[strong]
        if len(sub) < 30:
            continue
        hit = (sub["label_pm_5d_net"] > 0).mean()
        lo, hi = _wilson(hit, len(sub))
        print(
            f"    └→ 仅近5日累计>{thr}% (n={len(sub)}, {len(sub) / n_dates:.1f}/日): "
            f"命中={hit * 100:.2f}% ({lo * 100:.2f}-{hi * 100:.2f})  平均5d净={sub['label_pm_5d_net'].mean() * 100:+.3f}%"
        )
    # 净增持(增持-减持)近5日 > 0
    sub = d5[d5["inc_5"].fillna(0) - d5["dec_5"].fillna(0) > 0]
    if len(sub) >= 30:
        hit = (sub["label_pm_5d_net"] > 0).mean()
        lo, hi = _wilson(hit, len(sub))
        print(
            f"  [净增持 近5日>0          ] n={len(sub):<7d} {len(sub) / n_dates:6.1f}/日  "
            f"命中={hit * 100:.2f}% ({lo * 100:.2f}-{hi * 100:.2f})  平均5d净={sub['label_pm_5d_net'].mean() * 100:+.3f}%"
        )

    # 7. 单调性: 近5日净比率五分位 → 命中率
    print("\n===== 近5日累计净比率五分位 → 5d净命中率 (单调性) =====")
    q = d5.copy()
    q["net5"] = q["inc_5"].fillna(0) - q["dec_5"].fillna(0)
    q = q.dropna(subset=["net5", "label_pm_5d_net"])
    # 先去掉全零 (无事件) 行, 在事件行里分位
    evq = q[q["net5"].ne(0)].copy()
    if len(evq) >= 100:
        evq["bin"] = pd.qcut(evq["net5"].rank(method="first"), 5, labels=False)
        g = evq.groupby("bin").agg(
            n=("label_pm_5d_net", "size"),
            hit=("label_pm_5d_net", lambda s: (s > 0).mean()),
            mean=("label_pm_5d_net", "mean"),
            x=("net5", "mean"),
        )
        g["hit_%"] = g["hit"] * 100
        g["mean_%"] = g["mean"] * 100
        print(g[["n", "x", "hit_%", "mean_%"]].round(4).to_string())
        h0, h4 = g["hit"].iloc[0], g["hit"].iloc[-1]
        print(f"  Q5-Q1 命中率差 = {(h4 - h0) * 100:+.2f}pp")

    print(
        f"\n成本口径: COST={COST:.4f}, 滑点单边 {slip.min():.4f}~{slip.max():.4f} → round-trip "
        f"{(cost_total.min() * 100):.3f}%~{(cost_total.max() * 100):.3f}%"
    )


if __name__ == "__main__":
    main()
