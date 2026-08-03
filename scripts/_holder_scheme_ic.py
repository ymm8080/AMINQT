# -*- coding: utf-8 -*-
"""GLM 3特征 vs Kimi 6特征 — 日线截面 Rank IC 对比 (一次性 A/B 测试).

数据:
  - 面板: D:\\AMINQT\\PARQUET\\panel_full_enriched_v3.parquet (仅取 symbol/date/close_hfq/board)
  - 原始事件: data/_holder_cmp_raw.parquet (含 change_ratio/holder_type/evt 起止)

两套方案在同一行子集上度量, 消除覆盖差异:
  - scope=full: 全部主板行
  - scope=active: 每股距最近公告 ≤30 交易日的行 (事件活跃窗口)

标签按 B9 验收口径 label_pm_kd = close_hfq[T+1+k]/close_hfq[T+1] - 1 (k=1/3/5),
另附研究口径 label_1d = close_hfq[T+1]/close_hfq[T] - 1.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
RAW = os.path.join(ROOT, "data", "_holder_cmp_raw.parquet")

MIN_X_UNIQUE = 2
MIN_Y_UNIQUE = 2


# ────────────────────────── 标签 (LabelEngine PM 口径) ──────────────────────────
def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    c = df["close_hfq"]
    g = df.groupby("symbol")["close_hfq"]
    # 研究口径: label_1d = close[T+1]/close[T] - 1
    df["label_1d"] = g.shift(-1) / c - 1
    # PM 执行口径: label_pm_kd = close[T+1+k]/close[T+1] - 1
    c1 = g.shift(-1)
    for k in (1, 3, 5):
        df[f"label_pm_{k}d"] = g.shift(-(k + 1)) / c1 - 1
    return df


# ────────────────────────── 特征计算 ──────────────────────────
def per_symbol_features(df: pd.DataFrame) -> pd.DataFrame:
    """在已 merge 每日事件聚合的面板上, 按 symbol 计算两套特征."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    out = []
    for sym, g in df.groupby("symbol", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        evt = g["net_ratio"].notna()  # 当日有公告记录

        # ── GLM 三特征 (T=行日期, A=最近公告日, S=变动开始, E=变动结束) ──
        ann = g["date"].where(evt).ffill()  # A
        s = g["evt_start_date"].ffill()  # S
        e = g["evt_end_date"].ffill()  # E
        d1 = (g["date"] - ann).dt.days
        d2 = (g["date"] - e).dt.days
        g["glm_ann_decay"] = (1.0 / (1.0 + d1.clip(lower=0))).where(ann.notna())
        g["glm_end_decay"] = pd.Series(
            np.where(d2.ge(0), 1.0 / (1.0 + d2), 0.0), index=g.index
        ).where(e.notna())
        g["glm_is_executing"] = (
            ((g["date"] >= s) & (g["date"] < e))
            .astype(float)
            .where(s.notna() & e.notna())
        )

        # ── Kimi 六特征 (无公告日 = 0, 非 NA) ──
        net = g["net_ratio"].fillna(0.0)
        g["kimi_net_ratio"] = net  # Feat_Net_Change_Ratio
        g["kimi_g_ratio"] = g["g_ratio"].fillna(0.0)  # Feat_G_Change_Ratio
        g["kimi_p_ratio"] = g["p_ratio"].fillna(0.0)  # Feat_P_Change_Ratio
        g["kimi_c_ratio"] = g["c_ratio"].fillna(0.0)  # Feat_C_Change_Ratio
        # Feat_Recent_30d_Sum: 30 交易日滚动求和 (含 T; spec 防前视清单要求 shift(1), 两者都报)
        g["kimi_ratio_30d"] = net.rolling(30, min_periods=1).sum()
        g["kimi_ratio_30d_shift"] = net.rolling(30, min_periods=1).sum().shift(1)
        g["kimi_ann_decay"] = g["glm_ann_decay"]  # 共享特征

        # ── 事件窗口: 每行距最近一次公告 ≤30 天 (±1 个月, 前后对称) ──
        if evt.any():
            ev_arr = g.loc[evt, "date"].values.astype("datetime64[D]").astype(int)
            t_arr = g["date"].values.astype("datetime64[D]").astype(int)
            g["_near_evt_days"] = np.abs(t_arr[:, None] - ev_arr[None, :]).min(axis=1)
        else:
            g["_near_evt_days"] = np.inf
        g["_active"] = g["_near_evt_days"].le(30).astype(float)
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ────────────────────────── 日度截面 Rank IC ──────────────────────────
def daily_ic(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    sub = df[["date", x_col, y_col]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    ics = []
    dates = []
    for d, grp in sub.groupby("date", observed=True):
        x = grp[x_col]
        y = grp[y_col]
        if len(x) < 2 or x.nunique() < MIN_X_UNIQUE or y.nunique() < MIN_Y_UNIQUE:
            continue
        try:
            ics.append(float(spearmanr(x, y).statistic))
            dates.append(d)
        except (ValueError, TypeError):
            continue
    return pd.Series(ics, index=dates)


def ic_metrics(ser: pd.Series) -> dict:
    if ser.empty:
        return {"ic_mean": np.nan, "ic_ir": np.nan, "abs_ic": np.nan, "days": 0}
    m = ser.mean()
    return {
        "ic_mean": m,
        "ic_ir": m / ser.std() if ser.std() > 0 else np.nan,
        "abs_ic": ser.abs().mean(),
        "days": len(ser),
    }


GLM_COLS = ["glm_ann_decay", "glm_end_decay", "glm_is_executing"]
KIMI_COLS = [
    "kimi_net_ratio",
    "kimi_g_ratio",
    "kimi_p_ratio",
    "kimi_c_ratio",
    "kimi_ratio_30d",
    "kimi_ratio_30d_shift",
    "kimi_ann_decay",
]


def main() -> None:
    # 1. 面板
    df = pd.read_parquet(PANEL, columns=["symbol", "date", "close_hfq", "board"])
    df = df[df["board"] == "main"].copy()
    df = df.dropna(subset=["close_hfq"])
    df["date"] = pd.to_datetime(df["date"])
    print(f"panel main-board rows: {len(df)}  ({df['symbol'].nunique()} stocks)")

    # 2. 原始事件 → 每日聚合 (按股东类型拆分, 向量化)
    raw = pd.read_parquet(RAW)
    raw["date"] = pd.to_datetime(raw["date"])
    raw["sr_g"] = np.where(raw["holder_type"] == "G", raw["signed_ratio"], 0.0)
    raw["sr_p"] = np.where(raw["holder_type"] == "P", raw["signed_ratio"], 0.0)
    raw["sr_c"] = np.where(raw["holder_type"] == "C", raw["signed_ratio"], 0.0)
    agg = raw.groupby(["symbol", "date"], as_index=False).agg(
        net_ratio=("signed_ratio", "sum"),
        g_ratio=("sr_g", "sum"),
        p_ratio=("sr_p", "sum"),
        c_ratio=("sr_c", "sum"),
        evt_start_date=("evt_start_date", "min"),
        evt_end_date=("evt_end_date", "max"),
    )
    print(f"event rows aggregated: {len(agg)}")
    df = df.merge(agg, on=["symbol", "date"], how="left")

    # 3. 特征 + 标签
    df = per_symbol_features(df)
    df = add_labels(df)
    print(f"feature/label cols done, rows={len(df)}")

    # 4. IC 对比 (full=全主板; ht_stocks=只在有增减持数据的股票上; active=事件活跃行; 组合)
    ht_symbols = set(raw["symbol"].unique())
    df["_has_ht"] = df["symbol"].isin(ht_symbols).astype(bool)
    for scope, sub in (
        ("full", df),
        ("ht_stocks", df[df["_has_ht"]]),
        ("active", df[df["_active"] == 1.0]),
        ("ht_stocks+active", df[df["_has_ht"] & (df["_active"] == 1.0)]),
    ):
        print()
        print("=" * 92)
        print(f"SCOPE: {scope}  (rows={len(sub)}, stocks={sub['symbol'].nunique()})")
        print("=" * 92)
        hdr = f"{'feature':<22s} {'ic1':>7s} {'ic3':>7s} {'ic5':>7s} {'ir1':>7s} {'days1':>6s} {'nan%':>6s}"
        print(hdr)
        print("-" * 92)
        for col in GLM_COLS + KIMI_COLS:
            if col not in sub.columns:
                continue
            nan_pct = float(sub[col].isna().mean() * 100)
            ic1 = ic_metrics(daily_ic(sub, col, "label_pm_1d"))
            ic3 = ic_metrics(daily_ic(sub, col, "label_pm_3d"))
            ic5 = ic_metrics(daily_ic(sub, col, "label_pm_5d"))
            tag = "GLM " if col.startswith("glm") else "KIMI"
            print(
                f"{col:<22s} {ic1['ic_mean']:>7.4f} {ic3['ic_mean']:>7.4f} "
                f"{ic5['ic_mean']:>7.4f} {ic1['ic_ir']:>7.2f} "
                f"{ic1['days']:>6d} {nan_pct:>6.1f}   {tag}"
            )
        # 研究口径 label_1d 复核 (事件日 T→T+1)
        print("-" * 92)
        print("research label_1d (T→T+1):")
        for col in GLM_COLS + KIMI_COLS:
            if col not in sub.columns:
                continue
            ic1 = ic_metrics(daily_ic(sub, col, "label_1d"))
            print(
                f"  {col:<22s} ic={ic1['ic_mean']:>7.4f} ir={ic1['ic_ir']:>7.2f} days={ic1['days']:>6d}"
            )

    # 5. 分组单调性 (净比率 → 未来收益) — 事件日样本
    print()
    print("=" * 92)
    print("GROUPED forward return by net_ratio decile (event rows only, label_pm_1d)")
    print("=" * 92)
    evt_rows = df[df["net_ratio"].notna() & df["label_pm_1d"].notna()].copy()
    evt_rows["decile"] = pd.qcut(
        evt_rows["net_ratio"], 5, labels=False, duplicates="drop"
    )
    if evt_rows["decile"].notna().any():
        grp = evt_rows.groupby("decile", observed=True).agg(
            n=("label_pm_1d", "size"),
            fwd=("label_pm_1d", "mean"),
            net=("net_ratio", "mean"),
        )
        print(grp.to_string())
        g0 = grp.iloc[0]["fwd"]
        g4 = grp.iloc[-1]["fwd"]
        print(f"Q5-Q1 spread: {g4 - g0:+.4f}")

    # 6. 汇总决策表
    print()
    print("=" * 92)
    print("SCHEME SUMMARY (ht_stocks + ±1mo window, label_pm_1d/3d/5d)")
    print("=" * 92)
    sub = df[df["_has_ht"] & (df["_active"] == 1.0)]
    for scheme, cols in (("GLM", GLM_COLS), ("KIMI", KIMI_COLS)):
        row = {}
        for col in cols:
            best = max(
                (
                    ic_metrics(daily_ic(sub, col, f"label_pm_{k}d"))["abs_ic"]
                    for k in (1, 3, 5)
                ),
                default=0.0,
            )
            row[col] = best
        n_pass = sum(1 for v in row.values() if v > 0.02)
        best_col = max(row, key=row.get)
        print(
            f"{scheme}: features={len(cols)}, |IC|>0.02 pass={n_pass}, best={best_col} absIC={row[best_col]:.4f}"
        )


if __name__ == "__main__":
    main()
