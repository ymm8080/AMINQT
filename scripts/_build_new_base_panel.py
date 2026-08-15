"""全市场宇宙修复 Step 3: 新股票基础面板构建 (2026-08-15).

把 _pull_universe_raw.py 拉回的数据组装成与生产面板同语义的 base 面板
(公式逐条对齐 _daily_fetch.py 的 merge/derive 块):

  - OHLCV raw + hfq (raw × adj_factor, 4 列)
  - amount 千元→元 (×1000); volume = amount/close (新股票统一 gu 惯例)
  - daily_basic 估值/换手/股本/市值 + free_float_turnover_rate(=turnover_rate_f)
  - stk_limit up/down_limit_raw; suspend → is_suspended
  - board=board_of; industry=东财 meta map (缺失 UNKNOWN)
  - cyq: cost_50pct/cost_95pct/weight_avg/winner_rate →
    winner_ratio(=winner_rate/100) / avg_cost(=cost_50pct) /
    pct_90_high(=cost_95pct) / pct_90_con=(c95-c5)/(c95+c5)
  - margin 4 列 + lhb 3 列 (恢复的缓存, 面板口径)
  - 入库 gate: 上市交易日数 <150 剔行 (与 ingest_scan 同公式, 逐行向量化)
  - 派生特征 (groupby rolling, 禁 for 循环):
    intraday_range / pctChg / bias_5..250 (close_hfq) / bias_5_20_cross /
    bias_20_60_cross / ma_vol_ratio_5_20 / amplitude_5d / vol_surge / amt_surge /
    cost_bias / conc_trend_20d / conc_90_industry_rank

WORM: data/new_symbols_panel/base_new_<ts>.parquet
"""

from __future__ import annotations

import glob
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pipeline1.cleaning_pipeline import board_of  # noqa: E402
from config.settings import INGEST_MIN_LIST_DAYS  # noqa: E402

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
OUT_DIR = "data/new_symbols_raw"
ALT_DIR = "data/supply_cache/alt_data"
OUT_PANEL_DIR = "data/new_symbols_panel"


def _load_glob(pat: str, cols: list[str] | None = None) -> pd.DataFrame:
    files = sorted(glob.glob(pat))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f, columns=cols) for f in files]
    return pd.concat(frames, ignore_index=True)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.where((b.notna() & (b != 0)), a / b, np.nan)


def _load_meta_industry() -> dict[str, str]:
    """东财行业 map (生产语义, 缺省 UNKNOWN). 用当日缓存, 失败则尝试拉取."""
    from app.pipeline1.panel_builder import load_or_fetch_meta

    try:
        ind_map, _ = load_or_fetch_meta()
        if ind_map:
            return ind_map
    except Exception as exc:  # noqa: BLE001
        print(f"[industry] meta 拉取失败: {exc}", flush=True)
    # 回退: 已缓存的任意一日 stock_meta
    metas = sorted(glob.glob(r"D:\AMINQT\DATA OTHERS\data\processed\stock_meta_*.json"))
    if metas:
        import json

        with open(metas[-1], encoding="utf-8") as fh:
            return json.load(fh).get("industry_map", {})
    return {}


def main() -> None:
    universe = pd.read_parquet(
        sorted(glob.glob("data/new_universe/new_symbols_*.parquet"))[-1]
    )
    newsyms = set(universe["symbol"].astype(str).str.strip())
    print(f"[base] new symbols={len(newsyms)}", flush=True)

    # ── 1. raw 输入 ──
    daily = _load_glob(os.path.join(OUT_DIR, "daily", "daily_*.parquet"))
    basic = _load_glob(os.path.join(OUT_DIR, "daily_basic", "daily_basic_*.parquet"))
    susp = _load_glob(os.path.join(OUT_DIR, "suspend", "suspend_*.parquet"))
    adj = _load_glob(os.path.join(ALT_DIR, "adj_factor", "adj_*.parquet"))
    limit = _load_glob(os.path.join(ALT_DIR, "stk_limit", "*_all__.parquet"))
    cyq = _load_glob(os.path.join(ALT_DIR, "cyq_tushare", "cyq_full_*.parquet"))
    margin = _load_glob(os.path.join(ALT_DIR, "margin_panel_*.parquet"))
    lhb = _load_glob(os.path.join(ALT_DIR, "lhb", "all_*.parquet"))

    daily = daily[daily["symbol"].isin(newsyms)].copy()
    basic = basic[basic["symbol"].isin(newsyms)].copy()
    susp = susp[susp["symbol"].isin(newsyms)].copy()
    adj = adj[adj["symbol"].isin(newsyms)].copy()
    limit = limit[limit["symbol"].isin(newsyms)].copy()
    cyq = cyq[cyq["symbol"].isin(newsyms)].copy()
    margin = margin[margin["symbol"].isin(newsyms)].copy()
    lhb = lhb[lhb["symbol"].isin(newsyms)].copy()

    for d in (daily, basic, susp, adj, limit, cyq, margin, lhb):
        if "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"])
        if "trade_date" in d.columns:
            d["trade_date"] = d["trade_date"].astype(str)
    daily = daily.drop_duplicates(subset=["symbol", "date"])
    basic = basic.drop_duplicates(subset=["symbol", "date"])
    susp = susp.drop_duplicates(subset=["symbol", "date"])
    adj = adj.drop_duplicates(subset=["symbol", "trade_date"])
    limit = limit.drop_duplicates(subset=["symbol", "date"])
    cyq = cyq.drop_duplicates(subset=["symbol", "trade_date"])
    margin = margin.drop_duplicates(subset=["symbol", "date"])
    lhb = lhb.drop_duplicates(subset=["symbol", "date"])
    print(
        f"[base] daily={len(daily):,} basic={len(basic):,} susp={len(susp):,} "
        f"adj={len(adj):,} limit={len(limit):,} cyq={len(cyq):,} "
        f"margin={len(margin):,} lhb={len(lhb):,}",
        flush=True,
    )
    if not len(daily):
        raise SystemExit("FATAL: daily 空 — 拉取未完成, 先跑 _pull_universe_raw.py")

    # ── 2. 主表: daily + adj_factor → hfq ──
    df = daily[
        ["symbol", "date", "open", "high", "low", "close", "pre_close", "amount", "vol"]
    ].copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * 1000.0  # 千元→元
    adj_m = adj[["symbol", "trade_date", "adj_factor"]].rename(
        columns={"trade_date": "date"}
    )
    adj_m["date"] = pd.to_datetime(adj_m["date"], format="%Y%m%d", errors="coerce")
    df = df.merge(adj_m, on=["symbol", "date"], how="left")
    factor = pd.to_numeric(df["adj_factor"], errors="coerce")
    for c in ["open", "high", "low", "close"]:
        df[f"{c}_hfq"] = df[c] * factor
    df["adj_factor"] = factor

    # volume = amount/close (股, 新股票统一 gu 惯例 — 无面板历史可判 hand)
    valid = df["amount"].notna() & df["close"].notna() & (df["close"] > 0)
    df["volume"] = np.nan
    df.loc[valid, "volume"] = df.loc[valid, "amount"] / df.loc[valid, "close"]

    # ── 3. daily_basic ──
    b_cols = [
        c
        for c in [
            "volume_ratio",
            "pe_ttm",
            "pb",
            "ps_ttm",
            "total_share",
            "float_share",
            "free_share",
            "total_mv",
            "circ_mv",
            "dv_ratio",
            "dv_ttm",
            "turnover_rate",
            "turnover_rate_f",
        ]
        if c in basic.columns
    ]
    basic_m = basic[["symbol", "date"] + b_cols].copy()
    df = df.merge(basic_m, on=["symbol", "date"], how="left")
    if "turnover_rate_f" in df.columns:
        df["free_float_turnover_rate"] = df["turnover_rate_f"]
    else:
        df["free_float_turnover_rate"] = df["turnover_rate"]

    # ── 4. stk_limit / suspend / board / industry ──
    if len(limit):
        df = df.merge(
            limit[["symbol", "date", "up_limit_raw", "down_limit_raw"]],
            on=["symbol", "date"],
            how="left",
        )
    else:
        df["up_limit_raw"] = np.nan
        df["down_limit_raw"] = np.nan
    susp["_sus"] = 1
    df = df.merge(susp[["symbol", "date", "_sus"]], on=["symbol", "date"], how="left")
    df["is_suspended"] = df["_sus"].fillna(0).astype(int)
    df = df.drop(columns=["_sus"])
    df["board"] = df["symbol"].map(board_of)
    ind_map = _load_meta_industry()
    df["industry"] = df["symbol"].map(ind_map).fillna("UNKNOWN")

    # ── 5. cyq base + 派生 ──
    if len(cyq):
        cyq_m = cyq[
            [
                "symbol",
                "trade_date",
                "cost_50pct",
                "cost_95pct",
                "weight_avg",
                "winner_rate",
                "cost_5pct",
            ]
        ].rename(columns={"trade_date": "date"})
        cyq_m["date"] = pd.to_datetime(cyq_m["date"], format="%Y%m%d", errors="coerce")
        df = df.merge(cyq_m, on=["symbol", "date"], how="left")
    else:
        for c in ["cost_50pct", "cost_95pct", "weight_avg", "winner_rate", "cost_5pct"]:
            df[c] = np.nan
    df["winner_ratio"] = pd.to_numeric(df["winner_rate"], errors="coerce") / 100.0
    df["avg_cost"] = df["cost_50pct"]
    df["pct_90_high"] = df["cost_95pct"]
    df["pct_90_con"] = _safe_div(
        pd.to_numeric(df["cost_95pct"], errors="coerce")
        - pd.to_numeric(df["cost_5pct"], errors="coerce"),
        pd.to_numeric(df["cost_95pct"], errors="coerce")
        + pd.to_numeric(df["cost_5pct"], errors="coerce"),
    )
    df = df.drop(columns=["cost_5pct"])

    # ── 6. margin / lhb (恢复缓存, 已是面板口径) ──
    for src, cols in [
        (
            margin,
            ["margin_balance", "short_balance", "margin_buy_amt", "short_sell_vol"],
        ),
        (lhb, ["lhb_net_buy", "lhb_buy_amt", "lhb_sell_amt"]),
    ]:
        if len(src):
            avail = [c for c in cols if c in src.columns]
            df = df.merge(
                src[["symbol", "date"] + avail], on=["symbol", "date"], how="left"
            )
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan

    # ── 7. 入库 gate: 上市 <150 交易日剔行 (与 ingest_scan 同公式) ──
    cal = pd.DatetimeIndex(
        sorted(pd.read_parquet(PANEL, columns=["date"])["date"].unique())
    )
    ld_map = dict(
        zip(
            universe["symbol"].astype(str).str.strip(),
            universe["list_date"],
            strict=False,
        )
    )
    ld = pd.to_datetime(df["symbol"].map(ld_map), format="%Y%m%d", errors="coerce")
    left = cal.searchsorted(ld, side="left")
    right = cal.searchsorted(df["date"], side="right")
    list_days = right - left
    before = len(df)
    df = df[list_days >= INGEST_MIN_LIST_DAYS].copy()
    print(
        f"[gate] 剔次新 {before - len(df):,} 行 (list_days < {INGEST_MIN_LIST_DAYS})",
        flush=True,
    )

    # ── 8. 派生特征 (向量化 groupby) ──
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["intraday_range"] = (df["high"] - df["low"]) / df["pre_close"].replace(0, np.nan)
    df["pctChg"] = (df["close"] / df["pre_close"] - 1) * 100

    g = df.groupby("symbol", sort=False)
    hc = df["close_hfq"]
    for w in (5, 10, 20, 60, 120, 250):
        ma = g["close_hfq"].rolling(w, min_periods=w).mean()
        ma.index = ma.index.droplevel(0)
        df[f"bias_{w}"] = hc / ma - 1
    df["bias_5_20_cross"] = (
        np.sign(df["bias_5"] - df["bias_20"]).groupby(df["symbol"]).diff().fillna(0)
    )
    df["bias_20_60_cross"] = (
        np.sign(df["bias_20"] - df["bias_60"]).groupby(df["symbol"]).diff().fillna(0)
    )
    ma5v = g["volume"].rolling(5, min_periods=3).mean()
    ma5v.index = ma5v.index.droplevel(0)
    ma20v = g["volume"].rolling(20, min_periods=10).mean()
    ma20v.index = ma20v.index.droplevel(0)
    df["ma_vol_ratio_5_20"] = _safe_div(ma5v, ma20v)
    # amplitude_5d: 滚动 5 日均 (high-low)/pre_close
    amp = (df["high"] - df["low"]) / df["pre_close"].replace(0, np.nan)
    a5 = amp.groupby(df["symbol"]).rolling(5, min_periods=3).mean()
    a5.index = a5.index.droplevel(0)
    df["amplitude_5d"] = a5
    for col, surge in [("volume", "vol_surge"), ("amount", "amt_surge")]:
        m20 = g[col].rolling(20, min_periods=10).mean()
        m20.index = m20.index.droplevel(0)
        s20 = g[col].rolling(20, min_periods=10).std()
        s20.index = s20.index.droplevel(0)
        df[surge] = _safe_div(df[col] - m20, s20)
    # cyq 派生
    df["cost_bias"] = _safe_div(
        df["close_hfq"] - df["cost_50pct"],
        pd.to_numeric(df["cost_50pct"], errors="coerce"),
    )
    p90 = df["pct_90_con"].groupby(df["symbol"]).shift(19)
    df["conc_trend_20d"] = _safe_div(df["pct_90_con"], p90)
    df["conc_90_industry_rank"] = (
        df.groupby(["date", "industry"], observed=True)["pct_90_con"]
        .rank(pct=True)
        .fillna(0.5)
    )

    # ── 9. 保存 (WORM) ──
    os.makedirs(OUT_PANEL_DIR, exist_ok=True)
    ts_ = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_PANEL_DIR, f"base_new_{ts_}.parquet")
    df.to_parquet(out, index=False)
    print(f"[save] {out}", flush=True)
    print(
        f"[stat] rows={len(df):,} symbols={df['symbol'].nunique()} "
        f"dates={df['date'].min().date()}..{df['date'].max().date()} "
        f"cols={len(df.columns)}",
        flush=True,
    )
    cov = (
        df[
            [
                "close",
                "close_hfq",
                "volume",
                "turnover_rate",
                "pe_ttm",
                "up_limit_raw",
                "weight_avg",
                "winner_ratio",
                "margin_balance",
                "lhb_net_buy",
                "industry",
            ]
        ]
        .notna()
        .mean()
        .round(3)
    )
    print("[coverage]")
    print(cov.to_string(), flush=True)
    print("BASE BUILD DONE", flush=True)


if __name__ == "__main__":
    main()
