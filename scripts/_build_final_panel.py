"""Build final new symbols panel with all sources."""

import glob
import os
import sys

import pandas as pd

OUT_DIR = "data/new_symbols_raw"
ALT_DIR = "data/supply_cache/alt_data"
PANEL_OUT = "panel_final.parquet"


def load_parquets(pattern: str) -> pd.DataFrame:
    fs = sorted(glob.glob(pattern))
    if not fs:
        return pd.DataFrame()
    dfs = []
    for f in fs:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"WARN: {f} bad parquet: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def main():
    print("[daily]")
    daily = load_parquets(os.path.join(OUT_DIR, "daily", "daily_*.parquet"))
    print(f"  {len(daily)} rows")
    print("[adj_factor]")
    adj = load_parquets(os.path.join(ALT_DIR, "adj_factor", "adj_*.parquet"))
    print(f"  {len(adj)} rows")
    print("[daily_basic]")
    basic = load_parquets(os.path.join(OUT_DIR, "daily_basic", "daily_basic_*.parquet"))
    print(f"  {len(basic)} rows")
    print("[stk_limit]")
    limit = load_parquets(os.path.join(ALT_DIR, "stk_limit", "*_all__.parquet"))
    print(f"  {len(limit)} rows")
    print("[suspend]")
    susp = load_parquets(os.path.join(OUT_DIR, "suspend", "suspend_*.parquet"))
    print(f"  {len(susp)} rows")
    print("[margin]")
    margin = load_parquets(os.path.join(ALT_DIR, "margin_*.parquet"))
    print(f"  {len(margin)} rows")
    print("[lhb]")
    lhb = load_parquets(os.path.join(ALT_DIR, "lhb", "lhb_*.parquet"))
    print(f"  {len(lhb)} rows")
    print("[block_trade]")
    bt = load_parquets(os.path.join(ALT_DIR, "block_trade", "bt_*.parquet"))
    print(f"  {len(bt)} rows")
    print("[cyq]")
    cyq = load_parquets(os.path.join(ALT_DIR, "cyq_tushare", "cyq_*.parquet"))
    print(f"  {len(cyq)} rows")
    print("[top_inst]")
    topinst = load_parquets(os.path.join(OUT_DIR, "top_inst", "top_inst_*.parquet"))
    print(f"  {len(topinst)} rows")
    print("[fina]")
    fina = load_parquets(os.path.join(OUT_DIR, "fina", "fina_*.parquet"))
    print(f"  {len(fina)} rows")

    for df in [daily, adj, basic, limit, susp, margin, lhb, bt, cyq, topinst, fina]:
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

    panel = daily
    if adj.empty:
        print("ERROR: no adj_factor")
        return 1
    adj_ = adj.groupby(["symbol", "trade_date"], as_index=False)["adj_factor"].last()
    panel = pd.merge(panel, adj_, on=["symbol", "trade_date"], how="left")
    panel["adj_factor"] = panel["adj_factor"].fillna(1.0)
    for c in ["open", "high", "low", "close", "pre_close"]:
        if c in panel.columns:
            panel[f"{c}_hfq"] = panel[c] * panel["adj_factor"]
    panel = panel.sort_values(["symbol", "trade_date"])
    panel["adj_ratio"] = panel.groupby("symbol")["adj_factor"].transform(
        lambda x: x / x.shift(1).fillna(1.0)
    )

    if not basic.empty:
        panel = pd.merge(panel, basic, on=["symbol", "trade_date"], how="left")
    if not limit.empty:
        panel = pd.merge(
            panel,
            limit[["symbol", "trade_date", "up_limit_raw", "down_limit_raw"]],
            on=["symbol", "trade_date"],
            how="left",
        )
    if not susp.empty:
        susp["is_suspend"] = 1
        panel = pd.merge(
            panel,
            susp[["symbol", "trade_date", "is_suspend"]],
            on=["symbol", "trade_date"],
            how="left",
        )
        panel["is_suspend"] = panel["is_suspend"].fillna(0)
    if not margin.empty:
        m = margin[
            [
                "symbol",
                "date",
                "margin_balance",
                "short_balance",
                "margin_buy_amt",
                "short_sell_vol",
            ]
        ].copy()
        m.rename(columns={"date": "trade_date"}, inplace=True)
        panel = pd.merge(panel, m, on=["symbol", "trade_date"], how="left")
    if not lhb.empty:
        l = lhb[["symbol", "date", "lhb_buy_amt", "lhb_sell_amt", "lhb_net_buy"]].copy()
        l.rename(columns={"date": "trade_date"}, inplace=True)
        panel = pd.merge(panel, l, on=["symbol", "trade_date"], how="left")
    if not bt.empty:
        bt_ = bt[
            ["symbol", "trade_date", "price", "vol", "amount", "buyer", "seller"]
        ].copy()
        panel = pd.merge(
            panel, bt_, on=["symbol", "trade_date"], how="left", suffixes=("", "_bt")
        )
    if not cyq.empty:
        c = cyq[
            [
                "symbol",
                "trade_date",
                "his_low",
                "his_high",
                "cost_5pct",
                "cost_15pct",
                "cost_50pct",
                "cost_85pct",
                "cost_95pct",
                "weight_avg",
                "winner_rate",
            ]
        ].copy()
        panel = pd.merge(panel, c, on=["symbol", "trade_date"], how="left")
    if not topinst.empty:
        ti = (
            topinst.groupby(["symbol", "trade_date"])
            .agg(top_inst_buy=("buy", "sum"), top_inst_sell=("sell", "sum"))
            .reset_index()
        )
        panel = pd.merge(panel, ti, on=["symbol", "trade_date"], how="left")

    panel = panel.sort_values(["symbol", "trade_date"])
    panel["gap_up_5pct"] = (panel["open"] >= panel["pre_close"] * 1.05).astype(int)
    panel["gap_up_5pct_cnt"] = panel.groupby("symbol")["gap_up_5pct"].transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    panel["vol_20d"] = panel.groupby("symbol")["vol"].transform(
        lambda x: x.rolling(20, min_periods=5).mean()
    )
    panel["vol_chg_20d"] = panel["vol"] / (panel["vol_20d"] + 1e-8) - 1

    cols = ["symbol", "trade_date"] + [
        c for c in panel.columns if c not in ("symbol", "trade_date")
    ]
    panel = panel[cols]
    panel = panel.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    panel.to_parquet(os.path.join(OUT_DIR, PANEL_OUT), index=False)
    print(f"[done] {len(panel)} rows -> {OUT_DIR}/{PANEL_OUT}")
    print(f"  symbols={panel['symbol'].nunique()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
