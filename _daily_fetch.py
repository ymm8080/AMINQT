"""Daily fetch script — append one day's data to V3 panel from Tushare.
NEVER loads full panel into memory. Uses pyarrow streaming read/write.

Usage: python _daily_fetch.py [YYYYMMDD]  (default: today)

Sources fetched:
  1. OHLCV (daily) — all stocks, filtered to panel universe
  2. adj_factor → hfq prices
  3. daily_basic — turnover, PE, PB, float, MV
  4. stk_limit — up/down limit prices
  5. suspend_d — suspension markers
  6. moneyflow — main/super-large order flow
  7. cyq_perf — chip distribution (batch, one call)
  8. margin_detail — margin balance (per-stock)
  9. top_list — LHB (dragon-tiger board)
 10. sw_daily — sector index (Shenwan industry)

Derived (computed from panel history, reads only needed columns):
  - bias_5..250, bias_cross, ma_vol_ratio_5_20, amplitude_5d
  - vol_surge, amt_surge, ret_pct
  - pct_70_con, pct_90_con (cyq formula: (hi-lo)/(hi+lo), [0,1] range)
  - Forward-fill: financials (quarterly), margin(T+1 gap), announce_date

NOT fetched here (separate pipelines):
  - fina_indicator → run_announcement_pipeline.py (Pipeline 2, quarterly)
  - holdertrade/holdernumber → removed from panel (no longer tracked)
"""
import os
import sys
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import tushare as ts
from dotenv import load_dotenv
load_dotenv()

TRADE_DATE = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y%m%d")
PANEL = os.getenv("PANEL_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")

_token = os.getenv("TUSHARE_TOKEN") or ts.get_token()
if not _token:
    print("FATAL: No Tushare token found. Set TUSHARE_TOKEN or run `tushare.set_token('your_token')`")
    sys.exit(1)
pro = ts.pro_api(_token)

# ── 0. Get valid stock universe from panel ──
print("[0] Getting stock universe...")
yesterday = pq.read_table(PANEL, columns=["symbol", "date"]).to_pandas()
max_date = yesterday["date"].max()
valid_universe = set(yesterday[yesterday["date"] == max_date]["symbol"].unique())
print(f"    Universe: {len(valid_universe)} stocks, max date: {max_date.date()}")

if pd.Timestamp(TRADE_DATE) <= max_date:
    print(f"    {TRADE_DATE} already in panel (max={max_date.date()}), will replace today's rows.")

# ── 1. Fetch all Tushare sources ──
print(f"\n[1] Fetching Tushare data for {TRADE_DATE}...")

def to_symbol(df):
    if "ts_code" in df.columns:
        df["symbol"] = df["ts_code"].str.replace(".SZ", "").str.replace(".SH", "")
    return df

def safe_fetch(fn, name, **kwargs):
    try:
        df = fn(**kwargs)
        print(f"    {name}: {len(df)} rows")
        return to_symbol(df)
    except Exception as e:
        print(f"    {name}: FAILED ({e})")
        return pd.DataFrame()

ohlcv = safe_fetch(pro.daily, "OHLCV", trade_date=TRADE_DATE)
adj   = safe_fetch(pro.adj_factor, "adj_factor", trade_date=TRADE_DATE)
basic = safe_fetch(pro.daily_basic, "daily_basic", trade_date=TRADE_DATE)
limit = safe_fetch(pro.stk_limit, "stk_limit", trade_date=TRADE_DATE)
susp  = safe_fetch(pro.suspend_d, "suspend", trade_date=TRADE_DATE)
money = safe_fetch(pro.moneyflow, "moneyflow", trade_date=TRADE_DATE)
cyq   = safe_fetch(pro.cyq_perf, "cyq_perf", trade_date=TRADE_DATE)
margin= safe_fetch(pro.margin_detail, "margin_detail", trade_date=TRADE_DATE)
lhb   = safe_fetch(pro.top_list, "LHB", trade_date=TRADE_DATE)
sw    = safe_fetch(pro.sw_daily, "sw_daily", trade_date=TRADE_DATE)

if not len(ohlcv):
    print("FATAL: No OHLCV data")
    sys.exit(1)

# ── 2. Build today's DataFrame ──
print("\n[2] Building today's frame...")
# Start from OHLCV
df = ohlcv.copy()
df["date"] = pd.Timestamp(TRADE_DATE)
# Filter to valid universe
df = df[df["symbol"].isin(valid_universe)].copy()
print(f"    After universe filter: {len(df)} rows")

# Read panel schema
schema = pq.read_schema(PANEL)
panel_cols = schema.names

# ── 3. Merge all sources ──
print("\n[3] Merging sources...")
merges = {
    "daily_basic": (basic, ["volume_ratio",
        "pe", "pe_ttm", "pb", "ps", "ps_ttm", "total_share", "float_share",
        "free_share", "total_mv", "circ_mv", "dv_ratio", "dv_ttm"]),
    "moneyflow": (money, ["net_mf_amount", "buy_elg_amount", "sell_elg_amount"]),
    "cyq_perf": (cyq, ["cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct",
        "cost_95pct", "weight_avg", "winner_rate",
        "benefit_part", "avg_cost", "pct_70_low", "pct_70_high", "pct_70_con",
        "pct_90_low", "pct_90_high", "pct_90_con"]),
    "margin_detail": (margin, ["rzye", "rqye", "rzmre", "rqyl"]),
}
# Aggregate LHB by symbol
if len(lhb):
    lhb = lhb.groupby("symbol").agg({"l_buy": "sum", "l_sell": "sum", "net_amount": "sum"}).reset_index()

for src_name, (src_df, cols) in merges.items():
    if not len(src_df):
        continue
    available = [c for c in cols if c in src_df.columns]
    if not available:
        continue
    smap = src_df.set_index("symbol")[available]
    for c in available:
        tgt = c
        if tgt in panel_cols and tgt not in df.columns:
            df[tgt] = df["symbol"].map(smap[c])

# Direct mapping: Tushare source cols -> panel target cols (names differ)
if len(cyq) and "winner_rate" in cyq.columns and "winner_rate" not in df.columns:
    wmap = cyq.set_index("symbol")["winner_rate"]
    df["winner_rate"] = df["symbol"].map(wmap)

if len(basic):
    bmap = basic.set_index("symbol")
    if "turnover_rate_f" in bmap.columns and "free_float_turnover_rate" in panel_cols:
        df["free_float_turnover_rate"] = df["symbol"].map(bmap["turnover_rate_f"])

# stk_limit direct mapping
if len(limit):
    lmap = limit.set_index("symbol")
    for src_col, tgt_col in [("up_limit", "up_limit_raw"), ("down_limit", "down_limit_raw")]:
        if src_col in lmap.columns and tgt_col in panel_cols and tgt_col not in df.columns:
            df[tgt_col] = df["symbol"].map(lmap[src_col])

# Rename mappings
rename_map = {
    "net_mf_amount": "main_money_flow",
    "winner_rate": "_winner_rate_tmp",

    "rzye": "margin_balance", "rqye": "short_balance",
    "rzmre": "margin_buy_amt", "rqyl": "short_sell_vol",
}
for old, new in rename_map.items():
    if old in df.columns:
        df[new] = df[old]

# free_float_turnover_rate = turnover_rate_f (free float turnover)
if "turnover_rate_f" in df.columns and "free_float_turnover_rate" in panel_cols:
    df["free_float_turnover_rate"] = df["turnover_rate_f"]

# LHB merge
if len(lhb):
    for src, tgt in [("l_buy", "lhb_buy_amt"), ("l_sell", "lhb_sell_amt"), ("net_amount", "lhb_net_buy")]:
        if tgt in panel_cols:
            smap = lhb.set_index("symbol")[src]
            df[tgt] = df["symbol"].map(smap)

# ── 4. Compute derived columns ──
print("\n[4] Computing derived columns...")

# --- hfq prices ---
if len(adj) and "adj_factor" in adj.columns:
    fmap = adj.set_index("symbol")["adj_factor"]
    factor = df["symbol"].map(fmap)
    for col in ["open", "high", "low", "close"]:
        tgt = f"{col}_hfq"
        if tgt in panel_cols:
            df[tgt] = df[col] * factor

# --- winner_ratio = winner_rate / 100 ([0,1] ratio) ---
if "winner_rate" in df.columns and "winner_ratio" in panel_cols:
    df["winner_ratio"] = df["winner_rate"] / 100.0

# --- avg_cost = cost_50pct ---
if "cost_50pct" in df.columns and "avg_cost" in panel_cols:
    df["avg_cost"] = df["cost_50pct"]

# --- pct_90/70 CI ---
# Formula from cyq_calculator.py: concentration = (hi-lo)/(hi+lo)
# Result in [0, 1]: smaller = more concentrated (bullish)
def safe_div(a, b):
    return np.where((b.notna() & (b != 0)), a / b, np.nan)

for lo, hi, llo, lhi, lcon in [
    ("cost_5pct", "cost_95pct", "pct_90_low", "pct_90_high", "pct_90_con"),
    ("cost_15pct", "cost_85pct", "pct_70_low", "pct_70_high", "pct_70_con"),
]:
    if all(c in df.columns or c in panel_cols for c in [llo, lhi, lcon]):
        if lo in df.columns and hi in df.columns:
            df[llo] = df[lo]
            df[lhi] = df[hi]
            df[lcon] = safe_div(df[hi] - df[lo], df[hi] + df[lo])

# --- super_large_order_net ---
if "buy_elg_amount" in df.columns and "sell_elg_amount" in df.columns:
    df["super_large_order_net"] = df["buy_elg_amount"] - df["sell_elg_amount"]

# --- is_suspended ---
if len(susp):
    suspended_set = set(susp["symbol"].unique())
    df["is_suspended"] = df["symbol"].isin(suspended_set).astype(int)

# --- stock_basic: board, is_st, industry, list_days ---
try:
    basic_info = pro.stock_basic(list_status="L", fields="ts_code,symbol,name,industry,list_date")
    basic_info["board"] = basic_info["ts_code"].apply(
        lambda x: "sh" if x.endswith(".SH") else ("sz" if x.endswith(".SZ") else "bj")
    )
    basic_info["is_st"] = basic_info["name"].str.contains(r"\*?ST", na=False).astype(int)
    list_date_ts = pd.to_datetime(basic_info["list_date"], format="%Y%m%d", errors="coerce")
    basic_info["list_days"] = (pd.Timestamp(TRADE_DATE) - list_date_ts).dt.days
    for col in ["board", "is_st", "industry", "list_days"]:
        if col in panel_cols:
            smap = basic_info.set_index("symbol")[col]
            df[col] = df["symbol"].map(smap)
except Exception as e:
    print(f"    stock_basic: {e}")

# --- Sector index (sw_daily) ---
# Map each stock's industry to SW index data
if len(sw) and "industry" in df.columns:
    # Build index_name -> {close, vol, pct_change} mapping
    sw_map = {}
    for _, row in sw.iterrows():
        idx_name = str(row.get("name", "")).strip()
        if idx_name:
            sw_map[idx_name] = {"close": row.get("close"), "vol": row.get("vol"), "pct_change": row.get("pct_change")}
    # Fuzzy match industry -> SW name
    ind_to_sw = {}
    for ind in df["industry"].dropna().unique():
        ind_s = str(ind).strip()
        for sw_n in sw_map:
            if ind_s in sw_n or sw_n in ind_s:
                ind_to_sw[ind_s] = sw_n
                break
    for tgt_col, sw_key in [("sw_index_close","close"),("sw_index_vol","vol"),("sw_ret_1d","pct_change")]:
        if tgt_col in panel_cols:
            vals = {}
            for _, row in df.iterrows():
                sw_n = ind_to_sw.get(str(row.get("industry","")).strip())
                if sw_n and pd.notna(sw_map[sw_n].get(sw_key)):
                    vals[row["symbol"]] = sw_map[sw_n][sw_key]
            if vals:
                df[tgt_col] = df["symbol"].map(vals)
    n_filled = df["sw_index_close"].notna().sum() if "sw_index_close" in df.columns else 0
    print(f"    sw_daily: {len(sw)} indices -> {n_filled}/{len(df)} stocks")

# --- Simple derived features (no history needed) ---
if all(c in df.columns for c in ["high", "low", "pre_close"]):
    df["intraday_range"] = (df["high"] - df["low"]) / df["pre_close"].replace(0, np.nan)

if all(c in df.columns for c in ["close", "pre_close"]):
    df["pctChg"] = (df["close"] / df["pre_close"] - 1) * 100

if all(c in df.columns for c in ["amount", "close"]):
    valid = df["amount"].notna() & df["close"].notna() & (df["close"] > 0)
    if "volume" in panel_cols and "volume" not in df.columns:
        df["volume"] = np.nan
    df.loc[valid, "volume"] = df.loc[valid, "amount"] / df.loc[valid, "close"]

if "pct_chg" in df.columns and "ret_pct" in panel_cols:
    df["ret_pct"] = df["pct_chg"] / 100

# ── 5. Read history for rolling features ──
print("\n[5] Computing rolling features from history...")
symbols = sorted(df["symbol"].unique())

# Read only needed columns from panel
hist_cols = ["symbol", "date", "close_hfq", "volume", "high", "low", "pre_close", "open", "amount"]
hist_cols = [c for c in hist_cols if c in panel_cols]
hist = pq.read_table(PANEL, columns=hist_cols).to_pandas()
hist = hist[hist["symbol"].isin(symbols)]
hist = hist[hist["date"] <= pd.Timestamp(TRADE_DATE)]
hist = hist.sort_values(["symbol", "date"])
print(f"    History: {len(hist):,} rows for {hist.symbol.nunique()} symbols")

close_today = df.set_index("symbol")["close_hfq"]

# bias features
bias_windows = {"bias_5": 5, "bias_10": 10, "bias_20": 20, "bias_60": 60, "bias_120": 120, "bias_250": 250}
for col_name, window in bias_windows.items():
    if col_name not in panel_cols:
        continue
    ma = hist.groupby("symbol")["close_hfq"].rolling(window, min_periods=window).mean()
    ma.index = ma.index.droplevel(0)
    last_ma = ma.groupby(hist["symbol"]).last()
    vals = {}
    for sym, m in last_ma.dropna().items():
        if sym in close_today.index and pd.notna(close_today[sym]) and m > 0:
            vals[sym] = close_today[sym] / m - 1
    df[col_name] = df["symbol"].map(vals)
    n_filled = df[col_name].notna().sum()
    if n_filled < len(df):
        print(f"    {col_name}: {n_filled}/{len(df)}")

# bias cross (set to 0 — no diff history for single day)
for cross_col in ["bias_5_20_cross", "bias_20_60_cross"]:
    if cross_col in panel_cols:
        df[cross_col] = 0.0

# ma_vol_ratio_5_20
if "ma_vol_ratio_5_20" in panel_cols and "volume" in hist.columns:
    g = hist.groupby("symbol")["volume"]
    ma5 = g.rolling(5, min_periods=3).mean()
    ma5.index = ma5.index.droplevel(0)
    ma20 = g.rolling(20, min_periods=10).mean()
    ma20.index = ma20.index.droplevel(0)
    l5, l20 = ma5.groupby(hist["symbol"]).last(), ma20.groupby(hist["symbol"]).last()
    vals = {}
    for sym in l5.index.intersection(l20.index):
        if pd.notna(l20[sym]) and l20[sym] > 0:
            vals[sym] = l5[sym] / l20[sym]
    df["ma_vol_ratio_5_20"] = df["symbol"].map(vals)

# amplitude_5d
if "amplitude_5d" in panel_cols and all(c in hist.columns for c in ["high", "low", "pre_close"]):
    denom = hist["pre_close"].replace(0, np.nan)
    hist["_amp"] = (hist["high"] - hist["low"]) / denom
    ma5 = hist.groupby("symbol")["_amp"].rolling(5, min_periods=3).mean()
    ma5.index = ma5.index.droplevel(0)
    df["amplitude_5d"] = df["symbol"].map(ma5.groupby(hist["symbol"]).last())

# vol_surge, amt_surge
for col, surge_col in [("volume", "vol_surge"), ("amount", "amt_surge")]:
    if surge_col in panel_cols and col in hist.columns:
        g = hist.groupby("symbol")[col]
        ma20 = g.rolling(20, min_periods=10).mean()
        ma20.index = ma20.index.droplevel(0)
        std20 = g.rolling(20, min_periods=10).std()
        std20.index = std20.index.droplevel(0)
        last_m = ma20.groupby(hist["symbol"]).last()
        last_s = std20.groupby(hist["symbol"]).last()
        today_vals = df.set_index("symbol")[col]
        vals = {}
        for sym in last_m.index.intersection(last_s.index):
            if (sym in today_vals.index and pd.notna(today_vals[sym])
                and pd.notna(last_s[sym]) and last_s[sym] > 0):
                vals[sym] = (today_vals[sym] - last_m[sym]) / last_s[sym]
        df[surge_col] = df["symbol"].map(vals)

# ── 6. Forward-fill quarterly & slow data ──
print("\n[6] Forward-filling quarterly/slow data...")
ffill_cols = [
    "roe", "roa", "gross_margin", "net_margin", "eps_yoy", "rev_yoy", "profit_yoy",
    "debt_ratio", "current_ratio", "asset_turnover", "inventory_turnover",
    "ocf_to_or", "eps", "bps", "ocfps", "revenue_ps",
    "roe_deducted", "roe_yoy", "q_roe",
    "ar_turnover", "profit_ratio",
    "holder_count", "sh_change_vol", "sh_change_amt", "sh_change_amt_total",
    "sh_net_change_sign", "sh_net_sign",
    "sw_index_close", "sw_index_vol", "sw_ret_1d",
    "margin_balance", "short_balance", "margin_buy_amt", "short_sell_vol",
    "dt_eps", "q_ocf_to_sales",
    "announce_date",
]
ffill_cols = [c for c in ffill_cols if c in panel_cols]
needed_ffill = [c for c in ffill_cols if c not in df.columns or df[c].isna().all()]

if needed_ffill:
    ffill_hist = pq.read_table(PANEL, columns=["symbol", "date"] + needed_ffill).to_pandas()
    ffill_hist = ffill_hist[ffill_hist["symbol"].isin(symbols)]
    ffill_hist = ffill_hist[ffill_hist["date"] < pd.Timestamp(TRADE_DATE)]
    ffill_hist = ffill_hist.sort_values(["symbol", "date"])
    for col in needed_ffill:
        if col in ffill_hist.columns:
            ffill_hist[col] = ffill_hist.groupby("symbol")[col].ffill()
    last_per_stock = ffill_hist.groupby("symbol").last()
    filled_count = 0
    for col in needed_ffill:
        if col in last_per_stock.columns:
            smap = last_per_stock[col].dropna()
            df[col] = df["symbol"].map(smap)
            if df[col].notna().sum() > 0:
                filled_count += 1
    print(f"    Forward-filled: {filled_count} columns")

# ── 7. Align to panel schema ──
print("\n[7] Aligning to panel schema...")
for c in panel_cols:
    if c not in df.columns:
        df[c] = pd.NA
df = df[list(panel_cols)]

# Cast types
for field in schema:
    if field.name in df.columns:
        try:
            df[field.name] = df[field.name].astype(field.type.to_pandas_dtype())
        except Exception:
            pass

# ── 8. Append to panel (streaming) ──
print("\n[8] Appending to panel...")
today_table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

pf = pq.ParquetFile(PANEL)
tmp_path = PANEL + ".tmp"
writer = pq.ParquetWriter(tmp_path, schema=schema)

for rg_idx in range(pf.metadata.num_row_groups):
    rg = pf.read_row_group(rg_idx)
    rg_df = rg.to_pandas()
    # Remove today's old rows (idempotent: re-run after crash rebuilds cleanly)
    rg_df = rg_df[rg_df["date"] != pd.Timestamp(TRADE_DATE)]
    if len(rg_df):
        writer.write_table(pa.Table.from_pandas(rg_df, schema=schema, preserve_index=False))

writer.write_table(today_table)
writer.close()
pf.close()

# Atomic replace
if os.path.exists(PANEL):
    os.remove(PANEL)
os.rename(tmp_path, PANEL)

# ── 9. Audit ──
print(f"\n{'='*55}")
print(f"DONE: {TRADE_DATE}")
print(f"{'='*55}")
pf2 = pq.ParquetFile(PANEL)
print(f"Panel: {pf2.metadata.num_rows:,} rows, {len(pf2.schema_arrow.names)} cols")
pf2.close()

final = pq.read_table(PANEL, filters=[("date", "=", pd.Timestamp(TRADE_DATE))]).to_pandas()
n_cols = len(final.columns)
filled = sum(final.notna().sum() > 0)
empty = n_cols - filled
print(f"Filled: {filled}/{n_cols} ({filled/n_cols:.1%})")
empty_cols = sorted([c for c in final.columns if final[c].notna().sum() == 0])
if empty_cols:
    print(f"Empty ({len(empty_cols)}): {', '.join(empty_cols)}")
else:
    print("All columns have data!")
