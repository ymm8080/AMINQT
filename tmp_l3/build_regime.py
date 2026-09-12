# -*- coding: utf-8 -*-
"""L3 sentiment-cycle regime columns (market-level daily aggregates).

Reads panel via pyarrow (only needed cols, date-prefiltered to bound RAM),
builds per-day regime table:
  limit_up_count, blow_rate, max_board_height, lu_premium_1d, limit_up_count_ma5
Output: tmp_l3/regime_daily.csv  (WORM: dated copy also saved)
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
OUT = r"D:\AMINQT\AMINQT CODES\tmp_l3\regime_daily.csv"
CUT = pd.Timestamp("2024-06-01")  # warmup margin before eval window (last 500 td)

tb = pq.read_table(PANEL, columns=["symbol", "date", "close", "high", "pre_close"],
                   filters=[("date", ">=", CUT)])
df = tb.to_pandas()
df["symbol"] = df["symbol"].astype("category")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

# daily return & limit thresholds: 300xxx (ChiNext) / 688xxx (STAR) 20% band -> 19.8%; else 10% -> 9.8%
sym = df["symbol"].astype(str)
thr = np.where(sym.str.startswith(("300", "301", "688", "689")), 0.198, 0.098)
pc = df["pre_close"].to_numpy(dtype="float64")
ret = df["close"].to_numpy(dtype="float64") / pc - 1.0
hi_p = df["high"].to_numpy(dtype="float64") / pc - 1.0
valid = np.isfinite(pc) & (pc > 0) & np.isfinite(ret)
ret = np.where(valid, ret, 0.0)
hi_p = np.where(valid & np.isfinite(hi_p), hi_p, -1.0)

is_lu = ret >= thr          # sealed limit-up at close
touched = hi_p >= thr       # touched limit at intraday high
df["ret"] = ret
df["is_lu"] = is_lu
df["touched"] = touched

# consecutive limit-up streak (vectorized block-cumcount)
blk = (df["is_lu"] != df.groupby("symbol", observed=True)["is_lu"].shift(1))
blk_id = blk.groupby(df["symbol"], observed=True).cumsum()
streak = df.groupby(["symbol", blk_id], observed=True).cumcount() + 1
df["streak"] = streak.where(df["is_lu"], 0).astype("int16")

g = df.groupby("date", observed=True)
reg = pd.DataFrame({
    "n_stocks": g["ret"].count(),
    "mkt_ret": g["ret"].mean(),
    "limit_up_count": g["is_lu"].sum().astype("int32"),
    "touched_count": g["touched"].sum().astype("int32"),
    "max_board_height": g["streak"].max().astype("int16"),
})
reg["blow_count"] = reg["touched_count"] - reg["limit_up_count"]
reg["blow_rate"] = np.where(reg["touched_count"] > 0,
                            reg["blow_count"] / reg["touched_count"], np.nan)

# lu_premium_1d: mean(ret at t+1 | limit-up at t) - mean(ret at t+1 | all)
# join yesterday's limit-up flag onto today's row, per symbol
tdays = np.sort(df["date"].unique())
next_td_map = pd.Series(np.append(tdays[1:], pd.NaT), index=tdays)
df["date_next"] = df["date"].map(next_td_map)
lu_keys = df.loc[df["is_lu"], ["symbol", "date_next"]].rename(columns={"date_next": "date"})
lu_keys["date"] = pd.to_datetime(lu_keys["date"])
lu_keys["was_lu"] = True
df = df.merge(lu_keys, on=["symbol", "date"], how="left")
prem = df[df["was_lu"] == True].groupby("date", observed=True)["ret"].agg(["count", "mean"])
reg["lu_premium_1d"] = prem["mean"].reindex(reg.index) - reg["mkt_ret"]
reg["lu_cont_n"] = prem["count"].reindex(reg.index).fillna(0).astype("int32")

reg["limit_up_count_ma5"] = reg["limit_up_count"].rolling(5).mean()

reg = reg.reset_index()
reg["date"] = reg["date"].dt.strftime("%Y-%m-%d")
reg.to_csv(OUT, index=False)
print(reg.describe().round(4).to_string())
print("\nsaved:", OUT, "rows:", len(reg))
print(reg.tail(3).to_string())
