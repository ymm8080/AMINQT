# -*- coding: utf-8 -*-
# 复算 002881 今晚派发闸口径 wr5: 主面板尾部 + 09-09 bar (缺则 Tushare 补) → compute_cyq_panel
import sys

sys.path.insert(0, ".")
import pandas as pd
import tushare as ts

from app.pipeline1.cyq_calculator import RANGE_DAYS, compute_cyq_panel

SYMBOL = "002881"
PANEL = "D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"

need_cols = ["symbol", "date", "open", "high", "low", "close", "turnover_rate"]
df = pd.read_parquet(PANEL, columns=need_cols, filters=[("symbol", "=", SYMBOL)])
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").tail(RANGE_DAYS + 15).reset_index(drop=True)
print("panel max date:", df["date"].max().date())

if df["date"].max() < pd.Timestamp("2026-09-09"):
    pro = ts.pro_api()
    bar = pro.daily(ts_code=f"{SYMBOL}.SZ", trade_date="20260909")
    b = pro.daily_basic(ts_code=f"{SYMBOL}.SZ", trade_date="20260909")
    bar = bar.merge(b[["ts_code", "turnover_rate"]], on="ts_code")
    row = pd.DataFrame(
        {
            "symbol": SYMBOL,
            "date": pd.to_datetime(["2026-09-09"]),
            "open": bar["open"].astype(float),
            "high": bar["high"].astype(float),
            "low": bar["low"].astype(float),
            "close": bar["close"].astype(float),
            "turnover_rate": bar["turnover_rate"].astype(float),
        }
    )
    df = pd.concat([df, row], ignore_index=True).sort_values("date").tail(
        RANGE_DAYS + 15
    ).reset_index(drop=True)
    print("appended 0909 bar:", row[["close", "turnover_rate"]].to_dict("records"))

cyq = compute_cyq_panel(df[need_cols])
cyq = cyq.sort_values("date").tail(8)
print(cyq[["date", "winner_ratio"]].to_string(index=False))

wr = cyq.set_index("date")["winner_ratio"].astype(float)
wr5 = wr.iloc[-1] - wr.iloc[-6]
print(f"\nwr5 = {wr.iloc[-1]:.4f} - {wr.iloc[-6]:.4f} = {wr5:+.4f} -> "
      f"{'触发派发闸(删)' if wr5 < 0 else '不触发(保留)'}")
