# -*- coding: utf-8 -*-
# 查 002881 今日派发闸口径 wr5 (winner_ratio 5日变化); 只读不写交付
import pandas as pd

df = pd.read_parquet("data/cyq_panel.parquet", filters=[("symbol", "=", "002881")])
df = df[["date", "winner_ratio"]].sort_values("date").tail(12)
print(df.to_string(index=False))
print("\npanel max date:", df["date"].max())

s = df.set_index("date")["winner_ratio"].astype(float)
if len(s) >= 6:
    wr5 = s.iloc[-1] - s.iloc[-6]
    print(f"\nwr5 (latest vs -5td): {wr5:+.4f}  ->  {'触发派发闸(删)' if wr5 < 0 else '不触发(保留)'}")
    for k in (1, 2, 3, 4, 5):
        print(f"  wr{k}: {s.iloc[-1] - s.iloc[-1-k]:+.4f}")
