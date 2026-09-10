# -*- coding: utf-8 -*-
# 直拉 Tushare cyq_perf 002881 最新数据, 口径对齐 cyq_panel.winner_ratio 后算 wr5
import tushare as ts
import pandas as pd

ts.set_token(ts.get_token())
pro = ts.pro_api()

df = pro.cyq_perf(ts_code="002881.SZ", start_date="20260828", end_date="20260909")
df = df[["trade_date", "winner_rate"]].sort_values("trade_date")
print(df.to_string(index=False))

wr = df.set_index("trade_date")["winner_rate"].astype(float)
if "20260909" in wr.index:
    base = wr.iloc[wr.index.get_loc("20260909") - 5]
    wr5 = wr["20260909"] - base
    print(f"\nbase(-5td)={wr.index[wr.index.get_loc('20260909') - 5]} winner_rate={base:.2f}")
    print(f"wr5 = {wr['20260909']:.2f} - {base:.2f} = {wr5:+.2f}pp -> "
          f"{'触发派发闸(删)' if wr5 < 0 else '不触发(保留)'}")
else:
    print("\n20260909 行未发布")
