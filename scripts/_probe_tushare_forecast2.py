# -*- coding: utf-8 -*-
"""forecast 接口第二次探针: ann_date 口径."""
import sys

import tushare as ts

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from config.settings import TUSHARE_TOKEN

pro = ts.pro_api(TUSHARE_TOKEN)

FIELDS = "ts_code,ann_date,end_date,type,p_change_min,p_change_max,net_profit_min,net_profit_max"

for label, kw in [
    ("single_day_0715", dict(ann_date="20260715", fields=FIELDS)),
    ("range_july", dict(ann_date="", start_date="20260701", end_date="20260731", fields=FIELDS)),
    ("single_2024_day", dict(ann_date="20240715", fields=FIELDS)),
]:
    try:
        df = pro.forecast(**kw)
        print(f"[{label}] rows={len(df)}")
        if len(df):
            print(df.head(3).to_string())
            if "type" in df:
                print("type 分布:", df["type"].value_counts().to_dict())
    except Exception as e:
        print(f"[{label}] FAIL: {e}")
