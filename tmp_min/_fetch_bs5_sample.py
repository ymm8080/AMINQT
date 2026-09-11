# -*- coding: utf-8 -*-
"""bs5 小样本拉取 (10股×2周) — L2 构建器开发验证用, 不与全量 ingest 竞争."""
import os
import sys

import baostock as bs
import pandas as pd

OUT = r"D:\AMINQT\AMINQT CODES\tmp_min\_bs5_sample"
os.makedirs(OUT, exist_ok=True)
SYMS = ["sh.600000", "sz.000001", "sz.000626", "sz.300765", "sz.300911",
        "sz.002881", "sh.688111", "sz.300865", "sh.601318", "sz.000858"]
START, END = "2026-08-24", "2026-09-08"
FIELDS = "date,time,code,open,high,low,close,volume,amount"

lg = bs.login()
print("login:", lg.error_code, lg.error_msg)
tot = 0
for s in SYMS:
    rs = bs.query_history_k_data_plus(
        code=s, start_date=START, end_date=END,
        frequency="5", adjustflag="3", fields=FIELDS)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    df.to_parquet(os.path.join(OUT, f"bs5_{s.split('.')[1]}.parquet"), index=False)
    tot += len(df)
    print(f"{s}: {len(df)} bars")
bs.logout()
print(f"SAMPLE_FETCH_DONE rows={tot}")
