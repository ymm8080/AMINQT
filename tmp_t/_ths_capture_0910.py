# -*- coding: utf-8 -*-
"""用 pywencai 抓同花顺看涨/看跌信号 探针 (一次性; 输出落盘)"""
import io
import json
import os
import sys

os.environ["NODE_OPTIONS"] = "--no-deprecation"
sys.stdout.reconfigure(encoding="utf-8")
import pywencai

OUT = r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_capture_0910_result.txt"
L = []

def P(s=""):
    print(s)
    L.append(str(s))

QUERIES = [
    "同花顺看涨信号",
    "智能诊股看涨",
    "诊股看涨信号",
    "看跌信号",
]
for q in QUERIES:
    try:
        df = pywencai.get(query=q, query_type="stock", loop=True)
        if df is None:
            P(f"\n== {q!r} -> None")
            continue
        P(f"\n== {q!r} -> {len(df)} 行")
        P(f"列: {list(df.columns)}")
        P(df.head(8).to_string())
    except Exception as e:
        P(f"\n== {q!r} -> FAIL: {type(e).__name__}: {str(e)[:400]}")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))
print(f"\nsaved -> {OUT}")
