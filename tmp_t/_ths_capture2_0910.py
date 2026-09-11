# -*- coding: utf-8 -*-
"""通道验证: 已知问财能答的查询 vs THS徽章类说法"""
import os
import sys

os.environ["NODE_OPTIONS"] = "--no-deprecation"
sys.stdout.reconfigure(encoding="utf-8")
import pywencai

OUT = r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_capture2_0910_result.txt"
L = []

def P(s=""):
    print(s)
    L.append(str(s))

QUERIES = [
    ("已知好查询", "kdj金叉"),
    ("已知好查询", "连续3天上涨"),
    ("技术等价", "kdj金叉 且 收盘价大于10日均线 且 收阳"),
    ("THS徽章", "看涨评级"),
    ("THS徽章", "同花顺诊股评级看涨"),
]
for tag, q in QUERIES:
    try:
        df = pywencai.get(query=q, query_type="stock", loop=True)
        if df is None:
            P(f"\n== [{tag}] {q!r} -> None")
            continue
        P(f"\n== [{tag}] {q!r} -> {len(df)} 行")
        P(f"列: {list(df.columns)[:15]}")
        P(df.head(3).to_string())
    except Exception as e:
        P(f"\n== [{tag}] {q!r} -> FAIL: {type(e).__name__}: {str(e)[:400]}")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))
print(f"\nsaved -> {OUT}")
