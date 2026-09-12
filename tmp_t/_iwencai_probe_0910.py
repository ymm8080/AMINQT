# -*- coding: utf-8 -*-
"""问财能否表达THS看涨类信号 探针 (一次性, 不入库)"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:/AMINQT/AMINQT CODES")
from app.models.iwencai_agent import IwencaiAgent

QUERIES = [
    "同花顺看涨信号",
    "个股看涨信号",
    "kdj金叉 且 站上10日均线 且 收阳",
    "智能诊股看涨",
]
ag = IwencaiAgent()
for q in QUERIES:
    try:
        rows = ag.query(q, top_n=10)
        print(f"\n== {q!r} -> {len(rows)} 行")
        for r in rows[:5]:
            print("   ", {k: r.get(k) for k in list(r)[:6]})
    except Exception as e:
        print(f"\n== {q!r} -> FAIL: {type(e).__name__}: {e}")
