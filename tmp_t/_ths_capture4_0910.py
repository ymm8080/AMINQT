# -*- coding: utf-8 -*-
"""curl_cffi 模拟Chrome TLS指纹 测问财连通性"""
import os
import sys

os.environ["NODE_OPTIONS"] = "--no-deprecation"
sys.stdout.reconfigure(encoding="utf-8")
from curl_cffi import requests as creq

url = "https://www.iwencai.com/customized/chart/get-robot-data"
payload = {
    "question": "kdj金叉",
    "perpage": 10,
    "page": 1,
    "secondary_intent": "stock",
    "log_info": '{"input_type":"typewrite"}',
    "source": "Ths_iwencai_Xuangu",
    "version": "2.0",
    "query_area": "",
    "block_list": "",
    "add_info": '{"urp":{"scene":1,"company":1,"business":1},"contentType":"json","searchInfo":true}',
}
for imp in ("chrome124", "chrome131", "edge101"):
    try:
        r = creq.post(url, data=payload, impersonate=imp, timeout=30)
        print(f"[{imp}] status={r.status_code} len={len(r.text)}")
        print("   body[:300]:", r.text[:300].replace("\n", " "))
        break
    except Exception as e:
        print(f"[{imp}] FAIL: {type(e).__name__}: {str(e)[:200]}")
