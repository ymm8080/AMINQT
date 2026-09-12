# -*- coding: utf-8 -*-
"""HTTP80直打问财API (最后一搏本机出口)"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
import requests

url = "http://www.iwencai.com/customized/chart/get-robot-data"
payload = {
    "question": "kdj金叉",
    "perpage": 10, "page": 1, "secondary_intent": "stock",
    "log_info": '{"input_type":"typewrite"}', "source": "Ths_iwencai_Xuangu",
    "version": "2.0", "query_area": "", "block_list": "",
    "add_info": '{"urp":{"scene":1,"company":1,"business":1},"contentType":"json","searchInfo":true}',
}
hd = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0.0.0",
      "Content-Type": "application/x-www-form-urlencoded",
      "Referer": "http://www.iwencai.com/"}
try:
    r = requests.post(url, data=payload, headers=hd, timeout=20, allow_redirects=False)
    print(f"POST80: status={r.status_code}, loc={r.headers.get('location')}, len={len(r.text)}")
    print("body[:300]:", r.text[:300].replace("\n", " "))
except Exception as e:
    print(f"POST80 FAIL: {type(e).__name__}: {str(e)[:200]}")
