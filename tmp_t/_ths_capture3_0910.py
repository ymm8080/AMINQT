# -*- coding: utf-8 -*-
"""裸HTTP诊断v2: 用pywencai自带get_token, 打印问财真实回包"""
import os
import sys

os.environ["NODE_OPTIONS"] = "--no-deprecation"
sys.stdout.reconfigure(encoding="utf-8")
import requests
from pywencai.headers import get_token, headers

token = get_token()
print(f"token len={len(token)}, head={token[:20]}...")

url = "https://www.iwencai.com/customized/chart/get-robot-data"
hd = dict(headers())
hd["Cookie"] = f"v={token}"
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
r = requests.post(url, data=payload, headers=hd, timeout=30)
print(f"status={r.status_code}, len={len(r.text)}")
print(f"content-type={r.headers.get('content-type')}")
print("body[:800]:", r.text[:800].replace("\n", " "))
