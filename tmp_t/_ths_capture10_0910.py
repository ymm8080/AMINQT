# -*- coding: utf-8 -*-
"""capture10: 403收敛测试 - 会话cookie/https/http/三端点 全组合"""
import os
import sys

os.environ["NODE_OPTIONS"] = "--no-deprecation"
sys.stdout.reconfigure(encoding="utf-8")
import time

import requests
import urllib3.util.connection as u3c

IP_IWENCAI = "121.12.127.73"
_orig_cc = u3c.create_connection


def fixed_cc(address, *a, **kw):
    host, port = address
    if host == "www.iwencai.com":
        address = (IP_IWENCAI, port)
    return _orig_cc(address, *a, **kw)


u3c.create_connection = fixed_cc

from pywencai.headers import headers as mk_headers

hd = mk_headers()
UA = hd["User-Agent"]

question_body = {
    'add_info': '{"urp":{"scene":1,"company":1,"business":1},"contentType":"json","searchInfo":true}',
    'perpage': '10', 'page': 1, 'source': 'Ths_iwencai_Xuangu',
    'log_info': '{"input_type":"click"}', 'version': '2.0',
    'secondary_intent': 'stock', 'question': 'kdj金叉',
}

s = requests.Session()
s.headers.update({"User-Agent": UA})

print("== 预热: GET主页集cookie")
try:
    r0 = s.get("https://www.iwencai.com/", timeout=15)
    print(f"   status={r0.status_code}, cookies={dict(s.cookies)}")
except Exception as e:
    print(f"   FAIL {type(e).__name__}: {str(e)[:120]}")
time.sleep(2)

# 补上hexin-v双写: header + cookie
s.headers.update({"hexin-v": hd["hexin-v"]})
s.cookies.set("v", hd["hexin-v"], domain="www.iwencai.com")


def probe(tag, url, json_body=None, form=None):
    try:
        r = s.post(url, json=json_body, data=form, timeout=20)
        print(f"[{tag}] {r.status_code}, len={len(r.text)}: {r.text[:260]}")
    except Exception as e:
        print(f"[{tag}] FAIL {type(e).__name__}: {str(e)[:150]}")
    time.sleep(2)


print("\n== 组合探测")
probe("A http+json+会话", "http://www.iwencai.com/customized/chart/get-robot-data", json_body=question_body)
probe("B https+json+会话", "https://www.iwencai.com/customized/chart/get-robot-data", json_body=question_body)

wap = {
    **question_body,
    "perpage": 100, "query_type": "stock",
}
probe("C wap find端点", "http://www.iwencai.com/unifiedwap/unified-wap/v2/stock-pick/find",
      form={"question": "kdj金叉", "query_type": "stock", "perpage": 10, "page": 1})

probe("D urp v7 getDataList", "http://www.iwencai.com/gateway/urp/v7/landing/getDataList",
      form={**question_body, "perpage": 100})
