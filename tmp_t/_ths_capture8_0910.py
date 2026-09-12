# -*- coding: utf-8 -*-
"""硬编码大陆CDN IP裸探: 问财API真实回包 + eq诊股域 (绕过geo-DNS漂移)"""
import os
import sys

os.environ["NODE_OPTIONS"] = "--no-deprecation"
sys.stdout.reconfigure(encoding="utf-8")
import time

import requests
from pywencai.headers import get_token, headers

# 大陆CDN IP (今天三次TLS握手验证过), SNI/Host仍用域名
IP_IWENCAI = "121.12.127.73"    # www.iwencai.com / pass / basic / stock 共用池
IP_EQ = "111.4.248.121"         # eq / news / comment / data 池

import requests.adapters
from urllib3.util.connection import create_connection


class FixedResolverAdapter(requests.adapters.HTTPAdapter):
    def __init__(self, ip, **kw):
        self.ip = ip
        super().__init__(**kw)

    def get_connection(self, host, port=None, **kw):
        # urllib3 2.x: host是元组(host,port)? 兼容处理
        if port is None and isinstance(host, tuple):
            host, port = host
        conn = super().get_connection(host, port, **kw) if hasattr(super(), "get_connection") else None
        return conn


# 简化: 直接monkey-patch urllib3的create_connection
_IP_MAP = {"www.iwencai.com": IP_IWENCAI, "eq.10jqka.com.cn": IP_EQ}
_orig_cc = create_connection


def fixed_create_connection(address, *a, **kw):
    host, port = address
    if host in _IP_MAP:
        address = (_IP_MAP[host], port)
    return _orig_cc(address, *a, **kw)


import urllib3.util.connection as u3c

u3c.create_connection = fixed_create_connection
print(f"[fix] iwencai->{IP_IWENCAI}, eq->{IP_EQ}")

token = get_token()
hd = dict(headers())
hd["Cookie"] = f"v={token}"
print(f"[token] len={len(token)}")

payload = {
    "question": "kdj金叉",
    "perpage": 10, "page": 1, "secondary_intent": "stock",
    "log_info": '{"input_type":"typewrite"}', "source": "Ths_iwencai_Xuangu",
    "version": "2.0", "query_area": "", "block_list": "",
    "add_info": '{"urp":{"scene":1,"company":1,"business":1},"contentType":"json","searchInfo":true}',
}

s = requests.Session()
s.headers.update(hd)

print("\n== GET 问财主页 (经121.12.127.73)")
try:
    r = s.get("https://www.iwencai.com/", timeout=15,
              headers={"User-Agent": hd.get("User-Agent", "Mozilla/5.0")})
    print(f"   status={r.status_code}, len={len(r.text)}, server={r.headers.get('server')}")
except Exception as e:
    print(f"   FAIL {type(e).__name__}: {str(e)[:150]}")
time.sleep(2)

print("\n== POST get-robot-data (kdj金叉)")
try:
    r = s.post("https://www.iwencai.com/customized/chart/get-robot-data",
               data=payload, timeout=25)
    print(f"   status={r.status_code}, len={len(r.text)}, ct={r.headers.get('content-type')}")
    print(f"   body[:900]: {r.text[:900]}")
except Exception as e:
    print(f"   FAIL {type(e).__name__}: {str(e)[:200]}")
time.sleep(2)

print("\n== GET eq诊股域根路径 (经111.4.248.121)")
try:
    r2 = requests.get("https://eq.10jqka.com.cn/", timeout=15,
                      headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0.0.0"})
    print(f"   status={r2.status_code}, len={len(r2.text)}")
    print(f"   body[:400]: {r2.text[:400]!r}")
except Exception as e:
    print(f"   FAIL {type(e).__name__}: {str(e)[:200]}")
