# -*- coding: utf-8 -*-
"""WAF解封探针: 解封=exit 0, 仍封=exit 1 (供until-loop轮询)"""
import sys

import requests
import urllib3.util.connection as u3c

_orig = u3c.create_connection


def fixed(address, *a, **kw):
    host, port = address
    if host == "www.iwencai.com":
        address = ("121.12.127.73", port)
    return _orig(address, *a, **kw)


u3c.create_connection = fixed

import pywencai  # noqa: E402

cookies = json = None
import json as _json  # noqa: E402

cookies = _json.load(open(r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_cookies_logged_0910.json",
                          encoding="utf-8"))
ck = {c["name"]: c["value"] for c in cookies
      if ("iwencai" in c["domain"] or "10jqka" in c["domain"])}
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36"
hexin = pywencai.headers.headers()["hexin-v"]
params = {"question": "20240205看涨信号的股票", "secondary_intent": "stock",
          "perpage": 10, "page": 1, "block_list": "", "add_info": "1",
          "r": "0.517", "source": "Ths_iwencai_Xuangu", "version": "2.0",
          "query_type": "stock", "hexin-v": hexin}
headers = {"User-Agent": UA, "hexin-v": hexin, "Content-Type": "application/json",
           "Referer": "https://www.iwencai.com/unifiedwap/result/get-robot-data",
           "Origin": "https://www.iwencai.com"}
try:
    resp = requests.post("http://www.iwencai.com/customized/chart/get-robot-data",
                         params=params, json={}, headers=headers, cookies=ck, timeout=25)
    if resp.status_code == 200 and resp.text.lstrip().startswith("{"):
        print("UNBLOCKED")
        sys.exit(0)
    print(f"still blocked: HTTP {resp.status_code} {resp.text[:60]!r}")
    sys.exit(1)
except Exception as e:
    print(f"probe error: {type(e).__name__} {str(e)[:60]}")
    sys.exit(1)
