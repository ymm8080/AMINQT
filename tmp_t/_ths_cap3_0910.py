# -*- coding: utf-8 -*-
"""cap3: 裸POST + 登录cookie, 看看信号查询的真实响应体"""
import json
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
import requests
import urllib3.util.connection as u3c

_orig = u3c.create_connection


def fixed(address, *a, **kw):
    host, port = address
    if host == "www.iwencai.com":
        address = ("121.12.127.73", port)
    return _orig(address, *a, **kw)


u3c.create_connection = fixed

import pywencai

cookies = json.load(open(r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_cookies_logged_0910.json", encoding="utf-8"))
ck = {c["name"]: c["value"] for c in cookies if ("iwencai" in c["domain"] or "10jqka" in c["domain"])}

hexin = pywencai.headers.headers()["hexin-v"]

QUERY = "同花顺看涨信号"
body = {
    "question": QUERY,
    "secondary_intent": "stock",
    "perpage": 100,
    "page": 1,
    "block_list": "",
    "add_info": "1",
    "r": "0.123456",
    "source": "Ths_iwencai_Xuangu",
    "version": "2.0",
    "query_type": "stock",
    "hexin-v": hexin,
}
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
headers = {
    "User-Agent": UA,
    "hexin-v": hexin,
    "Content-Type": "application/json",
    "Referer": "https://www.iwencai.com/unifiedwap/result/get-robot-data",
    "Origin": "https://www.iwencai.com",
}
r = requests.post("http://www.iwencai.com/customized/chart/get-robot-data",
                  params=body, json={}, headers=headers, cookies=ck, timeout=20)
print(f"param-style: {r.status_code} len={len(r.text)}")
print(r.text[:600])
