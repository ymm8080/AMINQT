# -*- coding: utf-8 -*-
"""从问财主页+选股页HTML挖真实登录入口"""
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
import requests
import urllib3.util.connection as u3c

IP = "111.4.248.121"
_orig = u3c.create_connection


def fixed(address, *a, **kw):
    host, port = address
    if host == "www.iwencai.com":
        address = (IP, port)
    return _orig(address, *a, **kw)


u3c.create_connection = fixed

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36"}

for path in ["/", "/screener"]:
    try:
        r = requests.get(f"https://www.iwencai.com{path}", headers=UA, timeout=15)
        html = r.text
        print(f"\n===== {path} status={r.status_code} len={len(html)}")
        pats = [r'https?://[^"\'\s<>]*pass\.10jqka[^"\'\s<>]*',
                r'https?://[^"\'\s<>]*login[^"\'\s<>]*',
                r'https?://[^"\'\s<>]*passport[^"\'\s<>]*',
                r'https?://[^"\'\s<>]*sso[^"\'\s<>]*']
        seen = set()
        for pat in pats:
            for m in re.findall(pat, html, re.I):
                if m not in seen:
                    seen.add(m)
                    print(f"  {m[:130]}")
    except Exception as e:
        print(f"{path} FAIL {type(e).__name__}: {str(e)[:120]}")
