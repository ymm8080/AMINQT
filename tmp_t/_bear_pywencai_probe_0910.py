# -*- coding: utf-8 -*-
"""探针2: pywencai.get 自带端点 perpage 大页是否生效 (样例日 20240305)."""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
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
df = pywencai.get(query="20240305看跌信号的股票",
                  cookie=ck, log=True, loop=True, perpage=100, sleep=1)
if df is None:
    print("pywencai.get -> None")
    sys.exit(2)
print(f"rows={len(df)}")
print("columns:", list(df.columns)[:25])
print(df.head(3).to_string())
