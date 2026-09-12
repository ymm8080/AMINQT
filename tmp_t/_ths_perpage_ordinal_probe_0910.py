# -*- coding: utf-8 -*-
"""探针B: ①perpage阶梯 (10/20/50/100) 看单页上限; ②序数切片 '第11名到第20名' 能否取到 p1 之外的股票.
目标日 20230904 (找到N=58). 预算5请求."""
import io
import json
import re
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import urllib3.util.connection as u3c

_orig = u3c.create_connection


def fixed(address, *a, **kw):
    host, port = address
    if host == "www.iwencai.com":
        address = ("121.12.127.73", port)
    return _orig(address, *a, **kw)


u3c.create_connection = fixed
import requests
import pywencai

cookies = json.load(open(r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_cookies_logged_0910.json", encoding="utf-8"))
ck = {c["name"]: c["value"] for c in cookies if ("iwencai" in c["domain"] or "10jqka" in c["domain"])}
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36"


def q(question, perpage=10):
    headers = {"User-Agent": UA, "hexin-v": pywencai.headers.headers()["hexin-v"],
               "Content-Type": "application/json",
               "Referer": "https://www.iwencai.com/unifiedwap/result/get-robot-data",
               "Origin": "https://www.iwencai.com"}
    params = {"question": question, "secondary_intent": "stock",
              "perpage": perpage, "page": 1, "block_list": "", "add_info": "1",
              "r": round(time.time() % 1, 6), "source": "Ths_iwencai_Xuangu",
              "version": "2.0", "query_type": "stock", "hexin-v": headers["hexin-v"]}
    resp = requests.post("http://www.iwencai.com/customized/chart/get-robot-data",
                         params=params, json={}, headers=headers, cookies=ck, timeout=25)
    if not resp.text.lstrip().startswith("{"):
        raise RuntimeError(f"WAF HTTP{resp.status_code}")
    d = resp.json()
    if d.get("status_code") != 0:
        raise RuntimeError(f"API {d.get('status_code')} {str(d.get('status_msg'))[:60]}")
    comps = d["data"]["answer"][0]["txt"][0]["content"]["components"]
    txt = next((c for c in comps if c.get("show_type") == "txt2"), None)
    total = None
    if txt:
        m = re.search(r"找到(\d+)", re.sub("<[^>]+>", "", txt["data"].get("content", "")))
        total = int(m.group(1)) if m else None
    tb = next((c for c in comps if str(c.get("show_type", "")).startswith("xuangu_table")
               and c["data"].get("datas")), None)
    codes = []
    if tb:
        for row in tb["data"]["datas"]:
            v = row.get("股票代码", "")
            codes.append(str(v.get("value", v) if isinstance(v, dict) else v))
    return total, codes


base_codes = []
try:
    t, base_codes = q("20230904看涨信号的股票", 10)
    print(f"base p1(perpage=10): 找到N={t} rows={len(base_codes)}")
except Exception as e:
    print(f"base FAIL: {e}")
    sys.exit(1)
time.sleep(3)

for pp in (20, 50, 100):
    try:
        t, codes = q("20230904看涨信号的股票", pp)
        print(f"perpage={pp}: 找到N={t} rows={len(codes)} {'<<< 生效' if len(codes) > 10 else '(忽略,仍10)'}")
    except Exception as e:
        print(f"perpage={pp}: FAIL {str(e)[:60]}")
    time.sleep(3)

try:
    t, codes = q("20230904看涨信号的股票第11名到第20名", 10)
    new = [c for c in codes if c not in base_codes]
    print(f"序数切片[11-20]: 找到N={t} rows={len(codes)} 其中新代码={len(new)}")
    print("  codes:", codes)
    print("  新代码:", new)
except Exception as e:
    print(f"序数切片: FAIL {str(e)[:60]}")
print("PROBE-END")
