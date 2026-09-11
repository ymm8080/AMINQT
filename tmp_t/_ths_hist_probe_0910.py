# -*- coding: utf-8 -*-
"""hist-probe: 历史看涨/看跌抓取路线探测 (日期快照 vs 区间反查)"""
import json
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
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36"


def probe(q):
    params = {"question": q, "secondary_intent": "stock", "perpage": 10, "page": 1,
              "block_list": "", "add_info": "1", "r": "0.777", "source": "Ths_iwencai_Xuangu",
              "version": "2.0", "query_type": "stock", "hexin-v": hexin}
    headers = {"User-Agent": UA, "hexin-v": hexin, "Content-Type": "application/json",
               "Referer": "https://www.iwencai.com/unifiedwap/result/get-robot-data",
               "Origin": "https://www.iwencai.com"}
    try:
        r = requests.post("http://www.iwencai.com/customized/chart/get-robot-data",
                          params=params, json={}, headers=headers, cookies=ck, timeout=25)
        d = r.json()
        if d.get("status_code") != 0:
            return f"status={d.get('status_code')} {str(d.get('status_msg'))[:50]}"
        comps = d["data"]["answer"][0]["txt"][0]["content"]["components"]
        tb = next((c for c in comps if c.get("show_type", "").startswith("xuangu_table")
                   and c.get("data", {}).get("datas")), None)
        txt = next((c for c in comps if c.get("show_type") == "txt2"), None)
        summary = ""
        if txt:
            import re
            content = txt["data"].get("content", "")
            m = re.search(r"找到(\d+)", re.sub("<[^>]+>", "", content))
            summary = f"总数={m.group(1) if m else '?'} "
        if tb is None:
            return f"{summary}无表格"
        cols = [c.get("title") or c.get("key") for c in tb["data"]["columns"]]
        return f"{summary}行={len(tb['data']['datas'])} 列={cols[:8]}"
    except Exception as e:
        return f"FAIL {type(e).__name__}: {str(e)[:80]}"


PROBES = [
    "20260905看涨信号的股票",           # 路线1: 历史单日快照
    "20260829看跌信号的股票",           # 路线1b: 看跌历史
    "近三年出现看涨信号的股票",          # 路线2: 区间反查
    "最近一周看涨信号日期",              # 路线2b: 日期列
]

for q in PROBES:
    print(f"===== {q}")
    print(f"  {probe(q)}")

print("SCRIPT-END")
