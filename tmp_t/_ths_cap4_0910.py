# -*- coding: utf-8 -*-
"""cap4: 看涨信号措辞探针 — 找能命中选股表格的query"""
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

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"

VARIANTS = [
    "看涨信号的股票",
    "出现看涨信号的股票",
    "同花顺看涨的股票",
    "有看涨信号的股票有哪些",
]

for q in VARIANTS:
    params = {
        "question": q, "secondary_intent": "stock", "perpage": 20, "page": 1,
        "secondary_form": "", "block_list": "", "add_info": "1", "r": "0.987654",
        "source": "Ths_iwencai_Xuangu", "version": "2.0", "query_type": "stock",
        "hexin-v": hexin,
    }
    headers = {"User-Agent": UA, "hexin-v": hexin, "Content-Type": "application/json",
               "Referer": "https://www.iwencai.com/unifiedwap/result/get-robot-data",
               "Origin": "https://www.iwencai.com"}
    try:
        r = requests.post("http://www.iwencai.com/customized/chart/get-robot-data",
                          params=params, json={}, headers=headers, cookies=ck, timeout=20)
        txt = r.text
        has_rows = '"rows"' in txt
        # 提取总数与table列名线索
        total = ""
        if '"total_count"' in txt:
            i = txt.find('"total_count"')
            total = txt[i:i + 40]
        elif '"total"' in txt:
            i = txt.find('"total"')
            total = txt[i:i + 30]
        title = ""
        if '"sug_title"' in txt:
            i = txt.find('"sug_title"')
            title = txt[i:i + 60]
        print(f"===== {q}\n  status={r.status_code} len={len(txt)} rows={has_rows} {total} {title}")
        if has_rows:
            with open(rf"D:/AMINQT/AMINQT CODES/tmp_t/_ths_cap4_hit_{abs(hash(q)) % 1000}.json",
                      "w", encoding="utf-8") as f:
                f.write(txt)
            print(f"  [saved hit]")
    except Exception as e:
        print(f"===== {q}\n  FAIL {type(e).__name__}: {str(e)[:100]}")

print("SCRIPT-END")
