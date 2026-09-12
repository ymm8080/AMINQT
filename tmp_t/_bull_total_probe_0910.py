# -*- coding: utf-8 -*-
"""决定性探针: 看涨查询的 找到N 总数 vs 分页实际可取行数 (2个样例日)."""
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


def q(dstr, page):
    headers = {"User-Agent": UA, "hexin-v": pywencai.headers.headers()["hexin-v"],
               "Content-Type": "application/json",
               "Referer": "https://www.iwencai.com/unifiedwap/result/get-robot-data",
               "Origin": "https://www.iwencai.com"}
    params = {"question": f"{dstr}看涨信号的股票", "secondary_intent": "stock",
              "perpage": 10, "page": page, "block_list": "", "add_info": "1",
              "r": round(time.time() % 1, 6), "source": "Ths_iwencai_Xuangu", "version": "2.0",
              "query_type": "stock", "hexin-v": headers["hexin-v"]}
    resp = requests.post("http://www.iwencai.com/customized/chart/get-robot-data",
                         params=params, json={}, headers=headers, cookies=ck, timeout=25)
    if not resp.text.lstrip().startswith("{"):
        raise RuntimeError(f"WAF HTTP{resp.status_code}")
    d = resp.json()
    if d.get("status_code") != 0:
        raise RuntimeError(f"API {d.get('status_code')}")
    comps = d["data"]["answer"][0]["txt"][0]["content"]["components"]
    txt = next((c for c in comps if c.get("show_type") == "txt2"), None)
    total = None
    if txt:
        m = re.search(r"找到(\d+)", re.sub("<[^>]+>", "", txt["data"].get("content", "")))
        total = int(m.group(1)) if m else None
    tb = next((c for c in comps if c.get("show_type", "").startswith("xuangu_table")
               and c.get("data", {}).get("datas")), None)
    codes = []
    if tb:
        cols = tb["data"]["columns"]
        for row in tb["data"]["datas"]:
            k = next((c.get("key") or c.get("title") for c in cols
                      if (c.get("key") or c.get("title")) == "股票代码"), "股票代码")
            v = row.get(k, "")
            codes.append(str(v.get("value", v) if isinstance(v, dict) else v))
    return total, codes


for dstr in (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("20240305", "20230905"):
    try:
        total, p1 = q(dstr, 1)
        time.sleep(2)
        _, p2 = q(dstr, 2)
        time.sleep(2)
        _, p3 = q(dstr, 3)
        print(f"{dstr}: 找到N={total}")
        print(f"  p1={p1}")
        print(f"  p2={p2} (与p1重叠 {len(set(p1)&set(p2))}/10)")
        print(f"  p3={p3} (与p1重叠 {len(set(p1)&set(p3))}/10)")
    except Exception as e:
        print(f"{dstr}: FAIL {type(e).__name__} {str(e)[:60]}")
    time.sleep(3)
