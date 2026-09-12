# -*- coding: utf-8 -*-
"""单次探针: 看跌查询 perpage=100 是否生效 + 返回表 schema (2024-03-05 样例日)."""
import io
import json
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

DSTR = sys.argv[1] if len(sys.argv) > 1 else "20240305"
PERPAGE = int(sys.argv[2]) if len(sys.argv) > 2 else 100

headers = {"User-Agent": UA, "hexin-v": pywencai.headers.headers()["hexin-v"],
           "Content-Type": "application/json",
           "Referer": "https://www.iwencai.com/unifiedwap/result/get-robot-data",
           "Origin": "https://www.iwencai.com"}
params = {"question": f"{DSTR}看跌信号的股票", "secondary_intent": "stock",
          "perpage": PERPAGE, "page": 1, "block_list": "", "add_info": "1",
          "r": round(time.time() % 1, 6), "source": "Ths_iwencai_Xuangu", "version": "2.0",
          "query_type": "stock", "hexin-v": headers["hexin-v"]}
resp = requests.post("http://www.iwencai.com/customized/chart/get-robot-data",
                     params=params, json={}, headers=headers, cookies=ck, timeout=25)
if not resp.text.lstrip().startswith("{"):
    print(f"非JSON HTTP{resp.status_code}: {resp.text[:200]}")
    sys.exit(2)
d = resp.json()
if d.get("status_code") != 0:
    print(f"API err {d.get('status_code')} {str(d.get('status_msg'))[:80]}")
    sys.exit(2)
comps = d["data"]["answer"][0]["txt"][0]["content"]["components"]
tb = next((c for c in comps if c.get("show_type", "").startswith("xuangu_table")
           and c.get("data", {}).get("datas")), None)
txt = next((c for c in comps if c.get("show_type") == "txt2"), None)
total = None
if txt:
    import re
    m = re.search(r"找到(\d+)", re.sub("<[^>]+>", "", txt["data"].get("content", "")))
    total = int(m.group(1)) if m else None
print(f"perpage={PERPAGE} -> page1 rows={len(tb['data']['datas']) if tb else 0}, 找到N={total}")
if tb:
    cols = tb["data"]["columns"]
    print("columns:", [c.get("key") or c.get("title") for c in cols][:25])
    row0 = tb["data"]["datas"][0]
    for col in cols[:14]:
        k = col.get("key") or col.get("title")
        v = row0.get(k, "")
        v = v.get("value", v) if isinstance(v, dict) else v
        print(f"  {k!r}: {str(v)[:60]}")
