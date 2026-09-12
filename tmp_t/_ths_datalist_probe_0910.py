# -*- coding: utf-8 -*-
"""探针D: pywencai 式真分页 — get-robot-data 拿 footer_info.url → getDataList (form, perpage=100).
目标日 20230904 (找到N=58). 预算: 1 robot-data + 2 getDataList."""
import io
import json
import re
import sys
import time
from urllib.parse import urlparse, parse_qs

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
DSTR = "20230904"


def robot(question):
    headers = {"User-Agent": UA, "hexin-v": pywencai.headers.headers()["hexin-v"],
               "Content-Type": "application/json",
               "Referer": "https://www.iwencai.com/unifiedwap/result/get-robot-data",
               "Origin": "https://www.iwencai.com"}
    params = {"question": question, "secondary_intent": "stock",
              "perpage": 10, "page": 1, "block_list": "", "add_info": "1",
              "r": round(time.time() % 1, 6), "source": "Ths_iwencai_Xuangu",
              "version": "2.0", "query_type": "stock", "hexin-v": headers["hexin-v"]}
    resp = requests.post("http://www.iwencai.com/customized/chart/get-robot-data",
                         params=params, json={}, headers=headers, cookies=ck, timeout=25)
    if not resp.text.lstrip().startswith("{"):
        raise RuntimeError(f"WAF HTTP{resp.status_code}")
    d = resp.json()
    if d.get("status_code") != 0:
        raise RuntimeError(f"API {d.get('status_code')}")
    comps = d["data"]["answer"][0]["txt"][0]["content"]["components"]
    tb = next((c for c in comps if str(c.get("show_type", "")).startswith("xuangu_table")), None)
    if tb is None:
        raise RuntimeError("no table comp")
    footer_url = (tb.get("config", {}).get("other_info", {}) or {}).get("footer_info", {}).get("url", "")
    codes = []
    if tb.get("data", {}).get("datas"):
        for row in tb["data"]["datas"]:
            v = row.get("股票代码", "")
            codes.append(str(v.get("value", v) if isinstance(v, dict) else v))
    return codes, footer_url


def parse_qs_single(url):
    q = parse_qs(urlparse(url).query)
    return {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in q.items()}


codes1, furl = robot(f"{DSTR}看涨信号的股票")
print(f"p1: rows={len(codes1)} codes={codes1[:5]}")
print(f"footer_url: {furl[:200]}")
if not furl:
    print("无 footer_info.url, 探针结束")
    sys.exit(0)
up = parse_qs_single(furl)
print(f"url_params keys: {sorted(up.keys())}")
time.sleep(3)

headers = {"User-Agent": UA, "hexin-v": pywencai.headers.headers()["hexin-v"],
           "Referer": "http://www.iwencai.com/unifiedwap/result/get-robot-data"}
for page in (1, 2):
    data = {**up, "perpage": 100, "page": page}
    r = requests.post("http://www.iwencai.com/gateway/urp/v7/landing/getDataList",
                      data=data, headers=headers, cookies=ck, timeout=25)
    print(f"getDataList page{page}: HTTP{r.status_code} head={r.text[:120].replace(chr(10),' ')}")
    try:
        dj = r.json()
        datas = dj.get("answer", {}).get("components", [{}])[0].get("data", {}).get("datas", [])
        codes = [str(row.get("股票代码", row.get("code", ""))) for row in datas]
        new = [c for c in codes if c not in codes1] if page == 2 else codes
        print(f"  rows={len(datas)} 新vs_p1={len(new)} sample={codes[:8]}")
    except Exception as e:
        print(f"  parse FAIL {e}")
    time.sleep(3)
print("PROBE-END")
