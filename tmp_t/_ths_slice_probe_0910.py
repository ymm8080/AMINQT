# -*- coding: utf-8 -*-
"""探针C: ①代码前缀切片 (60/00开头) 解析与池数一致性 vs 基础找到N=58; ②前端JS翻页接口名摸底."""
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


seen = set()
for suf in ("", " 60开头", " 00开头", " 30开头", " 68开头"):
    try:
        t, codes = q(f"20230904看涨信号的股票{suf}", 10)
        new = [c for c in codes if c not in seen]
        seen.update(codes)
        print(f"[{suf.strip() or 'base'}]: 找到N={t} rows={len(codes)} 新={len(new)} codes={codes[:6]}")
    except Exception as e:
        print(f"[{suf.strip() or 'base'}]: FAIL {str(e)[:60]}")
    time.sleep(3)
print(f"并集去重={len(seen)} (基础找到N=58)")
time.sleep(2)

# 前端JS: 找翻页接口名 (静态资产, WAF较松)
try:
    r = requests.get("http://www.iwencai.com/unifiedwap/result?querytype=stock&q=600019",
                     headers={"User-Agent": UA}, cookies=ck, timeout=20)
    apis = sorted(set(re.findall(r'[\"\']([\w/\-]*get-[\w\-]+)[\"\']', r.text)))
    apis += sorted(set(re.findall(r'[\"\']([\w/\-]*page[\w\-]*\.do|[\w/\-]*data[\w\-]*\.json)[\"\']', r.text)))
    print(f"前端页面 HTTP{r.status_code} len={len(r.text)} API线索: {apis[:20]}")
    scripts = re.findall(r'src=[\"\'](http[^\"\']+\.js[^\"\']*)[\"\']', r.text)
    print(f"js bundles: {scripts[:8]}")
except Exception as e:
    print(f"前端页面 FAIL {str(e)[:60]}")
print("PROBE-END")
