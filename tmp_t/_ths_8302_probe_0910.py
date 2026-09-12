# -*- coding: utf-8 -*-
"""探针F: 定位 -8302 — 20260909 的 robot→datalist 流, 对比 perpage=100 vs 30, dump footer."""
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
DSTR = sys.argv[1] if len(sys.argv) > 1 else "20260909"


def robot(question):
    hv = pywencai.headers.headers()["hexin-v"]
    headers = {"User-Agent": UA, "hexin-v": hv, "Content-Type": "application/json",
               "Referer": "http://www.iwencai.com/unifiedwap/result/get-robot-data",
               "Origin": "http://www.iwencai.com"}
    body = {"add_info": '{"urp":{"scene":1,"company":1,"business":1},"contentType":"json","searchInfo":true}',
            "perpage": "10", "page": 1, "source": "Ths_iwencai_Xuangu",
            "log_info": '{"input_type":"click"}', "version": "2.0",
            "secondary_intent": "stock", "question": question}
    resp = requests.post("http://www.iwencai.com/customized/chart/get-robot-data",
                         json=body, headers=headers, cookies=ck, timeout=25)
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
    tb = next((c for c in comps if str(c.get("show_type", "")).startswith("xuangu_table")), None)
    furl = ""
    if tb is not None:
        oi = (tb.get("config") or {}).get("other_info") or {}
        furl = (oi.get("footer_info") or {}).get("url", "")
    return total, furl


total, furl = robot(f"{DSTR}看涨信号的股票")
print(f"robot: 找到N={total}")
print(f"footer: {furl[:300]}")
if not furl:
    sys.exit(0)
u = urlparse(furl if "://" in furl else "http://x.com" + (furl if furl.startswith("/") else "/" + furl))
up = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in parse_qs(u.query).items()}
time.sleep(3)
for pp in (30, 100):
    hv = pywencai.headers.headers()["hexin-v"]
    r = requests.post("http://www.iwencai.com/gateway/urp/v7/landing/getDataList",
                      data={**up, "perpage": pp, "page": 1},
                      headers={"User-Agent": UA, "hexin-v": hv,
                               "Referer": "http://www.iwencai.com/unifiedwap/result/get-robot-data"},
                      cookies=ck, timeout=25)
    head = r.text[:100].replace(chr(10), ' ')
    try:
        dj = r.json()
        datas = dj.get("answer", {}).get("components", [{}])[0].get("data", {}).get("datas", [])
        print(f"perpage={pp}: HTTP{r.status_code} status={dj.get('status_code')} rows={len(datas)} head={head}")
    except Exception:
        print(f"perpage={pp}: HTTP{r.status_code} 非JSON head={head}")
    time.sleep(3)
print("PROBE-END")
