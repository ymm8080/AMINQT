# -*- coding: utf-8 -*-
"""探针E: pywencai 原样 body 风格 robot-data (json body + urp add_info) → footer_info.url → getDataList 分页."""
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


def robot_body(question):
    hv = pywencai.headers.headers()["hexin-v"]
    headers = {"User-Agent": UA, "hexin-v": hv,
               "Content-Type": "application/json",
               "Referer": "http://www.iwencai.com/unifiedwap/result/get-robot-data",
               "Origin": "http://www.iwencai.com"}
    body = {
        "add_info": "{\"urp\":{\"scene\":1,\"company\":1,\"business\":1},\"contentType\":\"json\",\"searchInfo\":true}",
        "perpage": "10",
        "page": 1,
        "source": "Ths_iwencai_Xuangu",
        "log_info": "{\"input_type\":\"click\"}",
        "version": "2.0",
        "secondary_intent": "stock",
        "question": question,
    }
    resp = requests.post("http://www.iwencai.com/customized/chart/get-robot-data",
                         json=body, headers=headers, cookies=ck, timeout=25)
    if not resp.text.lstrip().startswith("{"):
        raise RuntimeError(f"WAF HTTP{resp.status_code}")
    d = resp.json()
    if d.get("status_code") != 0:
        raise RuntimeError(f"API {d.get('status_code')} {str(d.get('status_msg'))[:60]}")
    comps = d["data"]["answer"][0]["txt"][0]["content"]["components"]
    tb = next((c for c in comps if str(c.get("show_type", "")).startswith("xuangu_table")), None)
    if tb is None:
        raise RuntimeError("no table comp")
    cfg = tb.get("config") or {}
    oi = cfg.get("other_info") or {}
    furl = (oi.get("footer_info") or {}).get("url", "")
    print(f"  table config keys: {sorted(cfg.keys())}")
    if not furl:
        # 兜底: 全响应搜 landing/getDataList 或 footer url
        raw = json.dumps(d, ensure_ascii=False)
        m = re.search(r'https?://[^"\\\']+landing[^"\\\']*', raw)
        m2 = re.search(r'"url"\s*:\s*"(http[^"]+)"', raw)
        print(f"  全响应搜: landing={bool(m)} 首个url={bool(m2)}")
        if m:
            furl = m.group(0)
        elif m2:
            furl = m2.group(1)
    codes = []
    if tb.get("data", {}).get("datas"):
        for row in tb["data"]["datas"]:
            v = row.get("股票代码", "")
            codes.append(str(v.get("value", v) if isinstance(v, dict) else v))
    return codes, furl


codes1, furl = robot_body(f"{DSTR}看涨信号的股票")
print(f"p1: rows={len(codes1)} codes={codes1[:5]}")
print(f"footer_url: {furl[:250]}")
if not furl:
    print("仍无 url, 探针结束")
    sys.exit(0)
up = parse_qs(furl if "://" in furl else f"http://x.com{furl if furl.startswith('/') else '/'+furl}").__class__ and {}
u = urlparse(furl if "://" in furl else "http://x.com" + (furl if furl.startswith("/") else "/" + furl))
q = parse_qs(u.query)
up = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in q.items()}
print(f"url_params keys: {sorted(up.keys())[:20]}")
time.sleep(3)

headers = {"User-Agent": UA, "hexin-v": pywencai.headers.headers()["hexin-v"],
           "Referer": "http://www.iwencai.com/unifiedwap/result/get-robot-data"}
for page in (1, 2):
    data = {**up, "perpage": 100, "page": page}
    r = requests.post("http://www.iwencai.com/gateway/urp/v7/landing/getDataList",
                      data=data, headers=headers, cookies=ck, timeout=25)
    print(f"getDataList page{page}: HTTP{r.status_code} head={r.text[:150].replace(chr(10),' ')}")
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
