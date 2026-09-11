# -*- coding: utf-8 -*-
"""分页修复探针: 回放 p1 的 qid/sessionid(/ticket/sess_tk/urp) 请求 page=2, 看是否出新股.
目标日 20230904 (找到N=58, v1 只抓到10). 请求预算<=4次, 间隔3s."""
import io
import json
import re
import sys
import time
import urllib.parse

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


def post(page, perpage, extra_params=None, urp=None, r_val=None):
    headers = {"User-Agent": UA, "hexin-v": pywencai.headers.headers()["hexin-v"],
               "Content-Type": "application/json",
               "Referer": "https://www.iwencai.com/unifiedwap/result/get-robot-data",
               "Origin": "https://www.iwencai.com"}
    params = {"question": f"{DSTR}看涨信号的股票", "secondary_intent": "stock",
              "perpage": perpage, "page": page, "block_list": "", "add_info": "1",
              "r": r_val or round(time.time() % 1, 6), "source": "Ths_iwencai_Xuangu",
              "version": "2.0", "query_type": "stock", "hexin-v": headers["hexin-v"]}
    if extra_params:
        params.update(extra_params)
    body = {}
    if urp is not None:
        body = {"urp": json.dumps(urp, ensure_ascii=False)}
    resp = requests.post("http://www.iwencai.com/customized/chart/get-robot-data",
                         params=params, json=body, headers=headers, cookies=ck, timeout=25)
    if not resp.text.lstrip().startswith("{"):
        raise RuntimeError(f"WAF HTTP{resp.status_code}")
    d = resp.json()
    if d.get("status_code") != 0:
        raise RuntimeError(f"API {d.get('status_code')} {str(d.get('status_msg'))[:60]}")
    content = d["data"]["answer"][0]["txt"][0]["content"]
    comps = content["components"]
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
    return total, codes, content, tb["data"].get("meta", {}) if tb else {}


t0, c1, content1, meta1 = post(1, 10)
print(f"p1: 找到N={t0} rows={len(c1)}")
print("  codes:", c1)
qid = meta1.get("qid") or ""
sess = meta1.get("sessionid") or ""
print(f"  qid={qid} sessionid={sess[:20]}...")
time.sleep(3)

# V1 简单回放: page=2 + qid + sessionid
t2, c2, _, _ = post(2, 10, extra_params={"qid": qid, "sessionid": sess})
print(f"V1(page2+qid+sess): rows={len(c2)} 新代码={len(set(c2)-set(c1))}")
print("  codes:", c2[:10])
if len(set(c2) - set(c1)) == 0:
    time.sleep(3)
    # V2 全量回放: 从 request_params 提 urp, 改 page=2 重放
    rp = content1.get("request_params", "")
    urp_raw = ""
    for kv in rp.split("&"):
        if kv.startswith("urp="):
            urp_raw = urllib.parse.unquote(kv[4:])
            break
    if urp_raw:
        urp = json.loads(urp_raw.replace("'", '"'))
        urp["page"] = 2
        urp["perpage"] = 10
        t3, c3, _, _ = post(2, 10, extra_params={"qid": qid, "sessionid": sess}, urp=urp)
        print(f"V2(urp.page=2全回放): rows={len(c3)} 新代码={len(set(c3)-set(c1))}")
        print("  codes:", c3[:10])
    else:
        print("V2 skip: p1 响应无 urp")
print("PROBE-END")
