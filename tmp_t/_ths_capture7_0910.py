# -*- coding: utf-8 -*-
"""DNS修复后问财首捕: monkey-patch解析 + pywencai试捕 + eq诊股域探测"""
import os
import sys

os.environ["NODE_OPTIONS"] = "--no-deprecation"
sys.stdout.reconfigure(encoding="utf-8")
import json
import socket
import time

import requests

OUT = r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_capture7_0910_result.txt"
L = []


def P(s=""):
    print(s)
    L.append(str(s))


# ---- DNS修复: THS域 -> DNSPod真实IP ----
def doh_resolve(dom):
    r = requests.get("https://1.12.12.12/resolve", params={"name": dom, "type": "A"},
                     timeout=8, headers={"Accept": "application/dns-json"})
    return [a["data"] for a in r.json().get("Answer", []) if a.get("type") == 1]


THS_HOSTS = ["www.iwencai.com", "www.10jqka.com.cn", "pass.10jqka.com.cn",
             "eq.10jqka.com.cn", "basic.10jqka.com.cn", "d.10jqka.com.cn",
             "news.10jqka.com.cn", "comment.10jqka.com.cn", "data.10jqka.com.cn",
             "stock.10jqka.com.cn", "m.10jqka.com.cn", "i.10jqka.com.cn"]
FIXED = {}
for d in THS_HOSTS:
    try:
        ips = doh_resolve(d)
        if ips:
            FIXED[d] = ips
            P(f"[dns] {d} -> {ips[:3]}")
    except Exception as e:
        P(f"[dns] {d} FAIL {type(e).__name__}: {str(e)[:60]}")

_orig_getaddrinfo = socket.getaddrinfo


def patched_getaddrinfo(host, *a, **kw):
    h = host if isinstance(host, str) else host.decode(errors="ignore")
    if h in FIXED:
        ip = FIXED[h][int(time.time() * 10) % len(FIXED[h])]  # 简易轮换
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, kw.get("port", a[1] if len(a) > 1 else 443)))]
    return _orig_getaddrinfo(host, *a, **kw)


# 注意: getaddrinfo的port在位置参数里, 精确包装
def patched(host, port, *rest, **kw):
    h = host if isinstance(host, str) else host.decode(errors="ignore")
    if h in FIXED:
        ip = FIXED[h][int(time.time() * 10) % len(FIXED[h])]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]
    return _orig_getaddrinfo(host, port, *rest, **kw)


socket.getaddrinfo = patched
P("[dns] monkey-patch applied")

# ---- pywencai 试捕 ----
import pywencai

QUERIES = [
    ("THS徽章", "同花顺看涨信号"),
    ("已知好查询", "kdj金叉"),
]
for tag, q in QUERIES:
    try:
        df = pywencai.get(query=q, query_type="stock", loop=True)
        if df is None:
            P(f"\n== [{tag}] {q!r} -> None")
        else:
            P(f"\n== [{tag}] {q!r} -> {len(df)}行")
            P(f"列({len(df.columns)}): {list(df.columns)[:12]}")
            P(df.head(3).to_string()[:1500])
    except Exception as e:
        P(f"\n== [{tag}] {q!r} -> FAIL {type(e).__name__}: {str(e)[:300]}")
    time.sleep(3)

# ---- eq诊股域探测 ----
P("\n== eq.10jqka.com.cn 探测")
for path in ["/", "/open/api/"]:
    try:
        r = requests.get(f"https://eq.10jqka.com.cn{path}", timeout=12,
                         headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0.0.0"})
        P(f"-- {path}: {r.status_code}, len={len(r.text)}")
        P(f"   {r.text[:300]!r}")
    except Exception as e:
        P(f"-- {path} FAIL {type(e).__name__}: {str(e)[:150]}")
    time.sleep(1)

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))
P(f"\nsaved -> {OUT}")
