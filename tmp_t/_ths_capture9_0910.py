# -*- coding: utf-8 -*-
"""capture9: urllib3级IP修正 + pywencai标准请求 + 失败时裸打原始回包"""
import os
import sys

os.environ["NODE_OPTIONS"] = "--no-deprecation"
sys.stdout.reconfigure(encoding="utf-8")
import time

# ---- IP修正 (在import pywencai之前) ----
import requests.adapters
import urllib3.util.connection as u3c

IP_IWENCAI = "121.12.127.73"
_orig_cc = u3c.create_connection


def fixed_create_connection(address, *a, **kw):
    host, port = address
    if host == "www.iwencai.com":
        address = (IP_IWENCAI, port)
    return _orig_cc(address, *a, **kw)


u3c.create_connection = fixed_create_connection
print(f"[fix] www.iwencai.com -> {IP_IWENCAI}")

import pywencai

QUERIES = [
    ("THS徽章", "同花顺看涨信号"),
    ("已知好查询", "kdj金叉"),
]
ok = False
for tag, q in QUERIES:
    try:
        df = pywencai.get(query=q, query_type="stock", loop=True, log=True)
        if df is None:
            print(f"\n== [{tag}] {q!r} -> None (10次重试全败)")
        else:
            ok = True
            print(f"\n== [{tag}] {q!r} -> {len(df)}行")
            print(f"列({len(df.columns)}): {list(df.columns)[:14]}")
            print(df.head(3).to_string()[:1600])
    except Exception as e:
        print(f"\n== [{tag}] {q!r} -> EXC {type(e).__name__}: {str(e)[:300]}")
    time.sleep(3)

# ---- pywencai失败则裸打 (完全复刻: http+json+hexin-v头) ----
if not ok:
    print("\n== 裸打复刻 (http + json body + hexin-v header)")
    import requests
    from pywencai.headers import headers as mk_headers

    hd = mk_headers()
    data = {
        'add_info': '{"urp":{"scene":1,"company":1,"business":1},"contentType":"json","searchInfo":true}',
        'perpage': '10', 'page': 1, 'source': 'Ths_iwencai_Xuangu',
        'log_info': '{"input_type":"click"}', 'version': '2.0',
        'secondary_intent': 'stock', 'question': 'kdj金叉',
    }
    try:
        r = requests.post("http://www.iwencai.com/customized/chart/get-robot-data",
                          json=data, headers=hd, timeout=25)
        print(f"   status={r.status_code}, len={len(r.text)}, ct={r.headers.get('content-type')}")
        print(f"   body[:1000]: {r.text[:1000]}")
    except Exception as e:
        print(f"   FAIL {type(e).__name__}: {str(e)[:250]}")
