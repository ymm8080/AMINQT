# -*- coding: utf-8 -*-
"""cap2: 看涨/看跌dict结构解剖 + 组合查询(自选股带信号列)"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
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
pairs = [f"{c['name']}={c['value']}" for c in cookies
         if ("iwencai" in c["domain"] or "10jqka" in c["domain"])]
cookie_str = "; ".join(pairs)


def dump_struct(obj, prefix="", depth=0):
    if depth > 4:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                print(f"{prefix}{k}: {type(v).__name__}[{len(v)}]")
                dump_struct(v, prefix + "  ", depth + 1)
            else:
                s = str(v)[:80]
                print(f"{prefix}{k} = {s}")
    elif isinstance(obj, list) and obj:
        print(f"{prefix}[0]: {type(obj[0]).__name__}")
        dump_struct(obj[0], prefix + "  ", depth + 1)


QUERIES = [
    "同花顺看涨信号",
    "我的自选股 看涨信号",
]

for q in QUERIES:
    print(f"\n========== {q}")
    try:
        r = pywencai.get(query=q, query_type="stock", log=False, cookie=cookie_str)
        print(f"type: {type(r).__name__}")
        if hasattr(r, "columns"):
            print(f"DataFrame {len(r)}行: {list(r.columns)[:20]}")
            safe = q.replace(" ", "_")[:20]
            r.to_parquet(rf"D:/AMINQT/AMINQT CODES/tmp_t/_ths_cap2_{safe}_0910.parquet")
            print(r.head(10).to_string()[:1200])
        else:
            dump_struct(r)
            with open(rf"D:/AMINQT/AMINQT CODES/tmp_t/_ths_cap2_raw_{abs(hash(q)) % 10000}_0910.json",
                      "w", encoding="utf-8") as f:
                json.dump(r, f, ensure_ascii=False, indent=1, default=str)
    except Exception as e:
        print(f"FAIL {type(e).__name__}: {str(e)[:200]}")

print("\nSCRIPT-END")
