# -*- coding: utf-8 -*-
"""登录后抓取: 自选股看涨/看跌 + THS看涨信号池 vs D-rule 对照"""
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

# 1) 组装cookie串 (登录态: 从用户Edge profile原生解出)
cookies = json.load(open(r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_cookies_logged_0910.json", encoding="utf-8"))
pairs = [f"{c['name']}={c['value']}" for c in cookies
         if ("iwencai" in c["domain"] or "10jqka" in c["domain"])]
cookie_str = "; ".join(pairs)
has_login = any(c["name"] in ("userid", "user_id", "snuid") for c in cookies)
print(f"[cookie] {len(pairs)}对, 登录态={has_login}")

QUERIES = [
    "同花顺看涨信号",        # 全池: 看涨标的
    "同花顺看跌信号",        # 对照池
    "我的自选股",            # 用户自选(需登录)
]

results = {}
for q in QUERIES:
    print(f"\n===== query: {q}")
    try:
        df = pywencai.get(query=q, query_type="stock", loop=True, log=True, cookie=cookie_str)
        if df is None:
            print("   -> None (while_do吞错, 落raw)")
            results[q] = {"ok": False}
            continue
        n = len(df)
        cols = list(df.columns)[:25]
        print(f"   -> {n}行 x {len(df.columns)}列")
        print(f"   cols: {cols}")
        results[q] = {"ok": True, "n": n, "cols": cols}
        safe = q.replace("/", "_")[:20]
        out = rf"D:/AMINQT/AMINQT CODES/tmp_t/_ths_cap_{safe}_0910.parquet"
        df.to_parquet(out)
        print(f"   saved: {out}")
        print(df.head(8).to_string()[:1500])
    except Exception as e:
        print(f"   FAIL {type(e).__name__}: {str(e)[:200]}")
        results[q] = {"ok": False, "err": str(e)[:200]}

with open(r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_cap_result_0910.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print("\nSCRIPT-END")
