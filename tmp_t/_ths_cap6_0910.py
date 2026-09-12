# -*- coding: utf-8 -*-
"""cap6: 看涨/看跌信号全池抓取 — xuangu_tableV1 → DataFrame → parquet"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import requests
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
ck = {c["name"]: c["value"] for c in cookies if ("iwencai" in c["domain"] or "10jqka" in c["domain"])}
hexin = pywencai.headers.headers()["hexin-v"]
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36"


def fetch_table(question, perpage=100):
    params = {"question": question, "secondary_intent": "stock", "perpage": perpage, "page": 1,
              "block_list": "", "add_info": "1", "r": "0.321", "source": "Ths_iwencai_Xuangu",
              "version": "2.0", "query_type": "stock", "hexin-v": hexin}
    headers = {"User-Agent": UA, "hexin-v": hexin, "Content-Type": "application/json",
               "Referer": "https://www.iwencai.com/unifiedwap/result/get-robot-data",
               "Origin": "https://www.iwencai.com"}
    r = requests.post("http://www.iwencai.com/customized/chart/get-robot-data",
                      params=params, json={}, headers=headers, cookies=ck, timeout=25)
    d = r.json()
    if d.get("status_code") != 0:
        print(f"  API status_code={d.get('status_code')} {str(d.get('status_msg'))[:60]}")
        return None
    comps = d["data"]["answer"][0]["txt"][0]["content"]["components"]
    for c in comps:
        if c.get("show_type", "").startswith("xuangu_table") and c.get("data", {}).get("datas"):
            cols = [col.get("title") or col.get("key") for col in c["data"]["columns"]]
            rows = []
            for row in c["data"]["datas"]:
                vals = {}
                for i, col in enumerate(c["data"]["columns"]):
                    key = col.get("key") or (cols[i] if i < len(cols) else f"c{i}")
                    cell = row.get(key, row[i] if isinstance(row, list) else "")
                    vals[key] = cell.get("value", cell) if isinstance(cell, dict) else cell
                rows.append(vals)
            df = pd.DataFrame(rows)
            return df
    return None


for label, q in [("bull", "看涨信号的股票"), ("bear", "看跌信号的股票")]:
    print(f"===== {label}: {q}")
    df = fetch_table(q)
    if df is None:
        print("  无表格")
        continue
    print(f"  {len(df)}行 x {len(df.columns)}列: {list(df.columns)[:12]}")
    out = rf"D:/AMINQT/AMINQT CODES/tmp_t/_ths_signal_{label}_0910.parquet"
    df.to_parquet(out)
    print(f"  saved: {out}")
    print(df.head(30).to_string()[:2500])

print("SCRIPT-END")
