# -*- coding: utf-8 -*-
"""看涨池缺口补填 — 忽略state从头扫全部日期, 只抓缺失文件 (resume-jump坑的解法).

续跑脚本从 last_done 之后开始, 散布在 last_done 之前的失败日永远不会被复跑填上.
本脚本无状态扫描: bull_YYYYMMDD.parquet 缺失才抓. 用法与主脚本同一通道/同限速.
"""
import io
import json
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import importlib.util

spec = importlib.util.spec_from_file_location(
    "bullbf", r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_backfill3y_0910.py")
# 不能 import (模块级会执行主循环) — 只复用 fetch_day: 手动加载到 fetch_day 定义为止不可行,
# 因此这里独立实现精简版 fetch (与主脚本逐行同逻辑, 独立演化风险自担).

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
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36"


def fresh_headers():
    import random as _rnd
    return {"User-Agent": UA, "hexin-v": pywencai.headers.headers()["hexin-v"],
            "Content-Type": "application/json",
            "Referer": "https://www.iwencai.com/unifiedwap/result/get-robot-data",
            "Origin": "https://www.iwencai.com"}, round(_rnd.random(), 6)


OUT_DIR = Path(r"D:/AMINQT/AMINQT CODES/tmp_t/ths_bull_daily_0910")

pf = pd.read_parquet("D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet",
                     columns=["date", "symbol"])
dates = sorted(pd.to_datetime(pf["date"].unique()))
dates = [d for d in dates if d >= pd.Timestamp("2023-09-01")]
del pf

missing = [d for d in dates if not (OUT_DIR / f"bull_{d.strftime('%Y%m%d')}.parquet").exists()]
print(f"[cal] 缺口 {len(missing)}/{len(dates)} 天")
if not missing:
    print("SCRIPT-END")
    sys.exit(0)


def fetch_day(dstr):
    import re
    rows, seen = [], set()
    for page in range(1, 41):
        headers, rval = fresh_headers()
        params = {"question": f"{dstr}看涨信号的股票", "secondary_intent": "stock",
                  "perpage": 10, "page": page, "block_list": "", "add_info": "1",
                  "r": rval, "source": "Ths_iwencai_Xuangu", "version": "2.0",
                  "query_type": "stock", "hexin-v": headers["hexin-v"]}
        resp = requests.post("http://www.iwencai.com/customized/chart/get-robot-data",
                             params=params, json={}, headers=headers, cookies=ck, timeout=25)
        if not resp.text.lstrip().startswith("{"):
            raise RuntimeError(f"WAF 非JSON HTTP{resp.status_code}")
        d = resp.json()
        if d.get("status_code") != 0:
            raise RuntimeError(f"API {d.get('status_code')}")
        comps = d["data"]["answer"][0]["txt"][0]["content"]["components"]
        tb = next((c for c in comps if c.get("show_type", "").startswith("xuangu_table")
                   and c.get("data", {}).get("datas")), None)
        if tb is None:
            break
        cols = tb["data"]["columns"]
        page_codes = []
        for row in tb["data"]["datas"]:
            rec = {}
            for col in cols:
                key = col.get("key") or col.get("title")
                cell = row.get(key, "")
                rec[key] = cell.get("value", cell) if isinstance(cell, dict) else cell
            rec["_query_date"] = dstr
            page_codes.append(str(rec.get("股票代码", "")))
            rows.append(rec)
        if page_codes and all(c in seen for c in page_codes):
            rows = rows[:len(rows) - len(page_codes)]
            break
        seen.update(page_codes)
        if len(tb["data"]["datas"]) < 10:
            break
        time.sleep(0.8)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


consec = 0
for k, d in enumerate(missing):
    dstr = d.strftime("%Y%m%d")
    try:
        df = fetch_day(dstr)
        df.to_parquet(OUT_DIR / f"bull_{dstr}.parquet")
        consec = 0
        print(f"[gap {k+1}/{len(missing)}] {dstr}: {len(df)}行")
        time.sleep(3.0)
    except Exception as e:
        consec += 1
        print(f"[FAIL] {dstr}: {type(e).__name__} {str(e)[:60]} consec={consec}")
        if "WAF" in str(e):
            time.sleep(300)
        else:
            time.sleep(15)
        if consec >= 5:
            print("[ABORT] 连续5失败")
            sys.exit(2)
print(f"[done] 缺口补填完成")
print("SCRIPT-END")
