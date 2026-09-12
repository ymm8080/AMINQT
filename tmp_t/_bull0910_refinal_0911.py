# -*- coding: utf-8 -*-
"""bull 20260910 盘中残照(14:31, 24只)终值重抓 — 独立单日脚本, 不碰sweep锁.

流程: 备份盘中版 → robot-data(找到N) → getDataList真分页 → 写tmp+cache → 同步bull_audit行
"""
import json
import re
import shutil
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

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

DS = "20260910"
TMP = Path(r"D:/AMINQT/AMINQT CODES/tmp_t/ths_bull_daily_0910")
CACHE = Path(r"D:/AMINQT/AMINQT CODES/data/supply_cache/ths_signal")
AUDIT = TMP / "bull_audit.csv"

cookies = json.load(open(r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_cookies_logged_0910.json", encoding="utf-8"))
ck = {c["name"]: c["value"] for c in cookies if ("iwencai" in c["domain"] or "10jqka" in c["domain"])}
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36"
URL_ROBOT = "http://www.iwencai.com/customized/chart/get-robot-data"
URL_LIST = "http://www.iwencai.com/gateway/urp/v7/landing/getDataList"
ADD_INFO = '{"urp":{"scene":1,"company":1,"business":1},"contentType":"json","searchInfo":true}'


def fresh_headers():
    import random as _rnd
    return {"User-Agent": UA, "hexin-v": pywencai.headers.headers()["hexin-v"],
            "Content-Type": "application/json",
            "Referer": "http://www.iwencai.com/unifiedwap/result/get-robot-data",
            "Origin": "http://www.iwencai.com"}, round(_rnd.random(), 6)


def _norm_code(v):
    s = str(v.get("value", v) if isinstance(v, dict) else v)
    return re.sub(r"\.\w+$", "", s.strip())


old = TMP / f"bull_{DS}.parquet"
if old.exists():
    n_old = pd.read_parquet(old, columns=["股票代码"])["股票代码"].nunique()
    shutil.copy2(old, TMP / f"bull_{DS}.parquet.intraday_1431")
    print(f"[bak] 盘中版 {n_old} 只 → .intraday_1431")

# ── robot-data: 找到N + footer_url ──
headers, _ = fresh_headers()
body = {"add_info": ADD_INFO, "perpage": "10", "page": 1,
        "source": "Ths_iwencai_Xuangu", "log_info": '{"input_type":"click"}',
        "version": "2.0", "secondary_intent": "stock",
        "question": f"{DS}看涨信号的股票"}
resp = requests.post(URL_ROBOT, json=body, headers=headers, cookies=ck, timeout=25)
assert resp.text.lstrip().startswith("{"), f"WAF HTTP{resp.status_code}"
d = resp.json()
assert d.get("status_code") == 0, d.get("status_msg")
comps = d["data"]["answer"][0]["txt"][0]["content"]["components"]
found_n = None
txt = next((c for c in comps if c.get("show_type") == "txt2"), None)
if txt:
    m = re.search(r"找到(\d+)", re.sub("<[^>]+>", "", txt["data"].get("content", "")))
    found_n = int(m.group(1)) if m else None
tb = next((c for c in comps if str(c.get("show_type", "")).startswith("xuangu_table")), None)
furl = ((tb.get("config") or {}).get("other_info") or {}).get("footer_info", {}).get("url", "") if tb else ""
print(f"[robot] found_n={found_n} furl={'有' if furl else '无'}")
assert furl, "无footer_url, 终止不改文件"  # found_n=None时同sweep语义: 继续翻页靠short_page/no_more收尾

u = urlparse(furl if "://" in furl else "http://x.com" + (furl if furl.startswith("/") else "/" + furl))
up = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in parse_qs(u.query).items()}

# ── getDataList 真分页 ──
rows, seen = [], set()
end_reason = "max_pages"
for page in range(1, 30):
    h, _ = fresh_headers()
    h = {k: v for k, v in h.items() if k != "Content-Type"}
    r = None
    for attempt in range(3):
        try:
            r = requests.post(URL_LIST, data={**up, "perpage": 100, "page": page},
                              headers=h, cookies=ck, timeout=(10, 60))
            break
        except requests.RequestException as e:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    assert r.text.lstrip().startswith("{"), f"WAF HTTP{r.status_code} p{page}"
    dd = r.json()
    assert str(dd.get("status_code")) == "0", dd.get("status_msg")
    datas = dd.get("answer", {}).get("components", [{}])[0].get("data", {}).get("datas", [])
    if not datas:
        end_reason = "no_more"
        break
    new_cnt = 0
    for row in datas:
        rec = {k: (cell.get("value", cell) if isinstance(cell, dict) else cell)
               for k, cell in row.items()}
        c = _norm_code(rec.get("股票代码", ""))
        if c and c not in seen:
            seen.add(c)
            rec["股票代码"] = c
            rows.append(rec)
            new_cnt += 1
    print(f"  p{page}: +{len(datas)} new={new_cnt}")
    if found_n is not None and len(seen) >= found_n:
        end_reason = "complete"
        break
    if new_cnt == 0:
        end_reason = "cycled"
        break
    if len(datas) < 100:
        end_reason = "short_page"
        break
    time.sleep(0.8)

df = pd.DataFrame(rows)
captured = int(df["股票代码"].nunique()) if len(df) else 0
print(f"[result] rows={len(df)} nunique={captured} found_n={found_n} end={end_reason}")
complete = ((found_n is not None and captured >= found_n) or end_reason in ("short_page", "no_more"))
if not complete:
    print("[FAIL-LOUD] 未抓全, 不落盘不更新audit")
    sys.exit(2)

df.to_parquet(old)
shutil.copy2(old, CACHE / old.name)
print(f"[write] tmp + cache 已更新 ({captured} 只)")

# ── 同步 bull_audit 行 ──
a = pd.read_csv(AUDIT, dtype={"date": str})
a = a[a["date"] != DS]
a = pd.concat([a, pd.DataFrame([{"date": DS, "found_n": found_n, "captured": captured,
                                 "end_reason": end_reason, "complete": True}])],
              ignore_index=True).sort_values("date")
a.to_csv(AUDIT, index=False)
print(f"[audit] {DS}: found_n={found_n} captured={captured} end={end_reason} complete=True")
