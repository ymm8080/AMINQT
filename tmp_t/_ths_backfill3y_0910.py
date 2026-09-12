# -*- coding: utf-8 -*-
"""3年看涨信号池逐日回填 v3 — getDataList 真分页版.

沿革: v1 dup-break 截断 (找到N=58只抓10); v2 found_n审计但端点仍锁10行/页;
v3 破局: pywencai 源码逆向 — robot-data(body风格) 返回 footer_info.url,
POST gateway/urp/v7/landing/getDataList (form) perpage=100 真生效,
20230904 探针 58/58 一页全中. 每日 ≈2-3请求全量.
保留: found_n 审计 (bull_audit.csv), 全量扫描 (无审计行/不完整→重抓), 单实例锁, fail-fast.
"""
import json
import os
import re
import subprocess
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


OUT_DIR = Path(r"D:/AMINQT/AMINQT CODES/tmp_t/ths_bull_daily_0910")
OUT_DIR.mkdir(exist_ok=True)
STATE = OUT_DIR / "_state.json"
AUDIT = OUT_DIR / "bull_audit.csv"
MAX_PAGES = 60

# 交易日历: 本地面板 (回填到09-09, 与面板对齐)
pf = pd.read_parquet("D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet",
                     columns=["date", "symbol"])
dates = sorted(pd.to_datetime(pf["date"].unique()))
dates = [d for d in dates if d >= pd.Timestamp("2023-09-01")]
del pf
print(f"[cal] {len(dates)}个交易日 {dates[0].date()} ~ {dates[-1].date()}")

state = {"last_done": "", "fail_date": "", "consec_fail": 0}
if STATE.exists():
    state.update(json.load(open(STATE, encoding="utf-8")))
# consec 是进程内概念: 上次abort遗留的计数若被继承, 新实例首败即触发>=8假abort
state["consec_fail"] = 0
STATE.write_text(json.dumps(state), encoding="utf-8")
state["consec_fail"] = 0  # 失败连击每进程清零: 不继承上实例残值(8940继承8首败即abort)

# 单实例锁: 活PID持锁, 死PID自动过期 (防 babysitter/runner 双开竞写审计)
LOCK = OUT_DIR / "_sweep.lock"


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                             capture_output=True, text=True, timeout=15).stdout
        return f" {pid} " in out.replace("\n", " ") + " "
    except Exception:
        return True


if LOCK.exists():
    try:
        opid = int(LOCK.read_text(encoding="utf-8").strip() or 0)
    except Exception:
        opid = 0
    if opid and opid != os.getpid() and _pid_alive(opid):
        print(f"[lock] 扫描实例 PID {opid} 在跑, 本实例退出 (exit 3)")
        sys.exit(3)
LOCK.write_text(str(os.getpid()), encoding="utf-8")


def _load_audit():
    if AUDIT.exists():
        for _ in range(3):
            try:
                return pd.read_csv(AUDIT, dtype={"date": str})
            except Exception:
                time.sleep(2)
    return pd.DataFrame(columns=["date", "found_n", "captured", "end_reason", "complete"])


audit = _load_audit()


def _audit_valid(dstr: str) -> bool:
    a = audit[audit["date"] == dstr] if len(audit) else pd.DataFrame()
    if a.empty:
        return False
    return bool(a.iloc[0]["complete"])


def _append_audit(dstr, found_n, captured, end_reason, complete):
    global audit
    audit = audit[audit["date"] != dstr]
    row = pd.DataFrame([{"date": dstr, "found_n": found_n, "captured": captured,
                         "end_reason": end_reason, "complete": bool(complete)}])
    audit = pd.concat([audit, row], ignore_index=True).sort_values("date")
    _tmp = AUDIT.with_suffix(".csv.tmp")
    audit.to_csv(_tmp, index=False)
    _tmp.replace(AUDIT)


def _norm_code(v):
    s = str(v.get("value", v) if isinstance(v, dict) else v)
    return re.sub(r"\.\w+$", "", s.strip())


def _row_rec(row, dstr):
    rec = {}
    for k, cell in row.items():
        rec[k] = cell.get("value", cell) if isinstance(cell, dict) else cell
    rec["_query_date"] = dstr
    return rec


def _robot_day(dstr, question):
    """body风格 robot-data → (found_n, footer_url)."""
    headers, _ = fresh_headers()
    body = {"add_info": ADD_INFO, "perpage": "10", "page": 1,
            "source": "Ths_iwencai_Xuangu", "log_info": '{"input_type":"click"}',
            "version": "2.0", "secondary_intent": "stock",
            "question": question.format(dstr=dstr)}
    resp = requests.post(URL_ROBOT, json=body, headers=headers, cookies=ck, timeout=25)
    if not resp.text.lstrip().startswith("{"):
        (OUT_DIR / f"_debug_{dstr}_robot.html").write_text(
            f"HTTP {resp.status_code}\n{resp.text[:500]}", encoding="utf-8")
        raise RuntimeError(f"WAF 非JSON HTTP{resp.status_code}")
    d = resp.json()
    if d.get("status_code") != 0:
        (OUT_DIR / f"_debug_{dstr}_robot.json").write_text(resp.text[:3000], encoding="utf-8")
        raise RuntimeError(f"API {d.get('status_code')} {str(d.get('status_msg'))[:40]}")
    comps = d["data"]["answer"][0]["txt"][0]["content"]["components"]
    found_n = None
    txt = next((c for c in comps if c.get("show_type") == "txt2"), None)
    if txt:
        m = re.search(r"找到(\d+)", re.sub("<[^>]+>", "", txt["data"].get("content", "")))
        found_n = int(m.group(1)) if m else None
    tb = next((c for c in comps if str(c.get("show_type", "")).startswith("xuangu_table")), None)
    furl = ""
    if tb is not None:
        oi = (tb.get("config") or {}).get("other_info") or {}
        furl = (oi.get("footer_info") or {}).get("url", "")
    return found_n, furl


def _datalist_page(up, page, dstr):
    # form POST: 去掉 Content-Type: application/json, 否则 -8302
    headers, _ = fresh_headers()
    headers = {k: v for k, v in headers.items() if k != "Content-Type"}
    r = None
    last_err = None
    for attempt in range(3):  # 大页payload重, 瞬时超时重试
        try:
            r = requests.post(URL_LIST, data={**up, "perpage": 100, "page": page},
                              headers=headers, cookies=ck, timeout=(10, 60))
            break
        except requests.RequestException as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
    if r is None:
        raise RuntimeError(f"list 连接3败: {last_err} (date={dstr} list p{page})")
    if not r.text.lstrip().startswith("{"):
        (OUT_DIR / f"_debug_{dstr}_list_p{page}.html").write_text(
            f"HTTP {r.status_code}\n{r.text[:500]}", encoding="utf-8")
        raise RuntimeError(f"WAF 非JSON HTTP{r.status_code}")
    d = r.json()
    if str(d.get("status_code")) != "0":
        (OUT_DIR / f"_debug_{dstr}_list_p{page}.json").write_text(r.text[:3000], encoding="utf-8")
        raise RuntimeError(f"list API {d.get('status_code')} {str(d.get('status_msg'))[:40]}")
    comps = d.get("answer", {}).get("components", [])
    datas = comps[0].get("data", {}).get("datas", []) if comps else []
    return [_row_rec(row, dstr) for row in datas]


def fetch_day(dstr, question_tpl):
    """返回 (df, found_n, end_reason)."""
    found_n, furl = _robot_day(dstr, question_tpl)
    if found_n == 0 or (not furl and not found_n):
        return pd.DataFrame(), (found_n or 0), "no_table"
    if not furl:
        raise RuntimeError("robot 无 footer_url (found_n>0)")
    u = urlparse(furl if "://" in furl else "http://x.com" + (furl if furl.startswith("/") else "/" + furl))
    up = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
          for k, v in parse_qs(u.query).items()}
    rows, seen = [], set()
    end_reason = "max_pages"
    for page in range(1, MAX_PAGES + 1):
        recs = _datalist_page(up, page, dstr)
        if not recs:
            end_reason = "no_more"
            break
        new_cnt = 0
        for rec in recs:
            c = _norm_code(rec.get("股票代码", ""))
            if c and c not in seen:
                seen.add(c)
                rec["股票代码"] = c
                rows.append(rec)
                new_cnt += 1
        if found_n is not None and len(seen) >= found_n:
            end_reason = "complete"
            break
        if new_cnt == 0:
            end_reason = "cycled"
            break
        if len(recs) < 100:
            end_reason = "short_page"
            break
        time.sleep(0.8)
    df = pd.DataFrame(rows)
    return df, found_n, end_reason


QUESTION = "{dstr}看涨信号的股票"
print("[sweep v3] 全量扫描: 无审计行/不完整的日子一律重抓")
t0 = time.time()
done_cnt = 0
for i, d in enumerate(dates):
    dstr = d.strftime("%Y%m%d")
    out_f = OUT_DIR / f"bull_{dstr}.parquet"
    if out_f.exists() and _audit_valid(dstr):
        state["last_done"] = dstr
        continue
    try:
        df, found_n, end_reason = fetch_day(dstr, QUESTION)
        df.to_parquet(out_f)
        captured = int(df["股票代码"].nunique()) if len(df) else 0
        complete = ((found_n is not None and captured >= found_n)
                    or end_reason in ("short_page", "no_more", "no_table"))
        _append_audit(dstr, found_n, captured, end_reason, complete)
        state.update({"last_done": dstr, "consec_fail": 0})
        STATE.write_text(json.dumps(state), encoding="utf-8")
        done_cnt += 1
        flag = "" if complete else "  <<< INCOMPLETE"
        if done_cnt % 25 == 0 or not complete or captured == 0:
            el = time.time() - t0
            print(f"[{i+1}/{len(dates)}] {dstr}: 找到N={found_n} 抓到{captured} "
                  f"({end_reason}){flag} | 本轮{done_cnt}天 {el/60:.0f}min")
        time.sleep(2.0 + (i % 7) * 0.2)
    except Exception as e:
        state["consec_fail"] += 1
        state["fail_date"] = dstr
        STATE.write_text(json.dumps(state), encoding="utf-8")
        print(f"[FAIL] {dstr}: {type(e).__name__} {str(e)[:80]} consec={state['consec_fail']}")
        if "WAF" in str(e) or "403" in str(e):
            time.sleep(120)
        else:
            time.sleep(10)
        if state["consec_fail"] >= 8:
            print("[ABORT] 连续8失败, fail-fast退出")
            sys.exit(2)

n_complete = int(audit["complete"].sum()) if len(audit) else 0
print(f"[done] sweep完成, audit complete {n_complete}/{len(dates)}, 最后 {state['last_done']}")
if n_complete < len(dates):
    print(f"[WARN] 不完整日子: {audit[~audit['complete']]['date'].tolist()[:30]}")
print("SCRIPT-END")
