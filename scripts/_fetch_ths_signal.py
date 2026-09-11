# -*- coding: utf-8 -*-
"""THS问财看涨/看跌信号 每日抓取 (daily automation step).

用法:
  python scripts/_fetch_ths_signal.py                # 抓 T-1 交易日 (默认)
  python scripts/_fetch_ths_signal.py --date 20260910

行为:
  - 看涨池全量(分页+代码去重早停) -> data/supply_cache/ths_signal/bull_{date}.parquet (WORM, 存在即跳过)
  - 看跌池个股级全量(perpage=100) -> data/supply_cache/ths_signal/bear_{date}.parquet (WORM)
  - 看跌池规模 -> data/supply_cache/ths_signal/bear_counts.csv (追加, 按日期去重)
  - cookie 失效 -> exit 2 (fail-fast, 需刷新 cookies.json)
cookie 来源: 用真实Edge挂真实profile导出 (tmp_t/_ths_v8_0910.py 的方法),
输出重定向到 data/supply_cache/ths_signal/cookies.json (已gitignore).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
import urllib3.util.connection as u3c

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "supply_cache" / "ths_signal"
COOKIES_F = CACHE_DIR / "cookies.json"
BEAR_CSV = CACHE_DIR / "bear_counts.csv"

# 本地ISP DNS把THS域名指到死WAF, 固定走已验证CDN IP (20260910实测)
IP_IWENCAI = "121.12.127.73"
_orig_create_connection = u3c.create_connection


def _fixed_create_connection(address, *a, **kw):
    host, port = address
    if host == "www.iwencai.com":
        address = (IP_IWENCAI, port)
    return _orig_create_connection(address, *a, **kw)


u3c.create_connection = _fixed_create_connection

import pywencai  # noqa: E402  (须在monkey-patch之后)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


def _headers_with_token():
    hexin = pywencai.headers.headers()["hexin-v"]
    headers = {"User-Agent": UA, "hexin-v": hexin, "Content-Type": "application/json",
               "Referer": "https://www.iwencai.com/unifiedwap/result/get-robot-data",
               "Origin": "https://www.iwencai.com"}
    return headers, round(random.random(), 6)


def _post(question: str, page: int, ck: dict, perpage: int = 10) -> dict:
    headers, rval = _headers_with_token()
    params = {"question": question, "secondary_intent": "stock", "perpage": perpage,
              "page": page, "block_list": "", "add_info": "1", "r": rval,
              "source": "Ths_iwencai_Xuangu", "version": "2.0",
              "query_type": "stock", "hexin-v": headers["hexin-v"]}
    resp = requests.post("http://www.iwencai.com/customized/chart/get-robot-data",
                         params=params, json={}, headers=headers, cookies=ck, timeout=25)
    return resp.json()


def _extract_table(d: dict):
    comps = d["data"]["answer"][0]["txt"][0]["content"]["components"]
    tb = next((c for c in comps if c.get("show_type", "").startswith("xuangu_table")
               and c.get("data", {}).get("datas")), None)
    if tb is None:
        return None
    cols = tb["data"]["columns"]
    out = []
    for row in tb["data"]["datas"]:
        rec = {}
        for col in cols:
            key = col.get("key") or col.get("title")
            cell = row.get(key, "")
            rec[key] = cell.get("value", cell) if isinstance(cell, dict) else cell
        out.append(rec)
    return out


def _find_total(d: dict):
    comps = d["data"]["answer"][0]["txt"][0]["content"]["components"]
    txt = next((c for c in comps if c.get("show_type") == "txt2"), None)
    if not txt:
        return None
    plain = re.sub("<[^>]+>", "", txt["data"].get("content", ""))
    m = re.search(r"找到(\d+)", plain)
    return int(m.group(1)) if m else None


def _validate_date_suffix(d: dict, dstr: str) -> None:
    """THS对非交易日查询会静默回退到别的日期 (20260501→[20260430],
    20260912→[20250912]), 用返回列的日期后缀守门, 不匹配=大声失败不落盘."""
    comps = d["data"]["answer"][0]["txt"][0]["content"]["components"]
    for c in comps:
        for col in (c.get("data", {}) or {}).get("columns", []):
            m = re.search(r"\[(\d{8})\]", col.get("title", "") + col.get("key", ""))
            if m and m.group(1) != dstr:
                raise RuntimeError(
                    f"THS日期回退: 请求{dstr} 实际[{m.group(1)}] (非交易日?) — 不落盘")


def _paginate_pool(question: str, dstr: str, ck: dict, perpage: int, max_pages: int) -> pd.DataFrame:
    """v3真分页: robot-data(body风格)拿 footer_info.url → getDataList (form) perpage=100 真生效.

    沿革: 旧版对 get-robot-data 传 page=N, 但该端点 page 参数被无视(永远第1页),
    perpage>10 也被忽略 → 池子>10 必截断 (20230904 实测 找到58 只抓10).
    pywencai 源码逆向出的 getDataList 是唯一真分页通道 (20230910 探针 58/58 一页全中).
    """
    # ① robot 会话: found_n + footer_info.url
    hv_headers = _headers_with_token()[0]
    body = {"add_info": '{"urp":{"scene":1,"company":1,"business":1},'
                        '"contentType":"json","searchInfo":true}',
            "perpage": "10", "page": 1, "source": "Ths_iwencai_Xuangu",
            "log_info": '{"input_type":"click"}', "version": "2.0",
            "secondary_intent": "stock", "question": question}
    resp = requests.post("http://www.iwencai.com/customized/chart/get-robot-data",
                         json=body, headers=hv_headers, cookies=ck, timeout=25)
    if not resp.text.lstrip().startswith("{"):
        raise RuntimeError(f"WAF 非JSON HTTP{resp.status_code} (date={dstr} robot)")
    d = resp.json()
    if d.get("status_code") != 0:
        raise RuntimeError(f"API status={d.get('status_code')} "
                           f"{str(d.get('status_msg'))[:60]} (date={dstr} robot)")
    _validate_date_suffix(d, dstr)
    total = _find_total(d)
    comps = d["data"]["answer"][0]["txt"][0]["content"]["components"]
    tb = next((c for c in comps if str(c.get("show_type", "")).startswith("xuangu_table")), None)
    furl = ""
    if tb is not None:
        oi = (tb.get("config") or {}).get("other_info") or {}
        furl = (oi.get("footer_info") or {}).get("url", "")
    if total == 0 or (not furl and not total):
        return pd.DataFrame()
    if not furl:
        raise RuntimeError(f"robot 无 footer_info.url (found_n={total}, date={dstr}) — 不落盘")
    u = urlparse(furl if "://" in furl else "http://x.com" + (furl if furl.startswith("/") else "/" + furl))
    up = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v)
          for k, v in parse_qs(u.query).items()}

    # ② getDataList 分页 (perpage=100 真生效)
    rows, seen = [], set()
    for page in range(1, max_pages + 1):
        # getDataList 是 form POST: 必须去掉 robot 调用的 Content-Type: application/json,
        # 否则表单体被声明成 JSON → 服务端 -8302 (20230910 实测)
        list_headers = {k: v for k, v in _headers_with_token()[0].items()
                        if k != "Content-Type"}
        lr = None
        last_err = None
        for attempt in range(3):  # 大页payload重, 瞬时超时重试; 鉴权/业务错仍大声失败
            try:
                lr = requests.post("http://www.iwencai.com/gateway/urp/v7/landing/getDataList",
                                   data={**up, "perpage": perpage, "page": page},
                                   headers=list_headers, cookies=ck, timeout=(10, 60))
                break
            except requests.RequestException as e:
                last_err = e
                time.sleep(5 * (attempt + 1))
        if lr is None:
            raise RuntimeError(f"list 连接3败: {last_err} (date={dstr} list p{page})")
        if not lr.text.lstrip().startswith("{"):
            raise RuntimeError(f"WAF 非JSON HTTP{lr.status_code} (date={dstr} list p{page})")
        ld = lr.json()
        if str(ld.get("status_code")) != "0":
            raise RuntimeError(f"list API status={ld.get('status_code')} "
                               f"{str(ld.get('status_msg'))[:60]} (date={dstr} list p{page})")
        lcomps = ld.get("answer", {}).get("components", [])
        datas = lcomps[0].get("data", {}).get("datas", []) if lcomps else []
        if not datas:
            break
        new_cnt = 0
        for row in datas:
            rec = {}
            for k, cell in row.items():
                rec[k] = cell.get("value", cell) if isinstance(cell, dict) else cell
            code = re.sub(r"\.\w+$", "", str(rec.get("股票代码", "")).strip())
            if code and code not in seen:
                seen.add(code)
                rec["股票代码"] = code
                rec["_query_date"] = dstr
                rows.append(rec)
                new_cnt += 1
        if total is not None and len(seen) >= total:
            break
        if new_cnt == 0:
            break  # 服务端循环重复
        if len(datas) < perpage:
            break
        time.sleep(0.3)
    df = pd.DataFrame(rows)
    if total is not None and len(seen) < total:
        raise RuntimeError(f"分页不完整: 找到{total} 抓到{len(seen)} (date={dstr}) — 不落盘")
    return df


def fetch_bull_pool(dstr: str, ck: dict) -> pd.DataFrame:
    return _paginate_pool(f"{dstr}看涨信号的股票", dstr, ck, perpage=100, max_pages=40)


def fetch_bear_pool(dstr: str, ck: dict) -> pd.DataFrame:
    """看跌池~2700只/日 (median 2869, max 5096), perpage=100 → ~28页; 只留代码列 (旗标语义).
    完整性由 _paginate_pool 内部 找到N vs 抓到数 守门, 不齐即 raise 不落盘."""
    df = _paginate_pool(f"{dstr}看跌信号的股票", dstr, ck, perpage=100, max_pages=60)
    if len(df) == 0:
        return df
    return df[["股票代码"] + [c for c in ("股票简称", "最新涨跌幅") if c in df.columns]]


def prev_trading_day(dstr: str) -> str:
    """T-1交易日: 从本地面板日历回退"""
    pf = pd.read_parquet("D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet",
                         columns=["date"])
    dates = sorted(pd.to_datetime(pf["date"].unique()))
    del pf
    target = pd.Timestamp(dstr)
    prev = [d for d in dates if d < target]
    if not prev:
        raise SystemExit(f"[abort] 日历中无 {dstr} 之前的交易日")
    return prev[-1].strftime("%Y%m%d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="YYYYMMDD (默认=T-1交易日)")
    args = ap.parse_args()

    if not COOKIES_F.exists():
        print(f"[FATAL] 缺cookie: {COOKIES_F}\n"
              f"刷新方法: 杀msedge -> python tmp_t/_ths_v8_0910.py (输出改写到cookies.json)")
        sys.exit(2)
    cookies = json.load(open(COOKIES_F, encoding="utf-8"))
    ck = {c["name"]: c["value"] for c in cookies
          if ("iwencai" in c["domain"] or "10jqka" in c["domain"])}

    # 默认=今日 (链在20:15收盘后跑, 当日session已完成); 周末回退上一交易日.
    # 节假日(工作日非session)由 _validate_date_suffix 守门, 大声失败不落盘.
    today = time.strftime("%Y%m%d")
    if args.date:
        dstr = args.date
    elif time.strptime(today, "%Y%m%d").tm_wday >= 5:
        dstr = prev_trading_day(today)
    else:
        dstr = today
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for label, fetcher in (("看涨池", fetch_bull_pool), ("看跌池", fetch_bear_pool)):
        out_f = CACHE_DIR / f"{'bull' if label == '看涨池' else 'bear'}_{dstr}.parquet"
        if out_f.exists():
            print(f"[skip] {out_f.name} 已存在 (WORM)")
            continue
        try:
            df = fetcher(dstr, ck)
        except RuntimeError as e:
            print(f"[FATAL] {e}\n若为鉴权类失败(403/-1091连续), 刷新cookies.json后重试")
            sys.exit(2)
        df.to_parquet(out_f)
        print(f"[ok] {dstr} {label} {len(df)}行 -> {out_f.name}")
        time.sleep(1.0)

    # 看跌池规模
    d = _post(f"{dstr}看跌信号的股票", 1, ck)
    bcount = _find_total(d) if d.get("status_code") == 0 else None
    row = {"date": dstr, "bear_count": bcount}
    if BEAR_CSV.exists():
        hist = pd.read_csv(BEAR_CSV)
        if dstr not in set(hist["date"].astype(str)):
            hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
            hist.to_csv(BEAR_CSV, index=False)
    else:
        pd.DataFrame([row]).to_csv(BEAR_CSV, index=False)
    print(f"[ok] {dstr} 看跌池 {bcount}")
    print("SCRIPT-END")


if __name__ == "__main__":
    main()
