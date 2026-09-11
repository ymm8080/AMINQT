# -*- coding: utf-8 -*-
"""_ingest_5min.py — Tushare stk_mins 5分钟线批量落地 (2026-09-09 Ingest A).

宇宙: panel_full_enriched_v3.parquet symbol 列去重 (实际 5252 只, 非预估1780).
布局: D:/AMINQT/PARQUET/intraday/5min/{symbol}_5min.parquet
      — 与 app/core/intraday_loader.py 的 _cache_path(symbol, '5') 命名一致,
        列 = loader._OUTPUT_COLS = [datetime,open,high,low,close,volume,amount],
        未来 IntradayLoader(cache_dir=该目录) 可直接读。
WORM: 文件已存在即跳过 (断点续跑)。
分页: stk_mins 单次上限 8000 行且返回按时间倒序 (2026-09-09 实测),
      end_date 逐次前移 (oldest-1s) 直到取尽, 每股通常 2 页/年。
限流: 500次/min → 每次调用间隔 >=0.16s; 异常指数退避重试最多5次。
校验: high>=low, high>=open/close, low<=open/close, volume>=0, OHLCV 非 NaN;
      物理不可能行剔除并逐行记日志计数 (不静默丢弃)。
终态: stdout 末行 INGEST_5MIN_DONE status=ok|partial|aborted
      + tmp_min/_ingest_5min_summary.json
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta

ROOT = r"D:\AMINQT\AMINQT CODES"
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

PANEL_PATH = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
OUT_DIR = r"D:\AMINQT\PARQUET\intraday\5min"
TMP_MIN = os.path.join(ROOT, "tmp_min")
FAIL_FILE = os.path.join(TMP_MIN, "_ingest_5min_failures.txt")
SUMMARY_FILE = os.path.join(TMP_MIN, "_ingest_5min_summary.json")

START = "2025-09-09 00:00:00"   # 最近1年 (今日 2026-09-09 盘前, 无当日数据)
END = "2026-09-08 23:59:59"     # 最后一个已完结交易日

FIELDS = "ts_code,trade_time,open,high,low,close,vol,amount"  # 显式 fields (项目惯例)
THROTTLE = 0.16
PAGE_CAP = 8000
MAX_RETRY = 5
MAX_PAGES_PER_STOCK = 6
PROGRESS_EVERY = 25
MILESTONE_EVERY = 500

OUT_COLS = ["datetime", "open", "high", "low", "close", "volume", "amount"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def to_ts_code(sym: str) -> str:
    if "." in sym:
        return sym
    if sym.startswith("6"):
        return sym + ".SH"
    if sym.startswith(("0", "3")):
        return sym + ".SZ"
    if sym.startswith(("4", "8", "9")):
        return sym + ".BJ"
    return sym


def load_universe() -> list:
    t = pq.read_table(PANEL_PATH, columns=["symbol"])
    syms = sorted(set(t.to_pandas()["symbol"].astype(str)))
    return syms


def call_with_retry(pro, **kwargs) -> pd.DataFrame:
    last_exc = None
    for attempt in range(MAX_RETRY):
        try:
            df = pro.stk_mins(**kwargs)
            time.sleep(THROTTLE)
            if df is None:
                return pd.DataFrame()
            return df
        except Exception as exc:  # 网络/限流退避
            last_exc = exc
            wait = 2 ** attempt
            log(f"  retry {attempt + 1}/{MAX_RETRY} after {wait}s: {type(exc).__name__}: {exc}")
            time.sleep(wait)
    raise RuntimeError(f"stk_mins 连续失败 x{MAX_RETRY}: {last_exc}")


def fetch_stock(pro, ts_code: str) -> pd.DataFrame:
    """倒序分页取整窗 [START, end], 返回原始行 (未清洗)."""
    pages = []
    end = END
    for _ in range(MAX_PAGES_PER_STOCK):
        df = call_with_retry(
            pro, ts_code=ts_code, freq="5min",
            start_date=START, end_date=end, fields=FIELDS,
        )
        if len(df) == 0:
            break
        pages.append(df)
        if len(df) < PAGE_CAP:
            break
        oldest = pd.to_datetime(df["trade_time"]).min()
        end = (oldest - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
        if end < START:
            break
    if not pages:
        return pd.DataFrame()
    raw = pd.concat(pages, ignore_index=True)
    return raw.drop_duplicates(subset=["ts_code", "trade_time"]).reset_index(drop=True)


def validate(raw: pd.DataFrame, ts_code: str, stats: dict) -> pd.DataFrame:
    """OHLCV 物理校验; 不可能行逐行记日志并剔除, 返回干净帧 (loader 列)."""
    df = raw.rename(columns={"trade_time": "datetime", "vol": "volume"})
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["amount"] = pd.to_numeric(df.get("amount"), errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"])

    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    bad = (
        o.isna() | h.isna() | l.isna() | c.isna() | v.isna()
        | (h < l) | (h < o) | (h < c) | (l > o) | (l > c) | (v < 0)
    )
    n_bad = int(bad.sum())
    if n_bad:
        for idx, row in df[bad].iterrows():
            why = []
            if row[["open", "high", "low", "close", "volume"]].isna().any():
                why.append("nan")
            if row["high"] < row["low"]:
                why.append("high<low")
            if row["high"] < row["open"] or row["high"] < row["close"]:
                why.append("high<oc")
            if row["low"] > row["open"] or row["low"] > row["close"]:
                why.append("low>oc")
            if row["volume"] < 0:
                why.append("vol<0")
            log(f"  INVALID {ts_code} {row['datetime']} reasons={','.join(why)}")
        df = df[~bad]
        stats["invalid_dropped"] += n_bad
    return df[OUT_COLS].sort_values("datetime").reset_index(drop=True)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TMP_MIN, exist_ok=True)

    # 按项目惯例: 重活前查 HEAVY_SENTINELS 活进程冲突 (fail-fast)
    from scripts._run_guard import find_conflicts
    conflicts = find_conflicts()
    if conflicts:
        log(f"FATAL: 重活冲突 {conflicts}, 中止")
        log("INGEST_5MIN_DONE status=aborted")
        return

    from dotenv import load_dotenv
    load_dotenv()
    import tushare as ts
    token = os.getenv("TUSHARE_TOKEN") or ts.get_token()
    if not token:
        log("FATAL: 无 Tushare token")
        log("INGEST_5MIN_DONE status=aborted")
        return
    pro = ts.pro_api(token)

    syms = load_universe()
    log(f"universe: {len(syms)} 只; window {START} ~ {END}")

    stats = {
        "n_total": len(syms), "n_ok": 0, "n_skipped": 0, "n_empty": 0,
        "n_fail": 0, "rows_written": 0, "invalid_dropped": 0,
        "bytes_written": 0, "dt_min": None, "dt_max": None,
    }
    t0 = time.time()
    fails = []
    for i, sym in enumerate(syms, 1):
        path = os.path.join(OUT_DIR, f"{sym}_5min.parquet")
        if os.path.exists(path):
            stats["n_skipped"] += 1
        else:
            ts_code = to_ts_code(sym)
            try:
                raw = fetch_stock(pro, ts_code)
                if len(raw) == 0:
                    stats["n_empty"] += 1
                else:
                    clean = validate(raw, ts_code, stats)
                    if len(clean) == 0:
                        stats["n_empty"] += 1
                    else:
                        clean.to_parquet(path, index=False)
                        stats["n_ok"] += 1
                        stats["rows_written"] += len(clean)
                        stats["bytes_written"] += os.path.getsize(path)
                        dmin, dmax = str(clean["datetime"].min()), str(clean["datetime"].max())
                        if stats["dt_min"] is None or dmin < stats["dt_min"]:
                            stats["dt_min"] = dmin
                        if stats["dt_max"] is None or dmax > stats["dt_max"]:
                            stats["dt_max"] = dmax
            except Exception as exc:
                stats["n_fail"] += 1
                reason = f"{type(exc).__name__}: {exc}"[:200]
                fails.append(f"{ts_code}|{reason}")
                log(f"  FAIL {ts_code}: {reason}")
        if i % PROGRESS_EVERY == 0 or i == len(syms):
            el = time.time() - t0
            log(f"progress {i}/{len(syms)} ok={stats['n_ok']} skip={stats['n_skipped']} "
                f"empty={stats['n_empty']} fail={stats['n_fail']} rows={stats['rows_written']} "
                f"elapsed={el / 60:.1f}min")
        if i % MILESTONE_EVERY == 0:
            log(f"MILESTONE 5min {i}/{len(syms)} fail={stats['n_fail']}")

    with open(FAIL_FILE, "a", encoding="utf-8") as f:
        for line in fails:
            f.write(line + "\n")
    stats["elapsed_min"] = round((time.time() - t0) / 60, 1)
    stats["failures_file"] = FAIL_FILE if fails else None
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log(f"SUMMARY {json.dumps(stats, ensure_ascii=False)}")
    log("INGEST_5MIN_DONE status=" + ("ok" if stats["n_fail"] == 0 else "partial"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("INGEST_5MIN_DONE status=aborted", flush=True)
