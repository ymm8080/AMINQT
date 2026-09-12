# -*- coding: utf-8 -*-
"""_ingest_moneyflow.py — Tushare moneyflow 3年日频回补 (2026-09-09 Ingest B).

沿用既有日快照缓存: data/supply_cache/alt_data/moneyflow_daily/mf_YYYYMMDD.parquet
(与 _diag_moneyflow_hold.py 管线同目录同格式; 列=Tushare moneyflow 全20列)
WORM: 已存在日不覆盖; 当日返回空则不落文件 (下次可补)。
交易日历: pro.trade_cal SSE; 显式 fields; 限流退避同 A。
终态: INGEST_MF_DONE status=ok|partial|aborted + tmp_min/_ingest_mf_summary.json
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime

ROOT = r"D:\AMINQT\AMINQT CODES"
sys.path.insert(0, ROOT)

CACHE_DIR = os.path.join(ROOT, "data", "supply_cache", "alt_data", "moneyflow_daily")
TMP_MIN = os.path.join(ROOT, "tmp_min")
FAIL_FILE = os.path.join(TMP_MIN, "_ingest_mf_failures.txt")
SUMMARY_FILE = os.path.join(TMP_MIN, "_ingest_mf_summary.json")

CAL_START, CAL_END = "20230909", "20260908"  # 3年, 止于最后完结交易日
MF_FIELDS = (
    "ts_code,trade_date,buy_sm_vol,buy_sm_amount,sell_sm_vol,sell_sm_amount,"
    "buy_md_vol,buy_md_amount,sell_md_vol,sell_md_amount,"
    "buy_lg_vol,buy_lg_amount,sell_lg_vol,sell_lg_amount,"
    "buy_elg_vol,buy_elg_amount,sell_elg_vol,sell_elg_amount,"
    "net_mf_vol,net_mf_amount"
)
THROTTLE = 0.35  # 沿用项目 moneyflow 限流惯例 (_diag_moneyflow_hold.py)
MAX_RETRY = 5


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def call_with_retry(fn, **kwargs):
    last_exc = None
    for attempt in range(MAX_RETRY):
        try:
            df = fn(**kwargs)
            time.sleep(THROTTLE)
            return df
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            log(f"  retry {attempt + 1}/{MAX_RETRY} after {wait}s: {type(exc).__name__}: {exc}")
            time.sleep(wait)
    raise RuntimeError(f"调用连续失败 x{MAX_RETRY}: {last_exc}")


def main() -> None:
    from scripts._run_guard import find_conflicts
    conflicts = find_conflicts()
    if conflicts:
        log(f"FATAL: 重活冲突 {conflicts}, 中止")
        log("INGEST_MF_DONE status=aborted")
        return

    from dotenv import load_dotenv
    load_dotenv()
    import tushare as ts
    token = os.getenv("TUSHARE_TOKEN") or ts.get_token()
    if not token:
        log("FATAL: 无 Tushare token")
        log("INGEST_MF_DONE status=aborted")
        return
    pro = ts.pro_api(token)

    cal = call_with_retry(
        pro.trade_cal, exchange="SSE", start_date=CAL_START, end_date=CAL_END,
        is_open="1", fields="cal_date,is_open",
    )
    days = sorted(cal["cal_date"].astype(str))
    log(f"trade days {days[0]}~{days[-1]}: {len(days)}")

    stats = {"n_days": len(days), "n_fetched": 0, "n_skipped_exist": 0,
             "n_empty": 0, "n_fail": 0, "rows_written": 0}
    t0 = time.time()
    fails = []
    for i, d in enumerate(days, 1):
        path = os.path.join(CACHE_DIR, f"mf_{d}.parquet")
        if os.path.exists(path):
            stats["n_skipped_exist"] += 1
            continue
        try:
            df = call_with_retry(pro.moneyflow, trade_date=d, fields=MF_FIELDS)
            if df is None or len(df) == 0:
                stats["n_empty"] += 1  # 不落空文件, 下次可补
            else:
                cols = MF_FIELDS.split(",")
                df = df[cols]
                df.to_parquet(path, index=False)
                stats["n_fetched"] += 1
                stats["rows_written"] += len(df)
        except Exception as exc:
            stats["n_fail"] += 1
            reason = f"{type(exc).__name__}: {exc}"[:200]
            fails.append(f"mf_{d}|{reason}")
            log(f"  FAIL {d}: {reason}")
        if i % 50 == 0 or i == len(days):
            log(f"progress {i}/{len(days)} fetched={stats['n_fetched']} "
                f"skip={stats['n_skipped_exist']} empty={stats['n_empty']} "
                f"fail={stats['n_fail']} elapsed={(time.time() - t0) / 60:.1f}min")
        if i % 200 == 0:
            log(f"MILESTONE mf {i}/{len(days)} fail={stats['n_fail']}")

    with open(FAIL_FILE, "a", encoding="utf-8") as f:
        for line in fails:
            f.write(line + "\n")
    stats["elapsed_min"] = round((time.time() - t0) / 60, 1)
    stats["failures_file"] = FAIL_FILE if fails else None
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    log(f"SUMMARY {json.dumps(stats, ensure_ascii=False)}")
    log("INGEST_MF_DONE status=" + ("ok" if stats["n_fail"] == 0 else "partial"))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("INGEST_MF_DONE status=aborted", flush=True)
