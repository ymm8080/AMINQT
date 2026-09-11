# -*- coding: utf-8 -*-
"""baostock 5min 全量回补 (2026-09-09, 探针 50/50 PASS 后用户路径确认).

范围: V3 宇宙 (面板近期活跃 symbol) × 1年 (2025-09-09~2026-09-08).
质量探针: tmp_min/_probe_baostock_5min.py — 48bar/日满覆盖, 聚合amount
对照Tushare真值 100% 日 1.5% 内一致, volume 股单位 (÷100=手).
WORM: data/intraday/baostock_5min/bs5_{symbol}.parquet 逐股落盘, 重跑跳过已存在.
串行单会话 (baostock 限制), 预计 15-40h, 可断点续传.
终态: INGEST_BS5_DONE status=ok|partial|aborted + tmp_min/_ingest_bs5_summary.json
"""
import json
import os
import sys
import time
import traceback
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq

ROOT = r"D:\AMINQT\AMINQT CODES"
sys.path.insert(0, ROOT)
PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
OUTDIR = os.path.join(ROOT, "data", "intraday", "baostock_5min")
TMP_MIN = os.path.join(ROOT, "tmp_min")
SUMMARY = os.path.join(TMP_MIN, "_ingest_bs5_summary.json")
FAIL_FILE = os.path.join(TMP_MIN, "_ingest_bs5_failures.txt")
START, END = "2025-09-09", "2026-09-08"
FIELDS = "date,time,code,open,high,low,close,volume,amount"
MAX_RETRY = 3


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    from scripts._run_guard import find_conflicts
    # 多日回补不抢跑: 只等前驱 (ohlc修复→ovd回补→cyq重算), 全量哨兵会互等死锁
    PRED = ("_ohlc_repair_v3_0909.py", "_ovd_backfill_v3_0909.py",
            "_cyq_recompute_v3_0909.py")
    conflicts = find_conflicts(sentinels=PRED)
    attempt = 0
    while conflicts and attempt < 60:
        sens = [c.get("sentinel", "?") for c in conflicts]
        log(f"run-guard 冲突, 等10min重试 ({attempt + 1}/60): {sens}")
        time.sleep(600)
        attempt += 1
        conflicts = find_conflicts(sentinels=PRED)
    if conflicts:
        log(f"FATAL run-guard 持续冲突 10h: {conflicts}")
        log("INGEST_BS5_DONE status=aborted")
        return

    os.makedirs(OUTDIR, exist_ok=True)
    syms = pq.read_table(PANEL, columns=["symbol", "date"],
                         filters=[("date", ">=", pd.Timestamp("2026-08-01"))]
                         ).to_pandas()["symbol"].unique()
    # baostock 无北交所: 剔 4/8/9 开头; 6->sh 其余->sz
    syms = sorted(s for s in syms
                  if len(s) == 6 and s[0] in "0236" and not s.startswith("20"))
    log(f"universe={len(syms)} window {START}~{END}")

    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        log(f"FATAL login: {lg.error_msg}")
        log("INGEST_BS5_DONE status=aborted")
        return

    stats = {"n_total": len(syms), "n_done": 0, "n_skip": 0,
             "n_fail": 0, "rows_written": 0}
    fails = []
    t0 = time.time()
    try:
        for i, sym in enumerate(syms, 1):
            path = os.path.join(OUTDIR, f"bs5_{sym}.parquet")
            if os.path.exists(path):
                stats["n_skip"] += 1
                continue
            code = ("sh." if sym.startswith("6") else "sz.") + sym
            df = None
            for attempt in range(MAX_RETRY):
                try:
                    rs = bs.query_history_k_data_plus(
                        code, FIELDS, start_date=START, end_date=END,
                        frequency="5", adjustflag="3")
                    if rs.error_code != "0":
                        raise RuntimeError(rs.error_msg[:100])
                    rows = []
                    while rs.next():
                        rows.append(rs.get_row_data())
                    df = pd.DataFrame(rows, columns=FIELDS.split(","))
                    break
                except Exception as exc:
                    log(f"  retry {attempt+1}/{MAX_RETRY} {sym}: "
                        f"{type(exc).__name__}: {str(exc)[:80]}")
                    time.sleep(2 ** attempt)
            if df is None:
                stats["n_fail"] += 1
                fails.append(f"{sym}|consecutive_fail")
                continue
            if len(df) == 0:
                # 空也落盘占位 (WORM 标记已取, 避免反复重打)
                df.to_parquet(path, index=False)
                stats["n_done"] += 1
                continue
            for c in ("open", "high", "low", "close", "volume", "amount"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["symbol"] = sym
            df.to_parquet(path, index=False)
            stats["n_done"] += 1
            stats["rows_written"] += len(df)
            if (stats["n_done"] + stats["n_skip"]) % 100 == 0:
                el = time.time() - t0
                done_n = stats["n_done"] + stats["n_skip"]
                eta_h = el / max(done_n - stats["n_skip"], 1) \
                    * (len(syms) - done_n) / 3600
                log(f"MILESTONE {done_n}/{len(syms)} fail={stats['n_fail']} "
                    f"rows={stats['rows_written']:,} elapsed={el/3600:.1f}h "
                    f"eta={eta_h:.1f}h")
    except Exception:
        traceback.print_exc()
        log("INGEST_BS5_DONE status=aborted")
        return
    finally:
        try:
            bs.logout()
        except Exception:
            pass

    with open(FAIL_FILE, "a", encoding="utf-8") as f:
        for line in fails:
            f.write(line + "\n")
    stats["elapsed_h"] = round((time.time() - t0) / 3600, 2)
    json.dump(stats, open(SUMMARY, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    log(f"SUMMARY {json.dumps(stats, ensure_ascii=False)}")
    log("INGEST_BS5_DONE status=" + ("ok" if stats["n_fail"] == 0 else "partial"))


if __name__ == "__main__":
    main()
