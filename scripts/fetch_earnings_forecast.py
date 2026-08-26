# -*- coding: utf-8 -*-
"""业绩预告 (Tushare forecast) 回填 — ann_date 单日循环, 按年 WORM 落盘.

窗口默认: 今天往前整 3 年; 拉取顺序 = 最近一年优先 (新数据先落地).
按年写 data/supply_cache/alt_data/forecast/all_<start>_<end>.parquet,
已存在的年文件跳过 (断点续拉). 事件语义: 每条 = (ts_code, ann_date, end_date)
一次披露, 修正公告保留 (update_flag 区分), 去重仅限完全重复行.
"""
import os
import sys
import time
from datetime import date, timedelta

import pandas as pd
import tushare as ts

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from config.settings import TUSHARE_TOKEN  # noqa: E402

OUT_DIR = os.path.join("data", "supply_cache", "alt_data", "forecast")
FIELDS = (
    "ts_code,ann_date,end_date,type,p_change_min,p_change_max,"
    "net_profit_min,net_profit_max,update_flag"
)
SLEEP_S = 0.35
YEARS = 3


def year_windows(end: date, n: int) -> list[tuple[date, date]]:
    """最近一年优先, 从后往前切 n 个连续年窗."""
    wins = []
    for i in range(n):
        e = end - timedelta(days=365 * i)
        s = e - timedelta(days=365) + timedelta(days=1)
        wins.append((s, e))
    return wins


def fetch_window(pro, s: date, e: date) -> pd.DataFrame:
    frames, failed = [], []
    d = e
    while d >= s:
        ds = d.strftime("%Y%m%d")
        for attempt in range(2):
            try:
                df = pro.forecast(ann_date=ds, fields=FIELDS)
                if len(df):
                    frames.append(df)
                break
            except Exception as exc:
                if attempt == 0:
                    time.sleep(2.0)
                else:
                    failed.append(ds)
                    print(f"[warn] {ds}: {exc}", flush=True)
        time.sleep(SLEEP_S)
        d -= timedelta(days=1)
    if failed:
        print(f"[retry] {len(failed)} 天失败重试一轮", flush=True)
        for ds in failed:
            try:
                df = pro.forecast(ann_date=ds, fields=FIELDS)
                if len(df):
                    frames.append(df)
            except Exception as exc:
                print(f"[FAIL] {ds}: {exc}", flush=True)
            time.sleep(SLEEP_S)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    out["symbol"] = out["ts_code"].str[:6]
    return out


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    pro = ts.pro_api(TUSHARE_TOKEN)
    end = date.today()
    for s, e in year_windows(end, YEARS):
        tag = f"all_{s:%Y%m%d}_{e:%Y%m%d}"
        path = os.path.join(OUT_DIR, f"{tag}.parquet")
        if os.path.exists(path):
            print(f"[skip] {tag} 已存在", flush=True)
            continue
        print(f"[fetch] {tag} ({(e - s).days + 1} 天)", flush=True)
        t0 = time.time()
        df = fetch_window(pro, s, e)
        if df.empty:
            print(f"[warn] {tag} 0 行, 不落盘", flush=True)
            continue
        df.to_parquet(path, index=False)
        print(
            f"[done] {tag}: {len(df)} 行, {df['ts_code'].nunique()} 只, "
            f"{time.time() - t0:.0f}s -> {path}",
            flush=True,
        )
    print("[all done]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
