"""Backfill Tushare limit_list_d (涨跌停/连板) history → cached feature panel.

一次性数据基建: 抓近 N 个交易日的 limit_list_d, 构建每股每日涨跌停特征,
落盘 parquet (WORM-ish, 按截止日命名). 供强势股/连板延续研究脚本复用,
避免每次分析都重抓 API.

Base features (per stock×date):
  is_limit_up / is_limit_down / is_zhaban  涨停/跌停/炸板 标记 (0/1)
  limit_times                              连板高度 (Tushare 原始连板数)
  fd_amount_ratio                          封单金额/流通市值 (0~0.5 clip)
  open_times                               开板次数
  seal_mins                                首次封板时刻距 09:30 的分钟数 (NaN=非涨停或解析失败)

Usage: python scripts/_build_limit_features.py [N_dates]
"""

import os
import sys
import time

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import tushare as ts

from config import settings

N_DATES = int(sys.argv[1]) if len(sys.argv) > 1 else 250
PANEL = settings.PANEL_V3_PATH
pro = ts.pro_api(settings.TUSHARE_TOKEN or ts.get_token())

OUT_DIR = "data/factor_registry"
os.makedirs(OUT_DIR, exist_ok=True)


def _dates():
    d = sorted(
        pd.to_datetime(pd.read_parquet(PANEL, columns=["date"])["date"].unique())
    )
    return d[-N_DATES:]


def fetch_raw(dates, save_path=None, checkpoint_every=25):
    """抓取并每 checkpoint_every 天落盘一次 (可中断续传)."""
    frames, empty = [], []
    for i, d in enumerate(dates):
        ds = d.strftime("%Y%m%d")
        got = False
        for _ in range(2):  # 重试一次
            try:
                df = pro.limit_list_d(trade_date=ds)
                got = True
                break
            except Exception:
                time.sleep(0.3)
        if not got:
            empty.append(ds)
            continue
        if len(df):
            frames.append(df)
        if save_path and frames and (i + 1) % checkpoint_every == 0:
            pd.concat(frames, ignore_index=True).to_parquet(save_path)
            print(
                f"    checkpoint {i + 1}/{len(dates)} → {os.path.basename(save_path)}",
                flush=True,
            )
        if i % 50 == 0:
            print(f"    fetched {i}/{len(dates)} ({d.date()})", flush=True)
        time.sleep(0.12)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), empty


def build_features(raw):
    raw["is_limit_up"] = (raw["limit"] == "U").astype(float)
    raw["is_limit_down"] = (raw["limit"] == "D").astype(float)
    raw["is_zhaban"] = (raw["limit"] == "Z").astype(float)
    raw["limit_times"] = pd.to_numeric(raw["limit_times"], errors="coerce").fillna(0)
    raw["fd_amount_ratio"] = raw["fd_amount"] / raw["float_mv"].replace(0, np.nan)
    raw["fd_amount_ratio"] = raw["fd_amount_ratio"].fillna(0).clip(0, 0.5)
    raw["open_times"] = pd.to_numeric(raw["open_times"], errors="coerce").fillna(0)

    def _seal_mins(s):
        if isinstance(s, str) and len(s) >= 4:
            try:
                return int(s[:2]) * 60 + int(s[2:4]) - 570
            except ValueError:
                return np.nan
        return np.nan

    raw["seal_mins"] = raw["first_time"].map(_seal_mins)
    feat = raw[
        [
            "ts_code",
            "trade_date",
            "is_limit_up",
            "is_limit_down",
            "is_zhaban",
            "limit_times",
            "fd_amount_ratio",
            "open_times",
            "seal_mins",
        ]
    ].copy()
    feat["symbol"] = feat["ts_code"].str.replace(r"\.(SZ|SH|BJ)$", "", regex=True)
    feat["date"] = pd.to_datetime(feat["trade_date"], format="%Y%m%d")
    return feat[
        [
            "symbol",
            "date",
            "is_limit_up",
            "is_limit_down",
            "is_zhaban",
            "limit_times",
            "fd_amount_ratio",
            "open_times",
            "seal_mins",
        ]
    ]


def main():
    dates = _dates()
    end = dates[-1].strftime("%Y%m%d")
    print(
        f"[1] Backfill limit_list_d: {len(dates)} dates "
        f"({dates[0].date()} .. {dates[-1].date()})",
        flush=True,
    )

    raw_path = os.path.join(OUT_DIR, f"limit_list_d_raw_{end}.parquet")
    if os.path.exists(raw_path):
        raw = pd.read_parquet(raw_path)
        got = set(pd.to_datetime(raw["trade_date"], format="%Y%m%d"))
        missing = [d for d in dates if d not in got]
        if not missing:
            print(f"    cache hit: {raw_path} ({len(raw)} rows)")
        else:
            print(f"    cache partial ({len(missing)} missing), refetching missing...")
            extra, empty = fetch_raw(missing)
            raw = pd.concat([raw, extra], ignore_index=True)
            raw.to_parquet(raw_path)
    else:
        print(f"    fetching fresh (checkpoint to {os.path.basename(raw_path)})...")
        raw, empty = fetch_raw(dates, save_path=raw_path)
        raw.to_parquet(raw_path)
        if empty:
            print(f"    empty dates: {len(empty)} {empty[:5]}")

    feat = build_features(raw)
    print(
        f"[2] feature rows: {len(feat)}, dates: {feat['date'].nunique()}, "
        f"up: {(feat['is_limit_up'] == 1).sum()}, zhaban: {(feat['is_zhaban'] == 1).sum()}, "
        f"down: {(feat['is_limit_down'] == 1).sum()}",
        flush=True,
    )
    out = os.path.join(OUT_DIR, f"limit_feat_{end}.parquet")
    feat.to_parquet(out)
    print(f"[3] saved: {out}")
    print("DONE")


if __name__ == "__main__":
    main()
