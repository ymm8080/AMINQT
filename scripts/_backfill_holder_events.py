#!/usr/bin/env python3
"""_backfill_holder_events.py — 回填 V3 面板 holdertrade 事件列缺口 (2026-08-14).

根因: _daily_fetch 长期不抓 stk_holdertrade → GLM 4 列 (sh_net_change_sign /
sh_change_amt_total / sh_change_vol / sh_net_sign) 每日 ffill 续旧值 (新公告被
旧事件值盖住), sh_evt_start/end_date 停在 08-11 (最后一次面板重建), sh_net/g/p/c_ratio
停在 08-03 (一次性回填快照日). 08-04..08-14 公告日事件 10 列缺失或陈旧.

做法: stk_holdertrade 07-25..08-14 窗口 (含迟发公告余量) → agg_holdertrade_daily
按 (symbol, ann_date) 聚合 (语义同 panel_builder agg_map + _backfill_holder_ratio) →
覆盖写回面板 08-04..14 事件日行 (WORM 备份 + row-group 流式重写, 同
_backfill_margin_t1). 非事件行不动: GLM 列 ffill 语义保留, evt/ratio 列稀疏语义保留.

用法:
    python scripts/_backfill_holder_events.py [--dry-run]
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline1.data_supply import DataSupplyChain  # noqa: E402
from app.pipeline1.holdertrade_agg import (  # noqa: E402
    HOLDER_PANEL_COLS,
    agg_holdertrade_daily,
)
from config.settings import PANEL_V3_PATH  # noqa: E402

FETCH_START = dt.date(2026, 7, 25)  # 拉取窗口起点 (迟发公告余量)
FETCH_END = dt.date(2026, 8, 14)
WRITE_START = dt.date(2026, 8, 4)  # 写回起点: ratio 列缺口自 08-04 (此前各列已全)
MTIME_GRACE_S = 1800  # 面板 mtime 距今 <30min → 疑似并发写入, 中止
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _ohlcv_violations(df: pd.DataFrame) -> int:
    """OHLCV 数据校验铁律: high>=low, high>=open/close, low<=open/close, volume>=0."""
    if not set(OHLCV_COLS).issubset(df.columns):
        return -1
    return int(
        (
            (df["high"] < df["low"])
            | (df["high"] < df[["open", "close"]].max(axis=1))
            | (df["low"] > df[["open", "close"]].min(axis=1))
            | (df["volume"] < 0)
        ).sum()
    )


def fetch_holder_with_retry() -> pd.DataFrame:
    """stk_holdertrade 窗口拉取, 网络抖动重试 (同 _backfill_margin_t1)."""
    supply = DataSupplyChain()
    for attempt in range(4):
        try:
            raw = supply.fetch_holdertrade(
                start_date=FETCH_START.strftime("%Y%m%d"),
                end_date=FETCH_END.strftime("%Y%m%d"),
                refresh=True,
            )
            if len(raw):
                return raw
            print(f"  attempt {attempt + 1}/4: 空数据, 重试")
        except Exception as exc:
            print(f"  attempt {attempt + 1}/4 failed: {exc!r}")
        time.sleep(5 * (attempt + 1))
    return pd.DataFrame()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只打印计划不写盘")
    args = ap.parse_args()

    if not PANEL_V3_PATH.exists():
        print(f"FATAL: 面板不存在 {PANEL_V3_PATH}")
        return 1

    # ── 0. 并发写入防护: 面板 mtime 过新则中止 ──
    age = time.time() - os.path.getmtime(PANEL_V3_PATH)
    if age < MTIME_GRACE_S and os.environ.get("HOLDER_BF_SKIP_MTIME_GUARD") != "1":
        print(
            f"FATAL: 面板 mtime 距今 {age:.0f}s (<{MTIME_GRACE_S}s), 疑似并发写入, 中止"
        )
        return 1

    # ── 1. 拉取 + 聚合 ──
    print(f"拉取 stk_holdertrade {FETCH_START}..{FETCH_END} ...")
    raw = fetch_holder_with_retry()
    if not len(raw):
        print("FATAL: 无任何 holdertrade 数据可取")
        return 1
    daily = agg_holdertrade_daily(raw)
    daily = daily[daily["date"].dt.date >= WRITE_START].reset_index(drop=True)
    print(
        f"事件 {len(raw)} 条 → 日聚合 {len(daily)} 行 (ann_date ≥ {WRITE_START}):\n"
        + daily.groupby(daily["date"].dt.date).size().to_string()
    )
    print(
        "holder_type 分布:",
        raw.get("sh_holder_type", "").astype(str).str.upper().value_counts().to_dict(),
    )

    # ── 2. 面板目标日确认 ──
    panel_dates = pd.to_datetime(
        pq.read_table(str(PANEL_V3_PATH), columns=["date"]).to_pandas()["date"]
    )
    max_panel_date = panel_dates.max().date()
    hit = daily["date"].dt.date.isin(set(panel_dates.dt.date))
    print(
        f"面板最大日期 {max_panel_date}: {int(hit.sum())}/{len(daily)} 聚合行有对应面板行"
        f" (缺失 {int((~hit).sum())} 行 — 非交易日公告/面板未覆盖, 设计内跳过)"
    )

    if args.dry_run:
        print("[dry-run] 不写盘")
        return 0

    # ── 3. 磁盘空间 + WORM 备份 ──
    free = shutil.disk_usage(os.path.dirname(PANEL_V3_PATH)).free
    panel_bytes = os.path.getsize(PANEL_V3_PATH)
    if free < panel_bytes * 2 + 1e9:
        print(
            f"FATAL: D: 剩余 {free / 1e9:.1f}GB < 面板 {panel_bytes / 1e9:.1f}GB * 2 + 1GB"
        )
        return 1
    bak = PANEL_V3_PATH.with_name(
        f"panel_full_enriched_v3_pre_holder_evt_{pd.Timestamp.now():%Y%m%d_%H%M%S}.parquet"
    )
    print(f"备份 → {bak}")
    shutil.copy2(PANEL_V3_PATH, bak)

    # ── 4. 写前 OHLCV 校验基线 ──
    before = pq.read_table(str(PANEL_V3_PATH), columns=OHLCV_COLS).to_pandas()
    before_viol = _ohlcv_violations(before)
    del before
    print(f"写前 OHLCV 违例: {before_viol}")

    # ── 5. 流式重写: 事件日覆盖 10 列, 非事件行不动 ──
    pf = pq.ParquetFile(PANEL_V3_PATH)
    tmp = PANEL_V3_PATH.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(tmp, pf.schema_arrow)
    daily_agg = daily.rename(columns={c: f"{c}_agg" for c in HOLDER_PANEL_COLS})
    n_patched = 0
    for rg_idx in range(pf.metadata.num_row_groups):
        g = pf.read_row_group(rg_idx).to_pandas()
        m = g[["symbol", "date"]].merge(daily_agg, on=["symbol", "date"], how="left")
        has_evt = m["sh_net_sign_agg"].notna().to_numpy()  # sum(sign) 恒非 NaN ↔ 事件行
        for c in HOLDER_PANEL_COLS:
            fresh = m[f"{c}_agg"].to_numpy()
            mask = has_evt & pd.notna(fresh)  # 事件行 + 新值可用才覆盖 (防 NaT 回退)
            if mask.any():
                g[c] = np.where(mask, fresh, g[c].to_numpy())
                n_patched += int(mask.sum())
        writer.write_table(
            pa.Table.from_pandas(g, schema=pf.schema_arrow, preserve_index=False)
        )
    writer.close()
    pf.close()
    os.remove(PANEL_V3_PATH)
    os.rename(tmp, PANEL_V3_PATH)
    print(f"完成: 覆盖 {n_patched} 格 (10 列 × 事件日行)")

    # ── 6. 写后验证 ──
    v = pq.read_table(
        str(PANEL_V3_PATH), columns=["symbol", "date"] + HOLDER_PANEL_COLS
    ).to_pandas()
    assert v.shape[0] == panel_dates.shape[0], (
        f"行数变化 {panel_dates.shape[0]} -> {v.shape[0]}"
    )
    twins = [c for c in v.columns if c.endswith("_x") or c.endswith("_y")]
    assert not twins, f"出现 _x/_y 孪生列: {twins}"
    for c in HOLDER_PANEL_COLS:
        sub = v[v[c].notna()]
        print(
            f"  {c}: 非空 {len(sub):>8}  最新 {sub['date'].max().date() if len(sub) else '-'}"
        )
    after_viol = _ohlcv_violations(
        pq.read_table(str(PANEL_V3_PATH), columns=OHLCV_COLS).to_pandas()
    )
    assert after_viol == before_viol, f"OHLCV 违例数变化 {before_viol} -> {after_viol}"
    print(f"OHLCV 违例数不变: {after_viol} (写后)")

    # spot-check: 08-12..14 事件行抽样
    spot = v[v["date"].dt.date >= dt.date(2026, 8, 12)].dropna(
        subset=["sh_evt_end_date"]
    )
    print(f"\n08-12..14 事件行抽样 ({len(spot)} 行):")
    if len(spot):
        print(spot.head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
