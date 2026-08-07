"""一次性脚本: 给现存 V3 面板回填 4 个 KIMI 式股东增减持比例列.

  sh_net_ratio  = sum(signed_ratio)          按公告日聚合 (ann_date, 无 shift → 无 look-ahead)
  sh_g_ratio    = sum(signed_ratio | holder_type=="G")   高管
  sh_p_ratio    = sum(signed_ratio | holder_type=="P")   个人
  sh_c_ratio    = sum(signed_ratio | holder_type=="C")   公司

聚合逻辑与 scripts/_holder_scheme_ic.py (GLM vs KIMI 对比) 完全一致 (向量化, 无 per-stock 循环).

Surgical merge — 只并 4 列, 不复跑 enrich_alt_data (会生成 _x/_y 孪生列).
Guard: 4 列任一已存在于面板 → 中止.
WORM: 先 shutil.copy2 备份 panel_full_enriched_v3_preholder_ratio_<ts>.parquet 再写回.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("backfill_holder_ratio")

PANEL_PATH = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
RAW_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "_holder_cmp_raw.parquet",
)
RATIO_COLS = ["sh_net_ratio", "sh_g_ratio", "sh_p_ratio", "sh_c_ratio"]
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


def main() -> None:
    # ── 0. 并发写入防护: 面板 mtime 过新则中止 ──
    #     HOLDER_BF_SKIP_MTIME_GUARD=1 跳过 (仅当已确认无并发写进程时使用).
    mtime = os.path.getmtime(PANEL_PATH)
    age = time.time() - mtime
    if age < MTIME_GRACE_S and os.environ.get("HOLDER_BF_SKIP_MTIME_GUARD") != "1":
        logger.error(
            "面板 mtime %s 距今 %.0fs (<%ds), 疑似并发写入, 中止回填",
            datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
            age,
            MTIME_GRACE_S,
        )
        raise SystemExit(1)
    logger.info(
        "面板 mtime %s (%.0fs 前), 无并发写入迹象",
        datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
        age,
    )

    # ── 1. 加载面板 + Guard: 4 列不得已存在 ──
    logger.info("加载面板: %s", PANEL_PATH)
    panel = pd.read_parquet(PANEL_PATH)
    logger.info(
        "面板: %d 行, %d symbols, 日期 %s ~ %s, 列数 %d",
        panel.shape[0],
        panel["symbol"].nunique(),
        panel["date"].min().strftime("%Y%m%d"),
        panel["date"].max().strftime("%Y%m%d"),
        len(panel.columns),
    )
    existing = [c for c in RATIO_COLS if c in panel.columns]
    if existing:
        logger.error("面板已存在 %s, 中止 (避免 _x/_y 孪生列)", existing)
        raise SystemExit(1)

    # ── 2. WORM 备份 + 磁盘空间检查 ──
    free = shutil.disk_usage(os.path.dirname(PANEL_PATH)).free
    panel_bytes = os.path.getsize(PANEL_PATH)
    if free < panel_bytes * 2 + 1e9:
        logger.error(
            "D: 剩余 %.1fGB < 面板 %d 字节 * 2 + 1GB, 中止", free / 1e9, panel_bytes
        )
        raise SystemExit(1)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{os.path.splitext(PANEL_PATH)[0]}_preholder_ratio_{ts}.parquet"
    logger.info("备份: %s", backup)
    shutil.copy2(PANEL_PATH, backup)

    # ── 3. 原始事件 → 每日聚合 (与 _holder_scheme_ic 一致, 向量化) ──
    raw = pd.read_parquet(RAW_PATH)
    raw["date"] = pd.to_datetime(raw["date"])
    raw["change_ratio"] = pd.to_numeric(raw["change_ratio"], errors="coerce")
    raw["holder_type"] = raw["holder_type"].fillna("").astype(str).str.upper()
    raw["signed_ratio"] = pd.to_numeric(raw["signed_ratio"], errors="coerce").fillna(
        0.0
    )
    raw["sr_g"] = np.where(raw["holder_type"] == "G", raw["signed_ratio"], 0.0)
    raw["sr_p"] = np.where(raw["holder_type"] == "P", raw["signed_ratio"], 0.0)
    raw["sr_c"] = np.where(raw["holder_type"] == "C", raw["signed_ratio"], 0.0)
    agg = raw.groupby(["symbol", "date"], as_index=False).agg(
        sh_net_ratio=("signed_ratio", "sum"),
        sh_g_ratio=("sr_g", "sum"),
        sh_p_ratio=("sr_p", "sum"),
        sh_c_ratio=("sr_c", "sum"),
    )
    logger.info("事件聚合行数: %d", len(agg))

    # ── 4. 记录写前 OHLCV 校验基线 ──
    before_viol = _ohlcv_violations(panel)
    before_rows, before_cols = panel.shape
    logger.info("写前 OHLCV 违例: %d", before_viol)

    # ── 5. 左并 4 列 → 写回 (SNAPPY, 与源压缩一致) ──
    panel = panel.merge(agg, on=["symbol", "date"], how="left")
    logger.info("写回: %s (列数 %d -> %d)", PANEL_PATH, before_cols, len(panel.columns))
    panel.to_parquet(PANEL_PATH, index=False)

    # ── 6. 写后验证 ──
    verify = pd.read_parquet(PANEL_PATH)
    assert verify.shape[0] == before_rows, (
        f"行数变化 {before_rows} -> {verify.shape[0]}"
    )
    twins = [c for c in verify.columns if c.endswith("_x") or c.endswith("_y")]
    assert not twins, f"出现 _x/_y 孪生列: {twins}"
    missing = [c for c in RATIO_COLS if c not in verify.columns]
    assert not missing, f"缺失列: {missing}"
    for c in RATIO_COLS:
        nn = int(verify[c].notna().sum())
        logger.info("  %s: 非空 %d 行, 覆盖 %.3f%%", c, nn, 100.0 * nn / before_rows)
    after_viol = _ohlcv_violations(verify)
    assert after_viol == before_viol, f"OHLCV 违例数变化 {before_viol} -> {after_viol}"
    logger.info("OHLCV 违例数不变: %d (写后)", after_viol)

    # spot-check 600519 公告日
    spot = verify[verify["symbol"] == "600519"][
        ["symbol", "date", "sh_net_ratio", "sh_g_ratio", "sh_p_ratio", "sh_c_ratio"]
    ]
    spot = spot[spot["sh_net_ratio"].notna()]
    logger.info("600519 公告日非空行 %d:", len(spot))
    if len(spot):
        logger.info("\n%s", spot.head(10).to_string(index=False))

    logger.info(
        "完成: 列数 %d -> %d, 行数 %d 不变",
        before_cols,
        len(verify.columns),
        before_rows,
    )
    logger.info("备份: %s", backup)


if __name__ == "__main__":
    main()
