# -*- coding: utf-8 -*-
"""修复 panel *_hfq 复权因子系统性跳变 + 受污染的存储衍生 bias 列.

根因 (2026-08-03 实盘确认):
  - 历史行 (<=2026-07-24) 经 data_supply._tushare_fetch_hist 写入 close_hfq=close
    (300065 无复权 mult=1.0) 或陈旧因子 (000001 mult=150.7, Tushare 实际 139.008)。
  - 2026-07-27 起 _daily_fetch 用 close × Tushare adj_factor(trade_date) → 正确口径。
  - 07-27 系统性跳变 (~2987 只) 污染跨缝的标签 (label_engine 用 close_hfq) 与特征,
    导致两模型 OOS IC 转负 (MAIN -0.0288 / DUAL -0.0519)。

修复 (与 _daily_fetch 日更口径完全一致):
  1) open_hfq/high_hfq/low_hfq/close_hfq = raw × Tushare adj_factor(date)
  2) bias_5..250 = close_hfq / MA_n(close_hfq) - 1   (公式同 _daily_fetch bias 块)
  3) bias_5_20_cross / bias_20_60_cross = sign(bias_a - bias_b).diff().fillna(0)
     (同 feature_engine_v35 交叉块)
  不动: intraday_range / pctChg / amplitude_5d / ma_vol_ratio / vol_surge / amt_surge
    (均基于 raw OHLC/volume, 与 hfq 无关)。

幂等: adj_factor 按日缓存于 data/supply_cache/alt_data/adj_factor/, 重跑不重复抓取。

用法: python scripts/_fix_close_hfq.py   (WORM 备份后原地覆写 PANEL_V3_PATH)
"""

import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")

import numpy as np
import pandas as pd

from config.settings import PANEL_V3_PATH

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("fix_close_hfq")

ADJ_CACHE = "data/supply_cache/alt_data/adj_factor"
HFQ_COLS = ["open_hfq", "high_hfq", "low_hfq", "close_hfq"]
RAW_COLS = ["open", "high", "low", "close"]
BIAS_WINDOWS = {
    "bias_5": 5,
    "bias_10": 10,
    "bias_20": 20,
    "bias_60": 60,
    "bias_120": 120,
    "bias_250": 250,
}
CROSS_PAIRS = [
    ("bias_5_20_cross", "bias_5", "bias_20"),
    ("bias_20_60_cross", "bias_20", "bias_60"),
]


def _to_symbol(df: pd.DataFrame) -> pd.DataFrame:
    if "ts_code" in df.columns:
        df = df.copy()
        df["symbol"] = (
            df["ts_code"]
            .str.replace(".SZ", "", regex=False)
            .str.replace(".SH", "", regex=False)
        )
    return df


def _safe_fetch(pro, dstr: str) -> pd.DataFrame:
    for attempt in range(1, 4):
        try:
            df = pro.adj_factor(trade_date=dstr)
            if len(df):
                return _to_symbol(df)
        except Exception as e:  # noqa: BLE001 — 重试瞬时限流/超时
            logger.warning("  %s attempt %d/%d FAILED: %s", dstr, attempt, 3, e)
        time.sleep(2 * attempt)
    return pd.DataFrame()


def _fetch_adj_factors(dates: pd.Series) -> pd.DataFrame:
    """按日抓 adj_factor, 缓存 parquet (幂等)."""
    import tushare as ts

    token = os.getenv("TUSHARE_TOKEN", "") or ts.get_token()
    if not token:
        raise SystemExit("FATAL: 无 TUSHARE_TOKEN")
    pro = ts.pro_api(token)
    os.makedirs(ADJ_CACHE, exist_ok=True)
    out = []
    for i, d in enumerate(dates):
        dstr = pd.Timestamp(d).strftime("%Y%m%d")
        cache = os.path.join(ADJ_CACHE, f"adj_{dstr}.parquet")
        if os.path.exists(cache):
            df = pd.read_parquet(cache)
        else:
            df = _safe_fetch(pro, dstr)
            if len(df):
                df.to_parquet(cache, index=False)
            time.sleep(0.35)
        if len(df):
            out.append(df[["symbol", "adj_factor"]].assign(date=pd.Timestamp(d)))
        if (i + 1) % 100 == 0:
            logger.info("  adj_factor fetched %d/%d dates", i + 1, len(dates))
    if not out:
        raise SystemExit("FATAL: 未抓到任何 adj_factor")
    return pd.concat(out, ignore_index=True)


def main() -> None:
    panel = pd.read_parquet(PANEL_V3_PATH)
    logger.info(
        "Panel: %d rows, %d symbols, %d cols; dates %s..%s",
        len(panel),
        panel["symbol"].nunique(),
        len(panel.columns),
        panel["date"].min().date(),
        panel["date"].max().date(),
    )

    # ── 0. WORM 备份 ──
    backup = str(PANEL_V3_PATH).replace(
        ".parquet", f"_prehfqfix_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    )
    panel.to_parquet(backup, index=False)
    logger.info("WORM backup -> %s (%.1f MB)", backup, os.path.getsize(backup) / 1e6)

    # ── 1. 抓 adj_factor ──
    dates = np.sort(panel["date"].unique())
    logger.info("fetching adj_factor for %d dates...", len(dates))
    adj = _fetch_adj_factors(dates)
    logger.info("adj_factor rows: %d", len(adj))

    # ── 2. 对齐 factor, 每股按日期 ff(日)fill (停牌日缺行) ──
    m = panel[["symbol", "date"]].merge(adj, on=["symbol", "date"], how="left")
    m = m.sort_values(["symbol", "date"])
    m["adj_factor"] = m.groupby("symbol")["adj_factor"].ffill()
    old_mult = (panel["close_hfq"] / panel["close"]).replace([np.inf, -np.inf], np.nan)
    factor = m["adj_factor"].fillna(old_mult)  # 残留缺失行沿用旧口径, 避免引入 NaN
    nan_rows = m["adj_factor"].isna().sum()
    if nan_rows:
        logger.warning("%d rows 无 adj_factor (沿用旧口径)", nan_rows)

    # ── 3. 重算 *_hfq ──
    panel = panel.copy()
    for raw, hfq in zip(RAW_COLS, HFQ_COLS):
        panel[hfq] = panel[raw] * factor.values

    # ── 4. 重算存储 bias_* + cross ──
    srt = panel.sort_values(["symbol", "date"])
    for col, w in BIAS_WINDOWS.items():
        if col in panel.columns:
            ma = srt.groupby("symbol")["close_hfq"].transform(
                lambda x: x.rolling(w, min_periods=w).mean()
            )
            srt[col] = srt["close_hfq"] / ma - 1
    for cross_col, a, b in CROSS_PAIRS:
        if cross_col in panel.columns and a in panel.columns and b in panel.columns:
            srt[cross_col] = (
                np.sign(srt[a] - srt[b]).groupby(srt["symbol"]).diff().fillna(0.0)
            )
    panel = srt.sort_index()

    # ── 5. 验证: 07-27 系统性跳变应消失 ──
    def mult_at(dstr: str) -> pd.Series:
        sub = panel[panel["date"] == pd.Timestamp(dstr)][
            ["symbol", "close_hfq", "close"]
        ]
        return sub.set_index("symbol")["close_hfq"] / sub.set_index("symbol")[
            "close"
        ].replace(0, np.nan)

    m24, m27, m28 = mult_at("2026-07-24"), mult_at("2026-07-27"), mult_at("2026-07-28")
    seam_before = (m24.dropna() != m27.dropna()).sum() if len(m24) else -1
    seam_after = (m24.dropna() != m28.dropna()).sum() if len(m24) else -1
    logger.info(
        "mult(07-24)!=mult(07-27): %d symbols 修复前跳变; mult(07-24)!=mult(07-28): %d 修复后",
        seam_before,
        seam_after,
    )
    for sym in ["300065", "000001", "600519"]:
        row = panel[
            (panel["symbol"] == sym) & (panel["date"] == pd.Timestamp("2026-07-28"))
        ]
        if len(row):
            r = row.iloc[0]
            logger.info(
                "  %s 07-28: close=%.2f close_hfq=%.2f mult=%.4f",
                sym,
                r["close"],
                r["close_hfq"],
                r["close_hfq"] / r["close"],
            )

    # ── 6. 落盘 (原地, 备份已留) ──
    panel.to_parquet(PANEL_V3_PATH, index=False)
    logger.info(
        "Saved fixed panel -> %s (%.1f MB)",
        PANEL_V3_PATH,
        os.path.getsize(PANEL_V3_PATH) / 1e6,
    )
    logger.info("DONE")


if __name__ == "__main__":
    main()
