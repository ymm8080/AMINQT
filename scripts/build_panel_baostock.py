#!/usr/bin/env python3
"""用 baostock 拉全A股日线 + 换手率，构建完整训练面板 (4000+ 只).

Usage:
    python scripts/build_panel_baostock.py --dry-run
    python scripts/build_panel_baostock.py
    python scripts/build_panel_baostock.py --resume
"""

import argparse
import logging
import os
import time
import pandas as pd
import baostock as bs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT = "data/panel_full.parquet"
PROGRESS = "data/panel_full_bs_progress.txt"
START_DATE = "2024-01-01"
END_DATE = "2026-07-27"
MIN_AMOUNT = 5e7  # 日均成交额 >= 5000 万


def get_all_symbols() -> list[str]:
    """获取全A股列表 (排除ST/退市/北交所)."""
    bs.login()
    rs = bs.query_stock_basic()
    data = []
    while rs.next():
        data.append(rs.get_row_data())
    df = pd.DataFrame(data, columns=rs.fields)
    bs.logout()
    # code format: sh.600000, sz.000001
    df["symbol"] = df["code"].str.split(".").str[1]
    # 排除北交所(8/4开头), 排除指数/ETF
    mask = df["symbol"].str.match(r"^[0-9]{6}$")
    df = df[mask]
    # 排除 8/4 开头 (北交所)
    df = df[~df["symbol"].str.startswith(("8", "4"))]
    # 取上市日期在 2024 年前
    if "ipoDate" in df.columns:
        df = df[df["ipoDate"] < "2024-01-01"]
    symbols = sorted(df["symbol"].unique())
    logger.info(f"Stock list: {len(symbols)}")
    return symbols


def pull_one_stock(symbol: str) -> pd.DataFrame | None:
    """拉一只股票全量日线."""
    code = f"{'sh' if symbol.startswith(('6', '5', '9')) else 'sz'}.{symbol}"
    bs.login()
    rs = bs.query_history_k_data_plus(
        code,
        "date,code,open,high,low,close,preclose,volume,amount,turn,tradestatus,isST",
        start_date=START_DATE,
        end_date=END_DATE,
        frequency="d",
        adjustflag="2",  # 前复权
    )
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    bs.logout()

    if not rows:
        return None
    df = pd.DataFrame(rows, columns=rs.fields)
    # 类型转换
    for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"])
    df["tradestatus"] = df["tradestatus"].astype(int)
    df["isST"] = df["isST"].astype(int)
    # 只保留正常交易日
    df = df[df["tradestatus"] == 1]
    df = df[df["isST"] == 0]
    return df[
        [
            "symbol",
            "date",
            "open",
            "high",
            "low",
            "close",
            "preclose",
            "volume",
            "amount",
            "turn",
        ]
    ].dropna(subset=["open", "close"])


def load_progress() -> set:
    if not os.path.exists(PROGRESS):
        return set()
    with open(PROGRESS) as f:
        return set(line.strip() for line in f)


def save_progress(sym: str):
    with open(PROGRESS, "a") as f:
        f.write(f"{sym}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    symbols = get_all_symbols()
    if args.dry_run:
        logger.info(f"DRY RUN: {len(symbols)} stocks")
        return

    done = load_progress() if args.resume else set()
    todo = [s for s in symbols if s not in done]
    logger.info(f"Todo: {len(todo)} / Total: {len(symbols)}")

    frames = []
    if args.resume and os.path.exists(OUTPUT):
        frames.append(pd.read_parquet(OUTPUT))

    t0 = time.time()
    failed = []
    for i, sym in enumerate(todo):
        try:
            df = pull_one_stock(sym)
            if df is not None and len(df) > 0:
                frames.append(df)
            save_progress(sym)
        except Exception as e:
            failed.append(sym)
            if i < 3:  # 只打印前几个错误
                logger.warning(f"  {sym}: {e}")

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 3600
            logger.info(
                f"Progress: {i + 1}/{len(todo)} ({(i + 1) / len(todo) * 100:.0f}%) "
                f"rate={rate:.0f}/hr ETA={(len(todo) - i - 1) / rate:.1f}hr"
            )

    elapsed = time.time() - t0
    logger.info(
        f"Pull done: {len(todo) - len(failed)}/{len(todo)} in {elapsed / 3600:.1f}hr"
    )

    if frames:
        panel = pd.concat(frames, ignore_index=True)
        panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

        # 过滤: 日均成交额 >= 5000万 + 至少200个交易日
        avg_amt = panel.groupby("symbol")["amount"].mean()
        valid_liq = avg_amt[avg_amt >= MIN_AMOUNT].index
        cnt = panel.groupby("symbol").size()
        valid_cnt = cnt[cnt >= 200].index
        valid = set(valid_liq) & set(valid_cnt)
        panel = panel[panel["symbol"].isin(valid)]

        # 重命名列以匹配 Pipeline1
        panel = panel.rename(columns={"turn": "turnover_rate", "preclose": "pre_close"})
        panel["close_hfq"] = panel["close"]
        panel["volume"] = panel["volume"] * 100  # baostock volume: 手→股 (待确认)

        panel.to_parquet(OUTPUT, index=False)
        logger.info(
            f"Saved {OUTPUT}: {len(panel)} rows, "
            f"{panel['symbol'].nunique()} stocks, "
            f"{panel['date'].min()} ~ {panel['date'].max()}"
        )
    else:
        logger.error("No data!")


if __name__ == "__main__":
    main()
