#!/usr/bin/env python3
"""从 baostock 拉取季度基本面数据，生成日频面板.

从以下5个 API 提取不在 v3 中的维度：
  - Balance:  quickRatio, cashRatio, assetToEquity
  - CashFlow: tangibleAssetToAsset, ebitToInterest, CFOToNP, CFOToGr
  - Profit:  (可选验证, 不新增列)

数据流: 季度 → forward-fill 到日频, 按 (symbol, date) 合并到 v3.

Usage:
    python scripts/pull_bs_quarterly_fundamentals.py              # 全量
    python scripts/pull_bs_quarterly_fundamentals.py --resume     # 断点续传
    python scripts/pull_bs_quarterly_fundamentals.py --dry-run    # 预览
"""
import argparse, logging, os, sys, time
import numpy as np
import pandas as pd
import baostock as bs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    force=True, stream=sys.stderr)
logger = logging.getLogger(__name__)

# ── 路径 ──────────────────────────────────────────────
V3_PANEL = "data/panel_full_enriched_v3.parquet"
OUTPUT_RAW = "data/bs_fundamentals_raw.parquet"
OUTPUT_FF = "data/bs_fundamentals_ff.parquet"
PROGRESS_F = "data/bs_fundamentals_progress.txt"

# 只拉最近 6 个季度 (2025Q1-2026Q2), 之前数据 forward-fill 为 NaN
QUARTERS = ["2025Q1", "2025Q2", "2025Q3", "2025Q4",
            "2026Q1", "2026Q2"]

# 需要从各个 API 提取的字段 (key=API名称, value=需要的列)
# 只提取不在 v3 中的新增列
FUNDAMENTAL_COLS = {
    "balance": ["quickRatio", "cashRatio", "assetToEquity"],
    "cashflow": ["tangibleAssetToAsset", "ebitToInterest", "CFOToNP", "CFOToGr"],
}

# API映射
API_MAP = {
    "balance":   bs.query_balance_data,
    "cashflow":  bs.query_cash_flow_data,
}


def get_symbols_from_v3() -> list[str]:
    df = pd.read_parquet(V3_PANEL, columns=["symbol"])
    symbols = sorted(df["symbol"].unique())
    logger.info(f"Symbols from v3: {len(symbols)}")
    return symbols


def quarter_to_year_q(q: str) -> tuple:
    """'2025Q2' → (2025, 2)"""
    parts = q.split("Q")
    return int(parts[0]), int(parts[1])


def load_progress() -> set:
    if not os.path.exists(PROGRESS_F):
        return set()
    with open(PROGRESS_F) as f:
        return set(line.strip() for line in f)


def save_progress(sym: str):
    with open(PROGRESS_F, "a") as f:
        f.write(f"{sym}\n")
        f.flush()


def _code_of(symbol: str) -> str:
    return f"{'sh' if symbol.startswith(('6', '5', '9')) else 'sz'}.{symbol}"


def pull_fundamentals_batch(symbols: list[str]) -> pd.DataFrame | None:
    """在一段 login session 内批量拉取多个股票的季度基本面."""
    bs.login()
    rows_all = []
    try:
        for sym in symbols:
            code = _code_of(sym)
            for api_name, api_fn in API_MAP.items():
                cols = FUNDAMENTAL_COLS[api_name]
                for q_label in QUARTERS:
                    year, quarter = quarter_to_year_q(q_label)
                    try:
                        rs = api_fn(code, year=year, quarter=quarter)
                        data = []
                        while rs.next():
                            data.append(rs.get_row_data())
                        if data:
                            df = pd.DataFrame(data, columns=rs.fields)
                            keep = ["code", "pubDate", "statDate"] + cols
                            keep = [c for c in keep if c in df.columns]
                            df = df[keep].copy()
                            df["symbol"] = sym
                            df["q_label"] = q_label
                            df["api"] = api_name
                            rows_all.append(df)
                    except Exception as e:
                        logger.warning(f"  {sym} {api_name} {q_label}: {e}")
    finally:
        bs.logout()

    if not rows_all:
        return None

    combined = pd.concat(rows_all, ignore_index=True)
    combined = combined.drop(columns=["q_label", "api"])

    # 去重: 同 stock-quarter, 取最新 pubDate
    combined = combined.sort_values("pubDate").drop_duplicates(
        subset=["symbol", "statDate"], keep="last"
    )

    # 类型转换
    for col in FUNDAMENTAL_COLS["balance"] + FUNDAMENTAL_COLS["cashflow"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined["statDate"] = pd.to_datetime(combined["statDate"])
    combined["pubDate"] = pd.to_datetime(combined["pubDate"])
    combined["avail_date"] = combined["pubDate"] + pd.Timedelta(days=1)
    combined = combined.dropna(subset=["avail_date"])

    meta = ["symbol", "statDate", "pubDate", "avail_date"]
    val_cols = [c for c in FUNDAMENTAL_COLS["balance"] + FUNDAMENTAL_COLS["cashflow"]
                if c in combined.columns]
    return combined[meta + val_cols]


BATCH_SIZE = 200  # 每 200 只重新 login 一次, 防 session 超时


def forward_fill_to_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """将季度数据 forward-fill 到日频, 对齐 v3 的交易日历.

    规则: 从 avail_date 开始使用该季度数据, 直到下一个季度的 avail_date.
    """
    logger.info(f"Forward-filling quarterly data to daily...")

    # 加载 v3 的交易日历 (获取所有 (symbol, date) 骨架)
    calendar = pd.read_parquet(V3_PANEL, columns=["symbol", "date"])
    logger.info(f"Calendar skeleton: {len(calendar)} rows")

    # 对每个 symbol 进行 forward-fill
    symbols = sorted(raw["symbol"].unique())
    logger.info(f"Symbols with fundamentals: {len(symbols)}")

    # 用 merge_asof 做 forward-fill 最干净
    # 先合并所有 quarter 数据
    ff_frames = []
    for sym in symbols:
        sym_raw = raw[raw["symbol"] == sym].copy()
        sym_cal = calendar[calendar["symbol"] == sym].copy()

        if sym_raw.empty or sym_cal.empty:
            continue

        # 按日期排序
        sym_raw = sym_raw.sort_values("avail_date")
        sym_cal = sym_cal.sort_values("date")

        val_cols = [c for c in sym_raw.columns
                    if c not in ("symbol", "statDate", "pubDate", "avail_date")]

        # merge_asof: 每个交易日取最近 (且早于等于该日) 的季度数据
        ff = pd.merge_asof(
            sym_cal,
            sym_raw[["avail_date"] + val_cols],
            left_on="date", right_on="avail_date",
            direction="backward",  # 取最近的可用的季度数据
        )
        # 如果 data 为空, 说明该日期没有可用季度数据 (最早 quarter 之前)
        ff_frames.append(ff)

    if not ff_frames:
        logger.warning("No forward-filled data produced!")
        return pd.DataFrame()

    result = pd.concat(ff_frames, ignore_index=True)
    result = result.drop(columns=["avail_date"])
    # 去掉还没有任何季度数据的行
    val_cols = [c for c in FUNDAMENTAL_COLS["balance"] + FUNDAMENTAL_COLS["cashflow"]
                if c in result.columns]
    result = result.dropna(subset=val_cols, how="all")
    logger.info(f"Forward-filled: {len(result)} rows, {result['symbol'].nunique()} stocks")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-pull", action="store_true",
                        help="跳过拉取, 直接从已有 raw 文件做 forward-fill")
    args = parser.parse_args()

    symbols = get_symbols_from_v3()
    if args.dry_run:
        logger.info(f"DRY RUN: {len(symbols)} stocks, "
                    f"{len(QUARTERS)} quarters, "
                    f"{len(API_MAP)} APIs")
        return

    # ── Step 1: 拉取原始季度数据 ──
    if not args.skip_pull:
        done = load_progress() if args.resume else set()
        todo = [s for s in symbols if s not in done]
        logger.info(f"Fundamentals pull: {len(todo)} todo / {len(symbols)} total")

        frames = []
        if args.resume and os.path.exists(OUTPUT_RAW):
            frames.append(pd.read_parquet(OUTPUT_RAW))

        t0 = time.time()
        failed = []
        n_done = 0
        for batch_start in range(0, len(todo), BATCH_SIZE):
            batch = todo[batch_start:batch_start + BATCH_SIZE]
            try:
                df = pull_fundamentals_batch(batch)
                if df is not None and len(df) > 0:
                    frames.append(df)
                for sym in batch:
                    save_progress(sym)
                n_done += len(batch)
            except Exception as e:
                logger.warning(f"Batch failed for {batch[0]}..{batch[-1]}: {e}, "
                              f"retrying individually...")
                for sym in batch:
                    try:
                        df = pull_fundamentals_batch([sym])
                        if df is not None and len(df) > 0:
                            frames.append(df)
                        save_progress(sym)
                    except Exception as e2:
                        failed.append(sym)
                        if len(failed) <= 3:
                            logger.warning(f"  {sym}: {e2}")
                n_done += len(batch)

            elapsed = time.time() - t0
            rate = n_done / elapsed * 3600 if elapsed > 0 else 0
            pct = n_done / len(todo) * 100 if todo else 100
            eta_h = (len(todo) - n_done) / rate if rate > 0 else 0
            logger.info(f"Fundamentals [{n_done}/{len(todo)}] {pct:.0f}% "
                        f"rate={rate:.0f}/hr eta={eta_h:.1f}hr")

        elapsed = time.time() - t0
        n_ok = len(todo) - len(failed)
        logger.info(f"Fundamentals pull done: {n_ok}/{len(todo)} in {elapsed/3600:.1f}hr")

        if frames:
            raw = pd.concat(frames, ignore_index=True)
            # 排序并去重
            raw = raw.sort_values(["symbol", "statDate", "pubDate"]).reset_index(drop=True)
            raw = raw.drop_duplicates(subset=["symbol", "statDate"], keep="last")
            raw.to_parquet(OUTPUT_RAW, index=False)
            logger.info(f"Saved {OUTPUT_RAW}: {len(raw)} rows, "
                        f"{raw['symbol'].nunique()} stocks")
        else:
            logger.warning("No raw fundamentals collected.")
            return
    else:
        if not os.path.exists(OUTPUT_RAW):
            logger.error(f"--skip-pull but {OUTPUT_RAW} not found!")
            return
        raw = pd.read_parquet(OUTPUT_RAW)
        logger.info(f"Loaded raw fundamentals: {len(raw)} rows, "
                    f"{raw['symbol'].nunique()} stocks")

    # ── Step 2: Forward-fill to daily ──
    ff = forward_fill_to_daily(raw)
    if len(ff) > 0:
        ff.to_parquet(OUTPUT_FF, index=False)
        logger.info(f"Saved {OUTPUT_FF}: {len(ff)} rows, "
                    f"{ff['symbol'].nunique()} stocks")
        # 打印覆盖率
        for col in [c for c in FUNDAMENTAL_COLS["balance"] + FUNDAMENTAL_COLS["cashflow"]
                    if c in ff.columns]:
            nna = ff[col].notna().sum()
            logger.info(f"  {col}: {nna} ({nna/len(ff)*100:.1f}%)")


if __name__ == "__main__":
    main()
