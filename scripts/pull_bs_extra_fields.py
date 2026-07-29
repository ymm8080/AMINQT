#!/usr/bin/env python3
"""从 baostock 拉取 pcfNcfTTM (Price/CashFlow TTM) + 本地计算 pctChg.

pctChg 直接从 v3 面板的 close 列计算, 无需网络.
pcfNcfTTM 从 baostock K-line API 拉取 (仅近 400 交易日 ≈ 2025-01-01 至今).

Usage:
    python scripts/pull_bs_extra_fields.py                 # 全量拉取
    python scripts/pull_bs_extra_fields.py --resume        # 断点续传
    python scripts/pull_bs_extra_fields.py --dry-run       # 预览
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
OUTPUT = "data/bs_pcfNcfTTM.parquet"
PROGRESS_K = "data/bs_pcfNcfTTM_progress.txt"
PCF_START = "2025-01-01"
PCF_END = "2026-07-28"


def get_symbols_from_v3() -> list[str]:
    """从 v3 面板获取股票列表."""
    df = pd.read_parquet(V3_PANEL, columns=["symbol"])
    symbols = sorted(df["symbol"].unique())
    logger.info(f"Symbols from v3: {len(symbols)}")
    return symbols


def load_progress() -> set:
    if not os.path.exists(PROGRESS_K):
        return set()
    with open(PROGRESS_K) as f:
        return set(line.strip() for line in f)


def save_progress(sym: str):
    with open(PROGRESS_K, "a") as f:
        f.write(f"{sym}\n")
        f.flush()


def _code_of(symbol: str) -> str:
    return f"{'sh' if symbol.startswith(('6', '5', '9')) else 'sz'}.{symbol}"


def query_pcf_batch(symbols: list[str]) -> list[pd.DataFrame]:
    """在一段 login session 内批量拉取多个股票的 pcfNcfTTM."""
    bs.login()
    frames = []
    try:
        for sym in symbols:
            code = _code_of(sym)
            rs = bs.query_history_k_data_plus(
                code,
                "date,code,pcfNcfTTM",
                start_date=PCF_START, end_date=PCF_END,
                frequency="d", adjustflag="2",
            )
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rows:
                df = pd.DataFrame(rows, columns=rs.fields)
                df["pcfNcfTTM"] = pd.to_numeric(df["pcfNcfTTM"], errors="coerce")
                df["symbol"] = sym
                df["date"] = pd.to_datetime(df["date"])
                df = df.dropna(subset=["pcfNcfTTM"])
                frames.append(df[["symbol", "date", "pcfNcfTTM"]])
    finally:
        bs.logout()
    return frames


# 每 N 个股票重新 login 一次, 避免 session 超时
BATCH_SIZE = 200  # 每 200 只重新 login 一次, 防 session 超时


def compute_pctChg_from_v3() -> pd.DataFrame:
    """从 v3 面板的 close 计算 pctChg = (close / close.shift(1) - 1) * 100."""
    logger.info("Computing pctChg from v3 close...")
    df = pd.read_parquet(V3_PANEL, columns=["symbol", "date", "close"])
    df = df.sort_values(["symbol", "date"])
    df["pctChg"] = df.groupby("symbol")["close"].pct_change() * 100
    result = df[["symbol", "date", "pctChg"]].copy()
    logger.info(f"pctChg computed: {result['pctChg'].notna().sum()} non-null rows")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # ── Step 1: 计算 pctChg ──
    pct_df = compute_pctChg_from_v3()
    if args.dry_run:
        logger.info(f"DRY RUN: pctChg would have {len(pct_df)} rows")
    else:
        pct_df.to_parquet("data/bs_pctChg.parquet", index=False)
        logger.info(f"Saved data/bs_pctChg.parquet: {len(pct_df)} rows")

    # ── Step 2: 拉取 pcfNcfTTM ──
    symbols = get_symbols_from_v3()
    done = load_progress() if args.resume else set()
    todo = [s for s in symbols if s not in done]
    logger.info(f"pcfNcfTTM pull: {len(todo)} todo / {len(symbols)} total")

    if args.dry_run:
        logger.info("DRY RUN: skipped pcfNcfTTM pull")
        return

    frames = []
    if args.resume and os.path.exists(OUTPUT):
        frames.append(pd.read_parquet(OUTPUT))

    # 分批次拉取: 每个 batch 共用一个 login session
    t0 = time.time()
    failed = []
    n_done = 0
    for batch_start in range(0, len(todo), BATCH_SIZE):
        batch = todo[batch_start:batch_start + BATCH_SIZE]
        try:
            batch_frames = query_pcf_batch(batch)
            frames.extend(batch_frames)
            for sym in batch:
                save_progress(sym)
            n_done += len(batch)
        except Exception as e:
            # 某批失败, 逐只重试 (单 stock login)
            logger.warning(f"Batch failed for {batch[0]}..{batch[-1]}: {e}, retrying individually...")
            for sym in batch:
                try:
                    bs.login()
                    rs = bs.query_history_k_data_plus(
                        _code_of(sym),
                        "date,code,pcfNcfTTM",
                        start_date=PCF_START, end_date=PCF_END,
                        frequency="d", adjustflag="2",
                    )
                    rows = []
                    while rs.next():
                        rows.append(rs.get_row_data())
                    bs.logout()
                    if rows:
                        df = pd.DataFrame(rows, columns=rs.fields)
                        df["pcfNcfTTM"] = pd.to_numeric(df["pcfNcfTTM"], errors="coerce")
                        df["symbol"] = sym
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.dropna(subset=["pcfNcfTTM"])
                        frames.append(df[["symbol", "date", "pcfNcfTTM"]])
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
        logger.info(f"pcfNcfTTM [{n_done}/{len(todo)}] {pct:.0f}% "
                    f"rate={rate:.0f}/hr eta={eta_h:.1f}hr")

    elapsed = time.time() - t0
    n_ok = len(todo) - len(failed)
    logger.info(f"pcfNcfTTM pull done: {n_ok}/{len(todo)} in {elapsed/3600:.1f}hr")

    if frames:
        panel = pd.concat(frames, ignore_index=True)
        panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
        panel.to_parquet(OUTPUT, index=False)
        logger.info(f"Saved {OUTPUT}: {len(panel)} rows, "
                    f"{panel['symbol'].nunique()} stocks, "
                    f"pcfNcfTTM coverage={panel['pcfNcfTTM'].notna().sum()}")
    else:
        logger.info("No pcfNcfTTM data collected.")


if __name__ == "__main__":
    main()
