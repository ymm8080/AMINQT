"""Fetch stock -> Shenwan (SW) L1/L2/L3 industry classification mapping.

Full rebuild mode (default):
  python fetch_sw_classification.py
  - Fetches all 346 L3 indices via Tushare (~9 min)

Incremental update mode:
  python fetch_sw_classification.py --incremental 000001.SZ,000002.SZ
  python fetch_sw_classification.py --incremental-file missing_stocks.txt
  - Queries index_member_all(ts_code=xxx) per stock (1 API call each, ~0.3s)
  - Appends new rows to existing CSV

Output: data/processed/sw_stock_classification.csv
  Columns: ts_code, symbol, name, sw_l1_code, sw_l1_name,
           sw_l2_code, sw_l2_name, sw_l3_code, sw_l3_name, in_date
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import data_others_path  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = data_others_path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "sw_stock_classification.csv"

FINAL_COLS = [
    "ts_code",
    "symbol",
    "name",
    "sw_l1_code",
    "sw_l1_name",
    "sw_l2_code",
    "sw_l2_name",
    "sw_l3_code",
    "sw_l3_name",
    "in_date",
    "fetch_date",
    "data_source",
]


def fetch_tushare_classification():
    """Fetch SW classification via Tushare index_classify + index_member.

    Returns:
        pd.DataFrame with columns: ts_code, symbol, name,
            sw_l1_code, sw_l1_name, sw_l2_code, sw_l2_name, sw_l3_code, sw_l3_name, in_date
        or None on failure.
    """
    try:
        import tushare as ts
        from dotenv import load_dotenv

        load_dotenv()

        token = os.getenv("TUSHARE_TOKEN") or ts.get_token()
        if not token:
            logger.warning("Tushare: no token")
            return None
        pro = ts.pro_api(token)

        # ── Step 1: Get SW2021 classification hierarchy ──
        classify = {}
        for level in ["L1", "L2", "L3"]:
            for src in ["SW2021", "SW2014", "SW"]:
                try:
                    df = pro.index_classify(level=level, src=src)
                    if len(df) > 0:
                        logger.info(
                            f"index_classify(level={level}, src={src}): {len(df)} rows"
                        )
                        classify[level] = df
                        break
                except Exception as e:
                    logger.debug(f"index_classify(level={level}, src={src}): {e}")

        if "L3" not in classify:
            logger.error("Cannot get L3 classification from Tushare")
            return None

        l1_df = classify.get("L1", pd.DataFrame())
        l2_df = classify.get("L2", pd.DataFrame())
        l3_df = classify["L3"]

        logger.info(f"Hierarchy: L1={len(l1_df)}, L2={len(l2_df)}, L3={len(l3_df)}")

        # Build L3 -> L2 -> L1 parent mapping from index_classify
        l3_to_parent = {}
        if "parent_code" in l3_df.columns:
            for _, row in l3_df.iterrows():
                l3_code = row["index_code"]
                parent = row.get("parent_code", "")
                l3_to_parent[l3_code] = {
                    "l2_code": parent,
                    "l3_name": row.get("industry_name", ""),
                }

            if len(l2_df) > 0 and "parent_code" in l2_df.columns:
                for _, row in l2_df.iterrows():
                    l2_code = row["index_code"]
                    for _l3c, info in l3_to_parent.items():
                        if info["l2_code"] == l2_code:
                            info["l1_code"] = row.get("parent_code", "")
                            info["l2_name"] = row.get("industry_name", "")

            if len(l1_df) > 0:
                l1_names = dict(zip(l1_df["index_code"], l1_df["industry_name"]))
                for _l3c, info in l3_to_parent.items():
                    info["l1_name"] = l1_names.get(info.get("l1_code", ""), "")
        else:
            l3_to_parent = None

        # ── Step 2: Get stock members for each L3 index ──
        l3_codes = l3_df["index_code"].tolist()
        logger.info(f"Fetching members for {len(l3_codes)} L3 indices...")

        all_members = []
        for i, l3_code in enumerate(l3_codes):
            try:
                members = pro.index_member_all(l3_code=l3_code)
                if len(members) > 0:
                    all_members.append(members)
                    if (i + 1) % 50 == 0:
                        logger.info(
                            f"  Progress: {i + 1}/{len(l3_codes)}, total {sum(len(x) for x in all_members)} rows"
                        )
                    time.sleep(0.25)
                    continue
            except Exception:
                pass

            try:
                members = pro.index_member(id=l3_code)
                if len(members) > 0:
                    members = members[members["is_new"] == "Y"].copy()
                    members["l3_code"] = l3_code
                    all_members.append(members)
                    if (i + 1) % 50 == 0:
                        logger.info(
                            f"  Progress: {i + 1}/{len(l3_codes)}, total {sum(len(x) for x in all_members)} rows"
                        )
            except Exception as e:
                logger.warning(f"  {l3_code}: {e}")
            time.sleep(0.25)

        if not all_members:
            logger.error("No members fetched from any L3 index")
            return None

        combined = pd.concat(all_members, ignore_index=True)
        logger.info(
            f"Total raw members: {len(combined)} rows, cols: {combined.columns.tolist()}"
        )

        # ── Step 3: Normalize ──
        if "l1_name" in combined.columns:
            logger.info("Using rich schema from index_member_all")
            if "is_new" in combined.columns:
                combined = combined[combined["is_new"] == "Y"].copy()
                logger.info(f"After is_new=Y filter: {len(combined)} rows")
            combined = combined.drop_duplicates(subset=["ts_code"], keep="first")
            logger.info(f"After dedup by ts_code: {len(combined)} rows")

            rename = {
                "l1_code": "sw_l1_code",
                "l1_name": "sw_l1_name",
                "l2_code": "sw_l2_code",
                "l2_name": "sw_l2_name",
                "l3_code": "sw_l3_code",
                "l3_name": "sw_l3_name",
            }
            combined = combined.rename(
                columns={k: v for k, v in rename.items() if k in combined.columns}
            )
        else:
            logger.info("Using sparse schema, joining with index_classify hierarchy")
            if "con_code" in combined.columns:
                combined = combined.rename(columns={"con_code": "ts_code"})
            if "is_new" in combined.columns:
                combined = combined[combined["is_new"] == "Y"].copy()
            combined = combined.drop_duplicates(subset=["ts_code"], keep="first")

            if l3_to_parent:
                combined["sw_l3_code"] = combined["index_code"]
                combined["sw_l3_name"] = combined["sw_l3_code"].map(
                    lambda x: l3_to_parent.get(x, {}).get("l3_name", "")
                )
                combined["sw_l2_code"] = combined["sw_l3_code"].map(
                    lambda x: l3_to_parent.get(x, {}).get("l2_code", "")
                )
                combined["sw_l2_name"] = combined["sw_l3_code"].map(
                    lambda x: l3_to_parent.get(x, {}).get("l2_name", "")
                )
                combined["sw_l1_code"] = combined["sw_l3_code"].map(
                    lambda x: l3_to_parent.get(x, {}).get("l1_code", "")
                )
                combined["sw_l1_name"] = combined["sw_l3_code"].map(
                    lambda x: l3_to_parent.get(x, {}).get("l1_name", "")
                )

        if "ts_code" in combined.columns:
            combined["symbol"] = (
                combined["ts_code"].str.replace(".SZ", "").str.replace(".SH", "")
            )

        for c in FINAL_COLS:
            if c not in combined.columns:
                combined[c] = pd.NA
        combined = combined[FINAL_COLS]

        return combined

    except Exception as e:
        logger.error(f"Tushare fetch failed: {e}")
        return None


def fetch_akshare_classification():
    """Fallback: akshare stock_industry_clf_hist_sw() from swsresearch.com."""
    try:
        import akshare as ak

        logger.info("akshare: calling stock_industry_clf_hist_sw()...")
        df = ak.stock_industry_clf_hist_sw()
        logger.info(f"akshare: {len(df)} rows, cols: {df.columns.tolist()}")
        if df.empty:
            return None

        col_map = {
            "股票代码": "symbol",
            "股票名称": "name",
            "行业代码": "sw_l3_code",
            "行业名称": "sw_l3_name",
            "申万一级": "sw_l1_name",
            "申万二级": "sw_l2_name",
            "开始日期": "in_date",
            "结束日期": "out_date",
            "是否最新": "is_new",
        }
        renamed = df.rename(
            columns={k: v for k, v in col_map.items() if k in df.columns}
        )

        if "symbol" in renamed.columns:
            renamed["ts_code"] = renamed["symbol"].apply(
                lambda x: f"{x}.SZ" if str(x).startswith(("0", "3")) else f"{x}.SH"
            )

        if "is_new" in renamed.columns:
            renamed["is_new"] = renamed["is_new"].astype(str).str.strip()
            current = renamed[renamed["is_new"].str.upper().str.startswith("Y")].copy()
            logger.info(f"Filtered to is_new=Y: {len(current)} / {len(renamed)}")
            renamed = current

        renamed = renamed.drop_duplicates(subset=["ts_code"], keep="first")

        for c in FINAL_COLS:
            if c not in renamed.columns:
                renamed[c] = pd.NA
        return renamed[FINAL_COLS]

    except Exception as e:
        logger.error(f"akshare failed: {e}")
        return None


def incremental_update(new_ts_codes):
    """Incrementally update SW classification CSV for new stocks.

    Queries Tushare index_member_all(ts_code=xxx) for each stock not yet
    in the CSV. Appends results to the existing file.

    Args:
        new_ts_codes (list[str]): ts_code strings to check/add.

    Returns:
        int: number of new rows added.
    """
    if not new_ts_codes:
        logger.info("Incremental update: no new stocks to fetch")
        return 0

    logger.info(f"Incremental update: {len(new_ts_codes)} new stocks to fetch")

    try:
        import tushare as ts
        from dotenv import load_dotenv

        load_dotenv()
        token = os.getenv("TUSHARE_TOKEN") or ts.get_token()
        if not token:
            logger.error("Tushare: no token")
            return 0
        pro = ts.pro_api(token)
    except Exception as e:
        logger.error(f"Tushare init failed: {e}")
        return 0

    # Load existing CSV
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH, encoding="utf-8-sig", dtype=str)
        existing_codes = set(existing["ts_code"].tolist())
    else:
        existing = pd.DataFrame()
        existing_codes = set()

    # Filter to truly missing stocks
    truly_new = [c for c in new_ts_codes if c not in existing_codes]
    if not truly_new:
        logger.info("All stocks already in CSV, nothing to do")
        return 0
    logger.info(f"  {len(truly_new)} stocks not in CSV (of {len(new_ts_codes)} given)")

    # Query index_member_all per stock (1 API call each, returns full L1/L2/L3)
    new_rows = []
    for i, ts_code in enumerate(truly_new):
        try:
            df = pro.index_member_all(ts_code=ts_code)
            if len(df) > 0:
                row = df[df["is_new"] == "Y"].head(1)
                if row.empty:
                    row = df.head(1)
                new_rows.append(row)
            if (i + 1) % 20 == 0:
                logger.info(
                    f"  Progress: {i + 1}/{len(truly_new)}, fetched {len(new_rows)} rows"
                )
            time.sleep(0.15)
        except Exception as e:
            logger.warning(f"  {ts_code}: {e}")

    if not new_rows:
        logger.warning("No SW classification data fetched for any new stock")
        return 0

    combined = pd.concat(new_rows, ignore_index=True)
    logger.info(
        f"  Fetched {len(combined)} rows for {combined['ts_code'].nunique()} stocks"
    )

    # Normalize (same as fetch_tushare_classification Step 3)
    rename = {
        "l1_code": "sw_l1_code",
        "l1_name": "sw_l1_name",
        "l2_code": "sw_l2_code",
        "l2_name": "sw_l2_name",
        "l3_code": "sw_l3_code",
        "l3_name": "sw_l3_name",
    }
    combined = combined.rename(
        columns={k: v for k, v in rename.items() if k in combined.columns}
    )

    if "ts_code" in combined.columns:
        combined["symbol"] = (
            combined["ts_code"].str.replace(".SZ", "").str.replace(".SH", "")
        )

    combined["fetch_date"] = datetime.now().strftime("%Y-%m-%d")
    combined["data_source"] = "tushare_incremental"

    for c in FINAL_COLS:
        if c not in combined.columns:
            combined[c] = pd.NA
    combined = combined[FINAL_COLS]

    # Append to existing CSV
    if len(existing) > 0:
        for c in combined.columns:
            if c not in existing.columns:
                existing[c] = pd.NA
        for c in existing.columns:
            if c not in combined.columns:
                combined[c] = pd.NA
        combined = combined[existing.columns]
        updated = pd.concat([existing, combined], ignore_index=True)
    else:
        updated = combined

    updated.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    logger.info(
        f"  Appended {len(combined)} rows. CSV now has {len(updated)} total stocks"
    )
    return len(combined)


def detect_and_update(new_ts_codes):
    """Public API: detect missing stocks and trigger incremental update.

    Used by _daily_fetch.py to auto-update SW classification when new
    stocks appear in the panel.

    Args:
        new_ts_codes (list[str]): ts_code strings to check/add.

    Returns:
        int: number of new rows added.
    """
    return incremental_update(new_ts_codes)


def main():
    """CLI entry point. Supports full rebuild (default) or incremental update."""
    parser = argparse.ArgumentParser(
        description="Fetch Shenwan SW stock classification"
    )
    parser.add_argument(
        "--incremental",
        type=str,
        default=None,
        help="Comma-separated ts_codes to add incrementally",
    )
    parser.add_argument(
        "--incremental-file",
        type=str,
        default=None,
        help="Path to text file with one ts_code per line",
    )
    args = parser.parse_args()

    # ── Incremental mode ──
    if args.incremental or args.incremental_file:
        codes = []
        if args.incremental:
            codes = [c.strip() for c in args.incremental.split(",") if c.strip()]
        if args.incremental_file:
            with open(args.incremental_file, encoding="utf-8") as f:
                codes = [line.strip() for line in f if line.strip()]
        if not codes:
            logger.error("No ts_codes provided for incremental update")
            sys.exit(1)
        n = incremental_update(codes)
        logger.info(f"Incremental update done: {n} new rows added")
        return

    # ── Full rebuild mode (default) ──
    logger.info("=" * 60)
    logger.info("Fetching SW (Shenwan) stock industry classification (full rebuild)")
    logger.info("=" * 60)

    logger.info("\n--- Strategy 1: Tushare index_classify + index_member ---")
    final_df = fetch_tushare_classification()
    source = "tushare"

    if final_df is None or len(final_df) == 0:
        logger.info("\n--- Strategy 2: akshare stock_industry_clf_hist_sw ---")
        final_df = fetch_akshare_classification()
        source = "akshare"

    if final_df is None or len(final_df) == 0:
        logger.error("All sources failed!")
        sys.exit(1)

    final_df["fetch_date"] = datetime.now().strftime("%Y-%m-%d")
    final_df["data_source"] = source
    final_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    logger.info(f"\nSaved {len(final_df)} rows to {OUTPUT_PATH}")

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Source: {source}")
    logger.info(f"Total stocks: {len(final_df)}")
    for col in ["sw_l1_name", "sw_l2_name", "sw_l3_name"]:
        if col in final_df.columns:
            n = final_df[col].notna().sum()
            u = final_df[col].nunique()
            logger.info(f"  {col}: {n} filled, {u} unique values")

    if "sw_l1_name" in final_df.columns:
        logger.info("\nL1 distribution (top 10):")
        for name, count in final_df["sw_l1_name"].value_counts().head(10).items():
            logger.info(f"  {name}: {count} stocks")

    display_cols = [
        c
        for c in ["ts_code", "symbol", "name", "sw_l1_name", "sw_l2_name", "sw_l3_name"]
        if c in final_df.columns
    ]
    logger.info("\nSample (first 10):")
    print(final_df[display_cols].head(10).to_string())

    # Coverage check
    try:
        import tushare as ts
        from dotenv import load_dotenv

        load_dotenv()
        pro = ts.pro_api(os.getenv("TUSHARE_TOKEN") or ts.get_token())
        sb = pro.stock_basic(list_status="L", fields="ts_code,symbol,name")
        panel_stocks = set(sb["ts_code"].tolist())
        mapped_stocks = set(final_df["ts_code"].tolist())
        overlap = panel_stocks & mapped_stocks
        pct = len(overlap) / len(panel_stocks) * 100 if panel_stocks else 0
        logger.info(
            f"\nCoverage: {len(overlap)}/{len(panel_stocks)} stocks ({pct:.1f}%)"
        )
        missing = panel_stocks - mapped_stocks
        if missing and len(missing) <= 30:
            logger.info(f"Missing: {sorted(missing)[:30]}")
        elif missing:
            logger.info(f"Missing: {len(missing)} stocks not in SW classification")
    except Exception as e:
        logger.debug(f"Coverage check failed: {e}")


if __name__ == "__main__":
    main()
