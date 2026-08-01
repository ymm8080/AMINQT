# -*- coding: utf-8 -*-
"""Patch _daily_fetch.py with all fixes."""

import pathlib

p = pathlib.Path("_daily_fetch.py")
c = p.read_text(encoding="utf-8")

# PANEL path
c = c.replace(
    'PANEL = "data/panel_full_enriched_v3.parquet"',
    'PANEL = os.getenv("PANEL_PATH", r"D:\\AMINQT\\PARQUET\\panel_full_enriched_v3.parquet")',
)

# Token
c = c.replace(
    'pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))',
    '_token = os.getenv("TUSHARE_TOKEN") or ts.get_token()\n'
    'if not _token:\n    print("FATAL: No Tushare token"); sys.exit(1)\npro = ts.pro_api(_token)',
)

# sw_daily fetch
c = c.replace(
    'lhb   = safe_fetch(pro.top_list, "LHB", trade_date=TRADE_DATE)',
    'lhb   = safe_fetch(pro.top_list, "LHB", trade_date=TRADE_DATE)\n'
    'sw    = safe_fetch(pro.sw_daily, "sw_daily", trade_date=TRADE_DATE)',
)

# Remove turnover_rate, turnover_rate_f, stk_limit from merges
c = c.replace(
    '"daily_basic": (basic, ["turnover_rate", "turnover_rate_f", "volume_ratio",',
    '"daily_basic": (basic, ["volume_ratio",',
)
c = c.replace('    "stk_limit": (limit, ["up_limit", "down_limit"]),\n', "")

# Add special cases + stk_limit fix before rename_map
special = """# Special: columns not in panel_cols but needed for derived computation
if len(cyq) and "winner_rate" in cyq.columns and "winner_rate" not in df.columns:
    wmap = cyq.set_index("symbol")["winner_rate"]
    df["winner_rate"] = df["symbol"].map(wmap)

if len(basic):
    bmap = basic.set_index("symbol")
    for col in ["turnover_rate", "turnover_rate_f"]:
        if col in bmap.columns and col not in df.columns:
            df[col] = df["symbol"].map(bmap[col])

# stk_limit: map source cols to panel column names directly
if len(limit):
    lmap = limit.set_index("symbol")
    for src_col, tgt_col in [("up_limit", "up_limit_raw"), ("down_limit", "down_limit_raw")]:
        if src_col in lmap.columns and tgt_col in panel_cols and tgt_col not in df.columns:
            df[tgt_col] = df["symbol"].map(lmap[src_col])

# Rename mappings"""
c = c.replace("# Rename mappings", special)

# Fix rename_map
c = c.replace(
    '    "up_limit": "up_limit_raw", "down_limit": "down_limit_raw",\n    ', ""
)

# Add free_float_turnover_rate mapping
c = c.replace(
    "# LHB merge",
    "# free_float_turnover_rate = turnover_rate_f\n"
    'if "turnover_rate_f" in df.columns and "free_float_turnover_rate" in panel_cols:\n'
    '    df["free_float_turnover_rate"] = df["turnover_rate_f"]\n\n# LHB merge',
)

# Remove old turn mapping
c = c.replace(
    'if "turn" in panel_cols and "turnover_rate" in df.columns:\n    df["turn"] = df["turnover_rate"]\n\n',
    "",
)

# Add sw_daily sector index block
sw_block = """# --- Sector index (sw_daily) ---
if len(sw) and "industry" in df.columns:
    sw_map = {}
    for _, row in sw.iterrows():
        idx_name = str(row.get("name", "")).strip()
        if idx_name:
            sw_map[idx_name] = {"close": row.get("close"), "vol": row.get("vol"), "pct_change": row.get("pct_change")}
    for tgt_col, sw_key in [("sw_index_close", "close"), ("sw_index_vol", "vol"), ("sw_ret_1d", "pct_change")]:
        if tgt_col in panel_cols:
            vals = {}
            for _, row in df.iterrows():
                ind = str(row.get("industry", "")).strip()
                if ind in sw_map and pd.notna(sw_map[ind].get(sw_key)):
                    vals[row["symbol"]] = sw_map[ind][sw_key]
            if vals:
                df[tgt_col] = df["symbol"].map(vals)
    n_filled = df["sw_index_close"].notna().sum() if "sw_index_close" in df.columns else 0
    print(f"    sw_daily: {len(sw)} indices -> {n_filled}/{len(df)} stocks mapped")

"""
c = c.replace(
    "# --- Simple derived features", sw_block + "# --- Simple derived features"
)

# Fix ffill list
old_ffill = """ffill_cols = [
    "roe", "roa", "gross_margin", "net_margin", "eps_yoy", "rev_yoy", "profit_yoy",
    "debt_ratio", "current_ratio", "asset_turnover", "inventory_turnover",
    "ocf_to_or", "eps", "bps", "ocfps", "revenue_ps",
    "roe_deducted", "roe_yoy", "q_roe",
    "ar_turnover", "free_float_turnover_rate",
    "holder_count", "sh_change_amt", "sh_change_amt_total",
    "sh_change_vol", "sh_net_change_sign", "sh_net_sign",
    "sw_index_close", "sw_index_vol", "sw_ret_1d",
    "chip_concentration", "profit_ratio",
    "margin_balance", "short_balance", "margin_buy_amt", "short_sell_vol",
    "dt_eps", "q_ocf_to_sales",
]"""
new_ffill = """ffill_cols = [
    "roe", "roa", "gross_margin", "net_margin", "eps_yoy", "rev_yoy", "profit_yoy",
    "debt_ratio", "current_ratio", "asset_turnover", "inventory_turnover",
    "ocf_to_or", "eps", "bps", "ocfps", "revenue_ps",
    "roe_deducted", "roe_yoy", "q_roe",
    "ar_turnover",
    "holder_count", "sh_change_vol", "sh_change_amt", "sh_change_amt_total",
    "sh_net_change_sign", "sh_net_sign",
    "sw_index_close", "sw_index_vol", "sw_ret_1d",
    "margin_balance", "short_balance", "margin_buy_amt", "short_sell_vol",
    "dt_eps", "q_ocf_to_sales",
    "announce_date",
]"""
c = c.replace(old_ffill, new_ffill)

# Fix audit
c = c.replace(
    'print(f"Panel: {pf2.metadata.num_rows:,} rows")',
    'print(f"Panel: {pf2.metadata.num_rows:,} rows, {len(pf2.schema_arrow.names)} cols")',
)
c = c.replace(
    """if empty_cols:
    print(f"Empty ({len(empty_cols)}): {', '.join(empty_cols)}")""",
    """if empty_cols:
    print(f"Empty ({len(empty_cols)}): {', '.join(empty_cols)}")
else:
    print("All columns have data!")""",
)

# Fix pct_con comment
c = c.replace(
    "# Formula from cyq_calculator.py: concentration = (hi-lo)/(hi+lo)",
    "# Formula from cyq_calculator.py: concentration = (hi-lo)/(hi+lo)\n# Result in [0, 1]: smaller = more concentrated (bullish)",
)

# Update docstring
c = c.replace(
    "- pct_70_con, pct_90_con (cyq_calculator formula)",
    "- pct_70_con, pct_90_con (cyq formula: (hi-lo)/(hi+lo), [0,1] range)",
)
c = c.replace(
    "- Forward-fill: financials, margin(T+1 gap), SW indices, shareholders",
    "- Forward-fill: financials (quarterly), margin(T+1 gap), announce_date\n\n"
    "NOT fetched here (separate pipelines):\n"
    "  - fina_indicator -> run_announcement_pipeline.py (Pipeline 2, quarterly)\n"
    "  - holdertrade/holdernumber -> run_announcement_pipeline.py (Pipeline 2, quarterly/event)\n"
    "  Both are forward-filled from panel history in this script.",
)

p.write_text(c, encoding="utf-8")
print(f"OK: patched, {len(c.splitlines())} lines")
