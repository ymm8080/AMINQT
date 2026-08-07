"""Final comprehensive patch for _daily_fetch.py."""

import pathlib

p = pathlib.Path("_daily_fetch.py")
c = p.read_text(encoding="utf-8")

# 1. PANEL path
c = c.replace(
    'PANEL = "data/panel_full_enriched_v3.parquet"',
    'PANEL = os.getenv("PANEL_PATH", r"D:\\AMINQT\\PARQUET\\panel_full_enriched_v3.parquet")',
)

# 2. Token
c = c.replace(
    'pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))',
    '_token = os.getenv("TUSHARE_TOKEN") or ts.get_token()\n'
    'if not _token:\n    print("FATAL: No Tushare token"); sys.exit(1)\npro = ts.pro_api(_token)',
)

# 3. sw_daily fetch
c = c.replace(
    'lhb   = safe_fetch(pro.top_list, "LHB", trade_date=TRADE_DATE)',
    'lhb   = safe_fetch(pro.top_list, "LHB", trade_date=TRADE_DATE)\n'
    'sw    = safe_fetch(pro.sw_daily, "sw_daily", trade_date=TRADE_DATE)',
)

# 4. Remove turnover_rate, turnover_rate_f, stk_limit from merges
c = c.replace(
    '"daily_basic": (basic, ["turnover_rate", "turnover_rate_f", "volume_ratio",',
    '"daily_basic": (basic, ["volume_ratio",',
)
c = c.replace('    "stk_limit": (limit, ["up_limit", "down_limit"]),\n', "")

# 5. Add special cases after merge loop
old_after = "# Rename mappings"
new_after = """# Special: columns not in panel_cols but needed for derived
if len(cyq) and "winner_rate" in cyq.columns and "winner_rate" not in df.columns:
    wmap = cyq.set_index("symbol")["winner_rate"]
    df["winner_rate"] = df["symbol"].map(wmap)

if len(basic):
    bmap = basic.set_index("symbol")
    if "turnover_rate_f" in bmap.columns and "free_float_turnover_rate" in panel_cols:
        df["free_float_turnover_rate"] = df["symbol"].map(bmap["turnover_rate_f"])

# stk_limit direct mapping
if len(limit):
    lmap = limit.set_index("symbol")
    for src_col, tgt_col in [("up_limit", "up_limit_raw"), ("down_limit", "down_limit_raw")]:
        if src_col in lmap.columns and tgt_col in panel_cols:
            df[tgt_col] = df["symbol"].map(lmap[src_col])

# Rename mappings"""
c = c.replace(old_after, new_after)

# 6. Fix rename_map
c = c.replace(
    '    "up_limit": "up_limit_raw", "down_limit": "down_limit_raw",\n    ', ""
)
c = c.replace('"winner_rate": "_winner_rate_tmp",\n    ', "")

# 7. Replace benefit_part computation with winner_ratio
c = c.replace(
    '# --- winner_ratio = winner_rate / 100 ---\nif "_winner_rate_tmp" in df.columns:\n    df["winner_ratio"] = df["_winner_rate_tmp"] / 100.0',
    '# --- winner_ratio = winner_rate (0-100, percentage) ---\nif "winner_rate" in df.columns and "winner_ratio" in panel_cols:\n    df["winner_ratio"] = df["winner_rate"]',
)

# 8. Remove turn mapping
c = c.replace(
    'if "turn" in panel_cols and "turnover_rate" in df.columns:\n    df["turn"] = df["turnover_rate"]\n\n',
    "",
)

# 9. Add sw_daily sector index block before simple derived
sw_block = """# --- Sector index (sw_daily) ---
if len(sw) and "industry" in df.columns:
    sw_map = {}
    for _, row in sw.iterrows():
        idx_name = str(row.get("name", "")).strip()
        if idx_name:
            sw_map[idx_name] = {"close": row.get("close"), "vol": row.get("vol"), "pct_change": row.get("pct_change")}
    # Fuzzy match industry -> SW name
    ind_to_sw = {}
    for ind in df["industry"].dropna().unique():
        ind_s = str(ind).strip()
        for sw_n in sw_map:
            if ind_s in sw_n or sw_n in ind_s:
                ind_to_sw[ind_s] = sw_n
                break
    for tgt_col, sw_key in [("sw_index_close","close"),("sw_index_vol","vol"),("sw_ret_1d","pct_change")]:
        if tgt_col in panel_cols:
            vals = {}
            for _, row in df.iterrows():
                sw_n = ind_to_sw.get(str(row.get("industry","")).strip())
                if sw_n and pd.notna(sw_map[sw_n].get(sw_key)):
                    vals[row["symbol"]] = sw_map[sw_n][sw_key]
            if vals:
                df[tgt_col] = df["symbol"].map(vals)
    n_filled = df["sw_index_close"].notna().sum() if "sw_index_close" in df.columns else 0
    print(f"    sw_daily: {len(sw)} indices -> {n_filled}/{len(df)} stocks")

"""
c = c.replace(
    "# --- Simple derived features", sw_block + "# --- Simple derived features"
)

# 10. Fix ffill list
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
    "ar_turnover", "profit_ratio",
    "holder_count", "sh_change_vol", "sh_change_amt", "sh_change_amt_total",
    "sh_net_change_sign", "sh_net_sign",
    "sw_index_close", "sw_index_vol", "sw_ret_1d",
    "margin_balance", "short_balance", "margin_buy_amt", "short_sell_vol",
    "dt_eps", "q_ocf_to_sales", "announce_date",
]"""
c = c.replace(old_ffill, new_ffill)

# 11. Fix ffill logic
c = c.replace(
    '    last_per_stock = ffill_hist.groupby("symbol").last()',
    '    for col in needed_ffill:\n        if col in ffill_hist.columns:\n            ffill_hist[col] = ffill_hist.groupby("symbol")[col].ffill()\n    last_per_stock = ffill_hist.groupby("symbol").last()',
)

# 12. Fix audit
c = c.replace(
    'print(f"Panel: {pf2.metadata.num_rows:,} rows")',
    'print(f"Panel: {pf2.metadata.num_rows:,} rows, {len(pf2.schema_arrow.names)} cols")',
)

p.write_text(c, encoding="utf-8")
print(f"Patched: {len(c.splitlines())} lines")

# Verify
import py_compile

py_compile.compile(str(p), doraise=True)
print("Syntax OK")
