# -*- coding: utf-8 -*-
"""Fix duplicate blocks in _daily_fetch.py caused by patch script."""

p = "_daily_fetch.py"
with open(p, encoding="utf-8") as f:
    text = f.read()

# Remove duplicate sw_daily fetch line (keep first)
text = text.replace(
    'lhb   = safe_fetch(pro.top_list, "LHB", trade_date=TRADE_DATE)\n'
    'sw    = safe_fetch(pro.sw_daily, "sw_daily", trade_date=TRADE_DATE)\n'
    'sw    = safe_fetch(pro.sw_daily, "sw_daily", trade_date=TRADE_DATE)',
    'lhb   = safe_fetch(pro.top_list, "LHB", trade_date=TRADE_DATE)\n'
    'sw    = safe_fetch(pro.sw_daily, "sw_daily", trade_date=TRADE_DATE)'
)

# Remove duplicate stk_limit block (keep first)
old_stk = '''# stk_limit: map source cols to panel column names directly
if len(limit):
    lmap = limit.set_index("symbol")
    for src_col, tgt_col in [("up_limit", "up_limit_raw"), ("down_limit", "down_limit_raw")]:
        if src_col in lmap.columns and tgt_col in panel_cols and tgt_col not in df.columns:
            df[tgt_col] = df["symbol"].map(lmap[src_col])

# stk_limit: map source cols to panel column names directly
if len(limit):
    lmap = limit.set_index("symbol")
    for src_col, tgt_col in [("up_limit", "up_limit_raw"), ("down_limit", "down_limit_raw")]:
        if src_col in lmap.columns and tgt_col in panel_cols and tgt_col not in df.columns:
            df[tgt_col] = df["symbol"].map(lmap[src_col])'''
new_stk = '''# stk_limit: map source cols to panel column names directly
if len(limit):
    lmap = limit.set_index("symbol")
    for src_col, tgt_col in [("up_limit", "up_limit_raw"), ("down_limit", "down_limit_raw")]:
        if src_col in lmap.columns and tgt_col in panel_cols and tgt_col not in df.columns:
            df[tgt_col] = df["symbol"].map(lmap[src_col])'''
text = text.replace(old_stk, new_stk)

# Remove duplicate free_float_turnover_rate block
old_fft = '''# free_float_turnover_rate = turnover_rate_f (free float turnover)
if "turnover_rate_f" in df.columns and "free_float_turnover_rate" in panel_cols:
    df["free_float_turnover_rate"] = df["turnover_rate_f"]

# free_float_turnover_rate = turnover_rate_f
if "turnover_rate_f" in df.columns and "free_float_turnover_rate" in panel_cols:
    df["free_float_turnover_rate"] = df["turnover_rate_f"]'''
new_fft = '''# free_float_turnover_rate = turnover_rate_f
if "turnover_rate_f" in df.columns and "free_float_turnover_rate" in panel_cols:
    df["free_float_turnover_rate"] = df["turnover_rate_f"]'''
text = text.replace(old_fft, new_fft)

# Remove duplicate sw_daily sector index block
# Find the second occurrence and remove it
sw_block_start = "# --- Sector index (sw_daily) ---"
parts = text.split(sw_block_start)
if len(parts) >= 3:
    # Keep first occurrence, remove subsequent
    first = parts[0] + sw_block_start + parts[1]
    # Find where the block ends (next # --- or # ──)
    for p2 in parts[2:]:
        first += p2
    text = first

# Fix duplicate else at end
lines = text.rstrip().split("\n")
# Remove trailing empty lines
while lines and lines[-1].strip() == "":
    lines.pop()
# Check for duplicate else block
if len(lines) >= 4:
    last4 = [line.strip() for line in lines[-4:]]
    if last4[2] == "else:" and last4[3] == 'print("All columns have data!")':
        if len(lines) >= 6:
            last6 = [line.strip() for line in lines[-6:]]
            if last6[4] == "else:" and last6[5] == 'print("All columns have data!")':
                # Remove last 2 lines
                lines = lines[:-2]

text = "\n".join(lines) + "\n"

with open(p, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Fixed: {len(lines)} lines")
