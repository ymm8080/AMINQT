"""Check stock_basic.industry: source, coverage, history."""

import os

import tushare as ts
from dotenv import load_dotenv

load_dotenv()

pro = ts.pro_api(ts.get_token() or os.getenv("TUSHARE_TOKEN"))

# 1. stock_basic industry field
sb = pro.stock_basic(list_status="L", fields="ts_code,symbol,name,industry,list_date")
print(f"stock_basic: {len(sb)} stocks")
print(f"Unique industries: {sb['industry'].nunique()}")
print(f"NaN industries: {sb['industry'].isna().sum()}")

# 2. Check list_date range (history)
sb["list_date"] = sb["list_date"].astype(str)
oldest = sb["list_date"].min()
newest = sb["list_date"].max()
print(f"List date range: {oldest} ~ {newest}")

# 3. Check if industry has changed over time (is it point-in-time?)
# stock_basic is current snapshot - NOT historical
print("\nNOTE: stock_basic is a CURRENT SNAPSHOT")
print("  - industry = Tushare's OWN classification (not SW)")
print("  - No historical changes tracked (no start/end date)")
print("  - For historical industry: need pro.stock_company or daily_basic")

# 4. Check stock_company for historical industry
try:
    sc = pro.stock_company(
        fields="ts_code,exchange,province,city,introduction,main_business"
    )
    print(f"\nstock_company: {len(sc)} rows, columns: {sc.columns.tolist()}")
except Exception as e:
    print(f"\nstock_company: {e}")

# 5. Check daily_basic for industry-equivalent fields
try:
    db = pro.daily_basic(trade_date="20260731")
    print(f"\ndaily_basic columns: {db.columns.tolist()}")
except Exception as e:
    print(f"\ndaily_basic: {e}")
