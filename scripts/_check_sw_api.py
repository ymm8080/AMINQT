"""Check Tushare APIs for SW industry membership."""

import os

import tushare as ts
from dotenv import load_dotenv

load_dotenv()

pro = ts.pro_api(ts.get_token() or os.getenv("TUSHARE_TOKEN"))

# Write to file for proper UTF-8
with open("scripts/_sw_api_check.txt", "w", encoding="utf-8") as f:
    # 1. index_classify - SW classification levels
    try:
        ic = pro.index_classify(level="L1", src="SW")
        f.write(f"index_classify L1 SW: {len(ic)} rows\n")
        f.write(f"  Columns: {ic.columns.tolist()}\n")
        for _, r in ic.iterrows():
            f.write(f"  {r['index_code']}  {r['industry_name']}\n")
    except Exception as e:
        f.write(f"index_classify L1: {e}\n")

    f.write("\n")
    try:
        ic2 = pro.index_classify(level="L2", src="SW")
        f.write(f"index_classify L2 SW: {len(ic2)} rows\n")
        f.write(f"  Columns: {ic2.columns.tolist()}\n")
        for _, r in ic2.iterrows():
            f.write(f"  {r['index_code']}  {r['industry_name']}\n")
    except Exception as e:
        f.write(f"index_classify L2: {e}\n")

    # 2. index_member - stock membership in SW indices
    f.write("\n")
    try:
        im = pro.index_member(id="801010.SI")
        f.write(f"index_member 801010.SI (农林牧渔): {len(im)} rows\n")
        f.write(f"  Columns: {im.columns.tolist()}\n")
        f.write(f"  Sample:\n{im.head(3).to_string()}\n")
    except Exception as e:
        f.write(f"index_member: {e}\n")

    # 3. stock_basic with all fields
    f.write("\n")
    try:
        sb = pro.stock_basic(
            list_status="L", fields="ts_code,symbol,name,area,industry,market,list_date"
        )
        f.write(
            f"stock_basic: {len(sb)} stocks, {sb['industry'].nunique()} unique industries\n"
        )
        f.write(f"  Columns: {sb.columns.tolist()}\n")
    except Exception as e:
        f.write(f"stock_basic: {e}\n")

    # 4. Try stock_company
    f.write("\n")
    try:
        sc = pro.stock_company(
            fields="ts_code,exchange,province,city,introduction,main_business,employee_count"
        )
        f.write(f"stock_company: {len(sc)} rows\n")
        f.write(f"  Columns: {sc.columns.tolist()}\n")
    except Exception as e:
        f.write(f"stock_company: {e}\n")

    # 5. Check daily_basic for industry fields
    f.write("\n")
    try:
        db = pro.daily_basic(trade_date="20260731")
        f.write(f"daily_basic: {len(db)} rows\n")
        f.write(f"  Columns: {db.columns.tolist()}\n")
    except Exception as e:
        f.write(f"daily_basic: {e}\n")

    # 6. Try sw_index_daily or other SW APIs
    f.write("\n")
    try:
        # Check if daily_basic has industry_name or similar
        f.write(f"daily_basic sample ts_code: {db['ts_code'].iloc[0]}\n")
    except Exception:
        pass

print("Done - see scripts/_sw_api_check.txt")
