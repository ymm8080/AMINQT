#!/usr/bin/env python3
"""检查 Tushare token 权限等级."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv

load_dotenv()
import tushare as ts

token = os.getenv("TUSHARE_TOKEN")
if not token:
    print("TUSHARE_TOKEN 未配置!")
    sys.exit(1)

ts.set_token(token)
pro = ts.pro_api()

# 查看用户积分
try:
    # 尝试获取用户信息
    import requests

    url = "http://api.tushare.pro"
    payload = {
        "api_name": "user",
        "token": token,
        "params": {},
        "fields": "",
    }
    resp = requests.post(url, json=payload, timeout=10)
    data = resp.json()
    if data.get("code") == 0:
        print("用户信息:")
        items = data.get("data", {}).get("items", [])
        if items:
            for k, v in items[0].items():
                print(f"  {k}: {v}")
        else:
            print("  (无 items)")
        print(f"  fields: {data.get('data', {}).get('fields', [])}")
    else:
        print(f"API 返回: {data}")
except Exception as e:
    print(f"查询用户信息失败: {e}")

# 测试 cyq_perf 权限
print("\n测试 cyq_perf 权限:")
try:
    df = pro.cyq_perf(ts_code="000001.SZ", start_date="20250101", end_date="20250110")
    print(f"  成功! 返回 {len(df)} 行")
    print(f"  列: {df.columns.tolist()}")
except Exception as e:
    print(f"  失败: {e}")
