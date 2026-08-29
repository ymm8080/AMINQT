"""解禁 (share_float) + 业绩预告 (forecast) 接口可得性探针 (一次性, 只读).

回答三件事: 积分档位够不够 / 字段口径 / 历史深度. 不落盘任何业务数据.
"""

import sys

import tushare as ts

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from config.settings import TUSHARE_TOKEN

pro = ts.pro_api(TUSHARE_TOKEN)


def probe(name, func, **kw):
    try:
        df = func(**kw)
        print(f"[{name}] rows={len(df)}")
        if len(df):
            print(df.head(3).to_string())
        return df
    except Exception as e:
        print(f"[{name}] FAIL: {e}")
        return None


# 1) 限售解禁 — 近一个月 + 2024 同窗口 (验历史深度)
probe(
    "float_recent",
    pro.share_float,
    ann_date="",
    start_date="20260801",
    end_date="20260825",
    fields="ts_code,ann_date,float_date,float_share,float_ratio,holder_name",
)
probe(
    "float_2024",
    pro.share_float,
    start_date="20240801",
    end_date="20240825",
    fields="ts_code,ann_date,float_date,float_share,float_ratio,holder_name",
)

# 2) 业绩预告 — 2026 中报期 (刚披露完) + 2024 中报期 (验深度)
probe(
    "forecast_2026H1",
    pro.forecast,
    period="20260630",
    fields="ts_code,ann_date,end_date,type,p_change_min,p_change_max,net_profit_min,net_profit_max,update_flag",
)
probe(
    "forecast_2024H1",
    pro.forecast,
    period="20240630",
    fields="ts_code,ann_date,end_date,type,p_change_min,p_change_max",
)

# 3) 用户积分 (决定接口档位)
try:
    ui = pro.query("user_common")
    print("[user] ", ui.to_string() if len(ui) else "no info")
except Exception as e:
    print(f"[user] FAIL: {e}")
