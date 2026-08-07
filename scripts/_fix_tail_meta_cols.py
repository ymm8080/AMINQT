"""一次性: 修复 V3 面板 07-27..07-31 尾部的 board/industry/is_st/list_days 元数据列.

背景 (2026-08-03): 基建面板 (panel_builder) 的元数据语义是 —
  board     = board_of (main/GEM/STAR, 前缀判断)
  industry  = 东财行业板块 (28 类, 缺省 UNKNOWN)
  is_st     = 名称判断
  list_days = 每股累计交易日数 (cumcount, 每股 +1/交易日)
而旧的 _daily_fetch 曾用 stock_basic 覆盖为 交易所代码(sh/sz) / 109 行业 /
上市日历天数 (600519 748→9100). 尾部 07-27..31 被污染, 此脚本只改这 4 列、
延续基建语义: board 用 board_of 重算; is_st/industry 取每股尾部前的最近值延续;
list_days 按 07-27→07-31 顺序级联 +1.

写回只替换这 4 列的 Arrow array, 其余 97 列复用原 array → schema 精确保留.
运行前自动做 WORM 备份.
"""

import datetime
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from app.pipeline1.cleaning_pipeline import board_of

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
TAIL_DATES = pd.to_datetime(
    ["2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31"]
)
META_COLS = ["board", "is_st", "industry", "list_days"]

stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
bk = PANEL.replace(".parquet", f"_preturnmetafix_{stamp}.parquet")
shutil.copy2(PANEL, bk)
print(f"WORM 备份: {bk}")

t = pq.read_table(PANEL)
df = t.to_pandas()
df["date"] = pd.to_datetime(df["date"])
mask_tail = df["date"].dt.date.isin({d.date() for d in TAIL_DATES})
print(
    f"尾部行: {mask_tail.sum()} (日期: {sorted(set(df.loc[mask_tail, 'date'].dt.date.astype(str)))})"
)

# board: board_of 重算 (与基建一致), 替代 sh/sz
df.loc[mask_tail, "board"] = df.loc[mask_tail, "symbol"].map(board_of)

# is_st/industry: 每股取尾部前最近值延续
pre = df[~mask_tail]
last_pre = pre.sort_values("date").groupby("symbol")[["is_st", "industry"]].last()
for col in ["is_st", "industry"]:
    smap = last_pre[col].dropna()
    df.loc[mask_tail, col] = df.loc[mask_tail, "symbol"].map(smap)

# list_days: 顺序级联 +1 (处理 07-27 时读到的仍是基建值; 后续日期读到已更新的前一日)
for d in TAIL_DATES:
    m = df["date"].dt.date == d.date()
    prior = df[df["date"] < d].sort_values("date").groupby("symbol")["list_days"].last()
    df.loc[m, "list_days"] = df.loc[m, "symbol"].map(prior) + 1

# 校验: 尾部不应再有 sh/sz / 日历天数
tail = df[mask_tail]
assert not (tail["board"].isin(["sh", "sz"])).any(), "board 仍有 sh/sz"
assert tail["list_days"].isna().sum() == 0, "list_days 出现 NaN"
print("board 分布:", tail.groupby("board").size().to_dict())
print(
    "industry nunique:",
    tail["industry"].nunique(),
    "| UNKNOWN:",
    (tail["industry"] == "UNKNOWN").sum(),
)
print(
    "600519 tail list_days:",
    tail[tail["symbol"] == "600519"].sort_values("date")["list_days"].tolist(),
)

# 全 schema 保留写回: 只替换 4 列, 其余列复用原 arrow array
cols = t.column_names
arrays = []
for c in cols:
    if c in META_COLS:
        orig = t.schema.field(c).type
        if c == "is_st":
            arrays.append(pa.array(df[c].fillna(False), type=orig))
        elif c == "list_days":
            arrays.append(pa.array(df[c].fillna(0).astype("int64"), type=orig))
        else:
            arrays.append(pa.array(df[c], type=orig))
    else:
        arrays.append(t.column(c))
new_t = pa.Table.from_arrays(arrays, schema=t.schema)
pq.write_table(new_t, PANEL)
print("写回完成:", PANEL, "| rows:", len(df), "| cols:", len(cols))
