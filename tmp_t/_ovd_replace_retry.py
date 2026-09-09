# -*- coding: utf-8 -*-
"""ovd 回补 os.replace 重试 (backfill 18:12:42 WinError5 后的补刀).

前提: .tmp_ovd_0909 已完整写出 (traceback 发生在 replace 而非 write),
.bak_0909_ovd WORM 备份已建, 面板本体=ohlc修复版未动.
10s 轮询最多 3min; 成功后校验 schema 带 ovd 列 + 行数, 改写 result json.
"""
import json
import os
import time
from datetime import datetime

import pyarrow.parquet as pq

TMP = "D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet.tmp_ovd_0909"
DST = "D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
RESULT = r"D:\AMINQT\AMINQT CODES\tmp_t\_ovd_backfill_result.json"

assert os.path.exists(TMP), "tmp 文件不存在, 不能重试"

ok = False
for i in range(18):
    try:
        os.replace(TMP, DST)
        ok = True
        print(f"[{datetime.now():%H:%M:%S}] REPLACE_OK attempt={i + 1}")
        break
    except PermissionError as e:
        print(f"[{datetime.now():%H:%M:%S}] retry {i + 1}/18: {e}", flush=True)
        time.sleep(10)

if not ok:
    print("REPLACE_RETRY_FAILED — 句柄被持续占用, 需人工定位占用进程")
    raise SystemExit(2)

pf = pq.ParquetFile(DST)
names = pf.schema_arrow.names
assert "ovd_wdist" in names and "ovd_15p" in names, f"ovd 列缺失: 只剩核对"
print(f"verify: rows={pf.num_rows:,} cols={pf.num_columns} "
      f"ovd_wdist={'ovd_wdist' in names} ovd_15p={'ovd_15p' in names}")

res = {}
if os.path.exists(RESULT):
    with open(RESULT, encoding="utf-8") as f:
        res = json.load(f)
res["replace_manual_fix"] = {
    "fixed_at": datetime.now().isoformat(timespec="seconds"),
    "note": "18:12:42 WinError5 后 os.replace 重试成功 (tmp 完整, 面板未受损)",
    "rows": int(pf.num_rows), "cols": int(pf.num_columns),
}
with open(RESULT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("OVD_REPLACE_FIX_DONE")
