# -*- coding: utf-8 -*-
"""ovd resume (09-09): 18:12:42 os.replace 因瞬时文件锁失败 — tmp 已完整写出,
面板未动 (mtime 16:28). 校验 tmp 完整性后原子换入 + 写 result json.
时限: 须在 cyq 等待者下一 tick (18:18:06) 前完成, 否则 cyq 读到无 ovd 列旧面板."""
import datetime
import json
import os
import time

import pyarrow.parquet as pq

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
TMP = PANEL + ".tmp_ovd_0909"
RESULT = "tmp_t/_ovd_backfill_result.json"
LOG = open("tmp_t/_ovd_backfill_v3_0909.log", "a", encoding="utf-8")


def log(m):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {m}", file=LOG, flush=True)


pf = pq.ParquetFile(TMP)
names = pf.schema_arrow.names
n = pf.num_rows
assert n > 3_000_000, f"tmp 行数异常 {n}"
assert "ovd_wdist" in names and "ovd_15p" in names, "tmp 缺 ovd 列"
nulls = pf.read(columns=["ovd_wdist"]).column(0).null_count
log(f"resume: tmp 校验 rows={n:,} cols={len(names)} ovd_wdist null={nulls:,}")
assert nulls < n, "ovd_wdist 全空"

for i in range(3):
    try:
        os.replace(TMP, PANEL)
        break
    except PermissionError as e:
        log(f"replace 重试 #{i + 1}: {e}")
        if i == 2:
            raise
        time.sleep(5)

ok = pq.ParquetFile(PANEL)
has = [c for c in ("ovd_wdist", "ovd_15p") if c in ok.schema_arrow.names]
assert len(has) == 2, f"回读缺列: {has}"
n2 = ok.num_rows
n2_nulls = ok.read(columns=["ovd_wdist"]).column(0).null_count
cov = {
    "rows_total": int(n2),
    "rows_ovd_nonnan": int(n2 - n2_nulls),
    "coverage_pct": round((n2 - n2_nulls) / n2 * 100, 2),
    "status": "ok",
    "resumed_from_tmp": True,
    "finished": datetime.datetime.now().isoformat(timespec="seconds"),
}
with open(RESULT, "w", encoding="utf-8") as f:
    json.dump(cov, f, ensure_ascii=False, indent=1)
log(f"OVD_BACKFILL_DONE status=ok (resume) coverage={cov['coverage_pct']}% -> {RESULT}")
print("RESUME_OK")
