# -*- coding: utf-8 -*-
"""只读探针: 核实 3 个探索代理报告的活跃静默停更 (2026-09-02 审计).

核对象:
1. sw_daily_history.parquet 冻结 @07-31 (dim28 上游, 无日更任务)
2. V3 panel announce_date 冻结 @08-14 (anns_d 无权限, dim31 上游)
3. V3 panel sw_ret_1d 08-27/09-01 整列空 (index_daily 静默失败)
4. V3 panel fina 列 (roe) 近期非空数恒定 = 冻结 (Pipeline2 只写缓存)
5. parallel 检查点 vs 面板日期差
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, r"D:\AMINQT\AMINQT CODES")
from config.settings import PANEL_V3_PATH, DATA_DIR  # noqa: E402

TAIL_N = 12


def max_date(path):
    schema = pq.read_schema(path)
    col = "date" if "date" in schema.names else "trade_date"
    d = pq.read_table(path, columns=[col])
    return pd.to_datetime(d[col].to_pandas()).max()


print("=" * 72)
print("[1] sw_daily_history.parquet (dim28 上游 39 列)")
p = DATA_DIR / "processed" / "sw_daily_history.parquet"
print(f"    exists={p.exists()}  max(date)={max_date(p)}")

print("[5a] parallel 检查点 main / dual")
for tag in ("main", "dual"):
    p = DATA_DIR / f"_diag_stage_{tag}_3y.parquet"
    print(f"    {tag}: exists={p.exists()}  max(date)={max_date(p)}")
for tag in ("", "_full"):
    p = DATA_DIR / f"cyq_panel{tag}.parquet"
    print(f"[5b] cyq_panel{tag or ''}.parquet: exists={p.exists()}  max(date)={max_date(p)}")

print("=" * 72)
print("[2][3][4] V3 面板尾部逐日核查")
cols = ["date", "symbol", "announce_date", "sw_ret_1d", "roe"]
t = pq.read_table(PANEL_V3_PATH, columns=cols).to_pandas()
t["date"] = pd.to_datetime(t["date"])
print(f"    panel max(date)={t['date'].max()}  rows={len(t):,}")

tail = t[t["date"] >= t["date"].max() - pd.Timedelta(days=TAIL_N * 2)]
g = tail.groupby("date")
summary = pd.DataFrame({
    "rows": g.size(),
    "ann_max": g["announce_date"].max(),
    "sw_ret_nonnull": g["sw_ret_1d"].count(),
    "sw_ret_nonzero": g["sw_ret_1d"].apply(lambda s: (s.fillna(0) != 0).sum()),
    "roe_nonnull": g["roe"].count(),
})
pd.set_option("display.width", 120)
print(summary.to_string())
print("=" * 72)
print("判读基准: ann_max 应≈当日(事件型,滞后可接受); sw_ret_nonnull 应≈rows;")
print("          roe_nonnull 若尾部恒定不变 = fina 面板列冻结")
