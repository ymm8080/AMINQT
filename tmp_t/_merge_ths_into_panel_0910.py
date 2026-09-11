# -*- coding: utf-8 -*-
"""把 ths_signal parts (+7列) 并入生产面板 V3: tmp写入→备份→原子替换→验证."""
import io
import os
import shutil
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pandas as pd

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
PART = r"D:/AMINQT/AMINQT CODES/data/enrich_parts/ths_signal.parquet"
BAK = PANEL + ".bak_0910_ths"
TMP = PANEL + ".tmp_ths"

panel = pd.read_parquet(PANEL)
part = pd.read_parquet(PART)
print(f"[pre] panel {len(panel)} rows {len(panel.columns)} cols | part {len(part)} rows")

ths_cols = [c for c in part.columns if c not in ("symbol", "date")]
overlap = [c for c in ths_cols if c in panel.columns]
if overlap:
    # 回填完成后的复跑: 覆盖旧列 (回填快照扩大 → 值必须整体刷新)
    if "--replace" not in sys.argv:
        raise SystemExit(f"面板已有 ths 列, 复跑请加 --replace: {overlap}")
    print(f"[replace] drop 旧 ths 列 {len(overlap)} 个后重并")
    panel = panel.drop(columns=overlap)
n0 = len(panel)
panel = panel.merge(part, on=["symbol", "date"], how="left")
assert len(panel) == n0, "合并改变行数!"
print(f"[merge] +{len(ths_cols)} cols: {ths_cols}")
for c in ths_cols:
    nn = int(panel[c].notna().sum())
    print(f"  {c}: non-null {nn}")

panel.to_parquet(TMP, index=False)
if not os.path.exists(BAK):
    shutil.copy2(PANEL, BAK)
    print(f"[bak] {BAK}")
import time as _time

for attempt in range(6):
    try:
        os.replace(TMP, PANEL)
        break
    except PermissionError as e:
        if attempt == 5:
            raise
        print(f"[lock] 目标被占用 ({e.strerror}), 20s后重试 {attempt + 2}/6")
        _time.sleep(20)
print("[done] 生产面板已替换")

# ── 验证: 重读, 抽样 + 覆盖统计 ──
v = pd.read_parquet(PANEL, columns=["symbol", "date"] + ths_cols)
print(f"[verify] reload {len(v)} rows, {len(v.columns)} cols")
row = v[(v["symbol"] == "688559") & (v["date"] == pd.Timestamp("2023-09-05"))]
if len(row):
    r = row.iloc[0]
    print(f"[sample] 688559@2023-09-05: ths_bull={r['ths_bull']} tech_n={r['ths_bull_tech_n']} "
          f"tech_pattern={r['ths_bull_tech_pattern']!r} buy={r['ths_bull_buy_signals']!r}")
per_day = v[v["ths_bull"] == 1.0].groupby(v["date"].dt.strftime("%Y%m%d"))["ths_bull"].sum()
print(f"[cover] ths_bull=1 天数={len(per_day)} 首={per_day.index[0]} 末={per_day.index[-1]} "
      f"日均={per_day.mean():.1f}只")
print(f"[verify] max_date={v['date'].max().date()} (须仍为 2026-09-09)")
