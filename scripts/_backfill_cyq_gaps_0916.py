# -*- coding: utf-8 -*-
"""0916: CYQ 缺行回填 (幂等, 只补缺, 不覆盖已有值).

来源两类:
  A. 2023-09-18..11-14 260 对 (date,symbol) 缺 CYQ 5 列:
     - Tushare 可召回的 (000595×5 / 000705×22 / 000757×1 = 28 对) 按源填入;
     - 上游全窗无源 (000505/000517/000552/000553/000736) 232 对 → 保持 NaN 不硬造.
  B. cost_bias 派生空洞 (2026-03-20 缺1695 / 2026-06-10 缺1711 / 2023 两日随A):
     公式 = (close_hfq - cost_50pct) / cost_50pct, 与 _daily_fetch.py:603 一致.

写法: 读全 panel → 改缺失位 → 全量重写 parquet (15.8G RAM, pyarrow 全读会 OOM,
改用 row-group 流式: 只需改行的 rg 才物化, 其余 rg 原样字节复制).
保留旧档 backup 后原子 rename, WORM 不覆盖.
"""

import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tushare as ts

ts.set_token("ff1a00b005486505d1bdd87c72d63206d72f6a3ec0cdc062ec867a96")
pro = ts.pro_api()

PANEL = os.getenv("PANEL_PATH", r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet")
BACKUP = PANEL.replace(".parquet", "_pre_cyq_backfill_20260916.parquet")
CYQ_COLS = ["cost_50pct", "cost_95pct", "pct_90_con", "pct_90_high", "winner_ratio"]

# ── 1. 读需要的列 (列式读, 面板 350w 行无压力) ──
need = [
    "date",
    "symbol",
    "close_hfq",
    "cost_50pct",
    "cost_95pct",
    "pct_90_con",
    "pct_90_high",
    "winner_ratio",
    "cost_bias",
]
t_all = pq.read_table(PANEL, columns=need).to_pandas()
t_all["date"] = pd.to_datetime(t_all["date"])

# ── 2. A 段: 2023 缺行 → Tushare 拉取 ──
# 窗口 = 面板全部 2023 (2023-01-03..2023-11-14), 9-16 前的月份一并审计
base_t = t_all[t_all["date"] <= pd.Timestamp("2023-11-14")].copy()
miss_mask = base_t["cost_50pct"].isna()
miss = base_t[miss_mask]
syms_missing = sorted(miss["symbol"].unique())
print(f"[A] 2023 缺行: {len(miss)} 对, {len(syms_missing)} 票")

cnt = miss.groupby("symbol").size().sort_values(ascending=False)
print(cnt.head(12).to_string())

fetched = {}
if len(miss):
    for sym in syms_missing:
        suffix = (
            ".SZ"
            if sym.startswith(("0", "3"))
            else ".SH"
            if sym.startswith("6")
            else None
        )
        if suffix is None:
            print(f"    {sym}: 非 SZ/SH, 跳过")
            continue
        d = base_t[(base_t["symbol"] == sym) & base_t["cost_50pct"].isna()]["date"]
        d0 = d.min().strftime("%Y%m%d")
        d1 = d.max().strftime("%Y%m%d")
        # 限速: cyq 逐只拉, 温和节流
        time.sleep(0.12)
        try:
            r = pro.cyq_perf(ts_code=sym + suffix, start_date=d0, end_date=d1)
        except Exception as e:
            print(f"    {sym}: fetch ERR {e}; skip")
            continue
        if not len(r):
            print(f"    {sym}: 源无数据 (0 rows)")
            continue
        r = r.rename(columns={"trade_date": "date"})
        r["symbol"] = r["ts_code"].str[:6]
        r["date"] = pd.to_datetime(r["date"])
        # Tushare 原始字段 → 面板口径 (与 _daily_fetch.py:396-404 公式一致):
        #   pct_90_high = cost_95pct
        #   pct_90_con  = (cost_95pct - cost_5pct) / (cost_95pct + cost_5pct)
        #   winner_ratio = winner_rate / 100
        r["pct_90_high"] = r["cost_95pct"]
        r["pct_90_con"] = (r["cost_95pct"] - r["cost_5pct"]) / (
            r["cost_95pct"] + r["cost_5pct"]
        )
        r["winner_ratio"] = r["winner_rate"] / 100.0
        fetched[sym] = r.set_index("date")[
            ["cost_50pct", "cost_95pct", "pct_90_con", "pct_90_high", "winner_ratio"]
        ].dropna(how="all")
        print(f"    {sym}: fetched {len(r)} rows in [{d0}..{d1}]")

# ── 3. 构造写入映射 (idx → dict-of-col-vals) ──
t_all = t_all.set_index(["date", "symbol"], drop=False)
changes_A = 0
changes_B = 0

fill_A = {}
for sym, fdf in fetched.items():
    for d, row in fdf.iterrows():
        if (d, sym) in t_all.index:
            fill_A[(d, sym)] = {c: row.get(c, np.nan) for c in CYQ_COLS}

# ── 4. B 段: cost_bias 重算 (只对 NaN 处) ──
cb_mask = (
    t_all["cost_bias"].isna() & t_all["cost_50pct"].notna() & t_all["close_hfq"].notna()
)
print(f"[B] cost_bias NaN 但 base 有值可算: {cb_mask.sum()} 行")
cb_target = t_all[cb_mask]
chg_by_date = cb_target["date"].value_counts().sort_index()
print("    per day:")
print(chg_by_date.to_string())

# ── 5. 落盘: row-group 流式 ──
pf = pq.ParquetFile(PANEL)
schema = pf.schema_arrow
tmp_path = PANEL + ".tmp"
writer = pq.ParquetWriter(tmp_path, schema=schema)

examples_printed = 0
for rg in range(pf.metadata.num_row_groups):
    rg_table = pf.read_row_group(rg)
    df_rg = rg_table.to_pandas()

    df_rg_idx = pd.MultiIndex.from_arrays(
        [pd.to_datetime(df_rg["date"]), df_rg["symbol"]]
    )

    # A: base CYQ 填充
    a_here = [i for i, key in enumerate(df_rg_idx) if key in fill_A]
    for i in a_here:
        key = df_rg_idx[i]
        vals = fill_A[key]
        cur = df_rg.iloc[i]
        # 幂等守卫: 只补全 NaN 位置
        for c in CYQ_COLS:
            if pd.isna(df_rg.at[df_rg.index[i], c]) and pd.notna(vals[c]):
                df_rg.at[df_rg.index[i], c] = vals[c]
                changes_A += 1
        # A 段 cost_bias 重算
        c50 = df_rg.at[df_rg.index[i], "cost_50pct"]
        if (
            pd.notna(c50)
            and c50 != 0
            and pd.notna(df_rg.at[df_rg.index[i], "close_hfq"])
        ):
            nv = (df_rg.at[df_rg.index[i], "close_hfq"] - c50) / c50
            if pd.isna(df_rg.at[df_rg.index[i], "cost_bias"]):
                df_rg.at[df_rg.index[i], "cost_bias"] = nv
                changes_B += 1
        if examples_printed < 3:
            print(
                f"    A-fill sample {key}: cost_50pct={c50}, cost_bias={df_rg.at[df_rg.index[i], 'cost_bias']}"
            )
            examples_printed += 1

    # B: cost_bias 重算 (含 2026 两日大洞)
    b_here_mask = (
        df_rg["cost_bias"].isna()
        & df_rg["cost_50pct"].notna()
        & df_rg["close_hfq"].notna()
    )
    if b_here_mask.any():
        n_before = int(b_here_mask.sum())
        c50 = df_rg.loc[b_here_mask, "cost_50pct"]
        chb = df_rg.loc[b_here_mask, "close_hfq"]
        df_rg.loc[b_here_mask, "cost_bias"] = (chb - c50) / c50.replace(0, np.nan)
        changes_B += n_before

    writer.write_table(pa.Table.from_pandas(df_rg, schema=schema, preserve_index=False))

writer.close()
pf.close()

print(f"\n[A] 补值 cell 数: {changes_A}; [B] cost_bias 补值: {changes_B}")

# ── 6. 原子替换 + 备份 ──
if os.path.exists(BACKUP):
    print(f"WARN: backup 已存在, 保留原备份不覆盖: {BACKUP}")
else:
    os.replace(PANEL, BACKUP)
    print(f"backup: {BACKUP}")

os.replace(tmp_path, PANEL)
print(f"panel replaced: {PANEL}")

# ── 7. 审计 ──
v = pq.read_table(
    PANEL, columns=["date", "symbol", "cost_50pct", "cost_bias"]
).to_pandas()
v["date"] = pd.to_datetime(v["date"])
w = v[(v["date"] >= "2023-09-18") & (v["date"] <= "2026-09-16")]
g = w.groupby("date").agg(
    rows=("symbol", "count"),
    c50_miss=("cost_50pct", lambda s: s.isna().sum()),
    bias_miss=("cost_bias", lambda s: s.isna().sum()),
)
after = g[(g["c50_miss"] > 0) | (g["bias_miss"] > 50)]
print("\n[AFTER] 仍然缺失的日子 (>0 base 或 >50 bias):")
print(after.to_string())
print(f"\n总日期数: {len(g)}, c50 全零缺日: {(g['c50_miss'] == 0).sum()}")
