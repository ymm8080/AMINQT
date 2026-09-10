# -*- coding: utf-8 -*-
"""
历史 OHLC qfq 污染修复 (2026-09-09). 修复对象: D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet

定性 (Tushare 裸价三例实锤: 002864 panel/raw=0.746, 600262=0.681, 688663=0.679;
000001=1.0 对照): 面板逐 symbol 混源 — 部分票历史 close 列是 qfq 复权源 (非裸价),
非2位小数浮点痕迹 1,758,727 行 (44.03%, 2482 symbol; 主板分红小的票缩放≈1
藏在 ratio 自然带里, 688/300/301 复权距离大才显形)。amount/volume 是真值。
连带: close_hfq=qfq×adj 双重缩放; pctChg/intraday_range/amplitude/bias 族/cost_bias
全歪; close(qfq) vs up_limit_raw(裸价) 涨停检测失真; vwap 族不可算。

本脚本:
  受累 symbol (非2dp close 任一行 ∪ volume 重构残留行) 全史拉 Tushare
  pro.daily (裸 OHLC+pre_close+vol) + pro.adj_factor:
    - open/high/low/close/pre_close ← 裸价
    - *_hfq ← 裸价 × adj (生产公式)
    - volume 重构行 (|r-1|<=1e-7 或 |r-100|<=1e-5) ← Tushare vol×100(股), 按
      symbol 惯例写回 (手 ÷100)
    - 派生重算 (仅受累 symbol 全史, 公式逐条对齐 _build_new_base_panel.py):
      pctChg / intraday_range / amplitude_5d / bias_5..250 /
      bias_5_20_cross / bias_20_60_cross / cost_bias
  Tushare 缺行的行保留旧值并计数. WORM 备份 + 原子替换.
"""

import json
import os
import shutil
import sys
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "tmp_t" else HERE
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from scripts._run_guard import find_conflicts

conflicts = find_conflicts()
if conflicts:
    print(f"[run-guard] 重活进程冲突, 拒启: {conflicts}", flush=True)
    sys.exit(2)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
OUT = os.path.join(HERE, "_ohlc_repair_result.json")
TOL = 1e-4  # 非2dp 判据 (裸价×100 与最近整数差; float64 表示误差 ~1e-9)
RECON_TOL = 1e-7

DERIVED = [
    "pctChg",
    "intraday_range",
    "amplitude_5d",
    "bias_5",
    "bias_10",
    "bias_20",
    "bias_60",
    "bias_120",
    "bias_250",
    "bias_5_20_cross",
    "bias_20_60_cross",
    "cost_bias",
]
BIAS_W = [5, 10, 20, 60, 120, 250]

t0 = time.time()
pf = pq.ParquetFile(PANEL)
comp = pf.metadata.row_group(0).column(0).compression
n_rows = pf.metadata.num_rows
pf.close()
print(f"panel rows={n_rows:,} compression={comp}", flush=True)

print("== 1. 载入 + 检测 ==", flush=True)
need = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "open_hfq",
    "high_hfq",
    "low_hfq",
    "close_hfq",
    "volume",
    "amount",
    "cost_50pct",
] + DERIVED
tbl = pq.read_table(PANEL)
missing_cols = [c for c in need if c not in tbl.schema.names]
if missing_cols:
    raise SystemExit(f"FATAL: 面板缺列 {missing_cols}")
pdf = tbl.select(need).to_pandas()
pdf["symbol"] = pdf["symbol"].astype(str).str.split(".").str[0].str.zfill(6)

c = pdf["close"]
bad = (np.abs(c * 100 - np.round(c * 100)) > TOL) & c.notna() & (c > 0)
v, a = pdf["volume"], pdf["amount"]
r = (a / (c * v)).where((c > 0) & (v > 0) & (a > 0))
recon = (((r - 1).abs() <= RECON_TOL) | ((r - 100).abs() <= RECON_TOL * 100)).fillna(
    False
)
print(
    f"非2dp close 行={int(bad.sum()):,} ({bad.mean():.2%}) / symbol={pdf.loc[bad, 'symbol'].nunique()}",
    flush=True,
)
print(f"volume 重构残留行={int(recon.sum()):,}", flush=True)

bad_syms = set(pdf.loc[bad, "symbol"].unique())
recon_syms = set(pdf.loc[recon, "symbol"].unique())
aff_syms = sorted(bad_syms | recon_syms)
print(
    f"受累 symbol={len(aff_syms)} (bad={len(bad_syms)} ∪ recon={len(recon_syms)})",
    flush=True,
)

# 每 symbol 惯例: ①非重构行中位 ratio>50=手 ②全重构: 恒等式≡100=手 ③默认 gu
true_r = r.where(~recon)
med = true_r.groupby(pdf["symbol"]).median()
exact100 = ((r - 100).abs() <= RECON_TOL * 100).groupby(pdf["symbol"]).sum()
hand = set(med[med > 50].index) | set(exact100[exact100 > 0].index)
print(f"hand(手)惯例 symbol={len(hand)}", flush=True)

print("== 2. Tushare 全史拉取 (裸 OHLC+pre_close+vol+adj) ==", flush=True)
import tushare as ts

from config import settings

token = settings.TUSHARE_TOKEN or ts.get_token()
pro = ts.pro_api(token)


def ts_code_of(sym):
    if sym.startswith("6"):
        return f"{sym}.SH"
    if sym.startswith(("0", "3", "1")):
        return f"{sym}.SZ"
    return f"{sym}.BJ"


ohlc_lut = {}  # (sym,date) -> (o,h,l,c,pc)
adj_lut = {}  # (sym,date) -> adj_factor
vol_lut = {}  # (sym,date) -> vol_ts (手)

sym_grp = (
    pdf[pdf["symbol"].isin(aff_syms)].groupby("symbol")["date"].agg(["min", "max"])
)
fails = []
for i, sym in enumerate(aff_syms):
    lo = sym_grp.loc[sym, "min"].strftime("%Y%m%d")
    hi = sym_grp.loc[sym, "max"].strftime("%Y%m%d")
    tcd = ts_code_of(sym)
    got = False
    for attempt in range(3):
        try:
            raw = pro.daily(
                ts_code=tcd,
                start_date=lo,
                end_date=hi,
                fields="trade_date,open,high,low,close,pre_close,vol",
            )
            adj = pro.adj_factor(
                ts_code=tcd, start_date=lo, end_date=hi, fields="trade_date,adj_factor"
            )
            got = True
            break
        except Exception as e:
            time.sleep(2.0 * (attempt + 1))
            if attempt == 2:
                fails.append(sym)
                print(f"  {sym} FAILED: {e}", flush=True)
    if got:
        for _, row in raw.iterrows():
            k = (sym, pd.to_datetime(row["trade_date"]))
            ohlc_lut[k] = (
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row["pre_close"],
            )
            vol_lut[k] = row["vol"]
        for _, row in adj.iterrows():
            adj_lut[(sym, pd.to_datetime(row["trade_date"]))] = float(row["adj_factor"])
    if (i + 1) % 100 == 0:
        print(
            f"  {i + 1}/{len(aff_syms)} ohlc键={len(ohlc_lut):,} adj键={len(adj_lut):,} "
            f"fail={len(fails)} {time.time() - t0:.0f}s",
            flush=True,
        )
    time.sleep(0.25)
print(
    f"拉取完成 ohlc键={len(ohlc_lut):,} adj键={len(adj_lut):,} fail={len(fails)} "
    f"elapsed={time.time() - t0:.0f}s",
    flush=True,
)

print("== 3. 重写 OHLC/hfq/volume (向量化 join) ==", flush=True)
new = {
    col: pdf[col].copy()
    for col in [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "open_hfq",
        "high_hfq",
        "low_hfq",
        "close_hfq",
        "volume",
    ]
}

lut_df = pd.DataFrame.from_dict(
    ohlc_lut, orient="index", columns=["o", "h", "l", "c", "pc"]
)
lut_df.index = pd.MultiIndex.from_tuples(lut_df.index, names=["symbol", "date"])
adj_s = pd.Series(adj_lut, dtype="float64")
adj_s.index = pd.MultiIndex.from_tuples(adj_s.index, names=["symbol", "date"])

j = pdf.loc[bad, ["symbol", "date"]].join(lut_df, on=["symbol", "date"])
j = j.join(adj_s.rename("adj"), on=["symbol", "date"])
valid = j["c"].notna() & (j["c"] > 0) & j[["o", "h", "l", "pc"]].notna().all(axis=1)
tgt = j.index[valid]
n_ohlc = len(tgt)
n_adj_miss = int((j.loc[tgt, "adj"].isna()).sum())
for col, srccol in [
    ("open", "o"),
    ("high", "h"),
    ("low", "l"),
    ("close", "c"),
    ("pre_close", "pc"),
]:
    new[col].loc[tgt] = j.loc[tgt, srccol].values
adjv = j.loc[tgt, "adj"]
has_adj = adjv.notna()
for col, srccol in [
    ("open_hfq", "o"),
    ("high_hfq", "h"),
    ("low_hfq", "l"),
    ("close_hfq", "c"),
]:
    new[col].loc[tgt] = (j.loc[tgt, srccol] * adjv).where(has_adj).values
print(f"OHLC 重写 {n_ohlc:,} 行 (adj 缺失保留旧 hfq {n_adj_miss:,})", flush=True)

# volume 重构行: Tushare vol(手) ×100=股; hand 惯例 symbol 原值(手)
vol_s = pd.Series(
    {k: x for k, x in vol_lut.items() if x is not None and not pd.isna(x) and x > 0},
    dtype="float64",
)
vol_s.index = pd.MultiIndex.from_tuples(vol_s.index, names=["symbol", "date"])
jr = pdf.loc[recon, ["symbol", "date"]].join(
    vol_s.rename("vol_ts"), on=["symbol", "date"]
)
tgtr = jr.index[jr["vol_ts"].notna()]
fac = pd.Series(np.where(jr.loc[tgtr, "symbol"].isin(hand), 1.0, 100.0), index=tgtr)
new["volume"].loc[tgtr] = (jr.loc[tgtr, "vol_ts"] * fac).values
n_vol = len(tgtr)
print(f"volume 重写 {n_vol:,} 行 (hand={len(hand & recon_syms)})", flush=True)

print("== 4. 派生重算 (受累 symbol 全史, 向量化 groupby) ==", flush=True)
aff_mask = pdf["symbol"].isin(aff_syms)
sub = pdf.loc[aff_mask].copy()
for col in new:
    sub[col] = new[col]
sub = sub.sort_values(["symbol", "date"])

pc0 = sub["pre_close"].replace(0, np.nan)
sub["intraday_range"] = (sub["high"] - sub["low"]) / pc0
sub["pctChg"] = (sub["close"] / pc0 - 1) * 100

g = sub.groupby("symbol", sort=False)
hc = sub["close_hfq"]
for w in BIAS_W:
    ma = g["close_hfq"].rolling(w, min_periods=w).mean()
    ma.index = ma.index.droplevel(0)
    sub[f"bias_{w}"] = hc / ma - 1
for a_, b_, name in [
    ("bias_5", "bias_20", "bias_5_20_cross"),
    ("bias_20", "bias_60", "bias_20_60_cross"),
]:
    sub[name] = np.sign(sub[a_] - sub[b_]).groupby(sub["symbol"]).diff().fillna(0)
amp = (sub["high"] - sub["low"]) / pc0
a5 = amp.groupby(sub["symbol"]).rolling(5, min_periods=3).mean()
a5.index = a5.index.droplevel(0)
sub["amplitude_5d"] = a5
cost = pd.to_numeric(sub["cost_50pct"], errors="coerce")
sub["cost_bias"] = np.where(
    cost.notna() & (cost != 0), (sub["close_hfq"] - cost) / cost, np.nan
)

for col in DERIVED:
    pdf.loc[sub.index, col] = sub[col]
for col in new:
    pdf[col] = new[col]

print("== 5. 终验 ==", flush=True)
c2, v2, a2 = pdf["close"], pdf["volume"], pdf["amount"]
bad2 = (np.abs(c2 * 100 - np.round(c2 * 100)) > TOL) & c2.notna() & (c2 > 0)
r2 = (a2 / (c2 * v2)).where((c2 > 0) & (v2 > 0) & (a2 > 0))
recon2 = (((r2 - 1).abs() <= RECON_TOL) | ((r2 - 100).abs() <= RECON_TOL * 100)).fillna(
    False
)
print(
    f"修后 非2dp={int(bad2.sum()):,} ({bad2.mean():.2%}, 原 {bad.mean():.2%}); "
    f"重构率={recon2.mean():.2%} (原 {recon.mean():.2%})",
    flush=True,
)
for sym, td in [
    ("002864", "2024-01-05"),
    ("688663", "2023-01-03"),
    ("600262", "2024-01-05"),
]:
    row = pdf[(pdf["symbol"] == sym) & (pdf["date"] == td)]
    if len(row):
        print(
            f"  spot {sym}@{td}: close={row['close'].iloc[0]:.2f} "
            f"hfq={row['close_hfq'].iloc[0]:.2f}",
            flush=True,
        )
ok_spot = True
for s, d, x in [
    ("002864", "2024-01-05", 38.27),
    ("688663", "2023-01-03", 47.07),
    ("600262", "2024-01-05", 19.32),
]:
    row = pdf[(pdf["symbol"] == s) & (pdf["date"] == d)]
    if not len(row) or abs(row["close"].iloc[0] - x) >= 0.01:
        ok_spot = False
print(f"裸价锚点校验: {'PASS' if ok_spot else 'FAIL'}", flush=True)

print("== 6. 备份 + 原子写 ==", flush=True)
tag = datetime.now().strftime("%Y%m%d_%H%M%S")
bak = f"{PANEL}.pre_ohlcfix_{tag}"
shutil.copy2(PANEL, bak)
print(f"备份 → {bak} ({os.path.getsize(bak) / 1e9:.2f} GB)", flush=True)
tbl2 = tbl
for col in [
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "open_hfq",
    "high_hfq",
    "low_hfq",
    "close_hfq",
    "volume",
] + DERIVED:
    tbl2 = tbl2.drop([col]).append_column(
        col, pa.array(pdf[col].values, type=tbl.schema.field(col).type)
    )
tmp = f"{PANEL}.tmp_ohlcfix_{tag}"
pq.write_table(tbl2, tmp, compression=comp)
os.replace(tmp, PANEL)
print(
    f"写回完成 ({os.path.getsize(PANEL) / 1e9:.2f} GB) elapsed={time.time() - t0:.0f}s",
    flush=True,
)

json.dump(
    {
        "panel": PANEL,
        "backup": bak,
        "rows": int(len(pdf)),
        "affected_symbols": len(aff_syms),
        "bad_close_before": int(bad.sum()),
        "bad_close_after": int(bad2.sum()),
        "recon_before": int(recon.sum()),
        "recon_after": int(recon2.sum()),
        "ohlc_fixed": int(n_ohlc),
        "adj_missing": int(n_adj_miss),
        "vol_fixed": int(n_vol),
        "hand_symbols": len(hand),
        "pull_failures": fails,
        "spot_check_pass": bool(ok_spot),
    },
    open(OUT, "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=1,
)
print(f"saved {OUT}", flush=True)
print("OHLC REPAIR DONE", flush=True)
