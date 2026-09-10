# -*- coding: utf-8 -*-
"""
volume 重构污染修复 (2026-09-09). 修复对象: D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet
定性 (根因审计推翻早期假设): amount 是真值 (Tushare×1000 元); 被 _daily_fetch/_build_new_base_panel
用 amount/close 重构的是 volume (±4% vwap 近似), 使 amount/(close×volume) 恒等 1(gu)/100(hand),
vwap 族特征全窗不可算. 代码根因已修 (_daily_fetch.py / _build_new_base_panel.py 改用 Tushare vol).
本脚本修存量的重构行:
  重构行 = |amount/(close×volume) − 1| ≤ 1e-7 (gu) 或 |ratio − 100| ≤ 1e-5 (hand)
  真值来源 (转成股): panel_4000 (vol 手, 2024-01..2026-07-27) > panel_baostock (vol 股) >
    Tushare pro.daily vol (近期窗口按 trade_date 全市场 + 缺口按个股全史)
  写回按各 symbol 惯例: gu=股原值, hand(历史中位 ratio>50)÷100=手
  修不动 (无真源) 的重构行保留旧近似值并报告. WORM 备份 + 原子替换.
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
SRC_4000 = os.path.join(ROOT, "data", "panel_4000.parquet")
SRC_BS = os.path.join(ROOT, "data", "panel_baostock.parquet")
OUT = os.path.join(HERE, "_volume_repair_result.json")
RECENT_START = "2026-07-27"  # 日更重构起点; 之前为 base_new_full 全史重构
TOL = 1e-7

t0 = time.time()
pf = pq.ParquetFile(PANEL)
comp = pf.metadata.row_group(0).column(0).compression
n_rows = pf.metadata.num_rows
pf.close()
print(f"panel rows={n_rows:,} compression={comp}", flush=True)

print("== 1. 载入面板 + 重构掩码 ==", flush=True)
tbl = pq.read_table(PANEL)
pdf = tbl.select(["date", "symbol", "close", "volume", "amount"]).to_pandas()
pdf["symbol"] = pdf["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
c, v, a = pdf["close"], pdf["volume"], pdf["amount"]
ratio = (a / (c * v)).where((c > 0) & (v > 0) & (a > 0))
recon = ((ratio - 1).abs() <= TOL) | ((ratio - 100).abs() <= TOL * 100)
recon = recon.fillna(False)
print(f"重构行={recon.sum():,} / {len(pdf):,} ({recon.mean():.1%})", flush=True)
for y, g in pdf.groupby(pdf["date"].dt.year):
    print(f"  {y}: {recon.loc[g.index].sum():,}/{len(g):,}", flush=True)

print("== 2. 每 symbol 惯例 (真值行中位 ratio; 全重构=新股票默认 gu) ==", flush=True)
true_ratio = ratio.where(~recon)
med = true_ratio.groupby(pdf["symbol"]).median()
hand = set(med[med > 50].index)
sym_is_hand = pdf["symbol"].isin(hand)
print(f"hand(手)={len(hand)} / 有真值史 symbol={med.notna().sum()} / "
      f"全重构(默认gu)={pdf['symbol'].nunique() - med.notna().sum()}", flush=True)


def src_vol_shares(path, name):
    """读真源 → {(sym,date): 真股数}. 每 symbol 自检中位 ratio 判 vol 单位 (≈1=股/≈100=手)."""
    if not os.path.exists(path):
        print(f"[{name}] 不存在, 跳过", flush=True)
        return {}
    df = pq.read_table(path, columns=["date", "symbol", "close", "volume", "amount"]).to_pandas()
    df["symbol"] = df["symbol"].astype(str).str.split(".").str[0].str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["close"] > 0) & (df["volume"] > 0) & (df["amount"] > 0)]
    r = (df["amount"] / (df["close"] * df["volume"]))
    m = r.groupby(df["symbol"]).median()
    fac = pd.Series(np.where(m > 50, 100.0, 1.0), index=m.index)  # 手→×100=股
    df["vol_sh"] = df["volume"] * df["symbol"].map(fac).fillna(1.0)
    lut = {(s, d): x for s, d, x in zip(df["symbol"], df["date"], df["vol_sh"])}
    n_hand_src = int((m > 50).sum())
    print(f"[{name}] rows={len(df):,} 键={len(lut):,} (源内手惯例symbol={n_hand_src}, "
          f"{df['date'].min().date()}..{df['date'].max().date()})", flush=True)
    return lut


lut = {}
for path, name in [(SRC_4000, "panel_4000"), (SRC_BS, "baostock")]:
    for k, x in src_vol_shares(path, name).items():
        lut.setdefault(k, x)
print(f"本地真源键={len(lut):,} elapsed={time.time()-t0:.0f}s", flush=True)

print("== 3. 第一轮回补 (本地源) ==", flush=True)
keys = list(zip(pdf.loc[recon, "symbol"], pdf.loc[recon, "date"]))
mapped = pd.Series([lut.get(k, np.nan) for k in keys], index=pdf.index[recon])
fixed1 = recon & False
fixed1[pdf.index[recon][mapped.notna().values]] = True
print(f"本地回补 {int(fixed1.sum()):,} / {int(recon.sum()):,}", flush=True)

print("== 4. Tushare 补缺口 ==", flush=True)
import tushare as ts

from config import settings

token = settings.TUSHARE_TOKEN or ts.get_token()
pro = ts.pro_api(token)
ts_lut = {}


def ts_code_of(sym):
    if sym.startswith("6"):
        return f"{sym}.SH"
    if sym.startswith(("0", "3", "1")):
        return f"{sym}.SZ"
    return f"{sym}.BJ"


def ingest_ts(raw):
    if raw is None or len(raw) == 0:
        return
    sub = raw[["ts_code", "trade_date", "vol"]].dropna()
    sub = sub[sub["vol"] > 0]
    for tc, td, vv in zip(sub["ts_code"], sub["trade_date"], sub["vol"]):
        ts_lut.setdefault((tc.split(".")[0], pd.to_datetime(td)), float(vv) * 100.0)


# 4a. 近期窗口全市场 (按 trade_date)
d_hi = pdf["date"].max()
cal = pro.trade_cal(exchange="SSE", start_date=RECENT_START.replace("-", ""),
                    end_date=d_hi.strftime("%Y%m%d"), fields="cal_date,is_open")
tdates = [d for d, o in zip(cal["cal_date"], cal["is_open"]) if o == 1]
for i, td in enumerate(tdates):
    try:
        ingest_ts(pro.daily(trade_date=td))
    except Exception as e:
        print(f"  trade_date {td} FAILED: {e}", flush=True)
    if (i + 1) % 10 == 0:
        print(f"  trade_date {i+1}/{len(tdates)} 累计键={len(ts_lut):,} {time.time()-t0:.0f}s", flush=True)
    time.sleep(0.35)
print(f"近期窗口 Tushare 键={len(ts_lut):,}", flush=True)

# 4b. 仍缺的 symbol 全史按个股拉
gap_idx = pdf.index[recon & ~fixed1]
gap = pdf.loc[gap_idx]
mapped2 = pd.Series([ts_lut.get(k, np.nan) for k in zip(gap["symbol"], gap["date"])],
                     index=gap_idx)
still = gap.loc[mapped2.isna().values]
gap_syms = sorted(still["symbol"].unique())
print(f"按个股补拉 symbol={len(gap_syms)} 行={len(still):,}", flush=True)
for i, sym in enumerate(gap_syms):
    g = still[still["symbol"] == sym]
    try:
        ingest_ts(pro.daily(ts_code=ts_code_of(sym),
                            start_date=g["date"].min().strftime("%Y%m%d"),
                            end_date=g["date"].max().strftime("%Y%m%d")))
    except Exception as e:
        print(f"  {sym} FAILED: {e}", flush=True)
    if (i + 1) % 100 == 0:
        print(f"  个股 {i+1}/{len(gap_syms)} 累计键={len(ts_lut):,} {time.time()-t0:.0f}s", flush=True)
    time.sleep(0.25)

print("== 5. 合成新 volume + 终验 ==", flush=True)
mapped2b = pd.Series([ts_lut.get(k, np.nan) for k in keys], index=pdf.index[recon])
new_vol = pdf["volume"].copy()
idx1 = pdf.index[recon][mapped.notna().values]
new_vol.loc[idx1] = mapped[mapped.notna()]
idx2 = pdf.index[recon][mapped2b.notna().values]
new_vol.loc[idx2] = mapped2b[mapped2b.notna()]
# 写回按惯例: hand ÷100
hand_idx = idx1.union(idx2).intersection(pdf.index[sym_is_hand])
new_vol.loc[hand_idx] = new_vol.loc[hand_idx] / 100.0
resid = int(recon.sum() - len(idx1) - len(idx2))
print(f"回补: 本地{len(idx1):,} + Tushare{len(idx2):,} (hand÷100 {len(hand_idx):,}); "
      f"修不动保留近似={resid:,}", flush=True)

ratio2 = (a / (c * new_vol)).where((c > 0) & (new_vol > 0) & (a > 0))
recon2 = ((ratio2 - 1).abs() <= TOL) | ((ratio2 - 100).abs() <= TOL * 100)
recon2 = recon2.fillna(False)
print(f"修后重构率={recon2.mean():.2%} (原 {recon.mean():.2%})", flush=True)
ok_range = (ratio2.dropna().between(0.85, 1.15) | ratio2.dropna().between(85, 115)).mean()
print(f"修后 ratio 自然散布检验 (0.85-1.15 或 85-115): {ok_range:.1%}", flush=True)

print("== 6. 备份 + 原子写 ==", flush=True)
tag = datetime.now().strftime("%Y%m%d_%H%M%S")
bak = f"{PANEL}.pre_volfix_{tag}"
shutil.copy2(PANEL, bak)
print(f"备份 → {bak} ({os.path.getsize(bak)/1e9:.2f} GB)", flush=True)
tbl2 = tbl.drop(["volume"]).append_column("volume", pa.array(new_vol.values, type=pa.float64()))
tmp = f"{PANEL}.tmp_volfix_{tag}"
pq.write_table(tbl2, tmp, compression=comp)
os.replace(tmp, PANEL)
print(f"写回完成 {PANEL} ({os.path.getsize(PANEL)/1e9:.2f} GB) elapsed={time.time()-t0:.0f}s", flush=True)

json.dump({
    "panel": PANEL, "backup": bak, "rows": int(len(pdf)),
    "recon_before": int(recon.sum()), "fixed_local": int(len(idx1)),
    "fixed_tushare": int(len(idx2)), "hand_div100": int(len(hand_idx)),
    "resid_kept_approx": resid, "recon_after": int(recon2.sum()),
    "hand_symbols": len(hand),
}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"saved {OUT}", flush=True)
