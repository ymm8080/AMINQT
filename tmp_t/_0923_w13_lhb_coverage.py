# -*- coding: utf-8 -*-
"""W13-B: 首板事件 LHB 席位数据覆盖审计 — 席位特征喂首板连板模型, 面够不够 (A/B 可行性闸).

口径:
- 事件 = 生产同源 scripts/_firstboard_pages.build_event_features
  (D0=主板首板 pctChg>=9.5 且 前10交易日无板 rolling(10).max().shift(1)==0)
- A 样本 = 生产训练过滤 hist_ok & mature10 & k2/next_board 非空 & need6 非空
  (对拍生产规模 n_train≈16141 / n_te≈5608)
- TR = 2023-01-01..2025-12-31, TE = 2026-01-01..2026-09-22
- LHB = D:/AMINQT/PARQUET/lhb_toplist.parquet (~60,583 行, reason)
       + D:/AMINQT/PARQUET/lhb_seat_detail.parquet (~573,841 行, 席位级 side)
- symbol 对齐: ts_code.str[:6] ↔ 面板 symbol (6位字符串); 主板 = 00/60 前缀
- 窗口定义: D-5..D0 = D0 及其前5个交易日 (rolling 6, 含 D0)
            D-20..D0 = rolling 21, 含 D0
            pre20 (对拍) = D-20..D-1 严格板前 (rolling 20 after shift(1))
            ⚠ 0923: 生产侧 _pre20 两列(lhb_pre20_net/both)已作死代码删除, 本脚本改为就地复算同口径
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(r"D:\AMINQT\AMINQT CODES")
sys.path.insert(0, str(ROOT / "scripts"))
import _firstboard_pages as fbp  # noqa: E402  (生产事件口径, 只读 import)

PANEL = r"D:/AMINQT/PARQUET/panel_full_enriched_v3.parquet"
TOP_LIST = r"D:/AMINQT/PARQUET/lhb_toplist.parquet"
TOP_INST = r"D:/AMINQT/PARQUET/lhb_seat_detail.parquet"

TR0, TR1 = pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-31")
TE0, TE1 = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-09-22")
NEED6 = ["wr1", "ret60", "gap", "seal_hard", "cmv", "ftr"]


def sec(title):
    print(f"\n{'=' * 72}\n== {title}\n{'=' * 72}", flush=True)


# ---------------------------------------------------------------- 1. 事件 (生产同源)
sec("1. 面板 + 首板事件 (生产 build_event_features 同源)")
df = fbp.load_mainboard(PANEL, tail_dates=None)
print(f"面板: {len(df)} 行, {df['symbol'].nunique()} 只, 日期 {df['date'].min():%Y-%m-%d}..{df['date'].max():%Y-%m-%d}")

ev = fbp.build_event_features(df)
a_mask = (ev["hist_ok"] & ev["mature10"] & ev["k2"].notna() & ev["next_board"].notna()
          & ev[NEED6].notna().all(axis=1))
# 对拍生产规模: n_train = A & date<=2025-12-31 (tr|va 并集), n_te = A & date>2025-12-31
n_train_prod_like = int((a_mask & (ev["date"] <= TR1)).sum())
n_te_prod_like = int((a_mask & (ev["date"] > TR1)).sum())
print(f"事件总数 {len(ev)}; A 样本 {int(a_mask.sum())}")
print(f"[对拍] A&date<=2025-12-31 = {n_train_prod_like} (生产 n_train≈16141, "
      f"差 {100 * (n_train_prod_like - 16141) / 16141:+.2f}%)")
print(f"[对拍] A&date>2025-12-31  = {n_te_prod_like} (生产 n_te≈5608, "
      f"差 {100 * (n_te_prod_like - 5608) / 5608:+.2f}%)")
pre2023 = int((a_mask & (ev["date"] < TR0)).sum())
print(f"其中 A 样本里 date<2023-01-01 的事件数 = {pre2023} (席位数据 2023-01-03 起, TR 外不计)")

tr_m = a_mask & ev["date"].between(TR0, TR1)
te_m = a_mask & ev["date"].between(TE0, TE1)
tr_raw = ev["date"].between(TR0, TR1)
te_raw = ev["date"].between(TE0, TE1)
print(f"TR 事件: raw {int(tr_raw.sum())} / A {int(tr_m.sum())};  TE 事件: raw {int(te_raw.sum())} / A {int(te_m.sum())}")

# ---------------------------------------------------------------- 2. LHB 数据
sec("2. LHB 数据 (toplist + seat_detail)")
tl = pq.read_table(TOP_LIST, columns=["trade_date", "ts_code", "net_amount", "reason"]).to_pandas()
tl["symbol"] = tl["ts_code"].str[:6]
tl["date"] = pd.to_datetime(tl["trade_date"].astype(str), format="%Y%m%d")
tl_mb = tl[tl["symbol"].str[:2].isin(["00", "60"])].copy()
print(f"toplist {len(tl)} 行 / {tl['date'].nunique()} 日 ({tl['date'].min():%Y-%m-%d}..{tl['date'].max():%Y-%m-%d}); "
      f"主板行 {len(tl_mb)}; (symbol,date) 重复行 {len(tl_mb) - len(tl_mb.drop_duplicates(['symbol', 'date']))}")

st = pq.read_table(TOP_INST, columns=["trade_date", "ts_code", "side", "net_buy"]).to_pandas()
st["side"] = pd.to_numeric(st["side"], errors="coerce")  # parquet 里是字符串 "0"/"1", 必须 numeric 化
print(f"side 值域: {sorted(st['side'].dropna().unique().tolist())}")
st["symbol"] = st["ts_code"].str[:6]
st["date"] = pd.to_datetime(st["trade_date"].astype(str), format="%Y%m%d")
st_mb = st[st["symbol"].str[:2].isin(["00", "60"])].copy()
print(f"seat_detail {len(st)} 行 / {st['date'].nunique()} 日; 主板行 {len(st_mb)}")
side_chk = st_mb.groupby("side")["net_buy"].agg(["mean", "size", lambda s: (s > 0).mean()])
side_chk.columns = ["net_buy均值", "行数", "net>0占比"]
print("side 语义验证 (net_buy 按侧):\n", side_chk.to_string())

# LHB 日期空洞 vs 面板交易日
pan_dates = pd.DatetimeIndex(sorted(df["date"].unique()))
lhb_dates = pd.DatetimeIndex(sorted(tl["date"].unique()))
for name, lo, hi in (("TR", TR0, TR1), ("TE", TE0, TE1)):
    pd_ = pan_dates[(pan_dates >= lo) & (pan_dates <= hi)]
    missing = pd_.difference(lhb_dates)
    print(f"{name} 面板交易日 {len(pd_)} 个, LHB 无数据日 {len(missing)} 个"
          + (f": {[d.strftime('%Y-%m-%d') for d in missing[:10]]}" if len(missing) else ""))

# ---------------------------------------------------------------- 3. 面板网格上的覆盖指示
sec("3. 覆盖率 (Q1/Q2/Q4)")
day_any = tl_mb.assign(_v=1).groupby(["symbol", "date"], as_index=False)["_v"].max()
day_net = (tl_mb.assign(_n=(pd.to_numeric(tl_mb["net_amount"], errors="coerce") > 0).astype(int))
           .groupby(["symbol", "date"], as_index=False)["_n"].max())
grid = df[["symbol", "date"]].merge(day_any, on=["symbol", "date"], how="left").merge(
    day_net, on=["symbol", "date"], how="left")
ind_any = grid["_v"].fillna(0).astype(float)
ind_net = grid["_n"].fillna(0).astype(float)
sym = df["symbol"]


def roll_flag(v, win):
    return (v.groupby(sym, sort=False)
            .transform(lambda x: x.rolling(win, min_periods=1).max()).fillna(0) > 0)


# 0923: 下面两列原先由 fbp.build_event_features 产出(死代码, 已删) → 就地复算, 口径逐字照抄
# (股票级 lhb_net_buy>0 / 净买与机构双正, shift(1).rolling(20) 严格板前), 保证对拍数字不变
_net_p = pd.to_numeric(df["lhb_net_buy"], errors="coerce")
_inst_p = pd.to_numeric(df["lhb_inst_buy"], errors="coerce")
_pos_p = (_net_p > 0).astype(float)
_both_p = ((_net_p > 0) & (_inst_p > 0)).astype(float)

cov = pd.DataFrame({
    "d0": ind_any > 0,                                     # Q1
    "w5": roll_flag(ind_any, 6),                           # D-5..D0 (含)
    "w20": roll_flag(ind_any, 21),                         # D-20..D0 (含)
    "pre20_any": (ind_any.groupby(sym, sort=False)
                  .transform(lambda x: x.shift(1).rolling(20, min_periods=1).max()).fillna(0) > 0),
    "pre20_net": (ind_net.groupby(sym, sort=False)
                  .transform(lambda x: x.shift(1).rolling(20, min_periods=1).max()).fillna(0) > 0),
    "lhb_pre20_net": (_pos_p.groupby(sym, sort=False)
                      .transform(lambda x: x.shift(1).rolling(20, min_periods=1).max()).fillna(0) > 0),
    "lhb_pre20_both": (_both_p.groupby(sym, sort=False)
                       .transform(lambda x: x.shift(1).rolling(20, min_periods=1).max()).fillna(0) > 0),
}, index=df.index)
# 注意: build_event_features 返回前 reset_index — 必须 (symbol,date) 按键 merge, 勿用 ev.index 对齐
cov = df[["symbol", "date"]].merge(cov, left_index=True, right_index=True)
ev = ev.merge(cov, on=["symbol", "date"], how="left")

rows = []
for name, m in (("TR", tr_m), ("TE", te_m)):
    c = ev.loc[m, ["d0", "w5", "w20", "pre20_any", "pre20_net"]]
    prod_net = ev.loc[m, "lhb_pre20_net"]
    prod_both = ev.loc[m, "lhb_pre20_both"]
    rows.append({
        "窗口": name, "n事件A": int(m.sum()),
        "Q1 D0上榜%": 100 * c["d0"].mean(),
        "D-5..D0%": 100 * c["w5"].mean(),
        "D-20..D0%": 100 * c["w20"].mean(),
        "pre20任意%(我)": 100 * c["pre20_any"].mean(),
        "pre20净买%(我)": 100 * c["pre20_net"].mean(),
        "lhb_pre20_net=1%(生产)": 100 * prod_net.mean(),
        "lhb_pre20_both=1%(生产)": 100 * prod_both.mean(),
    })
cov_tbl = pd.DataFrame(rows).set_index("窗口")
print(cov_tbl.T.round(2).to_string())

net_col = pd.to_numeric(df["lhb_net_buy"], errors="coerce")
for name, lo, hi in (("TR", TR0, TR1), ("TE", TE0, TE1)):
    dm = df["date"].between(lo, hi)
    print(f"{name} 面板主板行 lhb_net_buy 非空率 = {100 * net_col[dm].notna().mean():.2f}% "
          f"(n={int(dm.sum())}) — 日级 LHB 覆盖对照")

# 对拍差异诊断: 面板 net>0 日 vs toplist 日级 (symbol,date) 集合差
panel_pos = df.loc[net_col > 0, ["symbol", "date"]].drop_duplicates()
mine_any = day_any[["symbol", "date"]]
pp = set(map(tuple, panel_pos.values))
mm_any = set(map(tuple, mine_any.values))
mm_net = set(map(tuple, day_net[day_net["_n"] == 1][["symbol", "date"]].values))
print(f"日级对拍: 面板net>0对 {len(pp)} | toplist任意 {len(mm_any)} | toplist净>0 {len(mm_net)}")
print(f"  面板net>0 vs toplist净>0: 交 {len(pp & mm_net)}, 面板独有 {len(pp - mm_net)}, toplist独有 {len(mm_net - pp)}")
for name, lo, hi in (("TR", TR0, TR1), ("TE", TE0, TE1)):
    ppw = {t for t in pp if lo <= t[1] <= hi}
    anw = {t for t in mm_any if lo <= t[1] <= hi}
    ntw = {t for t in mm_net if lo <= t[1] <= hi}
    print(f"  {name}: 面板net>0 {len(ppw)} | toplist任意 {len(anw)} | toplist净>0 {len(ntw)} | "
          f"面板独有(净>0口径) {len(ppw - ntw)} | toplist净>0独有 {len(ntw - ppw)}")

# ---------------------------------------------------------------- 4. Q3 席位深读
sec("4. Q3: D0 覆盖事件的席位深读 (seat_detail)")
seat_cnt = st_mb.groupby(["symbol", "date"]).size().rename("n_seat").reset_index()
seat_buy = (st_mb.assign(_b=(st_mb["side"] == 0).astype(int))
            .groupby(["symbol", "date"], as_index=False)["_b"].mean())
for name, m in (("TR", tr_m), ("TE", te_m)):
    evc = ev.loc[m, ["symbol", "date", "d0"]]
    cov_ev = evc[evc["d0"]]
    j = cov_ev.merge(seat_cnt, on=["symbol", "date"], how="left")
    j = j.merge(seat_buy, on=["symbol", "date"], how="left")
    print(f"\n[{name}] D0 上榜事件 {len(cov_ev)}/{len(evc)}:")
    print(f"  席位行/事件: mean {j['n_seat'].mean():.1f}, median {j['n_seat'].median():.0f}, "
          f"p10 {j['n_seat'].quantile(.1):.0f}, p90 {j['n_seat'].quantile(.9):.0f}")
    print(f"  买方席位占比(side==0): 事件均值 {100 * j['_b'].mean():.1f}%")
    # reason 分布 (toplist 行级, 覆盖事件 join)
    rr = tl_mb.merge(cov_ev[["symbol", "date"]], on=["symbol", "date"], how="inner")
    top5 = rr["reason"].value_counts(normalize=True).head(5)
    print(f"  reason top5 (行={len(rr)}, 一事件可多 reason 行):")
    for k, v in top5.items():
        print(f"    {100 * v:5.1f}%  {k}")

# ---------------------------------------------------------------- 5. Q5 k2 选择性
sec("5. Q5: D0 上榜 vs 未上榜的 k2(5日内再板) 率")
for name, m in (("TR", tr_m), ("TE", te_m)):
    evc = ev.loc[m, ["k2", "d0"]]
    for lab, flag in (("D0上榜", True), ("未上榜", False)):
        sub = evc[evc["d0"] == flag]
        print(f"[{name}] {lab}: n={len(sub)}, k2率={100 * sub['k2'].mean():.2f}%")
    print(f"[{name}] 全体基线 k2率={100 * evc['k2'].mean():.2f}% (TE 已知≈27.4%)")

sec("DONE")
