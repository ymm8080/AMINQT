# -*- coding: utf-8 -*-
"""L2 日内/资金聚合特征构建器试点 (任务#4, 09-09).

两个独立源:
  A. bs5 5min bar (baostock, volume=股, amount=元) → 每股每日:
     - eop_vol_ratio   尾盘30min量/全日量 (拉尾盘指纹)
     - open_vol_ratio  开盘30min量/全日量
     - intraday_recovery (收-低)/(高-低) 日内V型修复
     - vwap_dev_close  收盘/真实VWAP-1 (收盘相对全日成交成本)
  B. moneyflow 资金流 (Tushare, 725日, amount千元→比率不受单位影响) → 每股每日:
     - mf_lg_net_ratio / mf_elg_net_ratio  大单/特大单净额占比
     - mf_net_ratio                        全单净流入占比
     - mf_lg_net5_ratio / mf_elg_net5_ratio 5日滚动净额占比
试点输出 tmp_min (全量版等 ingest 完成后走 scripts/ + 入面板 A/B).
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = r"D:\AMINQT\AMINQT CODES"
SAMPLE = os.path.join(ROOT, "tmp_min", "_bs5_sample")
MFDIR = os.path.join(ROOT, "data", "supply_cache", "alt_data", "moneyflow_daily")
OUT_BS5 = os.path.join(ROOT, "tmp_min", "_l2_bs5_feats_pilot.parquet")
OUT_MF = os.path.join(ROOT, "tmp_min", "_l2_mf_feats.parquet")


def build_bs5(src_dir):
    frames = []
    for p in sorted(glob.glob(os.path.join(src_dir, "bs5_*.parquet"))):
        sym = os.path.basename(p)[4:-8]
        d = pd.read_parquet(p)
        if d.empty:
            continue
        for c in ("open", "high", "low", "close", "volume", "amount"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
        d["bar_min"] = d["time"].str.slice(3, 5).astype(int) * 60 + d["time"].str.slice(5, 7).astype(int)
        rows = []
        for dt, g in d.groupby("date"):
            g = g.sort_values("bar_min")
            vol, amt = g["volume"].to_numpy(), g["amount"].to_numpy()
            day_vol = vol.sum()
            if day_vol <= 0:
                continue
            hi, lo = g["high"].max(), g["low"].min()
            c_close = g["close"].iloc[-1]
            vwap = amt.sum() / day_vol
            rng = hi - lo
            rows.append({
                "symbol": sym, "date": pd.Timestamp(dt),
                "eop_vol_ratio": vol[-6:].sum() / day_vol,
                "open_vol_ratio": vol[:6].sum() / day_vol,
                "intraday_recovery": (c_close - lo) / rng if rng > 0 else np.nan,
                "vwap_dev_close": c_close / vwap - 1.0 if vwap > 0 else np.nan,
                "n_bars": len(g),
            })
        frames.append(pd.DataFrame(rows))
    out = pd.concat(frames, ignore_index=True)
    return out


def build_mf():
    frames = [pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(MFDIR, "mf_*.parquet")))]
    d = pd.concat(frames, ignore_index=True)
    d["symbol"] = d["ts_code"].str.split(".").str[0]
    d["date"] = pd.to_datetime(d["trade_date"], format="%Y%m%d")
    buy_tot = sum(d[f"buy_{k}_amount"] for k in ("sm", "md", "lg", "elg"))
    sell_tot = sum(d[f"sell_{k}_amount"] for k in ("sm", "md", "lg", "elg"))
    d["tot_amt"] = buy_tot + sell_tot
    d["mf_lg_net_ratio"] = (d["buy_lg_amount"] - d["sell_lg_amount"]) / d["tot_amt"]
    d["mf_elg_net_ratio"] = (d["buy_elg_amount"] - d["sell_elg_amount"]) / d["tot_amt"]
    d["mf_net_ratio"] = d["net_mf_amount"] / d["tot_amt"]
    d = d.sort_values(["symbol", "date"])
    for k in ("lg", "elg"):
        b = d.groupby("symbol")[f"buy_{k}_amount"].transform(lambda s: s.rolling(5).sum())
        s = d.groupby("symbol")[f"sell_{k}_amount"].transform(lambda s: s.rolling(5).sum())
        t = d.groupby("symbol")["tot_amt"].transform(lambda s: s.rolling(5).sum())
        d[f"mf_{k}_net5_ratio"] = (b - s) / t
    return d[["symbol", "date", "mf_lg_net_ratio", "mf_elg_net_ratio", "mf_net_ratio",
              "mf_lg_net5_ratio", "mf_elg_net5_ratio"]]


def report(name, df, cols):
    print(f"\n== {name}: rows={len(df):,} stocks={df['symbol'].nunique()} "
          f"dates {df['date'].min().date()}~{df['date'].max().date()}")
    for c in cols:
        s = df[c].dropna()
        print(f"  {c:22s} nan={df[c].isna().mean():5.1%} "
              f"mean={s.mean():+.4f} p5={s.quantile(.05):+.3f} p95={s.quantile(.95):+.3f}")


if __name__ == "__main__":
    bs5 = build_bs5(SAMPLE)
    bs5.to_parquet(OUT_BS5, index=False)
    report("bs5 L2 (10股样本)", bs5,
           ["eop_vol_ratio", "open_vol_ratio", "intraday_recovery", "vwap_dev_close"])
    print("bars/day:", sorted(bs5["n_bars"].unique()))

    mf = build_mf()
    mf.to_parquet(OUT_MF, index=False)
    report("moneyflow L2 (725日全量)", mf,
           ["mf_lg_net_ratio", "mf_elg_net_ratio", "mf_net_ratio",
            "mf_lg_net5_ratio", "mf_elg_net5_ratio"])
    print("L2_BUILD_PILOT_DONE")
