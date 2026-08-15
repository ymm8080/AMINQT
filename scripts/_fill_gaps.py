"""全市场宇宙修复 Step 2b: 拉取缺口检测 + 回填 (2026-08-15).

_pull_universe_raw.py 的 _fetch_retry 4 次全败会永久跳过该日期该源
(Tushare 超时潮期间发生). 本脚本在拉取完成后跑, 检测每个源缺失的日期并重拉:

  A. 单日文件源: adj_factor/adj_YYYYMMDD.parquet, stk_limit/YYYYMMDD_all__.parquet
     — 按文件名对比日历, 缺哪个补哪个 (与拉取同列同格式)
  B. 累计源: cyq/margin/lhb/bt — 读全部 batch+canonical 文件的日期集合,
     缺失日期重拉 → 写 fill 批次 → 出新的 canonical WORM 文件 (builders 端
     dedup, 新旧 canonical 可共存)
  C. 新符号 daily/daily_basic/suspend — 日期集合对比, 缺失重拉过滤到新符号

重试 6 次 + 长退避 (缺口少, 慢而稳). 结束打印各源剩余缺口 (应为 0).

WORM: fill 批次 + cyq_full/margin_panel/all_/block_trade_full 新时间戳 canonical.
"""

from __future__ import annotations

import glob
import os
import random
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tushare as ts  # noqa: E402

from config import settings  # noqa: E402

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
OUT_DIR = "data/new_symbols_raw"
ALT_DIR = "data/supply_cache/alt_data"


def _calendar() -> list[str]:
    p = pd.read_parquet(PANEL, columns=["date"])
    return sorted(p["date"].astype(str).str[:10].str.replace("-", "").unique())


def _to_symbol(df: pd.DataFrame) -> pd.DataFrame:
    if "ts_code" in df.columns:
        df["symbol"] = (
            df["ts_code"]
            .str.replace(".SZ", "", regex=False)
            .str.replace(".SH", "", regex=False)
            .str.replace(".BJ", "", regex=False)
        )
    return df


def _fetch(pro, fn, name, **kw) -> pd.DataFrame:
    for i in range(1, 7):
        try:
            df = fn(**kw)
            if df is not None and len(df):
                return _to_symbol(df)
        except Exception as e:  # noqa: BLE001
            print(f"    {name}: FAIL attempt {i}: {e}", flush=True)
        time.sleep(4 * i + random.random())
    return pd.DataFrame()


def _date_set(files, col: str) -> set[str]:
    if not files:
        return set()
    out = set()
    for f in files:
        try:
            s = pd.read_parquet(f, columns=[col])[col]
        except Exception:  # noqa: BLE001
            continue
        s = pd.to_datetime(s, errors="coerce").dropna().dt.strftime("%Y%m%d")
        out |= set(s.unique())
    return out


def main() -> None:
    cal = _calendar()
    d0, d1 = cal[0], cal[-1]
    cal_set = set(cal)
    print(f"[fill] calendar={len(cal)} ({d0}..{d1})", flush=True)
    universe = pd.read_parquet(
        sorted(glob.glob("data/new_universe/new_symbols_*.parquet"))[-1]
    )
    newsyms = set(universe["symbol"])
    pro = ts.pro_api(settings.TUSHARE_TOKEN)

    # ── 检测缺口 ──
    adj_present = {
        os.path.basename(f)[4:12]
        for f in glob.glob(os.path.join(ALT_DIR, "adj_factor", "adj_*.parquet"))
    }
    lim_present = {
        os.path.basename(f)[:8]
        for f in glob.glob(os.path.join(ALT_DIR, "stk_limit", "*_all__.parquet"))
    }
    cyq_files = glob.glob(os.path.join(ALT_DIR, "cyq_tushare", "cyq_*.parquet"))
    marg_files = glob.glob(os.path.join(ALT_DIR, "margin_*.parquet"))
    lhb_files = glob.glob(os.path.join(ALT_DIR, "lhb", "*.parquet"))
    bt_files = glob.glob(os.path.join(ALT_DIR, "block_trade", "*.parquet"))
    daily_files = glob.glob(os.path.join(OUT_DIR, "daily", "daily_*.parquet"))
    basic_files = glob.glob(
        os.path.join(OUT_DIR, "daily_basic", "daily_basic_*.parquet")
    )
    susp_files = glob.glob(os.path.join(OUT_DIR, "suspend", "suspend_*.parquet"))

    gaps = {
        "adj": sorted(cal_set - adj_present),
        "limit": sorted(cal_set - lim_present),
        "cyq": sorted(cal_set - _date_set(cyq_files, "trade_date")),
        "margin": sorted(cal_set - _date_set(marg_files, "date")),
        "lhb": sorted(cal_set - _date_set(lhb_files, "date")),
        "bt": sorted(cal_set - _date_set(bt_files, "trade_date")),
        "daily": sorted(cal_set - _date_set(daily_files, "date")),
        "basic": sorted(cal_set - _date_set(basic_files, "date")),
        "susp": sorted(cal_set - _date_set(susp_files, "date")),
    }
    for k, v in gaps.items():
        print(f"[gap] {k}: {len(v)} missing", flush=True)
    if not any(gaps.values()):
        print("[fill] 无缺口", flush=True)
        return

    t0 = time.time()
    done: dict[str, list[str]] = {k: [] for k in gaps}

    # ── 单日文件源 (adj/limit) ──
    for ds in gaps["adj"]:
        a = _fetch(pro, pro.adj_factor, f"adj {ds}", trade_date=ds)
        if len(a):
            a[["ts_code", "trade_date", "adj_factor", "symbol"]].to_parquet(
                os.path.join(ALT_DIR, "adj_factor", f"adj_{ds}.parquet"), index=False
            )
            done["adj"].append(ds)
    for ds in gaps["limit"]:
        l = _fetch(pro, pro.stk_limit, f"limit {ds}", trade_date=ds)
        if len(l):
            l2 = l[["symbol", "trade_date", "up_limit", "down_limit"]].rename(
                columns={"up_limit": "up_limit_raw", "down_limit": "down_limit_raw"}
            )
            l2["date"] = pd.Timestamp(ds)
            l2.to_parquet(
                os.path.join(ALT_DIR, "stk_limit", f"{ds}_all__.parquet"), index=False
            )
            done["limit"].append(ds)

    # ── 累计源 (cyq/margin/lhb/bt) ──
    acc_cyq, acc_marg, acc_lhb, acc_bt = [], [], [], []
    for ds in sorted(
        set(gaps["cyq"]) | set(gaps["margin"]) | set(gaps["lhb"]) | set(gaps["bt"])
    ):
        if ds in gaps["cyq"]:
            c = _fetch(pro, pro.cyq_perf, f"cyq {ds}", trade_date=ds)
            if len(c):
                keep = [
                    x
                    for x in [
                        "symbol",
                        "trade_date",
                        "his_low",
                        "his_high",
                        "cost_5pct",
                        "cost_15pct",
                        "cost_50pct",
                        "cost_85pct",
                        "cost_95pct",
                        "weight_avg",
                        "winner_rate",
                    ]
                    if x in c.columns
                ]
                acc_cyq.append(c[keep])
                done["cyq"].append(ds)
        if ds in gaps["margin"]:
            m = _fetch(pro, pro.margin_detail, f"margin {ds}", trade_date=ds)
            if len(m):
                m2 = m[
                    ["symbol", "trade_date", "rzye", "rqye", "rzmre", "rqyl"]
                ].rename(
                    columns={
                        "rzye": "margin_balance",
                        "rqye": "short_balance",
                        "rzmre": "margin_buy_amt",
                        "rqyl": "short_sell_vol",
                        "trade_date": "date",
                    }
                )
                m2["date"] = pd.Timestamp(ds)
                acc_marg.append(m2)
                done["margin"].append(ds)
        if ds in gaps["lhb"]:
            t = _fetch(pro, pro.top_list, f"lhb {ds}", trade_date=ds)
            if len(t):
                g = (
                    t.groupby("symbol")
                    .agg(
                        lhb_buy_amt=("l_buy", "sum"),
                        lhb_sell_amt=("l_sell", "sum"),
                        lhb_net_buy=("net_amount", "sum"),
                    )
                    .reset_index()
                )
                g["date"] = pd.Timestamp(ds)
                acc_lhb.append(g)
                done["lhb"].append(ds)
        if ds in gaps["bt"]:
            b = _fetch(pro, pro.block_trade, f"bt {ds}", trade_date=ds)
            if len(b):
                keep = [
                    x
                    for x in [
                        "symbol",
                        "ts_code",
                        "trade_date",
                        "price",
                        "vol",
                        "amount",
                        "buyer",
                        "seller",
                    ]
                    if x in b.columns
                ]
                b2 = b[keep].copy()
                b2["date"] = pd.Timestamp(ds)
                acc_bt.append(b2)
                done["bt"].append(ds)
        if (
            len(done["cyq"]) + len(done["margin"]) + len(done["lhb"]) + len(done["bt"])
        ) % 25 == 0:
            print(
                f"[fill] 累计源已补 {len(done['cyq'])}/{len(done['margin'])}/"
                f"{len(done['lhb'])}/{len(done['bt'])} 日期 "
                f"({(time.time() - t0) / 60:.0f} min)",
                flush=True,
            )

    ts_ = datetime.now().strftime("%Y%m%d_%H%M%S")
    if acc_cyq:
        d = os.path.join(ALT_DIR, "cyq_tushare")
        os.makedirs(d, exist_ok=True)
        pd.concat(acc_cyq, ignore_index=True).to_parquet(
            os.path.join(d, f"cyq_fill_{ts_}.parquet"), index=False
        )
    if acc_marg:
        pd.concat(acc_marg, ignore_index=True).to_parquet(
            os.path.join(ALT_DIR, f"margin_fill_{ts_}.parquet"), index=False
        )
    if acc_lhb:
        d = os.path.join(ALT_DIR, "lhb")
        os.makedirs(d, exist_ok=True)
        pd.concat(acc_lhb, ignore_index=True).to_parquet(
            os.path.join(d, f"lhb_fill_{ts_}.parquet"), index=False
        )
    if acc_bt:
        d = os.path.join(ALT_DIR, "block_trade")
        os.makedirs(d, exist_ok=True)
        pd.concat(acc_bt, ignore_index=True).to_parquet(
            os.path.join(d, f"bt_fill_{ts_}.parquet"), index=False
        )

    # ── 新符号 daily/basic/susp ──
    for name, key, src_fn in [
        ("daily", "daily", pro.daily),
        ("basic", "daily_basic", pro.daily_basic),
        ("susp", "suspend", pro.suspend_d),
    ]:
        parts = []
        for ds in gaps[key]:
            d_ = _fetch(pro, src_fn, f"{name} {ds}", trade_date=ds)
            if not len(d_):
                continue
            if key == "daily":
                keep = [
                    x
                    for x in [
                        "symbol",
                        "trade_date",
                        "open",
                        "high",
                        "low",
                        "close",
                        "pre_close",
                        "amount",
                        "vol",
                    ]
                    if x in d_.columns
                ]
            elif key == "daily_basic":
                keep = [
                    x
                    for x in [
                        "symbol",
                        "trade_date",
                        "turnover_rate",
                        "turnover_rate_f",
                        "volume_ratio",
                        "pe_ttm",
                        "pb",
                        "ps_ttm",
                        "total_share",
                        "float_share",
                        "free_share",
                        "total_mv",
                        "circ_mv",
                        "dv_ratio",
                        "dv_ttm",
                    ]
                    if x in d_.columns
                ]
            else:
                keep = ["symbol", "trade_date"]
            sub = d_[d_["symbol"].isin(newsyms)][keep].copy()
            if len(sub):
                sub["date"] = pd.Timestamp(ds)
                parts.append(sub)
                done[key].append(ds)
        if parts:
            d = os.path.join(OUT_DIR, key)
            os.makedirs(d, exist_ok=True)
            pd.concat(parts, ignore_index=True).to_parquet(
                os.path.join(d, f"{key}_fill_{ts_}.parquet"), index=False
            )

    # ── 剩余缺口报告 ──
    print("\n[fill] 剩余缺口 (应为空):", flush=True)
    leftover = 0
    for k in gaps:
        rem = sorted(set(gaps[k]) - set(done[k]))
        if rem:
            leftover += len(rem)
            print(f"  {k}: {len(rem)} 仍缺 {rem[:6]}", flush=True)
    print(f"[fill] DONE, 仍缺 {leftover} 日期源次", flush=True)


if __name__ == "__main__":
    main()
