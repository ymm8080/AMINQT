# -*- coding: utf-8 -*-
"""全市场宇宙修复 Step 2: 全市场原始数据拉取 + supply 缓存恢复 (2026-08-15).

背景: 2026-08-15 02:10 data/supply_cache/alt_data/ 被外部清空 (面板本身完好).
本脚本一次拉齐:
  A. 恢复生产缓存 (全市场, 按旧约定路径/格式):
     - adj_factor/adj_YYYYMMDD.parquet           (单日文件: ts_code/trade_date/adj_factor/symbol)
     - stk_limit/YYYYMMDD_all__.parquet          (单日文件: symbol/date/up_limit_raw/down_limit_raw)
     - cyq_tushare/cyq_full_<ts>.parquet         (cost_* + weight_avg + winner_rate)
     - margin_panel_<ts>.parquet                 (symbol/date/margin_balance/short_balance/margin_buy_amt/short_sell_vol)
     - lhb/all_<d0>_<d1>_<ts>.parquet            (symbol/date/lhb_net_buy/lhb_buy_amt/lhb_sell_amt)
     - block_trade/block_trade_full_<ts>.parquet (symbol/date/ts_code/trade_date/price/vol/amount/buyer/seller)
  B. 新股票构建输入 (data/new_symbols_raw/):
     - daily / daily_basic / suspend             (逐日全市场拉, 过滤到待补符号)
     - top_inst                                  (全市场, 仅 LHB 日期 — 同时当缓存恢复)
     - fina                                      (逐 symbol 全期, 24 字段)

断点续传: data/new_symbols_raw/progress.json. 用法:
    python scripts/_pull_universe_raw.py --dry-run
    python scripts/_pull_universe_raw.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
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
FINA_FIELDS = (
    "ts_code,ann_date,end_date,roe,roe_dt,roa,np_margin,gross_margin,"
    "eps,ocfps,bps,revenue_ps,eps_yoy,or_yoy,profit_yoy,debt_to_assets,"
    "current_ratio,assets_turn,ar_turn,inv_turn,ocf_to_or,"
    "dt_eps,roe_yoy,q_roe,q_ocf_to_sales"
)
CALL_SLEEP = 0.12
FLUSH_EVERY = 100


def _to_symbol(df: pd.DataFrame) -> pd.DataFrame:
    if "ts_code" in df.columns:
        df["symbol"] = (
            df["ts_code"].str.replace(".SZ", "", regex=False)
            .str.replace(".SH", "", regex=False)
            .str.replace(".BJ", "", regex=False)
        )
    return df


def _fetch_retry(fn, name, attempts=4, **kwargs) -> pd.DataFrame:
    for i in range(1, attempts + 1):
        try:
            df = fn(**kwargs)
            if df is not None and len(df):
                return _to_symbol(df)
            print(f"    {name}: empty (attempt {i})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"    {name}: FAIL attempt {i}: {e}", flush=True)
        time.sleep(3 * i)
    return pd.DataFrame()


class Progress:
    def __init__(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        self.path = os.path.join(OUT_DIR, "progress.json")
        self.data = (
            json.load(open(self.path, encoding="utf-8"))
            if os.path.exists(self.path) else {}
        )

    def get(self, src: str):
        return self.data.get(src)

    def set(self, src: str, val):
        self.data[src] = val
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1)


def _panel_calendar() -> pd.DatetimeIndex:
    p = pd.read_parquet(PANEL, columns=["date"])
    return pd.DatetimeIndex(sorted(p["date"].unique()))


def _new_symbols() -> pd.DataFrame:
    f = sorted(glob.glob("data/new_universe/new_symbols_*.parquet"))[-1]
    return pd.read_parquet(f)


def _save(src: str, df: pd.DataFrame, subdir: str):
    os.makedirs(os.path.join(OUT_DIR, subdir), exist_ok=True)
    ts_ = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_parquet(os.path.join(OUT_DIR, subdir, f"{src}_{ts_}.parquet"), index=False)


def _save_alt(df: pd.DataFrame, fname: str, subdir: str):
    d = os.path.join(ALT_DIR, subdir)
    os.makedirs(d, exist_ok=True)
    df.to_parquet(os.path.join(d, fname), index=False)


def _batch_count(base: str, subdir: str) -> int:
    return len(glob.glob(os.path.join(ALT_DIR, subdir, f"{base}_b*.parquet")))


def _flush_batch(parts, base: str, subdir: str, seq: int):
    if not parts:
        return seq
    d = os.path.join(ALT_DIR, subdir)
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, f"{base}_b{seq:03d}.parquet")
    pd.concat(parts, ignore_index=True).to_parquet(f, index=False)
    return seq + 1


def _finalize(base: str, subdir: str, canonical: str) -> None:
    """合并批次文件 → canonical 单文件 (WORM 时间戳名), 删中间批次."""
    batches = sorted(glob.glob(os.path.join(ALT_DIR, subdir, f"{base}_b*.parquet")))
    if not batches:
        return
    df = pd.concat([pd.read_parquet(f) for f in batches], ignore_index=True)
    _save_alt(df, canonical, subdir)
    print(f"[finalize] {canonical}: {len(df):,} rows "
          f"(merged {len(batches)} batches)", flush=True)
    for f in batches:
        os.remove(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    universe = _new_symbols()
    newsyms = set(universe["symbol"])
    cal = _panel_calendar()
    d0, d1 = cal[0].strftime("%Y%m%d"), cal[-1].strftime("%Y%m%d")
    print(f"[plan] new symbols={len(newsyms)} | dates={len(cal)} ({d0}..{d1})")
    print("[plan] per-date full market: daily/daily_basic/adj_factor/cyq_perf/"
          "suspend/stk_limit/margin/top_list/block_trade")
    print("[plan] top_inst: LHB dates | fina: per new symbol")
    if args.dry_run:
        return

    prog = Progress()
    pro = ts.pro_api(settings.TUSHARE_TOKEN)
    done_date = prog.get("perdate") or ""
    todo = [d for d in cal if d.strftime("%Y%m%d") > done_date]
    print(f"[perdate] todo={len(todo)} (resume after {done_date or 'none'})", flush=True)

    today = datetime.now().strftime("%Y%m%d")
    cyq_base, marg_base, lhb_base, bt_base = "cyq", "margin", "lhb", "bt"
    seq_cyq = _batch_count(cyq_base, "cyq_tushare")
    seq_marg = _batch_count(marg_base, "")
    seq_lhb = _batch_count(lhb_base, "lhb")
    seq_bt = _batch_count(bt_base, "block_trade")
    acc_cyq, acc_margin, acc_lhb, acc_bt = [], [], [], []
    acc_daily, acc_basic, acc_susp = [], [], []
    t0 = time.time()
    for i, d in enumerate(todo):
        ds = d.strftime("%Y%m%d")
        # A: adj_factor / stk_limit — 单日文件 (旧约定: 每日期一个文件)
        a = _fetch_retry(pro.adj_factor, f"adj {ds}", trade_date=ds)
        if len(a):
            _save_alt(a[["ts_code", "trade_date", "adj_factor", "symbol"]],
                      f"adj_{ds}.parquet", "adj_factor")
        l = _fetch_retry(pro.stk_limit, f"limit {ds}", trade_date=ds)
        if len(l):
            l2 = l[["symbol", "trade_date", "up_limit", "down_limit"]].rename(
                columns={"up_limit": "up_limit_raw", "down_limit": "down_limit_raw"})
            l2["date"] = pd.Timestamp(d)
            _save_alt(l2, f"{ds}_all__.parquet", "stk_limit")
        # A: cyq_perf (全市场: cost 分位 + weight_avg + winner_rate)
        c = _fetch_retry(pro.cyq_perf, f"cyq {ds}", trade_date=ds)
        if len(c):
            keep = [x for x in ["symbol", "trade_date", "his_low", "his_high",
                                "cost_5pct", "cost_15pct", "cost_50pct",
                                "cost_85pct", "cost_95pct", "weight_avg",
                                "winner_rate"] if x in c.columns]
            acc_cyq.append(c[keep])
        # A: margin (全市场, 面板口径)
        m = _fetch_retry(pro.margin_detail, f"margin {ds}", trade_date=ds)
        if len(m):
            m2 = m[["symbol", "trade_date", "rzye", "rqye", "rzmre", "rqyl"]].rename(
                columns={"rzye": "margin_balance", "rqye": "short_balance",
                         "rzmre": "margin_buy_amt", "rqyl": "short_sell_vol",
                         "trade_date": "date"})
            m2["date"] = pd.Timestamp(d)
            acc_margin.append(m2)
        # A: top_list → lhb 缓存 (全市场, 面板口径 3 列)
        t = _fetch_retry(pro.top_list, f"lhb {ds}", trade_date=ds)
        if len(t):
            g = t.groupby("symbol").agg(
                lhb_buy_amt=("l_buy", "sum"),
                lhb_sell_amt=("l_sell", "sum"),
                lhb_net_buy=("net_amount", "sum"),
            ).reset_index()
            g["date"] = pd.Timestamp(d)
            acc_lhb.append(g)
        # A: block_trade (全市场, raw 缓存格式)
        b = _fetch_retry(pro.block_trade, f"bt {ds}", trade_date=ds)
        if len(b):
            keep = [x for x in ["symbol", "ts_code", "trade_date", "price",
                                "vol", "amount", "buyer", "seller"] if x in b.columns]
            b2 = b[keep].copy()
            b2["date"] = pd.Timestamp(d)
            acc_bt.append(b2)
        # B: daily / daily_basic / suspend → 过滤新符号
        d_ = _fetch_retry(pro.daily, f"daily {ds}", trade_date=ds)
        if len(d_):
            keep = [x for x in ["symbol", "trade_date", "open", "high", "low",
                                "close", "pre_close", "amount", "vol"] if x in d_.columns]
            sub = d_[d_["symbol"].isin(newsyms)][keep].copy()
            if len(sub):
                sub["date"] = pd.Timestamp(d)
                acc_daily.append(sub)
        db = _fetch_retry(pro.daily_basic, f"basic {ds}", trade_date=ds)
        if len(db):
            keep = [x for x in ["symbol", "trade_date", "turnover_rate",
                                "turnover_rate_f", "volume_ratio", "pe_ttm", "pb",
                                "ps_ttm", "total_share", "float_share", "free_share",
                                "total_mv", "circ_mv", "dv_ratio", "dv_ttm"] if x in db.columns]
            sub = db[db["symbol"].isin(newsyms)][keep].copy()
            if len(sub):
                sub["date"] = pd.Timestamp(d)
                acc_basic.append(sub)
        s = _fetch_retry(pro.suspend_d, f"susp {ds}", trade_date=ds)
        if len(s):
            sub = s[s["symbol"].isin(newsyms)][["symbol", "trade_date"]].copy()
            if len(sub):
                sub["date"] = pd.Timestamp(d)
                acc_susp.append(sub)
        time.sleep(CALL_SLEEP)
        if (i + 1) % FLUSH_EVERY == 0:
            for name, parts, sub in [
                ("daily", acc_daily, "daily"), ("daily_basic", acc_basic, "daily_basic"),
                ("suspend", acc_susp, "suspend"),
            ]:
                if parts:
                    _save(name, pd.concat(parts, ignore_index=True), sub)
            acc_daily, acc_basic, acc_susp = [], [], []
            seq_cyq = _flush_batch(acc_cyq, cyq_base, "cyq_tushare", seq_cyq)
            seq_marg = _flush_batch(acc_margin, marg_base, "", seq_marg)
            seq_lhb = _flush_batch(acc_lhb, lhb_base, "lhb", seq_lhb)
            seq_bt = _flush_batch(acc_bt, bt_base, "block_trade", seq_bt)
            acc_cyq, acc_margin, acc_lhb, acc_bt = [], [], [], []
            prog.set("perdate", ds)
            rate = (i + 1) / (time.time() - t0) * 3600
            print(f"[perdate] {i+1}/{len(todo)} ({rate:.0f}/hr) flushed @ {ds}", flush=True)
    # 尾部 flush
    for name, parts, sub in [("daily", acc_daily, "daily"), ("daily_basic", acc_basic, "daily_basic"),
                             ("suspend", acc_susp, "suspend")]:
        if parts:
            _save(name, pd.concat(parts, ignore_index=True), sub)
    _flush_batch(acc_cyq, cyq_base, "cyq_tushare", seq_cyq)
    _flush_batch(acc_margin, marg_base, "", seq_marg)
    _flush_batch(acc_lhb, lhb_base, "lhb", seq_lhb)
    _flush_batch(acc_bt, bt_base, "block_trade", seq_bt)
    if len(todo):
        prog.set("perdate", todo[-1].strftime("%Y%m%d"))
    print(f"[perdate] DONE in {(time.time()-t0)/60:.1f} min", flush=True)

    # 批次合并 (幂等: 无批次文件时跳过)
    _finalize(cyq_base, "cyq_tushare", f"cyq_full_{today}_{datetime.now().strftime('%H%M%S')}.parquet")
    _finalize(marg_base, "", f"margin_panel_{today}_{datetime.now().strftime('%H%M%S')}.parquet")
    _finalize(lhb_base, "lhb", f"all_{d0}_{d1}_{today}.parquet")
    _finalize(bt_base, "block_trade", f"block_trade_full_{today}_{datetime.now().strftime('%H%M%S')}.parquet")

    # top_inst: 全市场, 仅 LHB 日期 (同时恢复缓存)
    if prog.get("top_inst") != "done":
        lhb_files = glob.glob(os.path.join(ALT_DIR, "lhb", "all_*.parquet"))
        lhb = pd.concat([pd.read_parquet(f, columns=["symbol", "date"]) for f in lhb_files])
        lhb_dates = sorted(set(pd.to_datetime(lhb["date"]).dt.strftime("%Y%m%d")))
        print(f"[top_inst] LHB dates={len(lhb_dates)}", flush=True)
        acc = []
        for i, ds in enumerate(lhb_dates):
            t = _fetch_retry(pro.top_inst, f"top_inst {ds}", trade_date=ds)
            if len(t):
                # ts_code 必须保留: lhb_seats.seat_wide_from_top_inst 依赖它剥 symbol
                keep = [x for x in ["symbol", "ts_code", "trade_date", "exalter",
                                    "buy", "sell"] if x in t.columns]
                t2 = t[keep].copy()
                t2["date"] = pd.Timestamp(ds)
                acc.append(t2)
            time.sleep(CALL_SLEEP)
            if (i + 1) % FLUSH_EVERY == 0:
                if acc:
                    _save("top_inst", pd.concat(acc, ignore_index=True), "top_inst")
                    acc = []
                print(f"[top_inst] {i+1}/{len(lhb_dates)}", flush=True)
        if acc:
            _save("top_inst", pd.concat(acc, ignore_index=True), "top_inst")
        prog.set("top_inst", "done")
        print("[top_inst] DONE", flush=True)

    # fina: 逐新符号全期
    if prog.get("fina") != "done":
        ts_map = dict(zip(universe["symbol"], universe["ts_code"], strict=False))
        todo_fina = sorted(newsyms)
        print(f"[fina] todo={len(todo_fina)} symbols", flush=True)
        acc = []
        t0 = time.time()
        for i, sym in enumerate(todo_fina):
            code = ts_map.get(sym, sym + (".SH" if sym.startswith(("6", "5")) else ".SZ"))
            f = _fetch_retry(pro.fina_indicator, f"fina {sym}",
                             ts_code=code, start_date="20221001", end_date="20260815",
                             fields=FINA_FIELDS)
            if len(f):
                acc.append(f)
            time.sleep(CALL_SLEEP)
            if (i + 1) % FLUSH_EVERY == 0:
                if acc:
                    _save("fina", pd.concat(acc, ignore_index=True), "fina")
                    acc = []
                print(f"[fina] {i+1}/{len(todo_fina)} "
                      f"({(i+1)/(time.time()-t0)*3600:.0f}/hr)", flush=True)
        if acc:
            _save("fina", pd.concat(acc, ignore_index=True), "fina")
        prog.set("fina", "done")
        print("[fina] DONE", flush=True)
    print("ALL DONE")


if __name__ == "__main__":
    main()
