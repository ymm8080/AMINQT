# -*- coding: utf-8 -*-
"""诊断: 龙虎榜 (LHB) 出现过但 V3 面板 (3220 只) 里没有的股票, 逐只归类.

归类口径与生产 ingest gate 完全一致 (app/pipeline1/ingest_scan.py):
  - name_is_st: 名称以 ST/*ST/退 开头 → 剔
  - list_days < INGEST_MIN_LIST_DAYS(150) 个交易日 (按面板交易日历) → 剔
  - list_date 缺失 → ingest_scan 保守剔 (天数计 0), 单独归一类
  - 北交所 (4/8/92 开头): V3 只覆盖沪深 (symbol 剥 .SZ/.SH)
  - B股 (900/200 开头): pro.daily 无 B股, V3 设计上不含
  - 转债 (11/12 开头): AKShare LHB 含转债条目, V3 是股票面板
  - 退市/摘牌: stock_basic status=D 或无信息
  - 其余 → 未解释: 附面板覆盖缺口分析 (停牌/ST时段/单日缺口)

WORM: 输出 data/_diag_lhb_v3_gap_<ts>.parquet, 不覆盖旧文件.
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.universe_manager import name_is_st  # noqa: E402
from config import settings  # noqa: E402

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
LHB_DIR = "data/supply_cache/alt_data/lhb"
MIN_LIST_DAYS = settings.INGEST_MIN_LIST_DAYS
# 面板起始后 MIN_LIST_DAYS 个交易日之内的事件, 上市天数口径可能偏小, 单独标注
CAL_SLACK = MIN_LIST_DAYS + 20


def load_lhb_events() -> pd.DataFrame:
    files = glob.glob(os.path.join(LHB_DIR, "all_*.parquet"))
    parts = []
    for f in files:
        parts.append(pd.read_parquet(f, columns=["symbol", "date"]))
    for f in glob.glob(os.path.join(LHB_DIR, "_months*", "*.parquet")):
        parts.append(pd.read_parquet(f, columns=["symbol", "date"]))
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.strip()
    return df.drop_duplicates()


def main() -> None:
    panel = pd.read_parquet(PANEL, columns=["symbol", "date"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel["symbol"] = panel["symbol"].astype(str).str.strip()
    cal = pd.DatetimeIndex(sorted(panel["date"].unique()))
    panel_min, panel_max = cal.min(), cal.max()
    print(f"[panel] rows={len(panel):,} symbols={panel['symbol'].nunique()} "
          f"dates={panel_min.date()}..{panel_max.date()}")
    sym_dates = panel.groupby("symbol")["date"].apply(sorted).to_dict()

    lhb = load_lhb_events()
    lhb = lhb[(lhb["date"] >= panel_min) & (lhb["date"] <= panel_max)]
    print(f"[lhb] events in panel window={len(lhb):,} "
          f"symbols={lhb['symbol'].nunique()}")

    panel_keys = set(zip(panel["symbol"], panel["date"]))
    missing = pd.DataFrame(
        [{"symbol": s, "date": d}
         for s, d in set(zip(lhb["symbol"], lhb["date"])) - panel_keys]
    )
    print(f"[missing] {len(missing):,} 事件, {missing['symbol'].nunique()} 只")

    # stock_basic (Tushare)
    basic = None
    try:
        import tushare as ts

        token = settings.TUSHARE_TOKEN or ts.get_token()
        pro = ts.pro_api(token)
        basic = pro.stock_basic(
            fields="ts_code,symbol,name,list_date,market,list_status"
        )
        print(f"[stock_basic] {len(basic):,} 只")
    except Exception as exc:  # noqa: BLE001
        print(f"[stock_basic] 拉取失败: {exc} — 按无基本信息归类")

    if basic is not None:
        basic = basic.drop_duplicates(subset="symbol", keep="first")
        info = basic.set_index(basic["symbol"].astype(str).str.strip())
        missing["name"] = missing["symbol"].map(info["name"]).fillna("")
        missing["list_date"] = missing["symbol"].map(info["list_date"])
        missing["market"] = missing["symbol"].map(info["market"]).fillna("")
        missing["status"] = missing["symbol"].map(info["list_status"]).fillna("")
    else:
        missing["name"] = ""
        missing["list_date"] = pd.NA
        missing["market"] = ""
        missing["status"] = ""

    # 上市交易日数: 事件日 - 首个 >=list_date 的交易日 (searchsorted, 同 ingest_scan)
    # 注意: 面板日历起点 2023-01-03, list_date 早于起点 → 天数截断, 不能据此判次新.
    ld = pd.to_datetime(missing["list_date"], format="%Y%m%d", errors="coerce")
    left = cal.searchsorted(ld)  # list_date 在面板日历前 → 0; NaT → len(cal)
    right = cal.searchsorted(pd.to_datetime(missing["date"]))
    missing["list_days"] = np.where(ld.isna(), np.nan, right - left)

    first_evt = missing.groupby("symbol")["date"].min()
    first_evt_pos = pd.Series(cal.searchsorted(pd.to_datetime(first_evt)),
                              index=first_evt.index)
    first_ld = missing.drop_duplicates(subset="symbol").set_index("symbol")["list_days"]
    # 上市晚于面板起点: 口径精确, 首次事件日 list_days < 150 即真次新.
    # 上市早于面板起点: 非次新; 但首次事件落在面板前 170 日内 → 上市天数口径截断, 标记模糊.
    is_cal_pre = pd.to_datetime(first_evt) <= cal[CAL_SLACK]

    def classify(row) -> str:
        sym = row["symbol"]
        if sym.startswith(("4", "8", "92")):
            return "北交所 (V3 不覆盖)"
        if sym.startswith("900") or sym.startswith("200"):
            return "B股 (V3 不覆盖)"
        if sym.startswith(("11", "12")):
            return "转债 (非股票)"
        if row["status"] == "D" or not row["name"]:
            return "退市/无基本信息"
        if name_is_st(row["name"]):
            return "ST/*ST 剔"
        ld_val = row["list_date"]
        if pd.isna(ld_val):
            return "list_date 缺失 (保守剔)"
        if pd.to_datetime(ld_val, format="%Y%m%d", errors="coerce") >= cal[0]:
            # 上市晚于面板起点 → list_days 精确
            if first_ld.get(sym, np.inf) < MIN_LIST_DAYS:
                return f"次新 <{MIN_LIST_DAYS}交易日"
            return "未解释"
        # 上市早于面板起点 → 非次新
        if is_cal_pre.get(sym, False):
            return "面板起始 170 日内 (口径模糊)"
        return "未解释"

    missing["reason"] = missing.apply(classify, axis=1)

    # 未解释 + 口径模糊: 面板覆盖缺口分析
    def gap_analysis(sym: str, dts: pd.DatetimeIndex) -> tuple[int, str, str]:
        rows = np.array(sym_dates.get(sym, []))
        if rows.size == 0:
            return -1, "从未进面板", "从未进面板"
        d = np.array(dts)
        out = []
        for dd in d:
            pos = np.searchsorted(rows, dd)
            if pos < rows.size and rows[pos] == dd:
                continue
            prev_ = rows[pos - 1] if pos > 0 else None
            nxt = rows[pos] if pos < rows.size else None
            gap = (nxt - prev_).days if prev_ is not None and nxt is not None else None
            out.append((dd, prev_, nxt, gap))
        return len(out), out[0][3] if out else 0, out[-1][3] if out else 0

    unk = missing[missing["reason"].isin(["未解释", "面板起始 170 日内 (口径模糊)"])].copy()
    if len(unk):
        g = unk.groupby("symbol")["date"].apply(list)
        meta = {}
        for sym, dts in g.items():
            n_events, first_gap, last_gap = gap_analysis(sym, pd.DatetimeIndex(dts))
            meta[sym] = (n_events, first_gap, last_gap)
        unk["miss_events"] = unk["symbol"].map(lambda s: meta[s][0])
        unk["first_gap_days"] = unk["symbol"].map(lambda s: meta[s][1])
        unk["last_gap_days"] = unk["symbol"].map(lambda s: meta[s][2])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join("data", f"_diag_lhb_v3_gap_{ts}.parquet")
    missing.to_parquet(out, index=False)
    print(f"[save] {out}")

    print("\n== 全期归类 (按股票) ==")
    per_sym = missing.drop_duplicates(subset="symbol")
    print(per_sym.groupby("reason").agg(n=("symbol", "size")).to_string())

    recent = missing[missing["date"] >= panel_max - pd.Timedelta(days=90)]
    print("\n== 最近 90 天: 每只缺失股票 ==")
    r2 = recent.groupby("symbol").agg(
        first_date=("date", "min"),
        n_events=("date", "size"),
        name=("name", "first"),
        list_days=("list_days", "first"),
        reason=("reason", "first"),
    ).sort_values(["first_date"], ascending=False)
    print(r2.to_string())

    if len(unk):
        print("\n== 未解释/口径模糊明细 (附面板缺口) ==")
        cols = [c for c in
                ["symbol", "name", "date", "list_days", "reason",
                 "miss_events", "first_gap_days", "last_gap_days"]
                if c in unk.columns]
        print(unk[cols].sort_values("date", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
