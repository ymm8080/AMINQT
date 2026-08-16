"""全市场宇宙修复 Step 5c: 补拉 full_fixed 缺口 (2026-08-16).

_build_new_alt 修复 (014452) 后仍缺:
  1. 5 个 sw 指数 (801010 农林牧渔/801200 商业贸易/801210 休闲服务/
     801740 国防军工/801790 非银金融) — 拉取超时 FAILED, 该行业 3 列全 NA
  2. 57 只 income 超时 FAIL → net_margin/eps_yoy/profit_yoy 缺失
  3. merge_asof 列冲突 announce_date_x/_y → 规整为 announce_date (用 _x,
     与生产同口径 fina_indicator ann_date; _y 为 income ann_date, 99% 相等)
  4. drop 5 中间列 (vol/adj_factor/turnover_rate_f/winner_rate/avg_cost)

只更新失败行业行 / 失败股票行, 不动已成功数据.
WORM: data/new_symbols_panel/base_new_full_<ts>.parquet (新 ts 输出).
"""

from __future__ import annotations

import glob
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tushare as ts  # noqa: E402

from app.pipeline1.data_supply import SW_INDEX_CODES  # noqa: E402
from config import settings  # noqa: E402

OUT_PANEL_DIR = "data/new_symbols_panel"
MID_COLS = {"vol", "adj_factor", "turnover_rate_f", "winner_rate", "avg_cost"}
FAIL_SW = {"801010", "801200", "801210", "801740", "801790"}
CALL_SLEEP = 0.5
RETRY = 3
RETRY_SLEEP = 5.0


def _pull_index(pro: ts.pro_api, code: str, start: str, end: str) -> pd.DataFrame | None:
    for _i in range(RETRY):
        try:
            idx = pro.index_daily(ts_code=f"{code}.SI", start_date=start, end_date=end)
            if idx is not None and len(idx):
                return idx
        except Exception:  # noqa: BLE001
            pass
        time.sleep(RETRY_SLEEP)
    return None


def pull_income_one(pro: ts.pro_api, sym: str) -> pd.DataFrame | None:
    """单只 income → PIT 行 [symbol, announce_date, net_margin, eps_yoy, profit_yoy].

    与 _fix_new_panel_alt.pull_income_yoy 同口径 (update_flag=1, end_date 去重,
    同报告期同比, 全历史).
    """
    code = sym + (".SH" if sym.startswith(("6", "5")) else ".SZ")
    for _i in range(RETRY):
        try:
            raw = pro.income(ts_code=code)
            break
        except Exception:  # noqa: BLE001
            raw = None
            time.sleep(RETRY_SLEEP)
    if raw is None or raw.empty:
        return None
    raw = raw[raw["update_flag"] == "1"].copy()
    if raw.empty:
        return None
    raw["end_date"] = raw["end_date"].astype(str)
    raw = raw.drop_duplicates(subset="end_date", keep="last").sort_values("end_date")
    end_to_row = {r["end_date"]: r for _, r in raw.iterrows()}
    rows = []
    for _, r in raw.iterrows():
        prev_end = f"{int(r['end_date'][:4]) - 1}{r['end_date'][4:]}"
        prev = end_to_row.get(prev_end)
        nm = (
            r["n_income"] / r["total_revenue"] * 100.0
            if r["total_revenue"] and r["total_revenue"] != 0
            else np.nan
        )
        eps_yoy = (
            (r["basic_eps"] / prev["basic_eps"] - 1.0) * 100.0
            if prev is not None and prev["basic_eps"] not in (None, 0)
            else np.nan
        )
        py_yoy = (
            (r["n_income_attr_p"] / prev["n_income_attr_p"] - 1.0) * 100.0
            if prev is not None and prev["n_income_attr_p"] not in (None, 0)
            else np.nan
        )
        rows.append(
            {
                "symbol": sym,
                "announce_date": r["ann_date"],
                "net_margin": nm,
                "eps_yoy": eps_yoy,
                "profit_yoy": py_yoy,
            }
        )
    if not rows:
        return None
    out = pd.DataFrame(rows)
    out["announce_date"] = pd.to_datetime(
        out["announce_date"], format="%Y%m%d", errors="coerce"
    )
    return out.dropna(subset=["announce_date"])


def main() -> None:
    f = sorted(glob.glob(os.path.join(OUT_PANEL_DIR, "base_new_full_*.parquet")))[-1]
    df = pd.read_parquet(f)
    print(
        f"[patch] input={os.path.basename(f)} rows={len(df):,} syms={df['symbol'].nunique()}",
        flush=True,
    )
    pro = ts.pro_api(settings.TUSHARE_TOKEN)

    # ── 1. 补 5 个 sw 指数 (只更新失败行业行) ──
    code2ind = {c: SW_INDEX_CODES[c] for c in FAIL_SW}
    fail_inds = set(code2ind.values())
    mask_sw = df["industry"].isin(fail_inds)
    print(
        f"[sw] 补拉 {len(FAIL_SW)} 指数, 受影响行 {int(mask_sw.sum()):,} "
        f"({mask_sw.mean():.1%})",
        flush=True,
    )
    idx_parts = []
    for code in sorted(FAIL_SW):
        idx = _pull_index(
            pro, code, df["date"].min().strftime("%Y%m%d"), df["date"].max().strftime("%Y%m%d")
        )
        if idx is None or not len(idx):
            print(f"[sw] {code} {code2ind[code]}: 仍 FAILED", flush=True)
            continue
        idx["date"] = pd.to_datetime(idx["trade_date"], format="%Y%m%d")
        idx = idx.rename(columns={"pct_chg": "_pct", "close": "_close", "vol": "_vol"})
        idx["sw_ret_1d"] = idx["_pct"] / 100.0
        idx["sw_index_close"] = idx["_close"]
        idx["sw_index_vol"] = idx["_vol"] / 1e6
        idx["industry"] = code2ind[code]
        idx_parts.append(idx[["date", "industry", "sw_ret_1d", "sw_index_close", "sw_index_vol"]])
        print(f"[sw] {code} {code2ind[code]}: {len(idx):,} 行", flush=True)
        time.sleep(CALL_SLEEP)
    if idx_parts:
        idx_all = pd.concat(idx_parts, ignore_index=True)
        sub = df.loc[mask_sw].copy()
        merged = sub.drop(columns=["sw_ret_1d", "sw_index_close", "sw_index_vol"]).merge(
            idx_all, on=["industry", "date"], how="left"
        )
        # merge 输出 RangeIndex, loc 赋值按索引对齐会错位 — 恢复 sub 索引
        merged = merged.set_axis(sub.index)
        df.loc[mask_sw] = merged
        print(
            f"[sw] 更新后失败行业 sw_ret_1d 覆盖 "
            f"{df.loc[mask_sw, 'sw_ret_1d'].notna().mean():.1%}",
            flush=True,
        )

    # ── 2. 补 57 只 income (只更新缺失股票行) ──
    fina_cols = ["net_margin", "eps_yoy", "profit_yoy"]
    miss = df["symbol"].isin(
        df.loc[df[fina_cols].isna().all(axis=1), "symbol"].unique()
    )
    symbols = sorted(df.loc[miss, "symbol"].unique())
    print(f"[fina] 补拉 {len(symbols)} 只 income", flush=True)
    parts = []
    for i, sym in enumerate(symbols):
        out = pull_income_one(pro, sym)
        if out is not None:
            parts.append(out)
        if (i + 1) % 10 == 0:
            print(f"[fina] {i + 1}/{len(symbols)}", flush=True)
        time.sleep(CALL_SLEEP)
    if parts:
        fina = pd.concat(parts, ignore_index=True)
        sub = df.loc[miss].copy().sort_values("date")
        merged = pd.merge_asof(
            sub,
            fina[["symbol", "announce_date"] + fina_cols].sort_values("announce_date"),
            left_on="date",
            right_on="announce_date",
            by="symbol",
            direction="backward",
        )
        merged = merged.drop(columns=["announce_date"])
        # merge_asof 输出 RangeIndex, loc 赋值按索引对齐会错位 — 恢复 sub 索引
        merged = merged.set_axis(sub.index)
        df.loc[miss] = merged[df.columns]
        print(f"[fina] 更新 {int(miss.sum()):,} 行, 覆盖 {df[fina_cols].notna().mean().round(3).to_dict()}", flush=True)

    # ── 3. 列规整: announce_date_x → announce_date; drop _y + 中间列 ──
    if "announce_date_x" in df.columns:
        df["announce_date"] = df["announce_date_x"]
        df = df.drop(columns=["announce_date_x", "announce_date_y"])
    df = df.drop(columns=[c for c in MID_COLS if c in df.columns])

    # ── 4. 保存 (WORM) ──
    ts_ = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_PANEL_DIR, f"base_new_full_{ts_}.parquet")
    df.to_parquet(out, index=False)
    print(f"[save] {out}", flush=True)
    print(f"[stat] rows={len(df):,} cols={len(df.columns)}", flush=True)
    cov = df[
        ["industry", "net_margin", "eps_yoy", "profit_yoy", "sw_ret_1d", "announce_date"]
    ].notna().mean().round(3)
    print("[coverage]")
    print(cov.to_string(), flush=True)
    print("PATCH DONE", flush=True)


if __name__ == "__main__":
    main()
