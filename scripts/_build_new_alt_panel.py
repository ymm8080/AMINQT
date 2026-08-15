"""全市场宇宙修复 Step 5: 新股票另类数据合并 (2026-08-15).

在 base_new_cyq 面板上合并剩余 alt 列 (语义逐条对齐生产):
  - fina 22 列: merge_asof by announce_date (PIT, backward) + announce_date 列
  - holdertrade 10 列: fetch_holdertrade(全窗口, 恢复被删缓存) → agg_holdertrade_daily
    → 4 GLM 列 ffill, evt/ratio 6 列稀疏
  - margin 4 列: 面板口径, groupby ffill (生产 ffill_cols 语义)
  - bt 4 列: 恢复的 block_trade_full 缓存逐日聚合 (规则复刻 _daily_fetch §6.5)
  - lhb 席位 8 列: top_inst 逐日 seat_wide_from_top_inst (生产权威函数)
  - sw_l1/2/3_name: fetch_sw_classification.incremental_update 增量补 CSV 后映射
  - sw_ret_1d/sw_index_close/sw_index_vol: 按行业 Tushare index_daily 全窗一次拉齐
    (pct_chg/100, close, vol/1e6 — 单位与 _daily_fetch §SW 一致)

WORM: data/new_symbols_panel/base_new_full_<ts>.parquet
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

from app.pipeline1.bt_snapshot import BT_COLS  # noqa: E402
from app.pipeline1.data_supply import (  # noqa: E402
    _FINA_COL_RENAME,
    SW_INDEX_CODES,
    DataSupplyChain,
)
from app.pipeline1.holdertrade_agg import agg_holdertrade_daily  # noqa: E402
from app.pipeline1.lhb_seats import seat_wide_from_top_inst  # noqa: E402
from config import settings  # noqa: E402

OUT_DIR = "data/new_symbols_raw"
ALT_DIR = "data/supply_cache/alt_data"
OUT_PANEL_DIR = "data/new_symbols_panel"
_INST_KW = (
    "机构专用",
    "QFII",
    "合格境外",
    "社保",
    "养老",
    "资产管理",
    "资管",
    "保险",
    "信托",
)


def _broker(seat) -> str:
    s = str(seat)
    idx = s.find("证券")
    return s[idx : idx + 2] if idx >= 0 else ""


def _bt_agg_day(bt_day: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """复刻 _daily_fetch §6.5 的当日大宗聚合 (输入单位: vol=万股, amount=万元)."""
    _bt = bt_day.merge(
        snap[["symbol", "close", "daily_amt", "circ_mv"]], on=["symbol"], how="left"
    )
    _bt = _bt[_bt["close"].notna()].copy()
    if not len(_bt):
        return pd.DataFrame(columns=BT_COLS)
    same_broker = (_bt["buyer"].map(_broker) == _bt["seller"].map(_broker)) & (
        _bt["buyer"] != _bt["seller"]
    )
    _bt["discount"] = (_bt["price"] - _bt["close"]) / _bt["close"].replace(0, np.nan)
    _bt["is_noise"] = (
        (_bt["buyer"] == _bt["seller"])
        | (same_broker & (_bt["discount"] < -0.1))
        | ((_bt["vol"] < 10) & (_bt["discount"].abs() < 0.01))
    )
    _bt["is_inst_buyer"] = _bt["buyer"].map(
        lambda s: any(k in str(s) for k in _INST_KW)
    )
    v = _bt[~_bt["is_noise"]]
    if not len(v):
        return pd.DataFrame(columns=BT_COLS)
    grp = v.groupby("symbol")
    total_amt = grp["amount"].sum()
    wavg = (v["price"] * v["vol"]).groupby(v["symbol"]).sum() / grp[
        "vol"
    ].sum().replace(0, np.nan)
    close = grp["close"].first()
    daily_amt = grp["daily_amt"].first()
    circ_mv = grp["circ_mv"].first()
    disc = (wavg - close) / close.replace(0, np.nan)
    any_inst = v.groupby("symbol")["is_inst_buyer"].max()
    return pd.DataFrame(
        {
            "bt_count": grp.size(),
            "bt_disc_raw": (-disc).clip(lower=0),
            "bt_inst_absorb": any_inst * total_amt / daily_amt.replace(0, np.nan),
            "bt_amt_ratio_float_mv": total_amt / circ_mv.replace(0, np.nan),
        }
    ).reset_index()


def main() -> None:
    f = sorted(glob.glob(os.path.join(OUT_PANEL_DIR, "base_new_cyq_*.parquet")))[-1]
    df = pd.read_parquet(f)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    newsyms = set(df["symbol"])
    d0, d1 = df["date"].min().strftime("%Y%m%d"), df["date"].max().strftime("%Y%m%d")
    print(
        f"[alt] input={f} rows={len(df):,} syms={df['symbol'].nunique()} ({d0}..{d1})",
        flush=True,
    )
    pro = ts.pro_api(settings.TUSHARE_TOKEN)

    # ── 1. fina (merge_asof PIT) ──
    fina_files = sorted(glob.glob(os.path.join(OUT_DIR, "fina", "fina_*.parquet")))
    if fina_files:
        fina = pd.concat([pd.read_parquet(x) for x in fina_files], ignore_index=True)
        fina = fina.rename(columns=_FINA_COL_RENAME)
        if "symbol" in fina.columns:
            fina["symbol"] = fina["symbol"].astype(str).str.strip()
        fina = fina[fina["symbol"].isin(newsyms)].copy()
        fina["announce_date"] = pd.to_datetime(
            fina["announce_date"], format="%Y%m%d", errors="coerce"
        )
        fina = fina.dropna(subset=["announce_date"]).sort_values("announce_date")
        fin_cols = [
            c
            for c in fina.columns
            if c
            not in (
                "symbol",
                "announce_date",
                "report_period",
                "_ts_code",
                "ts_code",
                "end_date",
            )
        ]
        if len(fina) and fin_cols:
            df = pd.merge_asof(
                df,
                fina[["symbol", "announce_date"] + fin_cols],
                left_on="date",
                right_on="announce_date",
                by="symbol",
                direction="backward",
            )
            print(f"[fina] merged {len(fina):,} rows, {len(fin_cols)} cols", flush=True)
            # 恢复被删的 fina 缓存 (build_full_panel 回退路径读它)
            cache_d = os.path.join(ALT_DIR, "fina_indicator")
            os.makedirs(cache_d, exist_ok=True)
            ts_ = datetime.now().strftime("%Y%m%d_%H%M%S")
            fina.to_parquet(
                os.path.join(cache_d, f"new_symbols_{ts_}.parquet"), index=False
            )
    else:
        print("[fina] WARN: 无 fina 输入 (拉取未完成?)", flush=True)

    # ── 2. holdertrade (fetch 全窗口 → agg → merge → 4 GLM ffill) ──
    try:
        ht = DataSupplyChain().fetch_holdertrade(
            start_date=d0, end_date=d1, refresh=True
        )
        print(f"[holdertrade] raw={len(ht):,} (缓存已重建)", flush=True)
        if len(ht):
            agg = agg_holdertrade_daily(ht)
            agg = agg[agg["symbol"].isin(newsyms)].copy()
            agg["date"] = pd.to_datetime(agg["date"])
            df = df.merge(agg, on=["symbol", "date"], how="left")
            for c in [
                "sh_change_vol",
                "sh_change_amt_total",
                "sh_net_change_sign",
                "sh_net_sign",
            ]:
                if c in df.columns:
                    df[c] = df.groupby("symbol")[c].ffill()
            print(f"[holdertrade] merged {len(agg):,} event rows", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[holdertrade] FAILED: {exc}", flush=True)

    # ── 3. margin ffill (面板 ffill_cols 语义) ──
    for c in ["margin_balance", "short_balance", "margin_buy_amt", "short_sell_vol"]:
        if c in df.columns:
            df[c] = df.groupby("symbol")[c].ffill()

    # ── 4. bt 4 列 (恢复缓存逐日聚合) ──
    bt_files = sorted(
        glob.glob(os.path.join(ALT_DIR, "block_trade", "block_trade_full_*.parquet"))
    )
    if bt_files:
        bt = pd.concat([pd.read_parquet(x) for x in bt_files], ignore_index=True)
        bt = bt[bt["symbol"].isin(newsyms)].copy()
        bt["trade_date"] = bt["trade_date"].astype(str)
        bt = bt.drop_duplicates(
            subset=["symbol", "trade_date", "buyer", "seller", "price", "vol", "amount"]
        )
        print(f"[bt] raw={len(bt):,}", flush=True)
        snap = df[["symbol", "date", "close", "amount", "circ_mv"]].copy()
        snap["daily_amt"] = snap["amount"] / 10.0  # 生产公式照抄 (面板 amount=元)
        bt_dates = sorted(bt["trade_date"].unique())
        parts = []
        for ds in bt_dates:
            day = bt[bt["trade_date"] == ds]
            s = snap[snap["date"] == pd.Timestamp(ds)]
            if not len(s):
                continue
            agg = _bt_agg_day(day, s)
            if len(agg):
                agg["date"] = pd.Timestamp(ds)
                parts.append(agg)
        if parts:
            bt_agg = pd.concat(parts, ignore_index=True)
            df = df.merge(bt_agg, on=["symbol", "date"], how="left")
            print(
                f"[bt] merged {len(bt_agg):,} event rows ({len(bt_dates)} dates)",
                flush=True,
            )
    else:
        print("[bt] WARN: 无 block_trade 恢复缓存", flush=True)

    # ── 5. lhb 席位 8 列 ──
    ti_files = sorted(
        glob.glob(os.path.join(OUT_DIR, "top_inst", "top_inst_*.parquet"))
    )
    if ti_files:
        ti = pd.concat([pd.read_parquet(x) for x in ti_files], ignore_index=True)
        if "ts_code" not in ti.columns:
            print(
                "[lhb seats] WARN: top_inst 缺 ts_code 列 (旧批次), 跳过该源",
                flush=True,
            )
        else:
            ti = ti[ti["symbol"].isin(newsyms)].copy()
            ti["trade_date"] = ti["trade_date"].astype(str)
            print(f"[lhb seats] raw={len(ti):,}", flush=True)
            parts = []
            for ds, day in ti.groupby("trade_date"):
                wide = seat_wide_from_top_inst(day)
                if len(wide):
                    wide["date"] = pd.Timestamp(ds)
                    parts.append(wide)
            if parts:
                seats = pd.concat(parts, ignore_index=True)
                df = df.merge(seats, on=["symbol", "date"], how="left")
                print(
                    f"[lhb seats] merged {len(seats):,} rows, "
                    f"{seats['date'].nunique()} dates",
                    flush=True,
                )
    else:
        print("[lhb seats] WARN: 无 top_inst 输入", flush=True)

    # ── 6. sw 分类 (增量补 CSV → 映射 3 列) ──
    try:
        sys.path.insert(
            0,
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
            ),
        )
        import fetch_sw_classification as fsc  # noqa: E402

        universe = pd.read_parquet(
            sorted(glob.glob("data/new_universe/new_symbols_*.parquet"))[-1]
        )
        tscodes = universe["ts_code"].dropna().astype(str).tolist()
        n_added = fsc.incremental_update(tscodes)
        print(f"[sw cls] incremental: +{n_added} rows", flush=True)
        cls = pd.read_csv(str(fsc.OUTPUT_PATH), encoding="utf-8-sig", dtype=str)
        cls = cls.drop_duplicates(subset="symbol", keep="first")
        cls["symbol"] = cls["symbol"].astype(str).str.strip()
        m = cls.set_index("symbol")[["sw_l1_name", "sw_l2_name", "sw_l3_name"]]
        for c in ["sw_l1_name", "sw_l2_name", "sw_l3_name"]:
            df[c] = df["symbol"].map(m[c])
        print(
            f"[sw cls] mapped: {df['sw_l1_name'].notna().mean():.1%} l1 coverage",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[sw cls] FAILED: {exc}", flush=True)

    # ── 7. sw 指数 3 列 (按行业一次拉全窗) ──
    try:
        _ind2code = {v: k for k, v in SW_INDEX_CODES.items()}
        _ind2code["电气设备"] = "801730"  # 老申万名别名 (同 _daily_fetch)
        industries = sorted(set(df["industry"].unique()) - {"UNKNOWN"})
        ind_map = {}
        for ind in industries:
            if ind in _ind2code:
                ind_map[ind] = _ind2code[ind]
            else:  # 模糊匹配 (build_full_panel 同策略)
                for sw_name, code in _ind2code.items():
                    if ind in sw_name or sw_name in ind:
                        ind_map[ind] = code
                        break
        print(
            f"[sw idx] industries={len(industries)}, mapped={len(ind_map)}", flush=True
        )
        idx_parts = []
        for code in sorted(set(ind_map.values())):
            try:
                idx = pro.index_daily(ts_code=f"{code}.SI", start_date=d0, end_date=d1)
            except Exception as exc:  # noqa: BLE001
                print(f"[sw idx] {code}: FAILED ({exc})", flush=True)
                continue
            if not len(idx):
                continue
            idx["date"] = pd.to_datetime(idx["trade_date"], format="%Y%m%d")
            idx = idx.rename(
                columns={
                    "pct_chg": "_pct",
                    "close": "_close",
                    "vol": "_vol",
                }
            )
            idx["sw_ret_1d"] = idx["_pct"] / 100.0
            idx["sw_index_close"] = idx["_close"]
            idx["sw_index_vol"] = idx["_vol"] / 1e6
            idx["_sw_code"] = code
            idx_parts.append(
                idx[["date", "sw_ret_1d", "sw_index_close", "sw_index_vol", "_sw_code"]]
            )
            time.sleep(0.12)
        if idx_parts:
            idx_all = pd.concat(idx_parts, ignore_index=True)
            code2ind = {c: i for i, c in ind_map.items()}
            idx_all["industry"] = idx_all["_sw_code"].map(code2ind)
            df = df.merge(
                idx_all.drop(columns=["_sw_code"]),
                on=["industry", "date"],
                how="left",
            )
            print(
                f"[sw idx] merged: sw_ret_1d NaN={df['sw_ret_1d'].isna().mean():.1%}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[sw idx] FAILED: {exc}", flush=True)

    # ── 8. 保存 ──
    ts_ = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_PANEL_DIR, f"base_new_full_{ts_}.parquet")
    df.to_parquet(out, index=False)
    print(f"[save] {out}", flush=True)
    cols = [
        "roe",
        "sh_net_change_sign",
        "margin_balance",
        "bt_count",
        "lhb_inst_buy",
        "sw_l1_name",
        "sw_ret_1d",
    ]
    cov = df[[c for c in cols if c in df.columns]].notna().mean().round(3)
    print("[coverage]")
    print(cov.to_string(), flush=True)
    print("ALT BUILD DONE", flush=True)


if __name__ == "__main__":
    main()
