"""全市场宇宙修复 Step 5b: 修复 base_new_full 面板 3 处缺口 (2026-08-16).

08-15 首轮 alt 构建暴露 3 个问题 (QC 全列覆盖率检查拦截):
  1. industry 全 UNKNOWN — base 构建时 sw CSV 拉取网络中断 (RemoteDisconnected)
     → 重映射老申万名 + 重算 conc_90_industry_rank (依赖 industry 的分组排名)
  2. fina 缺 net_margin/eps_yoy/profit_yoy — Tushare fina_indicator 对当前 token
     不返回这些字段 (权限分层, 接口静默省略)
     → 用 income 接口计算 (口径已对拍生产面板验证):
        net_margin = n_income / total_revenue x 100
        eps_yoy    = basic_eps 同报告期同比 x 100
        profit_yoy = n_income_attr_p 同报告期同比 x 100
     合并用 merge_asof by announce_date (PIT, backward), 与生产一致
  3. sw 指数 3 列 (sw_ret_1d/sw_index_close/sw_index_vol) — industry 全 UNKNOWN
     导致行业无法映射 → 修复 industry 后按行业拉 index_daily 合并

WORM: 读已有 base_new_full_<ts>.parquet, 输出新 base_new_full_<new_ts>.parquet.
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

OUT_DIR = "data/new_symbols_raw"
OUT_PANEL_DIR = "data/new_symbols_panel"
CALL_SLEEP = 0.25
FLUSH_EVERY = 50

# 与 _build_new_base_panel.py 同口径
_SW2OLD = {
    "基础化工": "化工",
    "商贸零售": "商业贸易",
    "社会服务": "休闲服务",
    "纺织服饰": "纺织服装",
    "电力设备": "电气设备",
}
_OLD_NAMES = {
    "农林牧渔",
    "化工",
    "钢铁",
    "有色金属",
    "电子",
    "家用电器",
    "食品饮料",
    "纺织服装",
    "轻工制造",
    "医药生物",
    "公用事业",
    "交通运输",
    "房地产",
    "商业贸易",
    "休闲服务",
    "综合",
    "建筑材料",
    "建筑装饰",
    "电气设备",
    "国防军工",
    "计算机",
    "传媒",
    "通信",
    "银行",
    "非银金融",
    "汽车",
    "机械设备",
}


def _load_industry_map() -> dict[str, str]:
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
        ),
    )
    import fetch_sw_classification as fsc  # noqa: PLC0415

    cls = pd.read_csv(str(fsc.OUTPUT_PATH), encoding="utf-8-sig", dtype=str)
    cls = cls.drop_duplicates(subset="symbol", keep="first")
    cls["symbol"] = cls["symbol"].astype(str).str.strip()
    ind_map: dict[str, str] = {}
    for _, r in cls.iterrows():
        name = str(r["sw_l1_name"])
        old = _SW2OLD.get(name, name if name in _OLD_NAMES else "")
        ind_map[r["symbol"]] = old or "UNKNOWN"
    return ind_map


def pull_income_yoy(pro, symbols: list[str]) -> pd.DataFrame:
    """逐股拉 income 全历史 → 计算 net_margin/eps_yoy/profit_yoy (PIT 行).

    返回: [symbol, announce_date, net_margin, eps_yoy, profit_yoy]
    """
    parts = []
    t0 = time.time()
    for i, sym in enumerate(symbols):
        code = sym + (".SH" if sym.startswith(("6", "5")) else ".SZ")
        try:
            raw = pro.income(ts_code=code)
        except Exception as exc:  # noqa: BLE001
            print(f"    [income] {sym}: FAIL ({exc})", flush=True)
            time.sleep(2)
            continue
        if raw is None or raw.empty:
            time.sleep(CALL_SLEEP)
            continue
        raw = raw[raw["update_flag"] == "1"].copy()
        if raw.empty:
            time.sleep(CALL_SLEEP)
            continue
        raw["end_date"] = raw["end_date"].astype(str)
        raw = raw.drop_duplicates(subset="end_date", keep="last")
        raw = raw.sort_values("end_date")
        end_to_row = {r["end_date"]: r for _, r in raw.iterrows()}
        rows = []
        for _, r in raw.iterrows():
            end_date = r["end_date"]
            prev_end = f"{int(end_date[:4]) - 1}{end_date[4:]}"
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
        if rows:
            parts.append(pd.DataFrame(rows))
        time.sleep(CALL_SLEEP)
        if (i + 1) % FLUSH_EVERY == 0:
            rate = (i + 1) / (time.time() - t0) * 3600
            print(f"[income] {i + 1}/{len(symbols)} ({rate:.0f}/hr)", flush=True)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["announce_date"] = pd.to_datetime(
        out["announce_date"], format="%Y%m%d", errors="coerce"
    )
    return out.dropna(subset=["announce_date"])


def main() -> None:
    f = sorted(glob.glob(os.path.join(OUT_PANEL_DIR, "base_new_full_*.parquet")))[-1]
    df = pd.read_parquet(f)
    print(
        f"[fix] input={os.path.basename(f)} rows={len(df):,} syms={df['symbol'].nunique()}",
        flush=True,
    )
    pro = ts.pro_api(settings.TUSHARE_TOKEN)

    # ── 1. industry 重映射 + conc_90_industry_rank 重算 ──
    ind_map = _load_industry_map()
    before = (df["industry"] == "UNKNOWN").sum()
    df["industry"] = df["symbol"].map(ind_map).fillna("UNKNOWN")
    after = (df["industry"] == "UNKNOWN").sum()
    print(
        f"[industry] UNKNOWN {before:,} → {after:,} ({len(ind_map)} syms mapped)",
        flush=True,
    )
    df["conc_90_industry_rank"] = (
        df.groupby(["date", "industry"], observed=True)["pct_90_con"]
        .rank(pct=True)
        .fillna(0.5)
    )
    print("[rank] conc_90_industry_rank 重算完成", flush=True)

    # ── 2. fina 3 列 (income 计算, PIT) ──
    symbols = sorted(df["symbol"].astype(str).unique())
    fina = pull_income_yoy(pro, symbols)
    if len(fina):
        df = df.sort_values("date").reset_index(drop=True)
        df = pd.merge_asof(
            df,
            fina[
                ["symbol", "announce_date", "net_margin", "eps_yoy", "profit_yoy"]
            ].sort_values("announce_date"),
            left_on="date",
            right_on="announce_date",
            by="symbol",
            direction="backward",
        )
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        # 与生产同: announce_date 列保留 (merge_asof 引入)
        print(
            f"[fina] merged {len(fina):,} rows, "
            f"net_margin 覆盖 {df['net_margin'].notna().mean():.1%}",
            flush=True,
        )
    else:
        print("[fina] FATAL: income 拉取全失败", flush=True)
        raise SystemExit(1)

    # ── 3. sw 指数 3 列 (industry 修复后映射) ──
    _ind2code = {v: k for k, v in SW_INDEX_CODES.items()}
    _ind2code["电气设备"] = "801730"  # 老申万名别名 (同 _daily_fetch)
    industries = sorted(set(df["industry"].unique()) - {"UNKNOWN"})
    ind_map2: dict[str, str] = {}
    for ind in industries:
        if ind in _ind2code:
            ind_map2[ind] = _ind2code[ind]
        else:
            for sw_name, code in _ind2code.items():
                if ind in sw_name or sw_name in ind:
                    ind_map2[ind] = code
                    break
    print(f"[sw idx] industries={len(industries)}, mapped={len(ind_map2)}", flush=True)
    idx_parts = []
    for code in sorted(set(ind_map2.values())):
        try:
            idx = pro.index_daily(
                ts_code=f"{code}.SI",
                start_date=df["date"].min().strftime("%Y%m%d"),
                end_date=df["date"].max().strftime("%Y%m%d"),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[sw idx] {code}: FAILED ({exc})", flush=True)
            continue
        if not len(idx):
            continue
        idx["date"] = pd.to_datetime(idx["trade_date"], format="%Y%m%d")
        idx = idx.rename(columns={"pct_chg": "_pct", "close": "_close", "vol": "_vol"})
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
        code2ind = {c: i for i, c in ind_map2.items()}
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

    # ── 4. 保存 (WORM) ──
    ts_ = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_PANEL_DIR, f"base_new_full_{ts_}.parquet")
    df.to_parquet(out, index=False)
    print(f"[save] {out}", flush=True)
    cov = (
        df[["industry", "net_margin", "eps_yoy", "profit_yoy", "sw_ret_1d"]]
        .notna()
        .mean()
        .round(3)
    )
    print("[coverage]")
    print(cov.to_string(), flush=True)
    print("ALT FIX DONE", flush=True)


if __name__ == "__main__":
    main()
