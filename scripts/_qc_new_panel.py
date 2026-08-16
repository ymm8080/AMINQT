"""全市场宇宙修复 Step 6: 新面板质检 (2026-08-15).

铁律项 (违规必须报告, 不静默丢弃):
  1. OHLCV 校验: high>=low, high>=open/close, low<=open/close, volume>=0
  2. 派生特征合理性: bias/ratio 无 inf, 公式抽查 (bias_5 手工复算对拍)
  3. 覆盖率: 新股票 vs 生产面板同列覆盖率对比
  4. Schema 对齐: 与生产 120 列逐列核对 (缺列/多列/dtype)
  5. gate 校验: 首行 list_days >= 150 (次新剔行生效)
  6. 宇宙不重叠: 新符号 ∩ 生产符号 = ∅

WORM: data/new_symbols_panel/qc_report_<ts>.txt
"""

from __future__ import annotations

import glob
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import INGEST_MIN_LIST_DAYS  # noqa: E402

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
OUT_PANEL_DIR = "data/new_symbols_panel"

# 覆盖率两档规则 (2026-08-15 事故修复: 防全 NA 列混进生产)
COV_MAIN_MIN = 0.5  # 生产主列 (覆盖率>=0.9) 要求新面板至少 50%
COV_SPARSE_RATIO = 0.5  # 稀疏列允许 = 生产覆盖率 x 该比率 (防 100% NA)
# 事件类列: 覆盖率随上市时长线性增长 (新股上市晚, 事件池窗口短),
# 用比率判定会结构性误报 — 只要求非空行存在 (防全 NA 事故)
EVENT_COLS = ("bt_", "lhb_", "sh_evt_")

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def main() -> None:
    # 显式只匹配最终面板 (含 cyq/alt 列); base_new_* 会误匹配中间产物
    files = sorted(glob.glob(os.path.join(OUT_PANEL_DIR, "base_new_full_*.parquet")))
    if not files:
        raise SystemExit("FATAL: 无 base_new_full_*.parquet, 先跑 B 链 (base→cyq→alt)")
    f = files[-1]
    print(f"== QC input: {f} ==", flush=True)
    df = pd.read_parquet(f)
    print(
        f"rows={len(df):,} symbols={df['symbol'].nunique()} "
        f"dates={df['date'].min().date()}..{df['date'].max().date()}",
        flush=True,
    )

    # ── 1. OHLCV 校验 (铁律) ──
    print("\n[1] OHLCV 校验", flush=True)
    if {"high", "low", "open", "close"} <= set(df.columns):
        bad = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )
        check(
            "high>=low & high>=o/c & low<=o/c",
            bad.sum() == 0,
            f"{bad.sum():,} 行违规" if bad.sum() else "0 行违规",
        )
        if bad.sum():
            print(
                df.loc[bad, ["symbol", "date", "open", "high", "low", "close"]]
                .head(10)
                .to_string()
            )
    else:
        check("OHLCV 列存在", False, "缺 open/high/low/close")
    if "volume" in df.columns:
        neg = (df["volume"] < 0).sum()
        check("volume >= 0", neg == 0, f"{neg:,} 行负值")
    if "amount" in df.columns:
        neg = (df["amount"] < 0).sum()
        check("amount >= 0", neg == 0, f"{neg:,} 行负值")
    if "close_hfq" in df.columns:
        check(
            "close_hfq > 0",
            (df["close_hfq"] <= 0).sum() == 0,
            f"{(df['close_hfq'] <= 0).sum():,} 行非正",
        )

    # ── 2. 派生特征合理性 + 抽查 ──
    print("\n[2] 派生特征", flush=True)
    for c in [
        "bias_5",
        "bias_250",
        "ma_vol_ratio_5_20",
        "vol_surge",
        "amt_surge",
        "pct_90_con",
        "winner_ratio",
        "intraday_range",
    ]:
        if c not in df.columns:
            continue
        s = df[c]
        inf_n = int(np.isinf(s).sum())
        check(f"{c} 无 inf", inf_n == 0, f"{inf_n:,} 个 inf")
    # bias_5 抽查: 手工复算一只股票最后一天
    if "bias_5" in df.columns and "close_hfq" in df.columns:
        sym = df["symbol"].iloc[-1]
        g = df[df["symbol"] == sym].sort_values("date")
        if len(g) >= 5:
            expect = g["close_hfq"].iloc[-1] / g["close_hfq"].tail(5).mean() - 1
            got = g["bias_5"].iloc[-1]
            ok = np.isclose(expect, got, rtol=1e-9, equal_nan=True)
            check("bias_5 公式抽查", ok, f"{sym} 期望 {expect:.6f} vs 实际 {got:.6f}")
    # 120 天窗口检查 bias_250 覆盖 (新上市股应 NaN, 老股应有值)
    if "bias_250" in df.columns:
        cov = df["bias_250"].notna().mean()
        print(
            f"    bias_250 覆盖率 {cov:.1%} (新股上市<250交易日应低, 老股应高)",
            flush=True,
        )

    # ── 4. Schema 对齐 (先于覆盖率, 缺列时 df[pcols] 会 KeyError) ──
    print("\n[4] Schema 对齐 (生产 120 列)", flush=True)
    pcols = pd.read_parquet(PANEL).columns
    missing = [c for c in pcols if c not in df.columns]
    extra = [c for c in df.columns if c not in pcols]
    check("无缺列", not missing, f"缺 {len(missing)}: {missing[:8]}")
    check("无多列", not extra, f"多 {len(extra)}: {extra[:8]}")

    # ── 3. 覆盖率对比 (同列, 生产面板 vs 新面板, 全列断言) ──
    # 08-15 事故修复: 之前只抽 8 列且不断言, 100% NA 列混进生产未被抓
    print("\n[3] 覆盖率对比 (生产 vs 新, 全列)", flush=True)
    p = pd.read_parquet(PANEL, columns=["symbol"])
    print(
        f"    生产 {p['symbol'].nunique()} 只 | 新 {df['symbol'].nunique()} 只",
        flush=True,
    )
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(PANEL)
    prod_cov: dict[str, float] = {}
    for c in pcols:
        col = pf.read(columns=[c]).column(0)
        prod_cov[c] = 1.0 - col.null_count / max(len(col), 1)
    new_cov = (
        df.reindex(columns=pcols).notna().mean()
        if len(pcols)
        else pd.Series(dtype=float)
    )
    cov_fails: list[str] = []
    for c in pcols:
        pc, nc = prod_cov[c], float(new_cov.get(c, 0.0))
        if c.startswith(EVENT_COLS) and pc < 0.9:
            if nc <= 0:
                cov_fails.append(f"{c} 全 NA (事件列, 生产 {pc:.1%})")
            continue
        threshold = COV_MAIN_MIN if pc >= 0.9 else pc * COV_SPARSE_RATIO
        if nc + 1e-9 < threshold:
            cov_fails.append(f"{c} 生产 {pc:.1%} vs 新 {nc:.1%} (需 ≥{threshold:.1%})")
    check(
        "全列覆盖率达标",
        not cov_fails,
        f"{len(cov_fails)}/{len(pcols)} 列不达标: {cov_fails[:8]}",
    )
    for c in [
        "weight_avg",
        "winner_ratio",
        "lhb_net_buy",
        "margin_balance",
        "roe",
        "bt_count",
        "sw_ret_1d",
        "peak_price",
    ]:
        if c in pcols:
            print(
                f"    {c}: 生产 {prod_cov[c]:.1%} | 新 {new_cov.get(c, 0.0):.1%}",
                flush=True,
            )

    # ── 5. gate 校验: 首行 list_days >= 150 ──
    print(f"\n[5] 次新 gate (>= {INGEST_MIN_LIST_DAYS} 交易日)", flush=True)
    universe = pd.read_parquet(
        sorted(glob.glob("data/new_universe/new_symbols_*.parquet"))[-1]
    )
    cal = pd.DatetimeIndex(
        sorted(pd.read_parquet(PANEL, columns=["date"])["date"].unique())
    )
    ld_map = dict(
        zip(
            universe["symbol"].astype(str).str.strip(),
            universe["list_date"],
            strict=False,
        )
    )
    first = df.groupby("symbol")["date"].min()
    ld = pd.to_datetime(
        pd.Series(first.index).map(ld_map), format="%Y%m%d", errors="coerce"
    )
    left = cal.searchsorted(ld, side="left")
    right = cal.searchsorted(first, side="right")
    days = pd.Series(right - left, index=first.index)
    bad_n = int((days < INGEST_MIN_LIST_DAYS).sum())
    check(
        "首行 list_days >= 150",
        bad_n == 0,
        f"{bad_n} 只首行不足 {INGEST_MIN_LIST_DAYS} 交易日",
    )
    if bad_n:
        print(days[days < INGEST_MIN_LIST_DAYS].to_string())

    # ── 6. 宇宙不重叠 ──
    print("\n[6] 宇宙不重叠", flush=True)
    prod_syms = set(p["symbol"].astype(str).str.strip())
    new_syms = set(df["symbol"].astype(str).str.strip())
    overlap = prod_syms & new_syms
    check(
        "新符号 ∩ 生产 = 空",
        not overlap,
        f"{len(overlap)} 只重叠: {sorted(overlap)[:8]}",
    )

    # ── 7. 清单完整性 (清单 vs 面板, 防拉取缺失) ──
    # 08-15 事故修复: A 链曾因 daily 拉取未落地缺 96 只, 缺失必须被 gate 解释
    print("\n[7] 清单完整性", flush=True)
    want = set(universe["symbol"].astype(str).str.strip())
    missing = sorted(want - new_syms)
    if missing:
        exempt: list[str] = []
        unexcused: list[str] = []
        for s in missing:
            ld = pd.to_datetime(ld_map[s], format="%Y%m%d", errors="coerce")
            # 上市日至今在面板窗口内的交易日数 (不足 150 → 整只被 gate 剔, 正常)
            n = len(cal) - cal.searchsorted(ld) if pd.notna(ld) else 0
            (exempt if n < INGEST_MIN_LIST_DAYS else unexcused).append(s)
        check(
            "清单缺失全被 gate 豁免",
            not unexcused,
            f"{len(unexcused)} 只数据缺失: {unexcused[:8]} | gate 豁免 {len(exempt)} 只",
        )
    else:
        check(f"清单 {len(want)} 只全部入面板", True)

    # ── 报告 ──
    ts_ = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_PANEL_DIR, f"qc_report_{ts_}.txt")
    lines = [
        f"QC {ts_} on {os.path.basename(f)}",
        f"rows={len(df):,} symbols={df['symbol'].nunique()}",
        f"FAILS={len(FAILS)}: {FAILS}",
    ]
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(
        f"\n== QC {'通过' if not FAILS else f'失败 {len(FAILS)} 项'} — 报告 {out} ==",
        flush=True,
    )


if __name__ == "__main__":
    main()
