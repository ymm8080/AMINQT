"""全市场宇宙修复 Step 7: 新面板正确合并进生产 (2026-08-15).

前置: _qc_new_panel.py 全过 (120 列 schema 对齐/OHLCV/gate/不重叠).

流程:
  1. 读生产面板 + base_new_full_<ts>.parquet
  2. schema 强校验 (缺列/多列 = 0, 否则拒绝合并)
  3. concat + sort (symbol, date) → 写临时文件
  4. 生产路径原子替换 (先 rename 旧文件为 .pre_merge_<ts> 备份)
  5. 终验: symbols/rows/无新 NA 列/新股票列覆盖率抽查

WORM: 旧生产面板保留为 .pre_merge_<ts> 备份, 不覆盖删除.
"""

from __future__ import annotations

import glob
import os
import sys
from datetime import datetime

import pandas as pd

PANEL = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
OUT_PANEL_DIR = "data/new_symbols_panel"

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def main() -> None:
    files = sorted(glob.glob(os.path.join(OUT_PANEL_DIR, "base_new_full_*.parquet")))
    if not files:
        raise SystemExit("FATAL: 无 base_new_full_*.parquet, 先跑 B 链 + QC")
    new_f = files[-1]
    ts_ = datetime.now().strftime("%Y%m%d_%H%M%S")

    prod = pd.read_parquet(PANEL)
    new = pd.read_parquet(new_f)
    print(
        f"[merge] 生产 {len(prod):,} rows / {prod['symbol'].nunique()} syms | "
        f"新 {len(new):,} rows / {new['symbol'].nunique()} syms ({new_f})",
        flush=True,
    )

    # ── 1. schema 强校验 ──
    pcols = list(prod.columns)
    missing = [c for c in pcols if c not in new.columns]
    extra = [c for c in new.columns if c not in pcols]
    check("无缺列", not missing, f"缺 {len(missing)}: {missing[:8]}")
    check("无多列", not extra, f"多 {len(extra)}: {extra[:8]}")
    if missing or extra:
        raise SystemExit("FATAL: schema 不对齐, 拒绝合并")

    # ── 2. 宇宙不重叠 ──
    overlap = set(prod["symbol"].astype(str)) & set(new["symbol"].astype(str))
    check("新符号 ∩ 生产 = ∅", not overlap, f"{len(overlap)} 只: {sorted(overlap)[:8]}")

    # ── 3. concat + 排序 ──
    merged = pd.concat([prod, new], ignore_index=True)
    merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)
    print(
        f"[merge] concat → {len(merged):,} rows / {merged['symbol'].nunique()} syms",
        flush=True,
    )

    # ── 4. 原子替换 (旧面板备份不删) ──
    bak = f"{PANEL}.pre_merge_{ts_}"
    os.rename(PANEL, bak)
    print(f"[merge] 旧面板备份: {bak}", flush=True)
    tmp = f"{PANEL}.tmp_{ts_}"
    merged.to_parquet(tmp, index=False)
    os.rename(tmp, PANEL)
    print(f"[merge] 新面板写入: {PANEL}", flush=True)

    # ── 5. 终验 ──
    v = pd.read_parquet(PANEL)
    check(
        "symbols = 生产+新",
        v["symbol"].nunique() == prod["symbol"].nunique() + new["symbol"].nunique(),
        f"{v['symbol'].nunique()} vs {prod['symbol'].nunique()}+{new['symbol'].nunique()}",
    )
    check(
        "rows = 生产+新",
        len(v) == len(prod) + len(new),
        f"{len(v):,} vs {len(prod):,}+{len(new):,}",
    )
    # 新股票列覆盖率对比 (全列, 两档规则同 QC; 08-15 事故修复: 之前只抽 8 列)
    new_syms = set(new["symbol"].astype(str))
    sub = v[v["symbol"].astype(str).isin(new_syms)]
    prod_cov = prod.notna().mean()
    new_cov = sub.notna().mean()
    cov_fails: list[str] = []
    for c in v.columns:
        pc, nc = float(prod_cov.get(c, 0.0)), float(new_cov.get(c, 0.0))
        threshold = 0.5 if pc >= 0.9 else pc * 0.5
        if nc + 1e-9 < threshold:
            cov_fails.append(f"{c} 生产 {pc:.1%} vs 新 {nc:.1%} (需 ≥{threshold:.1%})")
    check(
        "新股票全列覆盖率达标",
        not cov_fails,
        f"{len(cov_fails)}/{len(v.columns)} 列不达标: {cov_fails[:8]}",
    )
    for c in [
        "volume",
        "board",
        "industry",
        "pctChg",
        "weight_avg",
        "winner_ratio",
        "sw_l1_name",
        "sw_ret_1d",
    ]:
        if c in v.columns:
            print(f"    新股票 {c}: 覆盖率 {new_cov.get(c, 0.0):.1%} (生产 {prod_cov.get(c, 0.0):.1%})", flush=True)
    # 全局无新增全 NA 列 (全面板)
    all_na = [c for c in v.columns if v[c].notna().sum() == 0]
    check("无全局全 NA 列", not all_na, f"{all_na[:8]}")

    print(f"\n== MERGE {'通过' if not FAILS else f'失败 {len(FAILS)} 项'} ==", flush=True)
    if FAILS:
        raise SystemExit("FATAL: 合并终验失败")
    print("MERGE DONE", flush=True)


if __name__ == "__main__":
    main()
