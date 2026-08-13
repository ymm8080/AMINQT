"""_repair_holder_evt_tail.py — 修复 08-03+ 面板 evt/ratio 列污染 (一次性).

根因: _daily_fetch.py 曾把 sh_evt_start_date/end_date + sh_net/p/g/c_ratio 纳入
ffill_cols (commit 40a424df, 2026-08-03), 导致 08-03 起每个交易日把每只股票
"最后一个事件窗口"盖到今天行 → 伪造公告日 (sh_ann_decay 尖峰到 1.0) + 伪造
执行期窗口 (sh_is_executing)。ffill 列表已移除这些列 (见 _daily_fetch.py).

本脚本: 把 date >= 2026-08-03 的这 6 列重置为 NaN, 再从真实 holdertrade 缓存
(bulk + 增量) 与 KIMI 缓存 (_holder_cmp_raw) 重放真实公告行 (announce >= 08-03)。

默认 dry-run (只报影响面, 不写盘); --apply 才先备份再写盘。写后校验行数/列数/
OHLCV 违例与写前一致 (不静默丢数据)。
"""

import argparse
import datetime as _dt
import glob
import os
import shutil

import numpy as np
import pandas as pd

PANEL_PATH = r"D:\AMINQT\PARQUET\panel_full_enriched_v3.parquet"
HOLDERTRADE_GLOB = os.path.join(
    "data", "supply_cache", "alt_data", "holdertrade", "all_*.parquet"
)
KIMI_RAW = os.path.join("data", "_holder_cmp_raw.parquet")

CUTOFF = pd.Timestamp("2026-08-03")
EVT_COLS = ["sh_evt_start_date", "sh_evt_end_date"]
RATIO_COLS = ["sh_net_ratio", "sh_g_ratio", "sh_p_ratio", "sh_c_ratio"]
RESET_COLS = EVT_COLS + RATIO_COLS

OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def _ohlcv_violations(panel: pd.DataFrame) -> int:
    """OHLCV 校验 (量化铁律): high>=low, high>=open/close, low<=open/close, volume>=0."""
    v = 0
    o, h, lo, c = (panel[k] for k in ["open", "high", "low", "close"])
    v += int((h < lo).sum())
    v += int((h < o).sum() | (h < c).sum())
    v += int((lo > o).sum() | (lo > c).sum())
    if "volume" in panel.columns:
        v += int((panel["volume"] < 0).sum())
    return v


def load_evt_agg() -> pd.DataFrame:
    """真实公告行 → evt 聚合 (holdertrade 缓存 + KIMI 缓存, announce>=08-03)."""
    parts = []
    for p in sorted(glob.glob(HOLDERTRADE_GLOB)):
        df = pd.read_parquet(
            p, columns=["symbol", "announce_date", "evt_start_date", "evt_end_date"]
        )
        parts.append(
            df.rename(columns={"announce_date": "date"})[
                ["symbol", "date", "evt_start_date", "evt_end_date"]
            ]
        )
    if os.path.exists(KIMI_RAW):
        k = pd.read_parquet(
            KIMI_RAW, columns=["symbol", "date", "evt_start_date", "evt_end_date"]
        )
        parts.append(k)
    if not parts:
        return pd.DataFrame(columns=["symbol", "date"] + EVT_COLS)
    raw = pd.concat(parts, ignore_index=True)
    for c in ["date", "evt_start_date", "evt_end_date"]:
        raw[c] = pd.to_datetime(raw[c], errors="coerce")
    raw = raw[raw["date"] >= CUTOFF]
    agg = (
        raw.groupby(["symbol", "date"], as_index=False)
        .agg(
            sh_evt_start_date=("evt_start_date", "min"),
            sh_evt_end_date=("evt_end_date", "max"),
        )
        .dropna(subset=["sh_evt_start_date", "sh_evt_end_date"])
    )
    return agg


def load_ratio_agg() -> pd.DataFrame:
    """真实公告行 → ratio 聚合 (KIMI 缓存, date>=08-03)."""
    if not os.path.exists(KIMI_RAW):
        return pd.DataFrame(columns=["symbol", "date"] + RATIO_COLS)
    k = pd.read_parquet(
        KIMI_RAW, columns=["symbol", "date", "signed_ratio", "holder_type"]
    )
    k["date"] = pd.to_datetime(k["date"], errors="coerce")
    k["signed_ratio"] = pd.to_numeric(k["signed_ratio"], errors="coerce").fillna(0.0)
    k = k[k["date"] >= CUTOFF]
    ht = k["holder_type"].fillna("").astype(str).str.upper()
    k["sr_g"] = np.where(ht == "G", k["signed_ratio"], 0.0)
    k["sr_p"] = np.where(ht == "P", k["signed_ratio"], 0.0)
    k["sr_c"] = np.where(ht == "C", k["signed_ratio"], 0.0)
    return k.groupby(["symbol", "date"], as_index=False).agg(
        sh_net_ratio=("signed_ratio", "sum"),
        sh_g_ratio=("sr_g", "sum"),
        sh_p_ratio=("sr_p", "sum"),
        sh_c_ratio=("sr_c", "sum"),
    )


def repair(panel: pd.DataFrame) -> dict:
    """重置 08-03+ 6 列并重放真实公告行, 返回 before/after 计数."""
    evt_agg = load_evt_agg()
    ratio_agg = load_ratio_agg()
    mask = panel["date"] >= CUTOFF
    before = {
        c: int(panel.loc[mask, c].notna().sum())
        for c in RESET_COLS
        if c in panel.columns
    }
    sub = panel.loc[mask, ["symbol", "date"]].copy()
    sub = sub.merge(evt_agg, on=["symbol", "date"], how="left")
    sub = sub.merge(ratio_agg, on=["symbol", "date"], how="left")
    assert len(sub) == mask.sum(), f"重放合并行数异常 {len(sub)} vs {mask.sum()}"
    for c in RESET_COLS:
        if c in panel.columns:
            panel.loc[mask, c] = np.nan
    panel.loc[mask, RESET_COLS] = sub[RESET_COLS].to_numpy()
    after = {
        c: int(panel.loc[mask, c].notna().sum())
        for c in RESET_COLS
        if c in panel.columns
    }
    return {
        "mask_rows": int(mask.sum()),
        "evt_reapplied": int(evt_agg["symbol"].nunique()),
        "evt_rows": len(evt_agg),
        "ratio_reapplied": int(ratio_agg["symbol"].nunique()),
        "ratio_rows": len(ratio_agg),
        "before": before,
        "after": after,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="备份后写盘 (默认仅 dry-run)")
    args = ap.parse_args()

    def prog(msg):
        print(msg, flush=True)

    panel = pd.read_parquet(PANEL_PATH)
    prog(f"加载面板: rows={len(panel):,} cols={panel.shape[1]}")

    missing = [c for c in RESET_COLS if c not in panel.columns]
    if missing:
        prog(f"面板缺少列 (跳过): {missing}")
    before_viol = _ohlcv_violations(panel)
    prog(f"写前 OHLCV 违例: {before_viol}")

    r = repair(panel)
    prog(f"08-03+ 行: {r['mask_rows']:,}")
    prog(
        f"真实公告重放: evt {r['evt_rows']} 行/{r['evt_reapplied']} 股, "
        f"ratio {r['ratio_rows']} 行/{r['ratio_reapplied']} 股"
    )
    for c in RESET_COLS:
        if c in r["before"]:
            prog(f"  {c:<22} 修复前非空 {r['before'][c]:,} → 修复后 {r['after'][c]:,}")
    if not args.apply:
        prog("\n[dry-run] 未写盘. 加 --apply 执行 (先备份).")
        return

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.splitext(PANEL_PATH)[0] + f"_preholder_repair_{ts}.parquet"
    shutil.copy2(PANEL_PATH, backup)
    prog(f"备份: {backup}")
    panel.to_parquet(PANEL_PATH, index=False)
    prog(f"已写回: {PANEL_PATH}")

    verify = pd.read_parquet(PANEL_PATH)
    assert verify.shape == panel.shape, f"写后形状变化 {verify.shape} vs {panel.shape}"
    twins = [c for c in verify.columns if c.endswith("_x") or c.endswith("_y")]
    assert not twins, f"出现孪生列: {twins}"
    after_viol = _ohlcv_violations(verify)
    assert after_viol == before_viol, f"OHLCV 违例变化 {before_viol} -> {after_viol}"
    prog(f"校验通过: 行数/列数不变, OHLCV 违例 {after_viol} (与写前一致)")


if __name__ == "__main__":
    main()
