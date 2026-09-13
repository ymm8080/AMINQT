# -*- coding: utf-8 -*-
"""0912 ChipDistribution 网格前视修复 (方案b') 独立验证 — 不改生产代码.

背景: chip_distribution.py:52 网格 = linspace(全帧 low.min*0.9, high.max*1.1) →
未来价格拉伸网格改写历史行获利盘 (夜 agent 合成验证 max 漂移 58pp)。

方案 b' (固定绝对对数网格, 常数与数据/窗口/股票全无关):
    grid = np.exp(np.linspace(log(0.5), log(6000), n_bins))   # A股价格包络, 留边
    n_bins 400→2000 (12000x 范围下 bin 宽 ≈ 0.44% 相对, 与旧 per-stock 网格同档)
    winner() 不动 (grid<price 掩码对任意单调网格成立)。

验证 (合成, 种子固定, 无面板):
  1. 因果性: 前缀 build vs 全量 build → 前缀行获利盘必须逐位相等 (生产网格对照必漂移)
  2. 极端包络: x20 暴涨股不出界不崩 (值域 [0,100], 因果性仍成立)
  3. 概念保真: 修复网格 vs 生产网格 全量获利盘秩相关 (应高位, 概念不变仅数值微移)

用法: python tmp_t/_chip_fix_optionb_validate_0912.py  (轻量, 自写日志)
产物: DATA OTHERS/diag/chip_fix_optionb_0912_{ts}.json (WORM)
"""

import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from config.settings import data_others_path

TAG = "chip_fix_optionb_0912"
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_chip_fix_optionb_validate_0912.log")

log = logging.getLogger(TAG)
log.setLevel(logging.INFO)
for h in (logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    log.addHandler(h)

N_BINS_FIX = 2000
FIX_LO, FIX_HI = 0.5, 6000.0


def grid_fixed() -> np.ndarray:
    return np.exp(np.linspace(np.log(FIX_LO), np.log(FIX_HI), N_BINS_FIX))


def grid_production(df: pd.DataFrame) -> np.ndarray:
    return np.linspace(df["low"].min() * 0.9, df["high"].max() * 1.1, 400)


def _triangle(grid: np.ndarray, low: float, high: float, peak: float) -> np.ndarray:
    """复刻 chip_distribution._triangle (逐行同数学)."""
    w = np.zeros(len(grid))
    in_range = (grid >= low) & (grid <= high)
    if not in_range.any():
        w[np.argmin(np.abs(grid - peak))] = 1.0
        return w
    w[in_range] = np.where(
        grid[in_range] <= peak,
        (grid[in_range] - low) / max(peak - low, 1e-9),
        (high - grid[in_range]) / max(high - peak, 1e-9),
    )
    w = np.clip(w, 0, None)
    return w / w.sum()


def build_winner_series(df: pd.DataFrame, grid: np.ndarray, float_shares: float) -> np.ndarray:
    """复刻 build() 的换手迁移主循环, 只回获利盘序列 (与生产 :57-69 同数学)."""
    dist = np.zeros(len(grid))
    out = []
    for o, h, l, c, v in zip(df["open"], df["high"], df["low"], df["close"], df["volume"], strict=True):
        a01 = (c + o + l + h) / 4
        t = min(v / float_shares, 1.0)
        tri = _triangle(grid, l, h, a01)
        if dist.sum() == 0:
            dist = tri
        else:
            dist = dist * (1 - t) + t * tri
        out.append(dist[grid < c].sum() / max(dist.sum(), 1e-12) * 100)
    return np.array(out)


def synth_stock(rng: np.random.Generator, n: int, p0: float, drift: float, vol: float) -> pd.DataFrame:
    ret = rng.normal(drift, vol, n)
    close = p0 * np.exp(np.cumsum(ret))
    o = close * np.exp(rng.normal(0, 0.004, n))
    spread = np.abs(rng.normal(0, 0.008, n))
    hi = np.maximum(o, close) * (1 + spread)
    lo = np.minimum(o, close) * (1 - spread)
    vol_sh = rng.uniform(0.5e6, 2.0e6, n)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": close, "volume": vol_sh})


def main() -> int:
    rng = np.random.default_rng(42)
    fs = 5.0e8
    stocks = {
        "普通漫步": synth_stock(rng, 250, 10.0, 0.0, 0.02),
        "阴跌股": synth_stock(rng, 250, 30.0, -0.004, 0.02),
        "x20暴涨股": synth_stock(rng, 250, 3.0, 0.0125, 0.03),  # ~e^(0.0125*250)≈22x
    }
    rep = {"n_bins_fixed": N_BINS_FIX, "envelope": [FIX_LO, FIX_HI], "stocks": {}}
    ok_all = True
    for name, df in stocks.items():
        cut = len(df) // 2
        res = {}
        for tag, gfn in (("fixed", grid_fixed), ("production", grid_production)):
            full = build_winner_series(df, gfn(df) if tag == "production" else gfn(), fs)
            pref = build_winner_series(df.iloc[:cut], gfn(df.iloc[:cut]) if tag == "production" else gfn(), fs)
            drift = np.abs(full[:cut] - pref)
            res[tag] = {
                "prefix_max_drift_pp": float(drift.max()),
                "prefix_identical": bool(np.array_equal(full[:cut], pref)),
                "range": [float(np.nanmin(full)), float(np.nanmax(full))],
            }
        # 概念保真: 两网格全量获利盘的秩相关 (同概念仅数值微移则高位)
        f_fix = build_winner_series(df, grid_fixed(), fs)
        f_prod = build_winner_series(df, grid_production(df), fs)
        rc = float(pd.Series(f_fix).corr(pd.Series(f_prod), method="spearman"))
        causal = res["fixed"]["prefix_identical"] and res["fixed"]["prefix_max_drift_pp"] == 0.0
        prod_bad = res["production"]["prefix_max_drift_pp"] > 1.0
        ok_all &= causal and prod_bad
        rep["stocks"][name] = {**res, "spearman_fixed_vs_production": rc,
                               "causal_pass": causal, "production_drifts": prod_bad}
        log.info("[%s] fixed 逐位一致=%s (drift=%.1e) | production drift max=%.2fpp | 秩相关=%.4f",
                 name, res["fixed"]["prefix_identical"], res["fixed"]["prefix_max_drift_pp"],
                 res["production"]["prefix_max_drift_pp"], rc)

    out_dir = data_others_path("diag")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{TAG}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"verdict": "PASS" if ok_all else "FAIL", **rep}, fh, ensure_ascii=False, indent=2)
    log.info("[WORM] %s", path)
    log.info("[DONE] %s — 固定网格因果性逐位一致, 生产网格对照漂移复现", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
