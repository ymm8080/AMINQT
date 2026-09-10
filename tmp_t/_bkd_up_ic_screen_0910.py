# -*- coding: utf-8 -*-
"""双向短期特征族 IC 筛选 (bkd_ 破位 x10 + up_ 上涨 x6) — bkd 计划 Phase 1+2.

数据: data/_diag_stage_{board}_3y.parquet (585 列诊断帧, 自带原始 OHLCV/hfq/volume)。
全部特征 t 日收盘可知 (零前视), 向量化 groupby 计算, 禁 for 遍历个股。

筛选口径 (计划预设):
- 主判据 TS IC: 个股时序内 spearman(feature, label_10d_net), 全样本均值, |IC|>=0.01 保留;
- 参考 CS IC: 日截面 spearman 均值 (排名路径相关性, 仅报数);
- 共线去重: 与该板现产 feature_cols 的 max|spearman| (15 万行采样);
- 个案读数: 688228 @ 09-09 各特征取值; 连跌计数断言 (09-09 应=4, 09-04..09-09 四连阴)。

输出 WORM: diag/bkd_up_icscreen_{board}_{ts}.json
用法: python tmp_t/_bkd_up_ic_screen_0910.py [--board main|dual|both]
"""

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd

from config.settings import data_others_path  # noqa: E402

STAGE = {
    "main": "data/_diag_stage_main_3y.parquet",
    "dual": "data/_diag_stage_dual_3y.parquet",
}
RAW_COLS = [
    "symbol", "date", "open", "high", "low", "close",
    "open_hfq", "high_hfq", "low_hfq", "close_hfq", "volume", "label_10d_net",
]
KEEP_TS_IC = 0.01
DEDUP_SAMPLE_N = 150_000
DEDUP_RHO_WARN = 0.90
IC_MIN_OBS = 100
CASE_SYM, CASE_DATE = "688228", "2026-09-09"


def _bkd_up_features(df: pd.DataFrame) -> pd.DataFrame:
    """16 个候选, 全向量化 (groupby 窗口变换), 输入须已按 symbol,date 排序。"""
    g = df.groupby("symbol", sort=False)
    c = df["close_hfq"]
    pc = g["close_hfq"].shift(1)
    dn = (c < pc).astype("float64")
    up = (c > pc).astype("float64")

    ma5 = g["close_hfq"].transform(lambda s: s.rolling(5, min_periods=5).mean())
    ma10 = g["close_hfq"].transform(lambda s: s.rolling(10, min_periods=10).mean())
    ma20 = g["close_hfq"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    hi20 = g["high_hfq"].transform(lambda s: s.rolling(20, min_periods=20).max())
    lo20 = g["low_hfq"].transform(lambda s: s.rolling(20, min_periods=20).min())
    hi20p = g["high_hfq"].shift(1).groupby(df["symbol"], sort=False).transform(
        lambda s: s.rolling(20, min_periods=20).max()
    )
    hi5 = g["close_hfq"].transform(lambda s: s.rolling(5, min_periods=5).max())
    lo10 = g["low_hfq"].transform(lambda s: s.rolling(10, min_periods=10).min())
    hi60 = g["high_hfq"].transform(lambda s: s.rolling(60, min_periods=60).max())
    v20 = g["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())

    f = pd.DataFrame(index=df.index)
    # --- 破位族 bkd_ ---
    # 连跌分块: shift 必须在 symbol 组内 (跨股 shift 会串股边界 — 老 dn_streak 的坑)
    dn_prev = dn.groupby(df["symbol"], sort=False).shift(1)
    blk = (dn != dn_prev).groupby(df["symbol"], sort=False).cumsum()
    streak = dn.groupby([df["symbol"], blk], sort=False).cumsum()
    f["bkd_dn_streak"] = streak.clip(upper=10) * dn
    f["bkd_dn_days5"] = g_apply_rolling(dn, df["symbol"], 5, "sum")
    f["bkd_dd5_high20"] = c / hi20 - 1.0
    f["bkd_dd_high60"] = c / hi60 - 1.0
    f["bkd_min10_dist"] = c / lo10 - 1.0
    f["bkd_below_ma_cnt"] = (
        (c < ma5).astype("float64") + (c < ma10).astype("float64") + (c < ma20).astype("float64")
    )
    f["bkd_ma_bear_align"] = (
        (ma5 < ma10).astype("float64") + (ma10 < ma20).astype("float64") + (c < ma5).astype("float64")
    ) / 3.0
    f["bkd_ma5_slope5"] = ma5.groupby(df["symbol"], sort=False).pct_change(5)
    dn5 = g_apply_rolling(dn, df["symbol"], 5, "sum")
    dnvol5 = g_apply_rolling(dn * df["volume"], df["symbol"], 5, "sum")
    f["bkd_dnshrink5"] = (dnvol5 / dn5.replace(0.0, np.nan)) / v20
    rng = hi20 - lo20
    f["bkd_pos_range20"] = ((c - lo20) / rng.replace(0.0, np.nan)).clip(0.0, 1.0)
    # --- 上涨族 up_ ---
    up5 = g_apply_rolling(up, df["symbol"], 5, "sum")
    upvol5 = g_apply_rolling(up * df["volume"], df["symbol"], 5, "sum")
    f["up_vol_confirm5"] = (upvol5 / up5.replace(0.0, np.nan)) / v20
    body = (df["close_hfq"] - df["open_hfq"]) / pc  # 阳线实体为正 (hfq 口径)
    f["up_body5"] = g_apply_rolling(body, df["symbol"], 5, "mean")
    vr = df["volume"] / v20
    f["up_break20_vol"] = (c / hi20p - 1.0) * vr
    gap = df["open_hfq"] / pc - 1.0
    up_prev = (pc > g["close_hfq"].shift(2)).astype("float64")
    f["up_followthrough"] = g_apply_rolling((gap * up_prev).fillna(0.0), df["symbol"], 5, "mean")
    f["up_pullback_depth"] = c / hi5 - 1.0
    upgap_hold = (df["low_hfq"] > pc).astype("float64")
    f["up_gap_hold"] = g_apply_rolling(upgap_hold, df["symbol"], 10, "mean")
    return f


def g_apply_rolling(series: pd.Series, symbol: pd.Series, win: int, how: str) -> pd.Series:
    return series.groupby(symbol, sort=False).transform(lambda s: getattr(s.rolling(win, min_periods=win), how)())


def _grouped_spearman(x: pd.Series, y: pd.Series, key: pd.Series) -> tuple:
    """分组 spearman = 组内 rank 后按组 pearson (全向量化)。返回 (均值, 有效组数)。"""
    m = x.notna() & y.notna()
    if m.sum() < IC_MIN_OBS * 50:
        return float("nan"), 0
    xr = x[m].groupby(key[m]).rank()
    yr = y[m].groupby(key[m]).rank()
    d = pd.DataFrame({"k": key[m]})
    d["x"] = xr.astype("float64")
    d["y"] = yr.astype("float64")
    d["x2"] = d["x"] * d["x"]
    d["y2"] = d["y"] * d["y"]
    d["p"] = d["x"] * d["y"]
    gsum = d.groupby("k").agg(
        n=("x", "size"), mx=("x", "mean"), my=("y", "mean"),
        sxx=("x2", "sum"), syy=("y2", "sum"), sxy=("p", "sum"),
    )
    ok = gsum[gsum["n"] >= IC_MIN_OBS]
    if len(ok) == 0:
        return float("nan"), 0
    n = ok["n"].astype("float64")
    cov = ok["sxy"] / n - ok["mx"] * ok["my"]
    vx = ok["sxx"] / n - ok["mx"] ** 2
    vy = ok["syy"] / n - ok["my"] ** 2
    den = np.sqrt(vx.clip(lower=0) * vy.clip(lower=0))
    ic = (cov / den.replace(0.0, np.nan)).dropna()
    return float(ic.mean()), int(len(ic))


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", choices=("main", "dual", "both"), default="both")
    args = ap.parse_args()
    boards = ["main", "dual"] if args.board == "both" else [args.board]

    import psutil

    free_gb = psutil.virtual_memory().available / 1024**3
    if free_gb < 5.0:
        print(f"[ram] 空闲 {free_gb:.1f}GB < 5GB, 退出 (rc=3)", flush=True)
        return 3

    import pickle

    import pyarrow.parquet as pq
    from scipy.stats import spearmanr

    for board in boards:
        stage = STAGE[board]
        names = set(pq.ParquetFile(stage).schema_arrow.names)
        with open(os.path.join("models", "pipeline1", f"{board}_current.pkl"), "rb") as fh:
            prod_cols = list(pickle.load(fh)["feature_cols"])
        need = RAW_COLS + [c for c in prod_cols if c in names]
        print(f"[load] {stage} ({len(need)} cols)", flush=True)
        df = pd.read_parquet(stage, columns=sorted(set(need)))
        df["symbol"] = df["symbol"].astype(str)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)

        feats = _bkd_up_features(df)
        # 连跌计数断言: 09-04..09-09 四连阴 (老 dn_streak 算法此处读 1 = bug); 688228 只在 dual 板
        case = (df["symbol"] == CASE_SYM) & (df["date"] == CASE_DATE)
        if bool(case.any()):
            ci = int(np.flatnonzero(case.to_numpy())[0])
            st_case = float(feats["bkd_dn_streak"].iloc[ci])
            d5_case = float(feats["bkd_dn_days5"].iloc[ci])
            assert st_case == 4.0 and d5_case == 4.0, (
                f"连跌断言失败: streak={st_case} days5={d5_case} (应为 4/4)"
            )
            print(f"[assert] {CASE_SYM}@{CASE_DATE} 连跌 streak={st_case:.0f} days5={d5_case:.0f} OK", flush=True)
        else:
            ci = -1
            print(f"[assert] {CASE_SYM} 不在 {board} 板, 个案读数跳过", flush=True)

        label = df["label_10d_net"].astype("float64")
        symk = df["symbol"]
        datek = df["date"]
        rows = []
        for fname in feats.columns:
            x = feats[fname].astype("float64")
            ts_ic, n_sym = _grouped_spearman(x, label, symk)
            cs_ic, n_day = _grouped_spearman(x, label, datek)
            cv = float(x.notna().mean())
            case_v = None
            if ci >= 0:
                xv = x.iloc[ci]
                case_v = None if pd.isna(xv) else float(xv)
            rows.append({
                "feature": fname, "ts_ic": ts_ic, "n_symbols": n_sym,
                "cs_ic": cs_ic, "n_days": n_day, "coverage": cv,
                "case_688228_0909": case_v,
            })
            print(f"  {fname:22s} TS {ts_ic:+.4f} ({n_sym}) CS {cs_ic:+.4f} ({n_day}) cov {cv:.2f}", flush=True)

        # 共线去重 (采样, 与现产特征)
        rng = np.random.RandomState(7)
        si = rng.choice(len(df), size=min(DEDUP_SAMPLE_N, len(df)), replace=False)
        samp_x = feats.iloc[si].astype("float64")
        samp_p = df[need].iloc[si].drop(columns=[c for c in RAW_COLS if c in need], errors="ignore").astype("float64")
        for i, fname in enumerate(feats.columns):
            xx = samp_x[fname]
            best, bestc = 0.0, ""
            for pc_ in samp_p.columns:
                yy = samp_p[pc_]
                m = xx.notna() & yy.notna()
                if m.sum() < 1000:
                    continue
                r = spearmanr(xx[m], yy[m])[0]
                if np.isfinite(r) and abs(r) > abs(best):
                    best, bestc = float(r), pc_
            rows[i]["max_abs_rho_vs_prod"] = round(abs(best), 4)
            rows[i]["max_rho_col"] = bestc
            flag = " <DUP" if abs(best) > DEDUP_RHO_WARN else ""
            print(f"  {fname:22s} rho_max {abs(best):.3f} vs {bestc}{flag}", flush=True)
        del samp_x, samp_p
        gc.collect()

        keep = [r["feature"] for r in rows if abs(r["ts_ic"]) >= KEEP_TS_IC and r["max_abs_rho_vs_prod"] < DEDUP_RHO_WARN]
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = data_others_path("diag") / f"bkd_up_icscreen_{board}_{ts}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(
                {"board": board, "keep_ts_ic": KEEP_TS_IC, "dedup_rho": DEDUP_RHO_WARN,
                 "kept": keep, "features": rows},
                fh, ensure_ascii=False, indent=2,
            )
        n_keep_b = sum(1 for r in rows if abs(r["ts_ic"]) >= KEEP_TS_IC)
        print(f"[verdict] {board}: |TS IC|>={KEEP_TS_IC} {n_keep_b}/{len(rows)}, "
              f"去重后保留 {len(keep)}: {keep}", flush=True)
        print(f"[worm] {out} ({time.time() - t0:.0f}s)", flush=True)
        del df, feats, label
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
