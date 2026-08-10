"""Impact screen for Tushare limit_list_d (涨跌停/连板) candidate features.

双口径评估:
1. 条件收益分析 — 涨停/炸板/跌停/连板/封单强度/封板时间各组相对全池基线的
   T+2/3/5 净收益均值 + 上涨率 差 (稀疏二值列正确口径).
2. 截面 Rank IC — 二值列 min_x_unique=2, 连续/序数列 5; 连续列另算涨停子集内 IC.
分 main/dual 板, 半窗稳定性对照. 与项目 "上涨率" 验收口径一致
(label_pm_* 为验收权威 PM 标签, 日K近似). 结果落盘 JSON (WORM).
影响显著且稳定才允许接入 _daily_fetch.py 生产日更.

Usage: python scripts/_eval_limit_features.py [N_dates]
"""
import gc
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import tushare as ts
from scipy.stats import spearmanr

try:
    # Windows 控制台 GBK: 避免生僻 Unicode (如 U+2212) 打印时崩溃
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from config import settings
from app.pipeline1.cleaning_pipeline import CleaningPipeline
from app.pipeline1.label_engine import LabelEngine

N_DATES = int(sys.argv[1]) if len(sys.argv) > 1 else 250
PANEL = settings.PANEL_V3_PATH
TAG = datetime.now().strftime("%Y%m%d_%H%M%S")

LIMIT_FEATURES = [
    "is_limit_up",
    "is_limit_down",
    "is_zhaban",
    "limit_times",
    "fd_amount_ratio",
    "open_times",
    "seal_mins",
]
LABELS = ["label_pm_2d", "label_pm_3d", "label_pm_5d"]
LABEL_SFX = {"label_pm_2d": "2d", "label_pm_3d": "3d", "label_pm_5d": "5d"}

pro = ts.pro_api(settings.TUSHARE_TOKEN or ts.get_token())

# ── 1. Fetch limit_list_d for the panel's recent N trading dates ──
print(f"[1] Fetching limit_list_d for recent {N_DATES} trading dates...")
dates = sorted(pd.to_datetime(pd.read_parquet(PANEL, columns=["date"])["date"].unique()))[-N_DATES:]
print(f"    window: {dates[0].date()} .. {dates[-1].date()} ({len(dates)} dates)")
frames, empty = [], []
for i, d in enumerate(dates):
    ds = d.strftime("%Y%m%d")
    try:
        df = pro.limit_list_d(trade_date=ds)
    except Exception as e:
        empty.append(ds)
        continue
    if len(df):
        frames.append(df)
    time.sleep(0.12)
raw = pd.concat(frames, ignore_index=True)
print(f"    rows: {len(raw)} | empty dates: {len(empty)} {empty[:5]}")

# ── 2. Build per-stock daily candidate features ──
print("[2] Building limit features...")
raw["is_limit_up"] = (raw["limit"] == "U").astype(float)
raw["is_limit_down"] = (raw["limit"] == "D").astype(float)
raw["is_zhaban"] = (raw["limit"] == "Z").astype(float)
raw["limit_times"] = pd.to_numeric(raw["limit_times"], errors="coerce").fillna(0)
raw["fd_amount_ratio"] = raw["fd_amount"] / raw["float_mv"].replace(0, np.nan)
raw["fd_amount_ratio"] = raw["fd_amount_ratio"].fillna(0).clip(0, 0.5)
raw["open_times"] = pd.to_numeric(raw["open_times"], errors="coerce").fillna(0)


def _seal_mins(s):
    if isinstance(s, str) and len(s) >= 4:
        try:
            return int(s[:2]) * 60 + int(s[2:4]) - 570  # 09:30 = 570min
        except ValueError:
            return np.nan
    return np.nan


raw["seal_mins"] = raw["first_time"].map(_seal_mins)
feat = raw[["ts_code", "trade_date"] + LIMIT_FEATURES].copy()
feat["symbol"] = feat["ts_code"].str.replace(r"\.(SZ|SH|BJ)$", "", regex=True)
feat["date"] = pd.to_datetime(feat["trade_date"], format="%Y%m%d")
feat = feat[["symbol", "date"] + LIMIT_FEATURES]

# ── 3. Load panel window, join limit features ──
print("[3] Loading panel window + merging...")
panel = pd.read_parquet(PANEL)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel[panel["date"].isin(dates)]
print(f"    panel window: {len(panel)} rows, {panel.date.nunique()} dates, {panel.shape[1]} cols")
panel = panel.merge(feat, on=["symbol", "date"], how="left")
for c in LIMIT_FEATURES:
    if c != "seal_mins":
        panel[c] = panel[c].fillna(0.0)
del raw, feat
gc.collect()

# ── 4. Clean → Labels → 条件收益分析 (per board) ──
cleaner = CleaningPipeline()
main_df, dual_df = cleaner.run_train(panel)
del panel
gc.collect()


def _stats(sub, col):
    v = sub[col].dropna()
    if len(v) < 10:
        return None
    return {
        "n": int(len(v)),
        "mean": round(float(v.mean()), 5),
        "hit": round(float((v > 0).mean()), 4),
    }


def _diff(sub, col, base):
    s = _stats(sub, col)
    if s is None or base is None:
        return s
    s["delta_mean_pp"] = round((s["mean"] - base["mean"]) * 100, 3)
    s["delta_hit_pp"] = round((s["hit"] - base["hit"]) * 100, 2)
    return s


def _group_rows(df, lbl_col):
    """各组相对基线的收益差 (pp), 按标签列 lbl_col 计算."""
    base = _stats(df, lbl_col)
    up = df["is_limit_up"] == 1
    rows = []
    groups = [
        ("baseline", pd.Series(True, index=df.index)),
        ("涨停封住", up),
        ("炸板", df["is_zhaban"] == 1),
        ("跌停封住", df["is_limit_down"] == 1),
        ("首板(涨停1连板)", up & (df["limit_times"] == 1)),
        ("连板>=2(涨停)", up & (df["limit_times"] >= 2)),
        ("炸板且开板>=2次", (df["is_zhaban"] == 1) & (df["open_times"] >= 2)),
    ]
    for name, mask in groups:
        rows.append((name, _diff(df[mask], lbl_col, base)))
    # 封单强度 / 封板时间 (仅在涨停组内分档)
    up_df = df[up]
    if len(up_df) > 20:
        fd_med = up_df["fd_amount_ratio"].median()
        rows.append(("涨停+强封单(>=中位)", _diff(df[up & (df["fd_amount_ratio"] >= fd_med)], lbl_col, base)))
        rows.append(("涨停+弱封单(<中位)", _diff(df[up & (df["fd_amount_ratio"] < fd_med)], lbl_col, base)))
        seal = up_df["seal_mins"].dropna()
        if len(seal) > 20:
            s_med = seal.median()
            rows.append(("涨停+早封板(<=中位)", _diff(df[up & df["seal_mins"].notna() & (df["seal_mins"] <= s_med)], lbl_col, base)))
            rows.append(("涨停+晚封板(>中位)", _diff(df[up & df["seal_mins"].notna() & (df["seal_mins"] > s_med)], lbl_col, base)))
    return base, rows


# ── Rank IC (日度截面) — 二值列 min_x_unique=2, 连续/序数列 5 ──
_MIN_XU = {"is_limit_up": 2, "is_limit_down": 2, "is_zhaban": 2}
_CONT_IC_COLS = ["limit_times", "fd_amount_ratio", "open_times", "seal_mins"]


def _daily_rank_ic(df, x_col, y_col):
    ics = []
    for _d, g in df.groupby("date"):
        sub = g[[x_col, y_col]].dropna()
        min_xu = _MIN_XU.get(x_col, 5)
        if (
            len(sub) < 20
            or sub[x_col].nunique() < min_xu
            or sub[y_col].nunique() < 2
        ):
            continue
        r, _ = spearmanr(sub[x_col], sub[y_col])
        if not np.isnan(r):
            ics.append(r)
    return np.array(ics)


def _ic_block(df, x_col, y_col, min_days=20):
    a = _daily_rank_ic(df, x_col, y_col)
    if len(a) < min_days:
        return None
    s = a.std(ddof=1)
    return {
        "ic_mean": round(float(a.mean()), 5),
        "ic_ir": round(float(a.mean() / s) if s > 0 else 0.0, 4),
        "days": int(len(a)),
    }


def _ic_print(v):
    if v is None:
        return "   (样本不足)"
    return f"{v['ic_mean']:+.4f} {v['ic_ir']:>+7.2f} {v['days']:>5d}"


def _analyze(board_df, board):
    df = LabelEngine.build_labels(board_df)  # PM session 默认
    df = LabelEngine.mask_suspension(df)
    df = LabelEngine.mask_recent_days(df, days=6)
    n_all = int(len(df))
    print(f"\n[4] {board.upper()} rows={n_all}")

    out = {"n_rows": n_all, "labels": {}}
    for lbl in LABELS:
        if lbl not in df.columns:
            continue
        base, rows = _group_rows(df, lbl)
        if base is None:
            continue
        sfx = LABEL_SFX[lbl]
        print(f"\n    === T+{sfx} (PM 执行口径) 基线 mean={base['mean']*100:+.2f}% 上涨率={base['hit']*100:.1f}% ===")
        print(f"    {'组':24s} {'n':>6s} {'mean%':>8s} {'上涨率%':>8s} {'Δmeanpp':>8s} {'Δhitpp':>7s}")
        print("    " + "-" * 64)
        details = {}
        for name, s in rows:
            if s is None:
                print(f"    {name:24s}  (样本不足)")
                continue
            print(
                f"    {name:24s} {s['n']:>6d} {s['mean']*100:>+8.2f} {s['hit']*100:>8.1f} "
                f"{s.get('delta_mean_pp', 0):>+8.3f} {s.get('delta_hit_pp', 0):>+7.2f}"
            )
            details[name] = s
        out["labels"][sfx] = {"baseline": base, "groups": details}

    # 半窗稳定性: 涨停封住 vs 基线, 上半窗/下半窗
    dts = sorted(df["date"].unique())
    if len(dts) >= 20:
        mid = dts[len(dts) // 2]
        out["stability"] = {}
        for half_name, half_df in [("first_half", df[df["date"] < mid]), ("second_half", df[df["date"] >= mid])]:
            for lbl in LABELS:
                if lbl not in half_df.columns:
                    continue
                b = _stats(half_df, lbl)
                up = _stats(half_df[half_df["is_limit_up"] == 1], lbl)
                if b and up:
                    out["stability"].setdefault(LABEL_SFX[lbl], {})[half_name] = {
                        "base_hit": b["hit"], "up_hit": up["hit"],
                        "delta_hit_pp": round((up["hit"] - b["hit"]) * 100, 2),
                        "base_mean": b["mean"], "up_mean": up["mean"],
                        "delta_mean_pp": round((up["mean"] - b["mean"]) * 100, 3),
                        "n_up": up["n"],
                    }
        print("\n    === 半窗稳定性 (涨停封住 Δ 相对基线) ===")
        for sfx, halves in out["stability"].items():
            for hn, h in halves.items():
                print(f"    {sfx} {hn:12s} Δmean={h['delta_mean_pp']:+.3f}pp Δhit={h['delta_hit_pp']:+.2f}pp (n_up={h['n_up']})")

    # ── Rank IC: 全截面; 涨停子集只对连续/序数列 ──
    dts = sorted(df["date"].unique())
    out["ic"] = {}
    print("\n    === Rank IC (日度截面, 二值列 min_xu=2 / 连续列 5) ===")
    for lbl in LABELS:
        if lbl not in df.columns:
            continue
        sfx = LABEL_SFX[lbl]
        print(f"\n    T+{sfx}:")
        print(f"    {'feature':<16s} {'IC':>8s} {'ICIR':>7s} {'days':>5s} | {'涨停子集 IC':>12s} {'ICIR':>7s} {'days':>5s}")
        print("    " + "-" * 68)
        icg = {}
        for c in LIMIT_FEATURES:
            full = _ic_block(df, c, lbl)
            if c in _CONT_IC_COLS:
                wl = _ic_block(df[df["is_limit_up"] == 1], c, lbl)
                wl_s = _ic_print(wl)
            else:
                wl, wl_s = None, "n/a"
            icg[c] = {"full": full, "within_limit_up": wl}
            print(f"    {c:<16s} {_ic_print(full)} | {wl_s:>28s}")
        out["ic"][sfx] = icg

    # IC 半窗稳定性
    if len(dts) >= 20:
        mid = dts[len(dts) // 2]
        print("\n    === Rank IC 半窗稳定性 (ΔIC = 后半窗 - 前半窗) ===")
        out["ic_stability"] = {}
        for hn, h_df in [("first", df[df["date"] < mid]), ("second", df[df["date"] >= mid])]:
            for lbl in LABELS:
                if lbl not in h_df.columns:
                    continue
                sfx = LABEL_SFX[lbl]
                for c in LIMIT_FEATURES:
                    blk = _ic_block(h_df, c, lbl, min_days=10)
                    if blk is None:
                        continue
                    out["ic_stability"].setdefault(sfx, {}).setdefault(c, {})[hn] = blk
        for sfx, feats in out["ic_stability"].items():
            for c, halves in feats.items():
                if len(halves) != 2:
                    continue
                d_ic = halves["second"]["ic_mean"] - halves["first"]["ic_mean"]
                print(
                    f"    {sfx} {c:<16s} ΔIC={d_ic:+.4f} "
                    f"(first={halves['first']['ic_mean']:+.4f} second={halves['second']['ic_mean']:+.4f})"
                )
    return out


results_out = {}
for board, board_df in [("main", main_df), ("dual", dual_df)]:
    if len(board_df) == 0:
        continue
    results_out[board] = _analyze(board_df, board)

results_out["meta"] = {
    "tag": TAG,
    "window_dates": [str(x.date()) for x in dates],
    "n_dates": len(dates),
    "script": "_eval_limit_features.py",
    "source": "Tushare limit_list_d",
    "labels": "label_pm_* PM 执行口径 (验收权威, 日K近似)",
}
os.makedirs("data/factor_registry", exist_ok=True)
path = f"data/factor_registry/limit_eval_{TAG}.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(results_out, f, ensure_ascii=False, indent=1)
print(f"\nSaved: {path}")
print("DONE")
