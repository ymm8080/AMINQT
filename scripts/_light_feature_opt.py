"""Legacy MAIN 特征优化 —— 小样本方向测试 (light direction-finding).

目的: 在跑昂贵的全量 TopN OOS 门测前, 用 ~200 只股票的样本快速判断
legacy MAIN (bruteforce_dedup) 特征池的优化方向. 复用生产 BruteForceGenerator
保证与真实链路一致 (同一批 base 列 / 变换族 / 窗口).

对比方向 (全部用研究口径 forward-return label 评 IC):
  B   baseline   : 现状选择 = nan_filter(0.95) + dedup_l2(0.7, 方差降序 greedy keep)
  D2  ic-keep    : 同一 dedup, 但组内按 |IC| 降序 greedy keep (选信号最强的)
  D3  new-families: 候选新变换族 (skew/kurt/zscore/rsi/mdd/cv) 与现有族在同组核心列上的 |IC| 对比
  D4  thr-sweep  : dedup_threshold ∈ {0.5, 0.7, 0.9} 对池质量的影响

输出: 控制台紧凑对比 + WORM json (data/_diag_light_feature_opt/light_feature_opt_<ts>.json).
研究 label 仅用于评估特征质量, 不参与特征构造, 无 look-ahead.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline1.feature_selector import (
    BruteForceGenerator,
    dedup_l2,
    nan_filter,
)
from config.settings import DATA_OTHERS_DIR, PANEL_V3_PATH

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("light_feature_opt")

PANEL = str(PANEL_V3_PATH)
N_SYMS = int(os.getenv("LT_N_SYMS", "200"))
N_DAYS = int(
    os.getenv("LT_N_DAYS", "300")
)  # 采样每股最近的交易日数 (含 rolling warmup, 前 ~60 天弃用)
RNG = int(os.getenv("LT_RNG", "42"))
LABEL_FWD = 5  # 主评估视界: T+5 close-to-close
ALT_FWDS = (2, 5, 10)

CORE_COLS = [
    "close_hfq",
    "open_hfq",
    "high_hfq",
    "low_hfq",
    "volume",
    "amount",
    "turnover_rate",
    "free_float_turnover_rate",
]


# 候选新变换族: (name, windows, fn(s, w))  — fn 接收一维数组+窗口, 返回同长数组
def _rsi(s, w=14):
    """RSI(w) on a 1-D series (pct-change based)."""
    ser = pd.Series(s).astype(float)
    diff = ser.diff()
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / w, min_periods=w).mean()
    avg_loss = loss.ewm(alpha=1 / w, min_periods=w).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


NEW_TRANSFORMS = {
    "skew": {
        "windows": (20, 60),
        "fn": lambda s, w: pd.Series(s).rolling(w, min_periods=5).skew(),
    },
    "kurt": {
        "windows": (60,),
        "fn": lambda s, w: pd.Series(s).rolling(w, min_periods=20).kurt(),
    },
    "zscore": {
        "windows": (20, 60),
        "fn": lambda s, w: (
            (pd.Series(s) - pd.Series(s).rolling(w, min_periods=5).mean())
            / pd.Series(s).rolling(w, min_periods=5).std()
        ),
    },
    "rsi": {"windows": (14,), "fn": _rsi},
    "mdd": {
        "windows": (20, 60),
        "fn": lambda s, w: (
            pd.Series(s).rolling(w, min_periods=5).max() / pd.Series(s) - 1
        ),
    },
    "cv": {
        "windows": (20, 60),
        "fn": lambda s, w: (
            pd.Series(s).rolling(w, min_periods=5).std()
            / pd.Series(s).rolling(w, min_periods=5).mean()
        ),
    },
}


def load_sample() -> pd.DataFrame:
    """读回 200 只随机 symbol 最近 N_DAYS 的 V3 面板 (predicate pushdown)."""
    syms = pq.read_table(PANEL, columns=["symbol"]).column("symbol").to_pandas()
    pick = sorted(
        np.random.RandomState(RNG).choice(syms.unique(), size=N_SYMS, replace=False)
    )
    df = pq.read_table(PANEL, filters=[("symbol", "in", pick)]).to_pandas()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    # 取最近 N_DAYS (每股), 形成干净公共窗口
    df["_r"] = df.groupby("symbol").cumcount()
    per = df.groupby("symbol")["_r"].max() + 1
    cutoff = int(per.median()) - N_DAYS
    df = df[df["_r"] >= cutoff].drop(columns=["_r"]).reset_index(drop=True)
    return df


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """每股 close-to-close forward return (研究口径)."""
    out = {}
    for k in ALT_FWDS:
        out[f"fwd{k}"] = (
            df.groupby("symbol")["close_hfq"].shift(-k) / df["close_hfq"] - 1
        )
    return pd.DataFrame(out)


def daily_spearman_ic(
    feat_df: pd.DataFrame, label: pd.Series, date: pd.Series
) -> pd.Series:
    """每日期截面 Spearman IC, 返回 feature → mean|IC| over dates (NaN 安全)."""
    d = feat_df.astype(float).copy()
    d["__lab"] = label.values
    d["__date"] = date.values
    feat_cols = [c for c in d.columns if not c.startswith("__")]
    # 组内 rank → Spearman = rank 的 Pearson
    ranked = d.groupby("__date")[feat_cols + ["__lab"]].rank(pct=True)
    dts = d["__date"].values
    uniq = pd.unique(dts)
    corrs = {}
    for dt in uniq:
        idx = np.where(dts == dt)[0]
        sub = ranked.iloc[idx]
        corrs[dt] = sub[feat_cols].corrwith(sub["__lab"])
    return pd.DataFrame(corrs).T.mean()


def ic_stats(ic_mean: pd.Series) -> dict:
    return {
        "mean_abs_ic": float(ic_mean.abs().mean()),
        "pct_abs_ic_gt_2pct": float((ic_mean.abs() > 0.02).mean()),
        "n_feat": int(ic_mean.notna().sum()),
    }


def dedup_ic_keep(feats, df, ic_series, threshold=0.7):
    """dedup_l2 的 IC 版: 组内按 |IC| 降序 greedy keep."""
    groups = {}
    for c in feats:
        if "_brute_" in c:
            base = c.split("_brute_")[0]
        elif c.startswith("dim"):
            base = c.split("dim")[1][:2]
        else:
            base = c.split("_")[0] if "_" in c else c
        groups.setdefault(base, []).append(c)

    kept = []
    for _base, cols in groups.items():
        if len(cols) <= 1:
            kept.extend(cols)
            continue
        avail = [c for c in cols if c in df.columns]
        if len(avail) <= 1:
            kept.extend(avail)
            continue
        n_sample = min(5000, len(df))
        sample = df[avail].sample(n_sample, random_state=RNG)
        corr = sample.corr(method="spearman").abs()
        order = avail
        if ic_series is not None:
            order = sorted(
                avail, key=lambda c: abs(ic_series.get(c, 0.0)), reverse=True
            )
        dropped = set()
        for i, ci in enumerate(order):
            if ci in dropped:
                continue
            for cj in order[i + 1 :]:
                if cj in dropped:
                    continue
                if corr.loc[ci, cj] > threshold:
                    dropped.add(cj)
        kept.extend([c for c in avail if c not in dropped])
    return kept


def gen_new_families(df: pd.DataFrame) -> pd.DataFrame:
    """候选新变换族在 CORE_COLS 上的特征 (研究用途, 不注册生产)."""
    per_sym = []
    for _sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("date")
        feats = {}
        for col in CORE_COLS:
            if col not in g.columns:
                continue
            s = g[col].astype(float).values
            for fam, spec in NEW_TRANSFORMS.items():
                for w in spec["windows"]:
                    try:
                        vals = spec["fn"](s, w)
                    except Exception:
                        vals = pd.Series(np.nan, index=range(len(s)))
                    feats[f"{col}_{fam}{w}"] = np.asarray(
                        vals, dtype=np.float32
                    ).reshape(-1)
        per_sym.append(pd.DataFrame(feats, index=g.index))
    return pd.concat(per_sym).replace([np.inf, -np.inf], np.nan)


def main() -> None:
    t0 = time.time()
    out_dir = DATA_OTHERS_DIR / "_diag_light_feature_opt"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_sample()
    logger.info(
        "sample %d rows x %d cols in %.1fs", len(df), df.shape[1], time.time() - t0
    )
    labels = build_labels(df)
    label = labels[f"fwd{LABEL_FWD}"]
    # 丢弃无未来标签的最后 k 行 (每股)
    mask = label.notna()
    df, labels, label = (
        df[mask].reset_index(drop=True),
        labels[mask].reset_index(drop=True),
        label[mask].reset_index(drop=True),
    )

    # ── 基线 brute 特征 (真实生产 generator) ──
    gen = BruteForceGenerator()
    eligible = gen._eligible(df)
    logger.info("eligible base cols: %d", len(eligible))
    t1 = time.time()
    fams = {}
    for fam in [
        "pct_change",
        "rolling_mean",
        "rolling_std",
        "rolling_max",
        "diff",
        "momentum",
        "EMA",
    ]:
        new = gen.generate_family(df, fam, raw_cols=eligible)
        fams[fam] = new
        logger.info("  %s -> %d cols", fam, new.shape[1])
    F = pd.concat(fams.values(), axis=1)
    logger.info("brute frame %s in %.1fs", F.shape, time.time() - t1)

    # ── OOS 拆分: 选择只在 train 期评 IC, 池质量只在 test 期评估 (防 in-sample 循环) ──
    u_dates = np.sort(pd.unique(df["date"].values))
    cut = u_dates[int(len(u_dates) * 0.6)]  # train 60% / test 40%
    train_mask = df["date"].values <= cut
    test_mask = ~train_mask
    logger.info(
        "OOS split: train %d dates (<=%s) / test %d dates",
        int(train_mask.sum()),
        cut,
        int(test_mask.sum()),
    )

    def _ic_train():
        return daily_spearman_ic(
            F.iloc[train_mask], label.iloc[train_mask], df["date"].iloc[train_mask]
        )

    def _eval_test(sel):
        ic_test = daily_spearman_ic(
            F.iloc[test_mask], label.iloc[test_mask], df["date"].iloc[test_mask]
        )
        return ic_stats(ic_test.reindex(sel))

    t2 = time.time()
    ic_train = _ic_train()
    logger.info("train-IC computed in %.1fs", time.time() - t2)

    report = {
        "created": datetime.now().strftime("%Y%m%dT%H%M%S"),
        "sample": {
            "n_syms": N_SYMS,
            "n_rows": int(len(df)),
            "n_eligible_cols": int(len(eligible)),
        },
        "label_fwd": LABEL_FWD,
        "oos_split": {
            "train_dates": int(train_mask.sum()),
            "test_dates": int(test_mask.sum()),
            "cut_date": str(cut),
        },
        "directions": {},
    }

    # ── B: baseline (方差 dedup, 现状) ──
    valid = nan_filter(list(F.columns), F, 0.95)
    sel_b = dedup_l2(valid, F, 0.7)
    report["directions"]["B_baseline_var_dedup"] = {
        "selected_count": len(sel_b),
        "ic": _eval_test(sel_b),
    }

    # ── D2: IC-keep (同池, 组内按 train |IC| 选) ──
    sel_ic = dedup_ic_keep(valid, F, ic_train, 0.7)
    report["directions"]["D2_ic_dedup"] = {
        "selected_count": len(sel_ic),
        "ic": _eval_test(sel_ic),
    }

    # ── D4: threshold sweep (IC-keep, 不同阈值) ──
    for thr in (0.5, 0.9):
        sel = dedup_ic_keep(valid, F, ic_train, thr)
        report["directions"][f"D4_ic_dedup_thr{thr}"] = {
            "selected_count": len(sel),
            "ic": _eval_test(sel),
        }

    # ── D3: 新变换族 vs 现有族 (同核心列, 全在 test 期评估) ──
    core_mask = [c for c in F.columns if c.split("_brute_")[0] in CORE_COLS]
    newF = gen_new_families(df)
    ic_test_existing = daily_spearman_ic(
        F.iloc[test_mask].loc[:, core_mask],
        label.iloc[test_mask],
        df["date"].iloc[test_mask],
    )
    ic_test_new = daily_spearman_ic(
        newF.iloc[test_mask], label.iloc[test_mask], df["date"].iloc[test_mask]
    )
    report["directions"]["D3_new_vs_existing"] = {
        "existing_family_mean_abs_ic_test": float(ic_test_existing.abs().mean()),
        "existing_family_n": int(ic_test_existing.notna().sum()),
        "new_family_mean_abs_ic_test": float(ic_test_new.abs().mean()),
        "new_family_n": int(ic_test_new.notna().sum()),
        "new_top_features": ic_test_new.abs()
        .sort_values(ascending=False)
        .head(10)
        .to_dict(),
        "existing_top_features": ic_test_existing.abs()
        .sort_values(ascending=False)
        .head(10)
        .to_dict(),
    }

    # ── 备选视界稳健性 (B vs D2, test 期 fwd2/fwd10) ──
    for k in (2, 10):
        ik_test = daily_spearman_ic(
            F.iloc[test_mask],
            labels.iloc[test_mask][f"fwd{k}"],
            df["date"].iloc[test_mask],
        )
        report["directions"]["B_baseline_var_dedup"]["ic"][f"mean_abs_ic_fwd{k}"] = (
            float(ik_test.reindex(sel_b).abs().mean())
        )
        report["directions"]["D2_ic_dedup"]["ic"][f"mean_abs_ic_fwd{k}"] = float(
            ik_test.reindex(sel_ic).abs().mean()
        )

    ts = report["created"]
    out_path = out_dir / f"light_feature_opt_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 72)
    print("LEGACY MAIN FEATURE OPTIMIZATION — LIGHT DIRECTION TEST")
    print(
        f"sample: {N_SYMS} syms x {len(df)} rows | label fwd{LABEL_FWD} | eligible cols {len(eligible)}"
    )
    print("=" * 72)
    print(
        f"{'direction':<28}{'n_feat':>8}{'mean|IC|':>10}{'%|IC|>2%':>10}{'fwd2':>8}{'fwd10':>8}"
    )
    for name, d in report["directions"].items():
        if name.startswith("D3"):
            continue
        ic = d["ic"]
        print(
            f"{name:<28}{ic['n_feat']:>8}{ic['mean_abs_ic']:>10.4f}"
            f"{ic['pct_abs_ic_gt_2pct']:>10.1%}"
            f"{ic.get('mean_abs_ic_fwd2', np.nan):>8.4f}"
            f"{ic.get('mean_abs_ic_fwd10', np.nan):>8.4f}"
        )
    print("-" * 72)
    d3 = report["directions"]["D3_new_vs_existing"]
    print(
        f"D3 new-families : mean|IC| {d3['new_family_mean_abs_ic_test']:.4f} (n={d3['new_family_n']}) [test期]"
    )
    print(
        f"   vs existing  : mean|IC| {d3['existing_family_mean_abs_ic_test']:.4f} (n={d3['existing_family_n']}) on same core cols [test期]"
    )
    print(
        "\nnew-family top:",
        {k: round(v, 4) for k, v in list(d3["new_top_features"].items())[:6]},
    )
    print(
        "existing  top:",
        {k: round(v, 4) for k, v in list(d3["existing_top_features"].items())[:6]},
    )
    print(f"\nWORM: {out_path}")


if __name__ == "__main__":
    main()
