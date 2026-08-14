"""
Three-Layer Feature Selection Module.

Layer1: BruteForceGenerator + registry building
Layer2: DedupL2 (MAIN), GateD (DUAL), versioning, save/load
"""

import json
import logging
import os
import re
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config.settings import data_others_path

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# BruteForceGenerator (Layer1)
# ──────────────────────────────────────────────────────────


class BruteForceGenerator:
    """Generate ~3200 brute-force time-series features from raw panel columns.

    Per-symbol groupby, 8 transform families x multiple windows.
    Applied to all eligible numeric columns in the panel.
    """

    BASE_TRANSFORM_DEFS = {
        "pct_change": {"windows": (1, 2, 3, 5, 10, 20, 40, 60), "suffix": "pct"},
        "rolling_mean": {"windows": (5, 10, 20, 40, 60), "suffix": "ma"},
        "rolling_std": {"windows": (5, 10, 20, 40), "suffix": "std"},
        "rolling_max": {"windows": (10, 20, 40), "suffix": "max"},
        "rolling_min": {"windows": (10, 20, 40), "suffix": "min"},
        "diff": {"windows": (1, 5, 20), "suffix": "d"},
        "momentum": {"windows": (5, 20, 40), "suffix": "mom"},
        "EMA": {"windows": (5, 20, 40), "suffix": "ema"},
    }

    EXCLUDE_COLS = {
        "symbol",
        "date",
        "board",
        "industry",
        "announce_date",
        "is_suspended",
        "tradestatus",
        # ── Forward-filled quarterly (fina_indicator): step function,
        #    brute-force variants are constant within quarter → IC≈0, pure waste ──
        "roe",
        "roa",
        "gross_margin",
        "net_margin",
        "eps_yoy",
        "rev_yoy",
        "profit_yoy",
        "op_cf_ratio",
        "debt_ratio",
        "current_ratio",
        "asset_turnover",
        "inventory_turnover",
        "eps",
        "bps",
        "ocfps",
        "revenue_ps",
        "roe_deducted",
        "roe_yoy",
        "q_roe",
        "dt_eps",
        "q_ocf_to_sales",
        "ar_turnover",
        # ── dim22 outputs: dim22 already extracts QoQ/YoY/trend time-series
        #    signals; brute-force on these produces redundant 2nd-order derivatives ──
        "roe_qoq",
        "roa_qoq",
        "margin_chg",
        "growth_accel",
        "profit_accel",
        "debt_leveraging",
        "efficiency_chg",
        "ocf_stability",
        "roe_trend_4q",
        "margin_trend_4q",
        "rev_yoy_trend",
        "quality_momentum",
        # ── HS300×1y 与全市场×3y 双口径 |wIC|<0.02 低信息列
        #    (2026-08-04 _diag_hs300_exclude_cross 结论, A 层 brute 展开纯浪费) ──
        "circ_mv",
        "resistance_dist",
        "short_sell_vol",
        "chip_entropy",
        "peak_roc_20d",
        "chip_gini",
        "total_mv",
        "bias_5_20_cross",
        "short_balance",
        "volume_ratio",
        "conc_90_industry_rank",
        "margin_balance",
        "peak_roc_5d",
        "pct_90_con",
        "chip_skew_dist",
        "cost_bias",
        "pctChg",
    }

    def __init__(self, transforms=None, eligible_cols=None):
        self.transforms = transforms or self.BASE_TRANSFORM_DEFS
        self.eligible_cols = eligible_cols  # None = all numeric

    def _eligible(self, df):
        cols = self.eligible_cols
        if cols is None:
            cols = [
                c
                for c in df.columns
                if c not in self.EXCLUDE_COLS
                and not c.startswith("label_")
                and not c.startswith("dim")
                and df[c].dtype in ("float64", "int64")
            ]
        return [c for c in cols if c in df.columns]

    def generate(self, df, raw_cols=None):
        """Generate brute-force features, return new columns DataFrame."""
        t0 = time.time()
        raw = raw_cols or self._eligible(df)
        all_new = {}
        for sym, g in df.groupby("symbol"):
            g = g.sort_values("date")
            feats = {}
            for col in raw:
                if col not in g.columns:
                    continue
                s = g[col].astype(float).values
                n = len(s)

                for w in self.transforms.get("pct_change", {}).get("windows", ()):
                    o = np.full(n, np.nan)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        o[w:] = (s[w:] - s[:-w]) / np.abs(s[:-w]) * 100
                    feats[f"{col}_brute_pct{w}"] = o

                for w in self.transforms.get("rolling_mean", {}).get("windows", ()):
                    feats[f"{col}_brute_ma{w}"] = (
                        pd.Series(s).rolling(w, min_periods=1).mean().values
                    )

                for w in self.transforms.get("rolling_std", {}).get("windows", ()):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        feats[f"{col}_brute_std{w}"] = (
                            pd.Series(s).rolling(w, min_periods=1).std().values
                        )

                for w in self.transforms.get("rolling_max", {}).get("windows", ()):
                    r = pd.Series(s).rolling(w, min_periods=1)
                    feats[f"{col}_brute_max{w}"] = r.max().values
                    feats[f"{col}_brute_min{w}"] = r.min().values

                for w in self.transforms.get("diff", {}).get("windows", ()):
                    o = np.full(n, np.nan)
                    o[w:] = s[w:] - s[:-w]
                    feats[f"{col}_brute_d{w}"] = o

                for w in self.transforms.get("momentum", {}).get("windows", ()):
                    o = np.full(n, np.nan)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        o[w:] = s[w:] / np.abs(s[:-w])
                    feats[f"{col}_brute_mom{w}"] = o

                for w in self.transforms.get("EMA", {}).get("windows", ()):
                    feats[f"{col}_brute_ema{w}"] = (
                        pd.Series(s).ewm(span=w, min_periods=1).mean().values
                    )

            all_new[sym] = pd.DataFrame(feats, index=g.index).replace(
                [np.inf, -np.inf], np.nan
            )

        new = pd.concat(all_new.values())
        logger.info(
            "BruteForce: %d cols from %d raw cols (%.0fs)",
            len(new.columns),
            len(raw),
            time.time() - t0,
        )
        return new

    def _family_for_symbol(self, g, raw, family_name, windows, suffix):
        """单 symbol 的一族暴力特征 (与旧 generate_family 内层 for 逐字节一致).

        抽成独立函数, 供 generate_family (物化宽帧) 与 family_stats (流式统计)
        共用同一套特征数学, 避免两处实现漂移. 返回 (排序后的 g, feats dict).
        """
        g = g.sort_values("date")
        feats = {}
        for col in raw:
            if col not in g.columns:
                continue
            s = g[col].astype(float).values
            n = len(s)
            if family_name == "pct_change":
                for w in windows:
                    o = np.full(n, np.nan, dtype=np.float32)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        o[w:] = (s[w:] - s[:-w]) / np.abs(s[:-w]) * 100
                    feats[f"{col}_brute_{suffix}{w}"] = o
            elif family_name == "rolling_mean":
                for w in windows:
                    feats[f"{col}_brute_{suffix}{w}"] = (
                        pd.Series(s)
                        .rolling(w, min_periods=1)
                        .mean()
                        .values.astype(np.float32)
                    )
            elif family_name == "rolling_std":
                for w in windows:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        feats[f"{col}_brute_{suffix}{w}"] = (
                            pd.Series(s)
                            .rolling(w, min_periods=1)
                            .std()
                            .values.astype(np.float32)
                        )
            elif family_name in ("rolling_max", "rolling_min"):
                for w in windows:
                    feats[f"{col}_brute_max{w}"] = (
                        pd.Series(s)
                        .rolling(w, min_periods=1)
                        .max()
                        .values.astype(np.float32)
                    )
                    feats[f"{col}_brute_min{w}"] = (
                        pd.Series(s)
                        .rolling(w, min_periods=1)
                        .min()
                        .values.astype(np.float32)
                    )
            elif family_name == "diff":
                for w in windows:
                    o = np.full(n, np.nan, dtype=np.float32)
                    o[w:] = s[w:] - s[:-w]
                    feats[f"{col}_brute_{suffix}{w}"] = o.astype(np.float32)
            elif family_name == "momentum":
                for w in windows:
                    o = np.full(n, np.nan, dtype=np.float32)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        o[w:] = s[w:] / np.abs(s[:-w])
                    feats[f"{col}_brute_{suffix}{w}"] = o.astype(np.float32)
            elif family_name == "EMA":
                for w in windows:
                    feats[f"{col}_brute_{suffix}{w}"] = (
                        pd.Series(s)
                        .ewm(span=w, min_periods=1)
                        .mean()
                        .values.astype(np.float32)
                    )
        return g, feats

    def generate_family(self, df, family_name, raw_cols=None, dtype="float32"):
        """Generate brute-force features for ONE transform family.

        Memory-safe: one family at a time (7 loops), joined incrementally.
        Peak ~ base + 1 family, never holds all 3200 cols at once.
        Returns DataFrame with new feature columns in specified dtype.
        """
        t0 = time.time()
        raw = raw_cols or self._eligible(df)
        family_def = self.transforms.get(family_name)
        if family_def is None:
            raise ValueError(f"Unknown transform family: {family_name}")
        windows = family_def.get("windows", ())
        suffix = family_def.get("suffix", family_name)
        all_new = {}
        for sym, g in df.groupby("symbol"):
            g, feats = self._family_for_symbol(g, raw, family_name, windows, suffix)
            all_new[sym] = pd.DataFrame(feats, index=g.index).replace(
                [np.inf, -np.inf], np.nan
            )
        new = pd.concat(all_new.values())
        logger.info(
            "BruteForce[%s]: %d cols from %d raw (%.0fs, float32)",
            family_name,
            len(new.columns),
            len(raw),
            time.time() - t0,
        )
        return new

    def family_stats(self, df, family_name, sample_pos, raw_cols=None, dtype="float32"):
        """一族暴力特征的 每列 nan率 + 采样行值 — 免物化 (N×2544) 宽帧.

        OOM 根因 (2026-08-11): generate_family 内 pd.concat(all_new.values()) 把
        单家族 (pct_change) 的 (1223918, 2544) float32 拼成 11.6GB 连续块, 且
        all_new 已常驻同量 → 15.8GB 机触发 _ArrayMemoryError. 唯一消费者
        _run_bruteforce_dedup 只需要 每列 nan率 + 采样行值, 故按 symbol 逐列累计
        这两样统计, 常驻 ~50MB. 输出与 generate_family → 逐列统计 完全一致.
        """
        t0 = time.time()
        raw = raw_cols or self._eligible(df)
        family_def = self.transforms.get(family_name)
        if family_def is None:
            raise ValueError(f"Unknown transform family: {family_name}")
        windows = family_def.get("windows", ())
        suffix = family_def.get("suffix", family_name)
        pos_array = np.asarray(list(sample_pos))
        columns = None
        nan_count = None
        sample_vals = None
        for _sym, g in df.groupby("symbol"):
            g, feats = self._family_for_symbol(g, raw, family_name, windows, suffix)
            if columns is None:
                columns = list(feats.keys())
                nan_count = np.zeros(len(columns), dtype=np.int64)
                sample_vals = {
                    c: np.full(len(pos_array), np.nan, dtype=dtype) for c in columns
                }
            for i, c in enumerate(columns):
                arr = np.where(np.isinf(feats[c]), np.nan, feats[c])
                nan_count[i] += int(np.isnan(arr).sum())
            hit = np.isin(g.index.to_numpy(), pos_array)
            if hit.any():
                hit_pos = np.nonzero(hit)[0]
                sample_positions = sample_pos.get_indexer(g.index.to_numpy()[hit_pos])
                for c in columns:
                    sample_vals[c][sample_positions] = np.where(
                        np.isinf(feats[c][hit_pos]), np.nan, feats[c][hit_pos]
                    )
        total = float(len(df))
        nan_rate = {c: float(nan_count[i]) / total for i, c in enumerate(columns)}
        logger.info(
            "BruteForce[%s] stats: %d cols from %d raw (%.0fs, stream)",
            family_name,
            len(columns),
            len(raw),
            time.time() - t0,
        )
        return columns, nan_rate, sample_vals

    def generate_columns(self, df, family_name, need, raw_cols=None, dtype="float32"):
        """按需只生成 need 中属于本族的列 (免物化全族宽帧 OOM).

        2026-08-11 后注入 OOM 根因: 逐族 generate_family 把 pct_change 族
        (1223918, 2544) float32 物化成 11.6GB 单块 → _ArrayMemoryError →
        FeatureSelector 回退全量 → main 3d_cls 塌缩. 本方法逐 symbol 复用
        _family_for_symbol, 但只驻留 need 交集列 (常驻 ~#need 列而非全族),
        列命名/数学与 generate_family 逐字节一致. 无命中列返回 None.
        """
        t0 = time.time()
        raw = raw_cols or self._eligible(df)
        family_def = self.transforms.get(family_name)
        if family_def is None:
            return None
        windows = family_def.get("windows", ())
        suffix = family_def.get("suffix", family_name)
        all_new = {}
        for sym, g in df.groupby("symbol"):
            g, feats = self._family_for_symbol(g, raw, family_name, windows, suffix)
            pick = {c: v for c, v in feats.items() if c in need}
            if not pick:
                continue
            all_new[sym] = pd.DataFrame(pick, index=g.index).replace(
                [np.inf, -np.inf], np.nan
            )
        if not all_new:
            return None
        new = pd.concat(all_new.values())
        logger.info(
            "BruteForce[%s] 后注入 %d 列 (%d syms, %.0fs)",
            family_name,
            len(new.columns),
            len(all_new),
            time.time() - t0,
        )
        return new


# Brute-force 变换族顺序 — 与 generate() 内层 for 完全一致 (pct→ma→std→max/min→d→mom→ema).
# rolling_max 族同时产出 max+min 列 (见 generate_family), 故不含 rolling_min.
# build_features / FeatureSelector._run_bruteforce_dedup / train_runner 后注入共用,
# 保证候选列命名与族序恒定, 避免 RAM 优化后选择漂移.
BRUTE_FAMILIES = [
    "pct_change",
    "rolling_mean",
    "rolling_std",
    "rolling_max",
    "diff",
    "momentum",
    "EMA",
]


# ──────────────────────────────────────────────────────────
# Dedup L2 (Layer2, MAIN)
# ──────────────────────────────────────────────────────────


def dedup_l2(feats, df, threshold=0.7, order=None):
    """Correlation dedup within same base column group.

    For brute-force features, base column is prefix before '_brute_'.
    For curated features, base column is dim prefix or raw col name.
    Keeps features greedily: sort by variance descending, drop if
    |r| > threshold with any already-kept feature in the same group.

    order: pd.Series indexed by feature → keep score (e.g. mean|IC|).  When
    given, each group is sorted by |order| descending instead of variance
    (predictive-priority dedup).  Default None → 原方差降序行为不变.
    """
    groups = {}
    for c in feats:
        if "_brute_" in c:
            base = c.split("_brute_")[0]
        elif c.startswith("dim"):
            m = re.match(r"(dim\d+)", c)
            base = m.group(1) if m else c
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

        # Sample for speed
        n_sample = min(5000, len(df))
        sample = df[avail].sample(n_sample, random_state=42)
        corr = sample.corr(method="spearman").abs()

        # Sort by variance (proxy for importance) or by order score, keep if |r| < threshold
        if order is not None:
            ordered = sorted(avail, key=lambda c: abs(order.get(c, 0.0)), reverse=True)
        else:
            vars_ = sample.var().sort_values(ascending=False)
            ordered = [c for c in vars_.index if c in avail]
        dropped = set()
        for i, ci in enumerate(ordered):
            if ci in dropped:
                continue
            for cj in ordered[i + 1 :]:
                if cj in dropped:
                    continue
                if corr.loc[ci, cj] > threshold:
                    dropped.add(cj)

        kept.extend([c for c in avail if c not in dropped])

    logger.info(
        "DedupL2: %d -> %d features (|r|>%.2f)", len(feats), len(kept), threshold
    )
    return kept


def feature_mean_abs_ic(
    feat_frame: pd.DataFrame,
    date_ser: pd.Series,
    label_ser: pd.Series,
) -> dict[str, float]:
    """Per-feature 日度截面 |Spearman IC| 时间均值 (内存有界, 全量安全).

    用于 MAIN IC 排序 dedup (dedup_key="ic"): 在给定日期子集上, 对每个特征
    计算每日截面 Spearman IC 后取绝对值的时间均值. NaN 安全 (组内 rank + corrwith,
    与 light 测试验证口径一致). 只评估特征质量, 不参与特征构造, 无 look-ahead.
    float32 输入保持 float32 (仅 rank 输出为 float64). 逐日 corrwith 用
    按日排序边界切片, 免去每日全表布尔扫描.
    """
    feat_cols = list(feat_frame.columns)
    if not feat_cols:
        return {}
    d = feat_frame.copy()
    # reindex 按 index 标签对齐: feat_frame 可能来自 brute family (symbol-major
    # 顺序, 是 df index 的排列), 而 date/label 是 df 原始顺序, 位置对齐会错配标签.
    d["__lab"] = label_ser.reindex(d.index).to_numpy()
    d["__date"] = date_ser.reindex(d.index).to_numpy()
    # 组内 rank (NaN 保持 NaN) → Spearman = rank 的 Pearson.
    # rank 输出 float64 → 立即降 float32 (rank∈[0,1], 相关精度足够), 防大族
    # (rolling_max ~1260 列) 生成 12GB+ float64 秩矩阵触 commit 上限.
    ranked = (
        d.groupby("__date", observed=True)[feat_cols + ["__lab"]]
        .rank(pct=True)
        .astype(np.float32)
    )
    dts = d["__date"].to_numpy()
    order = np.argsort(dts, kind="stable")
    ranked = ranked.iloc[order].reset_index(drop=True)
    dto = dts[order]
    bounds = np.flatnonzero(np.diff(dto) != 0) + 1
    segs = np.concatenate(([0], bounds, [len(dto)]))
    acc: dict[str, list[float]] = {c: [] for c in feat_cols}
    for s, e in zip(segs[:-1], segs[1:]):
        block = ranked.iloc[s:e]
        corr = block[feat_cols].corrwith(block["__lab"])
        for c, v in corr.items():
            if not pd.isna(v):
                acc[c].append(float(abs(v)))
    return {c: float(np.nanmean(v)) for c, v in acc.items() if v}


# ──────────────────────────────────────────────────────────
# Gate D (Layer2, DUAL)
# ──────────────────────────────────────────────────────────


def gate_d_ablation(
    feats,
    df,
    label_col="label_1d_net",
    min_feats=30,
    sat_pct=0.95,
    lgb_params=None,
    random_state=42,
    metrics_out=None,
):
    """Importance forward ablation with saturation gate.

    1. Train full model on all feats, rank by gain importance
    2. Test ablation points (5,10,20,30,...,all,min_feats)
    3. Stop at 95% of best ICIR, clamped to min_feats

    metrics_out (dict|None): 提供则就地记录选择指标 (best_ir/best_n/sat_n/
    n_candidates/n_selected/ablation_log), 供调用方落盘快照. 返回类型不变 (list).
    """
    if metrics_out is not None:
        metrics_out.update(n_candidates=len(feats))
    if len(feats) <= min_feats:
        if metrics_out is not None:
            metrics_out.update(n_selected=len(feats), note="feats<=min_feats")
        return feats

    dates = sorted(df["date"].unique())
    split = int(len(dates) * 0.8)  # internal 80/20 for ablation
    tr = df[df["date"].isin(dates[:split])].dropna(subset=[label_col])
    te = df[df["date"].isin(dates[split:])].dropna(subset=[label_col])

    avail = [c for c in feats if c in df.columns]
    if len(avail) <= min_feats:
        if metrics_out is not None:
            metrics_out.update(n_selected=len(avail), note="avail<=min_feats")
        return avail

    base_params = dict(
        n_estimators=300,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        # 2026-08-12: gate_d 每周重训抽到不同结果 (r3=30 特征 vs 本轮=200, 输入完全一致),
        # 根因是 n_jobs=-1 多线程下 LGBM 直方图构建非确定 → gain 排序漂移 → 消融 ICIR
        # 曲线翻盘. deterministic+force_col_wise 固定直方图算法, 使消融结果可复现.
        deterministic=True,
        force_col_wise=True,
        bagging_seed=random_state + 1,
        feature_fraction_seed=random_state + 2,
        drop_seed=random_state + 3,
        n_jobs=-1,
        verbose=-1,
    )
    if lgb_params:
        base_params.update(lgb_params)

    # Full model importance
    full = lgb.LGBMRegressor(**base_params)
    full.fit(tr[avail], tr[label_col])
    imp = pd.DataFrame(
        {
            "feature": avail,
            "gain": full.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False)

    def _eval_icir(preds):
        df_e = te.copy()
        df_e["pred"] = preds
        ics = [
            spearmanr(g["pred"], g[label_col])[0]
            for _, g in df_e.groupby("date")
            if len(g) >= 10
        ]
        a = np.array([x for x in ics if not np.isnan(x)])
        # len(a)<=1 时 std 必然为 0 (原式也返回 0), 提前短路避免 numpy
        # "Degrees of freedom <= 0" RuntimeWarning (单样本 var).
        if len(a) > 1:
            sd = a.std()
            if sd > 0:
                return float(round(a.mean() / sd, 4))
        return 0.0

    # Quick ablation
    ab_params = dict(base_params)
    ab_params["n_estimators"] = 200
    ns = sorted(set([5, 10, 20, 30, 50, 75, 100, 150, 200, len(avail), min_feats]))
    best_n, best_ir = min_feats, 0.0
    ablation_log = []

    for n in ns:
        if n > len(avail):
            continue
        top = imp.head(n)["feature"].tolist()
        m = lgb.LGBMRegressor(**ab_params)
        m.fit(tr[top], tr[label_col])
        ir = _eval_icir(m.predict(te[top]))
        ablation_log.append({"n": n, "icir": ir})
        if ir > best_ir:
            best_n, best_ir = n, ir

    # 95% saturation
    sat_n = min_feats
    for log_entry in ablation_log:
        if log_entry["icir"] >= best_ir * sat_pct:
            sat_n = log_entry["n"]
            break
    sat_n = max(sat_n, min_feats)

    selected = imp.head(sat_n)["feature"].tolist()
    logger.info(
        "GateD: %d -> %d features (best_ir=%.4f @ n=%d, sat_n=%d)",
        len(avail),
        len(selected),
        best_ir,
        best_n,
        sat_n,
    )
    if metrics_out is not None:
        metrics_out.update(
            n_candidates=len(avail),
            n_selected=len(selected),
            best_ir=best_ir,
            best_n=best_n,
            sat_n=sat_n,
            ablation_log=ablation_log,
        )
    return selected


# ──────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────


def nan_filter(feats, df, threshold=0.95, min_support=0):
    """Drop features with NaN rate >= threshold.

    min_support > 0 → 事件类稀疏特征豁免: 只要非空行数 >= min_support 就保留,
    即便全局 NaN 率超 threshold (在事件 scope 内是稠密的, 全截面评价会误杀).
    """
    from pandas.api.types import is_numeric_dtype

    good = []
    for c in feats:
        if c not in df.columns or not is_numeric_dtype(df[c]):
            continue
        s = df[c]
        if s.isna().mean() < threshold or (
            min_support and s.notna().sum() >= min_support
        ):
            good.append(c)
    logger.info(
        "NaN filter: %d -> %d (threshold=%.2f, min_support=%d)",
        len(feats),
        len(good),
        threshold,
        min_support,
    )
    return good


# ──────────────────────────────────────────────────────────
# 子数据集 scope-IC 筛选 (事件类稀疏特征)
# ──────────────────────────────────────────────────────────
# 事件类特征 (增减持/龙虎榜/大宗) 全截面覆盖率极低 (holder ~0.4%),
# 在全截面被 nan 过滤 + 增益排序误杀。改为在其事件 scope
# (ht_stocks ± window 交易日) 内评日频 rank IC, |IC| >= ic_bar 者并入精选集。
# 这正是 "在子数据集上训练/评估后并入特征选择" 的落地。


def event_scope_mask(df, event_col, window=30):
    """事件 scope 掩码: 行落在任一事件前后 window 个交易日内的位置."""
    if event_col not in df.columns:
        return pd.Series(False, index=df.index)
    ev = df[event_col].notna().astype("int8")
    grp = ev.groupby(df["symbol"], sort=False)
    fwd = grp.transform(lambda s: s.rolling(window, min_periods=1).max())
    bwd = grp.transform(
        lambda s: s.iloc[::-1].rolling(window, min_periods=1).max().iloc[::-1]
    )
    return ((fwd > 0) | (bwd > 0)).fillna(False).astype(bool)


def daily_rank_ic(df, feat, label, mask, min_cross=10, min_dates=10):
    """scope 内按日 Spearman rank IC 均值 (跨日截面, 每日至少 min_cross 只)."""
    sub = df.loc[mask, [feat, label, "date"]].dropna(subset=[feat, label])
    if len(sub) < min_cross:
        return 0.0
    ics = []
    for _, g in sub.groupby("date"):
        if len(g) < min_cross:
            continue
        r = spearmanr(g[feat], g[label])[0]
        if not np.isnan(r):
            ics.append(r)
    if len(ics) < min_dates:
        return 0.0
    return float(np.mean(ics))


def scope_ic_union(
    selected,
    df,
    event_col,
    feats,
    label_cols,
    window=30,
    ic_bar=0.01,
    min_support=1000,
    min_cross=10,
    min_dates=10,
):
    """子数据集 IC 筛选 → 并入精选集. 返回扩展后的特征列表."""
    if event_col not in df.columns:
        return selected
    support = int(df[event_col].notna().sum())
    if support < min_support:
        logger.info(
            "scope_ic_union[%s]: 事件支撑不足 (%d < %d), 跳过",
            event_col,
            support,
            min_support,
        )
        return selected
    labels = [c for c in label_cols if c in df.columns]
    if not labels:
        logger.warning(
            "scope_ic_union[%s]: 无可用 label (%s), 跳过", event_col, label_cols
        )
        return selected
    mask = event_scope_mask(df, event_col, window=window)
    if not mask.any():
        return selected
    added = []
    for feat in feats:
        if feat in selected or feat not in df.columns:
            continue
        ics = [
            abs(daily_rank_ic(df, feat, lab, mask, min_cross, min_dates))
            for lab in labels
        ]
        best = max(ics) if ics else 0.0
        if best >= ic_bar:
            added.append(feat)
            logger.info(
                "scope_ic_union[%s]: %s scope-IC=%.4f (bar=%.2f) -> 并入",
                event_col,
                feat,
                best,
                ic_bar,
            )
    if added:
        logger.info(
            "scope_ic_union[%s]: 并入 %d 个特征 %s", event_col, len(added), added
        )
    return selected + added


# ──────────────────────────────────────────────────────────
# 事件类稀疏特征子数据集 IC 筛入配置 (scope_ic_union 驱动表)
# ──────────────────────────────────────────────────────────
# 全截面评价会误杀这些 ~0.4% 覆盖率的事件特征 (增减持/大宗); 在其事件
# scope 内评日频 rank IC, |IC| >= ic_bar 者并入精选集. 这是"在子数据集上
# 训练/评估后并入特征选择"的落地. 训练主链路 (train_runner.select_features)
# 与独立 Layer2 工具 (scripts/select_features.py) 共用本表, 口径一致.
EVENT_SCOPE_SCREENS = [
    {
        "name": "dim29_holdertrade",
        "event_col": "sh_net_ratio",
        "window": 30,
        "feats": ["sh_ratio_30d", "sh_net_ratio", "sh_g_ratio", "sh_c_ratio"],
        "label_cols": ["label_pm_1d_net", "label_pm_3d_net", "label_pm_5d_net"],
        "ic_bar": 0.01,
        "min_support": 1000,
    },
    {
        "name": "dim33_blocktrade",
        "event_col": "bt_count",
        "window": 30,
        "feats": [
            "bt_act_ewma",
            "bt_disc_ewma",
            "bt_inst_abs_ewma",
            "bt_mv_ratio_ewma",
        ],
        "label_cols": ["label_pm_1d_net", "label_pm_3d_net", "label_pm_5d_net"],
        "ic_bar": 0.01,
        "min_support": 1000,
    },
    {
        "name": "lhb_combined",
        "event_col": "lhb_net_buy",
        "window": 20,
        "feats": [
            # dim26_lhb_enhanced
            "lhb_inst_net_buy_5d",
            "lhb_inst_net_buy_20d",
            "lhb_inst_count_5d",
            "lhb_inst_buy_ratio",
            "lhb_abnormal_score",
            # dim32_lhb_glm
            "lhb_inst_flow",
            "lhb_retail_flow",
            "lhb_sell_pressure",
            "lhb_list_count_5d",
            # dim34_lhb_v2 (KIMI)
            "lhb2_inst_flow",
            "lhb2_inst_shock",
            "lhb2_top_flow",
            "lhb2_quant_flow",
            "lhb2_retail_flow",
            "lhb2_sell_pressure",
            "lhb2_sell_buy_ratio",
            "lhb2_list_count_5d",
            "lhb2_conboard_mem",
            "lhb2_inst_strength",
            "lhb2_inst_resolve",
            "lhb2_inst_conboard",
            "lhb2_inst_premium",
            "lhb2_inst_lock",
        ],
        "label_cols": ["label_pm_1d_net", "label_pm_3d_net", "label_pm_5d_net"],
        "ic_bar": 0.01,
        "min_support": 1000,
    },
]


def apply_event_scope_screens(selected, df, screens=None):
    """Run all event-scope IC screens, union qualifying features into `selected`.

    Shared by the training mainline (train_runner.select_features) and the
    standalone Layer2 tool (scripts/select_features.py) so both produce the
    same subset-evaluated feature list. 单个 screen 失败不阻断精选/训练.
    """
    screens = EVENT_SCOPE_SCREENS if screens is None else screens
    for screen in screens:
        try:
            before = len(selected)
            selected = scope_ic_union(
                selected,
                df,
                screen["event_col"],
                screen["feats"],
                screen["label_cols"],
                window=screen.get("window", 30),
                ic_bar=screen.get("ic_bar", 0.01),
                min_support=screen.get("min_support", 1000),
            )
            if len(selected) > before:
                logger.info(
                    "apply_event_scope_screens[%s]: 并入 %d 个特征",
                    screen["name"],
                    len(selected) - before,
                )
        except Exception as exc:
            logger.warning(
                "apply_event_scope_screens[%s] 失败 (%s), 忽略", screen["name"], exc
            )
    return selected


# ──────────────────────────────────────────────────────────
# 三频模型频率归属 (2026-08-04 全市场×3年 6格判定, _classify_freq_full.py)
# ──────────────────────────────────────────────────────────
# 每列 (feat → (freq, eval_type)): freq ∈ {月, 周, 日} 决定进哪个频率模型;
# eval_type ∈ {TS, XS} 是评估口径 (per-stock 时序 vs 日截面), 记录方法论文档用.
# 核心铁律: 月频特征不进日频模型. select_freq 按基列频率把选中特征路由到
# {月, 周, 日} 三张表; brute-force 变体 (col_brute_*) 继承基列频率.
# 事件类 (LHB/HOLDER/BT) 走事件池, 独立事件模块, 不进三频常规模.
FREQ_ORDER = ("月", "周", "日")

FREQ_ASSIGNMENT = {
    # ── chip 筹码 ──
    "pct_90_con": ("月", "TS"),
    "pct_90_high": ("月", "TS"),
    "weight_avg": ("月", "TS"),
    "conc_trend_20d": ("月", "TS"),
    "resistance_dist": ("月", "TS"),
    "chip_entropy": ("日", "TS"),
    "chip_gini": ("日", "TS"),
    "peak_roc_5d": ("日", "TS"),
    "chip_skew_dist": ("月", "XS"),
    "conc_90_industry_rank": ("月", "XS"),
    "peak_price": ("月", "XS"),
    "peak_roc_20d": ("周", "TS"),
    "cost_bias": ("周", "XS"),
    "support_dist": ("周", "TS"),
    # ── cost 成本线 ──
    "cost_50pct": ("月", "TS"),
    "cost_95pct": ("月", "TS"),
    # ── price 价格 ──
    "close_hfq": ("周", "TS"),
    # ── vol 量 ──
    "volume": ("月", "XS"),
    "amount": ("月", "XS"),
    "turnover_rate": ("月", "XS"),
    "free_float_turnover_rate": ("月", "XS"),
    "volume_ratio": ("月", "TS"),
    "ma_vol_ratio_5_20": ("月", "XS"),
    "vol_surge": ("月", "XS"),
    "amt_surge": ("月", "XS"),
    # ── ma 均线乖离 ──
    "bias_5": ("周", "TS"),
    "bias_10": ("周", "TS"),
    "bias_20": ("月", "TS"),
    "bias_60": ("日", "TS"),
    "bias_120": ("月", "TS"),
    "bias_250": ("月", "TS"),
    "bias_5_20_cross": ("日", "XS"),
    "bias_20_60_cross": ("日", "TS"),
    # ── volatility 波动 ──
    "amplitude_5d": ("月", "XS"),
    "intraday_range": ("月", "XS"),
    "winner_ratio": ("周", "TS"),
    "pctChg": ("周", "TS"),
    # ── valuation 估值市值 ──
    "pe_ttm": ("周", "TS"),
    "pb": ("周", "TS"),
    "ps_ttm": ("周", "TS"),
    "dv_ratio": ("月", "TS"),
    "total_mv": ("周", "TS"),
    "circ_mv": ("周", "TS"),
    # ── margin 两融 ──
    "margin_balance": ("月", "TS"),
    "short_balance": ("周", "TS"),
    "margin_buy_amt": ("月", "XS"),
    "short_sell_vol": ("周", "TS"),
    # ── fundamental 基本面 (盈利质量→月; 成长/每股→周) ──
    "roe": ("月", "TS"),
    "roe_deducted": ("月", "TS"),
    "roa": ("月", "TS"),
    "gross_margin": ("月", "TS"),
    "debt_ratio": ("月", "TS"),
    "current_ratio": ("月", "TS"),
    "asset_turnover": ("月", "TS"),
    "ar_turnover": ("月", "TS"),
    "inventory_turnover": ("月", "TS"),
    "rev_yoy": ("周", "TS"),
    "net_margin": ("周", "TS"),
    "eps_yoy": ("周", "TS"),
    "profit_yoy": ("周", "TS"),
    "ocfps": ("周", "TS"),
    "revenue_ps": ("周", "TS"),
    "bps": ("周", "TS"),
    "eps": ("周", "TS"),
    "dt_eps": ("周", "TS"),
    "roe_yoy": ("周", "TS"),
    "q_roe": ("周", "TS"),
    "q_ocf_to_sales": ("周", "TS"),
    "ocf_to_or": ("日", "TS"),
}

# ── 同族类比映射 (2026-08-04, 非单独6格判定) ──
# 生产选中特征池含大量 brute-force 变体, 基列多为已判定列的同族副本或训练时生成
# 变体 (_x/_y 后缀). 按与已判定同族列的函数一致性推导频率, 而非逐个重跑 6格:
#   - OHLCV 价格族 → 周 (同 close_hfq 判 TS·周)
#   - 换手/量/流动性 → 月·XS (同 turnover_rate 判 XS·月)
#   - 股本/市值 → 周·TS (同 total_mv/circ_mv 判 TS·周)
#   - 筹码/成本 _x/_y 变体 → 月·TS (同 pct_90_con/cost_50pct 判 TS·月)
#   - dv_ttm → 月 (同 dv_ratio), 涨跌停 → 日 (快信号), 行业相对收益 → 周 (return 族)
#   - 上市天数 → 月 (静态慢变)
# 真新族 (benefit_part/churn_suspect 等) 不猜, 仍归 '未分类' 由覆盖率报告暴露.
FAMILY_ANALOG = {
    # 价格族 (close_hfq: TS·周)
    "open": ("周", "TS"),
    "high": ("周", "TS"),
    "low": ("周", "TS"),
    "close": ("周", "TS"),
    "pre_close": ("周", "TS"),
    "open_hfq": ("周", "TS"),
    "high_hfq": ("周", "TS"),
    "low_hfq": ("周", "TS"),
    # 换手/量/流动性 (turnover_rate: XS·月)
    "turn": ("月", "XS"),
    "turnover_rate_f": ("月", "XS"),
    "rank_ff_turnover": ("月", "XS"),
    "rank_amount": ("月", "XS"),
    "liquidity_score": ("月", "XS"),
    "adv20": ("月", "XS"),
    "turnover_stability_5": ("月", "XS"),
    # 股本/市值 (total_mv/circ_mv: TS·周)
    "total_share": ("周", "TS"),
    "float_share": ("周", "TS"),
    "free_share": ("周", "TS"),
    # 筹码/成本 _x/_y 变体 (pct_90_con/cost_50pct: TS·月)
    "pct_70_con_x": ("月", "TS"),
    "pct_70_con_y": ("月", "TS"),
    "pct_70_high_x": ("月", "TS"),
    "pct_70_high_y": ("月", "TS"),
    "pct_70_low_x": ("月", "TS"),
    "pct_70_low_y": ("月", "TS"),
    "pct_90_con_x": ("月", "TS"),
    "pct_90_con_y": ("月", "TS"),
    "pct_90_high_x": ("月", "TS"),
    "pct_90_high_y": ("月", "TS"),
    "pct_90_low_x": ("月", "TS"),
    "pct_90_low_y": ("月", "TS"),
    "cost_5pct_x": ("月", "TS"),
    "cost_5pct_y": ("月", "TS"),
    "cost_15pct_x": ("月", "TS"),
    "cost_15pct_y": ("月", "TS"),
    "cost_50pct_x": ("月", "TS"),
    "cost_50pct_y": ("月", "TS"),
    "cost_85pct_x": ("月", "TS"),
    "cost_85pct_y": ("月", "TS"),
    "cost_95pct_x": ("月", "TS"),
    "cost_95pct_y": ("月", "TS"),
    "avg_cost_x": ("月", "TS"),
    "avg_cost_y": ("月", "TS"),
    "weight_avg_x": ("月", "TS"),
    "weight_avg_y": ("月", "TS"),
    # 股息 (dv_ratio: TS·月)
    "dv_ttm": ("月", "TS"),
    # 涨跌停 (快信号 → 日)
    "up_limit_raw": ("日", "TS"),
    "down_limit_raw": ("日", "TS"),
    # 行业/板块相对收益 (窗口后缀定频: _1d→日 _5d→周 _20d→月; 未6格判定)
    "sw_ret_1d": ("日", "TS"),
    "sw_ret_1d_x": ("日", "TS"),
    "sw_ret_5d": ("周", "TS"),
    "sector_return": ("周", "TS"),
    "sector_return_5d": ("周", "TS"),
    "sw_relative_strength": ("周", "TS"),
    "sw_rotation_position": ("周", "TS"),
    "sw_ret_20d": ("月", "TS"),
    "sw_vol_20d": ("月", "TS"),
    "ind_holder_trend_20d": ("月", "TS"),
    "sw_momentum_accel": ("月", "TS"),
    "ind_margin_accel": ("月", "TS"),
    "ind_margin_chg_5d": ("周", "TS"),
    "sw_index_close": ("周", "TS"),
    "sw_index_vol": ("周", "TS"),
    "market_turnover": ("周", "TS"),
    "market_turnover_ratio_5d": ("周", "TS"),
    "market_turnover_ratio_20d": ("月", "TS"),
    "market_limit_up": ("日", "TS"),
    # 波动/流动性 (同 amplitude_5d/turnover_rate: XS·月)
    "ATR_pct": ("月", "XS"),
    "amihud_illiq": ("月", "XS"),
    "amihud_illiquidity": ("月", "XS"),
    "sw_turnover_anomaly": ("月", "XS"),
    "free_float_turnover_rate_xrank": ("月", "XS"),
    "amount_xrank": ("月", "XS"),
    "turnover_f_chg_5d": ("月", "XS"),
    # 快价格信号 (日动量)
    "close_vs_low": ("日", "TS"),
    "overnight_ret": ("日", "TS"),
    "ROC_3d": ("日", "TS"),
    "gap_strength_5d": ("周", "TS"),
    "gap_strength_20d": ("月", "TS"),
    "gap_vs_ma5": ("周", "TS"),
    # 日历月 (季节效应)
    "month": ("月", "TS"),
    # 上市天数 (静态慢变 → 月)
    "list_days": ("月", "TS"),
    # benefit_part = winner_ratio 旧名 (settings.py:44 别名; winner_ratio 判 TS·周)
    "benefit_part_x": ("周", "TS"),
    "benefit_part_y": ("周", "TS"),
    # churn_suspect = 换手稳定性派生洗盘标志 (cleaning_pipeline:140, 同 turnover_stability_5)
    "churn_suspect": ("月", "XS"),
}

# 事件类基列前缀 → 事件模块 (不进三频常规模)
_EVENT_PREFIXES = ("sh_", "lhb", "bt_", "holder")


def freq_of(feature: str) -> str:
    """返回特征频率归属: 月/周/日 / 事件 / 未分类.

    解析顺序: 已测 FREQ_ASSIGNMENT → 同族类比 FAMILY_ANALOG → 事件前缀 →
    '未分类' (不静默默认, 由覆盖率报告暴露). brute-force 变体继承基列频率.
    """
    base = feature.split("_brute_")[0] if "_brute_" in feature else feature
    if base in FREQ_ASSIGNMENT:
        return FREQ_ASSIGNMENT[base][0]
    if base in FAMILY_ANALOG:
        return FAMILY_ANALOG[base][0]
    if base.startswith(_EVENT_PREFIXES):
        return "事件"
    return "未分类"


# ──────────────────────────────────────────────────────────
# FeatureSelector (Layer2 orchestrator + versioning)
# ──────────────────────────────────────────────────────────


class FeatureSelector:
    """Board-dispatched feature selection with versioning."""

    DEFAULT_CONFIG = {
        "main": {
            "pipeline": "bruteforce_dedup",
            "nan_threshold": 0.95,
            "dedup_threshold": 0.7,
            # [2026-08-12] 选择过大 OOM 修复: 全量 dedup 每次选 ~3506 → 注入/训练
            # OOM → 永远回退. 保底 FeatureEngine 基集, 剩余预算按方差补 brute.
            "max_features": 350,
        },
        "dual": {
            "pipeline": "gate_d",
            "nan_threshold": 0.95,
            "gate_d": {
                "min_features": 30,
                "saturation_pct": 0.95,
                "label": "label_pm_1d_net",
            },
        },
        "fallback": {"pipeline": "ic_screener"},
    }

    def __init__(self, config=None, registry_dir=None):
        self.config = config or self.DEFAULT_CONFIG
        self.registry_dir = registry_dir or str(
            data_others_path("data/factor_registry")
        )
        os.makedirs(self.registry_dir, exist_ok=True)

    # ── Selection ──

    def select(self, df, board, generator=None):
        """Run feature selection for a board. Returns list of feature names.

        每次选择结束都会落盘一个时间戳版本快照 (selected_{board}_{ts}.json,
        不覆盖旧文件, WORM), 便于回看历次 dedup_l2 / gate_d 选中特征.
        """
        board_cfg = self.config.get(board, self.config.get("fallback", {}))
        pipeline = board_cfg.get("pipeline", "ic_screener")

        metrics: dict = {}
        if pipeline == "bruteforce_dedup":
            result = self._run_bruteforce_dedup(
                df, board, board_cfg, generator, metrics_out=metrics
            )
        elif pipeline == "gate_d":
            result = self._run_gate_d(df, board, board_cfg, metrics_out=metrics)
        else:
            from app.pipeline1.feature_engine_v35 import FeatureEngineV35

            result = FeatureEngineV35.feature_columns(df)

        try:
            self.save_version(
                {
                    "board": board,
                    "pipeline": pipeline,
                    "created": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "selected_count": len(result),
                    "features": result,
                    "metrics": metrics,
                },
                board,
                activate=False,
            )
        except Exception as exc:  # 快照失败不阻断选择
            logger.warning("[%s] 保存选中特征快照失败: %s", board, exc)
        return result

    def select_freq(self, df, board, generator=None):
        """三频选择: 运行常规 select() 后按基列频率路由到 {月, 周, 日} 三张表.

        月频列特征 (含其 brute-force 变体) 只进月频表, 日频表不会混入月频特征
        (铁律: 月频特征不进日频模型). 事件类特征独立成 '事件' 桶 (只报告不落训练表);
        未在 FREQ_ASSIGNMENT 者进 '未分类' 桶并在覆盖率报告里暴露, 不静默默认.
        每张表 + 覆盖率报告均落盘时间戳版本 (WORM, 不覆盖). select() 行为不变.
        """
        selected = self.select(df, board, generator=generator)
        buckets = {freq: [] for freq in FREQ_ORDER}
        buckets["事件"] = []
        buckets["未分类"] = []
        for f in selected:
            buckets[freq_of(f)].append(f)

        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        for freq in FREQ_ORDER:
            path = os.path.join(self.registry_dir, f"selected_{board}_{freq}_{ts}.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "board": board,
                        "freq": freq,
                        "created": ts,
                        "selected_count": len(buckets[freq]),
                        "features": buckets[freq],
                    },
                    fh,
                    indent=2,
                    ensure_ascii=False,
                )
        cov_path = os.path.join(self.registry_dir, f"selected_{board}_freq_{ts}.json")
        with open(cov_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "board": board,
                    "ts": ts,
                    "coverage": {k: len(v) for k, v in buckets.items()},
                    "unknown": buckets["未分类"],
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(
            "[%s] select_freq 覆盖率: %s",
            board,
            {k: len(v) for k, v in buckets.items()},
        )
        return buckets

    def _run_bruteforce_dedup(self, df, board, cfg, generator=None, metrics_out=None):
        if generator is None:
            generator = BruteForceGenerator()
        raw_cols = generator._eligible(df)
        threshold = cfg.get("nan_threshold", 0.95)
        dedup_thr = cfg.get("dedup_threshold", 0.7)
        base_numeric = [
            c
            for c in df.columns
            if c not in BruteForceGenerator.EXCLUDE_COLS
            and not c.startswith("label_")
            and df[c].dtype in ("float64", "int64")
        ]
        # 内存安全: 绝不物化全量候选宽表 (2200+ 列 join 触 pandas block
        # consolidation → 需 ~9GB 连续块, 15.8GB 物理机 OOM).
        # 1) 逐列算 nan 率 (列无关, 无需宽表); 2) 只驻留固定 5000 行采样帧供
        # dedup_l2 用 (与旧 dedup_l2 内部 df.sample(5000, random_state=42)
        # 取同一行子集 → 相关性/方差相同 → 选择结果逐元素一致).
        n_sample = min(5000, len(df))
        sample_pos = df.sample(n_sample, random_state=42).index

        # IC 排序 dedup (dedup_key="ic", 默认关): 组内按 |IC| 降序 greedy keep,
        # 而非方差. IC 只在 ic_cut_date 之前的行上评估 (诚实, 防 OOS label 泄漏进选择).
        need_ic = cfg.get("dedup_key") == "ic"
        ic_label = cfg.get("ic_label", "label_pm_5d_net")
        ic_cut = cfg.get("ic_cut_date")
        if need_ic:
            if ic_cut is not None:
                pre_mask = df["date"] < pd.Timestamp(ic_cut)
            else:
                pre_mask = pd.Series(True, index=df.index)
            ic_date = df.loc[pre_mask, "date"]
            ic_lab = df.loc[pre_mask, ic_label]
        else:
            pre_mask = ic_date = ic_lab = None
        ic_accum: dict[str, float] = {}

        cand_nan: dict[str, float] = {}
        sample_cols: dict[str, np.ndarray] = {}
        for c in base_numeric:
            cand_nan[c] = float(df[c].isna().mean())
            sample_cols[c] = df.loc[sample_pos, c].to_numpy()
        if need_ic and base_numeric:
            # 一次批量评 IC (避免 90 次单独 groupby rank)
            ic_accum.update(
                feature_mean_abs_ic(df.loc[pre_mask, base_numeric], ic_date, ic_lab)
            )
        for fam in BRUTE_FAMILIES:
            if need_ic:
                # IC 排序 dedup 需全行逐日 cross-section Spearman → 无法逐 symbol 流式
                # 累计, 只能物化宽帧 (dedup_key="ic" 默认关, 且已被 TopN 门否决, 不设防).
                new = generator.generate_family(
                    df, fam, raw_cols=raw_cols, dtype="float32"
                )
                for c in new.columns:
                    if c in BruteForceGenerator.EXCLUDE_COLS or c.startswith("label_"):
                        continue
                    cand_nan[c] = float(new[c].isna().mean())
                    sample_cols[c] = new.loc[sample_pos, c].to_numpy()
                ic_accum.update(feature_mean_abs_ic(new.loc[pre_mask], ic_date, ic_lab))
                del new
            else:
                # 免物化宽帧 (2026-08-11 OOM 修复): 逐 symbol 流式累计 nan率+采样行值.
                columns, nan_rate, svals = generator.family_stats(
                    df, fam, sample_pos, raw_cols=raw_cols, dtype="float32"
                )
                for c in columns:
                    if c in BruteForceGenerator.EXCLUDE_COLS or c.startswith("label_"):
                        continue
                    cand_nan[c] = nan_rate[c]
                    sample_cols[c] = svals[c]

        valid = [c for c, rate in cand_nan.items() if rate < threshold]
        sample_frame = pd.DataFrame(sample_cols, index=sample_pos)
        order = pd.Series({c: ic_accum.get(c, 0.0) for c in valid}) if need_ic else None
        selected = dedup_l2(valid, sample_frame, dedup_thr, order=order)
        # [2026-08-12] 选择过大 OOM 修复 (main 每次 ~3506 → 注入/训练 OOM → 永远回退):
        # 保底含 FeatureEngine 基集 (=fallback, 已知过 OOS 门), 剩余预算按方差补最强
        # brute 变体 (保持 08-08 方差排序裁决, 非 IC 排序). OOS 门仍是最终验收.
        n_dedup = len(selected)
        max_feats = cfg.get("max_features", 350)
        if n_dedup > max_feats:
            from app.pipeline1.feature_engine_v35 import FeatureEngineV35

            base_feats = [c for c in FeatureEngineV35.feature_columns(df) if c in valid]
            vars_ = sample_frame[selected].var().sort_values(ascending=False)
            brute_only = [c for c in vars_.index if c not in set(base_feats)]
            budget = max(max_feats - len(base_feats), 0)
            selected = list(base_feats) + brute_only[:budget]
            logger.info(
                "DedupL2 选择过大 cap: %d -> %d (base=%d + brute=%d)",
                n_dedup,
                len(selected),
                len(base_feats),
                min(budget, len(brute_only)),
            )
        if metrics_out is not None:
            metrics_out.update(
                n_dedup=n_dedup,
                capped=n_dedup > max_feats,
                max_features=max_feats,
                dedup_threshold=dedup_thr,
            )
            if metrics_out["capped"]:
                metrics_out.update(
                    n_base=len(base_feats), n_brute=min(budget, len(brute_only))
                )
        return selected

    def _run_gate_d(self, df, board, cfg, metrics_out=None):
        from app.pipeline1.feature_engine_v35 import FeatureEngineV35

        all_feats = FeatureEngineV35.feature_columns(df)
        valid = nan_filter(all_feats, df, cfg.get("nan_threshold", 0.95))
        gcfg = cfg.get("gate_d", {})
        label = gcfg.get("label", "label_pm_1d_net")
        if label not in df.columns:
            label = "label_1d_net"
        return gate_d_ablation(
            valid,
            df,
            label_col=label,
            min_feats=gcfg.get("min_features", 30),
            sat_pct=gcfg.get("saturation_pct", 0.95),
            metrics_out=metrics_out,
        )

    # ── Versioning ──

    def _version_path(self, board, version_id=None):
        if version_id:
            return os.path.join(
                self.registry_dir, f"selected_{board}_{version_id}.json"
            )
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        return os.path.join(self.registry_dir, f"selected_{board}_{ts}.json")

    def _current_path(self, board):
        return os.path.join(self.registry_dir, f"selected_{board}_current.json")

    def save_version(self, result, board, activate=False):
        """Save feature selection result as timestamped version."""
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        path = self._version_path(board, ts)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        if activate:
            current = {
                "active_version": f"selected_{board}_{ts}.json",
                "board": board,
                "updated_at": ts,
            }
            with open(self._current_path(board), "w", encoding="utf-8") as f:
                json.dump(current, f, indent=2, ensure_ascii=False)
        return path

    def load_current(self, board):
        """Load active feature list for a board."""
        cp = self._current_path(board)
        if not os.path.exists(cp):
            raise FileNotFoundError(
                f"No current version for {board}. Run select first."
            )
        with open(cp, encoding="utf-8") as f:
            current = json.load(f)
        vp = os.path.join(self.registry_dir, current["active_version"])
        with open(vp, encoding="utf-8") as f:
            return json.load(f)

    def load_version(self, board, version_id):
        """Load a specific version."""
        path = self._version_path(board, version_id)
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def list_versions(self, board):
        """List all versions for a board, sorted newest first."""
        prefix = f"selected_{board}_"
        files = [
            f
            for f in os.listdir(self.registry_dir)
            if f.startswith(prefix) and f.endswith(".json") and "current" not in f
        ]
        files.sort(reverse=True)
        return [f[len(prefix) : -5] for f in files]

    def get_status(self, board):
        """Return current version status."""
        cp = self._current_path(board)
        if not os.path.exists(cp):
            return {
                "board": board,
                "status": "no_current",
                "versions_available": len(self.list_versions(board)),
            }
        with open(cp, encoding="utf-8") as f:
            current = json.load(f)
        vp = os.path.join(self.registry_dir, current["active_version"])
        if not os.path.exists(vp):
            return {
                "board": board,
                "status": "broken_pointer",
                "points_to": current["active_version"],
            }
        with open(vp, encoding="utf-8") as f:
            ver = json.load(f)
        return {
            "board": board,
            "status": "active",
            "active_version": current["active_version"],
            "updated_at": current.get("updated_at", "unknown"),
            "pipeline": ver.get("pipeline", "unknown"),
            "pool_size": ver.get("pool_size", 0),
            "selected_count": ver.get("selected_count", 0),
        }

    def diff_versions(self, old_features, new_features):
        """Compare two feature lists."""
        old_set = set(old_features)
        new_set = set(new_features)
        added = sorted(new_set - old_set)
        removed = sorted(old_set - new_set)
        return {
            "added_count": len(added),
            "removed_count": len(removed),
            "net_change": len(new_set) - len(old_set),
            "sample_added": added[:5],
            "sample_removed": removed[:5],
        }

    def rollback(self, board, version_id):
        """Point current to a previous version."""
        path = self._version_path(board, version_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Version {version_id} not found")
        current = {
            "active_version": f"selected_{board}_{version_id}.json",
            "board": board,
            "updated_at": datetime.now().strftime("%Y%m%dT%H%M%S"),
            "rollback": True,
        }
        with open(self._current_path(board), "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2, ensure_ascii=False)
        return current
