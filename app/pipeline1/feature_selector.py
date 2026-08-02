"""
Three-Layer Feature Selection Module.

Layer1: BruteForceGenerator + registry building
Layer2: DedupL2 (MAIN), GateD (DUAL), versioning, save/load
"""

import json
import os
import re
import time
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import lightgbm as lgb

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
        "is_st",
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
                    o[w:] = (s[w:] - s[:-w]) / np.abs(s[:-w]) * 100
                    feats[f"{col}_brute_pct{w}"] = o

                for w in self.transforms.get("rolling_mean", {}).get("windows", ()):
                    feats[f"{col}_brute_ma{w}"] = (
                        pd.Series(s).rolling(w, min_periods=1).mean().values
                    )

                for w in self.transforms.get("rolling_std", {}).get("windows", ()):
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


# ──────────────────────────────────────────────────────────
# Dedup L2 (Layer2, MAIN)
# ──────────────────────────────────────────────────────────


def dedup_l2(feats, df, threshold=0.7):
    """Correlation dedup within same base column group.

    For brute-force features, base column is prefix before '_brute_'.
    For curated features, base column is dim prefix or raw col name.
    Keeps features greedily: sort by variance descending, drop if
    |r| > threshold with any already-kept feature in the same group.
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
    for base, cols in groups.items():
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

        # Sort by variance (proxy for importance), keep if |r| < threshold
        vars_ = sample.var().sort_values(ascending=False)
        dropped = set()
        ordered = [c for c in vars_.index if c in avail]
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
):
    """Importance forward ablation with saturation gate.

    1. Train full model on all feats, rank by gain importance
    2. Test ablation points (5,10,20,30,...,all,min_feats)
    3. Stop at 95% of best ICIR, clamped to min_feats
    """
    if len(feats) <= min_feats:
        return feats

    dates = sorted(df["date"].unique())
    split = int(len(dates) * 0.8)  # internal 80/20 for ablation
    tr = df[df["date"].isin(dates[:split])].dropna(subset=[label_col])
    te = df[df["date"].isin(dates[split:])].dropna(subset=[label_col])

    avail = [c for c in feats if c in df.columns]
    if len(avail) <= min_feats:
        return avail

    base_params = dict(
        n_estimators=300,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
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
        return float(round(a.mean() / a.std() if a.std() > 0 else 0, 4))

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
    return selected


# ──────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────


def nan_filter(feats, df, threshold=0.95):
    """Drop features with NaN rate >= threshold."""
    from pandas.api.types import is_numeric_dtype

    good = []
    for c in feats:
        if (
            c in df.columns
            and is_numeric_dtype(df[c])
            and df[c].isna().mean() < threshold
        ):
            good.append(c)
    logger.info(
        "NaN filter: %d -> %d (threshold=%.2f)", len(feats), len(good), threshold
    )
    return good


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

    def __init__(
        self, config=None, registry_dir=str(data_others_path("data/factor_registry"))
    ):
        self.config = config or self.DEFAULT_CONFIG
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)

    # ── Selection ──

    def select(self, df, board, generator=None):
        """Run feature selection for a board. Returns list of feature names."""
        board_cfg = self.config.get(board, self.config.get("fallback", {}))
        pipeline = board_cfg.get("pipeline", "ic_screener")

        if pipeline == "bruteforce_dedup":
            return self._run_bruteforce_dedup(df, board, board_cfg, generator)
        elif pipeline == "gate_d":
            return self._run_gate_d(df, board, board_cfg)
        else:
            from app.pipeline1.feature_engine_v35 import FeatureEngineV35

            return FeatureEngineV35.feature_columns(df)

    def _run_bruteforce_dedup(self, df, board, cfg, generator=None):
        if generator is None:
            generator = BruteForceGenerator()
        raw_cols = generator._eligible(df)
        new = generator.generate(df, raw_cols=raw_cols)
        df_exp = df.join(new)
        all_cands = df_exp.columns.tolist()
        # Also include original raw + curated columns that might exist
        all_feats = [
            c
            for c in all_cands
            if c not in BruteForceGenerator.EXCLUDE_COLS
            and not c.startswith("label_")
            and df_exp[c].dtype in ("float64", "int64")
        ]
        valid = nan_filter(all_feats, df_exp, cfg.get("nan_threshold", 0.95))
        selected = dedup_l2(valid, df_exp, cfg.get("dedup_threshold", 0.7))
        return selected

    def _run_gate_d(self, df, board, cfg):
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
