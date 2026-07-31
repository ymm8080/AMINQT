#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
L2 Family Dedup — Feature Registry Deduplication Script (P0)
============================================================
Monthly offline script: within each dim_group, clusters highly correlated
(|r| > 0.7) active features and marks the weaker ones as substitutes
(active=False + substitute_for=<keep>).

Usage:
    python scripts/dedup_features.py
    python scripts/dedup_features.py --panel data/panel_full_enriched_v3.parquet
    python scripts/dedup_features.py --dry-run
"""

from __future__ import annotations

import argparse
import glob as glob_mod  # avoids shadowing the module with loop var
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Post-processing dims — structural, not signal features ──
SKIP_DIMS = {
    "_industry_neutralize",
    "_missingness_flags",
    "_time_series_changes",
    "_cross_sectional_ranks",
    "_auto_adopted",
}

DEFAULT_REGISTRY_PATH = "data/factor_registry/feature_registry.json"
REPORT_DIR = "data/factor_registry"

# Sampling params
N_STOCKS = 200
N_DAYS = 60
RANDOM_SEED = 42

# Correlation threshold
CORR_THRESHOLD = 0.7


# ── Helpers ─────────────────────────────────────────────────────────


def latest_panel(data_dir: str = "data") -> str | None:
    """Find the most recently modified .parquet file in ``data_dir``."""
    candidates = [
        f
        for f in glob_mod.glob(os.path.join(data_dir, "*.parquet"))
        if os.path.isfile(f)
    ]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def load_and_sample_panel(path: str) -> pd.DataFrame:
    """Load panel and sample *N_STOCKS* random stocks x *N_DAYS* most recent trading days."""
    logger.info("Loading panel: %s", path)
    df = pd.read_parquet(path)
    logger.info(
        "Panel: %d rows, %d cols, %d stocks, dates %s ~ %s",
        len(df),
        len(df.columns),
        df["symbol"].nunique(),
        df["date"].min(),
        df["date"].max(),
    )

    rng = np.random.RandomState(RANDOM_SEED)

    # Most recent N_DAYS trading days
    dates = sorted(df["date"].unique())
    recent_dates = dates[-N_DAYS:]
    df_recent = df[df["date"].isin(recent_dates)].copy()

    # Random subsample of stocks
    symbols = df_recent["symbol"].unique()
    if len(symbols) > N_STOCKS:
        picked = rng.choice(symbols, N_STOCKS, replace=False)
        df_sample = df_recent[df_recent["symbol"].isin(picked)]
    else:
        df_sample = df_recent

    logger.info(
        "Sampled: %d stocks x %d days = %d rows",
        df_sample["symbol"].nunique(),
        df_sample["date"].nunique(),
        len(df_sample),
    )
    return df_sample


def compute_spearman_matrix(
    df: pd.DataFrame, features: list[str]
) -> pd.DataFrame | None:
    """Compute Spearman correlation matrix (pairwise complete obs).

    Returns a square DataFrame indexed by feature name, or *None* if
    fewer than 2 features have sufficient non-NaN / non-constant data.
    """
    # Keep only features present in the panel
    available = [f for f in features if f in df.columns]
    if len(available) < 2:
        return None

    # Drop near-constant / all-NaN columns
    valid: list[str] = []
    for f in available:
        col = df[f]
        # Need enough non-NaN values and at least some variation
        if col.notna().sum() >= 30 and col.nunique() >= 5:
            valid.append(f)

    if len(valid) < 2:
        return None

    from scipy.stats import spearmanr

    data = df[valid].to_numpy()
    rho, _ = spearmanr(data, nan_policy="omit")
    return pd.DataFrame(rho, index=valid, columns=valid)


def _sort_score(fname: str, features_meta: dict[str, dict]) -> tuple[float, float]:
    """Return (primary_score, tiebreaker) for sorting features.

    Primary = ICIR.  If ICIR is missing/NaN, fall back to *ic_abs*
    as primary score.  If both are missing/NaN, score is 0.0 so the
    feature sorts to the end and becomes a substitute.

    The tiebreaker is always *ic_abs* so that when ICIR ties, higher
    |IC| wins.
    """
    meta = features_meta.get(fname, {})
    icir = meta.get("icir")
    ic_abs = meta.get("ic_abs", 0.0)

    # Sanitise — JSON may store None or NaN
    if ic_abs is None or (isinstance(ic_abs, float) and np.isnan(ic_abs)):
        ic_abs = 0.0

    if icir is None or (isinstance(icir, float) and np.isnan(icir)):
        # Fall back to ic_abs as primary score
        return (float(ic_abs), 0.0)

    return (float(icir), float(ic_abs))


def greedy_cluster(
    corr: pd.DataFrame,
    features_meta: dict[str, dict],
) -> dict[str, list[tuple[str, dict, float]]]:
    """Greedy clustering within a dim group.

    Sorts features by ICIR descending (fallback |IC|).  The top-ranked
    feature in each cluster is *kept*; all later features with
    ``|r| > CORR_THRESHOLD`` are marked *substitute*.

    Substitutes that have ``grade == "strong"`` are **never** marked —
    strong features are always kept.

    Returns ``{kept_name: [(sub_name, sub_meta, |r|), ...], ...}``
    """
    all_features = sorted(
        corr.columns,
        key=lambda fn: _sort_score(fn, features_meta),
        reverse=True,  # highest ICIR first
    )

    clusters: dict[str, list[tuple[str, dict, float]]] = {}
    substituted: set[str] = set()

    for i, f1 in enumerate(all_features):
        if f1 in substituted:
            continue
        # Strong features are never substituted, but they can still
        # be "keepers" for other features (unless the stronger one
        # came before them in the sorted list).
        # However, we still need to check: a strong feature should
        # never be marked as substitute, so we skip marking it
        # but still consider it as a potential keeper.

        cluster_members: list[tuple[str, dict, float]] = []

        for f2 in all_features[i + 1 :]:
            if f2 in substituted:
                continue
            r = corr.loc[f1, f2]
            if np.isnan(r):
                continue
            if abs(r) > CORR_THRESHOLD:
                meta2 = features_meta.get(f2, {})
                # Never mark grade="strong" as substitute
                if meta2.get("grade") == "strong":
                    continue
                cluster_members.append((f2, meta2, abs(r)))
                substituted.add(f2)

        if cluster_members:
            clusters[f1] = cluster_members

    return clusters


# ── Report formatting helpers ──────────────────────────────────────


def _icir_str(meta: dict) -> str:
    """Format ICIR or ic_abs as a readable string."""
    icir = meta.get("icir")
    if icir is not None and isinstance(icir, (int, float)) and not np.isnan(icir):
        return f"{icir:.4f}"
    ic_abs = meta.get("ic_abs")
    if ic_abs is not None and isinstance(ic_abs, (int, float)) and not np.isnan(ic_abs):
        return f"|IC|={ic_abs:.4f}"
    return "N/A"


def format_grade(meta: dict) -> str:
    """Return grade string with strong marker."""
    grade = meta.get("grade", "unknown")
    return f"**{grade}**" if grade == "strong" else grade


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L2 Family Dedup — Remove highly correlated features within dim groups",
    )
    parser.add_argument(
        "--panel",
        type=str,
        default=None,
        help="Path to panel parquet. Default: latest in data/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without modifying the registry — just print the report",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=DEFAULT_REGISTRY_PATH,
        help=f"Path to feature registry JSON (default: {DEFAULT_REGISTRY_PATH})",
    )
    args = parser.parse_args()

    # ── 1. Load registry ──────────────────────────────────────────
    registry_path = (
        os.path.abspath(args.registry)
        if not os.path.isabs(args.registry)
        else args.registry
    )
    if not os.path.exists(registry_path):
        logger.info(
            "Registry not found at %s — nothing to dedup. Exiting.", registry_path
        )
        return

    from app.pipeline1.feature_registry import FeatureRegistry

    registry = FeatureRegistry(registry_path)
    all_features = registry.get_all()
    if not all_features:
        logger.info("Registry is empty — nothing to dedup. Exiting.")
        return

    logger.info("Loaded registry: %d total features", len(all_features))

    # ── 2. Group active features by dim_group ────────────────────
    dim_features: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for name, meta in all_features.items():
        if not meta.get("active", True):
            continue
        dim_group = meta.get("dim_group", "unknown")
        if dim_group in SKIP_DIMS:
            continue
        dim_features[dim_group].append((name, meta))

    # Only dims with >= 3 active features qualify for dedup
    dim_features = {d: fl for d, fl in dim_features.items() if len(fl) >= 3}

    if not dim_features:
        logger.info("No dim groups with >= 3 active features found. Exiting.")
        return

    logger.info("Active dim groups with >= 3 features: %d", len(dim_features))
    for d, fl in sorted(dim_features.items()):
        logger.info("  %s: %d features", d, len(fl))

    # ── 3. Load panel data ───────────────────────────────────────
    panel_path = args.panel
    if panel_path is None:
        panel_path = latest_panel()
        if panel_path is None:
            logger.error("No panel parquet found in data/ — use --panel to specify.")
            return

    df_sample = load_and_sample_panel(panel_path)

    from app.pipeline1.feature_engine_v35 import FeatureEngineV35

    panel_feature_cols = set(FeatureEngineV35.feature_columns(df_sample))

    # ── 4 & 5. Per dim group: correlate + cluster ───────────────
    # {dim_group: {kept: [(sub_name, sub_meta, |r|), ...]}}
    all_clusters: dict[str, dict[str, list[tuple[str, dict, float]]]] = {}
    total_substituted = 0
    total_strong_protected = 0

    for dim_group, feats in sorted(dim_features.items()):
        feat_meta = {n: m for n, m in feats}

        # Only include features that exist in the panel
        available = [
            n for n in feat_meta if n in panel_feature_cols and n in df_sample.columns
        ]
        if len(available) < 3:
            logger.info(
                "  [%s] only %d features available in panel (<3), skipping",
                dim_group,
                len(available),
            )
            continue

        logger.info(
            "  [%s] computing Spearman correlation for %d features...",
            dim_group,
            len(available),
        )
        corr = compute_spearman_matrix(df_sample, available)
        if corr is None:
            logger.info(
                "  [%s] not enough valid data after filtering, skipping", dim_group
            )
            continue

        clusters = greedy_cluster(corr, feat_meta)
        if clusters:
            all_clusters[dim_group] = clusters
            n_sub = sum(len(v) for v in clusters.values())
            total_substituted += n_sub
            logger.info(
                "  [%s] %d cluster(s), %d feature(s) to substitute",
                dim_group,
                len(clusters),
                n_sub,
            )

            # ── 6. Update registry (skip on dry-run) ──
            if not args.dry_run:
                for kept, substitutes in clusters.items():
                    for sub_name, sub_meta, _ in substitutes:
                        # Track if a strong feature was "protected" (not marked)
                        # This shouldn't happen since we filter in greedy_cluster,
                        # but safeguard here too.
                        if sub_meta.get("grade") == "strong":
                            total_strong_protected += 1
                            logger.info(
                                "  [%s] PROTECTED strong feature: %s",
                                dim_group,
                                sub_name,
                            )
                            continue
                        sub_meta["active"] = False
                        sub_meta["substitute_for"] = kept
                        registry.register_new(sub_name, sub_meta)

    # ── 6b. Persist registry (if not dry-run) ────────────────────
    if not args.dry_run:
        if total_substituted > 0:
            registry.save()
            logger.info(
                "Registry saved: %d features deactivated as substitutes",
                total_substituted,
            )
        else:
            logger.info("No features to update — registry unchanged.")
    else:
        logger.info("DRY RUN — registry NOT modified.")

    # ── 7. Generate markdown report ──────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"dedup_report_{timestamp}.md")

    lines: list[str] = []
    lines.append("# L2 Family Dedup Report")
    lines.append("")
    lines.append(f"- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Registry:** {registry_path}")
    lines.append(f"- **Panel:** {panel_path}")
    lines.append(
        f"- **Mode:** {'DRY RUN (no changes applied)' if args.dry_run else 'LIVE'}"
    )
    lines.append(f"- **Correlation threshold:** |r| > {CORR_THRESHOLD}")
    lines.append("")

    total_before = sum(len(fl) for fl in dim_features.values())
    total_kept = sum(len(cl) for cl in all_clusters.values())

    for dim_group in sorted(dim_features):
        before = len(dim_features[dim_group])
        clusters = all_clusters.get(dim_group, {})
        kept = len(clusters)
        subbed = sum(len(v) for v in clusters.values())

        lines.append(f"## {dim_group}")
        lines.append("")
        lines.append(f"- **Active features before:** {before}")
        lines.append(f"- **Clusters formed:** {len(clusters)}")
        lines.append(f"- **Kept:** {kept}")
        lines.append(f"- **Substituted:** {subbed}")
        if before > 0:
            lines.append(f"- **Dim compression:** {subbed / before:.1%}")
        lines.append("")

        if not clusters:
            lines.append("_No high-correlation clusters found._")
            lines.append("")
            continue

        lines.append(
            "| Keeper | Grade(kept) | ICIR(kept) | Substitute | Grade(sub) | ICIR(sub) | |r| |"
        )
        lines.append(
            "|--------|-------------|------------|------------|------------|-----------|-----|"
        )

        for kept_name, substitutes in clusters.items():
            kept_meta = feat_meta.get(kept_name, {})
            kept_grade = format_grade(kept_meta)
            kept_icir = _icir_str(kept_meta)

            for sub_name, sub_meta, abs_r in substitutes:
                sub_grade = format_grade(sub_meta)
                sub_icir = _icir_str(sub_meta)
                lines.append(
                    f"| {kept_name} | {kept_grade} | {kept_icir} "
                    f"| {sub_name} | {sub_grade} | {sub_icir} | {abs_r:.2f} |"
                )
        lines.append("")

    # Summary
    lines.append("---")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- **Dim groups processed:** {len(all_clusters)} / {len(dim_features)}"
    )
    lines.append(f"- **Total active features (eligible dims):** {total_before}")
    lines.append(f"- **Features kept:** {total_kept}")
    lines.append(f"- **Features substituted:** {total_substituted}")
    if total_before > 0:
        ratio = total_substituted / total_before
        lines.append(
            f"- **Compression ratio:** {ratio:.1%} ({total_substituted} / {total_before})"
        )
    lines.append(f"- **Strong features protected:** {total_strong_protected}")
    lines.append("")

    if args.dry_run:
        lines.append("> **DRY RUN** — No changes were made to the registry.")
        lines.append("")

    report = "\n".join(lines)

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    logger.info("Report saved: %s", report_path)

    # Terminal summary
    print()
    print("=" * 72)
    print(f"  L2 Family Dedup {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"  Dim groups: {len(all_clusters)} / {len(dim_features)} with clusters")
    print(
        f"  Substituted: {total_substituted} / {total_before} features "
        f"({total_substituted / total_before:.1%})"
        if total_before > 0
        else "  No features to dedup."
    )
    print(f"  Report: {report_path}")
    if args.dry_run:
        print("  (registry NOT modified)")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
