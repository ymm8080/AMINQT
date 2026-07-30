# -*- coding: utf-8 -*-
"""IC pre-screen module (Gate A) for auto-adoption.

This module handles forward return label construction and Spearman rank IC
evaluation. The forward return label is a LABEL (legitimate future reference),
NOT a feature. It is kept in a separate module from feature_engine_v35.py to
ensure clean separation between label construction and feature computation.

Uses _label_reference (numpy slicing, no shift(-k)) to pass leakage_audit.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .label_engine import _label_reference, _safe_divide

logger = logging.getLogger(__name__)


def compute_forward_return_label(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Compute forward return label for IC pre-screening (LABEL construction ONLY).

    This is NOT a feature -- it is a temporary label used for IC/IR evaluation
    in auto-adoption, then dropped. Uses _label_reference (numpy slicing,
    no shift(-k)) to pass leakage_audit.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with 'symbol', 'date', 'close' columns.
    horizon : int
        Forward return horizon in trading days (default=1).

    Returns
    -------
    pd.Series
        Forward return series aligned to df index.
    """
    future_close = df.groupby("symbol")["close"].transform(
        lambda s, h=horizon: _label_reference(s, h)
    )
    return _safe_divide(future_close, df["close"]) - 1


def quick_ic_check(
    df: pd.DataFrame, col: str, label_col: str
) -> tuple[float, float, int]:
    """Compute mean IC, ICIR, and number of valid trading days.

    Groups by date, computes Spearman Rank IC against label_col per day,
    then aggregates across days.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with date, col, and label_col columns.
    col : str
        Feature column name to evaluate.
    label_col : str
        Label column name (forward return).

    Returns
    -------
    ic_mean : float
        Mean of daily rank IC values (signed).
    icir : float
        IC_mean / IC_std (using sample std, ddof=1).
    n_days : int
        Number of trading days with >= 10 valid observations.
    """
    daily_ics: list[float] = []
    for _date_val, grp in df.groupby("date"):
        valid = grp[[col, label_col]].dropna()
        if len(valid) < 10:
            continue
        try:
            ic, _ = spearmanr(valid[col], valid[label_col])
        except Exception:
            continue
        if np.isnan(ic):
            continue
        daily_ics.append(ic)

    n_days = len(daily_ics)
    if n_days < 20:
        return 0.0, 0.0, n_days

    ic_arr = np.array(daily_ics)
    ic_mean = ic_arr.mean()
    ic_std = ic_arr.std(ddof=1)
    if ic_std == 0.0:
        return 0.0, 0.0, n_days

    return ic_mean, ic_mean / ic_std if ic_std > 0 else 0.0, n_days


def prescreen_columns(
    df: pd.DataFrame,
    adoptable: list[str],
    ic_min: float,
    icir_min: float,
) -> tuple[list[str], dict[str, str]]:
    """Run IC/IR pre-screen on adoptable columns.

    Computes a 1-day forward return label (LABEL ONLY, not a feature),
    evaluates each column's IC/IR, and returns pass/fail lists.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with 'symbol', 'date', 'close' columns.
    adoptable : list[str]
        List of column names to screen.
        ic_min : float
        Minimum mean IC threshold (signed).
    icir_min : float
        Minimum ICIR threshold (signed).

    Returns
    -------
    screened_pass : list[str]
        Columns that passed the IC/IR gate.
    screened_fail : dict[str, str]
        Columns that failed, mapped to rejection reason.
    """
    label_col = "_fwd_ret_1d_ic"
    df_sorted = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df_sorted[label_col] = compute_forward_return_label(df_sorted, horizon=1)

    screened_pass: list[str] = []
    screened_fail: dict[str, str] = {}
    for col in adoptable:
        overlap = df_sorted[[col, label_col]].dropna()
        if len(overlap) == 0:
            screened_fail[col] = "no overlap with forward return label"
            continue
        overlap_nan = 1.0 - overlap[col].notna().mean()
        if overlap_nan > 0.5:
            screened_fail[col] = (
                f"insufficient overlap with label (source NaN={overlap_nan:.1%})"
            )
            continue

        ic_val, icir_val, n_days = quick_ic_check(df_sorted, col, label_col)
        if n_days < 20:
            screened_fail[col] = f"too few trading days ({n_days})"
            continue
        if ic_val < ic_min or icir_val < icir_min:
            screened_fail[col] = (
                f"IC/IR too weak (IC={ic_val:.4f}, ICIR={icir_val:.4f})"
            )
            continue
        screened_pass.append(col)

    logger.info(
        "Auto-Adopt IC Gate: %d/%d pass (IC>=%.2f & ICIR>=%.2f), %d fail",
        len(screened_pass),
        len(adoptable),
        ic_min,
        icir_min,
        len(screened_fail),
    )
    if screened_fail:
        for col, reason in sorted(screened_fail.items()):
            logger.info("Auto-Adopt IC Gate REJECT: %s -> %s", col, reason)

    return screened_pass, screened_fail
