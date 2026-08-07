#!/usr/bin/env python3
# =============================================================================
# Script for chip weekly analysis
# =============================================================================

import numpy as np


def calculate_chip_ic(df, labels, min_obs=200, n_bins=5):
    """
    Calculate IC for chip features

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with chip data
    labels : list
        List of label column names
    min_obs : int
        Minimum number of observations
    n_bins : int
        Number of bins for grouping

    Returns
    -------
    dict
        IC results
    """
    results = {}

    for i in range(len(df)):
        if i < min_obs:
            continue

        g = df.iloc[: i + 1].groupby("symbol").last()

        for tname in df.columns:
            if tname in ["symbol", "date"] + labels:
                continue

            f = g[tname]
            if f.notna().sum() < min_obs:
                continue

            for lab in labels:
                label = g[lab]
                m = f.notna() & label.notna()
                if m.sum() < min_obs:
                    continue

                fr = f[m].astype(float).rank()
                lr = label[m].astype(float).rank()
                c = fr.corr(lr)
                if c == c:
                    if lab not in results:
                        results[lab] = {}
                    if tname not in results[lab]:
                        results[lab][tname] = []
                    results[lab][tname].append(c)

    return {
        lab: {
            t: (float(np.mean(v)) if v else np.nan)
            for t, v in results.get(lab, {}).items()
        }
        for lab in labels
    }
