# -*- coding: utf-8 -*-
"""Round 4: Replace inline IC pre-screen with ic_prescreen module calls."""
import pathlib

p = pathlib.Path('app/pipeline1/feature_engine_v35.py')
text = p.read_text(encoding='utf-8')

# 1. Replace import: remove compute_forward_return_label, add ic_prescreen
old_import = 'from .label_engine import compute_forward_return_label'
new_import = 'from .ic_prescreen import prescreen_columns'
assert old_import in text, 'import not found'
text = text.replace(old_import, new_import, 1)

# 2. Remove _quick_ic_check static method entirely (moved to ic_prescreen)
old_quick_ic = '''    # ---------------- IC/IR Gate (Gate A) — quick pre-screen ----------------
    @staticmethod
    def _quick_ic_check(
        df: pd.DataFrame, col: str, label_col: str
    ) -> tuple[float, float, int]:
        """Compute |mean IC|, ICIR, and number of valid trading days.

        Groups by date, computes Spearman Rank IC against label_col per day,
        then aggregates across days.

        Returns
        -------
        abs_mean_ic : float
            Absolute mean of daily rank IC values.
        icir : float
            |IC_mean| / IC_std (using sample std, ddof=1).
        n_days : int
            Number of trading days with >= 10 valid observations.
        """
        daily_ics: list[float] = []
        for date_val, grp in df.groupby("date"):
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

        return ic_mean, ic_mean / ic_std, n_days

    # ---------------- Auto-Adoption (Phase 2) ----------------'''

new_quick_ic = '''    # ---------------- Auto-Adoption (Phase 2) ----------------'''

assert old_quick_ic in text, '_quick_ic_check method not found'
text = text.replace(old_quick_ic, new_quick_ic, 1)

# 3. Replace the entire IC pre-screen block with a call to prescreen_columns
old_block = '''        # ── IC/IR Gate pre-screen (Gate A) ──
        # Forward return label computed via label_engine (LABEL construction ONLY,
        # not a feature). Dropped before returning df. Delegated to label_engine
        # to keep feature_engine free of future-reference calls.
        label_col = "_fwd_ret_1d_ic"
        df_sorted = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        df_sorted[label_col] = compute_forward_return_label(df_sorted, horizon=1)

        screened_pass: list[str] = []
        screened_fail: dict[str, str] = {}
        for col in adoptable:
            # Check data overlap with the forward return label
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

            abs_ic, icir_val, n_days = self._quick_ic_check(
                df_sorted,
                col,
                label_col,
            )
            if n_days < 20:
                screened_fail[col] = f"too few trading days ({n_days})"
                continue
            if abs_ic < self._ADOPTION_IC_MIN or icir_val < self._ADOPTION_ICIR_MIN:
                screened_fail[col] = (
                    f"IC/IR too weak (|IC|={abs_ic:.4f}, ICIR={icir_val:.4f})"
                )
                continue
            screened_pass.append(col)

        # Clean up temporary label column
        df_sorted.drop(columns=[label_col], inplace=True)
        df = df_sorted

        # Log IC gate summary
        logger.info(
            "Auto-Adopt IC Gate: %d/%d pass (|IC|>=%.2f & ICIR>=%.2f), %d fail",
            len(screened_pass),
            len(adoptable),
            self._ADOPTION_IC_MIN,
            self._ADOPTION_ICIR_MIN,
            len(screened_fail),
        )
        if screened_fail:
            for col, reason in sorted(screened_fail.items()):
                logger.info("Auto-Adopt IC Gate REJECT: %s → %s", col, reason)

        adoptable = screened_pass'''

new_block = '''        # ── IC/IR Gate pre-screen (Gate A) ──
        # Delegated to ic_prescreen module: forward return label construction
        # and Spearman IC evaluation are LABEL operations (not features),
        # kept in a separate module for clean separation.
        adoptable, _screened_fail = prescreen_columns(
            df, adoptable, self._ADOPTION_IC_MIN, self._ADOPTION_ICIR_MIN
        )'''

assert old_block in text, 'IC pre-screen block not found'
text = text.replace(old_block, new_block, 1)

# 4. Remove unused spearmanr import (no longer used in feature_engine)
old_spearman_import = 'from scipy.stats import spearmanr\n'
new_spearman_import = ''
assert old_spearman_import in text, 'spearmanr import not found'
text = text.replace(old_spearman_import, new_spearman_import, 1)

p.write_text(text, encoding='utf-8')
print('Round 4: Extracted IC pre-screen to ic_prescreen.py module')
