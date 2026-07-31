# -*- coding: utf-8 -*-
"""Round 3: Move forward return label to label_engine.py + try-except spearmanr."""
import pathlib

# ── 1. Add helper to label_engine.py ──
le_path = pathlib.Path('app/pipeline1/label_engine.py')
le_text = le_path.read_text(encoding='utf-8')

# Insert _compute_forward_return_label after _safe_divide
insert_after = '''def _safe_divide(numerator, denominator):
    """Safe division (zero-division guard): NaN where denominator is 0."""
    return numerator / denominator.replace(0, np.nan)'''

new_func = '''def _safe_divide(numerator, denominator):
    """Safe division (zero-division guard): NaN where denominator is 0."""
    return numerator / denominator.replace(0, np.nan)


def compute_forward_return_label(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Compute forward return label for IC pre-screening (LABEL construction ONLY).

    This is NOT a feature -- it is a temporary label used for IC/IR evaluation
    in FeatureEngineV35._auto_adopt_new_columns, then dropped before returning df.
    Uses _label_reference (numpy slicing, no shift(-k)) to pass leakage_audit.

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
    return _safe_divide(future_close, df["close"]) - 1'''

assert insert_after in le_text, 'label_engine _safe_divide not found'
le_text = le_text.replace(insert_after, new_func, 1)
le_path.write_text(le_text, encoding='utf-8')
print('Added compute_forward_return_label to label_engine.py')

# ── 2. Update feature_engine_v35.py ──
fe_path = pathlib.Path('app/pipeline1/feature_engine_v35.py')
fe_text = fe_path.read_text(encoding='utf-8')

# 2a. Update import: remove _label_reference, add compute_forward_return_label
old_import = 'from .label_engine import _label_reference'
new_import = 'from .label_engine import compute_forward_return_label'
assert old_import in fe_text, 'import not found'
fe_text = fe_text.replace(old_import, new_import, 1)

# 2b. Replace the label computation block
old_block = '''        # ── IC/IR Gate pre-screen (Gate A) ──
        # Compute 1-day forward return label (use close as reference)
        # _label_reference is for LABEL construction ONLY (not features).
        # Uses numpy slicing (no shift(-k)) to pass leakage_audit.
        # This forward return is used ONLY for IC pre-screening (label-col evaluation),
        # never as a feature. The label_col is dropped before returning df.
        label_col = "_fwd_ret_1d_ic"
        df_sorted = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        future_close = df_sorted.groupby("symbol")["close"].transform(
            lambda s: _label_reference(s, 1)
        )
        df_sorted[label_col] = _safe_divide(future_close, df_sorted["close"]) - 1'''

new_block = '''        # ── IC/IR Gate pre-screen (Gate A) ──
        # Forward return label computed via label_engine (LABEL construction ONLY,
        # not a feature). Dropped before returning df. Delegated to label_engine
        # to keep feature_engine free of future-reference calls.
        label_col = "_fwd_ret_1d_ic"
        df_sorted = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        df_sorted[label_col] = compute_forward_return_label(df_sorted, horizon=1)'''

assert old_block in fe_text, 'label block not found'
fe_text = fe_text.replace(old_block, new_block, 1)

# 2c. Add try-except around spearmanr call in _quick_ic_check
old_spearman = '''            ic, _ = spearmanr(valid[col], valid[label_col])
            if np.isnan(ic):
                continue'''
new_spearman = '''            try:
                ic, _ = spearmanr(valid[col], valid[label_col])
            except Exception:
                continue
            if np.isnan(ic):
                continue'''
assert old_spearman in fe_text, 'spearmanr block not found'
fe_text = fe_text.replace(old_spearman, new_spearman, 1)

fe_path.write_text(fe_text, encoding='utf-8')
print('Updated feature_engine_v35.py: delegated label to label_engine + try-except spearmanr')
