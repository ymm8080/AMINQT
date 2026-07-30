# -*- coding: utf-8 -*-
"""Remove compute_forward_return_label from label_engine.py (now in ic_prescreen.py)."""
import pathlib

p = pathlib.Path('app/pipeline1/label_engine.py')
text = p.read_text(encoding='utf-8')

old_func = '''def compute_forward_return_label(df: pd.DataFrame, horizon: int = 1) -> pd.Series:
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
    return _safe_divide(future_close, df["close"]) - 1


def slippage_tier'''

new_func = '''def slippage_tier'''

assert old_func in text, 'compute_forward_return_label not found in label_engine'
text = text.replace(old_func, new_func, 1)
p.write_text(text, encoding='utf-8')
print('Removed compute_forward_return_label from label_engine.py')
