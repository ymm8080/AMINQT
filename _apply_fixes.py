# -*- coding: utf-8 -*-
"""Apply 4 fixes to feature_engine_v35.py for PR #49 code review."""
import pathlib

p = pathlib.Path('app/pipeline1/feature_engine_v35.py')
text = p.read_text(encoding='utf-8')

# 1. Add _label_reference import
old_import = 'from .cleaning_pipeline import get_limit_pct\n'
new_import = 'from .cleaning_pipeline import get_limit_pct\nfrom .label_engine import _label_reference\n'
assert old_import in text, 'import line not found'
text = text.replace(old_import, new_import, 1)

# 2. Replace shift(-1) with _label_reference
old_shift = '''        # ── IC/IR Gate pre-screen (Gate A) ──
        # Compute 1-day forward return label (use close as reference)
        label_col = "_fwd_ret_1d_ic"
        df_sorted = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        df_sorted[label_col] = df_sorted.groupby("symbol")["close"].transform(
            lambda s: s.shift(-1) / s - 1
        )'''
new_shift = '''        # ── IC/IR Gate pre-screen (Gate A) ──
        # Compute 1-day forward return label (use close as reference)
        # _label_reference uses numpy slicing (no shift(-k)) to pass leakage_audit
        label_col = "_fwd_ret_1d_ic"
        df_sorted = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        future_close = df_sorted.groupby("symbol")["close"].transform(
            lambda s: _label_reference(s, 1)
        )
        df_sorted[label_col] = _safe_divide(future_close, df_sorted["close"]) - 1'''
assert old_shift in text, 'shift(-1) block not found'
text = text.replace(old_shift, new_shift, 1)

# 3. Replace _per_stock_zscore division
old_zscore = '            g[zscore_col] = (s - mu) / sd.replace(0, np.nan)'
new_zscore = '            g[zscore_col] = _safe_divide(s - mu, sd)'
assert old_zscore in text, 'zscore division not found'
text = text.replace(old_zscore, new_zscore, 1)

# 4. Replace _per_stock_vol_adj division
old_vol = '                v_z = (v - v_mu) / v_sd.replace(0, 1.0)'
new_vol = '                v_z = _safe_divide(v - v_mu, v_sd)'
assert old_vol in text, 'vol_adj division not found'
text = text.replace(old_vol, new_vol, 1)

p.write_text(text, encoding='utf-8')
print('All 4 edits applied successfully')
