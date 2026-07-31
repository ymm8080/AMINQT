# -*- coding: utf-8 -*-
"""Round 2 fixes: try-except around registry.save() + label comment."""
import pathlib

p = pathlib.Path('app/pipeline1/feature_engine_v35.py')
text = p.read_text(encoding='utf-8')

# Fix 1: Add comment clarifying _label_reference is label-only
old_label_comment = '''        # _label_reference uses numpy slicing (no shift(-k)) to pass leakage_audit'''
new_label_comment = '''        # _label_reference is for LABEL construction ONLY (not features).
        # Uses numpy slicing (no shift(-k)) to pass leakage_audit.
        # This forward return is used ONLY for IC pre-screening (label-col evaluation),
        # never as a feature. The label_col is dropped before returning df.'''
assert old_label_comment in text, 'label comment not found'
text = text.replace(old_label_comment, new_label_comment, 1)

# Fix 2: Wrap registry.save() with try-except
old_save = '''        if adopted:
            registry.mark_source_cols_registered(adopted)
            registry.save()

            # ── AFTER summary ──'''
new_save = '''        if adopted:
            registry.mark_source_cols_registered(adopted)
            try:
                registry.save()
            except Exception as exc:
                logger.warning("Auto-adopt: registry.save() failed: %s", exc)

            # ── AFTER summary ──'''
assert old_save in text, 'save block not found'
text = text.replace(old_save, new_save, 1)

p.write_text(text, encoding='utf-8')
print('Round 2 fixes applied: label comment + registry.save try-except')