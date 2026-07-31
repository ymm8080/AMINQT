# -*- coding: utf-8 -*-
"""Run leakage_audit on feature_engine_v35.py."""
import sys
sys.path.insert(0, ".")
from app.pipeline1.leakage_audit import audit_source

with open("app/pipeline1/feature_engine_v35.py", encoding="utf-8") as f:
    source = f.read()

violations = audit_source(source, "feature_engine_v35.py")
print(f"Violations: {len(violations)}")
for v in violations:
    print(f"  L{v['line']}: {v['pattern']} -- {v['line_text'].strip()}")
