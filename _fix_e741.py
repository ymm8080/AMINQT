"""Rename standalone lowercase `l` to `low` to fix ruff E741 errors."""
import re

files = [
    r"d:\AMINQT\AMINQT CODES\app\core\intraday_factor_engine.py",
    r"d:\AMINQT\AMINQT CODES\app\core\ths_indicators.py",
    r"d:\AMINQT\AMINQT CODES\app\indicators\yimeng_dingdi.py",
    r"d:\AMINQT\AMINQT CODES\app\indicators\zhuli_lasheng.py",
    r"d:\AMINQT\AMINQT CODES\app\pipeline1\feature_engine_v35.py",
]

for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    # Count standalone `l` before replacement
    matches = list(re.finditer(r"\bl\b", content))
    if not matches:
        print(f"No standalone `l` found in {fpath}")
        continue

    new_content = re.sub(r"\bl\b", "low", content)

    with open(fpath, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Fixed {fpath}: {len(matches)} replacements")
