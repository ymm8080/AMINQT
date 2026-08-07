#!/usr/bin/env python3
"""Enforce that AMINQT CODES/DATA contains only parquet datasets.

Allowed entries under DATA/:
  - *.parquet files
  - Python source files (*.py) required by the adapters package
  - .gitkeep directory placeholders

Everything else must be written to DATA_OTHERS_DIR (D:/AMINQT/DATA OTHERS).

Usage:
    python scripts/check_data_only_parquet.py              # scan and report
    python scripts/check_data_only_parquet.py --ci         # exit 1 on violations
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Files/dirs that are allowed even though they are not parquet.
ALLOWED_SUFFIXES = {".parquet", ".py"}
ALLOWED_NAMES = {".gitkeep"}
IGNORED_DIR_NAMES = {"__pycache__"}


def find_violations(root: Path) -> list[Path]:
    """Return paths under root that violate the parquet-only rule."""
    violations: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in ALLOWED_SUFFIXES:
            continue
        if path.name in ALLOWED_NAMES:
            continue
        if IGNORED_DIR_NAMES.intersection(path.parts):
            continue
        violations.append(path)
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ensure DATA/ only contains parquet datasets."
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Exit with non-zero status if violations are found.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Directory to scan (default: project root data/)",
    )
    args = parser.parse_args()

    root = args.data_dir.resolve()
    violations = find_violations(root)
    if not violations:
        print(f"OK: {root} contains only parquet datasets and source files.")
        return 0

    print(f"FAIL: {len(violations)} non-parquet file(s) found under {root}:")
    for v in sorted(violations):
        print(f"  {v.relative_to(root)}")
    print("\nMove these files to DATA_OTHERS_DIR or update the writer.")
    return 1 if args.ci else 0


if __name__ == "__main__":
    sys.exit(main())
