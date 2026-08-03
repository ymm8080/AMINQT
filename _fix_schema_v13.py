# -*- coding: utf-8 -*-
"""Fix schema_version 1.2 -> 1.3 in tracked files."""
import pathlib

files_replacements = {
    "app/api/frontier_routes.py": [
        ('"schema_version": "1.2"', '"schema_version": "1.3"'),
    ],
    "app/pipeline1/prediction_db.py": [
        ("DEFAULT '1.2'", "DEFAULT '1.3'"),
        ('schema_version: str = "1.2"', 'schema_version: str = "1.3"'),
    ],
    "tests/test_frontier_api.py": [
        ('assert body["schema_version"] == "1.2"',
         'assert body["schema_version"] == "1.3"'),
    ],
    "tests/test_dashboard.py": [
        ('assert (df["schema_version"] == "1.2").all()',
         'assert (df["schema_version"] == "1.3").all()'),
    ],
}

for fpath, replacements in files_replacements.items():
    p = pathlib.Path(fpath)
    t = p.read_text(encoding="utf-8")
    for old, new in replacements:
        if old in t:
            t = t.replace(old, new)
            print(f"  Replaced in {fpath}: {old[:40]}...")
        else:
            print(f"  NOT FOUND in {fpath}: {old[:40]}...")
    p.write_text(t, encoding="utf-8")
    print(f"  Written: {fpath}")

print("Done.")
