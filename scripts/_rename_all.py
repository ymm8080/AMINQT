# -*- coding: utf-8 -*-
"""Replace benefit_part -> winner_ratio in all Python files.
Also change winner_rate/100 -> winner_rate (no division) where applicable.
"""
import pathlib

ROOT = pathlib.Path(".")

# Core files to update
CORE_FILES = [
    "_daily_fetch.py",
    "app/pipeline1/feature_engine_v35.py",
    "app/pipeline1/cyq_calculator.py",
]

# Scripts to update (less critical but keep consistent)
SCRIPT_FILES = [
    "scripts/data_fetch_pipeline.py",
    "scripts/enrich_v3_from_baostock.py",
    "scripts/eval_dim21_29.py",
    "scripts/fix_benefit_part.py",
    "scripts/rebuild_full_panel.py",
    "scripts/_check_cache_sources.py",
    "scripts/_compare_cyq.py",
    "scripts/_consolidate_xy_cols.py",
    "scripts/_fetch_cyq_tushare.py",
    "scripts/_fetch_cyq.py",
    "scripts/_fix_panel_schema.py",
    "scripts/_merge_cyq_to_v3.py",
    "scripts/_analyze_new_cols.py",
    "scripts/_analyze_nan_cols.py",
    "scripts/_verify_cols.py",
    "scripts/_verify_v3.py",
    "scripts/verify_cyq_akshare.py",
    "scripts/verify_cyq_after.py",
    "scripts/verify_600671.py",
    "scripts/v3_data_quality.py",
    "scripts/list_cyq.py",
    "scripts/ic_eval_direct.py",
    "scripts/fetch_cyq_remaining.py",
    "scripts/eval_dim21_29_full.py",
    "scripts/enrich_one_source.py",
    "scripts/assemble_enriched.py",
    "_ic_fast2.py",
    "_ic_eval_fast.py",
    "_eval_chg_targets.py",
]

all_files = CORE_FILES + SCRIPT_FILES
updated = 0

for fpath in all_files:
    p = ROOT / fpath
    if not p.exists():
        continue
    text = p.read_text(encoding="utf-8")
    if "benefit_part" not in text:
        continue
    
    # Global replace benefit_part -> winner_ratio
    text = text.replace("benefit_part", "winner_ratio")
    
    # Fix the /100 division: winner_ratio = winner_rate / 100.0 -> winner_ratio = winner_rate
    # This pattern appears in feature_engine_v35.py and _daily_fetch.py
    text = text.replace("df[\"winner_ratio\"] = df[\"winner_rate\"] / 100.0",
                        'df["winner_ratio"] = df["winner_rate"]')
    text = text.replace("df[\"winner_ratio\"] = df[\"_winner_rate_tmp\"] / 100.0",
                        'df["winner_ratio"] = df["winner_rate"]')
    
    # Fix rename_map: remove winner_rate -> _winner_rate_tmp (no longer needed)
    text = text.replace('"winner_rate": "_winner_rate_tmp",\n    ', '')
    text = text.replace('"winner_rate": "_winner_rate_tmp",', '')
    
    # Remove _winner_rate_tmp computation block
    text = text.replace(
        '# --- winner_ratio = winner_rate / 100 ---\nif "_winner_rate_tmp" in df.columns:\n    df["winner_ratio"] = df["winner_rate"]\n',
        '# --- winner_ratio = winner_rate (0-100, percentage) ---\nif "winner_rate" in df.columns and "winner_ratio" in panel_cols:\n    df["winner_ratio"] = df["winner_rate"]\n')
    text = text.replace(
        '# --- winner_ratio = winner_rate / 100 ---\nif "_winner_rate_tmp" in df.columns:\n    df["winner_ratio"] = df[\'winner_rate\']\n',
        '# --- winner_ratio = winner_rate (0-100, percentage) ---\nif "winner_rate" in df.columns and "winner_ratio" in panel_cols:\n    df["winner_ratio"] = df["winner_rate"]\n')
    
    p.write_text(text, encoding="utf-8")
    count = text.count("winner_ratio")
    print(f"  {fpath}: {count} occurrences of winner_ratio")
    updated += 1

print(f"\nUpdated {updated} files")
