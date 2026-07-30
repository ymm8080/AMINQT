# -*- coding: utf-8 -*-
import sqlite3
import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. List tables in predictions.db
conn = sqlite3.connect('data/predictions.db')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print("=== predictions.db tables ===")
print(tables)

# 2. Check fina_indicator table
if 'fina_indicator' in tables:
    print("\n=== fina_indicator schema ===")
    cur.execute("PRAGMA table_info(fina_indicator)")
    for col in cur.fetchall():
        print(col)

    print("\n=== fina_indicator row count ===")
    cur.execute("SELECT COUNT(*) FROM fina_indicator")
    print(cur.fetchone()[0])

    print("\n=== fina_indicator date range ===")
    cur.execute("SELECT MIN(date), MAX(date) FROM fina_indicator")
    print(cur.fetchone())

    print("\n=== fina_indicator sample (first 3 rows) ===")
    df_fina = pd.read_sql("SELECT * FROM fina_indicator LIMIT 3", conn)
    print(df_fina.columns.tolist())
    print(df_fina.head())

# 3. Check panel_3y columns
if os.path.exists('data/panel_3y.parquet'):
    panel = pd.read_parquet('data/panel_3y.parquet')
    print(f"\n=== panel_3y shape: {panel.shape} ===")
    print(f"Columns ({len(panel.columns)}): {panel.columns.tolist()}")

conn.close()
