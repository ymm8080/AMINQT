#!/usr/bin/env python3
"""Show latest prediction data from predictions.db"""
import sqlite3
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config.settings import data_others_path  # noqa: E402

conn = sqlite3.connect(str(data_others_path("data/predictions.db")))
conn.row_factory = sqlite3.Row

# Latest runs
runs = conn.execute("SELECT * FROM prediction_runs ORDER BY date DESC LIMIT 5").fetchall()
print("=== Last 5 Prediction Runs ===")
for r in runs:
    print(f"  Date: {r['date']} | Stocks: {r['n_stocks']} | Schema: {r['schema_version']} | Created: {r['created_at']}")

if not runs:
    print("  No prediction runs found.")
    conn.close()
    exit()

print()
latest = runs[0]["date"]
print(f"=== Latest Prediction: {latest} (Top 30 by prob_up) ===")

stocks = conn.execute("""
    SELECT s.*, o.actual_ret_1d, o.actual_ret_3d, o.actual_ret_5d,
           o.direction_correct_1d, o.pred_error_1d
    FROM prediction_stocks s
    LEFT JOIN prediction_outcomes o ON s.date = o.date AND s.symbol = o.symbol
    WHERE s.date = ? ORDER BY s.prob_up DESC LIMIT 30
""", (latest,)).fetchall()

header = f"{'#':>3} {'Symbol':>10} {'Board':>6} {'prob_up':>8} {'pred_1d':>8} {'pred_3d':>8} {'pred_5d':>8} {'act_1d':>10} {'act_3d':>10} {'act_5d':>10} {'dir_ok':>6}"
print(header)
print("-" * len(header))

for i, s in enumerate(stocks, 1):
    dir_str = str(s["direction_correct_1d"]) if s["direction_correct_1d"] is not None else "N/A"
    p1 = f"{s['prob_up']:.4f}" if s["prob_up"] else "N/A"
    r1 = f"{s['pred_ret_1d']:.4f}" if s["pred_ret_1d"] is not None else "N/A"
    r3 = f"{s['pred_ret_3d']:.4f}" if s["pred_ret_3d"] is not None else "N/A"
    r5 = f"{s['pred_ret_5d']:.4f}" if s["pred_ret_5d"] is not None else "N/A"
    a1 = f"{s['actual_ret_1d']:.4f}" if s["actual_ret_1d"] is not None else "N/A"
    a3 = f"{s['actual_ret_3d']:.4f}" if s["actual_ret_3d"] is not None else "N/A"
    a5 = f"{s['actual_ret_5d']:.4f}" if s["actual_ret_5d"] is not None else "N/A"
    print(f"{i:>3} {s['symbol']:>10} {s['board']:>6} {p1:>8} {r1:>8} {r3:>8} {r5:>8} {a1:>10} {a3:>10} {a5:>10} {dir_str:>6}")

# Summary
print()
total = conn.execute("SELECT COUNT(*) FROM prediction_stocks WHERE date=?", (latest,)).fetchone()[0]
has_outcome = conn.execute("SELECT COUNT(*) FROM prediction_outcomes WHERE date=?", (latest,)).fetchone()[0]
print(f"Total predictions: {total} | Outcomes backfilled: {has_outcome}")

if has_outcome > 0:
    q = conn.execute("""
        SELECT AVG(CASE WHEN direction_correct_1d=1 THEN 1.0 ELSE 0.0 END) as dir_acc,
               AVG(pred_error_1d) as bias_1d,
               AVG(ABS(pred_error_1d)) as mae_1d
        FROM prediction_outcomes WHERE date=?
    """, (latest,)).fetchone()
    print(f"Direction Accuracy: {q[0]:.2%} | Bias (pred-actual): {q[1]:.6f} | MAE: {q[2]:.6f}")

conn.close()
