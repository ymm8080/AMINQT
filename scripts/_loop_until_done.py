# -*- coding: utf-8 -*-
"""Loop until V3 universe is complete."""
import subprocess
import sys
import time

def run(cmd, name):
    print(f"\n=== {name} ===")
    print(f"CMD: {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"FAIL: {name}")
        return False
    print(f"OK: {name}")
    return True

def main():
    round = 0
    while True:
        round += 1
        print(f"\n🎯 Round {round}")

        # 1) Fill gaps for all sources
        if not run("python scripts/_fill_gaps.py", "fill_gaps"):
            print("Re-run fill_gaps next round")
            time.sleep(60)
            continue

        # 2) Pull top_inst & fina
        if not run("python scripts/_pull_top_inst_and_fina.py both", "top_inst+fina"):
            print("Retry next round")
            time.sleep(60)
            continue

        # 3) Build final panel
        if not run("python scripts/_build_final_panel.py", "build_final_panel"):
            print("Build failed, re-pull sources")
            time.sleep(60)
            continue

        # 4) Quality check (full)
        if not run("python scripts/_quality_check_panel.py", "quality_check"):
            print("QC failed, investigate")
            time.sleep(120)
            continue

        # 5) Check coverage (simple)
        import pandas as pd
        panel = pd.read_parquet("data/new_symbols_raw/panel_final.parquet")
        n_dates = panel["trade_date"].nunique()
        n_symbols = panel["symbol"].nunique()
        print(f"Coverage: {n_dates} days, {n_symbols} symbols")
        if n_dates >= 875 and n_symbols >= 1770:  # ~99.9%
            print("🎉 ALL DONE! Final panel ready.")
            break
        else:
            print(f"Still missing: days={876-n_dates}, symbols={1780-n_symbols}")
            print("Continuing...")
            time.sleep(60)

    print("✅ Loop complete. V3 universe frozen.")

if __name__ == "__main__":
    main()