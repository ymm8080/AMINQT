# -*- coding: utf-8 -*-
import json, time, sys
path = "data/new_symbols_raw/progress.json"
def main():
    try:
        with open(path) as f:
            d = json.load(f)
        for k, v in d.items():
            print(f"{k}={v}")
    except Exception as e:
        print(f"error: {e}")
if __name__ == "__main__":
    main()
    print(f"--- {time.time()}")