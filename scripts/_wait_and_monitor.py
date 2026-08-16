import time

while True:
    try:
        with open("data/new_symbols_raw/progress.json") as f:
            import json

            d = json.load(f)
        print(
            f"ps_daily={d.get('ps_daily', '')}, ps_daily_basic={d.get('ps_daily_basic', '')}"
        )
    except Exception as e:
        print(f"read progress: {e}")
    time.sleep(120)
    print("---", time.time())
