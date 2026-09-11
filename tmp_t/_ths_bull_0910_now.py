# -*- coding: utf-8 -*-
"""午间临时抓取: 仅看涨池 20260910 (轻量1-3请求, 避免与全速回填扫描争通道).
看跌池(28重请求)不在此抓 — 15:30 收盘后由刷新脚本统一抓终值."""
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:/AMINQT/AMINQT CODES/scripts")
import _fetch_ths_signal as f

DSTR = "20260910"
out_f = f.CACHE_DIR / f"bull_{DSTR}.parquet"
if out_f.exists():
    print(f"[skip] {out_f.name} 已存在 (WORM)")
    sys.exit(0)

cookies = json.load(open(f.COOKIES_F, encoding="utf-8"))
ck = {c["name"]: c["value"] for c in cookies
      if ("iwencai" in c["domain"] or "10jqka" in c["domain"])}
df = f.fetch_bull_pool(DSTR, ck)
df.to_parquet(out_f)
print(f"[ok] {DSTR} 看涨池 {len(df)}行 -> {out_f.name}")
print("SCRIPT-END")
