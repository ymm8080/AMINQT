# -*- coding: utf-8 -*-
"""v8: 真Edge + 真User Data目录 → 原生解v20 → CDP读全部cookie (Edge进程已杀, 无锁)"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

from playwright.sync_api import sync_playwright

REAL = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\User Data")
OUT = r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_cookies_logged_0910.json"

with sync_playwright() as p:
    br = p.chromium.launch_persistent_context(
        REAL, channel="msedge", headless=True, args=["--no-first-run", "--profile-directory=Default"])
    cookies = br.cookies()
    print(f"[read] cookie总量: {len(cookies)}")
    ths = [c for c in cookies if ("iwencai" in c["domain"] or "10jqka" in c["domain"] or "thsi" in c["domain"])]
    print(f"[read] THS域: {len(ths)}条")
    logged = False
    for c in ths:
        v = c["value"]
        if c["name"] in ("userid", "user_id", "snuid") and v:
            logged = True
        print(f"   {c['domain']:26s} {c['name']:16s} = {v[:24]}{'...' if len(v) > 24 else ''}")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(cookies, f, ensure_ascii=False, indent=1)
    print(f"[verdict] 登录态: {'成功获取!' if logged else '未获取'}")
    br.close()
print("SCRIPT-END")
