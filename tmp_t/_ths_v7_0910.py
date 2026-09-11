# -*- coding: utf-8 -*-
"""v7: 拷贝用户Edge profile → 真Edge原生解v20 → CDP读全部cookie"""
import os
import shutil
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

from playwright.sync_api import sync_playwright

SRC_ROOT = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\User Data")
TMP_PROFILE = os.path.join(tempfile.gettempdir(), "edge_ck_prof_0910")

# 最小拷贝集
items = [
    ("Local State", "Local State"),
    (r"Default\Network\Cookies", r"Default\Network\Cookies"),
    (r"Default\Preferences", r"Default\Preferences"),
]
if os.path.exists(TMP_PROFILE):
    shutil.rmtree(TMP_PROFILE, ignore_errors=True)
os.makedirs(os.path.join(TMP_PROFILE, "Default", "Network"), exist_ok=True)
for src_rel, dst_rel in items:
    s = os.path.join(SRC_ROOT, src_rel)
    d = os.path.join(TMP_PROFILE, dst_rel)
    try:
        shutil.copy2(s, d)
        print(f"[copy] {src_rel}: OK ({os.path.getsize(d)}B)")
    except Exception as e:
        print(f"[copy] {src_rel}: FAIL {e}")

OUT = r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_cookies_logged_0910.json"

with sync_playwright() as p:
    br = p.chromium.launch_persistent_context(
        TMP_PROFILE, channel="msedge", headless=True,
        args=["--no-first-run", "--disable-features=msEdgeBackslashTabCapture"])
    cookies = br.cookies()
    print(f"[read] profile内cookie总量: {len(cookies)}")
    ths = [c for c in cookies if ("iwencai" in c["domain"] or "10jqka" in c["domain"] or "thsi" in c["domain"])]
    print(f"[read] THS域: {len(ths)}条")
    logged = False
    for c in ths:
        v = c["value"]
        if c["name"] in ("userid", "user_id", "snuid") and v and not v.startswith("("):
            logged = True
        print(f"   {c['domain']:26s} {c['name']:16s} = {v[:24]}{'...' if len(v) > 24 else ''}")
    with open(OUT, "w", encoding="utf-8") as f:
        import json
        json.dump(cookies, f, ensure_ascii=False, indent=1)
    print(f"[verdict] 登录态: {'成功获取!' if logged else '未获取'}")
    print(f"[saved] {OUT}")
    br.close()
print("SCRIPT-END")
