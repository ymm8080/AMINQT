# -*- coding: utf-8 -*-
"""登录v2: iframe感知 + 全frame找输入框 + 截图诊断 + 10分钟等待"""
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from playwright.sync_api import sync_playwright

CREDS = json.load(open(r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_creds_0910.json", encoding="utf-8"))
USER, PW = CREDS["user"], CREDS["pw"]
PROFILE = r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_pw_profile"
OUT_COOKIES = r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_cookies_0910.json"
SHOT = r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_login2_step{}.png"

RULES = ("MAP www.iwencai.com 111.4.248.121,MAP pass.10jqka.com.cn 111.4.248.124,"
         "MAP basic.10jqka.com.cn 111.4.248.121,MAP eq.10jqka.com.cn 121.12.127.72,"
         "MAP news.10jqka.com.cn 121.12.127.84,MAP comment.10jqka.com.cn 121.12.127.73,"
         "MAP data.10jqka.com.cn 111.4.248.126,MAP captcha.10jqka.com.cn 111.4.248.124,"
         "MAP stock.10jqka.com.cn 123.139.125.227")

with sync_playwright() as p:
    browser = p.chromium.launch_persistent_context(
        PROFILE, channel="msedge", headless=False,
        viewport={"width": 1280, "height": 850},
        args=["--host-resolver-rules=" + RULES, "--lang=zh-CN"])
    page = browser.pages[0] if browser.pages else browser.new_page()
    page.goto("https://www.iwencai.com/", timeout=40000, wait_until="domcontentloaded")
    print(f"[nav] {page.title()[:40]}")
    time.sleep(3)

    # 已登录检测 (persistent profile可能保留上次会话)
    def have_userid():
        return any(c["name"] in ("userid", "user_id", "snuid") for c in browser.cookies())

    if have_userid():
        print("[skip] profile已有会话cookie, 无需登录")
    else:
        try:
            page.locator("text=请登录").first.click(timeout=3000)
            print("[login] clicked 请登录")
        except Exception:
            try:
                page.locator("text=登录").first.click(timeout=3000)
                print("[login] clicked 登录")
            except Exception as e:
                print(f"[login] click FAIL {str(e)[:80]}")
        time.sleep(4)

        # 诊断: 打印所有page和frame URL
        print("[diag] pages:")
        for i, pg in enumerate(browser.pages):
            print(f"   page{i}: {pg.url[:90]}")
        print("[diag] frames of main page:")
        for fr in page.frames:
            print(f"   frame: {fr.url[:90]}")
        page.screenshot(path=SHOT.format(1))

        # 全frame找登录输入框
        def find_pw_frame():
            for fr in page.frames:
                try:
                    if fr.locator("input[type='password']").first.is_visible(timeout=800):
                        return fr
                except Exception:
                    continue
            return None

        tgt = None
        for attempt in range(12):
            tgt = find_pw_frame()
            if tgt:
                break
            # 也可能新开tab
            for pg in browser.pages[1:]:
                tgt = pg
                break
            time.sleep(1)
        print(f"[login] 输入框所在frame: {tgt.url[:80] if tgt else None}")
        if tgt:
            try:
                tgt.locator("input[type='password']").first.wait_for(state="visible", timeout=5000)
                # 账号框: password前面的text输入
                filled_u = filled_p = False
                for s in ["input[type='text']", "input[name*='user' i]", "input[placeholder*='账号']"]:
                    try:
                        loc = tgt.locator(s).first
                        if loc.is_visible(timeout=1200):
                            loc.fill(USER)
                            filled_u = True
                            break
                    except Exception:
                        continue
                tgt.locator("input[type='password']").first.fill(PW)
                filled_p = True
                print(f"[login] filled user={filled_u} pass={filled_p}")
                tgt.screenshot(path=SHOT.format(2))
                time.sleep(1)
                clicked_btn = False
                for s in ["button:has-text('登')", "input[value*='登']",
                          "a:has-text('登')", "span:has-text('登')", "[class*='login'] button"]:
                    try:
                        loc = tgt.locator(s).first
                        if loc.is_visible(timeout=1000):
                            loc.click()
                            clicked_btn = True
                            print(f"[login] submit via {s}")
                            break
                    except Exception:
                        continue
                if not clicked_btn:
                    print("[login] 自动点登录按钮失败, 请手动点击登录")
            except Exception as e:
                print(f"[login] fill FAIL {type(e).__name__}: {str(e)[:100]}")
        else:
            print("[login] 没找到输入框, 请在浏览器窗口手动完成登录")

    print("[wait] 轮询登录cookie(最多10分钟; 滑块请手动滑)...")
    t0 = time.time()
    ok = False
    while time.time() - t0 < 600:
        if have_userid():
            ok = True
            break
        time.sleep(3)
    print(f"[wait] userid cookie: {'OK' if ok else 'NOT-FOUND'}")
    time.sleep(2)

    cookies_all = browser.cookies()
    with open(OUT_COOKIES, "w", encoding="utf-8") as f:
        json.dump(cookies_all, f, ensure_ascii=False, indent=1)
    iwc = [c for c in cookies_all if "iwencai" in c["domain"] or "10jqka" in c["domain"]]
    print(f"[dump] {len(cookies_all)}条cookie (THS域{len(iwc)}条) -> {OUT_COOKIES}")
    for c in iwc:
        v = c["value"]
        print(f"   {c['domain']:22s} {c['name']}={v[:20]}{'...' if len(v) > 20 else ''}")
    page.screenshot(path=SHOT.format(3))
    browser.close()
print("SCRIPT-END")
