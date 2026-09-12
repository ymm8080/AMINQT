# -*- coding: utf-8 -*-
"""登录v3: 枚举登录入口href + 直奔passport登录页 + 填表提交"""
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from playwright.sync_api import sync_playwright

CREDS = json.load(open(r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_creds_0910.json", encoding="utf-8"))
USER, PW = CREDS["user"], CREDS["pw"]
PROFILE = r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_pw_profile"
OUT_COOKIES = r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_cookies_0910.json"
SHOT = r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_login3_step{}.png"

RULES = ("MAP www.iwencai.com 111.4.248.121,MAP pass.10jqka.com.cn 111.4.248.124,"
         "MAP basic.10jqka.com.cn 111.4.248.121,MAP eq.10jqka.com.cn 121.12.127.72,"
         "MAP news.10jqka.com.cn 121.12.127.84,MAP comment.10jqka.com.cn 121.12.127.73,"
         "MAP data.10jqka.com.cn 111.4.248.126,MAP captcha.10jqka.com.cn 111.4.248.124,"
         "MAP stock.10jqka.com.cn 123.139.125.227")


def have_userid(br):
    return any(c["name"] in ("userid", "user_id", "snuid") for c in br.cookies())


with sync_playwright() as p:
    browser = p.chromium.launch_persistent_context(
        PROFILE, channel="msedge", headless=False,
        viewport={"width": 1280, "height": 850},
        args=["--host-resolver-rules=" + RULES, "--lang=zh-CN"])
    page = browser.pages[0] if browser.pages else browser.new_page()

    if have_userid(browser):
        print("[skip] 已有会话cookie")
    else:
        page.goto("https://www.iwencai.com/", timeout=40000, wait_until="domcontentloaded")
        time.sleep(3)
        # 枚举登录相关链接
        links = page.eval_on_selector_all(
            "a", "els => els.map(e => ({t: e.innerText.trim().slice(0,12), h: e.href}))"
                  ".filter(x => x.t.includes('登') || (x.h||'').includes('login') || (x.h||'').includes('pass'))")
        seen = set()
        for x in links:
            k = (x["t"], x["h"])
            if k not in seen:
                seen.add(k)
                print(f"[link] {x['t']!r} -> {x['h'][:100]}")

        # 找pass域登录链接直奔
        login_url = None
        for x in links:
            h = x.get("h") or ""
            if "pass.10jqka" in h and "login" in h:
                login_url = h
                break
        if login_url is None:
            login_url = "https://pass.10jqka.com.cn/login?return_to=https%3A%2F%2Fwww.iwencai.com%2F"
        print(f"[login] goto {login_url[:100]}")
        try:
            page.goto(login_url, timeout=40000, wait_until="domcontentloaded")
        except Exception as e:
            print(f"[login] goto FAIL {str(e)[:120]}")
        time.sleep(4)
        print(f"[login] now at: {page.url[:100]}")
        print("[diag] frames:")
        for fr in page.frames:
            print(f"   {fr.url[:90]}")
        page.screenshot(path=SHOT.format(1))

        # 全frame找输入框
        tgt = None
        for _ in range(10):
            for fr in page.frames:
                try:
                    if fr.locator("input[type='password']").first.is_visible(timeout=600):
                        tgt = fr
                        break
                except Exception:
                    continue
            if tgt:
                break
            time.sleep(1)
        print(f"[login] pw-frame: {tgt.url[:80] if tgt else None}")
        if tgt:
            try:
                # 账号tab
                for tab in ["text=账号登录", "text=帐号登录", "text=密码登录", "text=账号密码登录"]:
                    try:
                        tl = tgt.locator(tab).first
                        if tl.is_visible(timeout=800):
                            tl.click()
                            print(f"[login] tab {tab}")
                            time.sleep(1.2)
                            break
                    except Exception:
                        continue
                ok_u = ok_p = False
                for s in ["input[name*='user' i]", "input[type='text']",
                          "input[placeholder*='账号']", "input[placeholder*='用户名']",
                          "input[placeholder*='手机']"]:
                    try:
                        loc = tgt.locator(s).first
                        if loc.is_visible(timeout=1000):
                            loc.fill(USER)
                            ok_u = True
                            print(f"[login] user via {s}")
                            break
                    except Exception:
                        continue
                for s in ["input[type='password']"]:
                    try:
                        tgt.locator(s).first.fill(PW, timeout=3000)
                        ok_p = True
                        print(f"[login] pass via {s}")
                        break
                    except Exception:
                        continue
                tgt.screenshot(path=SHOT.format(2))
                if ok_u and ok_p:
                    time.sleep(1)
                    for s in ["button:has-text('登')", "input[value*='登']",
                              "a:has-text('登')", "[class*='btn'] :text('登')", ":text('登 录')"]:
                        try:
                            loc = tgt.locator(s).first
                            if loc.is_visible(timeout=900):
                                loc.click()
                                print(f"[login] submit via {s}")
                                break
                        except Exception:
                            continue
            except Exception as e:
                print(f"[login] fill/submit FAIL {type(e).__name__}: {str(e)[:120]}")
        else:
            print("[login] 仍未找到输入框, 请在浏览器窗口手动完成登录(账号:敏敏5us)")

    print("[wait] 轮询userid cookie(最多10分钟; 出现滑块请手动滑)...")
    t0 = time.time()
    ok = False
    while time.time() - t0 < 600:
        if have_userid(browser):
            ok = True
            break
        time.sleep(3)
    print(f"[wait] 登录{'成功' if ok else '未检测到'}")
    time.sleep(2)
    page.screenshot(path=SHOT.format(3))

    cookies_all = browser.cookies()
    with open(OUT_COOKIES, "w", encoding="utf-8") as f:
        json.dump(cookies_all, f, ensure_ascii=False, indent=1)
    iwc = [c for c in cookies_all if "iwencai" in c["domain"] or "10jqka" in c["domain"]]
    print(f"[dump] {len(cookies_all)}条cookie (THS域{len(iwc)}条)")
    for c in iwc:
        v = c["value"]
        print(f"   {c['domain']:22s} {c['name']}={v[:20]}{'...' if len(v) > 20 else ''}")
    browser.close()
print("SCRIPT-END")
