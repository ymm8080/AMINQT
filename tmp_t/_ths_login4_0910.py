# -*- coding: utf-8 -*-
"""登录v4: pass域根探测 + /screener自动弹登录框 + 网络监听抓真实登录端点"""
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
import requests
import urllib3.util.connection as u3c

# pass域固定解析
_orig = u3c.create_connection


def fixed(address, *a, **kw):
    host, port = address
    if host == "pass.10jqka.com.cn":
        address = ("111.4.248.124", port)
    elif host == "www.iwencai.com":
        address = ("111.4.248.121", port)
    return _orig(address, *a, **kw)


u3c.create_connection = fixed

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36"}

print("== pass.10jqka.com.cn 根路径探测")
try:
    r = requests.get("https://pass.10jqka.com.cn/", headers=UA, timeout=15, allow_redirects=False)
    print(f"   / status={r.status_code} loc={r.headers.get('location')} len={len(r.text)}")
    print(f"   body[:250]: {r.text[:250]!r}")
except Exception as e:
    print(f"   FAIL {type(e).__name__}: {str(e)[:120]}")

from playwright.sync_api import sync_playwright

CREDS = json.load(open(r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_creds_0910.json", encoding="utf-8"))
USER, PW = CREDS["user"], CREDS["pw"]
PROFILE = r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_pw_profile"
OUT_COOKIES = r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_cookies_0910.json"
SHOT = r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_login4_step{}.png"

RULES = ("MAP www.iwencai.com 111.4.248.121,MAP pass.10jqka.com.cn 111.4.248.124,"
         "MAP basic.10jqka.com.cn 111.4.248.121,MAP eq.10jqka.com.cn 121.12.127.72,"
         "MAP news.10jqka.com.cn 121.12.127.84,MAP comment.10jqka.com.cn 121.12.127.73,"
         "MAP data.10jqka.com.cn 111.4.248.126,MAP captcha.10jqka.com.cn 111.4.248.124,"
         "MAP stock.10jqka.com.cn 123.139.125.227")

hits = []


def note_hit(kind, url):
    low = url.lower()
    if any(k in low for k in ["login", "pass.", "passport", "sso", "uum", "reg"]):
        hits.append((kind, url))
        print(f"   [net:{kind}] {url[:130]}")


with sync_playwright() as p:
    browser = p.chromium.launch_persistent_context(
        PROFILE, channel="msedge", headless=False,
        viewport={"width": 1280, "height": 850},
        args=["--host-resolver-rules=" + RULES, "--lang=zh-CN"])
    page = browser.pages[0] if browser.pages else browser.new_page()
    browser.on("page", lambda pg: print(f"   [popup] {pg.url[:120]}"))
    page.on("request", lambda req: note_hit("req", req.url))
    page.on("frameattached", lambda fr: print(f"   [frame+] {fr.url[:100]}"))
    page.on("framenavigated", lambda fr: note_hit("frame", fr.url))

    page.goto("https://www.iwencai.com/screener", timeout=45000, wait_until="domcontentloaded")
    print(f"[nav] {page.title()[:40]} {page.url[:60]}")

    def have_userid():
        return any(c["name"] in ("userid", "user_id", "snuid") for c in browser.cookies())

    if not have_userid():
        time.sleep(10)
        print("[diag] frames after 10s:")
        for fr in page.frames:
            print(f"   {fr.url[:100]}")
        page.screenshot(path=SHOT.format(1))

        # 全frame找密码框 (登录框可能延迟渲染)
        tgt = None
        for _ in range(15):
            for fr in page.frames:
                try:
                    if fr.locator("input[type='password']").first.is_visible(timeout=500):
                        tgt = fr
                        break
                except Exception:
                    continue
            if tgt:
                break
        print(f"[login] pw-frame: {tgt.url[:90] if tgt else 'NOT-FOUND'}")
        if tgt:
            try:
                for tab in ["text=账号登录", "text=帐号登录", "text=密码登录"]:
                    try:
                        tl = tgt.locator(tab).first
                        if tl.is_visible(timeout=700):
                            tl.click()
                            print(f"[login] tab {tab}")
                            time.sleep(1)
                            break
                    except Exception:
                        continue
                ok_u = ok_p = False
                for s in ["input[name*='user' i]", "input[type='text']",
                          "input[placeholder*='账号']", "input[placeholder*='手机']",
                          "input[placeholder*='用户']"]:
                    try:
                        loc = tgt.locator(s).first
                        if loc.is_visible(timeout=900):
                            loc.fill(USER)
                            ok_u = True
                            print(f"[login] user via {s}")
                            break
                    except Exception:
                        continue
                try:
                    tgt.locator("input[type='password']").first.fill(PW, timeout=3000)
                    ok_p = True
                    print("[login] pass filled")
                except Exception:
                    pass
                tgt.screenshot(path=SHOT.format(2))
                if ok_u and ok_p:
                    time.sleep(0.8)
                    for s in ["button:has-text('登')", "input[value*='登']",
                              "a:has-text('登')", ":text('登 录')"]:
                        try:
                            loc = tgt.locator(s).first
                            if loc.is_visible(timeout=800):
                                loc.click()
                                print(f"[login] submit via {s}")
                                break
                        except Exception:
                            continue
            except Exception as e:
                print(f"[login] fill FAIL {type(e).__name__}: {str(e)[:120]}")
        else:
            print("[login] 请在浏览器窗口手动点登录并输入账号密码(敏敏5us)")

    print("[wait] 轮询userid cookie(最多10分钟; 滑块请手动滑)...")
    t0 = time.time()
    ok = False
    while time.time() - t0 < 600:
        if have_userid():
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
        print(f"   {c['domain']:22s} {c['name']}")
    browser.close()
print("SCRIPT-END")
