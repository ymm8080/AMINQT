# -*- coding: utf-8 -*-
"""自动登录问财: playwright+Edge(系统自带) + DoH定向解析 + cookie导出
滑块验证出现时窗口留在屏幕上等用户手动滑, 脚本轮询登录完成
"""
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
import requests

# ---- DoH 定向解析表 (绕开本机坏DNS) ----
DOMAINS = ["www.iwencai.com", "pass.10jqka.com.cn", "basic.10jqka.com.cn",
           "eq.10jqka.com.cn", "news.10jqka.com.cn", "comment.10jqka.com.cn",
           "data.10jqka.com.cn", "stock.10jqka.com.cn", "zc.10jqka.com.cn",
           "captcha.10jqka.com.cn", "files.10jqka.com.cn", "upload.10jqka.com.cn"]
rules = []
for d in DOMAINS:
    try:
        r = requests.get("https://1.12.12.12/resolve", params={"name": d, "type": "A"},
                         timeout=8, headers={"Accept": "application/dns-json"})
        ips = [a["data"] for a in r.json().get("Answer", []) if a.get("type") == 1 and a["data"] != "43.248.130.133"]
        if ips:
            rules.append(f"MAP {d} {ips[0]}")
            print(f"[dns] {d} -> {ips[0]}")
        else:
            print(f"[dns] {d} -> 无可用IP(跳过)")
    except Exception as e:
        print(f"[dns] {d} FAIL {str(e)[:60]}")

from playwright.sync_api import sync_playwright

CREDS = json.load(open(r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_creds_0910.json", encoding="utf-8"))
USER, PW = CREDS["user"], CREDS["pw"]

PROFILE = r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_pw_profile"
OUT_COOKIES = r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_cookies_0910.json"

with sync_playwright() as p:
    browser = p.chromium.launch_persistent_context(
        PROFILE,
        channel="msedge",
        headless=False,
        viewport={"width": 1280, "height": 850},
        args=["--host-resolver-rules=" + ",".join(rules), "--lang=zh-CN"],
    )
    page = browser.pages[0] if browser.pages else browser.new_page()
    page.goto("https://www.iwencai.com/", timeout=40000, wait_until="domcontentloaded")
    print(f"[nav] iwencai主页 title={page.title()[:50]}")
    time.sleep(3)
    page.screenshot(path=r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_login_step1.png")

    # 点登录 (主页可能有多个"登录", 取第一个可见的)
    clicked = False
    for sel in ["text=请登录", "text=登录"]:
        try:
            loc = page.locator(sel).first
            if loc.is_visible(timeout=2000):
                loc.click()
                clicked = True
                print(f"[login] clicked: {sel}")
                break
        except Exception:
            continue
    if not clicked:
        print("[login] 主页未找到登录入口, 可能已登录或需人工")
    time.sleep(4)

    # 登录可能在弹窗/新页面
    target = None
    for _ in range(10):
        for pg in browser.pages:
            u = pg.url
            if "pass.10jqka" in u or "login" in u.lower():
                target = pg
                break
        if target:
            break
        time.sleep(1)
    if target is None:
        target = page
        print("[login] 未检测到登录页跳转, 在主页上下文继续尝试")
    else:
        print(f"[login] 登录页: {target.url[:80]}")
    time.sleep(2)
    target.screenshot(path=r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_login_step2.png")

    # 填表 (多候选选择器)
    def try_fill(pg, sels, val):
        for s in sels:
            try:
                loc = pg.locator(s).first
                if loc.is_visible(timeout=1500):
                    loc.fill(val)
                    return s
            except Exception:
                continue
        return None

    # 若默认是扫码页, 尝试点"账号登录"tab
    for tab in ["text=账号登录", "text=帐号登录", "text=密码登录", "text=账号密码登录"]:
        try:
            loc = target.locator(tab).first
            if loc.is_visible(timeout=1000):
                loc.click()
                print(f"[login] tab: {tab}")
                time.sleep(1.5)
                break
        except Exception:
            continue

    us = try_fill(target, ["input[name='login_username']", "input[name='username']",
                           "input[placeholder*='账号']", "input[placeholder*='用户名']",
                           "input[type='text']"], USER)
    ps = try_fill(target, ["input[name='login_password']", "input[name='password']",
                           "input[placeholder*='密码']", "input[type='password']"], PW)
    print(f"[login] fill user={us}, pass={ps}")
    target.screenshot(path=r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_login_step3.png")

    if us and ps:
        for bsel in ["button:has-text('登录')", "input[value='登录']",
                     "a:has-text('登录')", "#login_button", "text=登 录"]:
            try:
                bloc = target.locator(bsel).first
                if bloc.is_visible(timeout=1500):
                    bloc.click()
                    print(f"[login] submit: {bsel}")
                    break
            except Exception:
                continue
    else:
        print("[login] 自动填表失败, 请在弹出的浏览器窗口手动输入账号密码点登录")

    # 轮询登录完成: .iwencai.com或.10jqka出现userid/会话cookie
    print("[wait] 轮询登录完成(如出现滑块请在浏览器窗口手动完成, 最多等5分钟)...")
    cookies_all = []
    ok = False
    t0 = time.time()
    while time.time() - t0 < 300:
        cookies_all = browser.cookies()
        names = {c["name"] for c in cookies_all}
        if {"userid", "user_id", "snuid"} & names:
            ok = True
            break
        time.sleep(3)
    if ok:
        print("[done] 登录成功! 检测到会话cookie")
    else:
        print("[warn] 5分钟内未检测到userid cookie, 导出当前全部cookie继续(可能已登录但cookie名不同)")

    with open(OUT_COOKIES, "w", encoding="utf-8") as f:
        json.dump(cookies_all, f, ensure_ascii=False, indent=1)
    iwc = [c for c in cookies_all if "iwencai" in c["domain"] or "10jqka" in c["domain"]]
    print(f"[dump] {len(cookies_all)}条cookie -> {OUT_COOKIES} (THS域: {len(iwc)}条)")
    for c in iwc:
        print(f"   {c['domain']:22s} {c['name']}={c['value'][:24]}{'...' if len(c['value'])>24 else ''}")

    time.sleep(2)
    browser.close()
print("SCRIPT-END")
