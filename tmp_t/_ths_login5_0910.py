# -*- coding: utf-8 -*-
"""登录v5: 枚举一切'登'元素逐个JS点击, 点击后扫密码框, 找到即填表"""
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from playwright.sync_api import sync_playwright

CREDS = json.load(open(r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_creds_0910.json", encoding="utf-8"))
USER, PW = CREDS["user"], CREDS["pw"]
PROFILE = r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_pw_profile"
OUT_COOKIES = r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_cookies_0910.json"
SHOT = r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_login5_step{}.png"

RULES = ("MAP www.iwencai.com 111.4.248.121,MAP pass.10jqka.com.cn 111.4.248.124,"
         "MAP basic.10jqka.com.cn 111.4.248.121,MAP eq.10jqka.com.cn 121.12.127.72,"
         "MAP news.10jqka.com.cn 121.12.127.84,MAP comment.10jqka.com.cn 121.12.127.73,"
         "MAP data.10jqka.com.cn 111.4.248.126,MAP captcha.10jqka.com.cn 111.4.248.124,"
         "MAP stock.10jqka.com.cn 123.139.125.227,"
         "MAP upass.iwencai.com 112.90.89.70,MAP s.thsi.cn 140.207.66.159")


def have_userid(br):
    return any(c["name"] in ("userid", "user_id", "snuid") for c in br.cookies())


def find_pw_frame(page, timeout_s=3):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        for fr in page.frames:
            try:
                if fr.locator("input[type='password']").first.is_visible(timeout=400):
                    return fr
            except Exception:
                continue
        time.sleep(0.5)
    return None


ENUM_JS = """
() => Array.from(document.querySelectorAll('a,button,div,span,li,p'))
  .filter(e => {
    const t = (e.innerText || '').replace(/\\s/g, '');
    return t.length > 0 && t.length <= 4 && t.includes('登');
  })
  .slice(0, 25)
  .map(e => ({
    tag: e.tagName,
    cls: (e.className || '').toString().slice(0, 60),
    href: e.href || '',
    txt: (e.innerText || '').trim().slice(0, 10),
    vis: !!(e.offsetWidth || e.offsetHeight)
  }))
"""

with sync_playwright() as p:
    browser = p.chromium.launch_persistent_context(
        PROFILE, channel="msedge", headless=False,
        viewport={"width": 1280, "height": 850},
        args=["--host-resolver-rules=" + RULES, "--lang=zh-CN"])
    page = browser.pages[0] if browser.pages else browser.new_page()
    page.on("frameattached", lambda fr: print(f"   [frame+] {fr.url[:90]}"))
    page.on("framenavigated", lambda fr: print(f"   [frame>] {fr.url[:90]}"))

    if have_userid(browser):
        print("[skip] 已登录")
    else:
        page.goto("https://www.iwencai.com/", timeout=45000, wait_until="domcontentloaded")
        time.sleep(5)
        page.screenshot(path=SHOT.format("0_home"))

        cands = page.evaluate(ENUM_JS)
        print(f"[enum] {len(cands)}个含'登'元素:")
        seen = set()
        uniq = []
        for c in cands:
            k = (c["tag"], c["cls"], c["txt"], c["vis"])
            if k not in seen:
                seen.add(k)
                uniq.append(c)
                print(f"   {'V' if c['vis'] else 'H'} <{c['tag']}> class={c['cls']!r} href={c['href'][:60]!r} txt={c['txt']!r}")

        tgt = None
        tried = 0
        # v5实证: SPAN.info/login 点击才弹iframe; class含login的优先
        order = sorted(range(len(uniq)), key=lambda i: 0 if "login" in uniq[i]["cls"] else (1 if "info" in uniq[i]["cls"] else 2))
        for i in order:
            c = uniq[i]
            if not c["vis"]:
                continue
            if c["href"].rstrip("/") == "https://www.iwencai.com/screener":
                continue  # v2教训: 跳去screener的是导航项不是登录框
            nth = i + 1
            try:
                page.evaluate(
                    """(n) => {
                        const els = Array.from(document.querySelectorAll('a,button,div,span,li,p'))
                          .filter(e => { const t=(e.innerText||'').replace(/\\s/g,'');
                                         return t.length>0 && t.length<=4 && t.includes('登'); });
                        els[n-1].click();
                    }""", nth)
            except Exception as e:
                print(f"   click#{nth} FAIL {str(e)[:60]}")
                continue
            tried += 1
            print(f"[try#{tried}] clicked <{c['tag']}> {c['txt']!r} cls={c['cls'][:30]!r}")
            tgt = find_pw_frame(page, 20)
            if tgt:
                print(f"[hit] 密码框出现在frame: {tgt.url[:90] or '(inline)'}")
                page.screenshot(path=SHOT.format("1_modal"))
                break
            print(f"   frames after try#{tried}: " + " | ".join((fr.url or '(blank)')[:60] for fr in page.frames))
            page.screenshot(path=SHOT.format(f"1_try{tried}"))

        if tgt is None:
            # 兜底: 允许点击 /screener 项再试一次
            print("[fallback] 可见候选尽, 试 /screener 项")
            for i, c in enumerate(uniq):
                if not c["vis"] or c["href"].rstrip("/") != "https://www.iwencai.com/screener":
                    continue
                page.evaluate(
                    """(n) => { const els = Array.from(document.querySelectorAll('a,button,div,span,li,p'))
                        .filter(e => { const t=(e.innerText||'').replace(/\\s/g,'');
                                       return t.length>0 && t.length<=4 && t.includes('登'); });
                        els[n-1].click(); }""", i + 1)
                print(f"[try] clicked screener项 {c['txt']!r}")
                tgt = find_pw_frame(page, 3)
                if tgt:
                    break

        if tgt is None:
            print("[login] 仍未出现密码框 — 请在打开的浏览器窗口手动点登录输密码(敏敏5us)")
        else:
            try:
                for tab in ["text=账号登录", "text=帐号登录", "text=密码登录"]:
                    try:
                        tl = tgt.locator(tab).first
                        if tl.is_visible(timeout=600):
                            tl.click()
                            time.sleep(1)
                            break
                    except Exception:
                        continue
                ok_u = ok_p = False
                for s in ["input[name*='user' i]", "input[type='text']",
                          "input[placeholder*='账号']", "input[placeholder*='手机']"]:
                    try:
                        loc = tgt.locator(s).first
                        if loc.is_visible(timeout=800):
                            loc.fill(USER)
                            ok_u = True
                            break
                    except Exception:
                        continue
                try:
                    tgt.locator("input[type='password']").first.fill(PW, timeout=2500)
                    ok_p = True
                except Exception:
                    pass
                print(f"[fill] user={ok_u} pass={ok_p}")
                tgt.screenshot(path=SHOT.format("2_filled"))
                if ok_u and ok_p:
                    time.sleep(0.8)
                    for s in ["button:has-text('登')", "input[value*='登']", "a:has-text('登')"]:
                        try:
                            loc = tgt.locator(s).first
                            if loc.is_visible(timeout=700):
                                loc.click()
                                print(f"[submit] {s}")
                                break
                        except Exception:
                            continue
            except Exception as e:
                print(f"[fill] FAIL {type(e).__name__}: {str(e)[:100]}")

    print("[wait] 轮询userid cookie 10分钟 (滑块请手动滑)...")
    t0 = time.time()
    ok = False
    while time.time() - t0 < 600:
        if have_userid(browser):
            ok = True
            break
        time.sleep(3)
    print(f"[wait] 登录{'成功' if ok else '未检测到'}")
    time.sleep(2)
    page.screenshot(path=SHOT.format("3_final"))
    cookies_all = browser.cookies()
    with open(OUT_COOKIES, "w", encoding="utf-8") as f:
        json.dump(cookies_all, f, ensure_ascii=False, indent=1)
    iwc = [c for c in cookies_all if "iwencai" in c["domain"] or "10jqka" in c["domain"]]
    print(f"[dump] {len(cookies_all)}条cookie (THS域{len(iwc)}条)")
    for c in iwc:
        print(f"   {c['domain']:22s} {c['name']}")
    browser.close()
print("SCRIPT-END")
