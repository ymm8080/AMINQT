# -*- coding: utf-8 -*-
"""判别封禁类型: 同IP真浏览器 chromium 打 robot-data API.
通=脚本指纹封(TLS级) -> 改浏览器路由; 403=纯IP封 -> 只能等自愈/换出口.
"""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
from playwright.sync_api import sync_playwright

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36"
cookies = json.load(open(r"D:/AMINQT/AMINQT CODES/tmp_t/_ths_cookies_logged_0910.json", encoding="utf-8"))
ck = [c for c in cookies if ("iwencai" in c.get("domain", "") or "10jqka" in c.get("domain", ""))]
BODY = {
    "add_info": '{"urp":{"scene":1,"company":1,"business":1},"contentType":"json","searchInfo":true}',
    "perpage": "10", "page": 1, "source": "Ths_iwencai_Xuangu",
    "log_info": '{"input_type":"click"}', "version": "2.0",
    "secondary_intent": "stock", "question": "20230901看涨信号的股票",
}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, channel="msedge",
                                args=["--host-resolver-rules=MAP www.iwencai.com 121.12.127.73"])
    ctx = browser.new_context(user_agent=UA)
    ctx.add_cookies([{k: c.get(k) for k in ("name", "value", "domain", "path")}
                     for c in ck])
    page = ctx.new_page()
    r = page.goto("http://www.iwencai.com/", timeout=30000, wait_until="domcontentloaded")
    print(f"[homepage] HTTP {r.status} title={page.title()[:40]!r}")
    res = page.evaluate(
        """async (body) => {
            const r = await fetch('/customized/chart/get-robot-data', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body),
            });
            const t = await r.text();
            return {status: r.status, head: t.slice(0, 300)};
        }""", BODY)
    print(f"[robot-data] HTTP {res['status']}")
    print(f"[body-head] {res['head']}")
    browser.close()
