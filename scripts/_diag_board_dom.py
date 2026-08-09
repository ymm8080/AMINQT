"""诊断: 抓取运行中看板选股池表格的列头与首几行数值."""
from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1500, "height": 1100})
    pg.goto("http://localhost:8501", wait_until="domcontentloaded")
    time.sleep(10)
    hdrs = pg.eval_on_selector_all("thead th", "els => els.map(e => e.innerText)")
    print("HEADERS:", hdrs)
    rows = pg.eval_on_selector_all("tbody tr", "els => els.slice(0, 6).map(tr => tr.innerText)")
    for i, r in enumerate(rows):
        # 表格前几个字段(代码/名称/模型/入选/评分/3d/5d/10d), 截断跳过 sparkline
        cells = r.split("\n")
        print(f"ROW{i}:", cells[:9])
    b.close()
