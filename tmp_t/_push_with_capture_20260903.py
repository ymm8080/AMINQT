# -*- coding: utf-8 -*-
"""09-03 THS 推送 v7: 整行高勾读(消假阴性) + 未勾行点行文字区切换 + 点击后复验."""
import sys, time, os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import scripts._ths_watchlist_push as P
from scripts._ths_ui import user_idle_seconds
from pathlib import Path

_n = [0]

def fixed_rows_from_img(img):
    text = img[:, 33:]
    rowfrac = (text.mean(axis=2) < 120).mean(axis=1)
    bands = []
    s = None
    for y, on in enumerate(rowfrac):
        if on and s is None:
            s = y
        elif not on and s is not None:
            if y - s >= 10:
                bands.append((s, y))
            s = None
    if s is not None and len(rowfrac) - s >= 10:
        bands.append((s, len(rowfrac)))
    rows = []
    for ya, yb in bands:
        yc = (ya + yb) // 2
        if yc < 7 or yc + 7 > img.shape[0]:
            continue
        cell = img[ya:yb, 17:28].mean(axis=2)
        rows.append((yc, bool(cell.min() < 80)))
    return rows

def wrap(img):
    rows = fixed_rows_from_img(img)
    _n[0] += 1
    try:
        from PIL import Image
        out = os.path.join(ROOT, "tmp_t", f"_dlg_rows_{_n[0]:02d}_n{len(rows)}.png")
        Image.fromarray(img).save(out)
    except Exception as exc:
        print("[capture] save fail:", exc, flush=True)
    return rows

P._dlg_rows_from_img = wrap

def ensure_checked_v2(dlg, expected, log=print, max_rounds=6):
    """行数核验 + 未勾行点行文字区切换, 每击后复验, 两击不响则整体放弃."""
    import uiautomation as auto
    from PIL import ImageGrab

    from scripts import _ths_ui as ui

    r = dlg.BoundingRectangle
    y0 = r.top + 62
    y1 = r.bottom - 58
    for rnd in range(max_rounds):
        if rnd:
            time.sleep(1.0)
        img = np.asarray(ImageGrab.grab(bbox=(r.left, y0, r.right - 10, y1)))
        rows = P._dlg_rows_from_img(img)
        if len(rows) != expected:
            log(f"[v7] 行数 {len(rows)} != {expected} (round {rnd + 1})")
            continue
        unchecked = [y0 + yc for yc, ck in rows if not ck]
        if not unchecked:
            if rnd:
                log(f"[v7] round {rnd + 1}: 全勾确认")
            return True
        log(f"[v7] round {rnd + 1}: {len(unchecked)} 行未勾, 点行文字区切换")
        for ay in unchecked:
            ui.assert_foreground_hexin(f"切换行 y={ay}")
            for attempt in (1, 2):
                auto.Click(r.left + 120, ay)
                time.sleep(0.6)
                img2 = np.asarray(ImageGrab.grab(bbox=(r.left, y0, r.right - 10, y1)))
                rows2 = P._dlg_rows_from_img(img2)
                hit = [ck for yc, ck in rows2 if abs((y0 + yc) - ay) <= 5]
                if hit and hit[-1]:
                    break
            else:
                log(f"[v7] y={ay} 两击未变勾, 放弃本批")
                return False
        time.sleep(0.5)
    return False

P._ensure_dialog_rows_checked = ensure_checked_v2
P.CHUNK_SIZE = 2  # 识别器每4行第3位不自动勾; 2行批全落安全位, 零点击纯核验

codes = [c for c in Path(
    r"D:\AMINQT\Daily Operation\STOCK LIST"
    r"\ths_watchlist_20260903__M20260903__D20260830excessfix.txt"
).read_text(encoding="utf-8").split() if c]
codes = [c for c in codes if c != "000985"]
if "688118" not in codes:
    codes.append("688118")

tmp = Path(ROOT) / "tmp_t" / "_ths_payload_final_20260903.txt"
tmp.write_text("\n".join(codes) + "\n", encoding="utf-8")
print(f"[v8] payload {len(codes)} 只 (000985 剔除, 688118 补回, CHUNK=2)", flush=True)

t0 = time.time()
while user_idle_seconds() < 90 and time.time() - t0 < 3600:
    time.sleep(10)
idle = user_idle_seconds()
print("waited %.0fs, idle=%.0fs" % (time.time() - t0, idle), flush=True)
if idle < 90:
    print("GAVE UP: machine never idle")
    sys.exit(1)

ok = P.push_via_ths(tmp)
print("[v8] push_via_ths ->", ok, flush=True)
sys.exit(0 if ok else 1)
