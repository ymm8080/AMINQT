# -*- coding: utf-8 -*-
"""从日常Edge profile提取THS域cookie (esentutl拷锁文件 + DPAPI/AES-GCM解密)"""
import base64
import ctypes
import ctypes.wintypes
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

SRC_COOKIE = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\User Data\Default\Network\Cookies")
SRC_LOCAL = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\User Data\Local State")
TMP = os.path.join(tempfile.gettempdir(), "edge_ck_0910")
os.makedirs(TMP, exist_ok=True)
DST_COOKIE = os.path.join(TMP, "Cookies")
DST_LOCAL = os.path.join(TMP, "Local State")

for src, dst in [(SRC_COOKIE, DST_COOKIE), (SRC_LOCAL, DST_LOCAL)]:
    if os.path.exists(dst):
        os.remove(dst)
    r = subprocess.run(["esentutl", "/y", src, "/d", dst, "/o"], capture_output=True, text=True)
    ok = os.path.exists(dst) and os.path.getsize(dst) > 0
    print(f"[copy] {os.path.basename(src)}: {'OK' if ok else 'FAIL'} ({os.path.getsize(dst) if os.path.exists(dst) else 0}B)")
    if not ok:
        print(r.stderr[:200])

# --- Local State: app-bound key (v10用os_crypt.encrypted_key) ---
b64key = ""
try:
    ls = json.load(open(DST_LOCAL, encoding="utf-8"))
    b64key = ls.get("os_crypt", {}).get("encrypted_key", "")
    print(f"[key] encrypted_key: {'有' if b64key else '无'} ({len(b64key)}ch)")
except Exception as e:
    print(f"[key] FAIL {e}")


def dpapi(data):
    class BUF(ctypes.Structure):
        _fields_ = [("cb", ctypes.wintypes.DWORD), ("pb", ctypes.POINTER(ctypes.c_char))]

    buf = ctypes.create_string_buffer(data, len(data))
    bin_ = BUF(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    bout = BUF()
    if not ctypes.windll.crypt32.CryptUnprotectData(
            ctypes.byref(bin_), None, None, None, None, 0, ctypes.byref(bout)):
        raise OSError("CryptUnprotectData failed")
    out = ctypes.string_at(bout.pb, bout.cb)
    ctypes.windll.kernel32.LocalFree(bout.pb)
    return out


aes_key = None
if b64key:
    raw = base64.b64decode(b64key)
    if raw[:3] == b"DPAPI":
        try:
            aes_key = dpapi(raw[5:])
            print(f"[key] AES key 解密OK ({len(aes_key)}B)")
        except Exception as e:
            print(f"[key] DPAPI FAIL {e}")

THS_HOSTS = ("iwencai", "10jqka", "thsi")

con = sqlite3.connect(DST_COOKIE)
rows = con.execute(
    "SELECT host_key, name, encrypted_value, value, expires_utc, is_secure FROM cookies").fetchall()
print(f"[db] {len(rows)}条cookie总量")

ths = [r for r in rows if any(h in r[0] for h in THS_HOSTS)]
print(f"[db] THS域 {len(ths)}条:")
LOGIN_NAMES = {"userid", "user_id", "snuid", "sessionid", "fid"}
logged = False
out = []
for host, name, ev, val, exp, sec in sorted(ths, key=lambda r: (r[0], r[1])):
    dec = ""
    if val:
        dec = val
    elif ev[:3] in (b"v10", b"v11"):
        try:
            from Crypto.Cipher import AES
            nonce, ct = ev[3:15], ev[15:]
            c = AES.new(aes_key, AES.MODE_GCM, nonce=nonce)
            dec = c.decrypt_and_verify(ct[:-16], ct[-16:]).decode("utf-8", "replace") if aes_key else "(无key)"
        except Exception as e:
            try:
                from cryptography.hazmat.primitives.ciphers.aead import AESGCM
                dec = AESGCM(aes_key).decrypt(nonce, ev[15:], None).decode("utf-8", "replace")
            except Exception as e2:
                dec = f"(解密失败 {str(e2)[:30]})"
    elif ev[:3] == b"v20":
        dec = "(v20 app-bound 暂不解)"
    else:
        dec = f"(prefix={ev[:3]!r} len={len(ev)})"
    if name in LOGIN_NAMES and dec and not dec.startswith("("):
        logged = True
    pv = dec if not dec.startswith("(") else dec
    print(f"   {host:28s} {name:14s} = {pv[:26]}{'...' if len(pv) > 26 else ''}")
    out.append({"domain": host, "name": name, "value": dec,
                "expires_utc": exp, "secure": bool(sec)})

with open(r"D:/AMINQT/AMINQT CODES/tmp_t/_edge_ths_cookies_0910.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"[verdict] 登录态cookie(userid等): {'存在!' if logged else '未见'}")
print("[saved] tmp_t/_edge_ths_cookies_0910.json")
