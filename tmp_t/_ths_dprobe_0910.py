# -*- coding: utf-8 -*-
"""探测未封子域d.10jqka.com.cn的内容 + 扫更多子域找漏网IP"""
import socket
import ssl
import sys

sys.stdout.reconfigure(encoding="utf-8")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0.0.0 Safari/537.36",
      "Referer": "https://www.10jqka.com.cn/"}

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def tls_get(host, path, ip=None, timeout=12):
    try:
        if ip is None:
            ip = socket.gethostbyname(host)
        s = socket.create_connection((ip, 443), timeout=timeout)
        w = ctx.wrap_socket(s, server_hostname=host)
        req = f"GET {path} HTTP/1.1\r\nHost: {host}\r\n"
        for k, v in UA.items():
            req += f"{k}: {v}\r\n"
        req += "Connection: close\r\n\r\n"
        w.sendall(req.encode())
        buf = b""
        while len(buf) < 65536:
            chunk = w.recv(65536)
            if not chunk:
                break
            buf += chunk
        w.close()
        head, _, body = buf.partition(b"\r\n\r\n")
        status = head.split(b"\r\n")[0].decode(errors="replace")
        return status, len(body), body[:400]
    except Exception as e:
        return f"FAIL {type(e).__name__}", 0, str(e)[:120].encode()


print("== d.10jqka.com.cn 内容探测")
for path in ["/", "/v6/line/hs_002377/1/all.js", "/v2/line/hs_002377/01/last.js"]:
    st, ln, body = tls_get("d.10jqka.com.cn", path)
    print(f"\n-- {path}\n   {st}, body={ln}")
    print(f"   {body[:350]!r}")

print("\n== 更多子域IP扫描 (找43.248.130.133之外的活IP)")
for dom in ["hq.10jqka.com.cn", "m.10jqka.com.cn", "wx.10jqka.com.cn",
            "i.10jqka.com.cn", "eqs.10jqka.com.cn", "t.10jqka.com.cn",
            "update.10jqka.com.cn", "sp.10jqka.com.cn", "cookie.10jqka.com.cn",
            "smartbox.iwencai.com", "s.iwencai.com"]:
    try:
        ip = socket.gethostbyname(dom)
    except Exception:
        print(f"{dom:26s} DNS-FAIL")
        continue
    tag = "SAME-BANNED-IP" if ip.startswith("43.248.130") else "OTHER-IP"
    ok = "-"
    if tag == "OTHER-IP":
        try:
            s = socket.create_connection((ip, 443), timeout=8)
            w = ctx.wrap_socket(s, server_hostname=dom)
            ok = f"TLS-OK {w.version()}"
            w.close()
        except Exception as e:
            ok = f"TLS-KILL {type(e).__name__}"
    print(f"{dom:26s} {ip:16s} {tag:14s} {ok}")
